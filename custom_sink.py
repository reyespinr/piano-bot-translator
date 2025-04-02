"""
Discord real-time audio processing module.

This module provides a customized implementation of Discord's audio sink
for real-time speech detection, transcription, and translation. It handles
audio streams from Discord voice channels, processes them to detect speech,
and uses machine learning models to transcribe and translate the speech.

Features:
- Context-aware speech processing with session-based priority
- User-specific processing toggles for selective filtering
- Adaptive speech detection thresholds based on conversation state
- Background worker threads for audio transcription
"""
import asyncio
import os
import time
import io
import wave
import queue
import threading
from dataclasses import dataclass
from queue import Queue
import numpy as np
from discord.sinks import WaveSink
import utils


@dataclass
class AudioConfig:
    """Configuration parameters for audio processing."""
    pause_threshold: float = 1.0
    silence_threshold: int = 10  # Frames of silence to consider speech ended
    force_process_timeout: float = 0.8  # Seconds to force process after silence


@dataclass
# pylint: disable=too-many-instance-attributes
class UserState:
    """Manages per-user state for audio processing."""
    last_packet_time: float = 0
    last_active_time: float = 0
    is_speaking: bool = False
    silence_frames: int = 0
    speech_detected: bool = False
    last_processed_time: float = 0
    speech_buffer: io.BytesIO = None
    pre_speech_buffer: list = None
    energy_history: list = None

    def __post_init__(self):
        """Initialize mutable default values."""
        if self.speech_buffer is None:
            self.speech_buffer = io.BytesIO()
        if self.pre_speech_buffer is None:
            self.pre_speech_buffer = []
        if self.energy_history is None:
            self.energy_history = []


@dataclass
class SessionTracker:
    """Tracks conversation session state."""
    start_time: float = 0
    state: str = "new"  # "new", "active", or "established"
    last_speaker_change: float = 0
    current_speakers: set = None
    last_activity_time: float = 0

    def __post_init__(self):
        """Initialize default values that need computation."""
        now = time.time()
        self.start_time = now
        self.last_speaker_change = now
        self.last_activity_time = now
        if self.current_speakers is None:
            self.current_speakers = set()


@dataclass
class WorkerManager:
    """Manages transcription worker threads."""
    num_workers: int
    event_loop: asyncio.AbstractEventLoop
    queue: Queue = None
    running: bool = True
    workers: list = None
    timer: threading.Timer = None

    def __post_init__(self):
        """Initialize mutable default values."""
        if self.queue is None:
            self.queue = queue.Queue(maxsize=10)
        if self.workers is None:
            self.workers = []


class RealTimeWaveSink(WaveSink):
    """Real-time audio processing sink for Discord voice data.

    This class extends Discord's WaveSink to provide real-time speech detection,
    transcription and translation capabilities. It processes incoming audio packets,
    detects speech activity, and manages the workflow of converting speech to text.

    Features:
    - Automatic speech detection based on audio energy levels
    - Buffering for smoother speech start/end handling
    - Silence detection for determining speech boundaries
    - Background processing of transcription tasks
    - Periodic checking for inactive speakers
    - Integration with GUI for displaying transcribed and translated text
    - Session-based priority filtering for improved conversation context
    - Selective user processing through toggle controls
    - Adaptive thresholds based on conversation phase

    Args:
        *args: Additional positional arguments to pass to the parent class.
        pause_threshold (float, optional): Time in seconds to detect a long pause.
            Defaults to 1.0.
        event_loop (asyncio.AbstractEventLoop, optional): Event loop for async operations.
            Defaults to None.
        num_workers (int, optional): Number of transcription worker threads.
            Defaults to 3.
        **kwargs: Additional keyword arguments to pass to the parent class.
    """

    def __init__(self, *args, pause_threshold=1.0, event_loop=None, num_workers=3, **kwargs):
        super().__init__(*args, **kwargs)

        # Initialize the parent class reference
        self.parent = None

        # Use composition to organize related attributes
        self.config = AudioConfig(
            pause_threshold=pause_threshold,
            silence_threshold=10,
            force_process_timeout=0.8
        )

        self.session = SessionTracker()

        self.workers = WorkerManager(num_workers, event_loop)

        # User state is stored in dictionaries mapping user IDs to UserState objects
        self.users = {}

        # Initialize worker threads
        self._start_worker_threads()

        # Start the timer to check for inactive speakers
        self._start_processing_timer()

    def _start_worker_threads(self):
        """Initialize and start worker threads."""
        for i in range(self.workers.num_workers):
            thread = threading.Thread(
                target=self._transcription_worker,
                daemon=True,
                name=f"\nTranscriptionWorker-{i}"
            )
            thread.start()
            self.workers.workers.append(thread)
            print(
                f"Started transcription worker {i+1}/{self.workers.num_workers}")

    def _start_processing_timer(self):
        """Start the timer to check for inactive speakers."""
        self.workers.timer = threading.Timer(
            1.0, self._check_inactive_speakers)
        self.workers.timer.daemon = True
        self.workers.timer.start()

    def _get_user_state(self, user):
        """Get or create UserState for a user."""
        if user not in self.users:
            self.users[user] = UserState()
        return self.users[user]

    def is_audio_active(self, audio_data, user):
        """Check if audio data contains active speech."""
        # Convert bytes to numpy array (assuming PCM signed 16-bit little-endian)
        audio_array = np.frombuffer(audio_data, dtype=np.int16)

        # Calculate energy of the current frame
        energy = np.mean(np.abs(audio_array))

        # Get user state
        user_state = self._get_user_state(user)

        # Initialize energy history if needed
        if not user_state.energy_history:
            user_state.energy_history = [energy] * 5

        # Update energy history
        user_state.energy_history.append(energy)
        # Keep last 5 frames
        user_state.energy_history = user_state.energy_history[-5:]

        # Calculate average energy over recent frames
        avg_energy = np.mean(user_state.energy_history)

        # Return if audio is considered active speech
        return avg_energy > 300  # Threshold for speech detection

    def write(self, data, user):
        """Process incoming audio data from Discord."""
        try:
            current_time = time.time()

            # Check if user is enabled for processing
            if (hasattr(self, 'parent') and
                    self.parent and
                    hasattr(self.parent, 'user_processing_enabled')):
                if (int(user) in self.parent.user_processing_enabled and
                        not self.parent.user_processing_enabled[int(user)]):
                    # User is disabled, just write to main buffer and skip processing
                    super().write(data, user)
                    return

            # Update last activity time when any audio is processed
            self.session.last_activity_time = current_time

            # Initialize user-specific data structures if needed
            user_state = self._get_user_state(user)

            # First time seeing this user?
            if user_state.last_packet_time == 0:
                user_state.last_packet_time = current_time

            # Check if the current packet contains active speech
            is_active = self.is_audio_active(data, user)

            # Update last active time if speech is detected
            if is_active:
                user_state.last_active_time = current_time

            # Calculate time differences
            time_diff = current_time - user_state.last_packet_time
            active_diff = current_time - user_state.last_active_time

            # Process speech if silent for too long
            if (user_state.is_speaking and
                    user_state.speech_detected and
                    active_diff > self.config.force_process_timeout):
                self._process_silent_speech(user)

            # Handle long pauses
            if time_diff > self.config.pause_threshold:
                self._handle_long_pause(user)

            # Store recent frames for smoother beginning of speech
            self._update_pre_speech_buffer(user, data)

            if is_active:
                self._handle_active_speech(user, data)
            else:
                self._handle_silence(user, data)

            # Update the last packet time
            user_state.last_packet_time = current_time

            # Write to the main buffer
            super().write(data, user)

        except (KeyError, TypeError, ValueError, AttributeError, IOError, RuntimeError) as e:
            print(f"Error in write method for user {user}: {e}")

    def _update_pre_speech_buffer(self, user, data):
        """Update the pre-speech buffer for smoother speech beginning."""
        user_state = self._get_user_state(user)
        user_state.pre_speech_buffer.append(data)
        if len(user_state.pre_speech_buffer) > 10:  # Keep ~200ms of audio
            user_state.pre_speech_buffer.pop(0)

    def _process_silent_speech(self, user):
        """Process speech after detecting significant silence."""
        print(
            f"Significant pause detected for user {user}. Processing speech.")
        user_state = self._get_user_state(user)
        user_state.is_speaking = False
        self.process_speech_buffer(user)
        user_state.speech_detected = False
        user_state.silence_frames = 0
        user_state.speech_buffer = io.BytesIO()

    def _handle_long_pause(self, user):
        """Handle a long pause in audio packets."""
        print(
            f"Long pause detected for user {user}. Processing any speech and resetting.")

        user_state = self._get_user_state(user)

        # Process any accumulated speech before resetting
        if user_state.is_speaking and user_state.speech_detected:
            print(f"Processing speech before resetting for user {user}")
            self.process_speech_buffer(user)

        # Reset state
        user_state.is_speaking = False
        user_state.silence_frames = 0
        user_state.speech_detected = False
        user_state.speech_buffer = io.BytesIO()
        user_state.pre_speech_buffer = []

    def _handle_active_speech(self, user, data):
        """Handle incoming active speech data."""
        user_state = self._get_user_state(user)

        # Reset silence counter when speech is detected
        user_state.silence_frames = 0

        # Mark that speech was detected in this session
        user_state.speech_detected = True

        # If this is the start of speech, add pre-speech buffer first
        if not user_state.is_speaking:
            print(f"Speech started for user {user}")
            user_state.is_speaking = True
            # Add pre-speech frames for smoother beginning
            for pre_data in user_state.pre_speech_buffer:
                user_state.speech_buffer.write(pre_data)

        # Add this audio data to the speech buffer
        user_state.speech_buffer.write(data)

    def _handle_silence(self, user, data):
        """Handle silence after speech."""
        user_state = self._get_user_state(user)

        # Add a few frames to the speech buffer for smoother transitions
        if user_state.is_speaking and user_state.silence_frames < 5:
            user_state.speech_buffer.write(data)

        # Increment silence counter
        user_state.silence_frames += 1

        # Check if silence has persisted long enough to consider speech ended
        if (user_state.is_speaking and
                user_state.silence_frames > self.config.silence_threshold):
            print(f"Speech ended for user {user}. Processing audio.")
            user_state.is_speaking = False

            # Only process if speech was detected in this session
            if user_state.speech_detected:
                self.process_speech_buffer(user)
                user_state.speech_detected = False
                user_state.speech_buffer = io.BytesIO()

    def _transcription_worker(self):
        """Background worker to process transcription queue."""
        thread_name = threading.current_thread().name
        print(f"{thread_name} started")

        while self.workers.running:
            try:
                # Get item from queue with timeout
                audio_file, user = self.workers.queue.get(timeout=1)
                print(f"{thread_name} processing audio for user {user}")

                # Process the transcription
                asyncio.run_coroutine_threadsafe(
                    self.transcribe_audio(audio_file, user),
                    self.workers.event_loop
                )

                # Wait a little before processing next item
                time.sleep(0.2)
                self.workers.queue.task_done()

            except queue.Empty:
                # No items in queue, just continue waiting
                pass
            except (asyncio.InvalidStateError, RuntimeError, ValueError, TypeError, OSError) as e:
                print(f"Error in {thread_name}: {e}")

        print(f"{thread_name} stopped")

    def process_speech_buffer(self, user):
        """Process the speech buffer for a user and queue for transcription."""
        user_state = self._get_user_state(user)

        # Update session state and tracking variables
        current_time = self._update_session_state(user)

        # Get audio duration and check if it meets minimum requirements
        duration_seconds = self._get_audio_duration(user_state.speech_buffer)
        min_duration = self._get_minimum_duration()

        if duration_seconds < min_duration:
            print(
                f"Speech too short ({duration_seconds:.2f}s < {min_duration:.2f}s), skipping.")
            return

        # Check if user is on cooldown
        if user_state.last_processed_time > 0:
            time_since_last = current_time - user_state.last_processed_time
            if time_since_last < 2.0:  # 2 second cooldown
                print(f"Cooldown active for user {user}, skipping.")
                return

        # Create and process audio file
        self._create_and_queue_audio_file(user, user_state, current_time)

    def _update_session_state(self, user):
        """Update session state based on user activity and timing."""
        current_time = time.time()

        # Check for new conversation session
        if user not in self.session.current_speakers:
            print(f"New speaker detected: {user}")
            self.session.state = "new"
            self.session.start_time = current_time
            self.session.current_speakers.add(user)
            self.session.last_speaker_change = current_time

        # Check for extended silence
        if current_time - self.session.last_activity_time > 10:
            print("Long silence detected, starting new conversation session")
            self.session.state = "new"
            self.session.start_time = current_time

        # Session state transitions
        time_in_session = current_time - self.session.start_time
        if self.session.state == "new" and time_in_session > 30:
            self.session.state = "active"
            print("Session transitioned to active state")
        elif self.session.state == "active" and time_in_session > 120:
            self.session.state = "established"
            print("Session transitioned to established state")

        # Update tracking variables
        self.session.last_activity_time = current_time
        return current_time

    def _get_audio_duration(self, speech_buffer):
        """Calculate audio duration from buffer size."""
        speech_buffer.seek(0, os.SEEK_END)
        buffer_size = speech_buffer.tell()
        # Calculate approximate duration (48000 Hz, 16-bit stereo)
        return buffer_size / 192000

    def _get_minimum_duration(self):
        """Get minimum duration threshold based on session state and queue size."""
        queue_not_busy = self.workers.queue.qsize() < 3

        if self.session.state == "new":
            # More permissive in new conversations
            min_duration = 0.75 if queue_not_busy else 1.5
            print(f"Using new session threshold: {min_duration:.2f}s")
        elif self.session.state == "active":
            # Normal threshold for active sessions
            min_duration = 1.0 if queue_not_busy else 2.0
        else:  # established
            # More strict for established conversations
            min_duration = 1.25 if queue_not_busy else 2.25

        return min_duration

    def _create_and_queue_audio_file(self, user, user_state, current_time):
        """Create WAV file from buffer and add to transcription queue."""
        # Create a temporary WAV file
        timestamp = int(time.time())
        temp_audio_file = f"{user}_{timestamp}_speech.wav"

        try:
            # Write the speech data to a WAV file
            user_state.speech_buffer.seek(0)  # Reset buffer pointer
            with open(temp_audio_file, 'wb') as out_f:
                wf = wave.Wave_write(out_f)
                wf.setnchannels(2)  # Stereo
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(48000)  # 48kHz (Discord's standard)
                wf.writeframes(user_state.speech_buffer.read())
                wf.close()

            print(f"Processing speech for user {user}, "
                  f"audio file size: {os.path.getsize(temp_audio_file)} bytes")

            # Add to transcription queue
            if not self.workers.queue.full():
                self.workers.queue.put((temp_audio_file, user), block=False)
                print(f"Added transcription task to queue for user {user}")
                user_state.last_processed_time = current_time
            else:
                print(
                    f"Transcription queue full, skipping audio for user {user}")
                os.remove(temp_audio_file)  # Clean up

        except (IOError, OSError, wave.Error) as e:
            print(f"Error creating WAV file for user {user}: {e}")

    async def transcribe_audio(self, audio_file, user):
        """Transcribe the audio file and update the GUI with results."""
        try:
            # Skip if file doesn't exist
            if not os.path.exists(audio_file):
                print(f"Audio file {audio_file} does not exist, skipping.")
                return

            # Get transcription and detected language
            transcribed_text, detected_language = await utils.transcribe(audio_file)

            # Skip processing if transcription was empty (failed confidence check)
            if not transcribed_text:
                print(
                    f"Empty transcription for user {user}, skipping processing.")
                self._cleanup_audio_file(audio_file)
                return

            # Determine if translation is needed
            needs_translation = await utils.should_translate(transcribed_text, detected_language)

            if needs_translation:
                translated_text = await utils.translate(transcribed_text)
                print(f"Translated from {detected_language} to English")
            else:
                # Skip translation for English or empty text
                translated_text = transcribed_text
                print(
                    f"Skipped translation - detected language: {detected_language}")

            # Debug output
            print(f"Transcription for user {user}: {transcribed_text}")

            # Update the GUI
            await self._update_gui(user, transcribed_text, translated_text)

            # Clean up temporary file
            self._cleanup_audio_file(audio_file)

        except (FileNotFoundError, IOError, ValueError, RuntimeError, asyncio.TimeoutError) as e:
            print(
                f"Error during transcription/translation for user {user}: {e}")
            self._cleanup_audio_file(audio_file)

    async def _update_gui(self, user, transcribed_text, translated_text):
        """Update the GUI with transcription and translation results."""
        try:
            if (hasattr(self, 'parent') and self.parent and
                    hasattr(self.parent, 'vc') and self.parent.vc):
                # Get the user's display name
                guild = self.parent.vc.guild
                member = guild.get_member(int(user))
                user_name = member.display_name if member else f"User {user}"

                # Format display text
                transcription = f"{user_name}: {transcribed_text}" if transcribed_text else ""
                translation = f"{user_name}: {translated_text}" if translated_text else ""

                if transcription and translation:
                    # Update the GUI
                    self.parent.update_text_display(
                        transcription, translation)
            else:
                print("Could not update GUI: Missing voice client or parent reference")
        except (AttributeError, TypeError, ValueError) as e:
            print(f"Error updating GUI for user {user}: {e}")

    def _cleanup_audio_file(self, audio_file):
        """Clean up temporary audio file."""
        try:
            if os.path.exists(audio_file):
                os.remove(audio_file)
        except (PermissionError, OSError, IOError) as e:
            print(f"Error removing audio file {audio_file}: {e}")

    def _check_inactive_speakers(self):
        """Periodically check for users who have stopped speaking but haven't been processed."""
        try:
            current_time = time.time()

            # Check each user
            for user, user_state in list(self.users.items()):
                if (user_state.is_speaking and
                        user_state.speech_detected):
                    # Check time since last packet
                    time_since_last = current_time - user_state.last_packet_time

                    # Process if inactive for too long
                    if time_since_last > self.config.force_process_timeout:
                        print(
                            f"Timer detected inactive speaker {user}. Processing speech.")
                        user_state.is_speaking = False
                        self.process_speech_buffer(user)
                        user_state.speech_detected = False
                        user_state.speech_buffer = io.BytesIO()
        except (KeyError, AttributeError, TypeError, ValueError) as e:
            print(f"Error checking inactive speakers: {e}")
        finally:
            # Reschedule the timer if still running
            if self.workers.running:
                self.workers.timer = threading.Timer(
                    0.5, self._check_inactive_speakers)
                self.workers.timer.daemon = True
                self.workers.timer.start()

    def cleanup(self):
        """Clean up resources when the sink is no longer needed."""
        self.workers.running = False
        if self.workers.timer:
            self.workers.timer.cancel()

        # Worker threads are daemon threads and will terminate when the program exits
        print(
            f"Cleaning up sink resources - {self.workers.num_workers} workers stopped")
