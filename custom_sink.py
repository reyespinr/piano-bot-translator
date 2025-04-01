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
import numpy as np
from discord.sinks import WaveSink
import utils


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
        **kwargs: Additional keyword arguments to pass to the parent class.
    """

    def __init__(self, *args, pause_threshold=1.0, event_loop=None, **kwargs):
        super().__init__(*args, **kwargs)

        # Initialize the parent class
        self.parent = None

        # Audio processing parameters
        self.pause_threshold = pause_threshold
        self.silence_threshold = 10  # Frames of silence to consider speech ended
        self.force_process_timeout = 0.8  # Seconds to force process after silence

        # User tracking
        self.user_last_packet_time = {}
        self.last_active_time = {}
        self.is_speaking = {}
        self.silence_frames = {}
        self.speech_detected = {}
        self.last_processed_time = {}

        # Audio buffers
        self.speech_buffers = {}
        self.pre_speech_buffers = {}
        self.energy_history = {}

        # Event loop reference for async operations
        self.event_loop = event_loop

        # Queue for managing transcription tasks
        self.transcription_queue = queue.Queue(maxsize=10)
        self.worker_running = True

        # Start worker thread for transcriptions
        self.worker_thread = threading.Thread(
            target=self._transcription_worker, daemon=True)
        self.worker_thread.start()

        # Start the timer to check for inactive speakers
        self.processing_timer = threading.Timer(
            1.0, self._check_inactive_speakers)
        self.processing_timer.daemon = True
        self.processing_timer.start()

        # Add session tracking variables
        self.session_start_time = time.time()
        self.session_state = "new"   # "new", "active", or "established"
        self.last_speaker_change = time.time()
        self.current_speakers = set()
        self.silence_duration = 0
        self.last_activity_time = time.time()

    def is_audio_active(self, audio_data, user):
        """Check if audio data contains active speech."""
        # Convert bytes to numpy array (assuming PCM signed 16-bit little-endian)
        audio_array = np.frombuffer(audio_data, dtype=np.int16)

        # Calculate energy of the current frame
        energy = np.mean(np.abs(audio_array))

        # Initialize energy history if needed
        if user not in self.energy_history:
            self.energy_history[user] = [energy] * 5

        # Update energy history
        self.energy_history[user].append(energy)
        # Keep last 5 frames
        self.energy_history[user] = self.energy_history[user][-5:]

        # Calculate average energy over recent frames
        avg_energy = np.mean(self.energy_history[user])

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
            self.last_activity_time = current_time

            # Initialize user-specific data structures if needed
            if user not in self.user_last_packet_time:
                self._initialize_user_data(user, current_time)

            # Check if the current packet contains active speech
            is_active = self.is_audio_active(data, user)

            # Update last active time if speech is detected
            if is_active:
                self.last_active_time[user] = current_time

            # Calculate time differences
            time_diff = current_time - self.user_last_packet_time[user]
            active_diff = current_time - self.last_active_time[user]

            # Process speech if silent for too long
            if self._should_process_due_to_silence(user, active_diff):
                self._process_silent_speech(user)

            # Handle long pauses
            if self._is_long_pause(time_diff):
                self._handle_long_pause(user)

            # Store recent frames for smoother beginning of speech
            self._update_pre_speech_buffer(user, data)

            if is_active:
                self._handle_active_speech(user, data)
            else:
                self._handle_silence(user, data)

            # Update the last packet time
            self.user_last_packet_time[user] = current_time

            # Write to the main buffer
            super().write(data, user)

        except (KeyError, TypeError, ValueError, AttributeError, IOError, RuntimeError) as e:
            print(f"Error in write method for user {user}: {e}")

    def _initialize_user_data(self, user, current_time):
        """Initialize data structures for a new user."""
        self.user_last_packet_time[user] = current_time
        self.is_speaking[user] = False
        self.silence_frames[user] = 0
        self.speech_detected[user] = False
        self.speech_buffers[user] = io.BytesIO()
        self.pre_speech_buffers[user] = []
        self.last_active_time[user] = 0

    def _should_process_due_to_silence(self, user, active_diff):
        """Check if we should process speech due to silence duration."""
        return (self.is_speaking[user] and
                self.speech_detected[user] and
                active_diff > self.force_process_timeout)

    def _is_long_pause(self, time_diff):
        """Check if there's been a long pause in receiving packets."""
        return time_diff > self.pause_threshold

    def _process_silent_speech(self, user):
        """Process speech after detecting significant silence."""
        print(
            f"Significant pause detected for user {user}. Processing speech.")
        self.is_speaking[user] = False
        self.process_speech_buffer(user)
        self.speech_detected[user] = False
        self.silence_frames[user] = 0
        self.speech_buffers[user] = io.BytesIO()

    def _handle_long_pause(self, user):
        """Handle a long pause in audio packets."""
        print(
            f"Long pause detected for user {user}. Processing any speech and resetting.")

        # Process any accumulated speech before resetting
        if self.is_speaking[user] and self.speech_detected[user]:
            print(f"Processing speech before resetting for user {user}")
            self.process_speech_buffer(user)

        # Reset state
        self.is_speaking[user] = False
        self.silence_frames[user] = 0
        self.speech_detected[user] = False
        self.speech_buffers[user] = io.BytesIO()
        self.pre_speech_buffers[user] = []

    def _update_pre_speech_buffer(self, user, data):
        """Update the pre-speech buffer for smoother speech beginning."""
        self.pre_speech_buffers[user].append(data)
        if len(self.pre_speech_buffers[user]) > 10:  # Keep ~200ms of audio
            self.pre_speech_buffers[user].pop(0)

    def _handle_active_speech(self, user, data):
        """Handle incoming active speech data."""
        # Reset silence counter when speech is detected
        self.silence_frames[user] = 0

        # Mark that speech was detected in this session
        self.speech_detected[user] = True

        # If this is the start of speech, add pre-speech buffer first
        if not self.is_speaking[user]:
            print(f"Speech started for user {user}")
            self.is_speaking[user] = True
            # Add pre-speech frames for smoother beginning
            for pre_data in self.pre_speech_buffers[user]:
                self.speech_buffers[user].write(pre_data)

        # Add this audio data to the speech buffer
        self.speech_buffers[user].write(data)

    def _handle_silence(self, user, data):
        """Handle silence after speech."""
        # Add a few frames to the speech buffer for smoother transitions
        if self.is_speaking[user] and self.silence_frames[user] < 5:
            self.speech_buffers[user].write(data)

        # Increment silence counter
        self.silence_frames[user] += 1

        # Check if silence has persisted long enough to consider speech ended
        if self.is_speaking[user] and self.silence_frames[user] > self.silence_threshold:
            print(f"Speech ended for user {user}. Processing audio.")
            self.is_speaking[user] = False

            # Only process if speech was detected in this session
            if self.speech_detected[user]:
                self.process_speech_buffer(user)
                self.speech_detected[user] = False
                self.speech_buffers[user] = io.BytesIO()

    def _transcription_worker(self):
        """Background worker to process transcription queue."""
        while self.worker_running:
            try:
                # Get item from queue with timeout
                audio_file, user = self.transcription_queue.get(timeout=1)

                # Process the transcription
                asyncio.run_coroutine_threadsafe(
                    self.transcribe_audio(audio_file, user),
                    self.event_loop
                )

                # Wait a little before processing next item
                time.sleep(0.5)
                self.transcription_queue.task_done()

            except queue.Empty:
                # No items in queue, just continue waiting
                pass
            except (asyncio.InvalidStateError, RuntimeError, ValueError, TypeError, OSError) as e:
                print(f"Error in transcription worker: {e}")

    def process_speech_buffer(self, user):
        """Process the speech buffer for a user and queue for transcription.

        This method handles the conversion of recorded audio to a file for processing.
        It implements several optimization techniques:
        - Session-based context awareness (new/active/established states)
        - Dynamic minimum duration thresholds based on session state
        - User-specific cooldown periods to prevent rapid transcriptions
        - Queue size-aware filtering to manage high-traffic periods

        Args:
            user (str): User ID of the speaker
        """
        if user not in self.speech_buffers:
            print(f"No speech buffer found for user {user}.")
            return

        # Update session state
        current_time = time.time()

        # Check for new conversation session
        if user not in self.current_speakers:
            print(f"New speaker detected: {user}")
            self.session_state = "new"
            self.session_start_time = current_time
            self.current_speakers.add(user)
            self.last_speaker_change = current_time

        # Check for extended silence (10+ seconds of no activity)
        if current_time - self.last_activity_time > 10:
            print("Long silence detected, starting new conversation session")
            self.session_state = "new"
            self.session_start_time = current_time

        # Session state transitions
        time_in_session = current_time - self.session_start_time
        if self.session_state == "new" and time_in_session > 30:
            self.session_state = "active"
            print("Session transitioned to active state")
        elif self.session_state == "active" and time_in_session > 120:
            self.session_state = "established"
            print("Session transitioned to established state")

        # Update tracking variables
        self.last_activity_time = current_time

        # Get the speech buffer
        speech_buffer = self.speech_buffers[user]
        speech_buffer.seek(0, os.SEEK_END)
        buffer_size = speech_buffer.tell()

        # Calculate approximate duration in seconds (48000 Hz, 16-bit stereo)
        duration_seconds = buffer_size / 192000

        # Set minimum duration based on session state and queue size
        if self.session_state == "new":
            # More permissive in new conversations
            min_duration = 0.75 if self.transcription_queue.qsize() < 3 else 1.5
            print(f"Using new session threshold: {min_duration:.2f}s")
        elif self.session_state == "active":
            # Normal threshold for active sessions
            min_duration = 1 if self.transcription_queue.qsize() < 3 else 2
        else:  # established
            # More strict for established conversations
            min_duration = 1.25 if self.transcription_queue.qsize() < 3 else 2.25

        if duration_seconds < min_duration:
            print(
                f"Speech too short ({duration_seconds:.2f}s < {min_duration:.2f}s), skipping.")
            return

        current_time = time.time()
        if user in self.last_processed_time:
            time_since_last = current_time - self.last_processed_time[user]
            if time_since_last < 2.0:  # 2 second cooldown
                print(f"Cooldown active for user {user}, skipping.")
                return

        # Create a temporary WAV file
        timestamp = int(time.time())
        temp_audio_file = f"{user}_{timestamp}_speech.wav"
        speech_buffer.seek(0)  # Reset buffer pointer

        try:
            # Write the speech data to a WAV file
            speech_buffer.seek(0)  # Reset buffer pointer
            with open(temp_audio_file, 'wb') as out_f:
                wf = wave.Wave_write(out_f)
                wf.setnchannels(2)  # Stereo
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(48000)  # 48kHz (Discord's standard)
                wf.writeframes(speech_buffer.read())
                wf.close()

            print(
                f"Processing speech for user {user}, "
                f"audio file size: {os.path.getsize(temp_audio_file)} bytes"
            )

            # Add to transcription queue
            if not self.transcription_queue.full():
                self.transcription_queue.put(
                    (temp_audio_file, user), block=False)
                print(f"Added transcription task to queue for user {user}")
                self.last_processed_time[user] = current_time
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

            # Get transcription and translation
            transcribed_text = await utils.transcribe(audio_file)
            translated_text = await utils.translate(transcribed_text) if transcribed_text else ""

            # Debug output
            print(f"Transcription for user {user}: {transcribed_text}")
            print(f"Translation: {translated_text}")

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
                if transcribed_text:
                    display_transcription = f"{user_name}: {transcribed_text}"
                else:
                    display_transcription = ""

                if translated_text:
                    display_translation = f"{user_name}: {translated_text}"
                else:
                    display_translation = ""

                if display_transcription and display_translation:
                    # Update the GUI
                    self.parent.update_text_display(
                        display_transcription, display_translation)
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

            # Check each active speaker
            for user, is_speaking in list(self.is_speaking.items()):
                if is_speaking and user in self.speech_detected and self.speech_detected[user]:
                    # Check time since last packet
                    time_since_last_packet = current_time - \
                        self.user_last_packet_time.get(user, 0)

                    # Process if inactive for too long
                    if time_since_last_packet > self.force_process_timeout:
                        print(
                            f"Timer detected inactive speaker {user}. Processing speech.")
                        self.is_speaking[user] = False
                        self.process_speech_buffer(user)
                        self.speech_detected[user] = False
                        self.speech_buffers[user] = io.BytesIO()
        except (KeyError, AttributeError, TypeError, ValueError) as e:
            print(f"Error checking inactive speakers: {e}")
        finally:
            # Reschedule the timer if still running
            if self.worker_running:
                self.processing_timer = threading.Timer(
                    0.5, self._check_inactive_speakers)
                self.processing_timer.daemon = True
                self.processing_timer.start()

    def cleanup(self):
        """Clean up resources when the sink is no longer needed."""
        self.worker_running = False
        if self.processing_timer:
            self.processing_timer.cancel()
        print("Cleaning up sink resources...")
