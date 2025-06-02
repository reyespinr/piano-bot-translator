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
import gc
import subprocess
from dataclasses import dataclass
from queue import Queue
import numpy as np
from discord.sinks import WaveSink
import utils
from logging_config import get_logger

logger = get_logger(__name__)


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

        # Initialize the translation callback to None - THIS IS THE KEY FIX
        self.translation_callback = None

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

        # Add tracking for last log time to prevent log spam
        self.last_block_log_time = {}

        # Initialize worker threads
        self._start_worker_threads()

        # Start the timer to check for inactive speakers
        self._start_processing_timer()

    def _start_worker_threads(self):
        """Initialize and start worker threads."""
        for i in range(self.workers.num_workers):
            thread = threading.Thread(
                target=self._transcription_worker,
                daemon=True,                name=f"\nTranscriptionWorker-{i}"
            )
            thread.start()
            self.workers.workers.append(thread)
            logger.info(
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

            # Check if user is enabled for processing - CRITICAL LOGIC
            if (hasattr(self, 'parent') and
                    self.parent and
                    hasattr(self.parent, 'user_processing_enabled')):

                # IMPORTANT: Force user ID to string type for consistency
                user_id = str(user)

                # COMPLETELY REDESIGNED: Check the actual dictionary at runtime for each packet
                if user_id in self.parent.user_processing_enabled:
                    # Get the CURRENT value directly from the dictionary (not a cached value)
                    enabled = bool(
                        self.parent.user_processing_enabled[user_id])

                    if not enabled:                        # Only log once per session
                        if user_id not in self.last_block_log_time:
                            logger.debug(
                                f"User {user_id} audio processing disabled")
                            # CRITICAL: Just write to buffer and IMMEDIATELY return
                            self.last_block_log_time[user_id] = current_time
                        super().write(data, user)
                        return

                else:
                    # Don't automatically enable users - require explicit setting
                    logger.info(
                        f"New user {user_id} detected, requiring manual enable")
                    self.parent.user_processing_enabled[user_id] = False
                    super().write(data, user)
                    return

        # Continue with normal audio processing for enabled users
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
                # Update the last packet time
                self._handle_silence(user, data)
            user_state.last_packet_time = current_time

            # Write to the main buffer
            super().write(data, user)

        except (KeyError, TypeError, ValueError, AttributeError, IOError, RuntimeError) as e:
            logger.error(f"Error in write method for user {user}: {e}")

    def _update_pre_speech_buffer(self, user, data):
        """Update the pre-speech buffer for smoother speech beginning."""
        user_state = self._get_user_state(user)
        user_state.pre_speech_buffer.append(data)
        if len(user_state.pre_speech_buffer) > 10:  # Keep ~200ms of audio
            user_state.pre_speech_buffer.pop(0)

    def _process_silent_speech(self, user):
        """Process speech after detecting significant silence."""
        logger.debug(
            f"Significant pause detected for user {user}. Processing speech.")
        user_state = self._get_user_state(user)
        user_state.is_speaking = False
        self.process_speech_buffer(user)
        user_state.speech_detected = False
        user_state.silence_frames = 0
        user_state.speech_buffer = io.BytesIO()

    def _handle_long_pause(self, user):
        """Handle a long pause in audio packets."""
        logger.debug(
            f"Long pause detected for user {user}. Processing any speech and resetting.")

        user_state = self._get_user_state(user)

        # Process any accumulated speech before resetting
        if user_state.is_speaking and user_state.speech_detected:
            logger.debug(f"Processing speech before resetting for user {user}")
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
        # If this is the start of speech, add pre-speech buffer first
        user_state.speech_detected = True
        if not user_state.is_speaking:
            logger.debug(f"Speech started for user {user}")
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
        # Check if silence has persisted long enough to consider speech ended
        user_state.silence_frames += 1
        if (user_state.is_speaking and
                user_state.silence_frames > self.config.silence_threshold):
            logger.debug(f"Speech ended for user {user}. Processing audio.")
            # Only process if speech was detected in this session
            user_state.is_speaking = False
            if user_state.speech_detected:
                self.process_speech_buffer(user)
                user_state.speech_detected = False
                user_state.speech_buffer = io.BytesIO()

    def _transcription_worker(self):
        """Background worker to process transcription queue."""
        thread_name = threading.current_thread().name
        logger.info(f"{thread_name} started")

        while self.workers.running:
            try:
                # Get item from queue with timeout (no debug message for waiting)
                audio_file, user = self.workers.queue.get(timeout=1)
                logger.debug(f"Processing audio for user {user}")

                # Process the transcription
                asyncio.run_coroutine_threadsafe(
                    self.transcribe_audio(audio_file, user),
                    self.workers.event_loop
                )

                # Wait a little before processing next item
                time.sleep(0.2)
                self.workers.queue.task_done()
            except queue.Empty:
                # No debug message for empty queue
                pass

            except Exception as e:
                logger.error(f"Error in transcription worker: {e}")

        logger.info(f"{thread_name} stopped")

    def process_speech_buffer(self, user):
        """Process the speech buffer for a user and queue for transcription."""
        user_state = self._get_user_state(user)

        # Update session state and tracking variables
        # Get audio duration and check if it meets minimum requirements
        current_time = self._update_session_state(user)
        duration_seconds = self._get_audio_duration(user_state.speech_buffer)
        min_duration = self._get_minimum_duration()

        # Also check minimum buffer size (Discord audio is 192KB/sec, so minimum ~38KB for 0.2s)
        user_state.speech_buffer.seek(0, os.SEEK_END)
        buffer_size = user_state.speech_buffer.tell()
        min_buffer_size = 38400  # ~0.2 seconds of audio

        if duration_seconds < min_duration or buffer_size < min_buffer_size:
            logger.debug(
                f"Speech too short ({duration_seconds:.2f}s < {min_duration:.2f}s) or buffer too small ({buffer_size} < {min_buffer_size} bytes), skipping.")
            return

        # Check if user is on cooldown
        if user_state.last_processed_time > 0:
            time_since_last = current_time - user_state.last_processed_time
            if time_since_last < 2.0:  # 2 second cooldown
                logger.debug(f"Cooldown active for user {user}, skipping.")
                return        # Create and process audio file
        self._create_and_queue_audio_file(user, user_state, current_time)

    def _update_session_state(self, user):
        """Update session state based on user activity and timing."""
        current_time = time.time()

        # Check for new conversation session
        if user not in self.session.current_speakers:
            logger.info(f"New speaker detected: {user}")
            self.session.state = "new"
            self.session.start_time = current_time
            self.session.current_speakers.add(user)
            self.session.last_speaker_change = current_time

        # Check for extended silence
        if current_time - self.session.last_activity_time > 10:
            logger.info(
                "Long silence detected, starting new conversation session")
            self.session.state = "new"
            self.session.start_time = current_time

        # Session state transitions
        time_in_session = current_time - self.session.start_time
        if self.session.state == "new" and time_in_session > 30:
            self.session.state = "active"
            logger.info("Session transitioned to active state")
        elif self.session.state == "active" and time_in_session > 120:
            self.session.state = "established"
            logger.info("Session transitioned to established state")

        # Update tracking variables
        self.session.last_activity_time = current_time
        return current_time

    def _get_audio_duration(self, speech_buffer):
        """Calculate audio duration from buffer size."""
        speech_buffer.seek(0, os.SEEK_END)
        # Calculate approximate duration (48000 Hz, 16-bit stereo)
        buffer_size = speech_buffer.tell()
        return buffer_size / 192000

    def _get_minimum_duration(self):
        """Get minimum duration threshold based on session state and queue size."""
        queue_not_busy = self.workers.queue.qsize() < 3

        if self.session.state == "new":
            # More permissive in new conversations
            min_duration = 0.75 if queue_not_busy else 1.5
            logger.debug(f"Using new session threshold: {min_duration:.2f}s")
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
            audio_data = user_state.speech_buffer.read()

            # Final validation - ensure we have actual audio data
            if len(audio_data) < 1920:  # Less than ~10ms of audio
                logger.debug(
                    f"Audio data too small ({len(audio_data)} bytes), skipping.")
                return

            with open(temp_audio_file, 'wb') as out_f:
                wf = wave.Wave_write(out_f)
                wf.setnchannels(2)  # Stereo
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(48000)  # 48kHz (Discord's standard)
                wf.writeframes(audio_data)
                wf.close()

            logger.info(f"Processing speech for user {user}, "
                        f"audio file size: {os.path.getsize(temp_audio_file)} bytes")

            # Add to transcription queue
            if not self.workers.queue.full():
                self.workers.queue.put((temp_audio_file, user), block=False)
                logger.debug(
                    f"Added transcription task to queue for user {user}")
                user_state.last_processed_time = current_time
            else:
                logger.warning(
                    f"Transcription queue full, skipping audio for user {user}")
                os.remove(temp_audio_file)  # Clean up

        except (IOError, OSError, wave.Error) as e:
            logger.error(f"Error creating WAV file for user {user}: {e}")

    async def transcribe_audio(self, audio_file, user):
        """Transcribe the audio file and update the GUI with results."""
        try:
            # Skip if file doesn't exist
            if not os.path.exists(audio_file):
                logger.warning(
                    f"Audio file {audio_file} does not exist, skipping.")
                return

            transcribed_text = None
            detected_language = None

            try:
                # Get transcription and detected language
                transcribed_text, detected_language = await utils.transcribe(audio_file)

                # Skip processing if transcription was empty (failed confidence check)
                if not transcribed_text:
                    logger.debug(
                        f"Empty transcription for user {user}, skipping processing.")
                    # Must delete file right after transcription
                    await self._force_delete_file(audio_file)
                    return
            except Exception as e:
                logger.error(f"Error during transcription: {e}")
                # Must delete file even if transcription fails
                await self._force_delete_file(audio_file)
                return

            try:
                # Determine if translation is needed
                needs_translation = await utils.should_translate(transcribed_text, detected_language)

                if needs_translation:
                    translated_text = await utils.translate(transcribed_text)
                    logger.info(
                        f"Translated from {detected_language} to English")
                else:
                    # Skip translation for English or empty text
                    translated_text = transcribed_text
                    logger.info(f"Processing speech in {detected_language}")

                # Clean log output
                logger.info(f"Transcription: {transcribed_text}")
                if needs_translation:
                    logger.info(f"Translation: {translated_text}")

                # Try updating GUI if available
                try:
                    await self._update_gui(user, transcribed_text, translated_text)
                except Exception as e:
                    logger.error(f"GUI update failed: {e}")
                    # Even if GUI update fails, try the translation callback directly
                    if hasattr(self, 'translation_callback') and self.translation_callback:
                        try:
                            await self.translation_callback(user, transcribed_text, message_type="transcription")
                            if needs_translation and translated_text != transcribed_text:
                                await self.translation_callback(user, translated_text, message_type="translation")
                            logger.debug("Transcription sent via callback")
                        except Exception as cb_error:
                            logger.error(
                                f"Error using translation callback: {cb_error}")
            finally:
                # CRITICAL: Must delete the file after we're done with it
                # This needs to be in the finally block to ensure it runs
                await self._force_delete_file(audio_file)

        except Exception as e:
            logger.error(
                f"Error during transcription/translation for user {user}: {e}")
            # One final attempt to delete the file
            await self._force_delete_file(audio_file)

    async def _force_delete_file(self, file_path):
        """Forcefully delete a file with multiple retries and GC."""
        if not file_path or not os.path.exists(file_path):
            return

        # Try to delete the file with multiple retries
        for attempt in range(3):
            try:
                # Try to force Python to release any file handles
                gc.collect()                # Try to delete
                os.remove(file_path)
                logger.debug(f"Deleted file: {os.path.basename(file_path)}")

                # Verify deletion
                if not os.path.exists(file_path):
                    return True

                logger.warning(
                    f"File still exists after deletion attempt: {file_path}")
            except (PermissionError, OSError) as e:
                logger.warning(f"Deletion attempt {attempt+1} failed: {e}")
                # Wait before retry
                await asyncio.sleep(0.5)

        # Last resort: try with Windows-specific commands on Windows
        if os.name == 'nt':
            try:
                subprocess.run(f'del /F "{file_path}"',
                               shell=True, check=False)
                logger.debug(
                    f"Attempted deletion with Windows command: {file_path}")
            except Exception as e:
                logger.error(f"Windows command deletion failed: {e}")

        logger.warning(
            f"Failed to delete file after multiple attempts: {file_path}")
        return False

    def _cleanup_audio_file(self, audio_file):
        """
        Legacy synchronous file cleanup method - 
        For backward compatibility only.
        Use _force_delete_file instead.
        """
        # Run the async delete in a non-blocking way
        if audio_file and os.path.exists(audio_file):
            # Create a task to delete the file
            asyncio.create_task(self._force_delete_file(audio_file))

    async def _update_gui(self, user, transcribed_text, translated_text):
        """Update the GUI with transcription and translation results."""
        try:
            # Try to update the GUI through parent object
            if hasattr(self, 'parent') and self.parent:
                # Check if the parent has a update_text_display method
                if hasattr(self.parent, 'update_text_display'):
                    self.parent.update_text_display(
                        transcribed_text, translated_text)
                    return

            # If we get here, try the translation callback
            if hasattr(self, 'translation_callback') and self.translation_callback:
                try:
                    # Send transcription first
                    await self.translation_callback(user, transcribed_text, message_type="transcription")

                    # Then send translation if different
                    if transcribed_text != translated_text:
                        await self.translation_callback(user, translated_text, message_type="translation")
                    else:
                        await self.translation_callback(user, translated_text, message_type="translation")

                    return
                except TypeError as e:
                    # Try without message_type parameter as a fallback                    logger.debug(f"Error with message_type parameter: {e}, trying without it")
                    await self.translation_callback(user, transcribed_text)
                    if transcribed_text != translated_text:
                        await self.translation_callback(user, translated_text)
                    return

        except (AttributeError, TypeError, ValueError) as e:
            logger.error(f"Error updating display for user {user}: {e}")
            import traceback
            logger.debug(f"TEXT-FLOW: Traceback: {traceback.format_exc()}")

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
                        logger.debug(
                            f"Timer detected inactive speaker {user}. Processing speech.")
                        user_state.is_speaking = False
                        self.process_speech_buffer(user)
                        user_state.speech_detected = False
                        user_state.speech_buffer = io.BytesIO()
        except (KeyError, AttributeError, TypeError, ValueError) as e:
            logger.error(f"Error checking inactive speakers: {e}")
        finally:
            # Reschedule the timer if still running
            if self.workers.running:
                self.workers.timer = threading.Timer(
                    0.5, self._check_inactive_speakers)
                self.workers.timer.daemon = True
                self.workers.timer.start()

    def cleanup(self):
        """Clean up resources when the sink is no longer needed."""
        logger.info("Starting sink cleanup process...")

        # Set running flag to False
        logger.debug("Setting worker running flag to False")
        self.workers.running = False

        # Cancel timer if it exists
        if self.workers.timer:
            logger.debug("Canceling worker timer")
            self.workers.timer.cancel()

        # Add a small delay to let threads terminate naturally
        logger.debug("Waiting for threads to terminate naturally")
        time.sleep(0.2)

        # Clear the queue to unblock any waiting threads
        try:
            logger.debug(
                f"Clearing queue (current size: {self.workers.queue.qsize()})")
            while not self.workers.queue.empty():
                try:
                    item = self.workers.queue.get_nowait()
                    logger.debug(f"Removed item from queue: {item[0]}")
                    self.workers.queue.task_done()
                except queue.Empty:
                    logger.debug("Queue empty exception while clearing")
                    break
        except (ValueError, RuntimeError) as e:
            logger.error(f"Error clearing queue: {e}")

        logger.info(
            f"Cleaning up sink resources - {self.workers.num_workers} workers stopped")
        # Print thread status
        for i, worker in enumerate(self.workers.workers):
            logger.debug(f"Worker {i} alive: {worker.is_alive()}")

        # Worker threads are daemon threads and will terminate when the program exits
