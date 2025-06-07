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
import wave
import queue
import threading
import gc
import subprocess
import traceback
import uuid
import struct
import io
import numpy as np
from dataclasses import dataclass
from queue import Queue
from typing import Optional, Callable, Awaitable
from discord.sinks import WaveSink
import utils
import translation
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
            self.queue = Queue(maxsize=50)
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

    def __init__(self, *args, pause_threshold=1.0, event_loop=None, num_workers=6, **kwargs):
        super().__init__(*args, **kwargs)

        # Initialize the parent class reference
        self.parent = None

        # Correctly annotate and initialize translation_callback only here
        self.translation_callback: Optional[Callable[...,
                                                     Awaitable[None]]] = None

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
                daemon=True, name=f"TranscriptionWorker-{i+1}"
            )
            thread.start()
            self.workers.workers.append(thread)
            logger.info("Started transcription worker %d/%d",
                        i+1, self.workers.num_workers)

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

    def write(self, data, user):
        """Process incoming audio data from Discord."""
        try:
            # PROTECTION: Validate incoming audio data before processing
            if not data or len(data) < 4:
                return

            current_time = time.time()

            # Check if user is enabled for processing - CRITICAL LOGIC
            if (hasattr(self, 'parent') and
                    self.parent and
                    hasattr(self.parent, 'user_processing_enabled')):
                if not self.parent.user_processing_enabled.get(str(user), True):
                    # Still write to main buffer but skip processing
                    try:
                        super().write(data, user)
                    except Exception as e:
                        logger.debug(
                            "Error writing to main buffer for disabled user %s: %s", user, str(e))
                    return

            # Continue with normal audio processing for enabled users
            # Update last activity time when any audio is processed
            self.session.last_activity_time = current_time

            # Initialize user-specific data structures if needed
            user_state = self._get_user_state(user)

            # First time seeing this user?
            if user_state.last_packet_time == 0:
                logger.debug("First audio packet from user %s", user)
                user_state.last_packet_time = current_time
                user_state.last_active_time = current_time

            # PROTECTION: Validate audio data for activity detection
            try:
                is_active = self.is_audio_active(data, user)
            except (ValueError, TypeError, OverflowError) as e:
                logger.debug(
                    "Audio activity detection failed for user %s: %s", user, str(e))
                is_active = False

            # Update last active time if speech is detected
            if is_active:
                user_state.last_active_time = current_time

            # Calculate time differences
            time_diff = current_time - user_state.last_packet_time
            active_diff = current_time - user_state.last_active_time

            # CRITICAL FIX: Only process long pause if we actually have speech detected
            # And increase the pause threshold to prevent the crazy loop
            if (user_state.is_speaking and
                    user_state.speech_detected and
                    active_diff > 2.0):  # Increased from 0.8 to 2.0 seconds
                self._process_silent_speech(user)

            # CRITICAL FIX: Increase pause threshold and add rate limiting
            if time_diff > 2.0:  # Increased from 1.0 to 2.0 seconds
                # Add rate limiting to prevent log spam
                now = time.time()
                last_log_time = self.last_block_log_time.get(user, 0)
                if now - last_log_time > 5.0:  # Only log once every 5 seconds per user
                    logger.debug(
                        "Long pause detected for user %s. Processing any speech and resetting.", user)
                    self.last_block_log_time[user] = now

                self._handle_long_pause(user)

            # Store recent frames for smoother beginning of speech
            self._update_pre_speech_buffer(user, data)

            if is_active:
                self._handle_active_speech(user, data)
            else:
                self._handle_silence(user, data)

            user_state.last_packet_time = current_time

            # PROTECTION: Write to the main buffer with error handling
            try:
                super().write(data, user)
            except Exception as e:
                logger.debug(
                    "Error writing to main buffer for user %s: %s", user, str(e))

        except (KeyError, TypeError, ValueError, AttributeError, IOError, RuntimeError) as e:
            logger.error("Error in write method for user %s: %s", user, str(e))
        except Exception as e:
            # Catch any other unexpected errors to prevent thread crashes
            logger.error(
                "Unexpected error in write method for user %s: %s", user, str(e))

    def is_audio_active(self, audio_data, user):
        """Check if audio data contains active speech."""
        try:
            # PROTECTION: Validate audio data before processing
            if not audio_data or len(audio_data) < 2:
                return False

            # Convert bytes to numpy array (assuming PCM signed 16-bit little-endian)
            audio_array = np.frombuffer(audio_data, dtype=np.int16)

            # PROTECTION: Check for empty array
            if len(audio_array) == 0:
                return False

            # Calculate energy of the current frame
            energy = np.mean(np.abs(audio_array))

            # Get user state
            user_state = self._get_user_state(user)

            # Initialize energy history if needed
            if not user_state.energy_history:
                user_state.energy_history = []

            # Update energy history
            user_state.energy_history.append(energy)
            # Keep last 5 frames
            user_state.energy_history = user_state.energy_history[-5:]

            # Calculate average energy over recent frames
            avg_energy = np.mean(user_state.energy_history)

            # IMPROVED: Slightly lower threshold for better sensitivity
            return avg_energy > 250  # Reduced from 300 for better pickup

        except (ValueError, TypeError, OverflowError, np.core._exceptions._ArrayMemoryError) as e:
            logger.warning(
                "Audio activity detection error for user %s: %s", user, str(e))
            return False  # Default to inactive for corrupted audio

    def _update_pre_speech_buffer(self, user, data):
        """Update the pre-speech buffer for smoother speech beginning."""
        user_state = self._get_user_state(user)
        user_state.pre_speech_buffer.append(data)
        if len(user_state.pre_speech_buffer) > 10:  # Keep ~200ms of audio
            user_state.pre_speech_buffer.pop(0)

    def _process_silent_speech(self, user):
        """Process speech after detecting significant silence."""
        logger.debug(
            "Significant pause detected for user %s. Processing speech.", user)
        user_state = self._get_user_state(user)
        user_state.is_speaking = False
        self.process_speech_buffer(user)
        user_state.speech_detected = False
        user_state.silence_frames = 0
        user_state.speech_buffer = io.BytesIO()

    def _handle_long_pause(self, user):
        """Handle a long pause in audio packets."""
        logger.debug(
            "Long pause detected for user %s. Processing any speech and resetting.", user)

        user_state = self._get_user_state(user)

        # Process any accumulated speech before resetting
        if user_state.is_speaking and user_state.speech_detected:
            logger.debug(
                "Processing speech before resetting for user %s", user)
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
            logger.debug("Speech started for user %s", user)
            user_state.is_speaking = True
            # Add pre-speech frames for smoother beginning
            for pre_data in user_state.pre_speech_buffer:
                user_state.speech_buffer.write(pre_data)

        # Add this audio data to the speech buffer
        try:
            user_state.speech_buffer.write(data)
        except (IOError, OSError) as e:
            logger.warning(
                "Failed to write speech data for user %s: %s", user, str(e))

    def _handle_silence(self, user, data):
        """Handle silence after speech."""
        user_state = self._get_user_state(user)

        # Add a few frames to the speech buffer for smoother transitions
        if user_state.is_speaking and user_state.silence_frames < 5:
            try:
                user_state.speech_buffer.write(data)
            except (IOError, OSError) as e:
                logger.warning(
                    "Failed to write silence data for user %s: %s", user, str(e))

        # Increment silence counter
        # Check if silence has persisted long enough to consider speech ended
        user_state.silence_frames += 1
        if (user_state.is_speaking and
                user_state.silence_frames > self.config.silence_threshold):
            logger.debug("Speech ended for user %s. Processing audio.", user)
            # Only process if speech was detected in this session
            if user_state.speech_detected:
                self.process_speech_buffer(user)
                user_state.speech_detected = False
                user_state.silence_frames = 0
                user_state.speech_buffer = io.BytesIO()

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
        min_buffer_size = 25000  # Reduced from 38400 (~0.13s instead of 0.2s)

        if duration_seconds < min_duration or buffer_size < min_buffer_size:
            logger.debug("Speech too short (%.2fs < %.2fs)" +
                         " or buffer too small (%d < %d bytes), skipping.",
                         duration_seconds, min_duration, buffer_size, min_buffer_size)
            return

        # IMPROVED: Reduce cooldown for better responsiveness
        if user_state.last_processed_time > 0:
            time_since_last = current_time - user_state.last_processed_time
            if time_since_last < 1.5:
                logger.debug("Cooldown active for user %s, skipping.", user)
                return

        # Create and process audio file
        self._create_and_queue_audio_file(user, user_state, current_time)

    def _create_and_queue_audio_file(self, user, user_state, current_time):
        """Create audio file and queue for transcription."""
        try:
            # Create unique filename
            audio_filename = f"speech_{user}_{int(current_time * 1000)}.wav"

            # Write audio buffer to file
            user_state.speech_buffer.seek(0)
            audio_data = user_state.speech_buffer.read()

            with wave.open(audio_filename, 'wb') as wav_file:
                wav_file.setnchannels(2)  # Discord uses stereo
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(48000)  # Discord sample rate
                wav_file.writeframes(audio_data)

            # Queue for transcription
            try:
                self.workers.queue.put((user, audio_filename), timeout=1.0)
                user_state.last_processed_time = current_time
                logger.debug("Queued audio file for user %s: %s",
                             user, audio_filename)
            except queue.Full:
                logger.warning(
                    "Transcription queue full, dropping audio for user %s", user)
                os.remove(audio_filename)

        except Exception as e:
            logger.error(
                "Error creating audio file for user %s: %s", user, str(e))

    def _update_session_state(self, user):
        """Update session state and return current time."""
        current_time = time.time()

        # Update session tracking
        if user not in self.session.current_speakers:
            self.session.current_speakers.add(user)
            self.session.last_speaker_change = current_time

        self.session.last_activity_time = current_time

        # Update session state based on activity
        session_duration = current_time - self.session.start_time
        if session_duration > 30:  # 30 seconds
            self.session.state = "established"
        elif len(self.session.current_speakers) > 1:
            self.session.state = "active"

        return current_time

    def _get_audio_duration(self, speech_buffer):
        """Calculate audio duration from buffer size."""
        if not speech_buffer:
            return 0.0

        # Get buffer size
        speech_buffer.seek(0, os.SEEK_END)
        buffer_size = speech_buffer.tell()

        # Discord audio is 48kHz, 16-bit, stereo = 192,000 bytes per second
        bytes_per_second = 48000 * 2 * 2  # sample_rate * channels * bytes_per_sample
        duration = buffer_size / bytes_per_second

        return duration

    def _get_minimum_duration(self):
        """Get minimum duration based on session state."""
        if self.session.state == "new":
            return 0.3  # 300ms for new sessions
        elif self.session.state == "active":
            return 0.2  # 200ms for active sessions
        else:  # established
            return 0.15  # 150ms for established sessions

    def _transcription_worker(self):
        """Background worker to process transcription queue."""
        thread_name = threading.current_thread().name
        logger.info("%s started", thread_name)

        # CRITICAL FIX: Create a proper event loop for this worker thread
        loop = None
        try:
            # CRITICAL FIX: Don't try to get existing loop, always create new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            logger.debug("%s created new event loop", thread_name)
        except Exception as e:
            logger.error("%s failed to create event loop: %s",
                         thread_name, str(e))
            return

        while self.workers.running:
            audio_file = None
            user = None

            try:
                # Get item from queue with timeout
                user, audio_file = self.workers.queue.get(
                    timeout=1.0)  # CRITICAL FIX: Correct order
                logger.debug("%s processing: %s for user %s",
                             thread_name, audio_file, user)

                # Process the transcription
                loop.run_until_complete(
                    self.transcribe_audio(audio_file, user))

            except queue.Empty:
                continue  # No work available, continue loop

            except Exception as e:
                logger.error("%s error processing queue item: %s",
                             thread_name, str(e))
                # Clean up the audio file if processing failed
                if audio_file and os.path.exists(audio_file):
                    try:
                        os.remove(audio_file)
                        logger.debug(
                            "Cleaned up failed audio file: %s", audio_file)
                    except Exception as cleanup_error:
                        logger.debug(
                            "Failed to cleanup audio file: %s", str(cleanup_error))
                continue

            finally:
                # Mark task as done if we got an item
                if 'audio_file' in locals() and audio_file is not None:
                    self.workers.queue.task_done()

        # CRITICAL FIX: Properly close the event loop when worker exits
        try:
            if loop and not loop.is_closed():
                loop.close()
                logger.debug("%s closed event loop", thread_name)
        except Exception as e:
            logger.error("%s error closing event loop: %s",
                         thread_name, str(e))

        logger.debug("%s exiting", thread_name)

    async def transcribe_audio(self, audio_file, user):
        """Transcribe the audio file with smart model routing and update the GUI with results."""
        logger.debug(
            "🔄 Starting transcription for user %s, file: %s", user, audio_file)

        try:
            # Skip if file doesn't exist
            if not os.path.exists(audio_file):
                logger.warning("Audio file not found: %s", audio_file)
                return

            transcribed_text = None
            detected_language = None

            try:
                logger.debug(
                    "📞 Calling utils.transcribe for file: %s", audio_file)
                transcribed_text, detected_language = await utils.transcribe(audio_file)
                logger.debug("🎯 Transcription result for user %s: text='%s', language=%s",
                             user, transcribed_text, detected_language)

            except Exception as e:
                logger.error(
                    "Transcription failed for user %s: %s", user, str(e))
                return

            # Skip processing if transcription was empty (failed confidence check)
            if not transcribed_text or not transcribed_text.strip():
                logger.debug("Empty transcription result for user %s", user)
                return

            logger.info("✅ Transcription for user %s: %s",
                        user, transcribed_text)

            # CRITICAL FIX: Process the transcription and translation
            try:
                logger.debug("📡 Sending transcription for user %s", user)
                await self._update_gui(user, transcribed_text, None, "transcription")

                # Check if translation is needed
                logger.debug("🌐 Starting translation check for user %s", user)
                should_translate_result = await utils.should_translate(transcribed_text, detected_language)
                logger.debug("🔍 Translation needed for user %s: %s",
                             user, should_translate_result)

                if should_translate_result:
                    # Translate the text
                    logger.debug("🌍 Translating text for user %s", user)
                    translated_text = await utils.translate(transcribed_text)
                    logger.debug(
                        "🌍 Translation result for user %s: %s", user, translated_text)

                    if translated_text and translated_text.strip():
                        logger.debug("📡 Sending translation for user %s", user)
                        await self._update_gui(user, translated_text, None, "translation")
                    else:
                        logger.warning("Translation failed for user %s", user)
                else:
                    # CRITICAL FIX: Even if no translation is needed (English),
                    # still send the original text to the translation container
                    logger.debug(
                        "⏭️ No translation needed for user %s (language: %s), sending original text to translation container", user, detected_language)
                    await self._update_gui(user, transcribed_text, None, "translation")

            except Exception as e:
                logger.error(
                    "Error processing transcription/translation for user %s: %s", user, str(e))

        except Exception as e:
            logger.error(
                "Error in transcribe_audio for user %s: %s", user, str(e))

        finally:
            logger.debug(
                "🏁 Transcription function completing for user %s", user)
            # Clean up the audio file
            if audio_file and os.path.exists(audio_file):
                await self._force_delete_file(audio_file)
            logger.debug(
                "🔚 transcribe_audio finally block completed for user %s", user)

    async def _force_delete_file(self, file_path):
        """Forcefully delete a file with multiple retries and GC."""
        if not file_path or not os.path.exists(file_path):
            return True

        # Try to delete the file with multiple retries
        for attempt in range(5):
            try:
                os.unlink(file_path)
                logger.debug("Deleted file: %s (attempt %d)",
                             os.path.basename(file_path), attempt + 1)
                return True
            except (OSError, PermissionError) as e:
                if attempt < 4:
                    await asyncio.sleep(0.1)
                    continue
                logger.debug("Delete failed attempt %d: %s",
                             attempt + 1, str(e))

        # Last resort: try with Windows-specific commands on Windows
        if os.name == 'nt':
            try:
                import subprocess
                subprocess.run(['del', '/f', file_path],
                               shell=True, check=False, capture_output=True)
                return True
            except:
                pass

        # Final attempt: rename the file to mark for deletion
        try:
            temp_name = file_path + '.delete_me'
            os.rename(file_path, temp_name)
            logger.debug("Renamed file for deletion: %s", temp_name)
            return True
        except (OSError, PermissionError) as e:
            logger.debug("Rename failed: %s", str(e))

        logger.error(
            "CRITICAL: Failed to delete file after all attempts: %s", file_path)
        return False

    async def _update_gui(self, user, text, translated_text, message_type):
        """Update the GUI with transcription and translation results."""
        logger.debug(
            "_update_gui called for user %s with message_type: %s", user, message_type)

        try:
            if self.translation_callback:
                logger.debug(
                    "Calling translation_callback for user %s with type %s", user, message_type)
                await self.translation_callback(user, text, message_type)
            else:
                logger.warning(
                    "No translation callback available for user %s", user)

        except Exception as e:
            logger.error("Error in _update_gui for user %s: %s", user, str(e))

    def _check_inactive_speakers(self):
        """Periodically check for users who have stopped speaking but haven't been processed."""
        try:
            current_time = time.time()

            # CRITICAL FIX: Add queue monitoring
            queue_size = self.workers.queue.qsize()
            if queue_size > 30:
                logger.warning("⚠️ High queue load: %d items", queue_size)
            elif queue_size > 0:
                logger.debug("Queue activity: %d items processing", queue_size)

            # Check each user
            for user, user_state in list(self.users.items()):
                try:
                    # Check for users who have been inactive for too long
                    if (user_state.is_speaking and
                        user_state.speech_detected and
                            current_time - user_state.last_active_time > 3.0):
                        logger.debug(
                            "Timer detected inactive speaker %s. Processing speech.", user)
                        self.process_speech_buffer(user)
                        user_state.is_speaking = False
                        user_state.speech_detected = False
                        user_state.silence_frames = 0
                        user_state.speech_buffer = io.BytesIO()
                except Exception as user_error:
                    logger.error(
                        "Error processing inactive user %s: %s", user, str(user_error))

        except (KeyError, AttributeError, TypeError, ValueError) as e:
            logger.error("Error checking inactive speakers: %s", e)
        finally:
            # Reschedule the timer if still running
            if self.workers.running:
                try:
                    self.workers.timer = threading.Timer(
                        1.0, self._check_inactive_speakers)
                    self.workers.timer.daemon = True
                    self.workers.timer.start()
                except Exception as timer_error:
                    logger.error("Failed to restart timer: %s",
                                 str(timer_error))

    def cleanup(self):
        """Clean up resources when the sink is no longer needed."""
        logger.info("Starting sink cleanup process...")

        # Set running flag to False
        logger.debug("Setting worker running flag to False")
        self.workers.running = False

        # Cancel timer if it exists
        if self.workers.timer:
            try:
                self.workers.timer.cancel()
            except:
                pass

        # Add a small delay to let threads terminate naturally
        logger.debug("Waiting for threads to terminate naturally")
        time.sleep(0.5)  # Increased from 0.2 to 0.5 for more threads

        # Clear the queue to unblock any waiting threads
        try:
            while not self.workers.queue.empty():
                try:
                    self.workers.queue.get_nowait()
                    self.workers.queue.task_done()
                except:
                    break
        except (ValueError, RuntimeError) as e:
            logger.debug("Queue clear error: %s", str(e))

        # Wait for all worker threads to finish with timeout
        logger.debug("Waiting for worker threads to join...")
        for i, worker in enumerate(self.workers.workers):
            try:
                worker.join(timeout=2.0)
                logger.debug("Worker thread %d successfully joined", i + 1)
            except:
                logger.warning("Worker thread %d failed to join", i + 1)

        logger.info("Cleaning up sink resources - %d workers processed",
                    self.workers.num_workers)

        # Print final thread status
        alive_threads = sum(
            1 for worker in self.workers.workers if worker.is_alive())
        if alive_threads > 0:
            logger.warning(
                "%d worker threads still alive after cleanup", alive_threads)
        else:
            logger.info("All worker threads successfully terminated")
