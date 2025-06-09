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
    # Add health monitoring fields
    worker_health: dict = None
    last_health_check: float = 0
    health_check_interval: float = 30.0  # Check every 30 seconds
    worker_restart_count: dict = None
    max_restarts_per_worker: int = 3

    def __post_init__(self):
        """Initialize mutable default values."""
        if self.queue is None:
            self.queue = Queue(maxsize=50)
        if self.workers is None:
            self.workers = []
        if self.worker_health is None:
            self.worker_health = {}
        if self.worker_restart_count is None:
            self.worker_restart_count = {}

    def mark_worker_healthy(self, worker_id: str):
        """Mark a worker as healthy with current timestamp."""
        self.worker_health[worker_id] = time.time()

    def is_worker_healthy(self, worker_id: str, timeout: float = 60.0) -> bool:
        """Check if a worker is healthy (responded within timeout)."""
        last_response = self.worker_health.get(worker_id, 0)
        return (time.time() - last_response) < timeout

    def get_unhealthy_workers(self, timeout: float = 60.0) -> list:
        """Get list of worker IDs that haven't responded within timeout."""
        current_time = time.time()
        unhealthy = []
        for worker_id, last_response in self.worker_health.items():
            if (current_time - last_response) > timeout:
                unhealthy.append(worker_id)
        return unhealthy


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
            worker_id = f"TranscriptionWorker-{i+1}"
            thread = threading.Thread(
                target=self._transcription_worker,
                args=(worker_id,),
                daemon=True,
                name=worker_id
            )
            thread.start()
            # Initialize worker health tracking
            self.workers.workers.append(thread)
            self.workers.mark_worker_healthy(worker_id)
            self.workers.worker_restart_count[worker_id] = 0
            logger.info("Started transcription worker %d/%d (%s)",
                        i+1, self.workers.num_workers, worker_id)

    def _start_processing_timer(self):
        """Start the timer to check for inactive speakers."""
        self.workers.timer = threading.Timer(
            1.0, self._check_inactive_speakers)
        self.workers.timer.daemon = True
        self.workers.timer.start()

    def _check_worker_health(self):
        """Check health of all worker threads and restart failed ones."""
        current_time = time.time()

        # Check health more frequently for better crash detection
        if (current_time - self.workers.last_health_check) < 15.0:  # Reduced from 30s to 15s
            return

        self.workers.last_health_check = current_time
        logger.debug("Performing enhanced worker health check...")

        # Check for unhealthy workers (reduced timeout for faster detection)
        unhealthy_workers = self.workers.get_unhealthy_workers(
            timeout=45.0)  # Reduced from 90s
        dead_workers = []

        # Check for dead threads
        for i, worker in enumerate(self.workers.workers):
            worker_id = worker.name
            if not worker.is_alive():
                dead_workers.append((i, worker_id))
                logger.warning(
                    "⚠️ Worker thread %s is dead - will restart", worker_id)

        # CRITICAL: Also check for workers that might be stuck
        stuck_workers = []
        queue_size = self.workers.queue.qsize()

        # If queue is backing up and workers aren't responding, they might be stuck
        if queue_size > 20:
            for worker_id in self.workers.worker_health.keys():
                last_health = self.workers.worker_health.get(worker_id, 0)
                if (current_time - last_health) > 30.0:  # Haven't checked in for 30s with high queue
                    stuck_workers.append((None, worker_id))
                    logger.warning("⚠️ Worker %s appears stuck (queue size: %d) - will restart",
                                   worker_id, queue_size)

        # Restart dead, unresponsive, or stuck workers
        workers_to_restart = dead_workers + \
            [(None, w) for w in unhealthy_workers if w not in [d[1] for d in dead_workers]] + \
            stuck_workers

        if workers_to_restart:
            logger.warning("🔧 Found %d workers that need restart (dead: %d, unhealthy: %d, stuck: %d)",
                           len(workers_to_restart), len(dead_workers), len(unhealthy_workers), len(stuck_workers))

        for worker_index, worker_id in workers_to_restart:
            restart_count = self.workers.worker_restart_count.get(worker_id, 0)

            if restart_count >= self.workers.max_restarts_per_worker:
                logger.error("❌ Worker %s has exceeded max restart attempts (%d), not restarting",
                             worker_id, self.workers.max_restarts_per_worker)
                continue

            logger.warning(
                "🔄 Restarting problematic worker: %s (restart #%d)", worker_id, restart_count + 1)

            # Create new worker thread
            new_thread = threading.Thread(
                target=self._transcription_worker,
                args=(worker_id,),
                daemon=True,
                name=worker_id
            )
            new_thread.start()

            # Update worker list
            if worker_index is not None:
                self.workers.workers[worker_index] = new_thread
            else:
                # Find and replace the worker by name
                for i, w in enumerate(self.workers.workers):
                    if w.name == worker_id:
                        self.workers.workers[i] = new_thread
                        break

            # Reset health tracking
            self.workers.mark_worker_healthy(worker_id)
            self.workers.worker_restart_count[worker_id] = restart_count + 1

            logger.info("✅ Successfully restarted worker: %s", worker_id)

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
                # PROTECTION: Validate audio data for activity detection using combined VAD
                user_state.last_active_time = current_time
            try:
                is_active = self._should_process_audio(data, user)
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

        except (ValueError, TypeError, OverflowError, MemoryError) as e:
            logger.warning(
                "Audio activity detection error for user %s: %s", user, str(e))
            # Default to inactive for corrupted audio
            return False

    def _update_pre_speech_buffer(self, user, data):
        """Update the pre-speech buffer for smoother speech beginning."""
        user_state = self._get_user_state(user)

        # CRITICAL FIX: Only keep pre-speech buffer if recent activity detected
        # Check if this frame has any energy to avoid accumulating silence
        try:
            # Quick energy check on current frame
            audio_array = np.frombuffer(data, dtype=np.int16)
            frame_energy = np.mean(np.abs(audio_array)) if len(
                audio_array) > 0 else 0

            # Only add to pre-speech buffer if there's some energy OR we're already speaking
            if frame_energy > 100 or user_state.is_speaking:
                user_state.pre_speech_buffer.append(data)
                if len(user_state.pre_speech_buffer) > 5:  # Reduced from 10 to 5 frames (~100ms)
                    user_state.pre_speech_buffer.pop(0)
            else:
                # Clear pre-speech buffer during silence to prevent accumulation
                if len(user_state.pre_speech_buffer) > 0:
                    user_state.pre_speech_buffer = []

        except (ValueError, TypeError) as e:
            # Fallback: just maintain buffer size
            user_state.pre_speech_buffer.append(data)
            if len(user_state.pre_speech_buffer) > 5:
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
        # Buffer is now reset immediately in _create_and_queue_audio_file

    def _handle_long_pause(self, user):
        """Handle a long pause in audio packets."""
        logger.debug(
            "Long pause detected for user %s. Processing any speech and resetting.", user)

        user_state = self._get_user_state(user)

        # Process any accumulated speech before resetting
        if user_state.is_speaking and user_state.speech_detected:
            logger.debug(
                "Processing speech before resetting for user %s", user)
            self.process_speech_buffer(user)        # Reset state
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

        # If this is the start of speech, reset buffer and add pre-speech buffer
        if not user_state.is_speaking:
            logger.debug("Speech started for user %s", user)
            user_state.is_speaking = True

            # CRITICAL FIX: Reset speech buffer to prevent accumulation of previous silence
            user_state.speech_buffer = io.BytesIO()

            # Add pre-speech frames for smoother beginning, but limit to avoid silence accumulation
            pre_speech_count = len(user_state.pre_speech_buffer)
            if pre_speech_count > 0:
                logger.debug(
                    "Adding %d pre-speech frames for user %s", pre_speech_count, user)
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
        # CRITICAL FIX: Only process speech once when threshold is first exceeded
        user_state.silence_frames += 1
        if (user_state.is_speaking and
            user_state.silence_frames == self.config.silence_threshold + 1 and  # Only trigger once
                user_state.speech_detected):  # Only if speech was actually detected
            logger.debug("Speech ended for user %s. Processing audio.", user)
            self.process_speech_buffer(user)

            # CRITICAL FIX: Reset state immediately to prevent re-processing
            user_state.is_speaking = False
            user_state.speech_detected = False
            user_state.silence_frames = 0
            # Buffer is now reset immediately in _create_and_queue_audio_file

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
            return        # IMPROVED: Reduce cooldown for better responsiveness
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
            audio_filename = f"{user}_{int(current_time * 1000)}_speech.wav"

            # Write audio buffer to file
            user_state.speech_buffer.seek(0)
            audio_data = user_state.speech_buffer.read()

            # CRITICAL FIX: Reset buffer immediately after reading to prevent accumulation
            user_state.speech_buffer = io.BytesIO()

            with wave.open(audio_filename, 'wb') as wav_file:
                wav_file.setnchannels(2)  # Discord uses stereo
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(48000)  # Discord sample rate
                # Validate audio file before queueing
                wav_file.writeframes(audio_data)
            is_valid, validation_msg = self._validate_audio_file(
                audio_filename)
            if not is_valid:
                logger.warning(
                    "Audio validation failed for user %s: %s", user, validation_msg)
                try:
                    os.remove(audio_filename)
                except Exception:
                    pass
                return

            # Manage queue overflow before adding new item
            self._manage_queue_overflow()

            # Queue for transcription
            try:
                self.workers.queue.put((user, audio_filename), timeout=1.0)
                user_state.last_processed_time = current_time
                logger.debug("Queued audio file for user %s: %s",
                             user, audio_filename)
            except queue.Full:
                logger.warning(
                    "Transcription queue full after overflow management, dropping audio for user %s", user)
                try:
                    os.remove(audio_filename)
                except Exception:
                    pass

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
            # 150ms for established sessions
            return 0.15

    def _transcription_worker(self, worker_id):
        """Background worker to process transcription queue with robust error handling and crash recovery."""
        thread_name = threading.current_thread().name
        logger.info("%s started with crash recovery", thread_name)

        consecutive_failures = 0
        max_consecutive_failures = 5
        processing_count = 0

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

        # Mark worker as healthy initially
        if worker_id:
            self.workers.mark_worker_healthy(worker_id)

        while self.workers.running:
            audio_file = None
            user = None
            queue_item_acquired = False

            try:
                # Get item from queue with timeout
                user, audio_file = self.workers.queue.get(timeout=1.0)
                queue_item_acquired = True

                logger.debug("%s processing: %s for user %s (count: %d)",
                             thread_name, audio_file, user, processing_count + 1)

                # CRITICAL: Pre-validate audio file exists before processing
                if not audio_file or not os.path.exists(audio_file):
                    logger.warning(
                        "%s skipping missing audio file: %s", thread_name, audio_file)
                    continue

                # Process the transcription with timeout protection
                try:
                    # CRITICAL: Add timeout to prevent hanging on transcription
                    transcription_task = self.transcribe_audio(
                        audio_file, user)
                    await_result = asyncio.wait_for(
                        transcription_task, timeout=60.0)
                    loop.run_until_complete(await_result)

                    # Successful processing - reset failure counter
                    consecutive_failures = 0
                    processing_count += 1

                    # Mark worker as healthy after successful processing
                    if worker_id:
                        self.workers.mark_worker_healthy(worker_id)

                    logger.debug("%s successfully processed item %d",
                                 thread_name, processing_count)

                except asyncio.TimeoutError:
                    logger.error("%s transcription timeout for file %s (user %s)",
                                 thread_name, audio_file, user)
                    consecutive_failures += 1
                    # Clean up the hung audio file
                    if audio_file and os.path.exists(audio_file):
                        try:
                            os.remove(audio_file)
                            logger.warning(
                                "%s cleaned up timeout audio file: %s", thread_name, audio_file)
                        except Exception:
                            pass

                except Exception as transcription_error:
                    logger.error("%s transcription error for %s: %s",
                                 thread_name, audio_file, str(transcription_error))
                    consecutive_failures += 1
                    # Clean up the failed audio file
                    if audio_file and os.path.exists(audio_file):
                        try:
                            os.remove(audio_file)
                            logger.warning(
                                "%s cleaned up failed audio file: %s", thread_name, audio_file)
                        except Exception:
                            pass

            except queue.Empty:
                # No work available - periodic health check
                if worker_id:
                    self.workers.mark_worker_healthy(worker_id)
                continue

            except Exception as queue_error:
                logger.error("%s critical error in queue processing: %s",
                             thread_name, str(queue_error))
                consecutive_failures += 1

                # CRITICAL: If we can't even process the queue, something is very wrong
                if consecutive_failures >= max_consecutive_failures:
                    logger.error("%s experienced %d consecutive failures, exiting to trigger restart",
                                 thread_name, consecutive_failures)
                    break

            finally:
                # CRITICAL: Always mark task as done if we acquired a queue item
                if queue_item_acquired:
                    try:
                        self.workers.queue.task_done()
                    except Exception as task_done_error:
                        logger.error("%s error marking task done: %s",
                                     thread_name, str(task_done_error))

        # Worker is exiting - log final statistics
        logger.warning("%s exiting after processing %d items (consecutive failures: %d)",
                       thread_name, processing_count, consecutive_failures)

        # CRITICAL FIX: Properly close the event loop when worker exits
        try:
            if loop and not loop.is_closed():
                loop.close()
                logger.debug("%s closed event loop", thread_name)
        except Exception as e:
            logger.error("%s error closing event loop: %s",
                         thread_name, str(e))

        logger.info("%s exiting", thread_name)

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
                # Graceful degradation: send error notice to UI
                error_msg = f"[Transcription Error: {str(e)[:50]}...]"
                try:
                    await self._update_gui(user, error_msg, None, "transcription")
                except Exception:
                    pass
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

            # CRITICAL FIX: Reduce queue monitoring frequency to prevent log spam
            queue_size = self.workers.queue.qsize()

            # Only log queue status every 10 seconds to reduce spam
            if not hasattr(self, '_last_queue_log_time'):
                self._last_queue_log_time = 0

            if current_time - self._last_queue_log_time > 10.0:  # Every 10 seconds
                if queue_size > 30:
                    logger.warning("⚠️ High queue load: %d items", queue_size)
                elif queue_size > 0:
                    logger.debug(
                        "Queue activity: %d items processing", queue_size)
                self._last_queue_log_time = current_time

            # Check each user with rate limiting
            for user, user_state in list(self.users.items()):
                try:
                    # CRITICAL FIX: Only process inactive speakers once per user
                    if (user_state.is_speaking and
                        user_state.speech_detected and
                            current_time - user_state.last_active_time > 3.0):

                        # Add rate limiting per user to prevent repeated processing
                        if not hasattr(user_state, 'last_inactive_check'):
                            # Only process if we haven't checked this user recently
                            user_state.last_inactive_check = 0
                        if current_time - user_state.last_inactive_check > 2.0:
                            logger.debug(
                                "Timer detected inactive speaker %s. Processing speech.", user)
                            self.process_speech_buffer(user)
                            user_state.is_speaking = False
                            user_state.speech_detected = False
                            user_state.silence_frames = 0
                            # Buffer is now reset immediately in _create_and_queue_audio_file
                            user_state.last_inactive_check = current_time

                except Exception as user_error:
                    logger.error(
                        "Error processing inactive user %s: %s", user, str(user_error))

            # CRITICAL: Perform worker health check
            self._check_worker_health()

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

    def _manage_queue_overflow(self):
        """Manage queue overflow by dropping oldest items when queue is too full."""
        queue_size = self.workers.queue.qsize()
        max_queue_size = 35  # Lower than maxsize=50 to prevent total overflow

        if queue_size > max_queue_size:
            dropped_count = 0
            # Drop up to 10 oldest items to make room
            max_drops = min(10, queue_size - 25)  # Target queue size of 25

            logger.warning("Queue overflow detected (%d items), dropping %d oldest items",
                           queue_size, max_drops)

            for _ in range(max_drops):
                try:
                    user, audio_file = self.workers.queue.get_nowait()
                    # Clean up the dropped audio file
                    if audio_file and os.path.exists(audio_file):
                        try:
                            os.remove(audio_file)
                            logger.debug(
                                "Cleaned up dropped audio file: %s", audio_file)
                        except Exception:
                            pass
                    dropped_count += 1
                    self.workers.queue.task_done()
                except queue.Empty:
                    break

            if dropped_count > 0:
                logger.warning(
                    "Dropped %d audio files due to queue overflow", dropped_count)

        return queue_size

    def _validate_audio_file(self, audio_file_path):
        """Validate audio file before processing to detect corruption."""
        try:
            if not os.path.exists(audio_file_path):
                return False, "File does not exist"

            # Check file size (minimum 10KB for meaningful audio)
            file_size = os.path.getsize(audio_file_path)
            if file_size < 10000:
                return False, f"File too small: {file_size} bytes"

            # Try to open with wave to validate format
            with wave.open(audio_file_path, 'rb') as wav_file:
                frames = wav_file.getnframes()
                framerate = wav_file.getframerate()
                duration = frames / framerate if framerate > 0 else 0

                if duration < 0.1:  # Less than 100ms
                    return False, f"Audio too short: {duration:.2f}s"

                if duration > 300.0:  # More than 5 minutes (300 seconds)
                    return False, f"Audio too long: {duration:.2f}s"

            return True, "Valid"

        except (wave.Error, OSError, IOError) as e:
            return False, f"Wave format error: {str(e)}"
        except Exception as e:
            return False, f"Validation error: {str(e)}"

    def _is_user_speaking_discord_vad(self, user):
        """Check if user is speaking according to Discord's built-in VAD.

        This uses Discord's own voice activity detection which is more accurate
        than energy-based detection as it accounts for things like:
        - Push-to-talk state
        - Voice activation detection
        - Mute/deafen states
        - Network conditions
        """
        try:
            # Access voice client through parent (VoiceTranslator instance)
            if (hasattr(self, 'parent') and self.parent and
                hasattr(self.parent, 'voice_client') and self.parent.voice_client and
                    hasattr(self.parent.voice_client, 'ws') and self.parent.voice_client.ws):

                voice_client = self.parent.voice_client
                ssrc_map = voice_client.ws.ssrc_map                # Find the SSRC for this user
                for info in ssrc_map.values():
                    if info.get("user_id") == user:
                        speaking_state = info.get("speaking", False)
                        logger.debug(
                            "Discord VAD for user %s: speaking=%s", user, speaking_state)
                        return speaking_state

                logger.debug("User %s not found in SSRC map", user)
                return False

        except Exception as e:
            logger.warning(
                "Error checking Discord VAD for user %s: %s", user, e)
            return False

        return False

    def _should_process_audio(self, data, user):
        """Combined VAD check using both Discord's VAD and energy-based fallback.

        This provides the most reliable voice activity detection by:
        1. First checking Discord's built-in VAD (most accurate)
        2. Falling back to energy-based VAD if Discord VAD unavailable
        3. Applying additional logic for silence detection
        """
        try:
            # Method 1: Try Discord's built-in VAD first (most reliable)
            discord_speaking = self._is_user_speaking_discord_vad(user)

            # Method 2: Energy-based VAD as fallback
            energy_active = self.is_audio_active(data, user)

            # Method 3: Check for silence frames
            # DEBUG: Log VAD decisions for troubleshooting (only when there's activity)
            is_silence_frame = self._is_silence_frame(data)
            if energy_active or discord_speaking:
                logger.debug("VAD check for user %s: discord=%s, energy=%s, silence=%s",
                             user, discord_speaking, energy_active, is_silence_frame)

            # Combine the signals for best accuracy
            if discord_speaking is not None:
                # If Discord says they're speaking, trust it
                if discord_speaking:
                    return True                # FIXED: Be more conservative when Discord says NOT speaking
                # Only override Discord's decision if energy is VERY high and it's clearly not silence
                elif energy_active and not is_silence_frame:
                    # Calculate actual energy to be more selective
                    try:
                        audio_array = np.frombuffer(data, dtype=np.int16)
                        if len(audio_array) > 0:
                            current_energy = np.mean(np.abs(audio_array))
                            # Only override if energy is significantly higher than normal speech threshold
                            if current_energy > 800:  # Much higher threshold than normal 250
                                logger.debug(
                                    "Overriding Discord VAD: very high energy %d detected", current_energy)
                                return True
                    except Exception:
                        pass
                # Trust Discord when it says user is not speaking
                return False
            else:
                # Fallback to energy-based VAD if Discord VAD unavailable
                return energy_active and not is_silence_frame
        except Exception as e:
            logger.warning(
                "Error in combined VAD check for user %s: %s", user, e)
            # Final fallback to simple energy check
            return self.is_audio_active(data, user)

    def _is_silence_frame(self, audio_data):
        """Check if audio frame is a silence frame sent by Discord."""
        try:
            # Discord sends specific silence frames: b"\xf8\xff\xfe"
            if len(audio_data) >= 3 and audio_data[:3] == b"\xf8\xff\xfe":
                return True

            # Also check for very low energy (near silence)
            if len(audio_data) >= 2:
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
                if len(audio_array) > 0:
                    energy = np.mean(np.abs(audio_array))
                    return energy < 50  # Very low energy threshold for silence

        except Exception as e:
            logger.debug("Error checking silence frame: %s", e)

        return False
