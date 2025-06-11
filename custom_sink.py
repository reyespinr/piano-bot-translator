"""
Simplified Discord real-time audio processing module.

This module provides a streamlined implementation of Discord's audio sink
for real-time speech detection, transcription, and translation. It coordinates
between specialized components for audio processing, worker management,
session tracking, and voice activity detection.

Features:
- Modular architecture with separated concerns
- Real-time speech detection and processing
- Background worker thread management
- Session-aware audio processing
- User-specific processing controls
"""
import asyncio
import os
import time
import threading
import subprocess
from typing import Optional, Callable, Awaitable
from discord.sinks import WaveSink

# Import our specialized components
from audio_worker_manager import AudioWorkerManager
from audio_session_manager import AudioSession
from voice_activity_detector import VoiceActivityDetector
from audio_buffer_manager import AudioBufferManager
import utils
from logging_config import get_logger

logger = get_logger(__name__)


class RealTimeWaveSink(WaveSink):
    """Simplified real-time audio processing sink for Discord voice data.

    This class extends Discord's WaveSink to provide real-time speech detection,
    transcription and translation capabilities. It coordinates between specialized
    components to process incoming audio packets, detect speech activity, and 
    manage the workflow of converting speech to text.

    Features:
    - Automatic speech detection using combined VAD methods
    - Modular component architecture for maintainability
    - Background processing of transcription tasks
    - Session-aware audio processing with adaptive thresholds
    - User-specific processing controls

    Args:
        *args: Additional positional arguments to pass to the parent class.
        pause_threshold (float, optional): Time in seconds to detect a long pause.
            Defaults to 1.0.
        event_loop (asyncio.AbstractEventLoop, optional): Event loop for async operations.
            Defaults to None.
        num_workers (int, optional): Number of transcription worker threads.
            Defaults to 6.
        **kwargs: Additional keyword arguments to pass to the parent class.
    """

    def __init__(self, *args, pause_threshold=1.0, event_loop=None, num_workers=6, **kwargs):
        super().__init__(*args, **kwargs)

        # Initialize components using composition
        self.worker_manager = AudioWorkerManager(num_workers, event_loop)
        self.session = AudioSession()
        self.vad = VoiceActivityDetector()
        self.buffer_manager = AudioBufferManager()

        # Simple configuration
        self.pause_threshold = pause_threshold
        self.silence_threshold = 10

        # Parent reference and callback
        self.parent = None
        self.translation_callback: Optional[Callable[...,
                                                     Awaitable[None]]] = None

        # Add tracking for last log time to prevent log spam
        self.last_block_log_time = {}

        # Set up worker callback and start components
        self.worker_manager.set_transcription_callback(self.transcribe_audio)
        self.worker_manager.start_workers()

        # Start the timer to check for inactive speakers
        self._start_processing_timer()

    def _start_processing_timer(self):
        """Start the timer to check for inactive speakers."""
        self.worker_manager.timer = threading.Timer(
            1.0, self._check_inactive_speakers)
        self.worker_manager.timer.daemon = True
        self.worker_manager.timer.start()

    def _should_process_user(self, user, data):
        """Check if user should be processed based on enabled state."""
        if (hasattr(self, 'parent') and
                self.parent and
                hasattr(self.parent, 'user_processing_enabled')):
            return self.parent.user_processing_enabled.get(str(user), True)
        return True

    def _get_voice_client(self):
        """Get voice client from parent if available."""
        if hasattr(self, 'parent') and self.parent and hasattr(self.parent, 'voice_client'):
            return self.parent.voice_client
        return None

    def write(self, data, user):
        """Main audio processing entry point - simplified."""
        try:
            # PROTECTION: Validate incoming audio data before processing
            if not data or len(data) < 4:
                return

            current_time = time.time()

            # Check if user is enabled for processing
            if not self._should_process_user(user, data):
                # Still write to main buffer but skip processing
                try:
                    super().write(data, user)
                except Exception as e:
                    logger.debug(
                        "Error writing to main buffer for disabled user %s: %s", user, str(e))
                return

            # Update session activity
            self.session.update_speaker(user)

            # Get user state
            user_state = self.buffer_manager.get_user_state(user)

            # First time seeing this user?
            if user_state.last_packet_time == 0:
                logger.debug("First audio packet from user %s", user)
                user_state.last_packet_time = current_time
                user_state.last_active_time = current_time

            # Check if audio is active using VAD
            try:
                is_active = self.vad.is_speech_active(
                    data, user, user_state, self._get_voice_client())
            except Exception as e:
                logger.debug(
                    "Audio activity detection failed for user %s: %s", user, str(e))
                is_active = False

            # Update last active time if speech is detected
            if is_active:
                user_state.last_active_time = current_time

            # Calculate time differences
            time_diff = current_time - user_state.last_packet_time
            active_diff = current_time - user_state.last_active_time

            # Process speech after significant pause
            if (user_state.is_speaking and
                    user_state.speech_detected and
                    active_diff > 2.0):  # 2.0 seconds
                self._process_silent_speech(user)

            # Handle long pauses
            if time_diff > 2.0:  # 2.0 seconds
                # Add rate limiting to prevent log spam
                now = time.time()
                last_log_time = self.last_block_log_time.get(user, 0)
                if now - last_log_time > 5.0:  # Only log once every 5 seconds per user
                    logger.debug(
                        "Long pause detected for user %s. Processing any speech and resetting.", user)
                    self.last_block_log_time[user] = now

                should_process = self.buffer_manager.process_long_pause(user)
                if should_process:
                    self.process_speech_buffer(user)

            # Update pre-speech buffer
            self.buffer_manager.update_pre_speech_buffer(user, data)

            # Handle active speech or silence
            if is_active:
                self.buffer_manager.handle_active_speech(user, data)
            else:
                should_process = self.buffer_manager.handle_silence(
                    user, data, self.silence_threshold)
                if should_process:
                    logger.debug(
                        "Speech ended for user %s. Processing audio.", user)
                    self.process_speech_buffer(user)
                    self.buffer_manager.reset_speech_state(user)

            user_state.last_packet_time = current_time

            # Write to the main buffer with error handling
            try:
                super().write(data, user)
            except Exception as e:
                logger.debug(
                    "Error writing to main buffer for user %s: %s", user, str(e))

        except Exception as e:
            logger.error("Error in write method for user %s: %s", user, str(e))

    def _process_silent_speech(self, user):
        """Process speech after detecting significant silence."""
        logger.debug(
            "Significant pause detected for user %s. Processing speech.", user)
        user_state = self.buffer_manager.get_user_state(user)
        user_state.is_speaking = False
        self.process_speech_buffer(user)
        user_state.speech_detected = False
        user_state.silence_frames = 0

    def process_speech_buffer(self, user):
        """Process the speech buffer for a user and queue for transcription."""
        user_state = self.buffer_manager.get_user_state(user)

        # Update session state and tracking variables
        current_time = self._update_session_state(user)
        min_duration = self.session.get_minimum_duration()

        # Check if buffer should be processed
        if not self.buffer_manager.should_process_buffer(user, min_duration):
            return

        # Check cooldown
        if self.buffer_manager.check_cooldown(user, current_time):
            return

        # Create and queue audio file
        self._create_and_queue_audio_file(user, current_time)

    def _create_and_queue_audio_file(self, user, current_time):
        """Create audio file and queue for transcription."""
        try:
            # Create audio file
            audio_filename, success = self.buffer_manager.create_audio_file(
                user, current_time)

            if not success:
                return

            # Manage queue overflow before adding new item
            self.worker_manager.manage_queue_overflow()

            # Queue for transcription
            if self.worker_manager.queue_audio_file(user, audio_filename):
                self.buffer_manager.update_processed_time(user, current_time)
                logger.debug("Queued audio file for user %s: %s",
                             user, audio_filename)

        except Exception as e:
            logger.error(
                "Error creating audio file for user %s: %s", user, str(e))

    def _update_session_state(self, user):
        """Update session state and return current time."""
        current_time = time.time()
        self.session.update_speaker(user)
        return current_time

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

            # Process the transcription and translation
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
                    # Even if no translation is needed (English),
                    # still send the original text to the translation container
                    logger.debug("⏭️ No translation needed for user %s (language: %s), sending original text",
                                 user, detected_language)
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

            # Reduce queue monitoring frequency to prevent log spam
            queue_size = self.worker_manager.get_queue_size()

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

            # Check for inactive speakers
            users_to_process = self.buffer_manager.check_inactive_speakers(
                current_time)

            for user in users_to_process:
                try:
                    logger.debug(
                        "Timer detected inactive speaker %s. Processing speech.", user)
                    self.process_speech_buffer(user)
                    user_state = self.buffer_manager.get_user_state(user)
                    user_state.is_speaking = False
                    user_state.speech_detected = False
                    user_state.silence_frames = 0
                except Exception as user_error:
                    logger.error(
                        "Error processing inactive user %s: %s", user, str(user_error))

            # Perform worker health check
            self.worker_manager.check_health()

        except (KeyError, AttributeError, TypeError, ValueError) as e:
            logger.error("Error checking inactive speakers: %s", e)
        finally:
            # Reschedule the timer if still running
            if self.worker_manager.running:
                try:
                    self.worker_manager.timer = threading.Timer(
                        1.0, self._check_inactive_speakers)
                    self.worker_manager.timer.daemon = True
                    self.worker_manager.timer.start()
                except Exception as timer_error:
                    logger.error("Failed to restart timer: %s",
                                 str(timer_error))

    def cleanup(self):
        """Clean up resources when the sink is no longer needed."""
        logger.info("Starting sink cleanup process...")

        # Delegate cleanup to worker manager
        self.worker_manager.cleanup()

        logger.info("Sink cleanup completed")
