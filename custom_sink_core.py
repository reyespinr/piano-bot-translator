"""
Core audio sink components with modular, testable architecture.

This module contains the refactored audio sink logic, broken down into
smaller, focused classes for better maintainability and testing.
"""
import asyncio
import gc
import os
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Callable, Tuple

import utils
from logging_config import get_logger

logger = get_logger(__name__)

# Constants for audio processing
MAX_DELETE_RETRIES = 3


@dataclass
class AudioSinkState:
    """Encapsulates audio sink state and configuration."""
    pause_threshold: float = 1.0
    silence_threshold: int = 10
    parent: Optional[Any] = None
    translation_callback: Optional[Callable] = None
    last_block_log_time: Dict[str, float] = None

    def __post_init__(self):
        """Initialize mutable defaults."""
        if self.last_block_log_time is None:
            self.last_block_log_time = {}


class AudioSinkInitializer:
    """Handles audio sink initialization and component setup."""

    @staticmethod
    def initialize_components(num_workers: int, event_loop: Optional[asyncio.AbstractEventLoop]) -> Tuple[Any, Any, Any, Any]:
        """Initialize all audio processing components."""
        from audio_worker_manager import AudioWorkerManager
        from audio_session_manager import AudioSession
        from voice_activity_detector import VoiceActivityDetector
        from audio_buffer_manager import AudioBufferManager

        worker_manager = AudioWorkerManager(num_workers, event_loop)
        session = AudioSession()
        vad = VoiceActivityDetector()
        buffer_manager = AudioBufferManager()

        return worker_manager, session, vad, buffer_manager

    @staticmethod
    def setup_worker_callback(worker_manager: Any, transcription_callback: Callable) -> None:
        """Set up worker manager callback and start workers."""
        worker_manager.set_transcription_callback(transcription_callback)
        worker_manager.start_workers()

    @staticmethod
    def start_processing_timer(worker_manager: Any, check_inactive_callback: Callable) -> None:
        """Start the timer to check for inactive speakers."""
        worker_manager.timer = threading.Timer(1.0, check_inactive_callback)
        worker_manager.timer.daemon = True
        worker_manager.timer.start()


class UserProcessingChecker:
    """Handles user processing state validation."""

    @staticmethod
    def should_process_user(sink_state: AudioSinkState, user: str, data: Any) -> bool:
        """Check if user should be processed based on enabled state."""
        if (sink_state.parent and
                hasattr(sink_state.parent, 'user_processing_enabled')):
            return sink_state.parent.user_processing_enabled.get(str(user), True)
        return True

    @staticmethod
    def get_voice_client(sink_state: AudioSinkState) -> Optional[Any]:
        """Get voice client from parent if available."""
        if (sink_state.parent and
                hasattr(sink_state.parent, 'voice_client')):
            return sink_state.parent.voice_client
        return None


class AudioDataValidator:
    """Handles audio data validation and preprocessing."""

    @staticmethod
    def validate_audio_data(data: Any) -> bool:
        """Validate incoming audio data before processing."""
        return data is not None and len(data) >= 4

    @staticmethod
    def is_first_packet(user_state: Any) -> bool:
        """Check if this is the first audio packet from user."""
        return user_state.last_packet_time == 0


class SpeechActivityProcessor:
    """Handles speech activity detection and processing logic."""

    @staticmethod
    def detect_speech_activity(vad: Any, data: Any, user: str, user_state: Any, voice_client: Any) -> bool:
        """Detect speech activity using VAD with error handling."""
        try:
            return vad.is_speech_active(data, user, user_state, voice_client)
        except Exception as e:
            logger.debug(
                "Audio activity detection failed for user %s: %s", user, str(e))
            return False

    @staticmethod
    def update_activity_times(user_state: Any, current_time: float, is_active: bool) -> None:
        """Update user activity timestamps."""
        if is_active:
            user_state.last_active_time = current_time

    @staticmethod
    def calculate_time_differences(user_state: Any, current_time: float) -> Tuple[float, float]:
        """Calculate time differences for pause detection."""
        time_diff = current_time - user_state.last_packet_time
        active_diff = current_time - user_state.last_active_time
        return time_diff, active_diff


class SilencePauseHandler:
    """Handles silence detection and pause processing."""

    @staticmethod
    def should_process_silent_speech(user_state: Any, active_diff: float) -> bool:
        """Check if silent speech should be processed."""
        return (user_state.is_speaking and
                user_state.speech_detected and
                active_diff > 2.0)

    @staticmethod
    def process_silent_speech(user: str, buffer_manager: Any) -> None:
        """Process speech after detecting significant silence."""
        logger.debug(
            "Significant pause detected for user %s. Processing speech.", user)
        user_state = buffer_manager.get_user_state(user)
        user_state.is_speaking = False
        buffer_manager.reset_speech_state(user)

    @staticmethod
    def should_process_long_pause(time_diff: float, sink_state: AudioSinkState, user: str) -> bool:
        """Check if long pause should be processed with rate limiting."""
        if time_diff <= 2.0:
            return False

        # Add rate limiting to prevent log spam
        now = time.time()
        last_log_time = sink_state.last_block_log_time.get(user, 0)
        if now - last_log_time > 5.0:  # Only log once every 5 seconds per user
            logger.debug(
                "Long pause detected for user %s. Processing any speech and resetting.", user)
            sink_state.last_block_log_time[user] = now

        return True


class AudioBufferProcessor:
    """Handles audio buffer operations and speech processing."""

    @staticmethod
    def process_speech_buffer(user: str, buffer_manager: Any, session: Any) -> None:
        """Process the speech buffer for a user and queue for transcription."""
        user_state = buffer_manager.get_user_state(user)

        # Update session state and tracking variables
        current_time = time.time()
        session.update_speaker(user)
        min_duration = session.get_minimum_duration()

        # Check if buffer should be processed
        if not buffer_manager.should_process_buffer(user, min_duration):
            return

        # Check cooldown
        if buffer_manager.check_cooldown(user, current_time):
            return

        # Create and queue audio file
        AudioBufferProcessor._create_and_queue_audio_file(
            user, current_time, buffer_manager)

    @staticmethod
    def _create_and_queue_audio_file(user: str, current_time: float, buffer_manager: Any) -> None:
        """Create audio file and queue for transcription."""
        try:
            # Create audio file
            audio_filename, success = buffer_manager.create_audio_file(
                user, current_time)
            if not success:
                return

            # Get worker manager and manage queue overflow
            worker_manager = getattr(buffer_manager, 'worker_manager', None)
            if worker_manager:
                worker_manager.manage_queue_overflow()

                # Queue for transcription
                if worker_manager.queue_audio_file(user, audio_filename):
                    buffer_manager.update_processed_time(user, current_time)
                    logger.debug("Queued audio file for user %s: %s",
                                 user, audio_filename)

        except Exception as e:
            logger.error(
                "Error creating audio file for user %s: %s", user, str(e))

    @staticmethod
    def handle_active_speech(user: str, data: Any, buffer_manager: Any) -> None:
        """Handle active speech processing."""
        buffer_manager.handle_active_speech(user, data)

    @staticmethod
    def handle_silence(user: str, data: Any, silence_threshold: int, buffer_manager: Any,
                       process_callback: Callable) -> None:
        """Handle silence processing with speech completion check."""
        should_process = buffer_manager.handle_silence(
            user, data, silence_threshold)
        if should_process:
            logger.debug("Speech ended for user %s. Processing audio.", user)
            process_callback(user)
            buffer_manager.reset_speech_state(user)


class TranscriptionProcessor:
    """Handles transcription and translation processing."""

    @staticmethod
    async def transcribe_audio(audio_file: str, user: str, gui_callback: Callable) -> None:
        """Transcribe the audio file with smart model routing and update the GUI with results."""
        logger.debug(
            "🔄 Starting transcription for user %s, file: %s", user, audio_file)

        try:
            # Skip if file doesn't exist
            if not os.path.exists(audio_file):
                logger.warning("Audio file not found: %s", audio_file)
                return

            # Perform transcription
            transcribed_text, detected_language = await TranscriptionProcessor._perform_transcription(audio_file, user, gui_callback)

            if not transcribed_text or not transcribed_text.strip():
                logger.debug("Empty transcription result for user %s", user)
                return

            logger.info("✅ Transcription for user %s: %s",
                        user, transcribed_text)

            # Process transcription and translation
            await TranscriptionProcessor._process_transcription_and_translation(
                transcribed_text, detected_language, user, gui_callback)

        except Exception as e:
            logger.error(
                "Error in transcribe_audio for user %s: %s", user, str(e))
        finally:
            # Clean up the audio file
            if audio_file and os.path.exists(audio_file):
                await FileCleanupManager.force_delete_file(audio_file)
            logger.debug(
                "🔚 transcribe_audio finally block completed for user %s", user)

    @staticmethod
    async def _perform_transcription(audio_file: str, user: str, gui_callback: Callable) -> Tuple[str, str]:
        """Perform the actual transcription with error handling."""
        try:
            logger.debug("📞 Calling utils.transcribe for file: %s", audio_file)
            transcribed_text, detected_language = await utils.transcribe(audio_file)
            logger.debug("🎯 Transcription result for user %s: text='%s', language=%s",
                         user, transcribed_text, detected_language)
            return transcribed_text, detected_language

        except Exception as e:
            logger.error("Transcription failed for user %s: %s", user, str(e))
            # Graceful degradation: send error notice to UI
            error_msg = f"[Transcription Error: {str(e)[:50]}...]"
            try:
                await gui_callback(user, error_msg, None, "transcription")
            except Exception:
                pass
            return "", "error"

    @staticmethod
    async def _process_transcription_and_translation(transcribed_text: str, detected_language: str,
                                                     user: str, gui_callback: Callable) -> None:
        """Process transcription and translation results."""
        try:
            # Send transcription to GUI
            logger.debug("📡 Sending transcription for user %s", user)
            await gui_callback(user, transcribed_text, None, "transcription")

            # Check if translation is needed
            logger.debug("🌐 Starting translation check for user %s", user)
            await TranscriptionProcessor._handle_translation(transcribed_text, detected_language, user, gui_callback)

        except Exception as e:
            logger.error(
                "Error processing transcription/translation for user %s: %s", user, str(e))

    @staticmethod
    async def _handle_translation(transcribed_text: str, detected_language: str,
                                  user: str, gui_callback: Callable) -> None:
        """Handle translation logic."""
        should_translate_result = await utils.should_translate(transcribed_text, detected_language)
        logger.debug("🔍 Translation needed for user %s: %s",
                     user, should_translate_result)

        if should_translate_result:
            # Translate the text
            logger.debug("🌍 Translating text for user %s", user)
            translated_text = await utils.translate(transcribed_text)
            logger.debug("🌍 Translation result for user %s: %s",
                         user, translated_text)

            if translated_text and translated_text.strip():
                logger.debug("📡 Sending translation for user %s", user)
                await gui_callback(user, translated_text, None, "translation")
            else:
                logger.warning("Translation failed for user %s", user)
        else:
            # Even if no translation is needed (English), send original text to translation container
            logger.debug("⏭️ No translation needed for user %s (language: %s), sending original text",
                         user, detected_language)
            await gui_callback(user, transcribed_text, None, "translation")


class QueueMonitor:
    """Handles queue monitoring and inactive speaker detection."""

    @staticmethod
    def check_inactive_speakers(worker_manager: Any, buffer_manager: Any, last_queue_log_time: Dict[str, float]) -> None:
        """Periodically check for users who have stopped speaking but haven't been processed."""
        try:
            current_time = time.time()

            # Monitor queue with reduced frequency to prevent log spam
            QueueMonitor._monitor_queue_status(
                worker_manager, current_time, last_queue_log_time)

            # Check for inactive speakers
            users_to_process = buffer_manager.check_inactive_speakers(
                current_time)

            for user in users_to_process:
                try:
                    QueueMonitor._process_inactive_user(
                        user, worker_manager, buffer_manager, current_time)
                except Exception as user_error:
                    logger.error(
                        "Error processing inactive user %s: %s", user, str(user_error))

            # Restart timer
            QueueMonitor._restart_timer(worker_manager)

        except Exception as e:
            logger.error("Error in _check_inactive_speakers: %s", str(e))

    @staticmethod
    def _monitor_queue_status(worker_manager: Any, current_time: float, last_queue_log_time: Dict[str, float]) -> None:
        """Monitor queue status with rate limiting."""
        queue_size = worker_manager.get_queue_size()

        # Only log queue status every 10 seconds to reduce spam
        if 'queue' not in last_queue_log_time:
            last_queue_log_time['queue'] = 0

        # Every 10 seconds
        if current_time - last_queue_log_time['queue'] > 10.0:
            if queue_size > 30:
                logger.warning("⚠️ High queue load: %d items", queue_size)
            elif queue_size > 0:
                logger.debug("Queue activity: %d items processing", queue_size)
            last_queue_log_time['queue'] = current_time

    @staticmethod
    def _process_inactive_user(user: str, worker_manager: Any, buffer_manager: Any, current_time: float) -> None:
        """Process an inactive user."""
        logger.debug("Processing inactive user: %s", user)

        # Create audio file
        audio_filename, success = buffer_manager.create_audio_file(
            user, current_time)
        if not success:
            return

        # Manage queue overflow
        worker_manager.manage_queue_overflow()

        # Queue for transcription
        if worker_manager.queue_audio_file(user, audio_filename):
            buffer_manager.update_processed_time(user, current_time)
            buffer_manager.reset_speech_state(user)
            logger.debug("Queued inactive user audio: %s", audio_filename)

    @staticmethod
    def _restart_timer(worker_manager: Any) -> None:
        """Restart the monitoring timer."""
        try:
            if hasattr(worker_manager, 'timer'):
                worker_manager.timer = threading.Timer(
                    1.0, lambda: None)  # Placeholder
                worker_manager.timer.daemon = True
                worker_manager.timer.start()
        except Exception as timer_error:
            logger.error("Failed to restart timer: %s", str(timer_error))


class FileCleanupManager:
    """Handles file cleanup operations."""

    @staticmethod
    async def force_delete_file(file_path: str) -> bool:
        """Forcefully delete a file with multiple retries and GC."""
        if not file_path or not os.path.exists(file_path):
            return False

        # Try to delete the file with multiple retries
        for attempt in range(MAX_DELETE_RETRIES):
            try:
                # Force garbage collection to release file handles
                gc.collect()

                # Delete and verify
                os.remove(file_path)
                logger.debug("Deleted file: %s", os.path.basename(file_path))

                if not os.path.exists(file_path):
                    return True

                logger.warning(
                    "File still exists after deletion attempt: %s", file_path)
            except (PermissionError, OSError) as e:
                logger.warning("Deletion attempt %d failed: %s",
                               attempt + 1, str(e))
                await asyncio.sleep(0.5)  # Wait before retry

        # Last resort: try with Windows-specific commands
        if os.name == 'nt':
            try:
                subprocess.run(f'del /F "{file_path}"',
                               shell=True, check=False)
                logger.debug(
                    "Attempted deletion with Windows command: %s", file_path)
                return not os.path.exists(file_path)
            except (OSError, subprocess.SubprocessError) as e:
                logger.error("Windows command deletion failed: %s", str(e))

        logger.warning(
            "Failed to delete file after multiple attempts: %s", file_path)
        return False


class GUIUpdateManager:
    """Handles GUI update operations."""

    @staticmethod
    async def update_gui(translation_callback: Optional[Callable], user: str, text: str,
                         translated_text: Optional[str], message_type: str) -> None:
        """Update the GUI with transcription and translation results."""
        logger.debug(
            "_update_gui called for user %s with message_type: %s", user, message_type)

        try:
            if translation_callback:
                logger.debug(
                    "Calling translation_callback for user %s with type %s", user, message_type)
                await translation_callback(user, text, message_type)
            else:
                logger.warning(
                    "No translation callback available for user %s", user)

        except Exception as e:
            logger.error("Error in _update_gui for user %s: %s", user, str(e))


class SinkCleanupManager:
    """Handles sink cleanup operations."""

    @staticmethod
    def cleanup_sink(worker_manager: Any) -> None:
        """Clean up resources when the sink is no longer needed."""
        logger.info("Starting sink cleanup process...")

        try:
            # Delegate cleanup to worker manager
            worker_manager.cleanup()
            logger.info("Sink cleanup completed")
        except Exception as e:
            logger.error("Error during sink cleanup: %s", str(e))
