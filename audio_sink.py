"""
Simplified Discord real-time audio processing module.

This module provides a streamlined implementation of Discord's audio sink
for real-time speech detection, transcription, and translation. It coordinates
between specialized components for audio processing, worker management,
session tracking, and voice activity detection.

REFACTORED: Large methods have been broken down into modular components
in audio_sink_core.py for better maintainability and testing.

Features:
- Modular architecture with separated concerns
- Real-time speech detection and processing
- Background worker thread management
- Session-aware audio processing
- User-specific processing controls
"""
import time
from discord.sinks import WaveSink

from audio_sink_core import (
    AudioSinkState,
    AudioSinkInitializer,
    UserProcessingChecker,
    AudioDataValidator,
    SpeechActivityProcessor,
    SilencePauseHandler,
    AudioBufferProcessor,
    TranscriptionProcessor,
    QueueMonitor,
    GUIUpdateManager,
    SinkCleanupManager
)
from logging_config import get_logger

logger = get_logger(__name__)


class RealTimeWaveSink(WaveSink):
    """Simplified real-time audio processing sink for Discord voice data.

    This class extends Discord's WaveSink to provide real-time speech detection,
    transcription and translation capabilities. It coordinates between specialized
    components to process incoming audio packets, detect speech activity, and 
    manage the workflow of converting speech to text.

    REFACTORED: Complex methods have been broken down into modular components
    for better maintainability and testing.

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

        # Initialize state
        self.state = AudioSinkState(
            pause_threshold=pause_threshold,
            silence_threshold=10
        )

        # Initialize components using the new modular approach
        self.worker_manager, self.session, self.vad, self.buffer_manager = AudioSinkInitializer.initialize_components(
            num_workers, event_loop
        )

        # Set up worker callback and start components
        AudioSinkInitializer.setup_worker_callback(
            self.worker_manager, self.transcribe_audio)

        # Start the timer to check for inactive speakers
        AudioSinkInitializer.start_processing_timer(
            self.worker_manager, self._check_inactive_speakers)

        # Initialize queue monitoring
        self._last_queue_log_time = {}

    def _should_process_user(self, user, data):
        """Check if user should be processed based on enabled state."""
        return UserProcessingChecker.should_process_user(self.state, user, data)

    def _get_voice_client(self):
        """Get voice client from parent if available."""
        return UserProcessingChecker.get_voice_client(self.state)

    @property
    def parent(self):
        """Get parent reference."""
        return self.state.parent

    @parent.setter
    def parent(self, value):
        """Set parent reference."""
        self.state.parent = value

    @property
    def translation_callback(self):
        """Get translation callback."""
        return self.state.translation_callback

    @translation_callback.setter
    def translation_callback(self, value):
        """Set translation callback."""
        self.state.translation_callback = value

    def write(self, data, user):
        """Main audio processing entry point - simplified using core components."""
        try:
            # Validate incoming audio data
            if not AudioDataValidator.validate_audio_data(data):
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

            # Handle first packet from user
            if AudioDataValidator.is_first_packet(user_state):
                logger.debug("First audio packet from user %s", user)
                user_state.last_packet_time = current_time
                user_state.last_active_time = current_time

            # Detect speech activity using core component
            is_active = SpeechActivityProcessor.detect_speech_activity(
                self.vad, data, user, user_state, self._get_voice_client())

            # Update activity times using core component
            SpeechActivityProcessor.update_activity_times(
                user_state, current_time, is_active)

            # Calculate time differences using core component
            time_diff, active_diff = SpeechActivityProcessor.calculate_time_differences(
                user_state, current_time)

            # Process silent speech using core component
            if SilencePauseHandler.should_process_silent_speech(user_state, active_diff):
                SilencePauseHandler.process_silent_speech(
                    user, self.buffer_manager)
                self.process_speech_buffer(user)

            # Handle long pauses using core component
            if SilencePauseHandler.should_process_long_pause(time_diff, self.state, user):
                should_process = self.buffer_manager.process_long_pause(user)
                if should_process:
                    self.process_speech_buffer(user)

            # Update pre-speech buffer
            self.buffer_manager.update_pre_speech_buffer(user, data)

            # Handle active speech or silence using core components
            if is_active:
                AudioBufferProcessor.handle_active_speech(
                    user, data, self.buffer_manager)
            else:
                AudioBufferProcessor.handle_silence(
                    user, data, self.state.silence_threshold,
                    self.buffer_manager, self.process_speech_buffer)

            # Write to the main buffer with error handling
            user_state.last_packet_time = current_time
            try:
                super().write(data, user)
            except Exception as e:
                logger.debug(
                    "Error writing to main buffer for user %s: %s", user, str(e))

        except Exception as e:
            logger.error("Error in write method for user %s: %s", user, str(e))

    def process_speech_buffer(self, user):
        """Process the speech buffer for a user and queue for transcription_service."""
        # Check if user processing is enabled before processing buffer
        if not self._should_process_user(user, None):
            logger.debug(
                "Skipping speech buffer processing for disabled user: %s", user)
            return

        AudioBufferProcessor.process_speech_buffer(
            user, self.buffer_manager, self.session, self.worker_manager)

    async def transcribe_audio(self, audio_file, user):
        """Transcribe the audio file with smart model routing and update the GUI with results."""
        await TranscriptionProcessor.transcribe_audio(audio_file, user, self._update_gui)

    async def _update_gui(self, user, text, translated_text, message_type):
        """Update the GUI with transcription and translation results."""
        await GUIUpdateManager.update_gui(
            self.translation_callback, user, text, translated_text, message_type)

    def _check_inactive_speakers(self):
        """Periodically check for users who have stopped speaking but haven't been processed."""
        QueueMonitor.check_inactive_speakers(
            self.worker_manager, self.buffer_manager, self._last_queue_log_time, self._check_inactive_speakers, self.state)

    def cleanup(self):
        """Clean up resources when the sink is no longer needed."""
        SinkCleanupManager.cleanup_sink(self.worker_manager)
