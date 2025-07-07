"""
Core translation components with modular, testable architecture.

This module contains the refactored translator logic, broken down into
smaller, focused classes for better maintainability and testing.
"""
import asyncio
import struct
import sys
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple, Any

from audio_processing_utils import CLEANUP_DELAY
from audio_sink import RealTimeWaveSink
from logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class VoiceTranslatorState:
    """Encapsulates voice translator state."""
    translation_callback: Optional[Callable] = None
    websocket_handler: Optional[Any] = None
    voice_client: Optional[Any] = None
    sink: Optional[RealTimeWaveSink] = None
    is_listening: bool = False
    user_processing_enabled: Dict[str, bool] = None
    active_voices: Dict[int, Any] = None
    connected_users: Dict[str, Dict] = None
    current_channel: Optional[Any] = None
    model_loaded: bool = False

    def __post_init__(self):
        """Initialize mutable defaults."""
        if self.user_processing_enabled is None:
            self.user_processing_enabled = {}
        if self.active_voices is None:
            self.active_voices = {}
        if self.connected_users is None:
            self.connected_users = {}


class AudioSinkSetup:
    """Handles audio sink creation and configuration."""

    @staticmethod
    def create_sink(loop: asyncio.AbstractEventLoop) -> RealTimeWaveSink:
        """Create and configure audio sink."""
        logger.info("Creating voice processing sink...")
        return RealTimeWaveSink(
            pause_threshold=1.0,
            event_loop=loop
        )

    @staticmethod
    def configure_sink_callback(sink: RealTimeWaveSink, callback: Callable) -> None:
        """Configure sink translation callback."""
        sink.translation_callback = callback

    @staticmethod
    def setup_user_processing(sink: RealTimeWaveSink, user_processing_enabled: Dict[str, bool]) -> None:
        """Set up user processing states on sink."""
        if not user_processing_enabled:
            return

        sink.parent = type('obj', (object,), {})()
        processing_settings = {}
        for k, v in user_processing_enabled.items():
            processing_settings[str(k)] = bool(v)

        sink.parent.user_processing_enabled = processing_settings
        settings_items = [f"{k}={v}" for k, v in processing_settings.items()]
        settings_str = ', '.join(settings_items)
        logger.debug("Active user settings: %s", settings_str)

    @staticmethod
    def fallback_setup(sink: RealTimeWaveSink, voice_client: Any) -> None:
        """Fallback setup for sink parent reference."""
        if not (hasattr(voice_client, 'guild') and voice_client.guild):
            return

        try:
            if 'server' in sys.modules:
                server_module = sys.modules['server']
            else:
                import server as server_module

            if hasattr(server_module, 'user_processing_enabled'):
                sink.parent = server_module
        except ImportError:
            logger.debug("Could not import server module")


class DiscordAudioProtection:
    """Handles Discord audio corruption protection."""

    @staticmethod
    def create_audio_callback() -> Callable:
        """Create robust audio callback."""
        def audio_callback(_sink, *args):
            async def robust_process():
                try:
                    logger.debug("Audio callback executed successfully")
                except Exception as e:
                    logger.error("Error in audio callback: %s", str(e))
                return
            return robust_process()
        return audio_callback

    @staticmethod
    def patch_audio_unpack(voice_client: Any) -> None:
        """Apply audio corruption protection patch."""
        original_unpack_audio = voice_client.unpack_audio

        def safe_unpack_audio(data):
            try:
                return original_unpack_audio(data)
            except (IndexError, struct.error, ValueError, TypeError, AttributeError) as e:
                logger.warning(
                    "Discord audio packet corruption (ignoring): %s", str(e))
                return
            except Exception as e:
                logger.error(
                    "Unexpected Discord audio error (ignoring): %s", str(e))
                return

        voice_client.unpack_audio = safe_unpack_audio
        logger.debug("Applied Discord audio corruption protection")


class AudioRecordingManager:
    """Manages audio recording start/stop operations."""

    @staticmethod
    async def start_recording(voice_client: Any, sink: RealTimeWaveSink, callback: Callable) -> Tuple[bool, str]:
        """Start audio recording with error handling."""
        try:
            voice_client.start_recording(sink, callback)
            logger.debug("Recording started successfully")
            return True, "Recording started"
        except Exception as e:
            logger.error("Failed to start recording: %s", str(e))
            return False, f"Recording failed: {str(e)}"

    @staticmethod
    async def stop_recording(voice_client: Any) -> Tuple[bool, str]:
        """Stop audio recording with graceful error handling."""
        try:
            voice_client.stop_recording()
            logger.debug("Recording stopped successfully")
            return True, "Recording stopped"
        except Exception as recording_error:
            error_msg = str(recording_error)
            if "Not currently recording audio" in error_msg:
                logger.info(
                    "Voice client was not recording - this is expected after navigation/reconnection")
                return True, "Recording was already stopped"
            else:
                logger.warning(
                    "Unexpected error stopping recording: %s", error_msg)
                return False, f"Stop recording error: {error_msg}"


class SinkCleanupManager:
    """Handles sink cleanup operations."""

    @staticmethod
    async def cleanup_sink(sink: Optional[RealTimeWaveSink]) -> None:
        """Clean up sink resources with error handling."""
        if not sink:
            logger.debug("No sink to cleanup")
            return

        try:
            logger.debug("Cleaning up sink")
            sink.cleanup()
            logger.debug("Sink cleanup completed")
        except (AttributeError, RuntimeError) as e:
            logger.error("Error during sink cleanup: %s", str(e))

    @staticmethod
    async def wait_for_workers() -> None:
        """Wait for workers to finish with appropriate delay."""
        logger.debug("Waiting for workers to finish...")
        await asyncio.sleep(CLEANUP_DELAY)


class ModelManager:
    """Handles model loading and validation."""

    @staticmethod
    async def load_models() -> Tuple[bool, str]:
        """Load and validate models."""
        try:
            logger.info("Loading translation models...")
            from faster_whisper_manager import faster_whisper_model_manager

            # Check if models are already loaded
            if faster_whisper_model_manager.stats["models_loaded"]:
                logger.info(
                    "✅ Models already loaded via Faster-Whisper ModelManager")
                return True, "Models already loaded"

            # Initialize models
            success = await faster_whisper_model_manager.initialize_models(warm_up=False)
            if success:
                logger.info(
                    "✅ Models loaded successfully via Faster-Whisper ModelManager")
                return True, "Models loaded successfully"

            logger.error(
                "❌ Failed to load models via Faster-Whisper ModelManager")
            return False, "Failed to load models"

        except Exception as e:
            logger.error("Error during model loading: %s", str(e))

            # Check if models were loaded despite the error
            try:
                from faster_whisper_manager import faster_whisper_model_manager
                if faster_whisper_model_manager.stats["models_loaded"]:
                    logger.info("✅ Models loaded successfully despite error")
                    return True, "Models loaded despite error"
            except Exception:
                pass

            return False, f"Model loading failed: {str(e)}"


class UserStateManager:
    """Manages user states and updates."""

    @staticmethod
    def initialize_user_states(current_channel: Any) -> Tuple[Dict[str, bool], Dict[str, Dict]]:
        """Initialize processing states for users currently in the voice channel."""
        user_processing_enabled = {}
        connected_users = {}

        if not current_channel:
            return user_processing_enabled, connected_users

        # Set all current members to enabled by default
        for member in current_channel.members:
            if not member.bot:  # Exclude bots
                user_id_str = str(member.id)
                user_processing_enabled[user_id_str] = True
                connected_users[user_id_str] = {
                    'id': user_id_str,
                    'name': member.display_name
                }
                logger.debug("Initialized user %s (%s) as enabled",
                             member.display_name, member.id)

        return user_processing_enabled, connected_users

    @staticmethod
    async def send_user_updates(websocket_handler: Any, current_channel: Any,
                                user_processing_enabled: Dict[str, bool]) -> None:
        """Send current user list and states to WebSocket clients."""
        if not websocket_handler or not current_channel:
            logger.warning(
                "No websocket handler or current channel available for user updates")
            return

        users = []
        for member in current_channel.members:
            if not member.bot:  # Exclude bots
                user_id_str = str(member.id)
                users.append({
                    'id': user_id_str,
                    'name': member.display_name,
                    'avatar': str(member.avatar.url) if member.avatar else None
                })
                # Ensure user is in processing states
                if user_id_str not in user_processing_enabled:
                    user_processing_enabled[user_id_str] = True

        # Send users update
        try:
            await websocket_handler.broadcast_message({
                'type': 'users_update',
                'users': users,
                'enabled_states': user_processing_enabled.copy()
            })
            logger.info(
                "Sent user updates: %d users with processing states", len(users))
        except Exception as e:
            logger.error("Failed to send user updates: %s", str(e))


class WebSocketBroadcaster:
    """Handles WebSocket message broadcasting."""

    @staticmethod
    async def broadcast_listen_status(websocket_handler: Any, is_listening: bool) -> None:
        """Broadcast listening status to all clients."""
        if websocket_handler:
            await websocket_handler.broadcast_listen_status(is_listening)

    @staticmethod
    async def broadcast_translation(websocket_handler: Any, user_id: str, text: str, message_type: str) -> None:
        """Broadcast translation/transcription to all clients."""
        if websocket_handler:
            await websocket_handler.broadcast_translation(user_id, text, message_type)
            logger.debug("✅ %s message sent to WebSocket clients",
                         message_type.capitalize())
        else:
            logger.warning(
                "❌ No websocket_handler available for %s", message_type)

    @staticmethod
    async def broadcast_user_toggle(websocket_handler: Any, user_id: str, enabled: bool) -> None:
        """Broadcast user toggle status."""
        if websocket_handler:
            await websocket_handler.broadcast_message({
                'type': 'user_toggle',
                'user_id': str(user_id),
                'enabled': enabled
            })

    @staticmethod
    async def broadcast_user_joined(websocket_handler: Any, user_data: Dict, enabled: bool) -> None:
        """Broadcast user joined event."""
        if websocket_handler:
            await websocket_handler.broadcast_user_joined(user_data, enabled)

    @staticmethod
    async def broadcast_user_left(websocket_handler: Any, user_id: str) -> None:
        """Broadcast user left event."""
        if websocket_handler:
            await websocket_handler.broadcast_user_left(user_id)


class VoiceStateUpdateHandler:
    """Handles voice state update events."""

    @staticmethod
    async def handle_user_joined(member: Any, current_channel: Any, user_processing_enabled: Dict[str, bool],
                                 websocket_handler: Any) -> None:
        """Handle user joining voice channel."""
        user_id_str = str(member.id)

        # Enable processing by default for new users
        user_processing_enabled[user_id_str] = True
        logger.info("👋 User %s joined voice channel", member.display_name)

        # Send user joined update
        user_data = {
            "id": user_id_str,
            "name": member.display_name,
            "avatar": str(member.avatar.url) if member.avatar else None
        }
        await WebSocketBroadcaster.broadcast_user_joined(websocket_handler, user_data, True)

    @staticmethod
    async def handle_user_left(member: Any, user_processing_enabled: Dict[str, bool],
                               websocket_handler: Any) -> None:
        """Handle user leaving voice channel."""
        user_id_str = str(member.id)

        # Remove from processing states
        user_processing_enabled.pop(user_id_str, None)
        logger.info("👋 User %s left voice channel", member.display_name)

        # Send user left update
        await WebSocketBroadcaster.broadcast_user_left(websocket_handler, user_id_str)
