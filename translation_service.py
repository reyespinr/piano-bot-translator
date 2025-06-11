"""
Voice translation module for Discord bot.

This module provides real-time voice transcription and translation capabilities
for Discord voice channels. It handles audio processing, language detection,
transcription, and translation using machine learning models.

REFACTORED: Large methods have been broken down into modular components
in translation_engine.py for better maintainability and testing.
"""
import asyncio
from typing import Callable

from translation_engine import (
    VoiceTranslatorState,
    AudioSinkSetup,
    DiscordAudioProtection,
    AudioRecordingManager,
    SinkCleanupManager,
    ModelManager,
    FileCleanupManager,
    UserStateManager,
    WebSocketBroadcaster,
    VoiceStateUpdateHandler
)
from logging_config import get_logger

logger = get_logger(__name__)


class VoiceTranslator:
    """Voice translation manager for Discord bot."""

    def __init__(self, translation_callback: Callable = None):
        self.state = VoiceTranslatorState(translation_callback=translation_callback)
        logger.info("✅ VoiceTranslator initialized")

    def set_websocket_handler(self, handler):
        """Set the WebSocket handler for sending user updates."""
        self.state.websocket_handler = handler

    async def join_voice_channel(self, channel):
        """Join a voice channel and set up audio processing."""
        try:
            logger.info("Joining voice channel: %s", channel.name)

            # Join the voice channel
            self.state.voice_client = await channel.connect()
            self.state.current_channel = channel

            # Initialize user processing states for current members
            self.state.user_processing_enabled, self.state.connected_users = UserStateManager.initialize_user_states(channel)

            # Send user list to WebSocket clients
            await UserStateManager.send_user_updates(
                self.state.websocket_handler, 
                self.state.current_channel, 
                self.state.user_processing_enabled
            )

            logger.info("✅ Successfully joined voice channel: %s", channel.name)
            return True

        except Exception as e:
            logger.error("Failed to join voice channel %s: %s", channel.name, str(e))
            return False

    def _handle_audio_finished(self, sink, *args):
        """Handle audio recording finished event."""
        logger.debug("Audio recording finished")

    async def process_audio_callback(self, user_id, text, message_type):
        """Process audio transcription/translation callback."""
        try:
            logger.debug("🔄 process_audio_callback called for user %s with message_type: %s", user_id, message_type)

            if message_type == "transcription":
                logger.info("📝 Received transcription for user %s: %s", user_id, text)
                await WebSocketBroadcaster.broadcast_translation(self.state.websocket_handler, user_id, text, "transcription")

            elif message_type == "translation":
                logger.info("🌍 Received translation for user %s: %s", user_id, text)
                await WebSocketBroadcaster.broadcast_translation(self.state.websocket_handler, user_id, text, "translation")

        except Exception as e:
            logger.error("Error in process_audio_callback: %s", str(e))

    async def _get_user_display_name(self, user_id):
        """Get user display name from Discord bot - simplified version."""
        return UserStateManager.get_user_display_name(self.state.current_channel, user_id)

    async def toggle_user_processing(self, user_id, enabled):
        """Toggle processing for a specific user."""
        try:
            user_id_str = str(user_id)
            self.state.user_processing_enabled[user_id_str] = enabled

            # Get user info for logging
            user_name = UserStateManager.get_user_display_name(self.state.current_channel, user_id)
            logger.info("User processing %s for %s (%s)", "enabled" if enabled else "disabled", user_name, user_id)

            # Send update to WebSocket clients
            await WebSocketBroadcaster.broadcast_user_toggle(self.state.websocket_handler, user_id_str, enabled)
            return True

        except Exception as e:
            logger.error("Error toggling user processing: %s", str(e))
            return False

    async def handle_voice_state_update(self, member, before, after):
        """Handle voice state updates (users joining/leaving)."""
        try:
            if not self.state.current_channel:
                return

            # Check if the update is for our current channel
            if self.state.current_channel in (before.channel, after.channel):
                if member.bot:  # Ignore bots
                    return

                # User joined our channel
                if after.channel == self.state.current_channel and before.channel != self.state.current_channel:
                    await VoiceStateUpdateHandler.handle_user_joined(
                        member, self.state.current_channel, 
                        self.state.user_processing_enabled, self.state.websocket_handler
                    )

                # User left our channel
                elif before.channel == self.state.current_channel and after.channel != self.state.current_channel:
                    await VoiceStateUpdateHandler.handle_user_left(
                        member, self.state.user_processing_enabled, self.state.websocket_handler
                    )

        except Exception as e:
            logger.error("Error handling voice state update: %s", str(e))

    async def load_models(self):
        """Verify model loading capabilities and warm up models with timeout protection"""
        success, message = await ModelManager.load_models()
        self.state.model_loaded = success
        logger.info("Model loading result: %s - %s", "SUCCESS" if success else "FAILED", message)
        return success

    def setup_voice_receiver(self, voice_client):
        """Set up the voice receiver for a Discord voice client"""
        if not self.state.model_loaded:
            logger.warning("Models not loaded yet!")
        
        # Store the voice client for later use
        self.state.active_voices[voice_client.guild.id] = voice_client

    async def start_listening(self, voice_client):
        """Start listening and processing audio"""
        if not voice_client or not voice_client.is_connected():
            return False, "Not connected to a voice channel"

        # If already listening, stop first to avoid conflicts
        if self.state.is_listening:
            logger.info("Already listening - stopping first to avoid state conflicts")
            await self.stop_listening(voice_client)

        try:
            # Create sink for real-time audio processing
            self.state.sink = AudioSinkSetup.create_sink(asyncio.get_event_loop())
            
            # Configure sink
            AudioSinkSetup.configure_sink_callback(self.state.sink, self.process_audio_callback)
            AudioSinkSetup.setup_user_processing(self.state.sink, self.state.user_processing_enabled)
            AudioSinkSetup.fallback_setup(self.state.sink, voice_client)

            # Create robust audio callback
            audio_callback = DiscordAudioProtection.create_audio_callback()

            # Apply Discord audio protection
            DiscordAudioProtection.patch_audio_unpack(voice_client)

            # Start recording
            success, message = await AudioRecordingManager.start_recording(voice_client, self.state.sink, audio_callback)
            if not success:
                return False, message

            self.state.is_listening = True
            logger.debug("Started listening in channel: %s", voice_client.channel.name)

            # Broadcast listening status to all clients
            await WebSocketBroadcaster.broadcast_listen_status(self.state.websocket_handler, True)

            return True, "Started listening"
            
        except Exception as e:
            logger.error("Error in start_listening: %s", str(e))
            return False, f"Error: {str(e)}"

    async def stop_listening(self, voice_client):
        """Stop listening and processing audio"""
        logger.debug("stop_listening called. Voice client valid: %s", voice_client is not None)
        if not voice_client:
            logger.debug("Voice client is None, returning")
            return False, "Not connected to a voice channel"

        try:
            # Stop recording with graceful error handling
            success, message = await AudioRecordingManager.stop_recording(voice_client)
            if not success and "Recording was already stopped" not in message:
                logger.warning("Recording stop issue: %s", message)

            # Wait for workers to finish
            await SinkCleanupManager.wait_for_workers()

            # Clean up sink resources
            await SinkCleanupManager.cleanup_sink(self.state.sink)
            self.state.sink = None

            self.state.is_listening = False
            logger.debug("Stopped listening")

            # Broadcast listening status to all clients
            await WebSocketBroadcaster.broadcast_listen_status(self.state.websocket_handler, False)

            return True, "Stopped listening"
            
        except Exception as e:
            logger.error("Error in stop_listening: %s", str(e))
            return False, f"Error: {str(e)}"

    async def toggle_listening(self, voice_client):
        """Toggle listening on/off"""
        if self.state.is_listening:
            return await self.stop_listening(voice_client)
        return await self.start_listening(voice_client)

    async def _force_delete_file(self, file_path):
        """Forcefully delete a file with multiple retries."""
        return await FileCleanupManager.force_delete_file(file_path)

    def _cleanup_audio_file(self, audio_file):
        """Legacy method for compatibility - use _force_delete_file instead."""
        if audio_file:
            asyncio.create_task(self._force_delete_file(audio_file))

    def cleanup(self):
        """Clean up resources"""
        try:
            # Stop all recordings
            for guild_id, voice_client in list(self.state.active_voices.items()):
                try:
                    if voice_client.is_connected():
                        asyncio.create_task(self.stop_listening(voice_client))
                except (AttributeError, ConnectionError) as e:
                    logger.error("Error during cleanup for guild %s: %s", guild_id, str(e))

            # Clean up sink
            if self.state.sink:
                try:
                    self.state.sink.cleanup()
                except Exception as e:
                    logger.error("Error during sink cleanup: %s", str(e))
                self.state.sink = None

            # Clear state
            self.state.active_voices.clear()
            self.state.connected_users.clear()
            self.state.user_processing_enabled.clear()
            self.state.is_listening = False

            logger.info("✅ VoiceTranslator cleanup completed")
            
        except Exception as e:
            logger.error("❌ Error during VoiceTranslator cleanup: %s", str(e))
