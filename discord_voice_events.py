"""
Discord voice event handlers.

Handles voice state updates, user join/leave events, and WebSocket notifications.
"""
import asyncio
from logging_config import get_logger

logger = get_logger(__name__)


class DiscordVoiceEventHandler:
    """Handles Discord voice state events and user management."""

    def __init__(self, bot_manager):
        """Initialize with reference to bot manager."""
        self.bot_manager = bot_manager

    async def handle_voice_state_update(self, member, before, after):
        """Handle voice state updates to track users joining/leaving."""
        if member.bot:
            return

        # Check WebSocket availability for notifications
        websocket_available = self._is_websocket_available()
        logger.debug(
            "🔍 Voice state update - WebSocket handler available: %s", websocket_available)

        # Get bot's current voice channel for comparison
        bot_voice_channel = self._get_bot_voice_channel()

        # Route to specific handlers based on voice state change
        if self._user_joined_channel(before, after):
            await self._handle_user_joined(member, after, bot_voice_channel, websocket_available)
        elif self._user_left_channel(before, after):
            await self._handle_user_left(member, before, bot_voice_channel, websocket_available)
        elif self._user_switched_channels(before, after):
            await self._handle_user_switched(member, before, after, bot_voice_channel, websocket_available)

    def _is_websocket_available(self):
        """Check if websocket handler is available for notifications."""
        return (hasattr(self.bot_manager, 'voice_translator') and
                self.bot_manager.voice_translator and
                hasattr(self.bot_manager.voice_translator, 'state') and
                hasattr(self.bot_manager.voice_translator.state, 'websocket_handler') and
                self.bot_manager.voice_translator.state.websocket_handler)

    def _get_bot_voice_channel(self):
        """Get the bot's current voice channel."""
        if (self.bot_manager.bot.voice_clients and
                len(self.bot_manager.bot.voice_clients) > 0):
            return self.bot_manager.bot.voice_clients[0].channel
        return None

    def _user_joined_channel(self, before, after):
        """Check if user joined a voice channel."""
        return before.channel is None and after.channel is not None

    def _user_left_channel(self, before, after):
        """Check if user left a voice channel."""
        return before.channel is not None and after.channel is None

    def _user_switched_channels(self, before, after):
        """Check if user switched between voice channels."""
        return (before.channel != after.channel and
                before.channel is not None and
                after.channel is not None)

    async def _handle_user_joined(self, member, after, bot_voice_channel, websocket_available):
        """Handle user joining a voice channel."""
        logger.info("👋 User %s joined voice channel: %s",
                    member.display_name, after.channel.name)

        user_id = str(member.id)

        # Only process if they joined our channel
        if bot_voice_channel and after.channel == bot_voice_channel:
            await self._enable_user_processing(member, user_id, websocket_available)

    async def _handle_user_left(self, member, before, bot_voice_channel, websocket_available):
        """Handle user leaving a voice channel."""
        logger.info("👋 User %s left voice channel: %s",
                    member.display_name, before.channel.name)

        user_id = str(member.id)

        # Only process if they left our channel
        if bot_voice_channel and before.channel == bot_voice_channel:
            await self._disable_user_processing(member, user_id, websocket_available)

    async def _handle_user_switched(self, member, before, after, bot_voice_channel, websocket_available):
        """Handle user switching between voice channels."""
        logger.info("👋 User %s switched from %s to %s",
                    member.display_name, before.channel.name, after.channel.name)

        user_id = str(member.id)

        # Check if user left our channel
        if bot_voice_channel and before.channel == bot_voice_channel:
            await self._disable_user_processing(member, user_id, websocket_available)

        # Check if user joined our channel
        elif bot_voice_channel and after.channel == bot_voice_channel:
            await self._enable_user_processing(member, user_id, websocket_available)

    async def _enable_user_processing(self, member, user_id, websocket_available):
        """Enable processing for a user and notify frontend."""
        if user_id not in self.bot_manager.user_processing_enabled:
            self.bot_manager.user_processing_enabled[user_id] = True
            logger.info("🔊 Enabled processing for new user: %s",
                        member.display_name)

        # Update voice translator's user settings
        if self.bot_manager.voice_translator:
            self.bot_manager.voice_translator.state.user_processing_enabled[user_id] = True

        # Prepare user data for frontend
        user_data = {
            "id": user_id,
            "name": member.display_name,
            "avatar": str(member.avatar.url) if member.avatar else None
        }

        # Notify frontend via WebSocket
        await self._notify_user_joined(user_data, websocket_available, member.display_name)

    async def _disable_user_processing(self, member, user_id, websocket_available):
        """Disable processing for a user and notify frontend."""
        # Remove from bot manager
        if user_id in self.bot_manager.user_processing_enabled:
            del self.bot_manager.user_processing_enabled[user_id]
            logger.info("🔇 Removed processing for user: %s",
                        member.display_name)

        # Remove from voice translator
        if (hasattr(self.bot_manager, 'voice_translator') and
            self.bot_manager.voice_translator and
                user_id in self.bot_manager.voice_translator.state.user_processing_enabled):
            del self.bot_manager.voice_translator.state.user_processing_enabled[user_id]

        # Notify frontend via WebSocket
        await self._notify_user_left(user_id, websocket_available, member.display_name)

    async def _notify_user_joined(self, user_data, websocket_available, display_name):
        """Notify frontend that a user joined."""
        if websocket_available:
            try:
                await self.bot_manager.voice_translator.state.websocket_handler.broadcast_user_joined(user_data, True)
                logger.info(
                    "📡 Notified frontend: user %s joined", display_name)
            except Exception as broadcast_error:
                logger.error("❌ Failed to broadcast user joined: %s",
                             str(broadcast_error))
        else:
            logger.warning(
                "⚠️ Cannot notify frontend - WebSocket handler not available")

    async def _notify_user_left(self, user_id, websocket_available, display_name):
        """Notify frontend that a user left."""
        if websocket_available:
            try:
                await self.bot_manager.voice_translator.state.websocket_handler.broadcast_user_left(user_id)
                logger.info("📡 Notified frontend: user %s left", display_name)
            except Exception as broadcast_error:
                logger.error("❌ Failed to broadcast user left: %s",
                             str(broadcast_error))
        else:
            logger.warning(
                "⚠️ Cannot notify frontend - WebSocket handler not available")
