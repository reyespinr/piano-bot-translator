"""
Discord channel operations manager.

Handles voice channel joining, leaving, and user state management.
"""
import asyncio
from logging_config import get_logger

logger = get_logger(__name__)


class DiscordChannelManager:
    """Manages Discord voice channel operations and user states."""

    def __init__(self, bot_manager):
        """Initialize with reference to bot manager."""
        self.bot_manager = bot_manager

    async def join_voice_channel(self, channel_id):
        """Join a voice channel and initialize user processing."""
        try:
            channel = self.bot_manager.bot.get_channel(int(channel_id))
            if not channel:
                # Leave current channel if connected
                return False, "Channel not found"
            await self._leave_current_channel()

            # Join new channel with timeout and reconnect settings
            voice_client = await channel.connect(
                timeout=60.0,  # Increase connection timeout
                reconnect=True  # Enable auto-reconnect
            )

            # Set deaf/mute status after connecting to stay "active"
            if voice_client.guild.me:
                try:
                    await voice_client.guild.me.edit(deafen=False, mute=False)
                    logger.debug(
                        "Set bot as undeafened and unmuted to stay active")
                except Exception as e:
                    logger.warning(
                        "Could not set deaf/mute status: %s", str(e))

            await self._setup_channel_connection(channel)

            # Initialize user processing for channel members
            await self._initialize_channel_users(channel)

            return True, f"Connected to {channel.name}"

        except Exception as e:
            logger.error("Error joining voice channel: %s", str(e))
            return False, str(e)

    async def leave_voice_channel(self):
        """Leave current voice channel."""
        try:
            if self.bot_manager.bot.voice_clients:
                await self.bot_manager.bot.voice_clients[0].disconnect()
                return True, "Disconnected from voice channel"
            return True, "Not connected to any channel"
        except Exception as e:
            return False, str(e)

    async def _leave_current_channel(self):
        """Leave the current voice channel if connected."""
        if self.bot_manager.bot.voice_clients:
            # Stop listening if active
            if (self.bot_manager.voice_translator and
                hasattr(self.bot_manager.voice_translator, 'state') and
                    self.bot_manager.voice_translator.state.is_listening):
                await self.bot_manager.voice_translator.stop_listening(
                    self.bot_manager.bot.voice_clients[0])

            await self.bot_manager.bot.voice_clients[0].disconnect()

    async def _setup_channel_connection(self, channel):
        """Set up the voice channel connection."""
        # Store current channel in voice translator
        if self.bot_manager.voice_translator:
            self.bot_manager.voice_translator.state.current_channel = channel

        # Add delay to ensure Discord updates member list
        await asyncio.sleep(0.5)

    async def _initialize_channel_users(self, channel):
        """Initialize user processing states for all connected users."""
        connected_users = [
            member for member in channel.members if not member.bot]

        # Log channel member details
        self._log_channel_details(channel, connected_users)

        # Set up user processing states
        await self._setup_user_processing_states(connected_users)

    def _log_channel_details(self, channel, connected_users):
        """Log detailed information about channel members."""
        logger.info("🔍 Channel member details:")
        logger.info("  - Total members in channel: %d", len(channel.members))
        logger.info("  - Bot members: %d",
                    len([m for m in channel.members if m.bot]))
        logger.info("  - Human members: %d",
                    len([m for m in channel.members if not m.bot]))

        for member in channel.members:
            logger.info("  - Member: %s (ID: %s, Bot: %s)",
                        member.display_name, member.id, member.bot)

        # Log connected users summary
        if connected_users:
            user_names = [member.display_name for member in connected_users]
            logger.info("👥 Users currently in voice channel '%s': %s",
                        channel.name, ", ".join(user_names))
        else:
            logger.info("👥 No other users in voice channel '%s'", channel.name)

    async def _setup_user_processing_states(self, connected_users):
        """Set up user processing states for channel members."""
        # Clear old user processing settings and set new ones
        current_user_ids = set()

        for member in connected_users:
            user_id = str(member.id)
            current_user_ids.add(user_id)

            if user_id not in self.bot_manager.user_processing_enabled:
                self.bot_manager.user_processing_enabled[user_id] = True

            logger.debug("Added user %s (%s) to processing enabled",
                         member.display_name, user_id)

        # Remove processing settings for users not in the new channel
        await self._cleanup_old_users(current_user_ids)

        # Update voice translator settings
        if self.bot_manager.voice_translator:
            self.bot_manager.voice_translator.state.user_processing_enabled = (
                self.bot_manager.user_processing_enabled.copy())

        logger.info("🔊 Processing enabled for %d users: %s",
                    len(self.bot_manager.user_processing_enabled),
                    list(self.bot_manager.user_processing_enabled.keys()))

    async def _cleanup_old_users(self, current_user_ids):
        """Remove processing settings for users not in the current channel."""
        users_to_remove = []

        for user_id in self.bot_manager.user_processing_enabled:
            if user_id not in current_user_ids:
                users_to_remove.append(user_id)

        for user_id in users_to_remove:
            del self.bot_manager.user_processing_enabled[user_id]

    def get_connected_users(self):
        """Get list of users connected to the current voice channel."""
        if not self.bot_manager.bot or not self.bot_manager.bot.voice_clients:
            logger.debug("🔍 get_connected_users: No bot or voice clients")
            return []

        voice_client = self.bot_manager.bot.voice_clients[0]
        if not voice_client.channel:
            logger.debug("🔍 get_connected_users: Voice client has no channel")
            return []

        users = []
        logger.debug("🔍 get_connected_users: Channel has %d total members",
                     len(voice_client.channel.members))

        for member in voice_client.channel.members:
            logger.debug("🔍 get_connected_users: Member %s (ID: %s, Bot: %s)",
                         member.display_name, member.id, member.bot)

            if not member.bot:
                user_data = {
                    "id": str(member.id),
                    "name": member.display_name,
                    "avatar": str(member.avatar.url) if member.avatar else None
                }
                users.append(user_data)
                logger.debug("🔍 get_connected_users: Added user %s", user_data)

        logger.debug(
            "🔍 get_connected_users: Returning %d users: %s", len(users), users)
        return users
