"""
Discord channel operations manager.

Handles voice channel joining, leaving, and user state management.
"""
import asyncio
import logging
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
                return False, "Channel not found"

            # Check if already connected to the same channel
            if (self.bot_manager.bot.voice_clients and
                    self.bot_manager.bot.voice_clients[0].channel.id == channel.id):
                logger.info("Already connected to channel: %s", channel.name)
                return True, f"Already connected to {channel.name}"

            # Leave current channel if connected to a different one
            await self._leave_current_channel()

            # Reduce delay for faster connection
            await asyncio.sleep(0.5)

            logger.info("Attempting to connect to channel: %s (ID: %s)",
                        channel.name, channel.id)

            # Join new channel with increased timeout and retry logic
            try:
                voice_client = await asyncio.wait_for(
                    channel.connect(timeout=30.0, reconnect=True),
                    timeout=35.0  # Overall timeout wrapper
                )
            except asyncio.TimeoutError:
                logger.error("Connection to voice channel timed out")
                return False, "Connection timeout - try again"

            logger.info(
                "Successfully connected to voice channel: %s", channel.name)

            # Set deaf/mute status after connecting to stay "active"
            if voice_client.guild.me:
                try:
                    await asyncio.wait_for(
                        voice_client.guild.me.edit(deafen=False, mute=False),
                        timeout=5.0
                    )
                    logger.debug(
                        "Set bot as undeafened and unmuted to stay active")
                except Exception as e:
                    # Quick setup without delays
                    logger.warning(
                        "Could not set deaf/mute status: %s", str(e))
            await self._setup_channel_connection_fast(channel, voice_client)

            # Verify connection stability
            if not await self._ensure_connection_stability(voice_client):
                logger.error("Connection unstable, aborting setup")
                await voice_client.disconnect()
                return False, "Connection unstable - try again"

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
            current_voice_client = self.bot_manager.bot.voice_clients[0]
            channel_name = current_voice_client.channel.name if current_voice_client.channel else "Unknown"

            logger.info("Leaving current voice channel: %s", channel_name)

            # Stop listening if active
            if (self.bot_manager.voice_translator and
                hasattr(self.bot_manager.voice_translator, 'state') and
                    self.bot_manager.voice_translator.state.is_listening):
                logger.debug("Stopping listening before disconnecting...")
                await self.bot_manager.voice_translator.stop_listening(current_voice_client)

            # Disconnect from voice channel
            await current_voice_client.disconnect()
            logger.info(
                "Successfully disconnected from voice channel: %s", channel_name)

            # Clear voice translator state
            if self.bot_manager.voice_translator:
                self.bot_manager.voice_translator.state.current_channel = None
                self.bot_manager.voice_translator.state.voice_client = None

    async def _setup_channel_connection(self, channel):
        """Set up the voice channel connection."""
        # Store current channel in voice translator
        if self.bot_manager.voice_translator:
            self.bot_manager.voice_translator.state.current_channel = channel

        # Add delay to ensure Discord updates member list
        await asyncio.sleep(0.5)

    async def _setup_channel_connection_fast(self, channel, voice_client):
        """Set up the voice channel connection quickly."""
        # Store current channel and voice client in voice translator
        if self.bot_manager.voice_translator:
            self.bot_manager.voice_translator.state.current_channel = channel
            self.bot_manager.voice_translator.state.voice_client = voice_client

        logger.debug(
            "Channel connection setup completed for: %s", channel.name)

    async def _initialize_channel_users(self, channel):
        """Initialize user processing states for all connected users."""
        try:
            # Quick user discovery
            connected_users = [
                member for member in channel.members if not member.bot]

            logger.info("👥 Found %d users in channel: %s",
                        len(connected_users), channel.name)

            # Fast user processing setup (no detailed logging during connection)
            await self._setup_user_processing_states_fast(connected_users)

            # Log details after setup is complete
            if logger.isEnabledFor(logging.DEBUG):
                self._log_channel_details(channel, connected_users)

        except Exception as e:
            logger.error("Error initializing channel users: %s", str(e))
            # Don't fail the entire connection for user setup issues

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

    async def _setup_user_processing_states_fast(self, connected_users):
        """Fast setup of user processing states without detailed logging."""
        current_user_ids = set()

        for member in connected_users:
            user_id = str(member.id)
            current_user_ids.add(user_id)

            if user_id not in self.bot_manager.user_processing_enabled:
                self.bot_manager.user_processing_enabled[user_id] = True

        # Remove processing settings for users not in the new channel
        await self._cleanup_old_users(current_user_ids)

        # Update voice translator settings
        if self.bot_manager.voice_translator:
            self.bot_manager.voice_translator.state.user_processing_enabled = (
                self.bot_manager.user_processing_enabled.copy())

        logger.info("🔊 Processing enabled for %d users",
                    len(self.bot_manager.user_processing_enabled))

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

    async def _ensure_connection_stability(self, voice_client):
        """Ensure the voice connection remains stable."""
        try:
            # Check if voice client is still connected
            if not voice_client.is_connected():
                logger.warning("Voice client disconnected during setup")
                return False

            # Send a keep-alive by checking latency
            latency = voice_client.latency
            logger.debug("Voice connection latency: %.2fms", latency * 1000)

            return True
        except Exception as e:
            logger.warning("Connection stability check failed: %s", str(e))
            return False
