"""
Discord bot management module.

Handles Discord bot initialization, events, and voice channel operations.
"""
import asyncio
import discord
from discord.ext import commands
from logging_config import get_logger

logger = get_logger(__name__)


class DiscordBotManager:
    """Manages Discord bot lifecycle and operations."""

    def __init__(self):
        self.bot = None
        self.voice_translator = None
        self.user_processing_enabled = {}

    def set_voice_translator(self, voice_translator):
        """Set the voice translator instance."""
        self.voice_translator = voice_translator

    def create_bot(self, translation_callback):
        """Create and configure the Discord bot."""
        intents = discord.Intents.default()
        intents.voice_states = True
        intents.guilds = True
        intents.message_content = True
        intents.members = True  # CRITICAL FIX: Enable member intent to see channel members

        self.bot = commands.Bot(command_prefix='!', intents=intents)

        @self.bot.event
        async def on_ready():
            logger.info("🤖 Discord bot logged in as: %s", self.bot.user)
            logger.info("🤖 Bot is ready and connected to Discord!")

        @self.bot.event
        async def on_voice_state_update(member, before, after):
            """Handle voice state updates to track users joining/leaving."""
            if member.bot:
                return

            # CRITICAL DEBUG: Check if websocket handler is available
            websocket_available = (hasattr(self, 'voice_translator') and
                                   self.voice_translator and
                                   hasattr(self.voice_translator, 'websocket_handler') and
                                   self.voice_translator.websocket_handler)

            logger.debug(
                "🔍 Voice state update - WebSocket handler available: %s", websocket_available)

            # Get bot's current voice channel for comparison
            bot_voice_channel = None
            if (self.bot.voice_clients and len(self.bot.voice_clients) > 0):
                bot_voice_channel = self.bot.voice_clients[0].channel

            # User joined a voice channel
            if before.channel is None and after.channel is not None:
                logger.info("👋 User %s joined voice channel: %s",
                            member.display_name, after.channel.name)
                user_id = str(member.id)

                # If they joined our channel, enable processing and notify frontend
                if bot_voice_channel and after.channel == bot_voice_channel:
                    if user_id not in self.user_processing_enabled:
                        self.user_processing_enabled[user_id] = True
                        logger.info(
                            "🔊 Enabled processing for new user: %s", member.display_name)

                    # Update voice translator's user settings
                    if self.voice_translator:
                        self.voice_translator.user_processing_enabled[user_id] = True

                    user_data = {
                        "id": user_id,
                        "name": member.display_name,
                        "avatar": str(member.avatar.url) if member.avatar else None
                    }

                    # CRITICAL FIX: Add more detailed debugging for WebSocket broadcast
                    if websocket_available:
                        try:
                            await self.voice_translator.websocket_handler.broadcast_user_joined(user_data, True)
                            logger.info(
                                "📡 Notified frontend: user %s joined", member.display_name)
                        except Exception as broadcast_error:
                            logger.error(
                                "❌ Failed to broadcast user joined: %s", str(broadcast_error))
                    else:
                        logger.warning(
                            "⚠️ Cannot notify frontend - WebSocket handler not available")

            # User left a voice channel
            elif before.channel is not None and after.channel is None:
                logger.info("👋 User %s left voice channel: %s",
                            member.display_name, before.channel.name)

                user_id = str(member.id)

                # If they left our channel, clean up and notify frontend
                if bot_voice_channel and before.channel == bot_voice_channel:
                    # Clean up user processing state
                    if user_id in self.user_processing_enabled:
                        del self.user_processing_enabled[user_id]
                        logger.info(
                            "🔇 Removed processing for user: %s", member.display_name)

                    # Update voice translator's user settings
                    if (hasattr(self, 'voice_translator') and self.voice_translator):
                        if user_id in self.voice_translator.user_processing_enabled:
                            del self.voice_translator.user_processing_enabled[user_id]

                    # CRITICAL FIX: Add more detailed debugging for WebSocket broadcast
                    if websocket_available:
                        try:
                            await self.voice_translator.websocket_handler.broadcast_user_left(user_id)
                            logger.info(
                                "📡 Notified frontend: user %s left", member.display_name)
                        except Exception as broadcast_error:
                            logger.error(
                                "❌ Failed to broadcast user left: %s", str(broadcast_error))
                    else:
                        logger.warning(
                            "⚠️ Cannot notify frontend - WebSocket handler not available")

            # User switched channels
            elif before.channel != after.channel and before.channel is not None and after.channel is not None:
                logger.info("👋 User %s switched from %s to %s",
                            member.display_name, before.channel.name, after.channel.name)

                user_id = str(member.id)

                # Check if user left our channel
                if bot_voice_channel and before.channel == bot_voice_channel:
                    # User left our channel
                    if user_id in self.user_processing_enabled:
                        del self.user_processing_enabled[user_id]
                        logger.info(
                            "🔇 Removed processing for user: %s (switched channels)", member.display_name)

                    # Update voice translator's user settings
                    if (hasattr(self, 'voice_translator') and self.voice_translator):
                        if user_id in self.voice_translator.user_processing_enabled:
                            del self.voice_translator.user_processing_enabled[user_id]

                    # CRITICAL FIX: Add more detailed debugging for WebSocket broadcast
                    if websocket_available:
                        try:
                            await self.voice_translator.websocket_handler.broadcast_user_left(user_id)
                            logger.info(
                                "📡 Notified frontend: user %s left our channel", member.display_name)
                        except Exception as broadcast_error:
                            logger.error(
                                "❌ Failed to broadcast user left: %s", str(broadcast_error))
                    else:
                        logger.warning(
                            "⚠️ Cannot notify frontend - WebSocket handler not available")

                # Check if user joined our channel
                elif bot_voice_channel and after.channel == bot_voice_channel:
                    # User joined our channel
                    if user_id not in self.user_processing_enabled:
                        self.user_processing_enabled[user_id] = True
                        logger.info(
                            "🔊 Enabled processing for user: %s (switched to our channel)", member.display_name)

                    # Update voice translator's user settings
                    if self.voice_translator:
                        self.voice_translator.user_processing_enabled[user_id] = True

                    user_data = {
                        "id": user_id,
                        "name": member.display_name,
                        "avatar": str(member.avatar.url) if member.avatar else None
                    }

                    # CRITICAL FIX: Add more detailed debugging for WebSocket broadcast
                    if websocket_available:
                        try:
                            await self.voice_translator.websocket_handler.broadcast_user_joined(user_data, True)
                            logger.info(
                                "📡 Notified frontend: user %s joined our channel", member.display_name)
                        except Exception as broadcast_error:
                            logger.error(
                                "❌ Failed to broadcast user joined: %s", str(broadcast_error))
                    else:
                        logger.warning(
                            "⚠️ Cannot notify frontend - WebSocket handler not available")

        return self.bot

    async def start_bot(self, token):
        """Start the Discord bot with the provided token."""
        if not self.bot:
            raise RuntimeError("Bot not created. Call create_bot() first.")

        try:
            await self.bot.start(token)
        except Exception as e:
            logger.error("Failed to start bot: %s", str(e))
            raise

    async def stop_bot(self):
        """Stop the Discord bot."""
        if self.bot and not self.bot.is_closed():
            await self.bot.close()

    async def get_guilds(self):
        """Get list of available guilds/servers."""
        if not self.bot or not self.bot.is_ready():
            return []

        guilds = []
        for guild in self.bot.guilds:
            guilds.append({
                "id": str(guild.id),
                "name": guild.name
            })
        return guilds

    def is_ready(self):
        """Check if the bot is ready."""
        return self.bot and self.bot.is_ready()

    async def get_voice_channels(self, guild_id):
        """Get voice channels for a specific guild."""
        if not self.bot or not self.bot.is_ready():
            return []

        try:
            guild_id_int = int(guild_id)
            guild = self.bot.get_guild(guild_id_int)
            if not guild:
                return []

            channels = []
            for channel in guild.voice_channels:
                channels.append({
                    "id": str(channel.id),
                    "name": channel.name
                })
            return channels
        except (ValueError, AttributeError):
            return []

    async def get_user_display_name(self, user_id):
        """Get display name for a user."""
        if not self.bot or not self.bot.is_ready():
            return f"User {user_id}"

        try:
            user_id_int = int(user_id)

            # First try to get from current voice channel if connected
            if self.bot.voice_clients:
                voice_client = self.bot.voice_clients[0]
                if voice_client.channel:
                    for member in voice_client.channel.members:
                        if member.id == user_id_int:
                            return member.display_name

            # Fallback to getting user from bot cache (synchronous)
            user = self.bot.get_user(user_id_int)
            if user:
                return user.display_name

            # Last resort: try to fetch user (but with timeout protection)
            try:
                user = await asyncio.wait_for(
                    self.bot.fetch_user(user_id_int),
                    timeout=2.0  # 2 second timeout
                )
                if user:
                    return user.display_name
            except (asyncio.TimeoutError, discord.NotFound, discord.HTTPException):
                # Fetch failed, fall through to default
                pass

        except (ValueError, AttributeError) as e:
            logger.debug(
                "Could not resolve user name for ID %s: %s", user_id, str(e))

        return f"User {user_id}"

    async def join_voice_channel(self, channel_id):
        """Join a voice channel."""
        try:
            channel = self.bot.get_channel(int(channel_id))
            if not channel:
                return False, "Channel not found"

            # Leave current channel if connected
            if self.bot.voice_clients:
                # Stop listening if active
                if (self.voice_translator and
                    hasattr(self.voice_translator, 'is_listening') and
                        self.voice_translator.is_listening):
                    await self.voice_translator.stop_listening(self.bot.voice_clients[0])

                await self.bot.voice_clients[0].disconnect()

            # Join new channel
            voice_client = await channel.connect()

            # Store current channel in voice translator
            if self.voice_translator:
                self.voice_translator.current_channel = channel

            # CRITICAL FIX: Add a small delay to ensure Discord updates member list
            await asyncio.sleep(0.5)

            # Initialize user processing states for all connected users
            connected_users = [
                member for member in channel.members if not member.bot]

            # CRITICAL FIX: Add detailed debugging for user detection
            logger.info("🔍 Channel member details:")
            logger.info("  - Total members in channel: %d",
                        len(channel.members))
            logger.info("  - Bot members: %d",
                        len([m for m in channel.members if m.bot]))
            logger.info("  - Human members: %d",
                        len([m for m in channel.members if not m.bot]))

            for member in channel.members:
                logger.info("  - Member: %s (ID: %s, Bot: %s)",
                            member.display_name, member.id, member.bot)

            # Log connected users when joining channel
            if connected_users:
                user_names = [
                    member.display_name for member in connected_users]
                logger.info("👥 Users currently in voice channel '%s': %s",
                            channel.name, ", ".join(user_names))
            else:
                logger.info(
                    "👥 No other users in voice channel '%s'", channel.name)

            # Clear old user processing settings and set new ones
            current_user_ids = set()
            for member in connected_users:
                user_id = str(member.id)
                current_user_ids.add(user_id)
                if user_id not in self.user_processing_enabled:
                    self.user_processing_enabled[user_id] = True
                logger.debug("Added user %s (%s) to processing enabled",
                             member.display_name, user_id)

            # Remove processing settings for users not in the new channel
            users_to_remove = []
            for user_id in self.user_processing_enabled:
                if user_id not in current_user_ids:
                    users_to_remove.append(user_id)

            for user_id in users_to_remove:
                del self.user_processing_enabled[user_id]

            # Update voice translator's user settings
            if self.voice_translator:
                self.voice_translator.user_processing_enabled = self.user_processing_enabled.copy()

            logger.info("🔊 Processing enabled for %d users: %s",
                        len(self.user_processing_enabled),
                        list(self.user_processing_enabled.keys()))

            return True, f"Connected to {channel.name}"
        except Exception as e:
            logger.error("Error joining voice channel: %s", str(e))
            return False, str(e)

    async def leave_voice_channel(self):
        """Leave current voice channel."""
        try:
            if self.bot.voice_clients:
                await self.bot.voice_clients[0].disconnect()
                return True, "Disconnected from voice channel"
            return True, "Not connected to any channel"
        except Exception as e:
            return False, str(e)

    async def toggle_listening(self):
        """Toggle listening state."""
        if not self.voice_translator:
            return False, "Voice translator not available", False

        if not self.bot.voice_clients:
            return False, "Not connected to a voice channel", False

        voice_client = self.bot.voice_clients[0]

        # Create fresh copy of processing settings
        string_user_processing_enabled = {}
        for user_id, enabled in self.user_processing_enabled.items():
            string_user_processing_enabled[str(user_id)] = enabled

        # Set in the translator
        self.voice_translator.user_processing_enabled = string_user_processing_enabled.copy()

        success, message = await self.voice_translator.toggle_listening(voice_client)
        return success, message, self.voice_translator.is_listening

    def get_connected_users(self):
        """Get list of users connected to the current voice channel."""
        if not self.bot or not self.bot.voice_clients:
            logger.debug("🔍 get_connected_users: No bot or voice clients")
            return []

        voice_client = self.bot.voice_clients[0]
        if not voice_client.channel:
            logger.debug("🔍 get_connected_users: Voice client has no channel")
            return []

        users = []
        logger.debug("🔍 get_connected_users: Channel has %d total members", len(
            voice_client.channel.members))

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
