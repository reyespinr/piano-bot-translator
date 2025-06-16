"""
Discord bot management module.

Handles Discord bot initialization, lifecycle, and coordination of bot components.
"""
import discord
from discord.ext import commands
from logging_config import get_logger
from discord_voice_events import DiscordVoiceEventHandler
from discord_channel_manager import DiscordChannelManager
from discord_user_manager import DiscordUserManager

logger = get_logger(__name__)


class DiscordBotManager:
    """Manages Discord bot lifecycle and operations."""

    def __init__(self):
        self.bot = None
        self.voice_translator = None
        self.user_processing_enabled = {}

        # Initialize component managers
        self.voice_event_handler = DiscordVoiceEventHandler(self)
        self.channel_manager = DiscordChannelManager(self)
        self.user_manager = DiscordUserManager(self)

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
            await self.voice_event_handler.handle_voice_state_update(member, before, after)

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
        return await self.user_manager.get_guilds()

    def is_ready(self):
        """Check if the bot is ready."""
        return self.bot and self.bot.is_ready()

    async def get_voice_channels(self, guild_id):
        """Get voice channels for a specific guild."""
        return await self.user_manager.get_voice_channels(guild_id)

    async def get_user_display_name(self, user_id):
        """Get display name for a user."""
        return await self.user_manager.get_user_display_name(user_id)

    async def join_voice_channel(self, channel_id):
        """Join a voice channel."""
        return await self.channel_manager.join_voice_channel(channel_id)

    async def leave_voice_channel(self):
        """Leave current voice channel."""
        return await self.channel_manager.leave_voice_channel()

    async def toggle_listening(self):
        """Toggle listening state."""
        if not self.voice_translator:
            return False, "Voice translator not available", False

        if not self.bot.voice_clients:
            return False, "Not connected to a voice channel", False

        # Create fresh copy of processing settings
        voice_client = self.bot.voice_clients[0]
        string_user_processing_enabled = {}

        for user_id, enabled in self.user_processing_enabled.items():
            string_user_processing_enabled[str(user_id)] = enabled

        # Set in the translator state (FIXED: Updated for refactored structure)
        self.voice_translator.state.user_processing_enabled = string_user_processing_enabled.copy()

        success, message = await self.voice_translator.toggle_listening(voice_client)
        return success, message, self.voice_translator.state.is_listening

    def get_connected_users(self):
        """Get list of users connected to the current voice channel."""
        return self.channel_manager.get_connected_users()
