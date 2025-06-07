"""
Discord bot management module.

Handles Discord bot initialization, events, and voice channel operations.
"""
import asyncio
import os
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

            # User joined a voice channel
            if before.channel is None and after.channel is not None:
                logger.info("👋 User %s joined voice channel: %s",
                            member.display_name, after.channel.name)
                user_id = str(member.id)
                if user_id not in self.user_processing_enabled:
                    self.user_processing_enabled[user_id] = True
                    logger.info(
                        "🔊 Enabled processing for new user: %s", member.display_name)

            # User left a voice channel
            elif before.channel is not None and after.channel is None:
                logger.info("👋 User %s left voice channel: %s",
                            member.display_name, before.channel.name)

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
            user = self.bot.get_user(int(user_id))
            return user.display_name if user else f"User {user_id}"
        except (ValueError, AttributeError):
            return f"User {user_id}"

    async def join_voice_channel(self, channel_id):
        """Join a voice channel."""
        try:
            channel = self.bot.get_channel(int(channel_id))
            if not channel:
                return False, "Channel not found"

            # Leave current channel if connected
            if self.bot.voice_clients:
                await self.bot.voice_clients[0].disconnect()

            # Join new channel
            await channel.connect()

            # Initialize user processing states for all connected users
            connected_users = [
                member for member in channel.members if not member.bot]
            for member in connected_users:
                user_id = str(member.id)
                if user_id not in self.user_processing_enabled:
                    self.user_processing_enabled[user_id] = True

            return True, f"Connected to {channel.name}"
        except Exception as e:
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
        success, message = await self.voice_translator.toggle_listening(voice_client)
        return success, message, self.voice_translator.is_listening

    def get_connected_users(self):
        """Get list of users connected to the current voice channel."""
        if not self.bot or not self.bot.voice_clients:
            return []

        voice_client = self.bot.voice_clients[0]
        if not voice_client.channel:
            return []

        users = []
        for member in voice_client.channel.members:
            if not member.bot:
                users.append({
                    "id": str(member.id),
                    "name": member.display_name
                })
        return users
