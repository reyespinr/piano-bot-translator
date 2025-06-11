"""
Discord user management utilities.

Handles user lookup, display name resolution, and user data operations.
"""
import asyncio
import discord
from logging_config import get_logger

logger = get_logger(__name__)


class DiscordUserManager:
    """Manages Discord user operations and data retrieval."""

    def __init__(self, bot_manager):
        """Initialize with reference to bot manager."""
        self.bot_manager = bot_manager

    async def get_user_display_name(self, user_id):
        """Get display name for a user with multiple fallback strategies."""
        if not self.bot_manager.bot or not self.bot_manager.bot.is_ready():
            return f"User {user_id}"

        try:
            user_id_int = int(user_id)

            # Strategy 1: Try to get from current voice channel if connected
            display_name = await self._get_name_from_voice_channel(user_id_int)
            if display_name:
                return display_name

            # Strategy 2: Fallback to getting user from bot cache (synchronous)
            display_name = self._get_name_from_cache(user_id_int)
            if display_name:
                return display_name

            # Strategy 3: Last resort - try to fetch user (with timeout protection)
            display_name = await self._fetch_user_with_timeout(user_id_int)
            if display_name:
                return display_name

        except (ValueError, AttributeError) as e:
            logger.debug(
                "Could not resolve user name for ID %s: %s", user_id, str(e))

        return f"User {user_id}"

    async def _get_name_from_voice_channel(self, user_id_int):
        """Try to get user display name from current voice channel members."""
        if not self.bot_manager.bot.voice_clients:
            return None

        voice_client = self.bot_manager.bot.voice_clients[0]
        if not voice_client.channel:
            return None

        for member in voice_client.channel.members:
            if member.id == user_id_int:
                return member.display_name

        return None

    def _get_name_from_cache(self, user_id_int):
        """Get user display name from bot's user cache."""
        user = self.bot_manager.bot.get_user(user_id_int)
        if user:
            return user.display_name
        return None

    async def _fetch_user_with_timeout(self, user_id_int):
        """Fetch user from Discord API with timeout protection."""
        try:
            user = await asyncio.wait_for(
                self.bot_manager.bot.fetch_user(user_id_int),
                timeout=2.0  # 2 second timeout
            )
            if user:
                return user.display_name
        except (asyncio.TimeoutError, discord.NotFound, discord.HTTPException):
            # Fetch failed, return None to fall through to default
            pass
        return None

    async def get_guilds(self):
        """Get list of available guilds/servers."""
        if not self.bot_manager.bot or not self.bot_manager.bot.is_ready():
            return []

        guilds = []
        for guild in self.bot_manager.bot.guilds:
            guilds.append({
                "id": str(guild.id),
                "name": guild.name
            })
        return guilds

    async def get_voice_channels(self, guild_id):
        """Get voice channels for a specific guild."""
        if not self.bot_manager.bot or not self.bot_manager.bot.is_ready():
            return []

        try:
            guild_id_int = int(guild_id)
            guild = self.bot_manager.bot.get_guild(guild_id_int)
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
