"""
WebSocket message routing for Piano Bot Translator.

Handles incoming WebSocket messages and routes them to appropriate handlers.
Manages all bot commands and user interactions.
"""
import asyncio
import json
import traceback
from typing import Dict, Any
from fastapi import WebSocket
from logging_config import get_logger

logger = get_logger(__name__)


class WebSocketMessageRouter:
    """Routes WebSocket messages to appropriate handlers."""

    def __init__(self, bot_manager, connection_manager, broadcaster, state_manager):
        self.bot_manager = bot_manager
        self.connection_manager = connection_manager
        self.broadcaster = broadcaster
        self.state_manager = state_manager

    async def handle_message(self, websocket: WebSocket, data: str):
        """Route incoming WebSocket message to appropriate handler."""
        try:
            message = json.loads(data)
            command = message.get("command")

            # Route to appropriate handler
            if command == "get_guilds":
                await self._handle_get_guilds(websocket)
            elif command == "get_channels":
                await self._handle_get_channels(websocket, message)
            elif command == "join_channel":
                await self._handle_join_channel(websocket, message)
            elif command == "leave_channel":
                await self._handle_leave_channel(websocket)
            elif command == "toggle_listen":
                await self._handle_toggle_listen(websocket)
            elif command == "toggle_user":
                await self._handle_toggle_user(websocket, message)
            else:
                logger.warning("Unknown command: %s", command)

        except json.JSONDecodeError:
            logger.error("Invalid JSON received from WebSocket")
        except Exception as e:
            logger.error("Error handling WebSocket message: %s", str(e))
            logger.debug("Message handling error traceback: %s",
                         traceback.format_exc())

    async def send_initial_state(self, websocket: WebSocket):
        """Send initial state to a newly connected client."""
        # Send bot status
        await self._send_bot_status(websocket)

        # If bot is connected to a channel, send current status
        if (self.bot_manager.bot and self.bot_manager.bot.voice_clients and
                len(self.bot_manager.bot.voice_clients) > 0):

            voice_client = self.bot_manager.bot.voice_clients[0]
            if voice_client.channel:
                connected_users = self.bot_manager.get_connected_users()

                status_message = {
                    "type": "status",
                    "bot_ready": self.bot_manager.is_ready(),
                    "connected_channel": voice_client.channel.name,
                    "translations": [],
                    "is_listening": getattr(self.bot_manager.voice_translator, 'is_listening', False),
                    "users": connected_users,
                    "enabled_states": self.state_manager.user_processing_enabled.copy()
                }

                await self.connection_manager.send_to_connection(websocket, status_message)
                logger.debug("📡 Sent current connection status to new client")

    async def _handle_get_guilds(self, websocket: WebSocket):
        """Handle get_guilds command."""
        try:
            guilds = await self.bot_manager.get_guilds()
            message = {
                "type": "guilds",
                "guilds": guilds
            }
            await self.connection_manager.send_to_connection(websocket, message)
            logger.debug("Sent %d guilds to client", len(guilds))
        except Exception as e:
            logger.error("Error sending guilds: %s", str(e))

    async def _handle_get_channels(self, websocket: WebSocket, message: Dict[str, Any]):
        """Handle get_channels command."""
        try:
            guild_id = message.get("guild_id")
            if not guild_id:
                return

            logger.debug("Received guild_id: '%s'", guild_id)
            channels = await self.bot_manager.get_voice_channels(guild_id)

            response = {
                "type": "channels",
                "channels": channels
            }
            await self.connection_manager.send_to_connection(websocket, response)

            # Get guild name for logging
            guild_name = await self._get_guild_name(guild_id)
            logger.info("✅ Sent %d voice channels for guild '%s' (ID: %s)",
                        len(channels), guild_name, guild_id)

        except Exception as e:
            logger.error("Error getting channels: %s", str(e))

    async def _handle_join_channel(self, websocket: WebSocket, message: Dict[str, Any]):
        """Handle join_channel command."""
        try:
            channel_id = message.get("channel_id")
            if not channel_id:
                await self.connection_manager.send_response(
                    websocket, "join_channel", False, "No channel ID provided"
                )
                return

            success, msg = await self.bot_manager.join_voice_channel(channel_id)
            await self.connection_manager.send_response(websocket, "join_channel", success, msg)

            if success:
                await self._handle_successful_join(msg)

        except Exception as e:
            logger.error("Error joining channel: %s", str(e))
            await self.connection_manager.send_response(
                websocket, "join_channel", False, f"Error: {str(e)}"
            )

    async def _handle_leave_channel(self, websocket: WebSocket):
        """Handle leave_channel command."""
        try:
            success, msg = await self.bot_manager.leave_voice_channel()
            await self.connection_manager.send_response(websocket, "leave_channel", success, msg)

            if success:
                await self.broadcaster.broadcast_connection_status("Not connected to any channel")

        except Exception as e:
            logger.error("Error leaving channel: %s", str(e))
            await self.connection_manager.send_response(
                websocket, "leave_channel", False, f"Error: {str(e)}"
            )

    async def _handle_toggle_listen(self, websocket: WebSocket):
        """Handle toggle_listen command."""
        try:
            success, msg, is_listening = await self.bot_manager.toggle_listening()

            await self.connection_manager.send_response(
                websocket, "toggle_listen", success, msg, is_listening=is_listening
            )

        except Exception as e:
            logger.error("Error toggling listen: %s", str(e))
            await self.connection_manager.send_response(
                websocket, "toggle_listen", False, f"Error: {str(e)}"
            )

    async def _handle_toggle_user(self, websocket: WebSocket, message: Dict[str, Any]):
        """Handle toggle_user command."""
        try:
            user_id = str(message.get("user_id", ""))
            enabled = bool(message.get("enabled", True))

            if not user_id:
                await self.connection_manager.send_response(
                    websocket, "toggle_user", False, "No user ID provided"
                )
                return

            # Check if user is still connected
            connected_users = self.bot_manager.get_connected_users()
            user_still_connected = any(
                user["id"] == user_id for user in connected_users)

            if not user_still_connected:
                logger.warning(
                    "⚠️ Attempted to toggle user %s who is no longer connected", user_id)
                await self.connection_manager.send_response(
                    websocket, "toggle_user", False, "User is no longer in the channel"
                )
                await self.broadcaster.broadcast_user_left(user_id)
                return            # Update user processing state locally
            self.state_manager.user_processing_enabled[user_id] = enabled

            # Update bot manager and translator components
            success = self.state_manager.update_bot_manager_and_translator(
                self.bot_manager)

            if success:
                await self.broadcaster.broadcast_user_toggle(user_id, enabled)
                await self.connection_manager.send_response(
                    websocket, "toggle_user", True, "User toggle updated"
                )
                logger.info("🔄 User %s processing %s", user_id,
                            "enabled" if enabled else "disabled")
            else:
                await self.connection_manager.send_response(
                    websocket, "toggle_user", False, "Failed to update user toggle"
                )

        except Exception as e:
            logger.error("Error toggling user: %s", str(e))
            await self.connection_manager.send_response(
                websocket, "toggle_user", False, f"Error: {str(e)}"
            )

    async def _send_bot_status(self, websocket: WebSocket):
        """Send bot status to a specific websocket."""
        try:
            is_ready = self.bot_manager.is_ready()
            message = {
                "type": "bot_status",
                "ready": is_ready
            }
            await self.connection_manager.send_to_connection(websocket, message)
        except Exception as e:
            logger.error("Error sending bot status: %s", str(e))

    async def _get_guild_name(self, guild_id: str) -> str:
        """Get guild name for logging purposes."""
        try:
            guilds = await self.bot_manager.get_guilds()
            for guild in guilds:
                if str(guild["id"]) == str(guild_id):
                    return guild["name"]
        except Exception:
            pass
        return "Unknown"

    async def _handle_successful_join(self, join_message: str):
        """Handle successful channel join - send status updates."""
        # Add a small delay to ensure users are properly detected
        await asyncio.sleep(0.2)

        # Get connected users and send comprehensive status update
        connected_users = self.bot_manager.get_connected_users()

        logger.debug(
            "🔍 DEBUG: Connected users from bot_manager: %s", connected_users)
        logger.debug("🔍 DEBUG: Bot manager user_processing_enabled: %s",
                     self.bot_manager.user_processing_enabled)

        # Sync user processing states
        self.state_manager.sync_with_bot_manager(self.bot_manager)

        # Try to reconstruct user list if needed
        if not connected_users and self.bot_manager.user_processing_enabled:
            connected_users = await self._reconstruct_user_list()

        # Send comprehensive status update to all clients
        status_message = {
            "type": "status",
            "bot_ready": self.bot_manager.is_ready(),
            "connected_channel": join_message.replace("Connected to ", ""),
            "translations": [],  # Empty for new connection
            "is_listening": False,
            "users": connected_users,
            "enabled_states": self.state_manager.user_processing_enabled.copy()
        }

        logger.info("📡 Broadcasting status update: %d users, %d enabled states",
                    len(connected_users), len(self.state_manager.user_processing_enabled))
        logger.debug("🔍 DEBUG: Final status message users: %s",
                     status_message["users"])

        await self.broadcaster.broadcast_to_all(status_message)

    async def _reconstruct_user_list(self) -> list:
        """Reconstruct user list from voice channel if needed."""
        try:
            if (self.bot_manager.bot and self.bot_manager.bot.voice_clients and
                    len(self.bot_manager.bot.voice_clients) > 0):

                voice_client = self.bot_manager.bot.voice_clients[0]
                if voice_client.channel:
                    reconstructed_users = []
                    for member in voice_client.channel.members:
                        if not member.bot:
                            reconstructed_users.append({
                                "id": str(member.id),
                                "name": member.display_name,
                                "avatar": str(member.avatar.url) if member.avatar else None
                            })

                    logger.info("🔧 Reconstructed %d users from voice channel", len(
                        reconstructed_users))
                    return reconstructed_users
        except Exception as e:
            logger.warning("Failed to reconstruct user list: %s", str(e))

        return []
