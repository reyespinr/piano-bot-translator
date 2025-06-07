"""
WebSocket handler for Piano Bot Translator.

Manages WebSocket connections and handles real-time communication between
the frontend and the Discord bot.
"""
import asyncio
import json
import traceback
from typing import List, Dict, Any, Optional
from fastapi import WebSocket, WebSocketDisconnect
from logging_config import get_logger

logger = get_logger(__name__)


class WebSocketManager:
    """Manages WebSocket connections and message broadcasting."""

    def __init__(self, bot_manager):
        self.bot_manager = bot_manager
        self.active_connections: List[WebSocket] = []
        self.user_processing_enabled: Dict[str, bool] = {}

    async def handle_connection(self, websocket: WebSocket):
        """Handle a new WebSocket connection."""
        await self.connect(websocket)
        try:
            while True:
                data = await websocket.receive_text()
                await self.handle_message(websocket, data)
        except WebSocketDisconnect:
            await self.disconnect(websocket)
        except Exception as e:
            logger.error("WebSocket error: %s", str(e))
            await self.disconnect(websocket)

    async def connect(self, websocket: WebSocket):
        """Accept a WebSocket connection."""
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info("🔌 New WebSocket connection (total: %d)",
                    len(self.active_connections))

        # Send initial state to new client
        await self.send_bot_status(websocket)

        # If bot is connected to a channel, send current status
        if (self.bot_manager.bot and self.bot_manager.bot.voice_clients and
                len(self.bot_manager.bot.voice_clients) > 0):

            voice_client = self.bot_manager.bot.voice_clients[0]
            if voice_client.channel:
                connected_users = self.bot_manager.get_connected_users()

                # Send current connection status
                status_message = {
                    "type": "status",
                    "bot_ready": self.bot_manager.is_ready(),
                    "connected_channel": voice_client.channel.name,
                    "translations": [],
                    "is_listening": getattr(self.bot_manager.voice_translator, 'is_listening', False),
                    "users": connected_users,
                    "enabled_states": self.bot_manager.user_processing_enabled.copy()
                }

                await websocket.send_text(json.dumps(status_message))
                logger.debug("📡 Sent current connection status to new client")

    async def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info("🔌 WebSocket disconnected (remaining: %d)",
                    len(self.active_connections))

    async def send_guilds(self, websocket: WebSocket):
        """Send available guilds to a specific websocket."""
        try:
            guilds = await self.bot_manager.get_guilds()
            message = {
                "type": "guilds",
                "guilds": guilds
            }
            await websocket.send_text(json.dumps(message))
            logger.debug("Sent %d guilds to client", len(guilds))
        except Exception as e:
            logger.error("Error sending guilds: %s", str(e))

    async def send_bot_status(self, websocket: WebSocket):
        """Send bot status to a specific websocket."""
        try:
            is_ready = self.bot_manager.is_ready()
            message = {
                "type": "bot_status",
                "ready": is_ready
            }
            await websocket.send_text(json.dumps(message))
        except Exception as e:
            logger.error("Error sending bot status: %s", str(e))

    async def handle_message(self, websocket: WebSocket, data: str):
        """Handle incoming WebSocket message."""
        try:
            message = json.loads(data)
            command = message.get("command")

            if command == "get_guilds":
                await self.send_guilds(websocket)
            elif command == "get_channels":
                await self.handle_get_channels(websocket, message)
            elif command == "join_channel":
                await self.handle_join_channel(websocket, message)
            elif command == "leave_channel":
                await self.handle_leave_channel(websocket)
            elif command == "toggle_listen":
                await self.handle_toggle_listen(websocket)
            elif command == "toggle_user":
                await self.handle_toggle_user(websocket, message)
            else:
                logger.warning("Unknown command: %s", command)

        except json.JSONDecodeError:
            logger.error("Invalid JSON received")
        except Exception as e:
            logger.error("Error handling message: %s", str(e))
            logger.debug("Message handling error traceback: %s",
                         traceback.format_exc())

    async def handle_get_channels(self, websocket: WebSocket, message: Dict[str, Any]):
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
            await websocket.send_text(json.dumps(response))

            # Get guild name for logging
            guild_name = "Unknown"
            guilds = await self.bot_manager.get_guilds()
            for guild in guilds:
                if str(guild["id"]) == str(guild_id):
                    guild_name = guild["name"]
                    break

            logger.info("✅ Sent %d voice channels for guild '%s' (ID: %s)",
                        len(channels), guild_name, guild_id)

        except Exception as e:
            logger.error("Error getting channels: %s", str(e))

    async def handle_join_channel(self, websocket: WebSocket, message: Dict[str, Any]):
        """Handle join_channel command."""
        try:
            channel_id = message.get("channel_id")
            if not channel_id:
                await self.send_response(websocket, "join_channel", False, "No channel ID provided")
                return

            success, msg = await self.bot_manager.join_voice_channel(channel_id)
            await self.send_response(websocket, "join_channel", success, msg)

            if success:
                # CRITICAL FIX: Add a small delay to ensure users are properly detected
                await asyncio.sleep(0.2)

                # Get connected users and send comprehensive status update
                connected_users = self.bot_manager.get_connected_users()

                # CRITICAL FIX: Debug logging to see what we're getting
                logger.info(
                    "🔍 DEBUG: Connected users from bot_manager: %s", connected_users)
                logger.info("🔍 DEBUG: Bot manager user_processing_enabled: %s",
                            self.bot_manager.user_processing_enabled)

                # Sync user processing states
                self.user_processing_enabled = self.bot_manager.user_processing_enabled.copy()

                # CRITICAL FIX: If we have no users but the bot manager has processing enabled users,
                # try to reconstruct the user list from the processing states
                if not connected_users and self.bot_manager.user_processing_enabled:
                    logger.warning(
                        "🔍 No users from get_connected_users but processing states exist, attempting reconstruction...")

                    # Try to get users from the current voice channel directly
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
                            connected_users = reconstructed_users
                            logger.info(
                                "🔧 Reconstructed %d users from voice channel", len(connected_users))

                # Send comprehensive status update to all clients
                status_message = {
                    "type": "status",
                    "bot_ready": self.bot_manager.is_ready(),
                    "connected_channel": msg.replace("Connected to ", ""),
                    "translations": [],  # Empty for new connection
                    "is_listening": False,
                    "users": connected_users,
                    "enabled_states": self.user_processing_enabled
                }

                logger.info("📡 Broadcasting status update: %d users, %d enabled states",
                            len(connected_users), len(self.user_processing_enabled))
                logger.info("🔍 DEBUG: Final status message users: %s",
                            status_message["users"])
                await self.broadcast_message(status_message)

        except Exception as e:
            logger.error("Error joining channel: %s", str(e))
            await self.send_response(websocket, "join_channel", False, f"Error: {str(e)}")

    async def handle_leave_channel(self, websocket: WebSocket):
        """Handle leave_channel command."""
        try:
            success, msg = await self.bot_manager.leave_voice_channel()
            await self.send_response(websocket, "leave_channel", success, msg)

            if success:
                await self.broadcast_connection_status("Not connected to any channel")

        except Exception as e:
            logger.error("Error leaving channel: %s", str(e))
            await self.send_response(websocket, "leave_channel", False, f"Error: {str(e)}")

    async def handle_toggle_listen(self, websocket: WebSocket):
        """Handle toggle_listen command."""
        try:
            success, msg, is_listening = await self.bot_manager.toggle_listening()

            response = {
                "type": "response",
                "command": "toggle_listen",
                "success": success,
                "message": msg,
                "is_listening": is_listening
            }
            await websocket.send_text(json.dumps(response))

        except Exception as e:
            logger.error("Error toggling listen: %s", str(e))
            await self.send_response(websocket, "toggle_listen", False, f"Error: {str(e)}")

    async def handle_toggle_user(self, websocket: WebSocket, message: Dict[str, Any]):
        """Handle toggle_user command."""
        try:
            user_id = str(message.get("user_id", ""))
            enabled = bool(message.get("enabled", True))

            if user_id:
                # CRITICAL FIX: Check if user is still in the channel before allowing toggle
                connected_users = self.bot_manager.get_connected_users()
                user_still_connected = any(
                    user["id"] == user_id for user in connected_users)

                if not user_still_connected:
                    logger.warning(
                        "⚠️ Attempted to toggle user %s who is no longer connected", user_id)
                    await self.send_response(websocket, "toggle_user", False, "User is no longer in the channel")

                    # Send user_left event to clean up frontend
                    await self.broadcast_user_left(user_id)
                    return

                # Update all user processing state locations
                self.user_processing_enabled[user_id] = enabled
                self.bot_manager.user_processing_enabled[user_id] = enabled

                logger.info("🔄 User %s processing %s", user_id,
                            "enabled" if enabled else "disabled")

                # Update the translator's user settings if available
                if (hasattr(self.bot_manager, 'voice_translator') and
                        self.bot_manager.voice_translator):

                    # Update translator's user settings
                    self.bot_manager.voice_translator.user_processing_enabled[user_id] = enabled

                    # Update sink if available
                    if (hasattr(self.bot_manager.voice_translator, 'sink') and
                            self.bot_manager.voice_translator.sink):

                        sink = self.bot_manager.voice_translator.sink
                        if not hasattr(sink, 'parent') or not sink.parent:
                            sink.parent = type('obj', (object,), {})()
                        sink.parent.user_processing_enabled = self.user_processing_enabled.copy()

                # Broadcast the change to all clients
                await self.broadcast_user_toggle(user_id, enabled)

                # Send response to requesting client
                await self.send_response(websocket, "toggle_user", True, "User toggle updated")

        except Exception as e:
            logger.error("Error toggling user: %s", str(e))
            await self.send_response(websocket, "toggle_user", False, f"Error: {str(e)}")

    async def send_response(self, websocket: WebSocket, command: str, success: bool, message: str):
        """Send a command response."""
        try:
            response = {
                "type": "response",
                "command": command,
                "success": success,
                "message": message
            }
            await websocket.send_text(json.dumps(response))
        except Exception as e:
            logger.error("Error sending response: %s", str(e))

    async def broadcast_translation(self, user_id: str, text: str, message_type: str = "transcription"):
        """Broadcast transcription/translation to all connected clients."""
        try:
            # Get user display name from bot manager with error handling
            try:
                user_name = await self.bot_manager.get_user_display_name(user_id)
            except Exception as name_error:
                logger.debug("Could not get user name for %s: %s",
                             user_id, str(name_error))
                user_name = f"User {user_id}"

            message = {
                "type": message_type,
                "data": {
                    "user_id": user_id,
                    "user": user_name,
                    "text": text
                }
            }

            await self.broadcast_message(message)
            logger.debug("Broadcasted %s for user %s (%s): %s",
                         message_type, user_name, user_id, text)

        except Exception as e:
            logger.error("Error broadcasting translation: %s", str(e))
            # Fallback: try to broadcast with just user ID
            try:
                fallback_message = {
                    "type": message_type,
                    "data": {
                        "user_id": user_id,
                        "user": f"User {user_id}",
                        "text": text
                    }
                }
                await self.broadcast_message(fallback_message)
                logger.debug(
                    "Fallback broadcast successful for user %s", user_id)
            except Exception as fallback_error:
                logger.error("Fallback broadcast also failed: %s",
                             str(fallback_error))

    async def broadcast_message(self, message: Dict[str, Any]):
        """Broadcast a message to all connected clients."""
        if not self.active_connections:
            return

        disconnected = []
        message_json = json.dumps(message)

        for websocket in self.active_connections:
            try:
                await websocket.send_text(message_json)
            except Exception as e:
                logger.debug("Failed to send message to client: %s", str(e))
                disconnected.append(websocket)

        # Remove disconnected clients
        for ws in disconnected:
            self.active_connections.discard(ws)

    async def broadcast_connection_status(self, status: str):
        """Broadcast connection status to all clients."""
        message = {
            "type": "connection_status",
            "status": status
        }
        await self.broadcast_message(message)

    async def broadcast_bot_status(self, ready: bool):
        """Broadcast bot status to all clients."""
        message = {
            "type": "bot_status",
            "ready": ready
        }
        await self.broadcast_message(message)

    async def broadcast_user_toggle(self, user_id: str, enabled: bool):
        """Broadcast user toggle change to all clients."""
        message = {
            "type": "user_toggle",
            "user_id": user_id,
            "enabled": enabled
        }
        await self.broadcast_message(message)

    async def broadcast_user_joined(self, user_data: Dict[str, Any], enabled: bool):
        """Broadcast user joined event to all clients."""
        try:
            # Add user to local state tracking
            user_id = user_data["id"]
            self.user_processing_enabled[user_id] = enabled

            message = {
                "type": "user_joined",
                "user": user_data,
                "enabled": enabled
            }
            await self.broadcast_message(message)
            logger.info("📡 Broadcasted user_joined event for user %s (%s)",
                        user_data["name"], user_id)

        except Exception as e:
            logger.error("Error broadcasting user joined event: %s", str(e))

    async def broadcast_user_left(self, user_id: str):
        """Broadcast user left event to all clients."""
        try:
            # CRITICAL FIX: Clean up local user processing state when user leaves
            if user_id in self.user_processing_enabled:
                del self.user_processing_enabled[user_id]
                logger.info(
                    "🔇 Cleaned up websocket user processing state for user %s", user_id)

            message = {
                "type": "user_left",
                "user_id": user_id
            }
            await self.broadcast_message(message)
            logger.info("📡 Broadcasted user_left event for user %s", user_id)

        except Exception as e:
            logger.error("Error broadcasting user left event: %s", str(e))

    async def broadcast_listen_status(self, is_listening: bool):
        """Broadcast listening status to all clients."""
        message = {
            "type": "listen_status",
            "is_listening": is_listening
        }
        await self.broadcast_message(message)
