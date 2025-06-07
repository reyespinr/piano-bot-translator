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

        # Send initial data
        await self.send_guilds(websocket)
        await self.send_bot_status(websocket)

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
                # Get connected users and send status update
                connected_users = self.bot_manager.get_connected_users()

                # Broadcast connection status to all clients with user data
                status_message = {
                    "type": "status",
                    "connected_channel": msg.replace("Connected to ", ""),
                    "users": connected_users,
                    "enabled_states": self.bot_manager.user_processing_enabled,
                    "is_listening": False
                }
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
                self.user_processing_enabled[user_id] = enabled
                logger.info("🔄 User %s processing %s", user_id,
                            "enabled" if enabled else "disabled")

                # Update the translator's user settings
                if (hasattr(self.bot_manager, 'voice_translator') and
                    self.bot_manager.voice_translator and
                    hasattr(self.bot_manager.voice_translator, 'sink') and
                        self.bot_manager.voice_translator.sink):

                    sink = self.bot_manager.voice_translator.sink
                    if not hasattr(sink, 'parent') or not sink.parent:
                        sink.parent = type('obj', (object,), {})
                    sink.parent.user_processing_enabled = self.user_processing_enabled.copy()

                # Broadcast the change to all clients
                await self.broadcast_user_toggle(user_id, enabled)

        except Exception as e:
            logger.error("Error toggling user: %s", str(e))

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
            # Get user display name
            user_name = await self.bot_manager.get_user_display_name(user_id)

            message = {
                "type": message_type,
                "data": {
                    "user_id": user_id,
                    "user": user_name,
                    "text": text
                }
            }

            await self.broadcast_message(message)
            logger.debug("Broadcasted %s for user %s: %s",
                         message_type, user_name, text)

        except Exception as e:
            logger.error("Error broadcasting translation: %s", str(e))

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
