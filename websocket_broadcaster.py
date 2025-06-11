"""
WebSocket message broadcasting for Piano Bot Translator.

Handles broadcasting messages to multiple WebSocket clients with
support for different connection types and message filtering.
"""
import json
from typing import Dict, Any, List
from fastapi import WebSocket
from logging_config import get_logger

logger = get_logger(__name__)


class WebSocketBroadcaster:
    """Handles broadcasting messages to WebSocket clients."""

    def __init__(self, connection_manager):
        self.connection_manager = connection_manager

    async def broadcast_to_all(self, message: Dict[str, Any]):
        """Broadcast a message to all connected clients."""
        await self._broadcast_to_connections(
            self.connection_manager.active_connections,
            message
        )

    async def broadcast_to_admins(self, message: Dict[str, Any]):
        """Broadcast a message only to admin connections."""
        admin_connections = self.connection_manager.get_connections_by_type(
            "admin")
        await self._broadcast_to_connections(admin_connections, message)

    async def broadcast_to_spectators(self, message: Dict[str, Any]):
        """Broadcast a message only to spectator connections."""
        spectator_connections = self.connection_manager.get_connections_by_type(
            "spectator")
        await self._broadcast_to_connections(spectator_connections, message)

    async def broadcast_translation(self, user_id: str, text: str, message_type: str = "transcription", bot_manager=None):
        """Broadcast transcription/translation to all connected clients."""
        try:
            # Get user display name with error handling
            user_name = await self._get_user_name(user_id, bot_manager)

            message = {
                "type": message_type,
                "data": {
                    "user_id": user_id,
                    "user": user_name,
                    "text": text
                }
            }

            await self.broadcast_to_all(message)
            logger.debug("Broadcasted %s for user %s (%s): %s",
                         message_type, user_name, user_id, text)

        except Exception as e:
            logger.error("Error broadcasting translation: %s", str(e))
            # Fallback: try to broadcast with just user ID
            await self._broadcast_fallback(user_id, text, message_type)

    async def broadcast_bot_status(self, ready: bool):
        """Broadcast bot status to all clients."""
        message = {
            "type": "bot_status",
            "ready": ready
        }
        await self.broadcast_to_all(message)

    async def broadcast_connection_status(self, status: str):
        """Broadcast connection status to all clients."""
        message = {
            "type": "connection_status",
            "status": status
        }
        await self.broadcast_to_all(message)

    async def broadcast_listen_status(self, is_listening: bool):
        """Broadcast listening status to all clients."""
        message = {
            "type": "listen_status",
            "is_listening": is_listening
        }
        await self.broadcast_to_all(message)

    async def broadcast_user_toggle(self, user_id: str, enabled: bool):
        """Broadcast user toggle change to admin clients only."""
        message = {
            "type": "user_toggle",
            "user_id": user_id,
            "enabled": enabled
        }
        await self.broadcast_to_admins(message)

    async def broadcast_user_joined(self, user_data: Dict[str, Any], enabled: bool):
        """Broadcast user joined event to all clients."""
        try:
            message = {
                "type": "user_joined",
                "user": user_data,
                "enabled": enabled
            }
            await self.broadcast_to_all(message)
            logger.info("📡 Broadcasted user_joined event for user %s (%s)",
                        user_data["name"], user_data["id"])

        except Exception as e:
            logger.error("Error broadcasting user joined event: %s", str(e))

    async def broadcast_user_left(self, user_id: str):
        """Broadcast user left event to all clients."""
        try:
            message = {
                "type": "user_left",
                "user_id": user_id
            }
            await self.broadcast_to_all(message)
            logger.info("📡 Broadcasted user_left event for user %s", user_id)

        except Exception as e:
            logger.error("Error broadcasting user left event: %s", str(e))

    async def _broadcast_to_connections(self, connections: List[WebSocket], message: Dict[str, Any]):
        """Internal method to broadcast to a specific list of connections."""
        if not connections:
            return

        disconnected = []
        message_json = json.dumps(message)

        for websocket in connections:
            success = await self.connection_manager.send_to_connection(websocket, message)
            if not success:
                disconnected.append(websocket)

        # Remove disconnected clients
        for ws in disconnected:
            await self.connection_manager.disconnect(ws)

    async def _get_user_name(self, user_id: str, bot_manager=None):
        """Get user display name with fallback."""
        if bot_manager:
            try:
                return await bot_manager.get_user_display_name(user_id)
            except Exception as name_error:
                logger.debug("Could not get user name for %s: %s",
                             user_id, str(name_error))

        return f"User {user_id}"

    async def _broadcast_fallback(self, user_id: str, text: str, message_type: str):
        """Fallback broadcast with minimal user info."""
        try:
            fallback_message = {
                "type": message_type,
                "data": {
                    "user_id": user_id,
                    "user": f"User {user_id}",
                    "text": text
                }
            }
            await self.broadcast_to_all(fallback_message)
            logger.debug("Fallback broadcast successful for user %s", user_id)
        except Exception as fallback_error:
            logger.error("Fallback broadcast also failed: %s",
                         str(fallback_error))
