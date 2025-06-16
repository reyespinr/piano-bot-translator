"""
Simplified WebSocket manager for Piano Bot Translator.

Coordinates between specialized WebSocket components to provide
clean, modular real-time communication between frontend and Discord bot.
"""
from typing import Dict, Any
from fastapi import WebSocket, WebSocketDisconnect
from logging_config import get_logger
from websocket_connection_manager import WebSocketConnectionManager
from websocket_broadcaster import WebSocketBroadcaster
from websocket_message_router import WebSocketMessageRouter
from websocket_state_manager import WebSocketStateManager

logger = get_logger(__name__)


class WebSocketManager:
    """Simplified WebSocket manager using composition pattern."""

    def __init__(self, bot_manager):
        self.bot_manager = bot_manager

        # Initialize components using composition
        self.connection_manager = WebSocketConnectionManager()
        self.broadcaster = WebSocketBroadcaster(self.connection_manager)
        self.state_manager = WebSocketStateManager()
        self.message_router = WebSocketMessageRouter(
            bot_manager, self.connection_manager, self.broadcaster, self.state_manager
        )

    async def handle_connection(self, websocket: WebSocket, connection_type: str = "admin"):
        """Handle a new WebSocket connection with connection type support."""
        await self.connection_manager.connect(websocket, connection_type)

        try:
            # Send initial state for admin connections
            if connection_type == "admin":
                await self.message_router.send_initial_state(websocket)
            elif connection_type == "spectator":
                # For spectators, only send basic status (no controls)
                await self._send_spectator_initial_state(websocket)

            # Main message handling loop
            while True:
                data = await websocket.receive_text()
                # Only admin connections can send commands
                if connection_type == "admin":
                    await self.message_router.handle_message(websocket, data)
                else:
                    logger.debug("Ignoring command from spectator connection")

        except WebSocketDisconnect:
            await self.connection_manager.disconnect(websocket)
        except Exception as e:
            logger.error("WebSocket error: %s", str(e))
            await self.connection_manager.disconnect(websocket)

    async def handle_admin_connection(self, websocket: WebSocket):
        """Handle an admin WebSocket connection (full control)."""
        await self.handle_connection(websocket, "admin")

    async def handle_spectator_connection(self, websocket: WebSocket):
        """Handle a spectator WebSocket connection (read-only)."""
        await self.handle_connection(websocket, "spectator")

    # Delegation methods for external access
    async def broadcast_translation(self, user_id: str, text: str, message_type: str = "transcription"):
        """Broadcast transcription/translation to all connected clients."""
        await self.broadcaster.broadcast_translation(user_id, text, message_type, self.bot_manager)

    async def broadcast_bot_status(self, ready: bool):
        """Broadcast bot status to all clients."""
        await self.broadcaster.broadcast_bot_status(ready)

    async def broadcast_connection_status(self, status: str):
        """Broadcast connection status to all clients."""
        await self.broadcaster.broadcast_connection_status(status)

    async def broadcast_listen_status(self, is_listening: bool):
        """Broadcast listening status to all clients."""
        await self.broadcaster.broadcast_listen_status(is_listening)

    async def broadcast_user_joined(self, user_data: Dict[str, Any], enabled: bool):
        """Broadcast user joined event to all clients."""
        # Update state manager
        user_id = user_data["id"]
        self.state_manager.add_user(user_id, enabled)
        self.state_manager.update_bot_manager_and_translator(self.bot_manager)

        # Broadcast the event
        await self.broadcaster.broadcast_user_joined(user_data, enabled)

    async def broadcast_user_left(self, user_id: str):
        """Broadcast user left event to all clients."""
        # Clean up state manager
        self.state_manager.remove_user(user_id)
        self.state_manager.update_bot_manager_and_translator(self.bot_manager)

        # Broadcast the event
        await self.broadcaster.broadcast_user_left(user_id)

    def get_connection_count(self) -> int:
        """Get total number of active connections."""
        return self.connection_manager.get_connection_count()

    def get_admin_connection_count(self) -> int:
        """Get number of admin connections."""
        return len(self.connection_manager.get_connections_by_type("admin"))

    def get_spectator_connection_count(self) -> int:
        """Get number of spectator connections."""
        return len(self.connection_manager.get_connections_by_type("spectator"))

    async def _send_spectator_initial_state(self, websocket: WebSocket):
        """Send initial state for spectator connections (read-only info)."""
        try:
            # Only send basic bot status and current translations
            if (self.bot_manager.bot and self.bot_manager.bot.voice_clients and
                    len(self.bot_manager.bot.voice_clients) > 0):

                voice_client = self.bot_manager.bot.voice_clients[0]
                if voice_client.channel:
                    spectator_message = {
                        "type": "spectator_status",
                        "connected_channel": voice_client.channel.name,
                        "is_listening": getattr(self.bot_manager.voice_translator, 'is_listening', False),
                        "translations": []  # Empty initially, they'll receive live updates
                    }

                    await self.connection_manager.send_to_connection(websocket, spectator_message)
                    logger.debug(
                        "📡 Sent spectator status to new spectator client")

        except Exception as e:
            logger.error("Error sending spectator initial state: %s", str(e))

    # Properties for backward compatibility
    @property
    def user_processing_enabled(self) -> Dict[str, bool]:
        """Get user processing enabled states (for backward compatibility)."""
        return self.state_manager.get_enabled_users()

    @property
    def active_connections(self):
        """Get active connections (for backward compatibility)."""
        return self.connection_manager.active_connections
