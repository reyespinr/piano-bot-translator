"""
WebSocket connection management for Piano Bot Translator.

Handles individual WebSocket connections, connection lifecycle, and
basic communication with clients.
"""
import json
from typing import List, Dict, Any
from fastapi import WebSocket, WebSocketDisconnect
from logging_config import get_logger

logger = get_logger(__name__)


class WebSocketConnectionManager:
    """Manages individual WebSocket connections and their lifecycle."""

    def __init__(self):
        self.active_connections: List[WebSocket] = []
        # For future admin/spectator distinction
        self.connection_types: Dict[WebSocket, str] = {}

    async def connect(self, websocket: WebSocket, connection_type: str = "admin"):
        """Accept and register a new WebSocket connection."""
        await websocket.accept()
        self.active_connections.append(websocket)
        self.connection_types[websocket] = connection_type

        logger.info("🔌 New %s WebSocket connection (total: %d)",
                    connection_type, len(self.active_connections))

        return True

    async def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            connection_type = self.connection_types.pop(websocket, "unknown")
            logger.info("🔌 %s WebSocket disconnected (remaining: %d)",
                        connection_type, len(self.active_connections))

    async def send_to_connection(self, websocket: WebSocket, message: Dict[str, Any]) -> bool:
        """Send a message to a specific WebSocket connection.

        Returns:
            bool: True if message was sent successfully, False if connection failed
        """
        try:
            await websocket.send_text(json.dumps(message))
            return True
        except Exception as e:
            logger.debug("Failed to send message to client: %s", str(e))
            return False

    async def send_response(self, websocket: WebSocket, command: str, success: bool, message: str, **extra_data):
        """Send a command response to a specific connection."""
        response = {
            "type": "response",
            "command": command,
            "success": success,
            "message": message,
            **extra_data
        }
        return await self.send_to_connection(websocket, response)

    def get_connections_by_type(self, connection_type: str = None) -> List[WebSocket]:
        """Get connections filtered by type. If None, returns all connections."""
        if connection_type is None:
            return self.active_connections.copy()

        return [ws for ws in self.active_connections
                if self.connection_types.get(ws) == connection_type]

    def get_connection_count(self) -> int:
        """Get total number of active connections."""
        return len(self.active_connections)

    def is_connected(self, websocket: WebSocket) -> bool:
        """Check if a WebSocket is still connected."""
        return websocket in self.active_connections
