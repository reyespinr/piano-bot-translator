"""
WebSocket state management for Piano Bot Translator.

Manages user processing states and synchronizes with the bot manager
and other components.
"""
from typing import Dict
from logging_config import get_logger

logger = get_logger(__name__)


class WebSocketStateManager:
    """Manages WebSocket-related state and user processing settings."""

    def __init__(self):
        self.user_processing_enabled: Dict[str, bool] = {}

    def sync_with_bot_manager(self, bot_manager):
        """Sync user processing states with the bot manager."""
        self.user_processing_enabled = bot_manager.user_processing_enabled.copy()
        logger.debug("Synced user processing states with bot manager: %d users",
                     len(self.user_processing_enabled))

    async def update_user_processing(self, user_id: str, enabled: bool) -> bool:
        """Update user processing state across all components.

        Returns:
            bool: True if update was successful
        """
        try:
            # Update local state
            self.user_processing_enabled[user_id] = enabled

            # This will be called from the router which has access to bot_manager
            # We'll update the bot_manager from the router level

            logger.debug("Updated user %s processing state to %s",
                         user_id, enabled)
            return True

        except Exception as e:
            logger.error("Error updating user processing state: %s", str(e))
            return False

    def add_user(self, user_id: str, enabled: bool = True):
        """Add a new user to processing state."""
        self.user_processing_enabled[user_id] = enabled
        logger.info(
            "Added user %s to processing state (enabled: %s)", user_id, enabled)

    def remove_user(self, user_id: str):
        """Remove a user from processing state."""
        if user_id in self.user_processing_enabled:
            del self.user_processing_enabled[user_id]
            logger.info("🔇 Removed user %s from processing state", user_id)

    def is_user_enabled(self, user_id: str) -> bool:
        """Check if a user is enabled for processing."""
        return self.user_processing_enabled.get(user_id, True)

    def get_enabled_users(self) -> Dict[str, bool]:
        """Get a copy of all user processing states."""
        return self.user_processing_enabled.copy()

    def get_user_count(self) -> int:
        """Get total number of users being tracked."""
        return len(self.user_processing_enabled)

    def clear_all_users(self):
        """Clear all user processing states."""
        cleared_count = len(self.user_processing_enabled)
        self.user_processing_enabled.clear()
        logger.info(
            "Cleared all user processing states (%d users)", cleared_count)

    def update_bot_manager_and_translator(self, bot_manager):
        """Update bot manager and translator with current user processing states."""
        try:
            # Update bot manager
            # Update translator if available
            bot_manager.user_processing_enabled = self.user_processing_enabled.copy()
            if (hasattr(bot_manager, 'voice_translator') and
                    bot_manager.voice_translator):
                # Update translator's user settings (FIXED: Updated for refactored structure)
                bot_manager.voice_translator.state.user_processing_enabled = self.user_processing_enabled.copy()

                # Update sink if available - FIXED: Access through state
                if (hasattr(bot_manager.voice_translator.state, 'sink') and
                        bot_manager.voice_translator.state.sink):

                    sink = bot_manager.voice_translator.state.sink
                    if not hasattr(sink, 'parent') or not sink.parent:
                        sink.parent = type('obj', (object,), {})()
                    sink.parent.user_processing_enabled = self.user_processing_enabled.copy()
                    logger.debug("Updated sink parent with user processing states: %s",
                                 list(self.user_processing_enabled.keys()))

            logger.debug(
                "Updated bot manager and translator with user processing states")
            return True

        except Exception as e:
            logger.error(
                "Error updating bot manager and translator: %s", str(e))
            return False
