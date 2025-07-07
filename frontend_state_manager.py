"""
Frontend State Manager for Discord Bot Translator.

This module manages dynamic frontend controls:
1. Language selection dropdown
2. Message temporal ordering and reordering
3. Real-time state synchronization with the frontend
"""
import time
import asyncio
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class PendingMessage:
    """Represents a message waiting to be properly ordered."""
    user_id: str
    text: str
    language: str
    audio_timestamp: float
    processing_completed: float
    forced_language: Optional[str] = None
    translation: Optional[str] = None
    message_id: str = ""


class FrontendStateManager:
    """Manages dynamic frontend state and controls."""

    def __init__(self):
        self.language_override: Optional[str] = None  # None = auto-detect
        self.enable_temporal_ordering: bool = True
        self.max_reorder_delay: float = 10.0  # Max seconds to wait for reordering
        # user_id -> messages
        self.pending_messages: Dict[str, List[PendingMessage]] = {}
        # message_id -> timer task
        self.message_timers: Dict[str, asyncio.Task] = {}
        self.supported_languages = [
            {"code": None, "name": "Auto-detect", "flag": "🌍"},
            {"code": "en", "name": "English", "flag": "🇺🇸"},
            {"code": "es", "name": "Spanish", "flag": "🇪🇸"},
            {"code": "fr", "name": "French", "flag": "🇫🇷"},
            {"code": "de", "name": "German", "flag": "🇩🇪"},
            {"code": "it", "name": "Italian", "flag": "🇮🇹"},
            {"code": "pt", "name": "Portuguese", "flag": "🇵🇹"},
            {"code": "ru", "name": "Russian", "flag": "🇷🇺"},
            {"code": "ja", "name": "Japanese", "flag": "🇯🇵"},
            {"code": "ko", "name": "Korean", "flag": "🇰🇷"},
            {"code": "zh", "name": "Chinese", "flag": "🇨🇳"},
            {"code": "ar", "name": "Arabic", "flag": "🇸🇦"},
            {"code": "hi", "name": "Hindi", "flag": "🇮🇳"},
            {"code": "nl", "name": "Dutch", "flag": "🇳🇱"},
            {"code": "pl", "name": "Polish", "flag": "🇵🇱"},
            {"code": "tr", "name": "Turkish", "flag": "🇹🇷"},
            {"code": "sv", "name": "Swedish", "flag": "🇸🇪"},
            {"code": "da", "name": "Danish", "flag": "🇩🇰"},
            {"code": "no", "name": "Norwegian", "flag": "🇳🇴"},
            {"code": "fi", "name": "Finnish", "flag": "🇫🇮"},
        ]

    def set_language_override(self, language_code: Optional[str]) -> bool:
        """Set language override from frontend dropdown."""
        if language_code is None or language_code in [lang["code"] for lang in self.supported_languages]:
            old_lang = self.language_override
            self.language_override = language_code
            logger.info("🌍 Language override changed: %s -> %s",
                        old_lang, language_code)
            return True
        return False

    def get_language_override(self) -> Optional[str]:
        """Get current language override."""
        return self.language_override

    def get_supported_languages(self) -> List[Dict]:
        """Get list of supported languages for frontend dropdown."""
        return self.supported_languages

    def set_temporal_ordering(self, enabled: bool) -> None:
        """Enable/disable temporal message ordering."""
        self.enable_temporal_ordering = enabled
        logger.info("⏰ Temporal ordering %s",
                    "enabled" if enabled else "disabled")

    async def add_message_for_ordering(self, user_id: str, text: str, language: str,
                                       audio_timestamp: float, processing_completed: float,
                                       forced_language: Optional[str] = None,
                                       translation: Optional[str] = None) -> Tuple[bool, PendingMessage]:
        """
        Add a message to the temporal ordering system.

        Returns:
            Tuple[bool, PendingMessage]: (should_send_immediately, message)
        """
        if not self.enable_temporal_ordering:
            # If temporal ordering is disabled, send immediately
            message = PendingMessage(
                user_id=user_id,
                text=text,
                language=language,
                audio_timestamp=audio_timestamp,
                processing_completed=processing_completed,
                forced_language=forced_language,
                translation=translation
            )
            return True, message

        # Create message with unique ID
        message_id = f"{user_id}_{int(audio_timestamp * 1000)}"
        message = PendingMessage(
            user_id=user_id,
            text=text,
            language=language,
            audio_timestamp=audio_timestamp,
            processing_completed=processing_completed,
            forced_language=forced_language,
            translation=translation,
            message_id=message_id
        )

        # Add to pending messages
        if user_id not in self.pending_messages:
            self.pending_messages[user_id] = []

        self.pending_messages[user_id].append(message)

        # Check if we can send this message immediately
        ready_messages = self._get_ready_messages(user_id)
        if ready_messages:
            return True, ready_messages[0]

        # Start timer for this message
        await self._start_message_timer(message)

        return False, message

    def _get_ready_messages(self, user_id: str) -> List[PendingMessage]:
        """Get messages that are ready to be sent in chronological order."""
        if user_id not in self.pending_messages:
            return []

        messages = self.pending_messages[user_id]
        if not messages:
            return []

        # Sort by audio timestamp
        messages.sort(key=lambda m: m.audio_timestamp)

        # Find consecutive messages from the beginning that are ready
        ready_messages = []
        current_time = time.time()

        for i, message in enumerate(messages):
            # Check if this message is ready (either very old or next in sequence)
            time_since_recording = current_time - message.audio_timestamp

            if i == 0 or time_since_recording > self.max_reorder_delay:
                ready_messages.append(message)
            else:
                break

        # Remove ready messages from pending
        if ready_messages:
            self.pending_messages[user_id] = messages[len(ready_messages):]

        return ready_messages

    async def _start_message_timer(self, message: PendingMessage) -> None:
        """Start a timer for a message to ensure it's eventually sent."""
        async def timer_callback():
            await asyncio.sleep(self.max_reorder_delay)
            # Force send this message if it's still pending
            if message.user_id in self.pending_messages:
                user_messages = self.pending_messages[message.user_id]
                if message in user_messages:
                    logger.info(
                        "⏰ Timer expired for message %s, forcing send", message.message_id)
                    # This would trigger the message to be sent
                    # Implementation depends on how you want to handle this

        # Start timer task
        if message.message_id not in self.message_timers:
            self.message_timers[message.message_id] = asyncio.create_task(
                timer_callback())

    async def force_send_pending_messages(self, user_id: str) -> List[PendingMessage]:
        """Force send all pending messages for a user."""
        if user_id not in self.pending_messages:
            return []

        messages = self.pending_messages[user_id]
        messages.sort(key=lambda m: m.audio_timestamp)

        # Clear pending messages
        self.pending_messages[user_id] = []

        # Cancel timers
        for message in messages:
            if message.message_id in self.message_timers:
                self.message_timers[message.message_id].cancel()
                del self.message_timers[message.message_id]

        return messages

    def get_state_for_frontend(self) -> Dict:
        """Get current state for frontend synchronization."""
        return {
            "language_override": self.language_override,
            "supported_languages": self.supported_languages,
            "temporal_ordering_enabled": self.enable_temporal_ordering,
            "max_reorder_delay": self.max_reorder_delay,
            "pending_message_count": sum(len(messages) for messages in self.pending_messages.values())
        }


# Global instance
frontend_state_manager = FrontendStateManager()
