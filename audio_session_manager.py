"""
Manages conversation session state and user tracking.
"""
import time
from dataclasses import dataclass, field
from typing import Set
from logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class AudioSession:
    """Tracks conversation session state."""
    start_time: float = field(default_factory=time.time)
    state: str = "new"  # "new", "active", or "established"
    last_speaker_change: float = field(default_factory=time.time)
    current_speakers: Set = field(default_factory=set)
    last_activity_time: float = field(default_factory=time.time)

    def update_speaker(self, user):
        """Update session with new speaker activity."""
        current_time = time.time()
        if user not in self.current_speakers:
            self.current_speakers.add(user)
            self.last_speaker_change = current_time
            logger.debug("New speaker added to session: %s", user)

        self.last_activity_time = current_time

        # Update session state based on activity
        session_duration = current_time - self.start_time
        old_state = self.state

        if session_duration > 30:  # 30 seconds
            self.state = "established"
        elif len(self.current_speakers) > 1:
            self.state = "active"

        if old_state != self.state:
            logger.debug("Session state changed from %s to %s",
                         old_state, self.state)

    def get_minimum_duration(self):
        """Get minimum audio duration based on session state."""
        if self.state == "new":
            return 0.3  # 300ms for new sessions
        elif self.state == "active":
            return 0.2  # 200ms for active sessions
        else:  # established
            return 0.15  # 150ms for established sessions

    def is_established(self) -> bool:
        """Check if session is in established state."""
        return self.state == "established"

    def get_speaker_count(self) -> int:
        """Get the number of unique speakers in this session."""
        return len(self.current_speakers)

    def get_session_duration(self) -> float:
        """Get total session duration in seconds."""
        return time.time() - self.start_time
