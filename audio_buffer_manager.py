"""
Manages audio buffering and file creation for users.
"""
import io
import os
import wave
from dataclasses import dataclass, field
from typing import Dict
import numpy as np
from logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class UserAudioState:
    """Per-user audio processing state."""
    last_packet_time: float = 0
    last_active_time: float = 0
    is_speaking: bool = False
    silence_frames: int = 0
    speech_detected: bool = False
    last_processed_time: float = 0
    speech_buffer: io.BytesIO = field(default_factory=io.BytesIO)
    pre_speech_buffer: list = field(default_factory=list)
    energy_history: list = field(default_factory=list)
    last_inactive_check: float = 0  # For rate limiting


class AudioBufferManager:
    """Manages audio buffers for all users."""

    def __init__(self):
        self.users: Dict[str, UserAudioState] = {}

    def get_user_state(self, user: str) -> UserAudioState:
        """Get or create user state."""
        if user not in self.users:
            self.users[user] = UserAudioState()
        return self.users[user]

    def update_pre_speech_buffer(self, user: str, data: bytes):
        """Update the pre-speech buffer for smoother speech beginning."""
        user_state = self.get_user_state(user)

        # CRITICAL FIX: Only keep pre-speech buffer if recent activity detected
        # Check if this frame has any energy to avoid accumulating silence
        try:
            # Quick energy check on current frame
            audio_array = np.frombuffer(data, dtype=np.int16)
            frame_energy = np.mean(np.abs(audio_array)) if len(
                audio_array) > 0 else 0

            # Only add to pre-speech buffer if there's some energy OR we're already speaking
            if frame_energy > 100 or user_state.is_speaking:
                user_state.pre_speech_buffer.append(data)
                if len(user_state.pre_speech_buffer) > 5:  # Reduced from 10 to 5 frames (~100ms)
                    user_state.pre_speech_buffer.pop(0)
            else:
                # Clear pre-speech buffer during silence to prevent accumulation
                if len(user_state.pre_speech_buffer) > 0:
                    user_state.pre_speech_buffer = []

        except (ValueError, TypeError) as e:
            # Fallback: just maintain buffer size
            logger.warning(
                "Error processing audio data for user %s: %s", user, str(e))
            user_state.pre_speech_buffer.append(data)
            if len(user_state.pre_speech_buffer) > 5:
                user_state.pre_speech_buffer.pop(0)

    def handle_active_speech(self, user: str, data: bytes):
        """Handle incoming active speech data."""
        user_state = self.get_user_state(user)

        # Reset silence counter when speech is detected
        user_state.silence_frames = 0
        # Mark that speech was detected in this session
        user_state.speech_detected = True

        # If this is the start of speech, reset buffer and add pre-speech buffer
        if not user_state.is_speaking:
            logger.debug("Speech started for user %s", user)
            user_state.is_speaking = True

            # CRITICAL FIX: Reset speech buffer to prevent accumulation of previous silence
            user_state.speech_buffer = io.BytesIO()

            # Add pre-speech frames for smoother beginning, but limit to avoid silence accumulation
            pre_speech_count = len(user_state.pre_speech_buffer)
            if pre_speech_count > 0:
                logger.debug(
                    "Adding %d pre-speech frames for user %s", pre_speech_count, user)
                for pre_data in user_state.pre_speech_buffer:
                    user_state.speech_buffer.write(pre_data)

        # Add this audio data to the speech buffer
        try:
            user_state.speech_buffer.write(data)
        except (IOError, OSError) as e:
            logger.warning(
                "Failed to write speech data for user %s: %s", user, str(e))

    def handle_silence(self, user: str, data: bytes, silence_threshold: int = 10):
        """Handle silence after speech."""
        user_state = self.get_user_state(user)

        # Add a few frames to the speech buffer for smoother transitions
        if user_state.is_speaking and user_state.silence_frames < 5:
            try:
                user_state.speech_buffer.write(data)
            except (IOError, OSError) as e:
                logger.warning(
                    "Failed to write silence data for user %s: %s", user, str(e))

        # Increment silence counter
        # CRITICAL FIX: Only process speech once when threshold is first exceeded
        user_state.silence_frames += 1

        # Return True if speech should be processed
        return (user_state.is_speaking and
                user_state.silence_frames == silence_threshold + 1 and  # Only trigger once
                user_state.speech_detected)  # Only if speech was actually detected

    def reset_speech_state(self, user: str):
        """Reset speech state for a user."""
        user_state = self.get_user_state(user)
        user_state.is_speaking = False
        user_state.speech_detected = False
        user_state.silence_frames = 0
        user_state.speech_buffer = io.BytesIO()

    def process_long_pause(self, user: str):
        """Handle a long pause in audio packets."""
        logger.debug(
            "Long pause detected for user %s. Processing any speech and resetting.", user)
        user_state = self.get_user_state(user)

        # Check if we should process accumulated speech
        should_process = user_state.is_speaking and user_state.speech_detected

        # Reset state
        user_state.is_speaking = False
        user_state.silence_frames = 0
        user_state.speech_detected = False
        user_state.speech_buffer = io.BytesIO()
        user_state.pre_speech_buffer = []

        return should_process

    def get_audio_duration(self, user: str) -> float:
        """Calculate audio duration from buffer size."""
        user_state = self.get_user_state(user)
        speech_buffer = user_state.speech_buffer

        if not speech_buffer:
            return 0.0

        # Get buffer size
        speech_buffer.seek(0, os.SEEK_END)
        buffer_size = speech_buffer.tell()

        # Discord audio is 48kHz, 16-bit, stereo = 192,000 bytes per second
        bytes_per_second = 48000 * 2 * 2  # sample_rate * channels * bytes_per_sample
        duration = buffer_size / bytes_per_second

        return duration

    def create_audio_file(self, user: str, current_time: float) -> tuple[str, bool]:
        """Create WAV file from user's speech buffer.

        Returns:
            tuple: (filename, success) where success indicates if file was created successfully
        """
        user_state = self.get_user_state(user)

        try:
            # Create unique filename
            audio_filename = f"{user}_{int(current_time * 1000)}_speech.wav"

            # Write audio buffer to file
            user_state.speech_buffer.seek(0)
            audio_data = user_state.speech_buffer.read()

            # CRITICAL FIX: Reset buffer immediately after reading to prevent accumulation
            user_state.speech_buffer = io.BytesIO()

            with wave.open(audio_filename, 'wb') as wav_file:
                wav_file.setnchannels(2)  # Discord uses stereo
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(48000)  # Discord sample rate
                wav_file.writeframes(audio_data)

            # Validate audio file before returning
            is_valid, validation_msg = self._validate_audio_file(
                audio_filename)
            if not is_valid:
                logger.warning(
                    "Audio validation failed for user %s: %s", user, validation_msg)
                try:
                    os.remove(audio_filename)
                except Exception:
                    pass
                return audio_filename, False

            return audio_filename, True

        except Exception as e:
            logger.error(
                "Error creating audio file for user %s: %s", user, str(e))
            return "", False

    def _validate_audio_file(self, audio_file_path: str) -> tuple[bool, str]:
        """Validate audio file before processing to detect corruption."""
        try:
            if not os.path.exists(audio_file_path):
                return False, "File does not exist"

            # Check file size (minimum 10KB for meaningful audio)
            file_size = os.path.getsize(audio_file_path)
            if file_size < 10000:
                return False, f"File too small: {file_size} bytes"

            # Try to open with wave to validate format
            with wave.open(audio_file_path, 'rb') as wav_file:
                frames = wav_file.getnframes()
                framerate = wav_file.getframerate()
                duration = frames / framerate if framerate > 0 else 0

                if duration < 0.1:  # Less than 100ms
                    return False, f"Audio too short: {duration:.2f}s"

                if duration > 300.0:  # More than 5 minutes (300 seconds)
                    return False, f"Audio too long: {duration:.2f}s"

            return True, "Valid"

        except (wave.Error, OSError, IOError) as e:
            return False, f"Wave format error: {str(e)}"
        except Exception as e:
            return False, f"Validation error: {str(e)}"

    def should_process_buffer(self, user: str, min_duration: float, min_buffer_size: int = 25000) -> bool:
        """Check if user's buffer should be processed based on duration and size."""
        user_state = self.get_user_state(user)

        # Get audio duration and check if it meets minimum requirements
        duration_seconds = self.get_audio_duration(user)

        # Also check minimum buffer size (Discord audio is 192KB/sec, so minimum ~38KB for 0.2s)
        user_state.speech_buffer.seek(0, os.SEEK_END)
        buffer_size = user_state.speech_buffer.tell()

        if duration_seconds < min_duration or buffer_size < min_buffer_size:
            logger.debug("Speech too short (%.2fs < %.2fs) or buffer too small (%d < %d bytes), skipping.",
                         duration_seconds, min_duration, buffer_size, min_buffer_size)
            return False

        return True

    def check_cooldown(self, user: str, current_time: float, cooldown_time: float = 1.5) -> bool:
        """Check if user is in cooldown period."""
        user_state = self.get_user_state(user)

        if user_state.last_processed_time > 0:
            time_since_last = current_time - user_state.last_processed_time
            if time_since_last < cooldown_time:
                logger.debug("Cooldown active for user %s, skipping.", user)
                return True
        return False

    def update_processed_time(self, user: str, current_time: float):
        """Update the last processed time for a user."""
        user_state = self.get_user_state(user)
        user_state.last_processed_time = current_time

    def check_inactive_speakers(self, current_time: float, inactive_threshold: float = 3.0, rate_limit: float = 2.0) -> list:
        """Check for users who have stopped speaking but haven't been processed.

        Returns:
            list: List of users who should be processed for inactivity
        """
        users_to_process = []

        for user, user_state in list(self.users.items()):
            try:
                # CRITICAL FIX: Only process inactive speakers once per user
                if (user_state.is_speaking and
                    user_state.speech_detected and
                        current_time - user_state.last_active_time > inactive_threshold):

                    # Add rate limiting per user to prevent repeated processing
                    if current_time - user_state.last_inactive_check > rate_limit:
                        logger.debug(
                            "Timer detected inactive speaker %s. Should process speech.", user)
                        users_to_process.append(user)
                        user_state.last_inactive_check = current_time

            except Exception as user_error:
                logger.error("Error processing inactive user %s: %s",
                             user, str(user_error))

        return users_to_process
