"""
Voice Activity Detection using multiple methods.
Combines Discord's built-in VAD with energy-based fallback.
"""
import numpy as np
from typing import Optional
from logging_config import get_logger

logger = get_logger(__name__)


class VoiceActivityDetector:
    """Handles voice activity detection using multiple methods."""

    def __init__(self, energy_threshold: float = 250):
        self.energy_threshold = energy_threshold

    def is_speech_active(self, audio_data, user, user_state, voice_client=None) -> bool:
        """Combined VAD check using Discord VAD and energy-based fallback.

        This provides the most reliable voice activity detection by:
        1. First checking Discord's built-in VAD (most accurate)
        2. Falling back to energy-based VAD if Discord VAD unavailable
        3. Applying additional logic for silence detection
        """
        try:
            # Method 1: Try Discord's built-in VAD first (most reliable)
            discord_speaking = self._check_discord_vad(user, voice_client)

            # Method 2: Energy-based VAD as fallback
            energy_active = self._check_energy_vad(audio_data, user_state)

            # Method 3: Check for silence frames
            is_silence_frame = self._is_silence_frame(audio_data)

            # DEBUG: Log VAD decisions for troubleshooting (only when there's activity)
            if energy_active or discord_speaking:
                logger.debug("VAD check for user %s: discord=%s, energy=%s, silence=%s",
                             user, discord_speaking, energy_active, is_silence_frame)

            # Combine the signals for best accuracy
            if discord_speaking is not None:
                # If Discord says they're speaking, trust it
                if discord_speaking:
                    return True

                # FIXED: Be more conservative when Discord says NOT speaking
                # Only override Discord's decision if energy is VERY high and it's clearly not silence
                elif energy_active and not is_silence_frame:
                    # Calculate actual energy to be more selective
                    try:
                        audio_array = np.frombuffer(audio_data, dtype=np.int16)
                        if len(audio_array) > 0:
                            current_energy = np.mean(np.abs(audio_array))
                            # Only override if energy is significantly higher than normal speech threshold
                            if current_energy > 800:  # Much higher threshold than normal 250
                                logger.debug(
                                    "Overriding Discord VAD: very high energy %d detected", current_energy)
                                return True
                    except Exception:
                        pass

                # Trust Discord when it says user is not speaking
                return False
            else:
                # Fallback to energy-based VAD if Discord VAD unavailable
                return energy_active and not is_silence_frame

        except Exception as e:
            logger.warning(
                "Error in combined VAD check for user %s: %s", user, e)
            # Final fallback to simple energy check
            return self._check_energy_vad(audio_data, user_state)

    def _check_discord_vad(self, user, voice_client) -> Optional[bool]:
        """Check Discord's built-in VAD.

        This uses Discord's own voice activity detection which is more accurate
        than energy-based detection as it accounts for things like:
        - Push-to-talk state
        - Voice activation detection
        - Mute/deafen states
        - Network conditions
        """
        try:
            # Access voice client through parent (VoiceTranslator instance)
            if (voice_client and
                hasattr(voice_client, 'ws') and voice_client.ws and
                    hasattr(voice_client.ws, 'ssrc_map')):

                ssrc_map = voice_client.ws.ssrc_map

                # Find the SSRC for this user
                for info in ssrc_map.values():
                    if info.get("user_id") == user:
                        speaking_state = info.get("speaking", False)
                        logger.debug(
                            "Discord VAD for user %s: speaking=%s", user, speaking_state)
                        return speaking_state

                logger.debug("User %s not found in SSRC map", user)
                return None

        except Exception as e:
            logger.warning(
                "Error checking Discord VAD for user %s: %s", user, e)
            return None

        return None

    def _check_energy_vad(self, audio_data, user_state) -> bool:
        """Energy-based voice activity detection."""
        try:
            # PROTECTION: Validate audio data before processing
            if not audio_data or len(audio_data) < 2:
                return False

            # Convert bytes to numpy array (assuming PCM signed 16-bit little-endian)
            audio_array = np.frombuffer(audio_data, dtype=np.int16)

            # PROTECTION: Check for empty array
            if len(audio_array) == 0:
                return False

            # Calculate energy of the current frame
            energy = np.mean(np.abs(audio_array))

            # Initialize energy history if needed
            if not hasattr(user_state, 'energy_history') or not user_state.energy_history:
                user_state.energy_history = []

            # Update energy history
            user_state.energy_history.append(energy)
            # Keep last 5 frames
            user_state.energy_history = user_state.energy_history[-5:]

            # Calculate average energy over recent frames
            avg_energy = np.mean(user_state.energy_history)

            # IMPROVED: Slightly lower threshold for better sensitivity
            return avg_energy > self.energy_threshold

        except (ValueError, TypeError, OverflowError, MemoryError) as e:
            logger.warning("Audio activity detection error: %s", str(e))
            # Default to inactive for corrupted audio
            return False

    def _is_silence_frame(self, audio_data):
        """Check if audio frame is a silence frame sent by Discord."""
        try:
            # Discord sends specific silence frames: b"\xf8\xff\xfe"
            if len(audio_data) >= 3 and audio_data[:3] == b"\xf8\xff\xfe":
                return True

            # Also check for very low energy (near silence)
            if len(audio_data) >= 2:
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
                if len(audio_array) > 0:
                    energy = np.mean(np.abs(audio_array))
                    return energy < 50  # Very low energy threshold for silence

        except Exception as e:
            logger.debug("Error checking silence frame: %s", e)

        return False
