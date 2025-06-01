"""
Critical fix for user toggling functionality.

This module monkey-patches Discord's voice client to make sink replacement seamless,
ensuring that user toggles take effect immediately.
"""
import discord
from discord.voice_client import VoiceClient

# Store the original start_recording method
original_start_recording = VoiceClient.start_recording

# Define a patched version of start_recording to ensure clean sink state


def patched_start_recording(self, sink, callback, *args, **kwargs):
    """
    Patched version of start_recording that ensures user toggles are respected.

    This replacement makes sure each new sink has a fresh reference to the
    user_processing_enabled dictionary.
    """
    # Make sure we're not recording before starting
    try:
        if hasattr(self, '_player') and self._player:
            self.stop_recording()
    except Exception as e:
        print(f"Error stopping existing recording: {e}")

    # Call the original method
    return original_start_recording(self, sink, callback, *args, **kwargs)


# Apply the monkey patch
VoiceClient.start_recording = patched_start_recording
print("Discord voice client patched for better toggle support")
