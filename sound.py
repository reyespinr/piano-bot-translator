"""
Sound device management and audio streaming.

This module provides functionality for handling audio devices and streaming audio
data from input devices to Discord voice channels. It includes tools for:
- Discovering available audio input devices
- Creating Discord-compatible audio streams
- Managing device connections and audio format conversion

The module uses sounddevice library for low-level audio device access and
integrates with Discord.py's audio subsystem.
"""
from pprint import pformat
import sounddevice as sd
import discord

DEFAULT = 0
sd.default.channels = 2
sd.default.dtype = "int16"
sd.default.latency = "low"
sd.default.samplerate = 48000


class PCMStream(discord.AudioSource):
    """Audio stream that captures from input devices and provides PCM data to Discord.

    This class implements Discord's AudioSource interface to capture audio from
    a system input device and convert it to the PCM format required by Discord.
    It handles device switching and audio stream management.
    """

    def __init__(self):
        """Initialize the PCM audio stream.

        Sets up the stream with default Discord audio parameters but doesn't
        start capturing until a device is selected.
        """
        discord.AudioSource.__init__(self)
        self.stream = None

        # Discord reads 20 ms worth of audio at a time (20 ms * 50 == 1000 ms == 1 sec)
        self.frames = int(sd.default.samplerate / 50)

    def read(self):
        """Read audio data from the input device.

        Called by Discord to get 20ms of audio data in PCM format.

        Returns:
            bytes: PCM audio data ready for Discord, or None if no stream is active
        """
        if self.stream is None:
            return None

        data = self.stream.read(self.frames)[0]

        # convert to pcm format
        return bytes(data)

    def change_device(self, num):
        """Change the audio input device.

        Closes any existing stream and opens a new one with the specified device.

        Args:
            num (int): The device index to use for input
        """
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()

        self.stream = sd.RawInputStream(device=num)
        self.stream.start()


class DeviceNotFoundError(Exception):
    """Exception raised when no suitable audio input devices are found.

    This exception includes detailed information about available devices and
    audio APIs to help diagnose device detection issues.
    """

    def __init__(self):
        """Initialize with system device information."""
        self.devices = sd.query_devices()
        self.host_apis = sd.query_hostapis()
        super().__init__("No Devices Found")

    def __str__(self):
        """Return formatted string representation of available devices."""
        return (
            f"Devices \n"
            f"{self.devices} \n "
            f"Host APIs \n"
            f"{pformat(self.host_apis)}"
        )


def query_devices():
    """Query available audio input devices.

    Identifies all input devices that can capture audio and are using
    the default host API.

    Returns:
        dict: Dictionary mapping device names to device indices

    Raises:
        DeviceNotFoundError: If no suitable input devices are found
    """
    options = {
        device.get("name"): index
        for index, device in enumerate(sd.query_devices())
        if (device.get("max_input_channels") > 0 and device.get("hostapi") == DEFAULT)
    }

    if not options:
        raise DeviceNotFoundError()

    return options
