"""
Audio processing utilities for Discord voice transcription.

This module provides audio file processing, validation, and creation utilities
for handling Discord voice data and preparing it for transcription.
"""
import wave
import numpy as np
from logging_config import get_logger

logger = get_logger(__name__)

# CRITICAL RESTORATION: Move common hallucinations here for shared access
# COMMON_HALLUCINATIONS = {
#     "thank you", "thanks", "thank", "um", "hmm", "okay", "ok", "yes",
#     "yeah", "no", "nope", "mm-hmm", "uh-huh", "uh", "ah", "oh", "wow",
#     "nice", "good", "great", "cool", "awesome", "amazing", "perfect",
#     "exactly", "right", "correct", "sure", "absolutely", "definitely",
#     "maybe", "i think", "i guess", "i know", "i see", "i understand",
#     "got it", "makes sense", "sounds good", "sounds great", "sounds cool"
# }
COMMON_HALLUCINATIONS = {
    "thank you", "thanks", "thank", "um", "hmm"
}


def get_audio_duration_from_file(audio_file_path):
    """Calculate audio duration from WAV file"""
    try:
        with wave.open(audio_file_path, 'rb') as wav_file:
            frames = wav_file.getnframes()
            sample_rate = wav_file.getframerate()
            duration = frames / sample_rate
            return duration
    except (wave.Error, IOError) as e:
        logger.warning("Could not determine audio duration: %s", e)
        return 0.0


def create_dummy_audio_file(filename="warmup_audio.wav"):
    """Create a small audio file for model warm-up.

    Args:
        filename (str): Name of the dummy audio file to create

    Returns:
        str: Path to the created audio file
    """
    # Create a 2-second file with actual speech-like noise for better warmup
    # using the format Discord expects (48kHz, 16-bit, stereo)
    sample_rate = 48000  # Changed to match Discord format
    duration = 2  # Increased to 2 seconds
    channels = 2  # Stereo for Discord compatibility

    # Create an array with speech-like noise patterns instead of pure random
    num_samples = sample_rate * duration

    # Generate a simple sine wave with noise to simulate speech
    t = np.linspace(0, duration, num_samples, False)
    # Mix of frequencies that resemble human speech
    signal = (np.sin(2 * np.pi * 440 * t) * 0.3 +  # A4 note
              np.sin(2 * np.pi * 880 * t) * 0.2 +  # A5 note
              np.random.normal(0, 0.1, num_samples))  # Background noise

    # Normalize and convert to 16-bit
    signal = np.clip(signal, -1, 1)
    audio_data = (signal * 32767).astype(np.int16)

    # Convert mono to stereo by duplicating the channel
    if channels == 2:
        audio_data = np.column_stack((audio_data, audio_data))

    # Write to WAV file
    with wave.Wave_write(filename) as wf:
        wf.setnchannels(1)  # Mono
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data.tobytes())

    return filename
