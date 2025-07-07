"""
Audio processing utilities for Discord voice transcription_service.

This module provides audio file processing, validation, and creation utilities
for handling Discord voice data and preparing it for transcription_service.
"""
import asyncio
import gc
import os
import subprocess
import wave
import numpy as np
from logging_config import get_logger

logger = get_logger(__name__)

# File cleanup constants
MAX_DELETE_RETRIES = 3
CLEANUP_DELAY = 0.5

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


def is_recoverable_error(error_str):
    """Check if the error is a recoverable PyTorch/FFmpeg error."""
    # Non-recoverable errors that should not be retried
    non_recoverable_errors = [
        "cannot import name 'dtw_kernel' from 'whisper.triton_ops'",
        "ImportError",
        "ModuleNotFoundError",
        "No module named",
        "CUDA error: invalid argument",  # GPU hardware/driver issues
        "CUDA error: out of memory",     # GPU memory exhaustion
        "CUDA error: device-side assert triggered",  # GPU kernel failures
        "CUDA kernel errors might be asynchronously reported",  # General CUDA issues
        "CUDA_LAUNCH_BLOCKING",         # CUDA debugging related
        "TORCH_USE_CUDA_DSA"           # PyTorch CUDA debugging
    ]

    # Check for non-recoverable errors first
    if any(error in error_str for error in non_recoverable_errors):
        return False

    recoverable_errors = [
        "Expected key.size",
        "Invalid argument",
        "Error muxing",
        "Error submitting",
        "Error writing trailer",
        "Error closing file",
        "RuntimeError: NYI",  # TorchScript VAD errors
        "Error submitting a packet to the muxer",  # FFmpeg packet errors
        "vad_annotator.py",  # Any VAD-related errors
        "Error muxing a packet",  # Additional FFmpeg muxer errors
        "Task finished with error code",  # FFmpeg task errors
        "Terminating thread with return code",  # FFmpeg thread errors
        "out#0/s16le",  # FFmpeg output format errors
        "aost#0:0/pcm_s16le",  # FFmpeg audio stream errors
        "ffmpeg",  # General FFmpeg errors
        "av_",  # FFmpeg/libav function errors
        "codec",  # Audio codec errors
        "format"  # Audio format errors
    ]

    return (any(error in error_str for error in recoverable_errors) or
            "tensor" in error_str.lower() or
            "NYI" in error_str or
            "TorchScript" in error_str or
            "ffmpeg" in error_str.lower() or
            "libav" in error_str.lower())  # Catch all FFmpeg/audio errors


async def force_delete_file(file_path: str) -> bool:
    """Forcefully delete a file with multiple retries and GC."""
    if not file_path or not os.path.exists(file_path):
        return False

    # Try to delete the file with multiple retries
    for attempt in range(MAX_DELETE_RETRIES):
        try:
            # Force garbage collection to release file handles
            gc.collect()

            # Delete and verify
            os.remove(file_path)
            logger.debug("Deleted file: %s", os.path.basename(file_path))

            if not os.path.exists(file_path):
                return True

            logger.warning(
                "File still exists after deletion attempt: %s", file_path)
        except (PermissionError, OSError) as e:
            logger.warning("Deletion attempt %d failed: %s",
                           attempt + 1, str(e))
            await asyncio.sleep(CLEANUP_DELAY)  # Wait before retry

    # Last resort: try with Windows-specific commands
    if os.name == 'nt':
        try:
            # Use PowerShell syntax for better compatibility
            subprocess.run(
                f'Remove-Item -Path "{file_path}" -Force', shell=True, check=False)
            logger.debug(
                "Attempted deletion with Windows PowerShell command: %s", file_path)
            return not os.path.exists(file_path)
        except (OSError, subprocess.SubprocessError) as e:
            logger.error(
                "Windows PowerShell command deletion failed: %s", str(e))

    logger.warning(
        "Failed to delete file after multiple attempts: %s", file_path)
    return False


def safe_remove_file(file_path: str) -> bool:
    """Safely remove a file with error handling."""
    try:
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
            logger.debug("Removed file: %s", os.path.basename(file_path))
            return True
        return False
    except (OSError, PermissionError) as e:
        logger.warning("Failed to remove file %s: %s", file_path, str(e))
        return False


def check_cuda_health():
    """Check CUDA availability and health."""
    try:
        import torch
        if not torch.cuda.is_available():
            return False, "CUDA not available"

        # Try a simple CUDA operation
        device = torch.device('cuda')
        test_tensor = torch.zeros(1, device=device)
        _ = test_tensor + 1  # Simple operation to test CUDA

        return True, f"CUDA healthy on device {torch.cuda.current_device()}"
    except Exception as e:
        return False, f"CUDA health check failed: {str(e)}"


def clear_cuda_cache():
    """Clear CUDA cache to free up memory."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()  # Wait for operations to complete
            logger.info("🧹 CUDA cache cleared")
            return True
    except Exception as e:
        logger.warning("Failed to clear CUDA cache: %s", str(e))
    return False
