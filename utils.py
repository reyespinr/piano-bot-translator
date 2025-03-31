"""
Audio transcription and translation utilities.

This module provides functions for transcribing audio to text using Whisper
and translating text between languages using DeepL's API. It includes a preloaded
Whisper model to improve performance across multiple transcription requests.
"""
import os
import wave
import numpy as np
import whisper
import requests


# Preload the Whisper model globally
print("Loading Whisper model...")
MODEL = whisper.load_model(
    "large-v3-turbo", device="cuda")  # Use "cuda" for GPU
# MODEL = whisper.load_model(
#     "base", device="cuda")  # Use "cuda" for GPU
print("Whisper model loaded successfully!")


async def transcribe(audio_file_path):
    """Transcribe speech from an audio file to text.

    Uses the preloaded Whisper model to convert speech in an audio file
    to text transcription.

    Args:
        audio_file_path (str): Path to the audio file to transcribe

    Returns:
        str: The transcribed text from the audio file
    """
    # Use the preloaded model for transcription
    result = MODEL.transcribe(audio_file_path, fp16=True)
    text = result["text"]
    return text


async def translate(text):
    """Translate text to English using DeepL's API.

    Sends the provided text to DeepL's translation API and returns
    the English translation.

    Args:
        text (str): The text to be translated

    Returns:
        str: The translated text in English
    """
    response = requests.post(
        "https://api-free.deepl.com/v2/translate",
        data={
            "auth_key": "5ac935ea-9ed2-40a7-bd4d-8153c941c79f:fx",
            "text": text,
            "target_lang": "EN"
        },
        timeout=10
    )
    translation = response.json()["translations"][0]["text"]
    return translation


def create_dummy_audio_file(filename="warmup_audio.wav"):
    """Create a small audio file for model warm-up.

    Args:
        filename (str): Name of the dummy audio file to create

    Returns:
        str: Path to the created audio file
    """
    # Create a 1-second file of silence (with a tiny bit of noise)
    # using the format Whisper expects (16kHz, 16-bit, mono)
    sample_rate = 16000
    duration = 1  # 1 second

    # Create an array of small random values (quiet noise)
    audio_data = np.random.normal(
        0, 0.01, sample_rate * duration).astype(np.int16)

    # Write to WAV file
    with wave.Wave_write(filename) as wf:
        wf.setnchannels(1)  # Mono
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data.tobytes())

    return filename


async def warm_up_pipeline():
    """Warm up the transcription pipeline.

    Runs a quick inference through the Whisper model to:
    - Pre-compile CUDA kernels
    - Allocate GPU memory
    - Initialize internal caches
    - Optimize execution paths

    This significantly reduces the delay for the first real transcription.
    """
    print("Warming up transcription pipeline...")
    try:
        # Create a dummy audio file
        dummy_file = create_dummy_audio_file()

        # Run through transcription
        await transcribe(dummy_file)

        # Clean up
        if os.path.exists(dummy_file):
            os.remove(dummy_file)

        print("Pipeline warm-up complete")
    except (IOError, FileNotFoundError, PermissionError, RuntimeError) as e:
        print(f"Warm-up error (non-critical): {e}")
