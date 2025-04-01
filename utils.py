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
import string


# Preload the Whisper model globally
print("Loading Whisper model...")
MODEL = whisper.load_model(
    "large-v3-turbo", device="cuda")  # Use "cuda" for GPU
# MODEL = whisper.load_model(
#     "base", device="cuda")  # Use "cuda" for GPU
print("Whisper model loaded successfully!")


async def transcribe(audio_file_path):
    """Transcribe speech from an audio file to text with confidence filtering."""
    # Use the preloaded model for transcription
    result = MODEL.transcribe(audio_file_path, fp16=True)

    # Extract the detected language
    detected_language = result.get("language", "")

    # Check confidence level
    if result["segments"]:
        # Get average log probability across all segments
        avg_log_prob = sum(s["avg_logprob"]
                           for s in result["segments"]) / len(result["segments"])

        # Filter out low-confidence outputs
        confidence_threshold = -1.5  # Adjust based on testing

        if avg_log_prob < confidence_threshold:
            print(
                f"Low confidence transcription rejected ({avg_log_prob:.2f}): '{result['text']}'")
            return "", detected_language  # Return empty text but still include language

        # Special case for common hallucinations with short audio
        text = result["text"].strip().lower()

        # Remove punctuation for comparison
        text_clean = text.translate(str.maketrans('', '', string.punctuation))

        if len(text_clean) < 15 and (
                text_clean == "thank you" or
                text_clean == "thanks" or
                text_clean == "thank" or
                text_clean == "um" or
                text_clean == "hmm"):
            # For these common short responses, require higher confidence
            stricter_threshold = -0.5
            if avg_log_prob < stricter_threshold:
                print(
                    f"Short statement '{text}' rejected with confidence {avg_log_prob:.2f}")
                return "", detected_language

        print(
            f"Transcription confidence: {avg_log_prob:.2f}, Language: {detected_language}")

    # Return both the text and detected language
    return result["text"], detected_language


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


async def should_translate(text, detected_language):
    """Determine if text needs translation based on language and content."""
    # Always skip empty text
    if not text:
        return False

    # Skip if dominant language is English
    if detected_language == "en":
        # Check if there are likely non-English segments
        # This simple heuristic checks for characters common in non-Latin alphabets
        non_ascii_ratio = len([c for c in text if ord(c) > 127]) / len(text)
        if non_ascii_ratio > 0.1:  # If more than 10% non-ASCII, translate anyway
            print(
                f"Detected mixed language content (non-ASCII ratio: {non_ascii_ratio:.2f})")
            return True
        return False

    # Non-English dominant language should be translated
    return True


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
