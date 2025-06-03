"""
Audio transcription and translation utilities.

This module provides functions for transcribing audio to text using stable-ts
(an enhanced version of Whisper) and translating text between languages using 
DeepL's API. It includes a preloaded model to improve performance across 
multiple transcription requests.

Features:
- Voice Activity Detection (VAD) to filter non-speech audio
- Confidence-based transcription filtering
- Common hallucination detection and prevention
- Automatic language detection
- Translation optimization to preserve API quota
- Pipeline warm-up for improved first-inference performance
- Multi-worker compatible model sharing
"""
import os
import wave
import string
import numpy as np
import requests
import stable_whisper
from logging_config import get_logger

# Initialize logger for this module
logger = get_logger(__name__)

# Model reference (will be loaded on demand, not at import time)
MODEL = None
MODEL_NAME = "large-v3-turbo"
# MODEL_NAME = "base"


# Common hallucinations to filter out
COMMON_HALLUCINATIONS = {
    "thank you", "thanks", "thank", "um", "hmm"
}


def _load_model_if_needed():
    """Lazy load the model only when needed - truly on-demand loading"""
    global MODEL
    if MODEL is None:
        logger.info("Loading stable-ts %s model...", MODEL_NAME)
        MODEL = stable_whisper.load_model(MODEL_NAME, device="cuda")
        logger.info("stable-ts model loaded successfully!")
    return MODEL


async def transcribe(audio_file_path):
    """Transcribe speech from an audio file to text with confidence filtering.

    This function processes audio through the stable-ts model with VAD (Voice
    Activity Detection) and applies multiple filtering layers to ensure quality:
    1. VAD to filter out non-speech audio
    2. Focus on human speech frequencies
    3. Additional confidence threshold filtering
    4. Language detection to determine if translation is needed

    Args:
        audio_file_path (str): Path to the audio file to transcribe

    Returns:
        tuple: (transcribed_text, detected_language)
            - transcribed_text (str): The transcribed text or empty if filtered
            - detected_language (str): ISO language code detected by Whisper
    """
    # Make sure model is loaded (true lazy loading)
    model = _load_model_if_needed()

    # Use the model for transcription with enhanced settings
    result = model.transcribe(
        audio_file_path,
        vad=True,                  # Enable Voice Activity Detection
        vad_threshold=0.35,        # VAD confidence threshold
        no_speech_threshold=0.6,   # Filter non-speech sections
        max_instant_words=0.3,     # Reduce hallucination words
        suppress_silence=True,     # Use silence detection for better timestamps
        only_voice_freq=True,      # Focus on human voice frequency range
        word_timestamps=True       # Important for proper segmentation
    )

    # Extract text and detected language
    transcribed_text = result.text if result.text else ""
    detected_language = result.language if result.language else ""

    # Handle Austrian German misidentified as Icelandic
    if detected_language == "is":  # "is" is the language code for Icelandic
        print("Detected Icelandic - likely Austrian German. Re-transcribing as German...")
        # Re-transcribe with German as forced language
        result = model.transcribe(
            audio_file_path,
            vad=True,
            vad_threshold=0.35,
            no_speech_threshold=0.6,
            max_instant_words=0.3,
            suppress_silence=True,
            only_voice_freq=True,
            word_timestamps=True,
            language="de"  # Force German language
        )
        transcribed_text = result.text if result.text else ""
        detected_language = "de"  # Override detected language to German
        print("Re-transcribed as German")

    # Apply additional confidence filtering as a safety net
    if hasattr(result, "segments") and result.segments:
        # Get confidence values
        confidences = []
        for segment in result.segments:
            if hasattr(segment, "avg_logprob"):
                confidences.append(segment.avg_logprob)

        # Apply our confidence thresholds if we have confidence data
        if confidences:
            avg_log_prob = sum(confidences) / len(confidences)

            # General confidence threshold
            confidence_threshold = -1.5
            if avg_log_prob < confidence_threshold:
                print(
                    f"Low confidence transcription rejected"
                    f"({avg_log_prob:.2f}): '{transcribed_text}'")
                return "", detected_language

            # Special case for common hallucinations
            text = transcribed_text.strip().lower()
            text_clean = text.translate(
                str.maketrans('', '', string.punctuation))

            if len(text_clean) < 15 and text_clean in COMMON_HALLUCINATIONS:
                # For these common short responses, require higher confidence
                stricter_threshold = -0.5
                if avg_log_prob < stricter_threshold:
                    print(
                        f"Short statement '{text}' rejected with confidence {avg_log_prob:.2f}")
                    return "", detected_language

            print(
                f"Transcription confidence: {avg_log_prob:.2f}, Language: {detected_language}")

    # Return both the text and detected language
    return transcribed_text, detected_language


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
    """Determine if text needs translation based on language and content.

    This function optimizes translation API usage by avoiding unnecessary
    translations of English text, while still detecting mixed-language content
    that might need translation.

    Args:
        text (str): The text to potentially translate
        detected_language (str): Language code detected by Whisper

    Returns:
        bool: True if text should be translated, False otherwise
    """
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

    Runs a quick inference through the model to:
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
