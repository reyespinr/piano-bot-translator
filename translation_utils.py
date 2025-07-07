"""
Audio transcription and translation utilities.

This module provides the main interface for transcription and translation
functionality, along with DeepL translation services and optimization logic.
"""
import requests
from logging_config import get_logger
from config_manager import get_config

# Import the faster-whisper model manager
from model_manager import faster_whisper_model_manager
import transcription_service

logger = get_logger(__name__)

# Translation functions (moved from translation.py)


async def translate(text):
    """Translate text to English using DeepL's API.

    Sends the provided text to DeepL's translation API and returns
    the English translation. Configuration is loaded from config.yaml.

    Args:
        text (str): The text to be translated

    Returns:
        str: The translated text in English
    """
    config = get_config()
    translation_config = config.translation

    api_key = translation_config.deepl_api_key
    api_url = translation_config.deepl_api_url
    target_lang = translation_config.target_language
    timeout = translation_config.timeout

    if not api_key:
        logger.error("DeepL API key not configured in config.yaml")
        raise ValueError("DeepL API key not configured")

    response = requests.post(
        api_url,
        data={
            "auth_key": api_key,
            "text": text,
            "target_lang": target_lang
        },
        timeout=timeout
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
            logger.info(
                "Detected mixed language content (non-ASCII ratio: %.2f - translating)",
                non_ascii_ratio)
            return True
        return False

    # Non-English dominant language should be translated
    return True

# Transcription utility functions


async def transcribe(audio_file, current_queue_size=0, concurrent_requests=0, active_transcriptions=0,
                     force_language=None, audio_timestamp=None):
    """
    Backward-compatible transcription function that returns only 2 values.

    This wrapper maintains API compatibility with the old system while using faster-whisper.
    Now supports dynamic language selection and temporal alignment.
    """
    transcribed_text, detected_language, result = await transcription_service.transcribe(
        audio_file, current_queue_size, concurrent_requests, active_transcriptions,
        force_language, audio_timestamp
    )
    return transcribed_text, detected_language

# Enhanced transcription function that returns full result for temporal alignment


async def transcribe_with_timestamp(audio_file, current_queue_size=0, concurrent_requests=0,
                                    active_transcriptions=0, force_language=None, audio_timestamp=None):
    """
    Enhanced transcription function that returns full result with temporal information.

    This is used when temporal alignment is needed for message reordering.
    """
    return await transcription_service.transcribe(
        audio_file, current_queue_size, concurrent_requests, active_transcriptions,
        force_language, audio_timestamp
    )


async def warm_up_pipeline():
    """
    Warm up transcription models using the faster-whisper model manager.
    This is the new, organized way to warm up models.
    """
    logger.info("🔥 Starting faster-whisper model warm-up pipeline...")

    try:
        # Use the faster-whisper model manager's warm-up functionality
        success = await faster_whisper_model_manager.warm_up_models()

        if success:
            logger.info(
                "🎯 Enhanced dual model pipeline warm-up complete! Ready for smart routing.")
            stats = faster_whisper_model_manager.get_stats()
            logger.info("📈 Accurate models: %d, Fast models: %d",
                        stats["accurate_models"], stats["fast_models"])
            logger.info("🚀 Total VRAM models loaded: %d models ready for parallel processing",
                        stats["accurate_models"] + stats["fast_models"])
            return True

        logger.warning(
            "⚠️ Model warm-up had some issues but models should still work")
        return False

    except Exception as e:
        logger.error("❌ Model warm-up failed: %s", str(e))
        logger.info("🔄 Models may still work without warm-up")
        return False
