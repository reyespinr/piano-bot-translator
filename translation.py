"""
Translation utilities for converting transcribed text to English.

This module provides translation services using DeepL's API and includes
optimization logic to avoid unnecessary translations.
"""
import requests
from logging_config import get_logger

logger = get_logger(__name__)


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
            logger.info(
                "Detected mixed language content (non-ASCII ratio: %.2f - translating)",
                non_ascii_ratio)
            return True
        return False

    # Non-English dominant language should be translated
    return True
