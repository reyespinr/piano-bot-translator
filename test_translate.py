"""
Test script for comparing translation approaches with stable-ts.

This script evaluates two approaches for translating audio to English:
1. Two-step approach: stable-ts transcription + DeepL translation
2. Direct approach: stable-ts translation to English in one step

It measures performance, accuracy, and quality differences between the approaches.

Usage:
    python test_translate.py
"""
import os
import time
import asyncio
import utils


async def test_current_approach(audio_file):
    """Test transcription with stable-ts + translation with DeepL."""
    start_time = time.time()

    # Get transcription and detected language
    transcribed_text, detected_language = await utils.transcribe(audio_file)

    # Track transcription time
    transcription_time = time.time() - start_time

    # Skip if empty transcription
    if not transcribed_text:
        return None, None, transcription_time, 0, 0, detected_language

    # Start translation timing
    translation_start = time.time()

    # Check if translation needed
    needs_translation = await utils.should_translate(transcribed_text, detected_language)

    if needs_translation:
        translated_text = await utils.translate(transcribed_text)
    else:
        translated_text = transcribed_text

    # Calculate times
    translation_time = time.time() - translation_start
    total_time = time.time() - start_time

    return (
        transcribed_text,
        translated_text,
        transcription_time,
        translation_time,
        total_time,
        detected_language
    )


async def test_whisper_direct(audio_file):
    """Test direct translation with stable-ts.

    This uses stable-ts's direct translation capability with the same
    VAD and filtering features used for transcription.
    """
    start_time = time.time()

    # Use stable-ts for translation with VAD and other enhancements
    translate_start = time.time()
    result = utils.MODEL.transcribe(
        audio_file,
        task="translate",         # Force translation to English
        vad=True,                 # Enable Voice Activity Detection
        vad_threshold=0.35,       # VAD confidence threshold
        no_speech_threshold=0.6,  # Filter non-speech sections
        max_instant_words=0.3,    # Reduce hallucination words
        suppress_silence=True,    # Use silence detection for better timestamps
        only_voice_freq=True      # Focus on human voice frequency range
    )

    # Extract results - stable-ts might have a different API
    translated_text = result.text if hasattr(result, "text") else ""
    detected_language = result.language if hasattr(result, "language") else ""

    # Calculate times
    translation_time = time.time() - translate_start
    total_time = time.time() - start_time

    print(f"stable-ts direct translation time: {translation_time:.2f}s")

    return translated_text, detected_language, total_time


async def main():
    """Run translation tests and compare approaches."""
    # Test files - we'll assume test_audio.wav exists
    test_file = "test_audio.wav"
    if not os.path.exists(test_file):
        print(f"Error: {test_file} not found")
        return

    # Warm up both pipelines
    print("Warming up pipelines...")
    await utils.warm_up_pipeline()
    print("Warm-up complete\n")

    # Test current approach (stable-ts + DeepL)
    print("===== Testing Current Approach: stable-ts transcribe → DeepL translate =====")
    (
        original,
        translated_deepl,
        transcribe_time,
        translate_time,
        total_time_current,
        language
    ) = await test_current_approach(test_file)

    print(f"Detected language: {language}")
    print(f"Original transcription: \"{original}\"")
    print(f"DeepL translation: \"{translated_deepl}\"")
    print(f"Transcription time: {transcribe_time:.2f}s")
    print(f"Translation time: {translate_time:.2f}s")
    print(f"Total time: {total_time_current:.2f}s\n")

    # Test stable-ts direct translation
    print("===== Testing Direct Approach: stable-ts translate =====")
    translated_whisper, language, total_time_whisper = await test_whisper_direct(test_file)

    print(f"Detected language: {language}")
    print(f"stable-ts direct translation: \"{translated_whisper}\"")
    print(f"Total time: {total_time_whisper:.2f}s\n")

    # Compare results
    print("===== Performance Comparison =====")
    print(f"Current approach (stable-ts+DeepL): {total_time_current:.2f}s")
    print(f"Direct approach (stable-ts translate): {total_time_whisper:.2f}s")
    time_diff = total_time_current - total_time_whisper
    percent = abs(time_diff)/min(total_time_current, total_time_whisper)*100
    faster = "faster" if time_diff > 0 else "slower"
    print(f"Difference: {abs(time_diff):.2f}s ({percent:.1f}% {faster})")

    # Quality assessment
    print("\n===== Translation Quality Comparison =====")
    print("Please review the quality of the two translations:")
    print(f"1. DeepL: \"{translated_deepl}\"")
    print(f"2. stable-ts direct: \"{translated_whisper}\"")


if __name__ == "__main__":
    asyncio.run(main())
