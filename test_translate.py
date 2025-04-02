import os
import time
import asyncio
import utils
import whisper


async def test_current_approach(audio_file):
    """Test transcription with Whisper + translation with DeepL."""
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

    return transcribed_text, translated_text, transcription_time, translation_time, total_time, detected_language


async def test_whisper_direct(audio_file):
    """Test direct translation with Whisper."""
    start_time = time.time()

    # Always use Whisper's translation for testing purposes
    # This helps with mixed-language audio where detection might be imprecise
    translate_start = time.time()
    result = utils.MODEL.transcribe(
        audio_file,
        fp16=True,
        task="translate"  # Force translation to English
    )
    translated_text = result.get("text", "")
    detected_language = result.get("language", "")

    # Calculate times
    translation_time = time.time() - translate_start
    total_time = time.time() - start_time

    print(f"Whisper translation time: {translation_time:.2f}s")

    return translated_text, detected_language, total_time


async def main():

    # Test files - we'll assume test_audio.wav exists
    test_file = "test_audio.wav"
    if not os.path.exists(test_file):
        print(f"Error: {test_file} not found")
        return

    # Warm up both pipelines
    print("Warming up pipelines...")
    await utils.warm_up_pipeline()
    print("Warm-up complete\n")

    # Test current approach (Whisper + DeepL)
    print("===== Testing Current Approach: Whisper transcribe → DeepL translate =====")
    original, translated_deepl, transcribe_time, translate_time, total_time_current, language = await test_current_approach(test_file)

    print(f"Detected language: {language}")
    print(f"Original transcription: \"{original}\"")
    print(f"DeepL translation: \"{translated_deepl}\"")
    print(f"Transcription time: {transcribe_time:.2f}s")
    print(f"Translation time: {translate_time:.2f}s")
    print(f"Total time: {total_time_current:.2f}s\n")

    # Test Whisper direct translation
    print("===== Testing Direct Approach: Whisper translate =====")
    translated_whisper, language, total_time_whisper = await test_whisper_direct(test_file)

    print(f"Detected language: {language}")
    print(f"Whisper translation: \"{translated_whisper}\"")
    print(f"Total time: {total_time_whisper:.2f}s\n")

    # Compare results
    print("===== Performance Comparison =====")
    print(f"Current approach (Whisper+DeepL): {total_time_current:.2f}s")
    print(f"New approach (Whisper direct): {total_time_whisper:.2f}s")
    time_diff = total_time_current - total_time_whisper
    percent = abs(time_diff)/min(total_time_current, total_time_whisper)*100
    faster = "faster" if time_diff > 0 else "slower"
    print(f"Difference: {abs(time_diff):.2f}s ({percent:.1f}% {faster})")

    # Quality assessment
    print("\n===== Translation Quality Comparison =====")
    print("Please review the quality of the two translations:")
    print(f"1. DeepL: \"{translated_deepl}\"")
    print(f"2. Whisper: \"{translated_whisper}\"")

if __name__ == "__main__":
    asyncio.run(main())
