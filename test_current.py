import os
import time
import asyncio
import utils


async def main():
    """Test the current transcription in utils.py and output timing data."""
    # Test file
    test_file = "test_audio4.wav"
    if not os.path.exists(test_file):
        print(f"Error: {test_file} not found")
        return

    # Warm up pipeline
    print("Warming up pipeline...")
    await utils.warm_up_pipeline()
    print("Warm-up complete\n")

    # Run and time the transcription
    print("===== Current Whisper Implementation =====")
    start_time = time.time()
    transcribed_text, detected_language = await utils.transcribe(test_file)
    elapsed_time = time.time() - start_time

    # Output results for copying
    print(f"Detected language: {detected_language}")
    print(f"Transcription: \"{transcribed_text}\"")
    print(f"Processing time: {elapsed_time:.2f}s\n")

    # Print values in format ready for test_stable.py
    print("===== Values for test_stable.py =====")
    print(f"current_text = \"{transcribed_text}\"")
    print(f"current_lang = \"{detected_language}\"")
    print(f"current_time = {elapsed_time:.2f}")

if __name__ == "__main__":
    asyncio.run(main())
