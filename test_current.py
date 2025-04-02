"""
Test script for evaluating the transcription performance of utils.py.

This script measures the performance and accuracy of the current transcription
implementation in utils.py by:
1. Loading the model and warming up the pipeline
2. Measuring the time required to transcribe a test audio file
3. Outputting the transcription results, language detection, and timing data

The output is formatted for easy copying into test_stable.py to enable direct 
comparisons between different transcription implementations.

Usage:
    python test_current.py
"""
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
