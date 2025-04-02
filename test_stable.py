"""
Test script for evaluating stable-ts transcription performance.

This script measures the performance and accuracy of the stable-ts implementation
of Whisper and compares it against previous results from the standard Whisper
implementation. It evaluates:
1. Transcription speed and efficiency
2. Language detection accuracy
3. Text quality and confidence scores
4. Hallucination prevention (especially for "thank you" hallucinations)

The script requires manually entering previous Whisper results for direct comparison.

Usage:
    python test_stable.py
"""
import os
import time
import asyncio
import wave
import stable_whisper
import numpy as np


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


async def test_stable_ts(audio_file):
    """Test the stable-ts implementation."""
    # Load the same model size as the current one
    model_size = "large-v3-turbo"  # This is fine if it's loading correctly

    # Load model
    print(f"Loading stable-ts model: {model_size}")
    model_load_start = time.time()
    model = stable_whisper.load_model(model_size, device='cuda')
    model_load_time = time.time() - model_load_start
    print(f"Model loading time: {model_load_time:.2f}s")

    # Warm up the model like in utils.py
    print("Warming up stable-ts model...")
    warmup_file = create_dummy_audio_file("warmup_stable.wav")
    _ = model.transcribe(warmup_file, vad=True)
    os.remove(warmup_file)
    print("Warm-up complete\n")

    # Start transcription timing (after model load and warm-up)
    print(f"Processing {audio_file}...")
    transcribe_start = time.time()

    # Transcribe with stable-ts using the correct parameters
    result = model.transcribe(
        audio_file,
        vad=True,                  # Enable Voice Activity Detection
        vad_threshold=0.35,        # Default from docs
        no_speech_threshold=0.6,   # Helps filter non-speech sections
        max_instant_words=0.3,     # Lower threshold to be more aggressive
        suppress_silence=True,     # Enable silence suppression
        only_voice_freq=True,      # Focus on human voice frequency range
        word_timestamps=True       # Important for proper segmentation
    )

    transcribed_text = result.text
    detected_language = result.language

    # Calculate transcription time
    transcribe_time = time.time() - transcribe_start

    # Get confidence data if available - FIXED for stable-ts objects
    confidence_info = ""
    if hasattr(result, "segments") and result.segments:
        try:
            # Access avg_logprob as an attribute, not with get()
            confidences = []
            for segment in result.segments:
                if hasattr(segment, "avg_logprob"):
                    confidences.append(segment.avg_logprob)

            if confidences:
                avg_confidence = sum(confidences) / len(confidences)
                confidence_info = f" (Avg confidence: {avg_confidence:.2f})"
        except (AttributeError, TypeError) as e:
            confidence_info = f" (Could not calculate confidence: {e})"

    return transcribed_text, detected_language, transcribe_time, confidence_info


async def main():
    """Run the stable-ts test and compare with previous Whisper results.

    This function loads a test audio file, processes it with stable-ts,
    and compares the results with previously recorded standard Whisper results
    for speed, accuracy, and hallucination prevention.
    """
    # Get previous results from the last run (to avoid re-running utils.transcribe)
    # Replace these values with your previous test results
    # Replace with actual previous result
    current_text = " Thank you."
    current_lang = "en"
    current_time = 1.36

    # Test file
    test_file = "test_audio4.wav"
    if not os.path.exists(test_file):
        print(f"Error: {test_file} not found")
        return

    print("===== Previous Whisper Implementation Results =====")
    print(f"Detected language: {current_lang}")
    print(f"Transcription: \"{current_text}\"")
    print(f"Processing time: {current_time:.2f}s\n")

    print("===== Testing stable-ts Implementation =====")
    text, lang, transcribe_time, confidence_info = await test_stable_ts(test_file)

    print(f"Detected language: {lang}")
    print(f"Transcription: \"{text}\"{confidence_info}")
    print(f"Processing time: {transcribe_time:.2f}s\n")

    # Compare transcription time (most relevant for your use case)
    print("===== Performance Comparison =====")
    print(f"Previous Whisper: {current_time:.2f}s")
    print(f"stable-ts: {transcribe_time:.2f}s")
    time_diff = current_time - transcribe_time
    percent = abs(time_diff)/min(current_time, transcribe_time)*100
    faster = "faster" if time_diff > 0 else "slower"
    print(
        f"Difference: {abs(time_diff):.2f}s "
        f"({percent:.1f}% {faster} than previous approach)")

    # Quality assessment
    print("\n===== Transcription Quality Comparison =====")
    print("Please review both transcriptions:")
    print(f"1. Previous: \"{current_text}\"")
    print(f"2. stable-ts: \"{text}\"")

    # Special hallucination test for "thank you"
    print("\n===== Special Hallucination Test =====")
    count_current = current_text.lower().count("thank you")
    count_stable = text.lower().count("thank you")
    print(f"'thank you' instances in Previous: {count_current}")
    print(f"'thank you' instances in stable-ts: {count_stable}")


if __name__ == "__main__":
    asyncio.run(main())
