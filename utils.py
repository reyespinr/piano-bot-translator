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
import asyncio
import threading
import stable_whisper
from logging_config import get_logger

# Initialize logger for this module
logger = get_logger(__name__)

# Common hallucinations that should be filtered with stricter confidence
COMMON_HALLUCINATIONS = {
    "thank you", "thanks", "okay", "ok", "yes", "yeah", "no", "nope",
    "mm-hmm", "uh-huh", "hmm", "um", "uh", "ah", "oh", "wow", "nice",
    "good", "great", "cool", "awesome", "amazing", "perfect", "exactly",
    "right", "correct", "sure", "absolutely", "definitely", "maybe",
    "i think", "i guess", "i know", "i see", "i understand", "got it",
    "makes sense", "sounds good", "sounds great", "sounds cool"
}

# Dual model architecture - load both models
MODEL_ACCURATE = None  # large-v3-turbo for quality
MODEL_FAST = None      # base for speed
MODEL_ACCURATE_NAME = "large-v3-turbo"
# MODEL_FAST_NAME = "large-v3-turbo"
MODEL_FAST_NAME = "base"

# Model usage tracking with thread-safe locks
MODEL_USAGE_STATS = {
    "accurate_model_busy": False,
    "fast_model_busy": False,
    "accurate_uses": 0,
    "fast_uses": 0,
    "queue_overflows": 0,
    "concurrent_requests": 0,  # Track concurrent requests
    "active_transcriptions": set(),  # Track active transcription IDs
    "stats_lock": threading.Lock(),  # Thread-safe access
    "accurate_model_lock": threading.Lock(),  # ADDED: Lock for accurate model
    "fast_model_lock": threading.Lock()  # ADDED: Lock for fast model
}


def _load_models_if_needed():
    """Lazy load both models only when needed - smart dual model loading"""
    global MODEL_ACCURATE, MODEL_FAST

    if MODEL_ACCURATE is None:
        logger.info("Loading ACCURATE model: %s...", MODEL_ACCURATE_NAME)
        MODEL_ACCURATE = stable_whisper.load_model(
            MODEL_ACCURATE_NAME, device="cuda")
        logger.info("Accurate model loaded successfully!")

    if MODEL_FAST is None:
        logger.info("Loading FAST model: %s...", MODEL_FAST_NAME)
        MODEL_FAST = stable_whisper.load_model(MODEL_FAST_NAME, device="cuda")
        logger.info("Fast model loaded successfully!")

    return MODEL_ACCURATE, MODEL_FAST


def _get_audio_duration_from_file(audio_file_path):
    """Calculate audio duration from WAV file"""
    try:
        with wave.open(audio_file_path, 'rb') as wav_file:
            frames = wav_file.getnframes()
            sample_rate = wav_file.getframerate()
            duration = frames / sample_rate
            return duration
    except (wave.Error, IOError) as e:
        logger.warning("Could not determine audio duration: %s", e)
        return 0.0


def _should_use_fast_model(audio_file_path, current_queue_size=0, concurrent_requests=0, active_transcriptions=0):
    """Determine which model to use based on audio duration and system load"""

    # Get audio duration
    duration = _get_audio_duration_from_file(audio_file_path)

    # Check if accurate model is currently busy
    accurate_busy = MODEL_USAGE_STATS["accurate_model_busy"]

    # DEBUG: Add detailed logging with both counters
    logger.debug("🔍 Routing decision: duration=%.1fs, accurate_busy=%s, concurrent=%d, active=%d, queue=%d",
                 duration, accurate_busy, concurrent_requests, active_transcriptions, current_queue_size)

    # AGGRESSIVE ROUTING: If accurate model is busy, route short audio to fast model
    if accurate_busy and duration < 3.0:  # Increased threshold to 3 seconds
        logger.info(
            "🚀 Routing audio (%.1fs) to FAST model (accurate model busy)", duration)
        MODEL_USAGE_STATS["fast_uses"] += 1
        return True

    # CONCURRENT PROCESSING: If we have multiple ACTIVE transcriptions, use fast model for shorter clips
    if active_transcriptions >= 1 and duration < 2.0:  # Use active transcriptions instead
        logger.info("⚡ Concurrent processing (%d active transcriptions), routing short audio (%.1fs) to FAST model",
                    active_transcriptions, duration)
        MODEL_USAGE_STATS["fast_uses"] += 1
        return True

    # QUEUE-BASED ROUTING: High queue load - use fast model more aggressively
    if current_queue_size > 2 and duration < 2.5:
        logger.info("🔥 Queue load (%d), routing audio (%.1fs) to FAST model",
                    current_queue_size, duration)
        MODEL_USAGE_STATS["fast_uses"] += 1
        MODEL_USAGE_STATS["queue_overflows"] += 1
        return True

    # EVERY 3rd SHORT CLIP: Route every 3rd short audio to fast model for load balancing
    total_uses = MODEL_USAGE_STATS["accurate_uses"] + \
        MODEL_USAGE_STATS["fast_uses"]
    if duration < 1.5 and total_uses > 0 and total_uses % 3 == 0:
        logger.info(
            "⚖️ Load balancing: routing short audio (%.1fs) to FAST model (every 3rd)", duration)
        MODEL_USAGE_STATS["fast_uses"] += 1
        return True

    # Default: Use accurate model for best quality
    logger.info("🎯 Routing audio (%.1fs) to ACCURATE model (concurrent: %d, active: %d, queue: %d)",
                duration, concurrent_requests, active_transcriptions, current_queue_size)
    MODEL_USAGE_STATS["accurate_uses"] += 1
    return False


def _is_recoverable_error(error_str):
    """Check if the error is a recoverable PyTorch/FFmpeg error."""
    recoverable_errors = [
        "Expected key.size",
        "Invalid argument",
        "Error muxing",
        "Error submitting",
        "Error writing trailer",
        "Error closing file",
        "RuntimeError: NYI",  # TorchScript VAD errors
        "Error submitting a packet to the muxer",  # FFmpeg packet errors
        "vad_annotator.py"  # Any VAD-related errors
    ]

    return (any(error in error_str for error in recoverable_errors) or
            "tensor" in error_str.lower() or
            "NYI" in error_str or
            "TorchScript" in error_str)  # Catch TorchScript errors


def _reset_model_state(model):
    """Reset model internal state to recover from corrupted state."""
    try:
        # Clear any cached states that might be corrupted
        if hasattr(model, 'model') and hasattr(model.model, 'decoder'):
            # Reset decoder state if available
            if hasattr(model.model.decoder, 'reset_state'):
                model.model.decoder.reset_state()

        # Force garbage collection to clear any corrupted tensors
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.debug("Model state reset completed")
        return True
    except Exception as e:
        logger.warning("Could not reset model state: %s", str(e))
        return False


async def transcribe_with_model(audio_file_path, model, model_name):
    """Transcribe using a specific model with consistent parameters - TRUE ASYNC VERSION"""

    # Generate unique ID for this transcription request
    import uuid
    transcription_id = str(uuid.uuid4())[:8]

    # Get the appropriate model lock
    model_lock = MODEL_USAGE_STATS["accurate_model_lock"] if model_name == "accurate" else MODEL_USAGE_STATS["fast_model_lock"]

    # Thread-safe tracking of active transcriptions
    with MODEL_USAGE_STATS["stats_lock"]:
        MODEL_USAGE_STATS["active_transcriptions"].add(transcription_id)
        active_count = len(MODEL_USAGE_STATS["active_transcriptions"])

        # Mark model as busy
        if model_name == "accurate":
            MODEL_USAGE_STATS["accurate_model_busy"] = True
        else:
            MODEL_USAGE_STATS["fast_model_busy"] = True

    try:
        logger.debug("🔄 [%s] Processing with %s model... (active: %d)",
                     transcription_id, model_name.upper(), active_count)

        # CRITICAL: Use model lock to prevent concurrent access to the same model
        def safe_transcribe():
            with model_lock:  # Serialize access to each model
                retry_count = 0
                max_retries = 2

                while retry_count <= max_retries:
                    try:
                        # First attempt: normal transcription with VAD
                        if retry_count == 0:
                            return model.transcribe(
                                audio_file_path,
                                vad=True,
                                vad_threshold=0.35,
                                no_speech_threshold=0.6,
                                max_instant_words=0.3,
                                suppress_silence=True,
                                only_voice_freq=True,
                                word_timestamps=True
                            )
                        # Second attempt: without VAD
                        elif retry_count == 1:
                            logger.warning(
                                "Retrying %s model without VAD", model_name)
                            return model.transcribe(
                                audio_file_path,
                                vad=False,  # Disable VAD
                                no_speech_threshold=0.6,
                                max_instant_words=0.3,
                                suppress_silence=True,
                                only_voice_freq=True,
                                word_timestamps=False  # Also disable word timestamps
                            )
                        # Third attempt: minimal settings
                        else:
                            logger.warning(
                                "Final retry for %s model with minimal settings", model_name)
                            return model.transcribe(
                                audio_file_path,
                                vad=False,
                                no_speech_threshold=0.8,  # Higher threshold
                                suppress_silence=False,   # Disable silence suppression
                                only_voice_freq=False,    # Disable frequency filtering
                                word_timestamps=False     # No word timestamps
                            )

                    except RuntimeError as e:
                        error_str = str(e)
                        retry_count += 1

                        if _is_recoverable_error(error_str):
                            logger.warning(
                                "PyTorch/FFmpeg/VAD error in %s model (attempt %d/%d): %s",
                                model_name, retry_count, max_retries + 1, error_str)

                            # Reset model state before retry
                            _reset_model_state(model)

                            # Wait progressively longer between retries
                            import time
                            time.sleep(0.1 * retry_count)

                            if retry_count > max_retries:
                                logger.error(
                                    "Max retries exceeded for %s model, skipping audio", model_name)
                                return None  # Return None to indicate failure
                            continue
                        else:
                            # Non-recoverable error, re-raise
                            raise

                return None  # Should not reach here

        # Run the blocking transcription in a thread pool to make it truly async
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, safe_transcribe)

        # Handle failure case
        if result is None:
            logger.warning(
                "Transcription failed for %s model after all retries", model_name)
            return "", "en", None

        # Extract text and detected language
        transcribed_text = result.text if result.text else ""
        detected_language = result.language if result.language else "en"

        logger.debug("✅ [%s] %s model completed transcription",
                     transcription_id, model_name.upper())
        return transcribed_text, detected_language, result

    except Exception as e:
        logger.error("Unexpected error in %s model: %s", model_name, str(e))
        # Reset model state on any unexpected error
        _reset_model_state(model)
        # Return empty result instead of crashing
        return "", "en", None
    finally:
        # Thread-safe cleanup
        with MODEL_USAGE_STATS["stats_lock"]:
            MODEL_USAGE_STATS["active_transcriptions"].discard(
                transcription_id)

            # Mark model as not busy only if no other requests using this model
            if model_name == "accurate":
                MODEL_USAGE_STATS["accurate_model_busy"] = False
            else:
                MODEL_USAGE_STATS["fast_model_busy"] = False


async def transcribe(audio_file_path, current_queue_size=0):
    """Smart transcribe with dual model routing based on audio duration and load.

    Routes audio to appropriate model:
    - ACCURATE model (large-v3-turbo) for most audio (best quality)
    - FAST model (base) only when accurate is busy AND audio is short

    Args:
        audio_file_path (str): Path to the audio file to transcribe
        current_queue_size (int): Current transcription queue size for load balancing

    Returns:
        tuple: (transcribed_text, detected_language)
    """

    # CRITICAL FIX: Track concurrent requests at the entry point with thread safety
    with MODEL_USAGE_STATS["stats_lock"]:
        MODEL_USAGE_STATS["concurrent_requests"] += 1
        current_concurrent = MODEL_USAGE_STATS["concurrent_requests"]
        active_transcriptions = len(MODEL_USAGE_STATS["active_transcriptions"])

    try:
        # Load both models if needed
        model_accurate, model_fast = _load_models_if_needed()

        # Determine which model to use (now with proper concurrent tracking)
        use_fast = _should_use_fast_model(audio_file_path, current_queue_size,
                                          current_concurrent, active_transcriptions)

        if use_fast:
            # Use fast model for quick processing
            transcribed_text, detected_language, result = await transcribe_with_model(
                audio_file_path, model_fast, "fast"
            )
        else:
            # Use accurate model for best quality
            transcribed_text, detected_language, result = await transcribe_with_model(
                audio_file_path, model_accurate, "accurate"
            )

        # Skip further processing if we got an empty result due to tensor errors
        if not transcribed_text and not detected_language:
            logger.debug("Skipping further processing due to model error")
            return "", "en"

        # Handle Austrian German misidentified as Icelandic (use accurate model for re-transcription)
        if detected_language == "is":
            logger.info(
                "Detected Icelandic - likely Austrian German. Re-transcribing with accurate model...")

            # PROTECTED: Use the same safe transcription approach for re-transcription
            def safe_retranscribe():
                with MODEL_USAGE_STATS["accurate_model_lock"]:
                    try:
                        return model_accurate.transcribe(
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
                    except RuntimeError as e:
                        error_str = str(e)
                        if _is_recoverable_error(error_str):
                            logger.warning(
                                "PyTorch/FFmpeg/VAD error during re-transcription (retrying without VAD): %s", error_str)
                            # Retry without VAD for problematic audio
                            import time
                            time.sleep(0.2)
                            return model_accurate.transcribe(
                                audio_file_path,
                                vad=False,  # Disable VAD on retry
                                no_speech_threshold=0.6,
                                max_instant_words=0.3,
                                suppress_silence=True,
                                only_voice_freq=True,
                                word_timestamps=True,
                                language="de"
                            )
                        return None

            try:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, safe_retranscribe)

                if result:
                    transcribed_text = result.text if result.text else ""
                    detected_language = "de"
                    logger.info(
                        "Re-transcribed as German using accurate model")
            except Exception as e:
                logger.warning(
                    "Re-transcription failed, using original result: %s", str(e))

        # Apply additional confidence filtering as a safety net
        if result and hasattr(result, "segments") and result.segments:
            # Get confidence values
            confidences = []
            for segment in result.segments:
                if hasattr(segment, "avg_logprob"):
                    confidences.append(segment.avg_logprob)

            # Apply our confidence thresholds if we have confidence data
            if confidences:
                # General confidence threshold
                avg_log_prob = sum(confidences) / len(confidences)
                confidence_threshold = -1.5
                if avg_log_prob < confidence_threshold:
                    logger.info(
                        "Low confidence transcription rejected (%s): '%s'",
                        f"{avg_log_prob:.2f}", transcribed_text)
                    return "", detected_language

                # Special case for common hallucinations
                text = transcribed_text.strip().lower()
                text_clean = text.translate(
                    str.maketrans('', '', string.punctuation))

                if len(text_clean) < 15 and text_clean in COMMON_HALLUCINATIONS:
                    # For these common short responses, require higher confidence
                    stricter_threshold = -0.5
                    if avg_log_prob < stricter_threshold:
                        logger.info(
                            "Short statement '%s' rejected with confidence %.2f", text, avg_log_prob)
                        return "", detected_language

                logger.info(
                    "Transcription confidence: %.2f, Language: %s", avg_log_prob, detected_language)

        # Log model usage stats periodically
        total_uses = MODEL_USAGE_STATS["accurate_uses"] + \
            MODEL_USAGE_STATS["fast_uses"]
        if total_uses % 10 == 0 and total_uses > 0:
            accuracy_rate = (
                MODEL_USAGE_STATS["accurate_uses"] / total_uses) * 100
            fast_rate = (MODEL_USAGE_STATS["fast_uses"] / total_uses) * 100
            logger.info("📊 Model usage: Accurate: %d (%.1f%%), Fast: %d (%.1f%%), Queue overflows: %d",
                        MODEL_USAGE_STATS["accurate_uses"], accuracy_rate,
                        MODEL_USAGE_STATS["fast_uses"], fast_rate,
                        MODEL_USAGE_STATS["queue_overflows"])

        return transcribed_text, detected_language

    finally:
        # CRITICAL FIX: Always decrement concurrent requests when done (thread-safe)
        with MODEL_USAGE_STATS["stats_lock"]:
            MODEL_USAGE_STATS["concurrent_requests"] -= 1


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
    """Warm up both transcription models for optimal performance."""
    logger.info("Warming up DUAL MODEL transcription pipeline...")
    try:
        # Create a dummy audio file
        dummy_file = create_dummy_audio_file()

        # Load both models explicitly
        logger.info("Loading both models for warm-up...")
        model_accurate, model_fast = _load_models_if_needed()

        # Warm up accurate model
        logger.info("Warming up accurate model...")
        await transcribe_with_model(dummy_file, model_accurate, "accurate")

        # Warm up fast model
        logger.info("Warming up fast model...")
        await transcribe_with_model(dummy_file, model_fast, "fast")

        # Clean up
        if os.path.exists(dummy_file):
            os.remove(dummy_file)

        logger.info(
            "🎯 Dual model pipeline warm-up complete! Ready for smart routing.")
        logger.info("📈 Accurate model: %s, Fast model: %s",
                    MODEL_ACCURATE_NAME, MODEL_FAST_NAME)

    except (IOError, FileNotFoundError, PermissionError, RuntimeError) as e:
        logger.error("Warm-up error (non-critical): %s", str(e))
