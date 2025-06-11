"""
Core transcription functionality using Whisper models.

This module handles the actual transcription of audio files using the loaded
Whisper models, with intelligent routing between accurate and fast models.
Now uses the unified ModelManager for better organization.

REFACTORED: Large functions have been broken down into modular components
in transcription_core.py for better maintainability and testing.
"""
import asyncio
import gc
import random
import string
import threading
import time
import traceback
import uuid
import torch
from model_manager import model_manager
import models  # For backward compatibility
import audio_utils
from audio_utils import COMMON_HALLUCINATIONS
from logging_config import get_logger

# Import refactored components
from transcription_core import transcribe_with_model_refactored

logger = get_logger(__name__)


async def transcribe(audio_file, current_queue_size=0, concurrent_requests=0, active_transcriptions=0):
    """
    Main transcription function with intelligent model routing.

    Args:
        audio_file: Path to audio file to transcribe
        current_queue_size: Current queue size for smart routing
        concurrent_requests: Number of concurrent requests
        active_transcriptions: Number of active transcriptions

    Returns:
        tuple: (transcribed_text, detected_language)
    """
    transcription_id = str(uuid.uuid4())[:8]
    logger.debug("🎬 [%s] ENTRY: transcribe() called for file: %s",
                 transcription_id, audio_file)

    try:
        # Load models using the new model manager
        logger.debug("🔧 [%s] Loading models...", transcription_id)

        # Check if models are loaded, if not, initialize them
        if not model_manager.stats["models_loaded"]:
            logger.warning(
                "[%s] Models not loaded! Initializing...", transcription_id)
            success = await model_manager.initialize_models(warm_up=False)
            if not success:
                logger.error("[%s] Failed to initialize models!",
                             transcription_id)
                return "", "error"

        # Get models from model manager
        model_accurate, model_fast_pool = model_manager.get_models()

        # Determine model routing based on audio characteristics and system load
        use_fast = should_use_fast_model(
            audio_file, current_queue_size, concurrent_requests, active_transcriptions)

        if use_fast:
            logger.info("🚀 [%s] Using FAST model...", transcription_id)
            fast_model, fast_lock, selected_model_index = model_manager.select_fast_model()
            if fast_model is None:
                logger.warning(
                    "[%s] No fast models available, falling back to accurate", transcription_id)
                model_accurate, accurate_lock = model_manager.get_accurate_model()
                transcribed_text, detected_language, result = await transcribe_with_model_refactored(
                    audio_file, model_accurate, "accurate"
                )
            else:
                transcribed_text, detected_language, result = await transcribe_with_model_refactored(
                    audio_file, fast_model, "fast", selected_model_index
                )
        else:
            logger.info("🎯 [%s] Using ACCURATE model...", transcription_id)
            model_accurate, accurate_lock = model_manager.get_accurate_model()
            transcribed_text, detected_language, result = await transcribe_with_model_refactored(
                audio_file, model_accurate, "accurate"
            )

        # Return the transcription results
        logger.info("🎉 [%s] SUCCESS: text='%s', language=%s",
                    transcription_id, transcribed_text[:50] if transcribed_text else "None", detected_language)
        return transcribed_text, detected_language

    except Exception as e:
        logger.error("❌ [%s] Error in transcribe(): %s",
                     transcription_id, str(e))
        logger.error("❌ [%s] Transcribe error traceback: %s",
                     transcription_id, traceback.format_exc())
        return "", "error"


def should_use_fast_model(audio_file_path, current_queue_size=0, concurrent_requests=0, active_transcriptions=0):
    """Determine which model to use based on audio duration and system load"""

    # Get audio duration
    duration = audio_utils.get_audio_duration_from_file(audio_file_path)

    # Get current system state from model manager
    stats = model_manager.get_stats()
    accurate_busy = stats["accurate_busy"]
    # DEBUG: Add detailed logging with pool information
    available_fast_models = stats["fast_available"]
    logger.debug("🔍 Routing decision: duration=%.1fs, accurate_busy=%s, concurrent=%d, active=%d, queue=%d, fast_available=%d/%d",
                 duration, accurate_busy, concurrent_requests, active_transcriptions, current_queue_size,
                 available_fast_models, stats["fast_models"])

    # AGGRESSIVE ROUTING: If accurate model is busy, route to fast models
    if accurate_busy and duration < 3.0 and available_fast_models > 0:
        logger.info("🚀 Routing audio (%.1fs) to FAST model pool (accurate model busy, %d models available)",
                    duration, available_fast_models)
        model_manager.update_stats(fast_uses=stats["fast_uses"] + 1)
        return True

    # CONCURRENT PROCESSING: If we have multiple ACTIVE transcriptions and fast models available
    if active_transcriptions >= 1 and duration < 2.0 and available_fast_models > 0:
        logger.info("⚡ Concurrent processing (%d active transcriptions), routing short audio (%.1fs) to FAST model pool (%d available)",
                    active_transcriptions, duration, available_fast_models)
        model_manager.update_stats(fast_uses=stats["fast_uses"] + 1)
        return True

    # QUEUE-BASED ROUTING: High queue load - use fast models more aggressively
    if current_queue_size > 2 and duration < 2.5 and available_fast_models > 0:
        logger.info("🔥 Queue load (%d), routing audio (%.1fs) to FAST model pool (%d available)",
                    current_queue_size, duration, available_fast_models)
        model_manager.update_stats(
            fast_uses=stats["fast_uses"] + 1,
            queue_overflows=stats["queue_overflows"] + 1
        )
        return True

    # LOAD BALANCING: If we have multiple fast models available, use them more often
    if available_fast_models >= 2 and duration < 3.0:
        logger.info("⚖️ Load balancing: routing audio (%.1fs) to FAST model pool (%d models available)",
                    duration, available_fast_models)
        model_manager.update_stats(fast_uses=stats["fast_uses"] + 1)
        return True

    # EVERY 3rd SHORT CLIP: Route every 3rd short audio to fast model for load balancing
    total_uses = stats["accurate_uses"] + stats["fast_uses"]
    if duration < 1.5 and total_uses > 0 and total_uses % 3 == 0 and available_fast_models > 0:
        logger.info(
            "⚖️ Periodic load balancing: routing short audio (%.1fs) to FAST model pool (every 3rd)", duration)
        model_manager.update_stats(fast_uses=stats["fast_uses"] + 1)
        return True

    # Default: Use accurate model for best quality
    logger.info("🎯 Routing audio (%.1fs) to ACCURATE model (concurrent: %d, active: %d, queue: %d, fast_available: %d)",
                duration, concurrent_requests, active_transcriptions, current_queue_size, available_fast_models)
    model_manager.update_stats(accurate_uses=stats["accurate_uses"] + 1)
    return False


def is_recoverable_error(error_str):
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
        "vad_annotator.py",  # Any VAD-related errors
        "Error muxing a packet",  # Additional FFmpeg muxer errors
        "Task finished with error code",  # FFmpeg task errors
        "Terminating thread with return code",  # FFmpeg thread errors
        "out#0/s16le",  # FFmpeg output format errors
        "aost#0:0/pcm_s16le",  # FFmpeg audio stream errors
        "ffmpeg",  # General FFmpeg errors
        "av_",  # FFmpeg/libav function errors
        "codec",  # Audio codec errors
        "format"  # Audio format errors
    ]

    return (any(error in error_str for error in recoverable_errors) or
            "tensor" in error_str.lower() or
            "NYI" in error_str or
            "TorchScript" in error_str or
            "ffmpeg" in error_str.lower() or
            "libav" in error_str.lower())  # Catch all FFmpeg/audio errors


def reset_model_state(model):
    """Reset model internal state to recover from corrupted state."""
    try:
        # Clear any cached states that might be corrupted
        if hasattr(model, 'model') and hasattr(model.model, 'decoder'):
            # Reset decoder state if available
            if hasattr(model.model.decoder, 'reset_state'):
                model.model.decoder.reset_state()

        # Force garbage collection to clear any corrupted tensors
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.debug("Model state reset completed")
        return True
    except Exception as e:
        logger.warning("Could not reset model state: %s", str(e))
        return False


# Backward compatibility wrapper for old transcribe_with_model function
async def transcribe_with_model(audio_file, model, model_name, model_index=None):
    """
    REFACTORED: This function now delegates to the modular implementation.

    Backward compatibility wrapper for the old massive function.
    The actual logic is now in transcription_core.py for better maintainability.
    """
    return await transcribe_with_model_refactored(audio_file, model, model_name, model_index)


def apply_confidence_filtering(result, transcribed_text, transcription_id):
    """Apply confidence filtering to transcription results using the original logic.

    Returns:
        tuple: (passed_filter: bool, confidence_score: float)
    """
    try:
        # Use the exact same logic as your original function
        confidences = []
        if hasattr(result, "segments") and result.segments:
            # Get confidence values from segments (exactly like your original)
            for segment in result.segments:
                if hasattr(segment, "avg_logprob"):
                    # Apply confidence filtering if we have confidence data (like your original)
                    confidences.append(segment.avg_logprob)
        if confidences:
            avg_log_prob = sum(confidences) / len(confidences)

            # General confidence threshold (same as your original)
            confidence_threshold = -1.5
            if avg_log_prob < confidence_threshold:
                logger.info("❌ [%s] Low confidence transcription rejected (%.2f): '%s'",
                            transcription_id, avg_log_prob, transcribed_text)
                return False, avg_log_prob
            text = transcribed_text.strip().lower()
            text_clean = text.translate(
                str.maketrans('', '', string.punctuation))

            if len(text_clean) < 15 and text_clean in COMMON_HALLUCINATIONS:
                # For these common short responses, require higher confidence (same as original)
                stricter_threshold = -0.5
                if avg_log_prob < stricter_threshold:
                    logger.info("❌ [%s] Short hallucination '%s' rejected with confidence %.2f",
                                transcription_id, text_clean, avg_log_prob)
                    return False, avg_log_prob

            logger.info("✅ [%s] Transcription confidence: %.2f, passed filtering",
                        transcription_id, avg_log_prob)
            return True, avg_log_prob
        else:
            # No confidence data available - this shouldn't happen with proper Whisper results
            logger.warning("[%s] No confidence data available from segments - result type: %s",
                           transcription_id, type(result).__name__)

            # Try to get confidence from result object directly as fallback
            if hasattr(result, 'avg_logprob'):
                confidence = getattr(result, 'avg_logprob', 0.0)
                logger.info(
                    "📊 [%s] Using result-level confidence: %.2f", transcription_id, confidence)
                return confidence >= -1.5, confidence

            # Basic text quality filtering as last resort
            if not transcribed_text or len(transcribed_text.strip()) < 2:
                return False, 0.0

            # Check for obvious hallucinations without confidence
            text_clean = transcribed_text.strip().lower().translate(
                str.maketrans('', '', string.punctuation))

            if len(text_clean) < 10 and text_clean in COMMON_HALLUCINATIONS:
                logger.info("❌ [%s] Short hallucination '%s' filtered without confidence data",
                            transcription_id, text_clean)
                return False, 0.0

            logger.warning(
                "[%s] No confidence data - accepting by default", transcription_id)
            return True, 0.0  # Default to accepting if no confidence data

    except Exception as e:
        logger.error("Error in confidence filtering: %s", str(e))
        return True, 0.0  # Default to accepting on error
