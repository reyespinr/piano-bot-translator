"""
Core transcription functionality using Whisper models.

This module handles the actual transcription of audio files using the loaded
Whisper models, with intelligent routing between accurate and fast models.
Now uses the unified ModelManager for better organization.

REFACTORED: Large functions have been broken down into modular components
in transcription_engine.py for better maintainability and testing.
"""
import traceback
import uuid
import audio_processing_utils
from model_manager import model_manager
from logging_config import get_logger
from transcription_engine import transcribe_with_model_refactored

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

    try:        # Load models using the new model manager
        logger.debug("🔧 [%s] Loading models...", transcription_id)

        # Check if models are loaded, if not, initialize them
        stats = model_manager.get_stats()
        if not stats["models_loaded"]:
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
    duration = audio_processing_utils.get_audio_duration_from_file(
        audio_file_path)

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
