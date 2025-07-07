"""
Audio transcription and translation utilities.

This module provides the main interface for transcription and translation
functionality, now using the unified ModelManager for better organization.
"""
from logging_config import get_logger

# Import the faster-whisper model manager
from faster_whisper_manager import faster_whisper_model_manager
import transcription_service
import translation

logger = get_logger(__name__)

# Re-export the main functions for backward compatibility


async def transcribe(audio_file, current_queue_size=0, concurrent_requests=0, active_transcriptions=0):
    """
    Backward-compatible transcription function that returns only 2 values.

    This wrapper maintains API compatibility with the old system while using faster-whisper.
    """
    transcribed_text, detected_language, result = await transcription_service.transcribe(
        audio_file, current_queue_size, concurrent_requests, active_transcriptions
    )
    return transcribed_text, detected_language

translate = translation.translate
should_translate = translation.should_translate


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
