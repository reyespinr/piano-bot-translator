"""
Audio transcription and translation utilities.

This module provides the main interface for transcription and translation
functionality, now using the unified ModelManager for better organization.
"""
from logging_config import get_logger

# Import the new model manager
from model_manager import model_manager
import transcription
import translation

logger = get_logger(__name__)

# Re-export the main functions for backward compatibility
transcribe = transcription.transcribe
translate = translation.translate
should_translate = translation.should_translate


async def warm_up_pipeline():
    """
    Warm up transcription models using the unified ModelManager.
    This is the new, organized way to warm up models.
    """
    logger.info("🔥 Starting unified model warm-up pipeline...")

    try:
        # Use the model manager's warm-up functionality
        success = await model_manager.warm_up_models()

        if success:
            logger.info("🎯 Enhanced dual model pipeline warm-up complete! Ready for smart routing.")
            stats = model_manager.get_stats()
            logger.info("📈 Accurate models: %d, Fast models: %d", 
                        stats["accurate_models"], stats["fast_models"])
            logger.info("🚀 Total VRAM models loaded: %d models ready for parallel processing",
                        stats["accurate_models"] + stats["fast_models"])
            return True

        logger.warning("⚠️ Model warm-up had some issues but models should still work")
        return False

    except Exception as e:
        logger.error("❌ Model warm-up failed: %s", str(e))
        logger.info("🔄 Models may still work without warm-up")
        return False
