"""
Audio transcription and translation utilities.

This module provides the main interface for transcription and translation
functionality, now using the unified ModelManager for better organization.
"""
import asyncio
import os
import uuid
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
            logger.info(
                "🎯 Enhanced dual model pipeline warm-up complete! Ready for smart routing.")
            stats = model_manager.get_stats()
            logger.info("📈 Accurate models: %d x %s, Fast models: %d x %s",
                        stats["accurate_models"], model_manager.accurate_config.name,
                        stats["fast_models"], model_manager.fast_config.name)
            logger.info("🚀 Total VRAM models loaded: %d models ready for parallel processing",
                        stats["accurate_models"] + stats["fast_models"])
            return True
        else:
            logger.warning(
                "⚠️ Model warm-up had some issues but models should still work")
            return False

    except Exception as e:
        logger.error("❌ Model warm-up failed: %s", str(e))
        logger.info("🔄 Models may still work without warm-up")
        return False


# Backward compatibility function (deprecated)
async def safe_warmup_transcribe(audio_file, model, model_name, model_index=None):
    """
    DEPRECATED: Use model_manager.warm_up_models() instead.
    Safe transcription for warmup with timeout and error handling.
    """
    logger.warning(
        "safe_warmup_transcribe() is deprecated. Use model_manager.warm_up_models() instead.")

    try:
        if model_name == "fast":
            model_display_name = f"FAST-{model_index + 1}"
            transcription_id = str(uuid.uuid4())[:8]

            # Get the lock from model manager
            if model_index < len(model_manager.fast_tier.locks):
                model_lock = model_manager.fast_tier.locks[model_index]
            else:
                logger.error("Invalid model index: %d", model_index)
                return False

            def warmup_transcribe():
                with model_lock:
                    return model.transcribe(audio_file, language="en", word_timestamps=False)

            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, warmup_transcribe)

            logger.debug(
                "Warmup transcription completed for %s model", model_display_name)
            return result is not None

        # Accurate model warmup - use normal transcribe_with_model
        transcribed_text, detected_language, result = await transcription.transcribe_with_model(
            audio_file, model, model_name, model_index
        )

        logger.debug("Warmup transcription completed for %s model",
                     str(model_name).upper())
        return result is not None

    except asyncio.TimeoutError:
        logger.warning("Warmup transcription timed out for %s model - continuing anyway",
                       model_name.upper() + (f"-{model_index + 1}" if model_index is not None else ""))
        return False
    except Exception as e:
        logger.warning("Warmup transcription failed for %s model: %s - continuing anyway",
                       model_name.upper() + (f"-{model_index + 1}" if model_index is not None else ""), str(e))
        return False


# Legacy function for backward compatibility (deprecated)
def _load_models_if_needed():
    """
    DEPRECATED: Use model_manager.initialize_models() instead.
    Legacy function for backward compatibility.
    """
    logger.warning(
        "_load_models_if_needed() is deprecated. Use model_manager.initialize_models() instead.")

    if not model_manager.stats["models_loaded"]:
        logger.warning(
            "Models not loaded via ModelManager! This should be done at startup.")
        # Try to load synchronously as fallback
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                logger.error(
                    "Cannot load models synchronously in async context!")
                return None, []
            else:
                success = loop.run_until_complete(
                    model_manager.initialize_models(warm_up=False))
                if not success:
                    return None, []
        except Exception as e:
            logger.error("Failed to load models: %s", str(e))
            return None, []

    return model_manager.get_models()
