"""
Audio transcription and translation utilities.

This module provides the main interface for transcription and translation
functionality, coordinating between the specialized modules.
"""
import asyncio
import os
import uuid
from logging_config import get_logger

# Change relative imports to absolute imports
import audio_utils
import models
import transcription
import translation

logger = get_logger(__name__)

# Re-export the main functions for backward compatibility
transcribe = transcription.transcribe
translate = translation.translate
should_translate = translation.should_translate
_load_models_if_needed = models.load_models_if_needed


async def safe_warmup_transcribe(audio_file, model, model_name, model_index=None):
    """Safe transcription for warmup with timeout and error handling."""
    try:
        if model_name == "fast":
            model_display_name = f"FAST-{model_index + 1}"
            transcription_id = str(uuid.uuid4())[:8]
            model_lock = models.MODEL_USAGE_STATS["fast_model_locks"][model_index]

            def warmup_transcribe():
                with model_lock:
                    return model.transcribe(audio_file, language="en", word_timestamps=False)

            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, warmup_transcribe)

            logger.debug(
                "Warmup transcription completed for %s model", model_display_name)
            return result is not None

        else:
            # Accurate model warmup - use normal transcribe_with_model
            transcribed_text, detected_language, result = await transcription.transcribe_with_model(
                audio_file, model, model_name, model_index
            )

            logger.debug(
                "Warmup transcription completed for %s model", model_name.upper())
            return result is not None

    except asyncio.TimeoutError:
        logger.warning("Warmup transcription timed out for %s model - continuing anyway",
                       model_name.upper() + (f"-{model_index + 1}" if model_index is not None else ""))
        return False
    except Exception as e:
        logger.warning("Warmup transcription failed for %s model: %s - continuing anyway",
                       model_name.upper() + (f"-{model_index + 1}" if model_index is not None else ""), str(e))
        return False


async def warm_up_pipeline():
    """Warm up both transcription models for optimal performance."""
    logger.info(
        "Warming up DUAL MODEL transcription pipeline with fast model pool...")
    dummy_files = []

    try:
        logger.info("Loading accurate model and fast model pool for warm-up...")
        model_accurate, model_fast_pool = models.load_models_if_needed()

        # Create a dummy audio file for the accurate model
        accurate_dummy_file = audio_utils.create_dummy_audio_file(
            "warmup_accurate.wav")
        dummy_files.append(accurate_dummy_file)
        logger.debug(
            "Created warmup audio file for accurate model: %s", accurate_dummy_file)

        # Warm up accurate model
        logger.info("Warming up accurate model...")
        accurate_success = await safe_warmup_transcribe(accurate_dummy_file, model_accurate, "accurate")

        if accurate_success:
            logger.info("✅ Accurate model warmed up successfully")
        else:
            logger.warning(
                "⚠️ Accurate model warmup had issues but continuing")

        # Warm up all fast models in the pool
        logger.info("Warming up fast model pool...")
        fast_successes = 0

        warmup_tasks = []
        for i, model_fast in enumerate(model_fast_pool):
            logger.info("Warming up fast model %d/%d...",
                        i+1, len(model_fast_pool))

            fast_dummy_file = audio_utils.create_dummy_audio_file(
                f"warmup_fast_{i+1}.wav")
            dummy_files.append(fast_dummy_file)
            logger.debug(
                "Created warmup audio file for fast model %d: %s", i+1, fast_dummy_file)

            warmup_task = safe_warmup_transcribe(
                fast_dummy_file, model_fast, "fast", i)
            warmup_tasks.append((warmup_task, i+1))

        # Wait for all fast model warmups to complete
        for warmup_task, model_num in warmup_tasks:
            try:
                success = await warmup_task
                if success:
                    fast_successes += 1
                    logger.info(
                        "✅ Fast model %d warmed up successfully", model_num)
                else:
                    logger.warning(
                        "⚠️ Fast model %d warmup had issues", model_num)
            except Exception as e:
                logger.warning(
                    "⚠️ Fast model %d warmup failed: %s", model_num, str(e))

        # Report results
        if fast_successes == len(model_fast_pool):
            logger.info("✅ All fast models warmed up successfully")
        else:
            logger.warning("⚠️ %d/%d fast models warmed up successfully",
                           fast_successes, len(model_fast_pool))

        logger.info(
            "🎯 Enhanced dual model pipeline warm-up complete! Ready for smart routing.")
        logger.info("📈 Accurate model: %s, Fast model pool: %d x %s",
                    models.MODEL_ACCURATE_NAME, models.MODEL_FAST_POOL_SIZE, models.MODEL_FAST_NAME)
        logger.info(
            "🚀 Total VRAM models loaded: %d models ready for parallel processing", 1 + models.MODEL_FAST_POOL_SIZE)

    except Exception as e:
        logger.error("Warmup error: %s", str(e))
        logger.info(
            "🔄 Models loaded but warmup incomplete - system will work normally")
    finally:
        # Clean up all dummy files
        for dummy_file in dummy_files:
            try:
                if os.path.exists(dummy_file):
                    os.remove(dummy_file)
                    logger.debug("Cleaned up warmup file: %s", dummy_file)
            except Exception as e:
                logger.debug(
                    "Failed to clean up warmup file %s: %s", dummy_file, str(e))
