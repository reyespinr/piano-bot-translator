"""
Core transcription functionality using Whisper models.

This module handles the actual transcription of audio files using the loaded
Whisper models, with intelligent routing between accurate and fast models.
"""
import asyncio
import gc
import string
import threading
import time
import traceback
import uuid
import torch
import models
import audio_utils
from audio_utils import COMMON_HALLUCINATIONS
from logging_config import get_logger

logger = get_logger(__name__)

# CRITICAL RESTORATION: Add back the common hallucinations filtering
# COMMON_HALLUCINATIONS moved to audio_utils.py


def should_use_fast_model(audio_file_path, current_queue_size=0, concurrent_requests=0, active_transcriptions=0):
    """Determine which model to use based on audio duration and system load"""

    # Get audio duration
    duration = audio_utils.get_audio_duration_from_file(audio_file_path)

    # Check if accurate model is currently busy
    accurate_busy = models.MODEL_USAGE_STATS["accurate_model_busy"]

    # Check how many fast models are available
    available_fast_models = models.count_available_fast_models()

    # DEBUG: Add detailed logging with pool information
    logger.debug("🔍 Routing decision: duration=%.1fs, accurate_busy=%s, concurrent=%d, active=%d, queue=%d, fast_available=%d/%d",
                 duration, accurate_busy, concurrent_requests, active_transcriptions, current_queue_size,
                 available_fast_models, models.MODEL_FAST_POOL_SIZE)

    # AGGRESSIVE ROUTING: If accurate model is busy, route to fast models
    if accurate_busy and duration < 3.0 and available_fast_models > 0:
        logger.info("🚀 Routing audio (%.1fs) to FAST model pool (accurate model busy, %d models available)",
                    duration, available_fast_models)
        models.MODEL_USAGE_STATS["fast_uses"] += 1
        return True

    # CONCURRENT PROCESSING: If we have multiple ACTIVE transcriptions and fast models available
    if active_transcriptions >= 1 and duration < 2.0 and available_fast_models > 0:
        logger.info("⚡ Concurrent processing (%d active transcriptions), routing short audio (%.1fs) to FAST model pool (%d available)",
                    active_transcriptions, duration, available_fast_models)
        models.MODEL_USAGE_STATS["fast_uses"] += 1
        return True

    # QUEUE-BASED ROUTING: High queue load - use fast models more aggressively
    if current_queue_size > 2 and duration < 2.5 and available_fast_models > 0:
        logger.info("🔥 Queue load (%d), routing audio (%.1fs) to FAST model pool (%d available)",
                    current_queue_size, duration, available_fast_models)
        models.MODEL_USAGE_STATS["fast_uses"] += 1
        models.MODEL_USAGE_STATS["queue_overflows"] += 1
        return True

    # LOAD BALANCING: If we have multiple fast models available, use them more often
    if available_fast_models >= 2 and duration < 3.0:
        logger.info("⚖️ Load balancing: routing audio (%.1fs) to FAST model pool (%d models available)",
                    duration, available_fast_models)
        models.MODEL_USAGE_STATS["fast_uses"] += 1
        return True

    # EVERY 3rd SHORT CLIP: Route every 3rd short audio to fast model for load balancing
    total_uses = models.MODEL_USAGE_STATS["accurate_uses"] + \
        models.MODEL_USAGE_STATS["fast_uses"]
    if duration < 1.5 and total_uses > 0 and total_uses % 3 == 0 and available_fast_models > 0:
        logger.info(
            "⚖️ Periodic load balancing: routing short audio (%.1fs) to FAST model pool (every 3rd)", duration)
        models.MODEL_USAGE_STATS["fast_uses"] += 1
        return True

    # Default: Use accurate model for best quality
    logger.info("🎯 Routing audio (%.1fs) to ACCURATE model (concurrent: %d, active: %d, queue: %d, fast_available: %d)",
                duration, concurrent_requests, active_transcriptions, current_queue_size, available_fast_models)
    models.MODEL_USAGE_STATS["accurate_uses"] += 1
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
        "vad_annotator.py"  # Any VAD-related errors
    ]

    return (any(error in error_str for error in recoverable_errors) or
            "tensor" in error_str.lower() or
            "NYI" in error_str or
            "TorchScript" in error_str)  # Catch TorchScript errors


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


async def transcribe_with_model(audio_file, model, model_name, model_index=None):
    """Enhanced transcribe function with comprehensive error handling and cleanup."""
    transcription_id = str(uuid.uuid4())[:8]

    if model_name == "fast":
        model_display_name = f"FAST-{model_index + 1}"
        model_lock = models.MODEL_USAGE_STATS["fast_model_locks"][model_index]
    else:
        model_display_name = model_name.upper()
        model_lock = models.MODEL_USAGE_STATS["accurate_model_lock"]

    # CRITICAL FIX: Ensure active_transcriptions is always an integer
    active_count = models.MODEL_USAGE_STATS["active_transcriptions"]
    if isinstance(active_count, set):
        active_count = len(active_count)
    elif not isinstance(active_count, int):
        active_count = 0

    logger.debug("🔄 [%s] Processing with %s model... (active: %d)",
                 transcription_id, model_display_name, active_count)

    try:
        logger.debug("🚀 [%s] Submitting %s model task to thread pool",
                     transcription_id, model_display_name)

        def transcribe_task():
            """Synchronous transcription task with comprehensive cleanup."""
            transcription_result = None
            lock_acquired = False

            try:
                logger.debug("🔒 [%s] Acquiring lock for %s model",
                             transcription_id, model_display_name)
                model_lock.acquire()
                lock_acquired = True
                logger.debug("🔓 [%s] Lock acquired for %s model",
                             transcription_id, model_display_name)

                logger.debug("🎤 [%s] Starting %s model transcription with enhanced settings",
                             transcription_id, model_display_name)

                # CRITICAL RESTORATION: Use enhanced transcription settings like the original
                transcription_result = model.transcribe(
                    audio_file,
                    vad=True,                  # Enable Voice Activity Detection
                    vad_threshold=0.35,        # VAD confidence threshold
                    no_speech_threshold=0.6,   # Filter non-speech sections
                    max_instant_words=0.3,     # Reduce hallucination words
                    suppress_silence=True,     # Use silence detection for better timestamps
                    only_voice_freq=True,      # Focus on human voice frequency range
                    word_timestamps=True       # Important for proper segmentation
                )

                logger.debug("✅ [%s] %s model transcription completed successfully",
                             transcription_id, model_display_name)

                return transcription_result

            except Exception as e:
                logger.error("❌ [%s] %s model transcription failed: %s",
                             transcription_id, model_display_name, str(e))
                return None
            finally:
                # CRITICAL FIX: Always release the lock with comprehensive error handling
                if lock_acquired:
                    try:
                        model_lock.release()
                        logger.debug("🔓 [%s] %s model lock released in finally block",
                                     transcription_id, model_display_name)
                    except Exception as lock_error:
                        logger.error("❌ [%s] Error releasing %s model lock: %s",
                                     transcription_id, model_display_name, str(lock_error))
                else:
                    logger.warning("⚠️ [%s] %s model lock was not acquired, skipping release",
                                   transcription_id, model_display_name)

        # Execute transcription in thread pool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, transcribe_task)

        logger.debug("🎯 [%s] %s model task completed from thread pool",
                     transcription_id, model_display_name)

        if result is None:
            logger.error("❌ [%s] %s model returned None result",
                         transcription_id, model_display_name)
            return None, None, None

        # CRITICAL RESTORATION: Extract text and language with proper handling
        try:
            if hasattr(result, 'text'):
                transcribed_text = getattr(result, 'text', '').strip()
                detected_language = getattr(result, 'language', 'unknown')
            elif isinstance(result, dict):
                transcribed_text = result.get('text', '').strip()
                detected_language = result.get('language', 'unknown')
            else:
                logger.warning("⚠️ [%s] Unknown result type: %s",
                               transcription_id, type(result))
                transcribed_text = str(result).strip() if result else ''
                detected_language = 'unknown'

        except Exception as result_error:
            logger.error("❌ [%s] Error extracting result data: %s",
                         transcription_id, str(result_error))
            return None, None, None

        # CRITICAL RESTORATION: Handle Austrian German misidentified as Icelandic
        if detected_language == "is":  # "is" is the language code for Icelandic
            logger.info("🇦🇹 [%s] Detected Icelandic - likely Austrian German. Re-transcribing as German...",
                        transcription_id)

            # Re-transcribe with German as forced language
            def retranscribe_task():
                try:
                    model_lock.acquire()
                    return model.transcribe(
                        audio_file,
                        vad=True,
                        vad_threshold=0.35,
                        no_speech_threshold=0.6,
                        max_instant_words=0.3,
                        suppress_silence=True,
                        only_voice_freq=True,
                        word_timestamps=True,
                        language="de"  # Force German language
                    )
                finally:
                    model_lock.release()

            loop = asyncio.get_event_loop()
            retry_result = await loop.run_in_executor(None, retranscribe_task)

            if retry_result:
                if hasattr(retry_result, 'text'):
                    transcribed_text = getattr(
                        retry_result, 'text', '').strip()
                elif isinstance(retry_result, dict):
                    transcribed_text = retry_result.get('text', '').strip()

                detected_language = "de"  # Override detected language to German
                result = retry_result  # Use the re-transcribed result for confidence analysis
                logger.info("🇩🇪 [%s] Re-transcribed as German: %s",
                            transcription_id, transcribed_text[:50])

        # CRITICAL RESTORATION: Apply confidence filtering
        confidence_passed, confidence_score = apply_confidence_filtering(
            result, transcribed_text, transcription_id)

        if not confidence_passed:
            logger.debug("🔇 [%s] Transcription filtered out due to low confidence: %.2f",
                         transcription_id, confidence_score)
            return "", detected_language, result

        logger.debug("✅ [%s] %s model completed transcription: text='%s' (length: %d), language='%s', confidence=%.2f",
                     transcription_id, model_display_name, transcribed_text,
                     len(transcribed_text), detected_language, confidence_score)

        # CRITICAL FIX: Always perform cleanup operations regardless of fast/accurate
        try:
            # CRITICAL FIX: Add comprehensive cleanup that ensures state is always reset
            logger.debug("🧹 [%s] %s model cleanup starting",
                         transcription_id, model_display_name)

            # CRITICAL FIX: For fast models, ensure usage stats are properly updated
            if model_name == "fast":
                try:
                    with models.MODEL_USAGE_STATS["stats_lock"]:
                        if model_index is not None and model_index < len(models.MODEL_USAGE_STATS["fast_model_usage"]):
                            models.MODEL_USAGE_STATS["fast_model_usage"][model_index] = max(0,
                                                                                            models.MODEL_USAGE_STATS["fast_model_usage"][model_index] - 1)
                            logger.debug("🔢 [%s] Fast model %d usage decremented to %d",
                                         transcription_id, model_index + 1,
                                         models.MODEL_USAGE_STATS["fast_model_usage"][model_index])
                except Exception as stats_error:
                    logger.error("❌ [%s] Error updating fast model stats: %s",
                                 transcription_id, str(stats_error))

            # CRITICAL FIX: Force a small delay to ensure all cleanup operations complete
            await asyncio.sleep(0.05)  # 50ms delay for cleanup completion

            logger.debug("🧹 [%s] %s model cleanup completed",
                         transcription_id, model_display_name)

        except Exception as cleanup_error:
            logger.error("❌ [%s] Error during %s model cleanup: %s",
                         transcription_id, model_display_name, str(cleanup_error))

        return transcribed_text, detected_language, result

    except Exception as e:
        logger.error("❌ [%s] Unexpected error in %s model processing: %s",
                     transcription_id, model_display_name, str(e))
        logger.error("❌ [%s] %s model error traceback: %s",
                     transcription_id, model_display_name, traceback.format_exc())
        return None, None, None


async def transcribe(audio_file, current_queue_size=0):
    """
    Enhanced transcribe function with bulletproof lock cleanup.

    CRITICAL FIX: Ensures that all transcriptions complete their callback chain
    by implementing comprehensive cleanup and state management.
    """
    transcription_id = str(uuid.uuid4())[:8]
    logger.debug("🎬 [%s] ENTRY: transcribe() called for file: %s",
                 transcription_id, audio_file)

    # CRITICAL FIX: Add comprehensive state tracking for debugging
    entry_stats = None
    cleanup_completed = False
    transcription_successful = False

    try:
        # CRITICAL FIX: Enhanced lock acquisition with better error handling
        max_lock_attempts = 5  # Increased from 3
        for attempt in range(max_lock_attempts):
            try:
                logger.debug(
                    "🔐 [%s] Attempting to acquire stats lock...", transcription_id)

                # CRITICAL FIX: Use a shorter timeout but more attempts
                # Reduced from 1.0 to 0.5
                if models.MODEL_USAGE_STATS["stats_lock"].acquire(timeout=0.5):
                    try:
                        models.MODEL_USAGE_STATS["concurrent_requests"] += 1
                        entry_stats = {
                            'concurrent': models.MODEL_USAGE_STATS["concurrent_requests"],
                            'active': models.MODEL_USAGE_STATS["active_transcriptions"]
                        }
                        logger.debug("🔢 [%s] Concurrent requests: %d, Active: %d (Lock acquired by: %s)",
                                     transcription_id, entry_stats['concurrent'], entry_stats['active'],
                                     threading.current_thread().name)
                    finally:
                        models.MODEL_USAGE_STATS["stats_lock"].release()
                        logger.debug(
                            "🔓 [%s] Stats lock released successfully", transcription_id)
                    break
                else:
                    logger.warning(
                        "⚠️ [%s] Stats lock busy on attempt %d, retrying...", transcription_id, attempt + 1)
                    # CRITICAL FIX: Exponential backoff with jitter
                    await asyncio.sleep(0.1 * (1.5 ** attempt) + random.uniform(0, 0.1))

            except Exception as lock_error:
                logger.error("❌ [%s] Lock acquisition error on attempt %d: %s",
                             transcription_id, attempt + 1, str(lock_error))
                await asyncio.sleep(0.1 * (attempt + 1))
        else:
            # All attempts failed
            logger.error("❌ [%s] Failed to acquire stats lock after %d attempts",
                         transcription_id, max_lock_attempts)

            # CRITICAL FIX: Emergency recovery - try emergency reset
            emergency_reset_successful = await models.emergency_lock_reset()
            if emergency_reset_successful:
                logger.warning(
                    "🔧 [%s] Emergency reset successful, retrying lock acquisition", transcription_id)
                try:
                    with models.MODEL_USAGE_STATS["stats_lock"]:
                        models.MODEL_USAGE_STATS["concurrent_requests"] += 1
                        entry_stats = {
                            'concurrent': models.MODEL_USAGE_STATS["concurrent_requests"],
                            'active': models.MODEL_USAGE_STATS["active_transcriptions"]
                        }
                    logger.info(
                        "✅ [%s] Lock acquired after emergency reset", transcription_id)
                except Exception as emergency_error:
                    logger.error("❌ [%s] Emergency reset also failed: %s",
                                 transcription_id, str(emergency_error))
                    raise RuntimeError(
                        f"Lock acquisition failed after emergency reset: {emergency_error}")
            else:
                raise RuntimeError(
                    "Failed to acquire stats lock and emergency reset failed")

        # Load models
        logger.debug("🔧 [%s] Loading models...", transcription_id)
        model_accurate, model_fast_pool = models.load_models_if_needed()

        # Determine model routing
        logger.debug("🎯 [%s] Determining model routing...", transcription_id)
        use_accurate, routing_reason = models.determine_model_routing(
            audio_file, current_queue_size, entry_stats)

        # Execute transcription based on routing decision
        if use_accurate:
            logger.debug("🎯 [%s] Using ACCURATE model...", transcription_id)
            transcribed_text, detected_language, result = await transcribe_with_model(
                audio_file, model_accurate, "accurate"
            )
        else:
            logger.debug("🚀 [%s] Using FAST model...", transcription_id)
            selected_model_index = models.select_fast_model()
            transcribed_text, detected_language, result = await transcribe_with_model(
                audio_file, model_fast_pool[selected_model_index], "fast", selected_model_index
            )

        # CRITICAL DEBUG: Add this log to see what we get back from transcription
        logger.debug("🚨 CRITICAL: [%s] transcribe_with_model returned: text='%s', language='%s'",
                     transcription_id, transcribed_text, detected_language)

        # CRITICAL RESTORATION: Extract confidence using the same method as your original code
        confidence = 0.0
        try:
            # Use the same confidence extraction logic as your original function
            if hasattr(result, "segments") and result.segments:
                # Get confidence values from segments (like your original code)
                confidences = []
                for segment in result.segments:
                    if hasattr(segment, "avg_logprob"):
                        confidences.append(segment.avg_logprob)
                
                # Apply your original confidence thresholds if we have confidence data
                if confidences:
                    confidence = sum(confidences) / len(confidences)
                    logger.info("Transcription confidence: %.2f, Language: %s",
                               confidence, detected_language)
                else:
                    logger.debug("[%s] No segment confidence data available", transcription_id)
            else:
                # Fallback: try to get confidence from result level
                if hasattr(result, 'avg_logprob'):
                    confidence = getattr(result, 'avg_logprob', 0.0)
                    logger.info("Transcription confidence: %.2f (result-level), Language: %s",
                               confidence, detected_language)
                elif isinstance(result, dict) and 'avg_logprob' in result:
                    confidence = result['avg_logprob']
                    logger.info("Transcription confidence: %.2f (dict), Language: %s",
                               confidence, detected_language)
                else:
                    logger.debug("[%s] No confidence information available from result type: %s",
                                transcription_id, type(result).__name__)

        except (AttributeError, TypeError) as e:
            logger.debug("[%s] Error extracting confidence: %s, using default", 
                        transcription_id, str(e))
            confidence = 0.0

        # Check transcription results
        if result is None or not transcribed_text:
            logger.warning("⚠️ [%s] Empty transcription result", transcription_id)
            return "", "unknown"

        # CRITICAL RESTORATION: Apply the same confidence threshold as your original
        if confidence < -1.0:
            logger.debug("🔇 [%s] Low confidence transcription (%.2f), returning empty", 
                        transcription_id, confidence)
            return "", detected_language

        # Mark transcription as successful
        transcription_successful = True
        logger.debug("🎉 [%s] SUCCESS: transcribe() returning: text='%s', lang=%s",
                     transcription_id, transcribed_text[:50] if transcribed_text else "None", detected_language)

        # CRITICAL FIX: Force a small delay before cleanup to ensure all async operations complete
        # 100ms delay to ensure callback chain readiness
        await asyncio.sleep(0.1)

        return transcribed_text.strip(), detected_language

    except Exception as e:
        logger.error("❌ [%s] Transcription error: %s",
                     transcription_id, str(e))
        logger.error("❌ [%s] Transcription traceback: %s",
                     transcription_id, traceback.format_exc())
        return "", "error"

    finally:
        # CRITICAL FIX: Enhanced cleanup with multiple retry attempts and bulletproof error handling
        logger.debug("🧹 [%s] Starting cleanup operations...", transcription_id)

        if entry_stats is not None:  # Only cleanup if we successfully incremented
            cleanup_max_attempts = 5  # Increased retry attempts
            for cleanup_attempt in range(cleanup_max_attempts):
                try:
                    logger.debug("🔐 [%s] Cleanup attempt %d: Trying to acquire stats lock...",
                                 transcription_id, cleanup_attempt + 1)

                    # CRITICAL FIX: Use shorter timeout but more attempts for cleanup
                    # Short timeout
                    if models.MODEL_USAGE_STATS["stats_lock"].acquire(timeout=0.3):
                        try:
                            if models.MODEL_USAGE_STATS["concurrent_requests"] > 0:
                                models.MODEL_USAGE_STATS["concurrent_requests"] -= 1
                                logger.debug("🔓 [%s] Successfully decremented concurrent requests (attempt %d) - now: %d",
                                             transcription_id, cleanup_attempt + 1, models.MODEL_USAGE_STATS["concurrent_requests"])
                                cleanup_completed = True
                            else:
                                logger.warning(
                                    "⚠️ [%s] Concurrent requests already at 0, skipping decrement", transcription_id)
                                cleanup_completed = True
                        finally:
                            models.MODEL_USAGE_STATS["stats_lock"].release()
                            logger.debug(
                                "🔓 [%s] Cleanup lock released successfully", transcription_id)

                        break  # Success, exit retry loop

                    else:
                        logger.warning("⚠️ [%s] Cleanup lock busy on attempt %d, retrying...",
                                       transcription_id, cleanup_attempt + 1)
                        # CRITICAL FIX: Shorter delays for cleanup retries
                        await asyncio.sleep(0.05 * (cleanup_attempt + 1))

                except Exception as cleanup_error:
                    logger.error("❌ [%s] Cleanup attempt %d failed: %s",
                                 transcription_id, cleanup_attempt + 1, str(cleanup_error))
                    await asyncio.sleep(0.05 * (cleanup_attempt + 1))

            if not cleanup_completed:
                logger.error("❌ [%s] CRITICAL: Cleanup failed after %d attempts - this may cause lock issues",
                             transcription_id, cleanup_max_attempts)
                # CRITICAL FIX: Force emergency reset if cleanup completely fails
                try:
                    logger.warning(
                        "🚨 [%s] Forcing emergency reset due to cleanup failure...", transcription_id)
                    await models.emergency_lock_reset()
                    logger.warning(
                        "✅ [%s] Emergency reset completed after cleanup failure", transcription_id)
                except Exception as emergency_error:
                    logger.error("💥 [%s] Emergency reset after cleanup failure also failed: %s",
                                 transcription_id, str(emergency_error))

        # CRITICAL FIX: Final status logging
        status = "SUCCESS" if transcription_successful else "FAILED"
        cleanup_status = "COMPLETED" if cleanup_completed else "FAILED"
        logger.debug("🧹 [%s] CLEANUP: transcribe() cleanup %s (transcription: %s)",
                     transcription_id, cleanup_status, status)


def apply_confidence_filtering(result, transcribed_text, transcription_id):
    """Apply confidence filtering to transcription results using the original logic.
    
    Returns:
        tuple: (passed_filter: bool, confidence_score: float)
    """
    try:
        # CRITICAL RESTORATION: Use the exact same logic as your original function
        confidences = []
        if hasattr(result, "segments") and result.segments:
            # Get confidence values from segments (exactly like your original)
            for segment in result.segments:
                if hasattr(segment, "avg_logprob"):
                    confidences.append(segment.avg_logprob)
        
        # Apply confidence filtering if we have confidence data (like your original)
        if confidences:
            avg_log_prob = sum(confidences) / len(confidences)
            
            # General confidence threshold (same as your original)
            confidence_threshold = -1.5
            if avg_log_prob < confidence_threshold:
                logger.debug("[%s] Low confidence transcription rejected (%.2f): '%s'",
                           transcription_id, avg_log_prob, transcribed_text)
                return False, avg_log_prob
            
            # CRITICAL RESTORATION: Special case for common hallucinations (same as original)
            text = transcribed_text.strip().lower()
            text_clean = text.translate(str.maketrans('', '', string.punctuation))
            
            if len(text_clean) < 15 and text_clean in COMMON_HALLUCINATIONS:
                # For these common short responses, require higher confidence (same as original)
                stricter_threshold = -0.5
                if avg_log_prob < stricter_threshold:
                    logger.debug("[%s] Short hallucination '%s' rejected with confidence %.2f",
                               transcription_id, text, avg_log_prob)
                    return False, avg_log_prob
            
            logger.debug("[%s] Transcription confidence: %.2f, passed filtering",
                        transcription_id, avg_log_prob)
            return True, avg_log_prob
        else:
            # No confidence data available - this shouldn't happen with proper Whisper results
            logger.warning("[%s] No confidence data available from segments - result type: %s",
                          transcription_id, type(result).__name__)
            
            # Try to get confidence from result object directly as fallback
            if hasattr(result, 'avg_logprob'):
                confidence = getattr(result, 'avg_logprob', 0.0)
                logger.debug("[%s] Using result-level confidence: %.2f", transcription_id, confidence)
                return confidence >= -1.5, confidence
            
            # Basic text quality filtering as last resort
            if not transcribed_text or len(transcribed_text.strip()) < 2:
                return False, 0.0
            
            # Check for obvious hallucinations without confidence
            text_clean = transcribed_text.strip().lower().translate(
                str.maketrans('', '', string.punctuation))
            
            if len(text_clean) < 10 and text_clean in COMMON_HALLUCINATIONS:
                logger.debug("[%s] Short hallucination '%s' filtered without confidence data",
                           transcription_id, text_clean)
                return False, 0.0
            
            logger.warning("[%s] No confidence data - accepting by default", transcription_id)
            return True, 0.0  # Default to accepting if no confidence data
            
    except Exception as e:
        logger.error("Error in confidence filtering: %s", str(e))
        return True, 0.0  # Default to accepting on error
