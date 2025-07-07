"""
Faster-Whisper Transcription Engine.

This module provides the faster-whisper implementation of transcription,
designed as a drop-in replacement for the stable-ts transcription engine.
"""
import asyncio
import gc
import string
import threading
import time
import traceback
import uuid
from typing import Tuple, Optional, Any
from dataclasses import dataclass
import torch
from faster_whisper import WhisperModel
from audio_processing_utils import COMMON_HALLUCINATIONS, is_recoverable_error, check_cuda_health, clear_cuda_cache
from model_manager import faster_whisper_model_manager
from logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class FasterWhisperTranscriptionRequest:
    """Encapsulates faster-whisper transcription request parameters."""
    audio_file: str
    model: WhisperModel
    model_name: str
    model_index: Optional[int] = None
    transcription_id: str = None
    force_language: Optional[str] = None  # NEW: Language override capability

    def __post_init__(self):
        if self.transcription_id is None:
            self.transcription_id = str(uuid.uuid4())[:8]


@dataclass
class FasterWhisperTranscriptionResult:
    """Encapsulates faster-whisper transcription results."""
    text: str
    language: str
    confidence_score: float
    segments: Any = None
    info: Any = None
    success: bool = True
    error_message: str = None


class FasterWhisperModelLockManager:
    """Manages faster-whisper model locks and display names."""

    @staticmethod
    def get_model_lock_and_name(model_name: str, model_index: Optional[int], transcription_id: str) -> Tuple[threading.RLock, str]:
        """Get the appropriate model lock and display name."""
        if model_name == "fast":
            model_display_name = f"FAST-{model_index + 1}"
            if model_index is not None and model_index < len(faster_whisper_model_manager.fast_tier.locks):
                model_lock = faster_whisper_model_manager.fast_tier.locks[model_index]
            else:
                logger.error("❌ [%s] Invalid fast model index: %d",
                             transcription_id, model_index)
                raise ValueError(f"Invalid fast model index: {model_index}")
        else:
            model_display_name = model_name.upper()
            if faster_whisper_model_manager.accurate_tier.locks:
                model_lock = faster_whisper_model_manager.accurate_tier.locks[0]
            else:
                logger.error(
                    "❌ [%s] No accurate model lock available", transcription_id)
                raise ValueError("No accurate model lock available")

        return model_lock, model_display_name


class FasterWhisperTranscriptionEngine:
    """Core faster-whisper transcription engine with retry logic."""

    def __init__(self, max_retries: int = 2):
        self.max_retries = max_retries

    def execute_transcription(self, request: FasterWhisperTranscriptionRequest, model_lock: threading.RLock,
                              model_display_name: str) -> Optional[Tuple[Any, Any]]:
        """Execute faster-whisper transcription with retry logic."""
        retry_count = 0

        while retry_count <= self.max_retries:
            try:
                return self._attempt_transcription(request, model_lock, model_display_name, retry_count)
            except Exception as e:
                error_str = str(e)
                logger.error("❌ [%s] %s model transcription failed (attempt %d): %s",
                             request.transcription_id, model_display_name, retry_count + 1, error_str)

                if self._should_retry(error_str, retry_count):
                    retry_count += 1
                    self._handle_retry(
                        request, model_lock, model_display_name, retry_count, error_str)
                    continue
                else:
                    # Check if this is a CUDA error and try recovery
                    if "CUDA error" in error_str:
                        logger.warning("🔥 [%s] CUDA error detected, attempting recovery: %s",
                                       request.transcription_id, error_str)
                        self._attempt_cuda_recovery()

                    logger.error("💥 [%s] Non-recoverable error or max retries exceeded: %s",
                                 request.transcription_id, error_str)
                    return None

        return None

    def _attempt_transcription(self, request: FasterWhisperTranscriptionRequest, model_lock: threading.RLock,
                               model_display_name: str, retry_count: int) -> Tuple[Any, Any]:
        """Single faster-whisper transcription attempt."""
        lock_acquired = False

        try:
            logger.debug("🔒 [%s] Acquiring lock for %s model (attempt %d)",
                         request.transcription_id, model_display_name, retry_count + 1)
            model_lock.acquire()
            lock_acquired = True
            logger.debug("🔓 [%s] Lock acquired for %s model",
                         request.transcription_id, model_display_name)

            logger.debug("🎤 [%s] Starting %s model transcription with faster-whisper (attempt %d)",
                         request.transcription_id, model_display_name, retry_count + 1)

            # Prepare transcription parameters
            transcribe_params = {
                "beam_size": 5,
                "best_of": 5,
                "temperature": 0.0,
                "condition_on_previous_text": False,
                "vad_filter": True,
                "vad_parameters": dict(min_silence_duration_ms=500)
            }

            # Add language parameter if forced
            if request.force_language:
                transcribe_params["language"] = request.force_language
                logger.debug("🌐 [%s] Forcing language: %s",
                             request.transcription_id, request.force_language)

            # Execute faster-whisper transcription
            segments, info = request.model.transcribe(
                request.audio_file,
                **transcribe_params
            )

            logger.debug("✅ [%s] %s model transcription completed successfully",
                         request.transcription_id, model_display_name)
            return segments, info

        finally:
            if lock_acquired:
                try:
                    model_lock.release()
                    logger.debug("🔓 [%s] %s model lock released in finally block",
                                 request.transcription_id, model_display_name)
                except Exception as lock_error:
                    logger.error("❌ [%s] Error releasing %s model lock: %s",
                                 request.transcription_id, model_display_name, str(lock_error))

    def _should_retry(self, error_str: str, retry_count: int) -> bool:
        """Determine if error is recoverable and retry should be attempted."""
        return retry_count < self.max_retries and is_recoverable_error(error_str)

    def _handle_retry(self, request: FasterWhisperTranscriptionRequest, model_lock: threading.RLock,
                      model_display_name: str, retry_count: int, error_str: str):
        """Handle retry logic including model state reset and backoff."""
        logger.warning("🔄 [%s] Recoverable error detected, retrying transcription (attempt %d/%d): %s",
                       request.transcription_id, retry_count, self.max_retries, error_str)

        # Reset model state if needed
        try:
            self._reset_model_state(request.model)
            logger.debug("🔄 [%s] Model state reset for retry",
                         request.transcription_id)
        except Exception as reset_error:
            logger.warning("⚠️ [%s] Failed to reset model state: %s",
                           request.transcription_id, str(reset_error))

        # Exponential backoff
        time.sleep(0.5 * retry_count)

    def _reset_model_state(self, model: WhisperModel):
        """Reset faster-whisper model internal state to recover from corrupted state."""
        try:
            # Faster-whisper uses CTranslate2, different cleanup approach
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            gc.collect()

        except Exception as e:
            logger.warning(
                "Failed to reset faster-whisper model state: %s", str(e))

    def _attempt_cuda_recovery(self):
        """Attempt to recover from CUDA errors by clearing cache and checking health."""
        try:
            logger.info("🔧 Attempting CUDA recovery...")

            # Check CUDA health first
            is_healthy, health_message = check_cuda_health()
            logger.info("🔍 CUDA health check: %s", health_message)

            if is_healthy:
                # Clear CUDA cache to free up memory
                clear_cuda_cache()
                logger.info("✅ CUDA recovery completed - cache cleared")
            else:
                logger.error(
                    "❌ CUDA recovery failed - GPU not healthy: %s", health_message)

        except Exception as e:
            logger.error("❌ CUDA recovery attempt failed: %s", str(e))


class FasterWhisperResultProcessor:
    """Processes and validates faster-whisper transcription results."""

    @staticmethod
    def extract_result_data(segments: Any, info: Any, transcription_id: str) -> Tuple[str, str]:
        """Extract text and language from faster-whisper results."""
        try:
            # Convert segments generator to list and extract text
            segment_list = list(segments)
            text = " ".join([segment.text for segment in segment_list]).strip()

            # Get detected language from info
            detected_language = info.language

            logger.debug("🎯 [%s] Extracted: text='%s' (length: %d), language='%s'",
                         transcription_id, text[:50] if text else "None",
                         len(text), detected_language)

            return text, detected_language

        except Exception as e:
            logger.error("❌ [%s] Error extracting faster-whisper result data: %s",
                         transcription_id, str(e))
            return "", "error"

    @staticmethod
    def apply_confidence_filtering(info: Any, transcribed_text: str, transcription_id: str) -> Tuple[bool, float]:
        """Apply confidence filtering to faster-whisper results."""
        try:
            # Faster-whisper provides language probability in info
            confidence_score = getattr(info, 'language_probability', 0.9)

            # Basic confidence threshold
            min_confidence = 0.3

            # Length-based filtering
            if len(transcribed_text.strip()) < 2:
                logger.debug("🔇 [%s] Text too short: '%s'",
                             transcription_id, transcribed_text)
                return False, confidence_score

            # Hallucination filtering
            text_lower = transcribed_text.lower().strip()
            if text_lower in COMMON_HALLUCINATIONS:
                logger.debug("🔇 [%s] Common hallucination detected: '%s'",
                             transcription_id, transcribed_text)
                return False, confidence_score

            # Confidence threshold
            if confidence_score < min_confidence:
                logger.debug("🔇 [%s] Low confidence: %.3f < %.3f",
                             transcription_id, confidence_score, min_confidence)
                return False, confidence_score

            return True, confidence_score

        except Exception as e:
            logger.warning(
                "⚠️ [%s] Error in confidence filtering: %s", transcription_id, str(e))
            return True, 0.5  # Default to accepting with medium confidence


class FasterWhisperModelCleanupManager:
    """Handles faster-whisper model cleanup and resource management."""

    @staticmethod
    def perform_cleanup(model_name: str, model_index: Optional[int], transcription_id: str, model_display_name: str):
        """Perform comprehensive cleanup operations."""
        try:
            logger.debug("🧹 [%s] %s model cleanup starting",
                         transcription_id, model_display_name)

            # For fast models, ensure usage stats are properly updated via model manager
            if model_name == "fast":
                try:
                    if model_index is not None:
                        faster_whisper_model_manager.release_fast_model(
                            model_index)
                        logger.debug("🔢 [%s] Fast model %d released via model manager",
                                     transcription_id, model_index + 1)
                except Exception as stats_error:
                    logger.error("❌ [%s] Error releasing fast model via model manager: %s",
                                 transcription_id, str(stats_error))

            # Force a small delay to ensure all cleanup operations complete
            time.sleep(0.05)  # 50ms delay for cleanup completion

            logger.debug("🧹 [%s] %s model cleanup completed",
                         transcription_id, model_display_name)

        except Exception as cleanup_error:
            logger.error("❌ [%s] Error during %s model cleanup: %s",
                         transcription_id, model_display_name, str(cleanup_error))


async def faster_whisper_transcribe_with_model(audio_file: str, model: WhisperModel, model_name: str,
                                               model_index: Optional[int] = None,
                                               force_language: Optional[str] = None) -> Tuple[Optional[str], Optional[str], Optional[Any]]:
    """
    Faster-whisper transcription function with modular components.

    This is the faster-whisper implementation that replaces the stable-ts version.
    """
    # Create transcription request
    request = FasterWhisperTranscriptionRequest(
        audio_file, model, model_name, model_index, force_language=force_language)

    logger.debug("🔄 [%s] Processing with faster-whisper %s model...",
                 request.transcription_id, f"{model_name.upper()}-{model_index + 1}" if model_index is not None else model_name.upper())

    try:
        # Get model lock and display name
        model_lock, model_display_name = FasterWhisperModelLockManager.get_model_lock_and_name(
            model_name, model_index, request.transcription_id)

        # Execute transcription with retry logic
        logger.debug("🚀 [%s] Submitting %s model task to thread pool",
                     request.transcription_id, model_display_name)

        transcription_engine = FasterWhisperTranscriptionEngine()

        def transcribe_task():
            return transcription_engine.execute_transcription(request, model_lock, model_display_name)

        # Execute transcription in thread pool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, transcribe_task)

        logger.debug("🎯 [%s] %s model task completed from thread pool",
                     request.transcription_id, model_display_name)

        if result is None:
            logger.error("❌ [%s] %s model returned None result",
                         request.transcription_id, model_display_name)
            return None, None, None

        segments, info = result

        # Extract text and language from result
        transcribed_text, detected_language = FasterWhisperResultProcessor.extract_result_data(
            segments, info, request.transcription_id)

        # Apply confidence filtering
        confidence_passed, confidence_score = FasterWhisperResultProcessor.apply_confidence_filtering(
            info, transcribed_text, request.transcription_id)

        if not confidence_passed:
            logger.debug("🔇 [%s] Transcription filtered out due to low confidence: %.2f",
                         request.transcription_id, confidence_score)
            return "", detected_language, info

        logger.info("✅ [%s] %s model completed: text='%s' (length: %d), language='%s', confidence=%.2f",
                    request.transcription_id, model_display_name, transcribed_text,
                    len(transcribed_text), detected_language, confidence_score)

        return transcribed_text, detected_language, info

    except Exception as e:
        logger.error("❌ [%s] Unexpected error in %s model processing: %s",
                     request.transcription_id, model_display_name if 'model_display_name' in locals() else 'unknown', str(e))
        logger.error("❌ [%s] %s model error traceback: %s",
                     request.transcription_id, model_display_name if 'model_display_name' in locals() else 'unknown', traceback.format_exc())
        return None, None, None

    finally:
        # Always perform cleanup operations
        if 'model_display_name' in locals():
            FasterWhisperModelCleanupManager.perform_cleanup(
                model_name, model_index, request.transcription_id, model_display_name)
