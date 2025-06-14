"""
Core transcription engine with modular, testable components.

This module contains the refactored transcription logic, broken down into
smaller, focused functions for better maintainability and testing.
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

from model_manager import model_manager
from audio_processing_utils import COMMON_HALLUCINATIONS
from logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class TranscriptionRequest:
    """Encapsulates transcription request parameters."""
    audio_file: str
    model: Any
    model_name: str
    model_index: Optional[int] = None
    transcription_id: str = None

    def __post_init__(self):
        if self.transcription_id is None:
            self.transcription_id = str(uuid.uuid4())[:8]


@dataclass
class TranscriptionResult:
    """Encapsulates transcription results."""
    text: str
    language: str
    confidence_score: float
    raw_result: Any = None
    success: bool = True
    error_message: str = None


class ModelLockManager:
    """Manages model locks and display names."""

    @staticmethod
    def get_model_lock_and_name(model_name: str, model_index: Optional[int], transcription_id: str) -> Tuple[threading.RLock, str]:
        """Get the appropriate model lock and display name."""
        if model_name == "fast":
            model_display_name = f"FAST-{model_index + 1}"
            if model_index is not None and model_index < len(model_manager.fast_tier.locks):
                model_lock = model_manager.fast_tier.locks[model_index]
            else:
                logger.error("❌ [%s] Invalid fast model index: %d",
                             transcription_id, model_index)
                raise ValueError(f"Invalid fast model index: {model_index}")
        else:
            model_display_name = model_name.upper()
            if model_manager.accurate_tier.locks:
                model_lock = model_manager.accurate_tier.locks[0]
            else:
                logger.error(
                    "❌ [%s] No accurate model lock available", transcription_id)
                raise ValueError("No accurate model lock available")

        return model_lock, model_display_name


class TranscriptionEngine:
    """Core transcription engine with retry logic."""

    def __init__(self, max_retries: int = 2):
        self.max_retries = max_retries

    def execute_transcription(self, request: TranscriptionRequest, model_lock: threading.RLock,
                              model_display_name: str) -> Optional[Any]:
        """Execute transcription with retry logic."""
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
                    logger.error("💥 [%s] Non-recoverable error or max retries exceeded: %s",
                                 request.transcription_id, error_str)
                    return None

        return None

    def _attempt_transcription(self, request: TranscriptionRequest, model_lock: threading.RLock,
                               model_display_name: str, retry_count: int) -> Any:
        """Single transcription attempt."""
        lock_acquired = False

        try:
            logger.debug("🔒 [%s] Acquiring lock for %s model (attempt %d)",
                         request.transcription_id, model_display_name, retry_count + 1)
            model_lock.acquire()
            lock_acquired = True
            logger.debug("🔓 [%s] Lock acquired for %s model",
                         request.transcription_id, model_display_name)

            logger.debug("🎤 [%s] Starting %s model transcription with enhanced settings (attempt %d)",
                         request.transcription_id, model_display_name, retry_count + 1)

            # Execute transcription with enhanced settings
            result = request.model.transcribe(
                request.audio_file,
                vad=True,                  # Enable Voice Activity Detection
                vad_threshold=0.35,        # VAD confidence threshold
                no_speech_threshold=0.6,   # Filter non-speech sections
                max_instant_words=0.3,     # Reduce hallucination words
                suppress_silence=True,     # Use silence detection for better timestamps
                only_voice_freq=True,      # Focus on human voice frequency range
                word_timestamps=True       # Important for proper segmentation
            )

            logger.debug("✅ [%s] %s model transcription completed successfully",
                         request.transcription_id, model_display_name)
            return result

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
        return retry_count < self.max_retries and self._is_recoverable_error(error_str)

    def _is_recoverable_error(self, error_str: str) -> bool:
        """Check if the error is a recoverable PyTorch/FFmpeg error."""
        recoverable_errors = [
            "ffmpeg",
            "Invalid audio format",
            "Audio file could not be read",
            "Temporary failure",
            "Resource temporarily unavailable",
            "cuda out of memory",
            "RuntimeError: CUDA",
            "torch.cuda.OutOfMemoryError",
            "RuntimeError: NYI",  # PyTorch TorchScript "Not Yet Implemented" errors
            "TorchScript interpreter",  # General TorchScript errors
            "vad/model/vad_annotator",  # VAD model specific errors
            # TorchScript operation failures
            "RuntimeError: The following operation failed in the TorchScript"
        ]
        return any(recoverable in error_str.lower() for recoverable in recoverable_errors)

    def _handle_retry(self, request: TranscriptionRequest, model_lock: threading.RLock,
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

    def _reset_model_state(self, model):
        """Reset model internal state to recover from corrupted state."""
        try:
            if hasattr(model, 'model') and hasattr(model.model, 'decoder'):
                model.model.decoder.reset_cache()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            gc.collect()

        except Exception as e:
            logger.warning("Failed to reset model state: %s", str(e))


class ResultProcessor:
    """Processes and validates transcription results."""

    @staticmethod
    def extract_result_data(raw_result: Any, transcription_id: str) -> Tuple[str, str]:
        """Extract text and language from raw transcription result."""
        try:
            if hasattr(raw_result, 'text'):
                transcribed_text = getattr(raw_result, 'text', '').strip()
                detected_language = getattr(raw_result, 'language', 'unknown')
            elif isinstance(raw_result, dict):
                transcribed_text = raw_result.get('text', '').strip()
                detected_language = raw_result.get('language', 'unknown')
            else:
                logger.warning("⚠️ [%s] Unknown result type: %s",
                               transcription_id, type(raw_result))
                transcribed_text = str(
                    raw_result).strip() if raw_result else ''
                detected_language = 'unknown'

            return transcribed_text, detected_language

        except Exception as result_error:
            logger.error("❌ [%s] Error extracting result data: %s",
                         transcription_id, str(result_error))
            raise

    @staticmethod
    async def handle_icelandic_detection(raw_result: Any, transcribed_text: str, detected_language: str,
                                         request: TranscriptionRequest, model_lock: threading.RLock) -> Tuple[str, str, Any]:
        """Handle Austrian German misidentified as Icelandic."""
        if detected_language == "is":  # "is" is the language code for Icelandic
            logger.info("🇦🇹 [%s] Detected Icelandic - likely Austrian German. Re-transcribing as German...",
                        request.transcription_id)

            # Re-transcribe with German as forced language
            def retranscribe_task():
                try:
                    model_lock.acquire()
                    return request.model.transcribe(
                        request.audio_file,
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
                raw_result = retry_result  # Use the re-transcribed result for confidence analysis
                logger.info("🇩🇪 [%s] Re-transcribed as German: %s",
                            request.transcription_id, transcribed_text[:50])

        return transcribed_text, detected_language, raw_result


class ConfidenceFilter:
    """Handles confidence filtering and validation."""

    @staticmethod
    def apply_confidence_filtering(result: Any, transcribed_text: str, transcription_id: str) -> Tuple[bool, float]:
        """Apply confidence filtering to transcription results."""
        try:
            # Get confidence values from segments
            confidences = []
            if hasattr(result, "segments") and result.segments:
                for segment in result.segments:
                    if hasattr(segment, "avg_logprob"):
                        confidences.append(segment.avg_logprob)

            if confidences:
                avg_log_prob = sum(confidences) / len(confidences)

                # General confidence threshold
                confidence_threshold = -1.5
                if avg_log_prob < confidence_threshold:
                    logger.info("❌ [%s] Low confidence transcription rejected (%.2f): '%s'",
                                transcription_id, avg_log_prob, transcribed_text)
                    return False, avg_log_prob

                # Special case for common hallucinations
                text = transcribed_text.strip().lower()
                text_clean = text.translate(
                    str.maketrans('', '', string.punctuation))

                if len(text_clean) < 15 and text_clean in COMMON_HALLUCINATIONS:
                    # For these common short responses, require higher confidence
                    stricter_threshold = -0.5
                    if avg_log_prob < stricter_threshold:
                        logger.info("❌ [%s] Short hallucination '%s' rejected with confidence %.2f",
                                    transcription_id, text_clean, avg_log_prob)
                        return False, avg_log_prob

                logger.info("✅ [%s] Transcription confidence: %.2f, passed filtering",
                            transcription_id, avg_log_prob)
                return True, avg_log_prob
            else:
                # No confidence data available - basic text quality filtering
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


class ModelCleanupManager:
    """Handles model cleanup and resource management."""

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
                        model_manager.release_fast_model(model_index)
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


async def transcribe_with_model_refactored(audio_file: str, model: Any, model_name: str,
                                           model_index: Optional[int] = None) -> Tuple[Optional[str], Optional[str], Optional[Any]]:
    """
    Refactored transcription function with modular components.

    This is the new, clean implementation that replaces the massive original function.
    """
    # Create transcription request
    request = TranscriptionRequest(audio_file, model, model_name, model_index)

    logger.debug("🔄 [%s] Processing with %s model...",
                 request.transcription_id, f"{model_name.upper()}-{model_index + 1}" if model_index is not None else model_name.upper())

    try:
        # Get model lock and display name
        model_lock, model_display_name = ModelLockManager.get_model_lock_and_name(
            model_name, model_index, request.transcription_id)

        # Execute transcription with retry logic
        logger.debug("🚀 [%s] Submitting %s model task to thread pool",
                     request.transcription_id, model_display_name)

        transcription_engine = TranscriptionEngine()

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

        # Extract text and language from result
        transcribed_text, detected_language = ResultProcessor.extract_result_data(
            result, request.transcription_id)        # Handle Austrian German misidentified as Icelandic
        transcribed_text, detected_language, result = await ResultProcessor.handle_icelandic_detection(
            result, transcribed_text, detected_language, request, model_lock)

        # Apply confidence filtering
        confidence_passed, confidence_score = ConfidenceFilter.apply_confidence_filtering(
            result, transcribed_text, request.transcription_id)

        if not confidence_passed:
            logger.debug("🔇 [%s] Transcription filtered out due to low confidence: %.2f",
                         request.transcription_id, confidence_score)
            return "", detected_language, result

        logger.info("✅ [%s] %s model completed: text='%s' (length: %d), language='%s', confidence=%.2f",
                    request.transcription_id, model_display_name, transcribed_text,
                    len(transcribed_text), detected_language, confidence_score)

        return transcribed_text, detected_language, result

    except Exception as e:
        logger.error("❌ [%s] Unexpected error in %s model processing: %s",
                     request.transcription_id, model_display_name if 'model_display_name' in locals() else 'unknown', str(e))
        logger.error("❌ [%s] %s model error traceback: %s",
                     request.transcription_id, model_display_name if 'model_display_name' in locals() else 'unknown', traceback.format_exc())
        return None, None, None

    finally:
        # Always perform cleanup operations
        if 'model_display_name' in locals():
            ModelCleanupManager.perform_cleanup(
                model_name, model_index, request.transcription_id, model_display_name)
