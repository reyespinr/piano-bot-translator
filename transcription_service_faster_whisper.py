"""
Core transcription functionality using faster-whisper models.

This module provides a simplified wrapper around faster-whisper to maintain
compatibility with the existing Discord bot while using the new faster-whisper backend.
"""
import traceback
import uuid
import audio_processing_utils
from faster_whisper_manager import faster_whisper_model_manager
from faster_whisper_service import faster_whisper_transcribe
from logging_config import get_logger

logger = get_logger(__name__)


async def transcribe(audio_file, current_queue_size=0, concurrent_requests=0, active_transcriptions=0):
    """
    Main transcription function with faster-whisper backend.

    This is a simplified wrapper that maintains API compatibility but uses faster-whisper.

    Args:
        audio_file: Path to audio file to transcribe
        current_queue_size: Current queue size (for compatibility, not used)
        concurrent_requests: Number of concurrent requests (for compatibility, not used)
        active_transcriptions: Number of active transcriptions (for compatibility, not used)

    Returns:
        tuple: (transcribed_text, detected_language, result_dict)
    """
    session_id = str(uuid.uuid4())[:8]
    logger.info("🎯 [%s] Starting transcription for: %s",
                session_id, audio_file)

    try:
        # Get stats before transcription
        stats = faster_whisper_model_manager.get_stats()

        # Check if models are initialized
        if not stats.get("models_loaded", False):
            logger.warning(
                "Models not initialized, attempting to initialize...")
            success = await faster_whisper_model_manager.initialize_models(warm_up=False)
            if not success:
                raise Exception("Failed to initialize faster-whisper models")

        # Get audio duration for logging
        duration = audio_processing_utils.get_audio_duration_from_file(
            audio_file)
        logger.info("🎵 [%s] Audio duration: %.2fs", session_id, duration)

        # Transcribe using faster-whisper
        transcribed_text, detected_language = await faster_whisper_transcribe(audio_file)

        # Create result dict for compatibility
        result = {
            "text": transcribed_text,
            "language": detected_language,
            "session_id": session_id,
            "audio_duration": duration,
            "model_type": "faster-whisper"
        }

        logger.info("✅ [%s] Transcription completed: '%s' (lang: %s)",
                    session_id, transcribed_text[:50], detected_language)

        return transcribed_text, detected_language, result

    except Exception as e:
        logger.error("❌ [%s] Transcription failed: %s", session_id, str(e))
        logger.error("❌ [%s] Traceback: %s",
                     session_id, traceback.format_exc())

        # Return empty result on failure
        return "", "unknown", {
            "text": "",
            "language": "unknown",
            "session_id": session_id,
            "error": str(e),
            "model_type": "faster-whisper"
        }
