"""
Model management module for audio transcription.

This module provides backward compatibility with the new unified ModelManager.
All model loading and management is now handled by model_manager.py for better organization.
"""
import warnings
from model_manager import (
    model_manager,
    load_models_if_needed,
    select_fast_model,
    get_available_fast_model,
    release_fast_model,
    count_available_fast_models,
    MODEL_USAGE_STATS,
    MODEL_ACCURATE_NAME,
    MODEL_FAST_NAME,
    MODEL_FAST_POOL_SIZE,
    update_legacy_stats
)
from logging_config import get_logger

logger = get_logger(__name__)

# Deprecation warning for direct imports
warnings.warn(
    "Direct imports from models.py are deprecated. Use model_manager.py for new code.",
    DeprecationWarning,
    stacklevel=2
)

logger.info("🔧 Model Configuration (via ModelManager): Fast pool size = %d",
            MODEL_FAST_POOL_SIZE)


# Backward compatibility functions - these now delegate to ModelManager
def determine_model_routing(audio_file, current_queue_size=0, entry_stats=None):
    """
    Determine which model to use based on audio duration and system load.

    Returns:
        tuple: (use_accurate: bool, routing_reason: str)
    """
    import os

    try:
        # Calculate audio duration from file size (approximate)
        file_size = os.path.getsize(audio_file)
        # Rough estimation: 48kHz * 2 channels * 2 bytes = 192KB per second
        duration_seconds = file_size / 192000.0

        # Get current system state from model manager
        stats = model_manager.get_stats()
        concurrent = entry_stats['concurrent'] if entry_stats else stats["concurrent_requests"]
        active = entry_stats['active'] if entry_stats else stats["active_transcriptions"]
        fast_available = stats["fast_available"]
        accurate_busy = stats["accurate_busy"]

        logger.debug("🔍 Routing decision: duration=%.1fs, accurate_busy=%s, concurrent=%d, active=%d, queue=%d, fast_available=%d/%d",
                     duration_seconds, accurate_busy, concurrent, active, current_queue_size, fast_available, stats["fast_models"])

        # Routing logic
        if duration_seconds >= 4.0:
            # Long audio - prefer accurate model
            if not accurate_busy:
                logger.info("🎯 Routing audio (%.1fs) to ACCURATE model (concurrent: %d, active: %d, queue: %d, fast_available: %d)",
                            duration_seconds, concurrent, active, current_queue_size, fast_available)
                return True, f"Long audio ({duration_seconds:.1f}s) routed to accurate model"
            elif fast_available > 0:
                logger.info("⚖️ Accurate model busy, routing long audio (%.1fs) to FAST model pool (%d models available)",
                            duration_seconds, fast_available)
                return False, f"Accurate busy, long audio routed to fast pool"
            else:
                logger.info("🎯 All fast models busy, waiting for ACCURATE model for long audio (%.1fs)",
                            duration_seconds)
                return True, f"All fast busy, waiting for accurate model"

        elif duration_seconds >= 2.5:
            # Medium audio - prefer accurate if not busy, otherwise fast
            if not accurate_busy and current_queue_size < 3:
                logger.info("🎯 Routing audio (%.1fs) to ACCURATE model (concurrent: %d, active: %d, queue: %d, fast_available: %d)",
                            duration_seconds, concurrent, active, current_queue_size, fast_available)
                return True, f"Medium audio ({duration_seconds:.1f}s) routed to accurate model"
            elif fast_available > 0:
                logger.info("⚖️ Load balancing: routing audio (%.1fs) to FAST model pool (%d models available)",
                            duration_seconds, fast_available)
                return False, f"Medium audio load-balanced to fast pool"
            else:
                logger.info("🎯 Routing audio (%.1fs) to ACCURATE model (concurrent: %d, active: %d, queue: %d, fast_available: %d)",
                            duration_seconds, concurrent, active, current_queue_size, fast_available)
                return True, f"No fast models available, using accurate"

        else:
            # Short audio - prefer fast models
            if fast_available > 0:
                logger.info("⚖️ Load balancing: routing audio (%.1fs) to FAST model pool (%d models available)",
                            duration_seconds, fast_available)
                return False, f"Short audio ({duration_seconds:.1f}s) routed to fast pool"
            elif not accurate_busy:
                logger.info("🎯 No fast models available, routing short audio (%.1fs) to ACCURATE model",
                            duration_seconds)
                return True, f"No fast models available, using accurate"
            else:
                logger.info("⚖️ All models busy, routing audio (%.1fs) to FAST model pool (will wait)",
                            duration_seconds)
                return False, f"All models busy, defaulting to fast pool"

    except Exception as e:
        logger.error("❌ Error in model routing: %s", str(e))
        # Default to fast model on error
        return False, f"Error in routing, defaulting to fast model: {str(e)}"


# Legacy functions that are no longer needed but kept for compatibility
def debug_lock_state():
    """Debug function - deprecated, functionality moved to ModelManager."""
    logger.warning(
        "debug_lock_state() is deprecated. Use model_manager.get_stats() instead.")
    return False


async def emergency_lock_reset():
    """Emergency function - deprecated, functionality moved to ModelManager."""
    logger.warning(
        "emergency_lock_reset() is deprecated. Use model_manager directly.")
    return False


# Make sure legacy stats are updated when accessed
def _update_legacy_stats_if_needed():
    """Update legacy stats structure when accessed."""
    if model_manager.stats["models_loaded"]:
        update_legacy_stats()


# Monkey patch MODEL_USAGE_STATS to auto-update
class LegacyStatsDict(dict):
    """Legacy stats dictionary that auto-updates from ModelManager."""

    def __getitem__(self, key):
        _update_legacy_stats_if_needed()
        return super().__getitem__(key)

    def get(self, key, default=None):
        _update_legacy_stats_if_needed()
        return super().get(key, default)


# Convert MODEL_USAGE_STATS to auto-updating dict
original_stats = MODEL_USAGE_STATS
MODEL_USAGE_STATS = LegacyStatsDict(original_stats)
