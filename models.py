"""
Model management module for audio transcription.

This module handles loading and managing Whisper models for transcription,
including both accurate and fast model pools for optimal performance.
"""
import threading
import time
import stable_whisper
import os
from logging_config import get_logger

logger = get_logger(__name__)

# Dual model architecture - load both models
MODEL_ACCURATE = None  # large-v3-turbo for quality
MODEL_FAST_POOL = []   # Pool of 3 base models for speed
MODEL_ACCURATE_NAME = "large-v3-turbo"
MODEL_FAST_NAME = "base"
# TESTING: Make pool size configurable for A/B testing (Windows PowerShell syntax)
# Default 3, but configurable
MODEL_FAST_POOL_SIZE = int(os.getenv('FAST_POOL_SIZE', '3'))
MODEL_FAST_POOL_SIZE = max(
    1, min(MODEL_FAST_POOL_SIZE, 5))  # Clamp between 1-5

logger.info("🔧 Configuration: Fast model pool size set to %d",
            MODEL_FAST_POOL_SIZE)

# Global model state management
MODEL_USAGE_STATS = {
    "stats_lock": threading.RLock(),
    "concurrent_requests": 0,
    # CRITICAL FIX: Ensure this is always an integer, not a set
    "active_transcriptions": 0,
    "accurate_busy": False,
    "fast_models_busy": False,
    "accurate_model_lock": threading.RLock(),
    "fast_model_locks": [],
    "fast_model_usage": [],
    # CRITICAL FIX: Add missing usage tracking counters
    "fast_uses": 0,
    "accurate_uses": 0,
    "queue_overflows": 0
}


def load_models_if_needed():
    """Lazy load accurate model and fast model pool only when needed"""
    global MODEL_ACCURATE, MODEL_FAST_POOL

    if MODEL_ACCURATE is None:
        logger.info("Loading ACCURATE model: %s...", MODEL_ACCURATE_NAME)
        MODEL_ACCURATE = stable_whisper.load_model(
            MODEL_ACCURATE_NAME, device="cuda")
        logger.info("Accurate model loaded successfully!")

    if len(MODEL_FAST_POOL) == 0:
        logger.info("Loading FAST model pool: %d x %s...",
                    MODEL_FAST_POOL_SIZE, MODEL_FAST_NAME)

        # Initialize tracking structures
        MODEL_USAGE_STATS["fast_model_locks"] = []
        MODEL_USAGE_STATS["fast_model_usage"] = []

        for i in range(MODEL_FAST_POOL_SIZE):
            logger.info("Loading fast model %d/%d...",
                        i+1, MODEL_FAST_POOL_SIZE)
            fast_model = stable_whisper.load_model(
                MODEL_FAST_NAME, device="cuda")
            MODEL_FAST_POOL.append(fast_model)

            # Create individual lock and tracking for each model
            MODEL_USAGE_STATS["fast_model_locks"].append(threading.RLock())
            MODEL_USAGE_STATS["fast_model_usage"].append(0)

            logger.info("Fast model %d/%d loaded successfully!",
                        i+1, MODEL_FAST_POOL_SIZE)

        logger.info(
            "🚀 Fast model pool loaded: %d models ready for parallel processing", MODEL_FAST_POOL_SIZE)

    return MODEL_ACCURATE, MODEL_FAST_POOL


def get_available_fast_model():
    """Get the least busy available fast model from the pool"""
    with MODEL_USAGE_STATS["stats_lock"]:
        # Find the least used available model
        available_models = []
        for i, busy in enumerate(MODEL_USAGE_STATS["fast_model_usage"]):
            if not busy:
                available_models.append(i)

        if available_models:
            # Sort by usage count to get the least used model
            available_models.sort(
                key=lambda x: MODEL_USAGE_STATS["fast_model_usage"][x])
            model_index = available_models[0]

            # Mark as busy and increment usage
            MODEL_USAGE_STATS["fast_model_usage"][model_index] += 1

            logger.debug("🎯 Selected fast model %d (usage: %d, available: %d/%d)",
                         model_index + 1,
                         MODEL_USAGE_STATS["fast_model_usage"][model_index],
                         len(available_models) - 1,
                         MODEL_FAST_POOL_SIZE)

            # All models busy, return first model (will wait for lock)
            return MODEL_FAST_POOL[model_index], MODEL_USAGE_STATS["fast_model_locks"][model_index], model_index
    logger.info("⚠️ All fast models busy, waiting for model 1...")
    return MODEL_FAST_POOL[0], MODEL_USAGE_STATS["fast_model_locks"][0], 0


def release_fast_model(model_index):
    """Release a fast model back to the available pool"""
    with MODEL_USAGE_STATS["stats_lock"]:
        if 0 <= model_index < len(MODEL_USAGE_STATS["fast_model_usage"]):
            MODEL_USAGE_STATS["fast_model_usage"][model_index] = False
            logger.debug("✅ Released fast model %d back to pool",
                         model_index + 1)


def count_available_fast_models():
    """Count how many fast models are currently available"""
    with MODEL_USAGE_STATS["stats_lock"]:
        return sum(1 for busy in MODEL_USAGE_STATS["fast_model_usage"] if not busy)


def debug_lock_state():
    """Debug the current lock state for troubleshooting deadlocks."""
    try:
        current_thread = threading.current_thread()
        lock = MODEL_USAGE_STATS["stats_lock"]

        # Try to get lock state information
        lock_acquired = lock.acquire(blocking=False)
        if lock_acquired:
            try:
                logger.debug("🔍 Lock is FREE - current thread: %s",
                             current_thread.name)
                MODEL_USAGE_STATS["lock_owner"] = None
                MODEL_USAGE_STATS["lock_acquired_time"] = None
                MODEL_USAGE_STATS["lock_stack_trace"] = None
            finally:
                lock.release()
        else:
            # Lock is busy
            owner = MODEL_USAGE_STATS.get("lock_owner", "UNKNOWN")
            acquired_time = MODEL_USAGE_STATS.get(
                "lock_acquired_time", "UNKNOWN")
            stack_trace = MODEL_USAGE_STATS.get("lock_stack_trace", "UNKNOWN")

            logger.error("🚨 LOCK IS BUSY - Owner: %s, Acquired: %s seconds ago",
                         owner,
                         f"{time.time() - acquired_time:.2f}" if acquired_time else "UNKNOWN")

            if stack_trace:
                logger.error("🚨 Lock acquired at:\n%s", stack_trace)

            # Check if lock has been held too long (over 30 seconds = definitely stuck)
            if acquired_time and (time.time() - acquired_time) > 30:
                logger.error(
                    "🚨 CRITICAL: Lock held for over 30 seconds - attempting emergency reset!")
                return True  # Signal that we need emergency recovery

        return False
    except Exception as e:
        logger.error("❌ Error debugging lock state: %s", str(e))
        return False


async def emergency_lock_reset():
    """Emergency function to reset all locks and states in case of deadlock."""
    try:
        logger.error("🚨 EMERGENCY LOCK RESET: Attempting to break deadlock...")

        # Create completely new locks to break any deadlocks
        MODEL_USAGE_STATS["stats_lock"] = threading.RLock()
        MODEL_USAGE_STATS["accurate_model_lock"] = threading.RLock()

        # Reset all state variables
        MODEL_USAGE_STATS["concurrent_requests"] = 0
        # CRITICAL FIX: Ensure integer
        MODEL_USAGE_STATS["active_transcriptions"] = 0
        MODEL_USAGE_STATS["accurate_busy"] = False
        MODEL_USAGE_STATS["fast_models_busy"] = False

        # Recreate fast model locks
        MODEL_USAGE_STATS["fast_model_locks"] = [
            threading.RLock() for _ in range(MODEL_FAST_POOL_SIZE)]
        MODEL_USAGE_STATS["fast_model_usage"] = [0] * MODEL_FAST_POOL_SIZE

        logger.error(
            "🔧 EMERGENCY RESET COMPLETE: All locks recreated, all states reset to clean state")
        logger.info("📊 Reset state: concurrent_requests=%d, active_transcriptions=%d, accurate_busy=%s, fast_models_busy=%s",
                    MODEL_USAGE_STATS["concurrent_requests"],
                    MODEL_USAGE_STATS["active_transcriptions"],
                    MODEL_USAGE_STATS["accurate_busy"],
                    MODEL_USAGE_STATS["fast_models_busy"])

        return True

    except Exception as e:
        logger.error("💥 EMERGENCY RESET FAILED: %s", str(e))
        return False


def determine_model_routing(audio_file, current_queue_size=0, entry_stats=None):
    """
    Determine which model to use based on audio duration and system load.

    Returns:
        tuple: (use_accurate: bool, routing_reason: str)
    """
    try:
        # Calculate audio duration from file size (approximate)
        file_size = os.path.getsize(audio_file)
        # Rough estimation: 48kHz * 2 channels * 2 bytes = 192KB per second
        duration_seconds = file_size / 192000.0

        # Get current system state
        concurrent = entry_stats['concurrent'] if entry_stats else MODEL_USAGE_STATS["concurrent_requests"]
        active = entry_stats['active'] if entry_stats else MODEL_USAGE_STATS["active_transcriptions"]

        # Count available fast models
        fast_available = sum(
            1 for usage in MODEL_USAGE_STATS["fast_model_usage"] if usage == 0)

        # Check if accurate model is busy
        accurate_busy = MODEL_USAGE_STATS["accurate_busy"]

        logger.debug("🔍 Routing decision: duration=%.1fs, accurate_busy=%s, concurrent=%d, active=%d, queue=%d, fast_available=%d/%d",
                     duration_seconds, accurate_busy, concurrent, active, current_queue_size, fast_available, len(MODEL_USAGE_STATS["fast_model_usage"]))

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


def select_fast_model():
    """
    Select the least used fast model for load balancing.

    Returns:
        int: Index of the selected fast model
    """
    try:
        with MODEL_USAGE_STATS["stats_lock"]:
            # Find the model with the lowest usage count
            min_usage = min(MODEL_USAGE_STATS["fast_model_usage"])
            available_models = [i for i, usage in enumerate(
                MODEL_USAGE_STATS["fast_model_usage"]) if usage == min_usage]

            # Select the first available model with minimum usage
            # Increment usage count
            selected_index = available_models[0]
            MODEL_USAGE_STATS["fast_model_usage"][selected_index] += 1

            logger.info("🎯 Selected fast model %d (usage: %d, available: %d/%d)",
                        selected_index + 1,
                        MODEL_USAGE_STATS["fast_model_usage"][selected_index],
                        len(available_models) - 1,
                        len(MODEL_USAGE_STATS["fast_model_usage"]))

            return selected_index

    except Exception as e:
        logger.error("❌ Error selecting fast model: %s", str(e))
        # Default to first model
        return 0
