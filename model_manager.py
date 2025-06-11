"""
Unified Model Manager for audio transcription.

This module provides a centralized, modular approach to loading and managing
Whisper models with proper initialization, warming up, and configuration.
Supports a two-tier architecture: accurate (large) and fast (small) model pools.
"""
import asyncio
import os
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import stable_whisper
import audio_utils
from logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class ModelConfig:
    """Configuration for model tiers."""
    name: str
    count: int
    description: str


@dataclass
class ModelTier:
    """Represents a tier of models (accurate or fast)."""
    config: ModelConfig
    models: List[any]
    locks: List[threading.RLock]
    usage_stats: List[int]
    global_lock: threading.RLock
    is_busy: bool = False


class ModelManager:
    """
    Centralized model manager with two-tier architecture.

    Tier 1 (Accurate): Large, high-quality models for accuracy
    Tier 2 (Fast): Small, fast models for speed and parallelization
    """

    def __init__(self):
        """Initialize the model manager with configurable tiers."""

        # Configuration - easily adjustable for different hardware
        self.accurate_config = ModelConfig(
            name="large-v3-turbo",
            count=1,  # Only 1 accurate model due to hardware limitations
            description="High-accuracy transcription"
        )

        # Get fast pool size from environment or default to 3
        fast_pool_size = int(os.getenv('FAST_POOL_SIZE', '3'))
        fast_pool_size = max(1, min(fast_pool_size, 5))  # Clamp between 1-5

        self.fast_config = ModelConfig(
            name="base",
            count=fast_pool_size,
            description="Fast parallel transcription"
        )

        logger.info("🔧 Model Manager Configuration:")
        logger.info("   Tier 1 (Accurate): %d x %s",
                    self.accurate_config.count, self.accurate_config.name)
        logger.info("   Tier 2 (Fast): %d x %s",
                    self.fast_config.count, self.fast_config.name)

        # Initialize model tiers
        self.accurate_tier = ModelTier(
            config=self.accurate_config,
            models=[],
            locks=[],
            usage_stats=[],
            global_lock=threading.RLock()
        )

        self.fast_tier = ModelTier(
            config=self.fast_config,
            models=[],
            locks=[],
            usage_stats=[],
            global_lock=threading.RLock()
        )

        # Global stats
        self.stats = {
            "stats_lock": threading.RLock(),
            "concurrent_requests": 0,
            "active_transcriptions": 0,
            "accurate_uses": 0,
            "fast_uses": 0,
            "queue_overflows": 0,
            "models_loaded": False,
            "models_warmed": False
        }

        logger.info("✅ Model Manager initialized")

    async def initialize_models(self, warm_up: bool = True) -> bool:
        """
        Initialize all models in both tiers.

        Args:
            warm_up: Whether to warm up models after loading

        Returns:
            bool: True if successful
        """
        try:
            logger.info("🚀 Starting model initialization...")

            # Load accurate tier
            success = await self._load_tier(self.accurate_tier, "Accurate")
            if not success:
                return False

            # Load fast tier
            success = await self._load_tier(self.fast_tier, "Fast")
            if not success:
                return False

            self.stats["models_loaded"] = True
            logger.info("✅ All model tiers loaded successfully")

            # Warm up if requested
            if warm_up:
                success = await self.warm_up_models()
                if success:
                    self.stats["models_warmed"] = True
                    logger.info("🎯 Model initialization complete with warm-up")
                else:
                    logger.warning("⚠️ Models loaded but warm-up had issues")

            # Report final status
            total_models = len(self.accurate_tier.models) + \
                len(self.fast_tier.models)
            logger.info("🚀 Total models ready: %d (%d accurate + %d fast)",
                        total_models, len(self.accurate_tier.models), len(self.fast_tier.models))

            return True

        except Exception as e:
            logger.error("❌ Model initialization failed: %s", str(e))
            return False

    async def _load_tier(self, tier: ModelTier, tier_name: str) -> bool:
        """Load all models for a specific tier."""
        try:
            logger.info("Loading %s tier: %d x %s...",
                        tier_name, tier.config.count, tier.config.name)

            for i in range(tier.config.count):
                logger.info("Loading %s model %d/%d...",
                            tier_name.lower(), i+1, tier.config.count)

                # Load the model
                model = stable_whisper.load_model(
                    tier.config.name, device="cuda")
                tier.models.append(model)

                # Create lock and usage tracking for this model
                tier.locks.append(threading.RLock())
                tier.usage_stats.append(0)

                logger.info("✅ %s model %d/%d loaded successfully",
                            tier_name, i+1, tier.config.count)

            logger.info("🚀 %s tier loaded: %d models ready",
                        tier_name, len(tier.models))
            return True

        except Exception as e:
            logger.error("❌ Failed to load %s tier: %s", tier_name, str(e))
            return False

    async def warm_up_models(self) -> bool:
        """Warm up all models in both tiers."""
        try:
            logger.info("🔥 Starting model warm-up process...")
            dummy_files = []

            # Warm up accurate tier
            accurate_success = await self._warm_up_tier(
                self.accurate_tier, "accurate", dummy_files
            )

            # Warm up fast tier
            fast_success = await self._warm_up_tier(
                self.fast_tier, "fast", dummy_files
            )

            # Clean up dummy files
            await self._cleanup_dummy_files(dummy_files)

            if accurate_success and fast_success:
                logger.info("🎯 All models warmed up successfully!")
                return True
            else:
                logger.warning(
                    "⚠️ Some models had warm-up issues but continuing")
                return False

        except Exception as e:
            logger.error("❌ Model warm-up failed: %s", str(e))
            return False

    async def _warm_up_tier(self, tier: ModelTier, tier_name: str, dummy_files: List[str]) -> bool:
        """Warm up all models in a specific tier."""
        try:
            logger.info("Warming up %s tier (%d models)...",
                        tier_name, len(tier.models))
            successes = 0

            # Create warmup tasks for all models in this tier
            warmup_tasks = []
            for i, model in enumerate(tier.models):
                model_display_name = f"{tier_name.upper()}-{i+1}" if len(
                    tier.models) > 1 else tier_name.upper()

                # Create dummy audio file for this model
                dummy_file = audio_utils.create_dummy_audio_file(
                    f"warmup_{tier_name}_{i+1}.wav")
                dummy_files.append(dummy_file)

                logger.debug("Created warmup audio file for %s: %s",
                             model_display_name, dummy_file)

                # Create warmup task
                warmup_task = self._safe_warmup_transcribe(
                    dummy_file, model, tier.locks[i], tier_name, i
                )
                warmup_tasks.append((warmup_task, model_display_name, i+1))

            # Wait for all warmup tasks to complete
            for warmup_task, model_display_name, model_num in warmup_tasks:
                try:
                    success = await warmup_task
                    if success:
                        successes += 1
                        logger.info("✅ %s warmed up successfully",
                                    model_display_name)
                    else:
                        logger.warning("⚠️ %s warmup had issues",
                                       model_display_name)
                except Exception as e:
                    logger.warning("⚠️ %s warmup failed: %s",
                                   model_display_name, str(e))

            # Report results for this tier
            if successes == len(tier.models):
                logger.info(
                    "✅ All %s tier models warmed up successfully", tier_name)
                return True
            else:
                logger.warning("⚠️ %d/%d %s tier models warmed up successfully",
                               successes, len(tier.models), tier_name)
                return False

        except Exception as e:
            logger.error("❌ Failed to warm up %s tier: %s", tier_name, str(e))
            return False

    async def _safe_warmup_transcribe(self, audio_file: str, model: any, model_lock: threading.RLock,
                                      tier_name: str, model_index: int) -> bool:
        """Safely perform warmup transcription with timeout and error handling."""
        try:
            transcription_id = str(uuid.uuid4())[:8]

            def warmup_transcribe():
                with model_lock:
                    return model.transcribe(audio_file, language="en", word_timestamps=False)

            # Run warmup transcription with timeout
            loop = asyncio.get_event_loop()
            result = await asyncio.wait_for(
                loop.run_in_executor(None, warmup_transcribe),
                timeout=30.0  # 30 second timeout
            )

            logger.debug("Warmup transcription completed for %s-%d",
                         tier_name.upper(), model_index + 1)
            return result is not None

        except asyncio.TimeoutError:
            logger.warning("Warmup transcription timed out for %s-%d - continuing anyway",
                           tier_name.upper(), model_index + 1)
            return False
        except Exception as e:
            logger.warning("Warmup transcription failed for %s-%d: %s - continuing anyway",
                           tier_name.upper(), model_index + 1, str(e))
            return False

    async def _cleanup_dummy_files(self, dummy_files: List[str]):
        """Clean up all dummy audio files."""
        for dummy_file in dummy_files:
            try:
                if os.path.exists(dummy_file):
                    os.remove(dummy_file)
                    logger.debug("Cleaned up warmup file: %s", dummy_file)
            except Exception as e:
                logger.debug(
                    "Failed to clean up warmup file %s: %s", dummy_file, str(e))

    def get_models(self) -> Tuple[any, List[any]]:
        """
        Get models for backward compatibility.

        Returns:
            Tuple of (accurate_model, fast_model_pool)
        """
        if not self.stats["models_loaded"]:
            logger.warning("Models not loaded yet!")
            return None, []

        accurate_model = self.accurate_tier.models[0] if self.accurate_tier.models else None
        fast_models = self.fast_tier.models.copy()

        return accurate_model, fast_models

    def select_fast_model(self) -> Tuple[any, threading.RLock, int]:
        """
        Select the least used fast model for load balancing.

        Returns:
            Tuple of (model, lock, index)
        """
        with self.stats["stats_lock"]:
            if not self.fast_tier.models:
                logger.error("No fast models available!")
                return None, None, -1

            # Find the model with the lowest usage count
            min_usage = min(self.fast_tier.usage_stats)
            available_models = [
                i for i, usage in enumerate(self.fast_tier.usage_stats)
                if usage == min_usage
            ]

            # Select the first available model with minimum usage
            selected_index = available_models[0]

            # Increment usage count
            self.fast_tier.usage_stats[selected_index] += 1
            self.stats["fast_uses"] += 1

            logger.info("🎯 Selected fast model %d (usage: %d, available: %d/%d)",
                        selected_index + 1,
                        self.fast_tier.usage_stats[selected_index],
                        len(available_models) - 1,
                        len(self.fast_tier.models))

            return (
                self.fast_tier.models[selected_index],
                self.fast_tier.locks[selected_index],
                selected_index
            )

    def release_fast_model(self, model_index: int):
        """Release a fast model back to the available pool."""
        with self.stats["stats_lock"]:
            if 0 <= model_index < len(self.fast_tier.usage_stats):
                if self.fast_tier.usage_stats[model_index] > 0:
                    self.fast_tier.usage_stats[model_index] -= 1
                    logger.debug("✅ Released fast model %d (usage now: %d)",
                                 model_index + 1, self.fast_tier.usage_stats[model_index])
                else:
                    logger.warning("⚠️ Attempted to release fast model %d that wasn't in use",
                                   model_index + 1)

    def get_accurate_model(self) -> Tuple[any, threading.RLock]:
        """
        Get the accurate model and its lock.

        Returns:
            Tuple of (model, lock)
        """
        if not self.accurate_tier.models:
            logger.error("No accurate model available!")
            return None, None

        with self.stats["stats_lock"]:
            self.stats["accurate_uses"] += 1

        return self.accurate_tier.models[0], self.accurate_tier.locks[0]

    def count_available_fast_models(self) -> int:
        """Count how many fast models are currently available (usage = 0)."""
        with self.stats["stats_lock"]:
            return sum(1 for usage in self.fast_tier.usage_stats if usage == 0)

    def is_accurate_busy(self) -> bool:
        """Check if the accurate model is currently busy."""
        return self.accurate_tier.is_busy

    def set_accurate_busy(self, busy: bool):
        """Set the accurate model busy status."""
        with self.stats["stats_lock"]:
            self.accurate_tier.is_busy = busy

    def get_stats(self) -> Dict:
        """Get current model usage statistics."""
        with self.stats["stats_lock"]:
            return {
                "models_loaded": self.stats["models_loaded"],
                "models_warmed": self.stats["models_warmed"],
                "concurrent_requests": self.stats["concurrent_requests"],
                "active_transcriptions": self.stats["active_transcriptions"],
                "accurate_uses": self.stats["accurate_uses"],
                "fast_uses": self.stats["fast_uses"],
                "queue_overflows": self.stats["queue_overflows"],
                "accurate_busy": self.accurate_tier.is_busy,
                "accurate_models": len(self.accurate_tier.models),
                "fast_models": len(self.fast_tier.models),
                "fast_available": self.count_available_fast_models(),
                "fast_usage": self.fast_tier.usage_stats.copy()
            }

    def update_stats(self, **kwargs):
        """Update statistics."""
        with self.stats["stats_lock"]:
            for key, value in kwargs.items():
                if key in self.stats:
                    self.stats[key] = value


# Global model manager instance
model_manager = ModelManager()


# Backward compatibility functions
def load_models_if_needed():
    """Legacy function for backward compatibility."""
    if not model_manager.stats["models_loaded"]:
        logger.warning(
            "Models not loaded via ModelManager! Loading synchronously...")
        # This is a fallback - ideally should use async initialize_models()
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


def select_fast_model():
    """Legacy function for backward compatibility."""
    _, _, index = model_manager.select_fast_model()
    return index


def get_available_fast_model():
    """Legacy function for backward compatibility."""
    model, lock, index = model_manager.select_fast_model()
    return model, lock, index


def release_fast_model(model_index):
    """Legacy function for backward compatibility."""
    model_manager.release_fast_model(model_index)


def count_available_fast_models():
    """Legacy function for backward compatibility."""
    return model_manager.count_available_fast_models()


# Export stats structure for backward compatibility
MODEL_USAGE_STATS = {
    "stats_lock": model_manager.stats["stats_lock"],
    "concurrent_requests": 0,
    "active_transcriptions": 0,
    "accurate_busy": False,
    "fast_models_busy": False,
    "accurate_model_lock": None,  # Will be set when models are loaded
    "fast_model_locks": [],       # Will be populated when models are loaded
    "fast_model_usage": [],       # Will be populated when models are loaded
    "fast_uses": 0,
    "accurate_uses": 0,
    "queue_overflows": 0
}


def update_legacy_stats():
    """Update legacy stats structure for backward compatibility."""
    stats = model_manager.get_stats()

    MODEL_USAGE_STATS["concurrent_requests"] = stats["concurrent_requests"]
    MODEL_USAGE_STATS["active_transcriptions"] = stats["active_transcriptions"]
    MODEL_USAGE_STATS["accurate_busy"] = stats["accurate_busy"]
    MODEL_USAGE_STATS["fast_uses"] = stats["fast_uses"]
    MODEL_USAGE_STATS["accurate_uses"] = stats["accurate_uses"]
    MODEL_USAGE_STATS["queue_overflows"] = stats["queue_overflows"]

    # Update model references
    if model_manager.accurate_tier.locks:
        MODEL_USAGE_STATS["accurate_model_lock"] = model_manager.accurate_tier.locks[0]

    MODEL_USAGE_STATS["fast_model_locks"] = model_manager.fast_tier.locks.copy()
    MODEL_USAGE_STATS["fast_model_usage"] = model_manager.fast_tier.usage_stats.copy()


# Model configuration constants for backward compatibility
MODEL_ACCURATE_NAME = model_manager.accurate_config.name
MODEL_FAST_NAME = model_manager.fast_config.name
MODEL_FAST_POOL_SIZE = model_manager.fast_config.count
