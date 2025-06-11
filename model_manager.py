"""
Refactored Model Manager for audio transcription.

This module provides a centralized, modular approach to loading and managing
Whisper models with YAML configuration support. Follows the "composition over inheritance"
principle with clean separation of concerns.
"""
import threading
from typing import Dict, List, Optional, Tuple, Any
from config_manager import get_config
from model_core import ModelTier, ModelLoader, ModelWarmup, ModelSelector, StatsManager
from logging_config import get_logger

logger = get_logger(__name__)


class ModelManager:
    """
    Centralized model manager with two-tier architecture and YAML configuration.

    Tier 1 (Accurate): Large, high-quality models for accuracy
    Tier 2 (Fast): Small, fast models for speed and parallelization
    """

    def __init__(self):
        """Initialize the model manager with YAML configuration."""
        # Load configuration
        self.config = get_config()

        # Initialize components
        self.stats_manager = StatsManager()
        self.model_selector = ModelSelector(self.stats_manager.stats_lock)

        # Initialize model tiers from configuration
        self.accurate_tier = ModelTier(config=self.config.models.accurate)
        self.fast_tier = ModelTier(config=self.config.models.fast)

        # Log configuration
        logger.info("🔧 Model Manager Configuration (from YAML):")
        logger.info("   Tier 1 (Accurate): %d x %s on %s",
                    self.accurate_tier.config.count,
                    self.accurate_tier.config.name,
                    self.accurate_tier.config.device)
        logger.info("   Tier 2 (Fast): %d x %s on %s",
                    self.fast_tier.config.count,
                    self.fast_tier.config.name,
                    self.fast_tier.config.device)

        logger.info("✅ Model Manager initialized with YAML configuration")

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
            success = await ModelLoader.load_tier(self.accurate_tier, "Accurate")
            if not success:
                return False

            # Load fast tier
            success = await ModelLoader.load_tier(self.fast_tier, "Fast")
            if not success:
                return False

            self.stats_manager.update_stats(models_loaded=True)
            logger.info("✅ All model tiers loaded successfully")

            # Warm up if requested
            if warm_up:
                success = await self._warm_up_models()
                if success:
                    self.stats_manager.update_stats(models_warmed=True)
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

    async def warm_up_models(self) -> bool:
        """
        Public method to warm up all models in both tiers.

        Returns:
            bool: True if successful
        """
        return await self._warm_up_models()

    async def _warm_up_models(self) -> bool:
        """Warm up all models in both tiers."""
        try:
            logger.info("🔥 Starting model warm-up process...")
            dummy_files = []

            # Warm up accurate tier
            accurate_success = await ModelWarmup.warm_up_tier(
                self.accurate_tier, "accurate", dummy_files
            )

            # Warm up fast tier
            fast_success = await ModelWarmup.warm_up_tier(
                self.fast_tier, "fast", dummy_files
            )

            # Clean up dummy files
            await ModelWarmup.cleanup_dummy_files(dummy_files)

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

    def get_models(self) -> Tuple[Any, List[Any]]:
        """
        Get models for backward compatibility.

        Returns:
            Tuple of (accurate_model, fast_model_pool)
        """
        if not self.stats_manager.stats["models_loaded"]:
            logger.warning("Models not loaded yet!")
            return None, []

        accurate_model = self.accurate_tier.models[0] if self.accurate_tier.models else None
        fast_models = self.fast_tier.models.copy()

        return accurate_model, fast_models

    def select_fast_model(self) -> Tuple[Any, threading.RLock, int]:
        """
        Select the least used fast model for load balancing.

        Returns:
            Tuple of (model, lock, index)
        """
        return self.model_selector.select_fast_model(self.fast_tier, self.stats_manager.stats)

    def release_fast_model(self, model_index: int):
        """Release a fast model back to the available pool."""
        self.model_selector.release_fast_model(self.fast_tier, model_index)

    def get_accurate_model(self) -> Tuple[Any, threading.RLock]:
        """
        Get the accurate model and its lock.

        Returns:
            Tuple of (model, lock)
        """
        if not self.accurate_tier.models:
            logger.error("No accurate model available!")
            return None, None

        with self.stats_manager.stats_lock:
            self.stats_manager.stats["accurate_uses"] += 1

        return self.accurate_tier.models[0], self.accurate_tier.locks[0]

    def count_available_fast_models(self) -> int:
        """Count how many fast models are currently available (usage = 0)."""
        return self.model_selector.count_available_fast_models(self.fast_tier)

    def is_accurate_busy(self) -> bool:
        """Check if the accurate model is currently busy."""
        return self.accurate_tier.is_busy

    def set_accurate_busy(self, busy: bool):
        """Set the accurate model busy status."""
        with self.stats_manager.stats_lock:
            self.accurate_tier.is_busy = busy

    def get_stats(self) -> Dict[str, Any]:
        """Get current model usage statistics."""
        return self.stats_manager.get_stats_snapshot(self.accurate_tier, self.fast_tier)

    def update_stats(self, **kwargs):
        """Update statistics."""
        self.stats_manager.update_stats(**kwargs)

    def reload_configuration(self) -> bool:
        """
        Reload configuration from YAML file.

        Note: This only reloads the config object, not the loaded models.
        Model reloading would require a full restart.
        """
        try:
            from config_manager import reload_config
            self.config = reload_config()
            logger.info("✅ Configuration reloaded from YAML")
            return True
        except Exception as e:
            logger.error("❌ Failed to reload configuration: %s", str(e))
            return False


class DynamicStats:
    """A dictionary-like object that dynamically fetches stats."""

    def __init__(self, model_manager):
        self.model_manager = model_manager

    def __getitem__(self, key):
        """Allow dictionary-style access."""
        stats = self.model_manager.get_stats()

        # Map old keys to new stats structure
        legacy_mapping = {
            "models_loaded": "models_loaded",
            "models_warmed": "models_warmed",
            "concurrent_requests": "concurrent_requests",
            "active_transcriptions": "active_transcriptions",
            "accurate_uses": "accurate_uses",
            "fast_uses": "fast_uses",
            "queue_overflows": "queue_overflows",
            "accurate_busy": "accurate_busy",
            "fast_models_busy": lambda: False,  # Legacy field
            "stats_lock": lambda: self.model_manager.stats_manager.stats_lock,
            "accurate_model_lock": lambda: self.model_manager.accurate_tier.locks[0] if self.model_manager.accurate_tier.locks else None,
            "fast_model_locks": lambda: self.model_manager.fast_tier.locks.copy(),
            "fast_model_usage": "fast_usage"
        }

        if key in legacy_mapping:
            value = legacy_mapping[key]
            if callable(value):
                return value()
            elif isinstance(value, str):
                return stats.get(value)
            else:
                return value
        else:
            raise KeyError(f"Unknown key: {key}")

    def get(self, key, default=None):
        """Dictionary-style get method."""
        try:
            return self[key]
        except KeyError:
            return default


class LegacyCompat:
    """Provides backward compatibility with the old model manager interface."""

    def __init__(self, model_manager: ModelManager):
        """Initialize legacy compatibility layer."""
        self.model_manager = model_manager

    @property
    def MODEL_USAGE_STATS(self) -> Dict[str, Any]:
        """Legacy stats structure for backward compatibility."""
        stats = self.model_manager.get_stats()

        legacy_stats = {
            "stats_lock": self.model_manager.stats_manager.stats_lock,
            "concurrent_requests": stats["concurrent_requests"],
            "active_transcriptions": stats["active_transcriptions"],
            "accurate_busy": stats["accurate_busy"],
            "fast_models_busy": False,  # Legacy field
            "accurate_model_lock": None,
            "fast_model_locks": [],
            "fast_model_usage": stats["fast_usage"],
            "fast_uses": stats["fast_uses"],
            "accurate_uses": stats["accurate_uses"],
            "queue_overflows": stats["queue_overflows"]
        }

        # Update model references if loaded
        if self.model_manager.accurate_tier.locks:
            legacy_stats["accurate_model_lock"] = self.model_manager.accurate_tier.locks[0]

        legacy_stats["fast_model_locks"] = self.model_manager.fast_tier.locks.copy()

        return legacy_stats

    def update_legacy_stats(self):
        """Update legacy stats structure for backward compatibility."""
        # This method exists for compatibility but stats are now always current
        pass

    @property
    def MODEL_ACCURATE_NAME(self) -> str:
        """Legacy accurate model name property."""
        return self.model_manager.accurate_tier.config.name

    @property
    def MODEL_FAST_NAME(self) -> str:
        """Legacy fast model name property."""
        return self.model_manager.fast_tier.config.name

    @property
    def MODEL_FAST_POOL_SIZE(self) -> int:
        """Legacy fast pool size property."""
        return self.model_manager.fast_tier.config.count


# Global model manager instance
model_manager = ModelManager()
legacy_compat = LegacyCompat(model_manager)


# Backward compatibility functions
def load_models_if_needed():
    """Legacy function for backward compatibility."""
    if not model_manager.stats_manager.stats["models_loaded"]:
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
    return model_manager.select_fast_model()


def release_fast_model(model_index):
    """Legacy function for backward compatibility."""
    model_manager.release_fast_model(model_index)


def count_available_fast_models():
    """Legacy function for backward compatibility."""
    return model_manager.count_available_fast_models()


# Export legacy constants and stats structure
MODEL_USAGE_STATS = DynamicStats(model_manager)
MODEL_ACCURATE_NAME = legacy_compat.MODEL_ACCURATE_NAME
MODEL_FAST_NAME = legacy_compat.MODEL_FAST_NAME
MODEL_FAST_POOL_SIZE = legacy_compat.MODEL_FAST_POOL_SIZE


def update_legacy_stats():
    """Update legacy stats structure for backward compatibility."""
    legacy_compat.update_legacy_stats()
