"""
Refactored Model Manager for audio transcription.

This module provides a centralized, modular approach to loading and managing
Whisper models with YAML configuration support. Follows the "composition over inheritance"
principle with clean separation of concerns.
"""
import threading
from typing import Dict, List, Tuple, Any
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
        Get models for current usage.

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


# Global model manager instance
model_manager = ModelManager()
