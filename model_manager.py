"""
Faster-Whisper Model Manager.

This module provides a centralized, modular approach to loading and managing
faster-whisper models with YAML configuration support. Designed as a drop-in
replacement for the stable-ts model manager.
"""
import threading
from typing import Dict, List, Tuple, Any, Optional
from config_manager import get_config
from model_core import (
    FasterWhisperModelTier,
    FasterWhisperModelLoader,
    FasterWhisperModelWarmup,
    FasterWhisperStatsManager
)
from logging_config import get_logger
import audio_processing_utils

logger = get_logger(__name__)


class FasterWhisperModelManager:
    """
    Centralized faster-whisper model manager with two-tier architecture.

    Tier 1 (Accurate): Large, high-quality models for accuracy
    Tier 2 (Fast): Small, fast models for speed and parallelization
    """

    def __init__(self):
        """Initialize the faster-whisper model manager with YAML configuration."""
        # Load configuration
        self.config = get_config()

        # Initialize components
        self.stats_manager = FasterWhisperStatsManager()

        # Initialize model tiers from configuration
        self.accurate_tier = FasterWhisperModelTier(
            config=self.config.models.accurate)
        self.fast_tier = FasterWhisperModelTier(config=self.config.models.fast)

        # Log configuration
        logger.info("🔧 Faster-Whisper Model Manager Configuration (from YAML):")
        logger.info("   Tier 1 (Accurate): %d x %s on %s",
                    self.accurate_tier.config.count,
                    self.accurate_tier.config.name,
                    self.accurate_tier.config.device)
        logger.info("   Tier 2 (Fast): %d x %s on %s",
                    self.fast_tier.config.count,
                    self.fast_tier.config.name,
                    self.fast_tier.config.device)

        logger.info(
            "✅ Faster-Whisper Model Manager initialized with YAML configuration")

    async def initialize_models(self, warm_up: bool = True) -> bool:
        """Initialize all faster-whisper models with optional warmup."""
        try:
            logger.info("🚀 Initializing faster-whisper models...")

            # Load accurate tier models
            accurate_success = await FasterWhisperModelLoader.load_tier(
                self.accurate_tier, "Accurate")

            # Load fast tier models
            fast_success = await FasterWhisperModelLoader.load_tier(
                self.fast_tier, "Fast")

            if not accurate_success and not fast_success:
                logger.error("❌ Failed to load any model tiers")
                return False

            # Update stats
            self.stats_manager.update_stats(
                models_loaded=True,
                accurate_models=len(self.accurate_tier.models),
                fast_models=len(self.fast_tier.models),
                fast_available=len(self.fast_tier.models)
            )

            logger.info("✅ Faster-whisper models loaded: %d accurate + %d fast = %d total",
                        len(self.accurate_tier.models),
                        len(self.fast_tier.models),
                        len(self.accurate_tier.models) + len(self.fast_tier.models))

            # Warm up models if requested
            if warm_up:
                warmup_success = await self.warm_up_models()
                if not warmup_success:
                    logger.warning(
                        "⚠️ Model warmup had issues, but models should still work")

            return True

        except Exception as e:
            logger.error(
                "❌ Failed to initialize faster-whisper models: %s", str(e))
            return False

    async def warm_up_models(self) -> bool:
        """Warm up all loaded faster-whisper models."""
        try:
            logger.info("🔥 Starting faster-whisper model warmup...")
            dummy_files = []

            try:
                # Warm up accurate tier
                accurate_warmup_success = await FasterWhisperModelWarmup.warm_up_tier(
                    self.accurate_tier, "Accurate", dummy_files)

                # Warm up fast tier
                fast_warmup_success = await FasterWhisperModelWarmup.warm_up_tier(
                    self.fast_tier, "Fast", dummy_files)

                # Update warmup status
                overall_success = accurate_warmup_success or fast_warmup_success
                self.stats_manager.update_stats(
                    warmup_completed=overall_success)

                if overall_success:
                    logger.info(
                        "🎯 Faster-whisper model warmup completed successfully!")
                else:
                    logger.warning("⚠️ Faster-whisper model warmup had issues")

                return overall_success

            finally:
                # Clean up dummy files
                for dummy_file in dummy_files:
                    try:
                        if dummy_file and audio_processing_utils.safe_remove_file(dummy_file):
                            logger.debug(
                                "Cleaned up warmup file: %s", dummy_file)
                    except Exception as cleanup_error:
                        logger.warning("Failed to clean up warmup file %s: %s",
                                       dummy_file, str(cleanup_error))

        except Exception as e:
            logger.error("❌ Faster-whisper model warmup failed: %s", str(e))
            return False

    def get_models(self) -> Tuple[Any, List[Any]]:
        """Get models for compatibility with existing code."""
        accurate_model = self.accurate_tier.models[0] if self.accurate_tier.models else None
        fast_models = self.fast_tier.models
        return accurate_model, fast_models

    def get_accurate_model(self) -> Tuple[Any, threading.RLock]:
        """Get accurate model and its lock."""
        if not self.accurate_tier.models:
            raise ValueError("No accurate models available")

        self.accurate_tier.is_busy = True
        self.stats_manager.update_stats(accurate_busy=True)

        return self.accurate_tier.models[0], self.accurate_tier.locks[0]

    def select_fast_model(self) -> Tuple[Optional[Any], Optional[threading.RLock], Optional[int]]:
        """Select an available fast model."""
        if not self.fast_tier.models:
            return None, None, None

        # Find least used fast model
        min_usage = min(self.fast_tier.usage_stats)
        selected_index = self.fast_tier.usage_stats.index(min_usage)

        # Update usage stats
        self.fast_tier.usage_stats[selected_index] += 1

        # Update available count
        available_count = max(0, len(self.fast_tier.models) - 1)
        self.stats_manager.update_stats(fast_available=available_count)

        return (
            self.fast_tier.models[selected_index],
            self.fast_tier.locks[selected_index],
            selected_index
        )

    def release_fast_model(self, model_index: int) -> None:
        """Release a fast model back to the pool."""
        if 0 <= model_index < len(self.fast_tier.models):
            # Update available count
            available_count = len(self.fast_tier.models)
            self.stats_manager.update_stats(fast_available=available_count)

    def get_stats(self) -> Dict[str, Any]:
        """Get current model statistics."""
        return self.stats_manager.get_stats()

    def update_stats(self, **kwargs) -> None:
        """Update model statistics."""
        self.stats_manager.update_stats(**kwargs)


# Global faster-whisper model manager instance
faster_whisper_model_manager = FasterWhisperModelManager()
