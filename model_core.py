"""
Core components for model management.

This module provides the fundamental building blocks for model management:
- Model tier structures
- Model loading and initialization
- Warmup operations
- Statistics tracking
"""
import asyncio
import os
import threading
import uuid
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional
import stable_whisper
import audio_processing_utils
from config_manager import ModelTierConfig
from logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class ModelTier:
    """Represents a tier of models with associated metadata."""
    config: ModelTierConfig
    models: List[Any] = field(default_factory=list)
    locks: List[threading.RLock] = field(default_factory=list)
    usage_stats: List[int] = field(default_factory=list)
    global_lock: threading.RLock = field(default_factory=threading.RLock)
    is_busy: bool = False


class ModelLoader:
    """Handles loading of models for different tiers."""

    @staticmethod
    async def load_tier(tier: ModelTier, tier_name: str) -> bool:
        """Load all models for a specific tier."""
        try:
            logger.info("Loading %s tier: %d x %s...",
                        tier_name, tier.config.count, tier.config.name)

            for i in range(tier.config.count):
                logger.info("Loading %s model %d/%d...",
                            tier_name.lower(), i+1, tier.config.count)

                # Load the model
                model = stable_whisper.load_model(
                    tier.config.name, device=tier.config.device)
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


class ModelWarmup:
    """Handles model warmup operations."""

    @staticmethod
    async def warm_up_tier(tier: ModelTier, tier_name: str, dummy_files: List[str]) -> bool:
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
                dummy_file = audio_processing_utils.create_dummy_audio_file(
                    f"warmup_{tier_name}_{i+1}.wav")
                dummy_files.append(dummy_file)

                logger.debug("Created warmup audio file for %s: %s",
                             model_display_name, dummy_file)

                # Create warmup task
                warmup_task = ModelWarmup._safe_warmup_transcribe(
                    dummy_file, model, tier.locks[i], tier_name, i, tier.config.warmup_timeout
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

    @staticmethod
    async def _safe_warmup_transcribe(audio_file: str, model: Any, model_lock: threading.RLock,
                                      tier_name: str, model_index: int, timeout: int) -> bool:
        """Safely perform warmup transcription with timeout and error handling."""
        try:
            def warmup_transcribe():
                with model_lock:
                    return model.transcribe(audio_file, language="en", word_timestamps=False)

            # Run warmup transcription with timeout
            loop = asyncio.get_event_loop()
            result = await asyncio.wait_for(
                loop.run_in_executor(None, warmup_transcribe),
                timeout=float(timeout)
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

    @staticmethod
    async def cleanup_dummy_files(dummy_files: List[str]):
        """Clean up all dummy audio files."""
        for dummy_file in dummy_files:
            try:
                if os.path.exists(dummy_file):
                    os.remove(dummy_file)
                    logger.debug("Cleaned up warmup file: %s", dummy_file)
            except Exception as e:
                logger.debug(
                    "Failed to clean up warmup file %s: %s", dummy_file, str(e))


class ModelSelector:
    """Handles model selection and load balancing."""

    def __init__(self, stats_lock: threading.RLock):
        """Initialize model selector with shared stats lock."""
        self.stats_lock = stats_lock

    def select_fast_model(self, tier: ModelTier, stats: Dict[str, Any]) -> Tuple[Any, threading.RLock, int]:
        """Select the least used fast model for load balancing."""
        with self.stats_lock:
            if not tier.models:
                logger.error("No fast models available!")
                return None, None, -1

            # Find the model with the lowest usage count
            min_usage = min(tier.usage_stats)
            available_models = [
                i for i, usage in enumerate(tier.usage_stats)
                if usage == min_usage
            ]

            # Select the first available model with minimum usage
            selected_index = available_models[0]

            # Increment usage count
            tier.usage_stats[selected_index] += 1
            stats["fast_uses"] += 1

            logger.info("🎯 Selected fast model %d (usage: %d, available: %d/%d)",
                        selected_index + 1,
                        tier.usage_stats[selected_index],
                        len(available_models) - 1,
                        len(tier.models))

            return (
                tier.models[selected_index],
                tier.locks[selected_index],
                selected_index
            )

    def release_fast_model(self, tier: ModelTier, model_index: int):
        """Release a fast model back to the available pool."""
        with self.stats_lock:
            if 0 <= model_index < len(tier.usage_stats):
                if tier.usage_stats[model_index] > 0:
                    tier.usage_stats[model_index] -= 1
                    logger.debug("✅ Released fast model %d (usage now: %d)",
                                 model_index + 1, tier.usage_stats[model_index])
                else:
                    logger.warning("⚠️ Attempted to release fast model %d that wasn't in use",
                                   model_index + 1)

    def count_available_fast_models(self, tier: ModelTier) -> int:
        """Count how many fast models are currently available (usage = 0)."""
        with self.stats_lock:
            return sum(1 for usage in tier.usage_stats if usage == 0)


class StatsManager:
    """Manages model usage statistics."""

    def __init__(self):
        """Initialize statistics manager."""
        self.stats_lock = threading.RLock()
        self.stats = {
            "concurrent_requests": 0,
            "active_transcriptions": 0,
            "accurate_uses": 0,
            "fast_uses": 0,
            "queue_overflows": 0,
            "models_loaded": False,
            "models_warmed": False
        }

    def update_stats(self, **kwargs):
        """Update statistics."""
        with self.stats_lock:
            for key, value in kwargs.items():
                if key in self.stats:
                    self.stats[key] = value

    def get_stats_snapshot(self, accurate_tier: ModelTier, fast_tier: ModelTier) -> Dict[str, Any]:
        """Get current model usage statistics snapshot."""
        with self.stats_lock:
            return {
                "models_loaded": self.stats["models_loaded"],
                "models_warmed": self.stats["models_warmed"],
                "concurrent_requests": self.stats["concurrent_requests"],
                "active_transcriptions": self.stats["active_transcriptions"],
                "accurate_uses": self.stats["accurate_uses"],
                "fast_uses": self.stats["fast_uses"],
                "queue_overflows": self.stats["queue_overflows"],
                "accurate_busy": accurate_tier.is_busy,
                "accurate_models": len(accurate_tier.models),
                "fast_models": len(fast_tier.models),
                "fast_available": sum(1 for usage in fast_tier.usage_stats if usage == 0),
                "fast_usage": fast_tier.usage_stats.copy()
            }
