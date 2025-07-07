"""
Faster-Whisper Model Core Components.

This module provides the faster-whisper implementation of model management,
designed to be a drop-in replacement for the stable-ts implementation.
"""
import asyncio
import os
import threading
import uuid
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional
from faster_whisper import WhisperModel
import audio_processing_utils
from config_manager import ModelTierConfig
from logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class FasterWhisperModelTier:
    """Represents a tier of faster-whisper models with associated metadata."""
    config: ModelTierConfig
    models: List[WhisperModel] = field(default_factory=list)
    locks: List[threading.RLock] = field(default_factory=list)
    usage_stats: List[int] = field(default_factory=list)
    global_lock: threading.RLock = field(default_factory=threading.RLock)
    is_busy: bool = False


class FasterWhisperModelLoader:
    """Handles loading of faster-whisper models for different tiers."""

    @staticmethod
    async def load_tier(tier: FasterWhisperModelTier, tier_name: str) -> bool:
        """Load all faster-whisper models for a specific tier."""
        try:
            logger.info("Loading %s tier (faster-whisper): %d x %s...",
                        tier_name, tier.config.count, tier.config.name)

            for i in range(tier.config.count):
                logger.info("Loading %s faster-whisper model %d/%d...",
                            tier_name.lower(), i+1, tier.config.count)

                # Load the faster-whisper model with auto compute type
                # Try different compute types for better compatibility
                compute_type = "auto" if tier.config.device == "cuda" else "int8"

                model = WhisperModel(
                    tier.config.name,
                    device=tier.config.device,
                    compute_type=compute_type
                )
                tier.models.append(model)

                # Create lock and usage tracking for this model
                tier.locks.append(threading.RLock())
                tier.usage_stats.append(0)

                logger.info("✅ %s faster-whisper model %d/%d loaded successfully",
                            tier_name, i+1, tier.config.count)

            logger.info("🚀 %s tier loaded (faster-whisper): %d models ready",
                        tier_name, len(tier.models))
            return True

        except Exception as e:
            logger.error(
                "❌ Failed to load %s tier (faster-whisper): %s", tier_name, str(e))
            return False


class FasterWhisperModelWarmup:
    """Handles faster-whisper model warmup operations."""

    @staticmethod
    async def warm_up_tier(tier: FasterWhisperModelTier, tier_name: str, dummy_files: List[str]) -> bool:
        """Warm up all faster-whisper models in a specific tier."""
        try:
            logger.info("Warming up %s tier (faster-whisper, %d models)...",
                        tier_name, len(tier.models))
            successes = 0

            # Create warmup tasks for all models in this tier
            warmup_tasks = []
            for i, model in enumerate(tier.models):
                model_display_name = f"{tier_name.upper()}-{i+1}" if len(
                    tier.models) > 1 else tier_name.upper()

                # Create dummy audio file for this model
                dummy_file = audio_processing_utils.create_dummy_audio_file(
                    f"warmup_faster_{tier_name}_{i+1}.wav")
                dummy_files.append(dummy_file)

                logger.debug("Created warmup audio file for %s: %s",
                             model_display_name, dummy_file)

                # Create warmup task
                warmup_task = FasterWhisperModelWarmup._safe_warmup_transcribe(
                    dummy_file, model, tier.locks[i], tier_name, i, tier.config.warmup_timeout
                )
                warmup_tasks.append((warmup_task, model_display_name, i+1))

            # Execute all warmup tasks in parallel
            logger.info("Executing %d warmup tasks for %s tier...",
                        len(warmup_tasks), tier_name)

            for warmup_task, model_display_name, model_index in warmup_tasks:
                try:
                    success = await warmup_task
                    if success:
                        successes += 1
                        logger.info("✅ %s warmup completed",
                                    model_display_name)
                    else:
                        logger.warning("⚠️ %s warmup failed",
                                       model_display_name)
                except Exception as e:
                    logger.error("❌ %s warmup error: %s",
                                 model_display_name, str(e))

            success_rate = successes / len(tier.models) if tier.models else 0
            logger.info("🎯 %s tier warmup complete: %d/%d models (%d%%)",
                        tier_name, successes, len(tier.models), int(success_rate * 100))

            return success_rate >= 0.5  # At least 50% models warmed up successfully

        except Exception as e:
            logger.error("❌ Failed to warm up %s tier: %s", tier_name, str(e))
            return False

    @staticmethod
    async def _safe_warmup_transcribe(audio_file: str, model: WhisperModel, model_lock: threading.RLock,
                                      tier_name: str, model_index: int, timeout: int) -> bool:
        """Safely perform warmup transcription with timeout and error handling."""
        try:
            def warmup_transcribe():
                with model_lock:
                    # Use faster-whisper transcribe method
                    segments, info = model.transcribe(
                        audio_file,
                        beam_size=1,  # Faster warmup
                        best_of=1,    # Faster warmup
                        temperature=0.0,
                        vad_filter=False  # Faster warmup
                    )
                    # Convert segments to list to ensure processing
                    return list(segments), info

            # Run warmup transcription with timeout
            loop = asyncio.get_event_loop()
            result = await asyncio.wait_for(
                loop.run_in_executor(None, warmup_transcribe),
                timeout=float(timeout)
            )

            logger.debug("Faster-whisper warmup transcription completed for %s-%d",
                         tier_name.upper(), model_index + 1)
            return result is not None

        except asyncio.TimeoutError:
            logger.warning("Faster-whisper warmup transcription timed out for %s-%d - continuing anyway",
                           tier_name.upper(), model_index + 1)
            return False
        except Exception as e:
            logger.warning("Faster-whisper warmup transcription failed for %s-%d: %s - continuing anyway",
                           tier_name.upper(), model_index + 1, str(e))
            return False


class FasterWhisperStatsManager:
    """Manages faster-whisper model statistics and usage tracking."""

    def __init__(self):
        """Initialize stats manager with thread-safe stats."""
        self.stats_lock = threading.RLock()
        self._stats = {
            "models_loaded": False,
            "accurate_models": 0,
            "fast_models": 0,
            "accurate_busy": False,
            "fast_available": 0,
            "accurate_uses": 0,
            "fast_uses": 0,
            "queue_overflows": 0,
            "total_uses": 0,
            "warmup_completed": False
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get a thread-safe copy of current stats."""
        with self.stats_lock:
            return self._stats.copy()

    def update_stats(self, **kwargs) -> None:
        """Update stats with thread safety."""
        with self.stats_lock:
            for key, value in kwargs.items():
                if key in self._stats:
                    self._stats[key] = value
                else:
                    logger.warning("Unknown stat key: %s", key)

    def increment_usage(self, model_type: str) -> None:
        """Increment usage counter for model type."""
        with self.stats_lock:
            if model_type == "accurate":
                self._stats["accurate_uses"] += 1
            elif model_type == "fast":
                self._stats["fast_uses"] += 1
            self._stats["total_uses"] += 1
