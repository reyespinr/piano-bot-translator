"""
Configuration Manager for Discord Bot Translator.

Provides centralized configuration loading from YAML files with validation
and default value handling.
"""
import os
import yaml
from dataclasses import dataclass
from typing import Dict, Any, Optional
from logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class ModelTierConfig:
    """Configuration for a model tier (accurate or fast)."""
    name: str
    count: int
    description: str
    device: str = "cuda"
    warmup_timeout: int = 30
    min_pool_size: Optional[int] = None
    max_pool_size: Optional[int] = None


@dataclass
class ModelsConfig:
    """Configuration for all model tiers."""
    accurate: ModelTierConfig
    fast: ModelTierConfig


@dataclass
class LoggingConfig:
    """Logging configuration."""
    console_level: str = "INFO"
    file_level: str = "DEBUG"
    max_file_size: int = 10
    backup_count: int = 4


@dataclass
class AppConfig:
    """Complete application configuration."""
    models: ModelsConfig
    logging: LoggingConfig


class ConfigManager:
    """Manages application configuration from YAML files."""

    def __init__(self, config_path: str = "config.yaml"):
        """Initialize configuration manager."""
        self.config_path = config_path
        self._config: Optional[AppConfig] = None

    def load_config(self) -> AppConfig:
        """Load configuration from YAML file with validation."""
        try:
            config_path = os.path.abspath(self.config_path)

            if not os.path.exists(config_path):
                logger.warning(
                    f"Config file not found: {config_path}, using defaults")
                return self._get_default_config()

            with open(config_path, 'r', encoding='utf-8') as f:
                raw_config = yaml.safe_load(f)

            if not raw_config:
                logger.warning("Empty config file, using defaults")
                return self._get_default_config()

            logger.info(f"✅ Loaded configuration from: {config_path}")
            return self._parse_config(raw_config)

        except Exception as e:
            logger.error(f"❌ Failed to load config: {e}, using defaults")
            return self._get_default_config()

    def _parse_config(self, raw_config: Dict[str, Any]) -> AppConfig:
        """Parse raw configuration into structured config objects."""
        # Parse models config
        models_raw = raw_config.get('models', {})

        # Parse accurate tier
        accurate_raw = models_raw.get('accurate', {})
        accurate_config = ModelTierConfig(
            name=accurate_raw.get('name', 'large-v3-turbo'),
            count=accurate_raw.get('count', 1),
            description=accurate_raw.get(
                'description', 'High-accuracy transcription'),
            device=accurate_raw.get('device', 'cuda'),
            warmup_timeout=accurate_raw.get('warmup_timeout', 30)
        )        # Parse fast tier - use only YAML configuration
        fast_raw = models_raw.get('fast', {})
        fast_pool_size = fast_raw.get('count', 3)
        min_pool = fast_raw.get('min_pool_size', 1)
        max_pool = fast_raw.get('max_pool_size', 5)

        # Apply pool size constraints from config
        fast_pool_size = max(min_pool, min(fast_pool_size, max_pool))

        fast_config = ModelTierConfig(
            name=fast_raw.get('name', 'base'),
            count=fast_pool_size,
            description=fast_raw.get(
                'description', 'Fast parallel transcription'),
            device=fast_raw.get('device', 'cuda'),
            warmup_timeout=fast_raw.get('warmup_timeout', 30),
            min_pool_size=min_pool,
            max_pool_size=max_pool
        )

        models_config = ModelsConfig(
            accurate=accurate_config, fast=fast_config)

        # Parse logging config
        logging_raw = raw_config.get('logging', {})
        logging_config = LoggingConfig(
            console_level=logging_raw.get('console_level', 'INFO'),
            file_level=logging_raw.get('file_level', 'DEBUG'),
            max_file_size=logging_raw.get('max_file_size', 10),
            backup_count=logging_raw.get('backup_count', 4)
        )

        return AppConfig(models=models_config, logging=logging_config)

    def _get_default_config(self) -> AppConfig:
        """Get default configuration when YAML loading fails."""
        accurate_config = ModelTierConfig(
            name="large-v3-turbo",
            count=1,
            description="High-accuracy transcription"
        )

        fast_config = ModelTierConfig(
            name="base",
            count=3,  # Default from YAML specification
            description="Fast parallel transcription",
            min_pool_size=1,
            max_pool_size=5
        )

        models_config = ModelsConfig(
            accurate=accurate_config, fast=fast_config)
        logging_config = LoggingConfig()

        return AppConfig(models=models_config, logging=logging_config)

    @property
    def config(self) -> AppConfig:
        """Get current configuration, loading if necessary."""
        if self._config is None:
            self._config = self.load_config()
        return self._config

    def reload_config(self) -> AppConfig:
        """Force reload configuration from file."""
        self._config = self.load_config()
        return self._config


# Global configuration manager instance
config_manager = ConfigManager()


def get_config() -> AppConfig:
    """Get the global application configuration."""
    return config_manager.config


def reload_config() -> AppConfig:
    """Reload configuration from file."""
    return config_manager.reload_config()
