"""
Centralized logging configuration for the Discord Voice Translator.

This module sets up comprehensive logging with:
- YAML-based configuration for easy customization
- File logging with rotation for persistent storage
- Console logging with configurable level (DEBUG/INFO)
- Proper filtering to control Discord.py verbosity
- Color formatting for better readability
- Performance optimizations for high-frequency operations
"""
import logging
import logging.handlers
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import yaml


class Colors:
    """ANSI color codes for console output formatting."""

    GREY = '\033[90m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD_RED = '\033[91;1m'
    RESET = '\033[0m'


class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds color codes to log messages for console output."""

    def __init__(self, fmt: Optional[str] = None) -> None:
        """Initialize the colored formatter.

        Args:
            fmt: Format string for log messages
        """
        super().__init__()
        self.fmt = fmt or "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

        # Color mapping for different log levels
        self.COLORS = {
            logging.DEBUG: Colors.GREY,
            logging.INFO: Colors.GREEN,
            logging.WARNING: Colors.YELLOW,
            logging.ERROR: Colors.RED,
            logging.CRITICAL: Colors.BOLD_RED
        }

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with appropriate color coding.

        Args:
            record: Log record to format

        Returns:
            Formatted log message with color codes
        """
        log_color = self.COLORS.get(record.levelno, Colors.RESET)

        # Apply color to the entire message
        formatter = logging.Formatter(
            f"%(asctime)s [{log_color}%(levelname)s{Colors.RESET}] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        return formatter.format(record)


class DiscordFilter(logging.Filter):
    """Filter to control Discord.py logging verbosity."""

    def __init__(self, max_level: int = logging.WARNING) -> None:
        """Initialize the Discord filter.

        Args:
            max_level: Maximum log level to allow for Discord logs
        """
        super().__init__()
        self.max_level = max_level

    def filter(self, record: logging.LogRecord) -> bool:
        """Filter Discord logs based on logger name and level.

        Args:
            record: Log record to filter

        Returns:
            True if record should be logged, False otherwise
        """
        # Allow all non-Discord logs
        if not record.name.startswith('discord'):
            return True

        # For Discord logs, only allow up to max_level
        return record.levelno >= self.max_level


class UvicornFilter(logging.Filter):
    """Filter to allow application logs and uvicorn, but block Discord."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Allow non-Discord logs and uvicorn logs.

        Args:
            record: Log record to filter

        Returns:
            True if record should be logged, False otherwise
        """
        # Block Discord logs
        if record.name.startswith('discord'):
            return False
        # Allow everything else (including uvicorn)
        return True


class NumbaFilter(logging.Filter):
    """Filter to suppress numba debug messages."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Suppress all numba debug messages.

        Args:
            record: Log record to filter

        Returns:
            True if record should be logged, False otherwise
        """
        # Suppress all numba debug messages
        if record.name.startswith('numba'):
            return record.levelno >= logging.WARNING
        return True


# Constants
CONFIG_FILE_PATH = Path("config.yaml")
LOG_DIR = Path("logs")
DEFAULT_CONFIG = {
    'logging': {
        'console_level': 'INFO',
        'file_level': 'DEBUG',
        'max_file_size': 10,
        'backup_count': 4
    }
}

# Global variable to store the log file path
log_file_path: Optional[str] = None


def load_config() -> Dict[str, Any]:
    """Load configuration from YAML file.

    Returns:
        Configuration dictionary with defaults if file doesn't exist
    """
    try:
        if CONFIG_FILE_PATH.exists():
            with open(CONFIG_FILE_PATH, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                # Merge with defaults to ensure all keys exist
                if 'logging' not in config:
                    config['logging'] = DEFAULT_CONFIG['logging']
                else:
                    for key, value in DEFAULT_CONFIG['logging'].items():
                        if key not in config['logging']:
                            config['logging'][key] = value
                return config
        else:
            # Create default config file
            with open(CONFIG_FILE_PATH, 'w', encoding='utf-8') as f:
                yaml.dump(DEFAULT_CONFIG, f,
                          default_flow_style=False, sort_keys=False)
            return DEFAULT_CONFIG
    except Exception as e:
        print(f"Warning: Could not load config.yaml, using defaults: {e}")
        return DEFAULT_CONFIG


def setup_logging() -> str:
    """Set up comprehensive logging configuration.

    Creates:
    - File handler with rotation for all application logs (always DEBUG)
    - Console handler with configurable level (DEBUG or INFO based on config)
    - Proper formatters and filters for each handler

    Returns:
        Path to the log file
    """
    global log_file_path

    # Load configuration
    config = load_config()
    logging_config = config['logging']

    # Parse console level from config
    console_level_str = logging_config.get('console_level', 'INFO').upper()
    console_level = getattr(logging, console_level_str, logging.INFO)

    # File level is always DEBUG
    file_level = logging.DEBUG

    # Create logs directory if it doesn't exist
    LOG_DIR.mkdir(exist_ok=True)

    # Generate log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = LOG_DIR / f"discord_translator_{timestamp}.log"

    # Store globally for later access
    log_file_path = str(log_filename)

    # Clear any existing handlers on root logger
    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    # Set root logger level to DEBUG to capture everything
    root_logger.setLevel(logging.DEBUG)

    # === FILE HANDLER SETUP ===
    # Create rotating file handler with config values
    max_bytes = logging_config.get(
        'max_file_size', 10) * 1024 * 1024  # Convert MB to bytes
    backup_count = logging_config.get('backup_count', 4)

    file_handler = logging.handlers.RotatingFileHandler(
        log_filename,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8'
    )
    file_handler.setLevel(file_level)  # Always DEBUG for file

    # File formatter (no colors, more detailed)
    file_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)-8s] %(name)-20s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    file_handler.addFilter(UvicornFilter())

    # === CONSOLE HANDLER SETUP ===
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)  # Use configured level

    # Console formatter with colors
    console_formatter = ColoredFormatter()
    console_handler.setFormatter(console_formatter)

    # Add filters to console handler
    console_handler.addFilter(UvicornFilter())
    console_handler.addFilter(DiscordFilter(max_level=logging.ERROR))
    console_handler.addFilter(NumbaFilter())

    # === ADD HANDLERS TO ROOT LOGGER ===
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    # === CONFIGURE SPECIFIC LOGGERS ===
    _configure_third_party_loggers()

    # === LOG STARTUP INFORMATION ===
    startup_logger = logging.getLogger('logging_setup')
    startup_logger.info("🎹 Discord Bot Translator - Logging initialized")
    startup_logger.info("📁 Log file: %s", log_filename)
    startup_logger.info("📊 Console level: %s, File level: %s",
                        console_level_str, "DEBUG")
    startup_logger.info("🔧 Configuration loaded from config.yaml")
    startup_logger.info("🔧 Root logger handlers: %d",
                        len(root_logger.handlers))

    return str(log_filename)


def _configure_third_party_loggers() -> None:
    """Configure third-party library loggers to reduce verbosity."""
    # Discord.py logging - reduce verbosity
    logging.getLogger('discord').setLevel(logging.ERROR)

    # Asyncio logging - reduce verbosity
    logging.getLogger('asyncio').setLevel(logging.ERROR)

    # HTTP and networking libraries
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('websockets').setLevel(logging.WARNING)
    logging.getLogger('aiohttp').setLevel(logging.WARNING)

    # Uvicorn logging
    logging.getLogger('uvicorn').setLevel(logging.INFO)
    logging.getLogger('uvicorn.access').setLevel(logging.INFO)

    # Suppress numba compilation messages
    numba_loggers = [
        'numba', 'numba.core', 'numba.core.byteflow',
        'numba.core.ssa', 'numba.core.interpreter'
    ]
    for logger_name in numba_loggers:
        level = logging.ERROR if 'byteflow' in logger_name or 'ssa' in logger_name or 'interpreter' in logger_name else logging.WARNING
        logging.getLogger(logger_name).setLevel(level)

    # Suppress torio debug messages
    torio_loggers = ['torio', 'torio._extension', 'torio._extension.utils']
    for logger_name in torio_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for the specified module.

    Args:
        name: Name of the module requesting the logger

    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


def _run_logging_tests() -> None:
    """Run logging tests when module is executed directly."""
    # Set up logging
    log_file = setup_logging()

    # Test different log levels
    test_logger = logging.getLogger('logging_test')
    test_logger.debug("🧪 Debug logging test")
    test_logger.info("🧪 Info logging test")
    test_logger.warning("🧪 Warning logging test")
    test_logger.error("🧪 Error logging test")

    # Test Discord logging (should be filtered on console)
    discord_test = logging.getLogger('discord.test')
    discord_test.debug("This DEBUG should NOT appear on console")
    discord_test.info("This INFO should NOT appear on console")
    discord_test.warning("This WARNING should NOT appear on console")
    discord_test.error("This ERROR should appear on console")

    print(f"\n✅ Logging test complete. Check log file: {log_file}")


# Initialize logging when module is imported
def _initialize_logging() -> None:
    """Initialize logging configuration on module import."""
    global log_file_path

    if log_file_path is None:
        try:
            setup_logging()
        except Exception as e:
            print(f"Failed to initialize logging: {e}")
            # Create a basic logger as fallback
            logging.basicConfig(level=logging.INFO)


# Test logging functionality if run directly
if __name__ == "__main__":
    _run_logging_tests()
else:
    # Initialize logging when imported
    _initialize_logging()
