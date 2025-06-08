"""
Centralized logging configuration for the Discord Voice Translator.

This module sets up comprehensive logging with:
- File logging with rotation for persistent storage
- Console logging for real-time monitoring  
- Proper filtering to control Discord.py verbosity
- Color formatting for better readability
- Performance optimizations for high-frequency operations
"""
import logging
import logging.handlers
import sys
import os
from datetime import datetime
from pathlib import Path

# Color codes for console output


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

    def __init__(self, fmt=None):
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

    def format(self, record):
        """Format log record with appropriate color coding."""
        # Create a copy of the record to avoid modifying the original
        log_color = self.COLORS.get(record.levelno, Colors.RESET)

        # Apply color to the entire message
        formatter = logging.Formatter(
            f"%(asctime)s [{log_color}%(levelname)s{Colors.RESET}] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        return formatter.format(record)


class DiscordFilter(logging.Filter):
    """Filter to control Discord.py logging verbosity."""

    def __init__(self, max_level=logging.WARNING):
        super().__init__()
        self.max_level = max_level

    def filter(self, record):
        """Filter Discord logs based on logger name and level."""
        # Allow all non-Discord logs
        if not record.name.startswith('discord'):
            return True

        # For Discord logs, only allow up to max_level
        return record.levelno >= self.max_level


class AppFilter(logging.Filter):
    """Filter to only allow application logs (non-Discord)."""

    def filter(self, record):
        """Only allow non-Discord logs."""
        return not record.name.startswith('discord')


class UvicornFilter(logging.Filter):
    """Filter to allow application logs and uvicorn, but block Discord."""

    def filter(self, record):
        """Allow non-Discord logs and uvicorn logs."""
        # Block Discord logs
        if record.name.startswith('discord'):
            return False
        # Allow everything else (including uvicorn)
        return True


# Global variable to store the log file path
log_file_path = None


def setup_logging():
    """
    Set up comprehensive logging configuration.

    Creates:
    - File handler with rotation for all application logs
    - Console handler with color formatting and Discord filtering
    - Proper formatters and filters for each handler

    Returns:
        str: Path to the log file
    """
    global log_file_path

    # Create logs directory if it doesn't exist
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    # Generate log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = logs_dir / f"discord_translator_{timestamp}.log"

    # Store globally for later access
    log_file_path = str(log_filename)

    # Clear any existing handlers on root logger
    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    # Set root logger level to DEBUG to capture everything
    root_logger.setLevel(logging.DEBUG)

    # === FILE HANDLER SETUP ===
    # Create rotating file handler (10MB max, 5 backups)
    file_handler = logging.handlers.RotatingFileHandler(
        log_filename,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)  # Capture everything in file

    # File formatter (no colors, more detailed)
    file_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)-8s] %(name)-20s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)

    # CHANGED: Use UvicornFilter instead of AppFilter to allow uvicorn in log file
    file_handler.addFilter(UvicornFilter())

    # === CONSOLE HANDLER SETUP ===
    console_handler = logging.StreamHandler(sys.stdout)
    # Will be filtered by DiscordFilter
    console_handler.setLevel(logging.DEBUG)

    # Console formatter with colors
    console_formatter = ColoredFormatter()
    console_handler.setFormatter(console_formatter)

    # CHANGED: Use UvicornFilter instead of DiscordFilter to allow uvicorn on console
    console_handler.addFilter(UvicornFilter())

    # Add Discord filter to console (only show WARNING+ from Discord)
    console_handler.addFilter(DiscordFilter(
        max_level=logging.ERROR))  # Only ERROR+ from Discord

    # Create a filter to suppress numba debug messages
    class NumbaFilter(logging.Filter):
        def filter(self, record):
            # Suppress all numba debug messages
            if record.name.startswith('numba'):
                return record.levelno >= logging.WARNING
            return True

    console_handler.addFilter(NumbaFilter())

    # === ADD HANDLERS TO ROOT LOGGER ===
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    # === CONFIGURE SPECIFIC LOGGERS ===

    # Discord.py logging - reduce verbosity
    discord_logger = logging.getLogger('discord')
    discord_logger.setLevel(logging.ERROR)  # Only show errors from Discord

    # Asyncio logging - reduce verbosity
    asyncio_logger = logging.getLogger('asyncio')
    asyncio_logger.setLevel(logging.ERROR)

    # urllib3 logging - reduce verbosity
    urllib3_logger = logging.getLogger('urllib3')
    urllib3_logger.setLevel(logging.ERROR)

    # Uvicorn logging - CHANGED: restore INFO level for visibility
    uvicorn_logger = logging.getLogger('uvicorn')
    uvicorn_logger.setLevel(logging.INFO)  # Restored from WARNING to INFO

    uvicorn_access_logger = logging.getLogger('uvicorn.access')
    # Restored from WARNING to INFO
    uvicorn_access_logger.setLevel(logging.INFO)

    # Our application modules - keep at DEBUG
    app_modules = [
        'server', 'bot_manager', 'translator', 'utils', 'custom_sink',
        'transcription', 'translation', 'models', 'audio_utils',
        'websocket_handler', '__main__'
    ]

    for module in app_modules:
        logger = logging.getLogger(module)
        # logger.setLevel(logging.DEBUG)
        logger.setLevel(logging.INFO)

    # Suppress specific third-party loggers
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('websockets').setLevel(logging.WARNING)
    logging.getLogger('aiohttp').setLevel(logging.WARNING)

    # Suppress numba compilation messages completely
    logging.getLogger('numba').setLevel(logging.WARNING)
    logging.getLogger('numba.core').setLevel(logging.WARNING)
    logging.getLogger('numba.core.byteflow').setLevel(logging.ERROR)
    logging.getLogger('numba.core.ssa').setLevel(logging.ERROR)
    logging.getLogger('numba.core.interpreter').setLevel(logging.ERROR)

    # Suppress torio debug messages
    logging.getLogger('torio').setLevel(logging.WARNING)
    logging.getLogger('torio._extension').setLevel(logging.WARNING)
    logging.getLogger('torio._extension.utils').setLevel(logging.WARNING)

    # === LOG STARTUP INFORMATION ===
    startup_logger = logging.getLogger('logging_setup')
    startup_logger.info("🎹 Discord Bot Translator - Logging initialized")
    startup_logger.info("📁 Log file: %s", log_filename)
    startup_logger.info(
        "📊 Console level: DEBUG (Discord filtered, Uvicorn visible), File level: DEBUG (App+Uvicorn)")
    startup_logger.info("🔧 Root logger handlers: %d",
                        len(root_logger.handlers))

    return str(log_filename)


def get_logger(name):
    """
    Get a logger instance for the specified module.

    Args:
        name (str): Name of the module requesting the logger

    Returns:
        logging.Logger: Configured logger instance
    """
    return logging.getLogger(name)


# Test logging functionality if run directly
if __name__ == "__main__":
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

# CRITICAL: Initialize logging immediately when this module is imported
if log_file_path is None:
    try:
        setup_logging()
    except Exception as e:
        print(f"Failed to initialize logging: {e}")
        # Create a basic logger as fallback
        logging.basicConfig(level=logging.INFO)
