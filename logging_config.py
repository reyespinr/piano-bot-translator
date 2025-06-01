"""
Logging configuration for the Discord Bot Translator.

This module provides centralized logging configuration with appropriate
levels and formatting for production use. It replaces scattered print
statements with proper logging that can be controlled by environment
variables or configuration.
"""
import logging
import os
import sys
from typing import Optional


class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to log levels in console output."""

    # Color codes for different log levels
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
    RESET = '\033[0m'

    def format(self, record):
        # Add color to the log level name
        if record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.RESET}"
        return super().format(record)


def setup_logging(log_level: Optional[str] = None) -> logging.Logger:
    """
    Set up logging configuration for the application.

    Args:
        log_level: Optional log level override. If not provided, uses environment
                  variable LOG_LEVEL or defaults to INFO.

    Returns:
        Logger instance configured for the application.
    """
    # Determine log level
    if log_level is None:
        log_level = os.getenv('LOG_LEVEL', 'INFO').upper()

    # Create logger
    logger = logging.getLogger('discord_translator')
    logger.setLevel(getattr(logging, log_level, logging.INFO))

    # Clear any existing handlers
    logger.handlers.clear()

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logger.level)

    # Create formatter
    formatter = ColoredFormatter(
        fmt='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)

    # Add handler to logger
    logger.addHandler(console_handler)

    # Prevent duplicate logs from propagating to root logger
    logger.propagate = False

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.

    Args:
        name: Name of the module (usually __name__)

    Returns:
        Logger instance for the module.
    """
    return logging.getLogger(f'discord_translator.{name}')


# Initialize the main logger
main_logger = setup_logging()
