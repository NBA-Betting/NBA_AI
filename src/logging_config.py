"""
logging_config.py

Overview:
This module handles the centralized configuration of logging for the application.
It ensures that logging is set up with a consistent format and level across all modules.
The logging level can be dynamically adjusted based on the needs of the application.

Enhanced with features from database_updater_CM:
- Rotating file handlers to prevent unbounded log growth
- Optional JSON structured logging for easier parsing
- Configurable console/file output
- Better default formatting with full timestamps

Functions:
- setup_logging(log_level, log_file, structured, log_to_console, max_file_size, backup_count):
    Configures the logging settings with advanced options.

Usage:
- Import the setup_logging function in any module where logging needs to be configured.
- Call setup_logging() at the beginning of the script to ensure logging is properly set up.

Example:
    from logging_config import setup_logging

    # Simple usage (backward compatible)
    setup_logging(log_level="DEBUG")

    # Advanced usage with file rotation and JSON logging
    setup_logging(
        log_level="INFO",
        log_file="nba_ai.log",
        structured=True,
        max_file_size=10_000_000,  # 10MB
        backup_count=3
    )
"""

import logging
from logging.handlers import RotatingFileHandler


def setup_logging(
    log_level="INFO",
    log_file=None,
    structured=False,
    log_to_console=True,
    max_file_size=5_000_000,
    backup_count=5,
):
    """
    Sets up logging configuration if not already configured.

    This function configures the logging settings for the application with support for:
    - Console and/or file output
    - Rotating file handlers to manage log file size
    - Optional JSON structured logging
    - Consistent formatting across all modules

    Parameters:
        log_level (str): The logging level to use. Default is "INFO".
            Available levels: "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL".
        log_file (str, optional): Path to log file. If None, only console logging is used.
        structured (bool): Use JSON formatted logs if True. Default is False.
        log_to_console (bool): Log to console if True. Default is True.
        max_file_size (int): Maximum size (in bytes) for log files before rotation. Default is 5MB.
        backup_count (int): Number of backup log files to retain. Default is 5.

    Example:
        # Simple usage (backward compatible)
        setup_logging(log_level="DEBUG")

        # With file output and rotation
        setup_logging(log_level="INFO", log_file="app.log")

        # With JSON logging for production
        setup_logging(
            log_level="INFO",
            log_file="nba_ai.log",
            structured=True,
            max_file_size=10_000_000,
            backup_count=3
        )
    """
    # Check if the logging has already been configured to avoid duplicate handlers
    if not logging.getLogger().hasHandlers():
        # Create formatter
        if structured:
            # Import JSON formatter only if needed
            try:
                from pythonjsonlogger import jsonlogger

                formatter = jsonlogger.JsonFormatter(
                    "%(asctime)s %(name)s %(levelname)s %(message)s"
                )
            except ImportError:
                logging.warning(
                    "pythonjsonlogger not installed, falling back to standard formatting"
                )
                structured = False

        if not structured:
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )

        # Create handlers
        handlers = []

        # Console handler
        if log_to_console:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            handlers.append(console_handler)

        # File handler with rotation
        if log_file:
            file_handler = RotatingFileHandler(
                log_file, maxBytes=max_file_size, backupCount=backup_count
            )
            file_handler.setFormatter(formatter)
            handlers.append(file_handler)

        # Configure logging
        logging.basicConfig(
            level=getattr(logging, log_level),
            handlers=handlers,
        )

        logger = logging.getLogger()
        if log_file:
            logger.info(
                f"Logging system configured (output: console={log_to_console}, file={log_file})"
            )
    else:
        # If logging is already configured, just update the logging level
        logger = logging.getLogger()
        logger.setLevel(getattr(logging, log_level))
        logger.debug(f"Logging level updated to {log_level}")
