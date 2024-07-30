"""
logging_config.py

Overview:
This module handles the centralized configuration of logging for the application. 
It ensures that logging is set up with a consistent format and level across all modules. 
The logging level can be dynamically adjusted based on the needs of the application.

Functions:
- setup_logging(log_level="INFO"): Configures the logging settings if not already configured.

Usage:
- Import the setup_logging function in any module where logging needs to be configured.
- Call setup_logging() at the beginning of the script to ensure logging is properly set up.

Example:
    from logging_config import setup_logging
    setup_logging(log_level="DEBUG")
"""

import logging


def setup_logging(log_level="INFO"):
    """
    Sets up logging configuration if not already configured.

    This function configures the logging settings for the application. It ensures
    that logging is set up with a consistent format and level across all modules.
    If logging has already been configured, it updates the logging level.

    Parameters:
        log_level (str): The logging level to use. Default is "INFO".
            Available levels: "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL".

    Example:
        setup_logging(log_level="DEBUG")
    """
    # Check if the logging has already been configured to avoid duplicate handlers
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(
            level=getattr(logging, log_level),  # Set the logging level
            format="%(asctime)s-%(filename)s[%(lineno)d]-%(levelname)s: %(message)s",
            datefmt="%H:%M",  # Set the date format to HH:MM
        )
    else:
        # If logging is already configured, just update the logging level
        logging.getLogger().setLevel(getattr(logging, log_level))
