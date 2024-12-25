import logging
from logging.handlers import RotatingFileHandler

from pythonjsonlogger import jsonlogger


def setup_logging(
    log_level="INFO",
    log_file="app.log",
    structured=False,
    log_to_console=True,
    max_file_size=5_000_000,
    backup_count=5,
):
    """
    Sets up centralized logging configuration.

    Args:
        log_level (str): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file (str): File path for log file storage.
        structured (bool): Use structured (JSON) logs if True.
        log_to_console (bool): Log to console if True.
        max_file_size (int): Maximum size (in bytes) for rotating log files.
        backup_count (int): Number of backup log files to retain.
    """
    if not logging.getLogger().hasHandlers():
        # Create formatters
        if structured:
            formatter = jsonlogger.JsonFormatter(
                "%(asctime)s %(name)s %(levelname)s %(message)s"
            )
        else:
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )

        # Create handlers
        handlers = []
        if log_to_console:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            handlers.append(console_handler)

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
        logger.info("Logging system configured")
    else:
        logger = logging.getLogger()
        logger.setLevel(getattr(logging, log_level))
        logger.info("Logging level updated to %s", log_level)


# Example of loading this logging setup into a data script
if __name__ == "__main__":
    # Configure the logging system for a data processing script
    setup_logging(
        log_level="DEBUG",
        log_file="data_script.log",
        structured=True,  # Use JSON logs for easier parsing
    )

    # Get a logger for the script
    logger = logging.getLogger("data_script")

    def process_data():
        logger.info("Starting data processing...")
        try:
            # Simulate data processing
            logger.info("Processing data")
            # Simulating an exception to demonstrate traceback logging
            raise ValueError("Simulated error for traceback demonstration")
        except Exception as e:
            logger.error("An error occurred during data processing", exc_info=True)

    # Main execution
    process_data()
