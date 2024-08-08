"""
start_app.py

Main entry point for the Flask web application.

This script sets up and launches the Flask web application with configurable
options for the prediction engine, logging level, and Flask debug mode.

Usage:
    python start_app.py --predictor=PredictorName --log_level=INFO --debug

Options:
    --predictor : str, optional
        The predictor to use for predictions. Default is 'Best'.
        Valid options are listed in the configuration.
    --log_level : str, optional
        The log level to use for logging. Options are DEBUG, INFO, WARNING, ERROR, CRITICAL. Default is 'INFO'.
    --debug : bool, optional
        If set, run the application in Flask debug mode. Default is False.
"""

import argparse

from src.config import config
from src.logging_config import setup_logging
from src.web_app.app import create_app

# Configuration
VALID_PREDICTORS = list(config["predictors"].keys())


def main():
    """
    Main entry point for launching the Flask application.

    This function handles the parsing of command-line arguments, sets up logging,
    initializes the Flask app, and runs the app with the specified configuration.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Launch the web app with a specified prediction engine"
    )
    parser.add_argument(
        "--predictor",
        default="Best",
        type=str,
        help=f"The predictor to use for predictions. Default is 'Best'. Valid options are: {', '.join(VALID_PREDICTORS)}",
    )
    parser.add_argument(
        "--log_level",
        default="INFO",
        type=str,
        help="The log level to use for logging. Options are DEBUG, INFO, WARNING, ERROR, CRITICAL.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Run the application in Flask debug mode.",
    )
    args = parser.parse_args()

    predictor = args.predictor
    log_level = args.log_level.upper()
    debug_mode = args.debug

    # Map "Best" to the actual predictor
    if predictor == "Best":
        best_predictor = config["predictors"].get("Best", None)
        if best_predictor:
            predictor = best_predictor
        else:
            raise ValueError("No predictor found for 'Best' in the configuration.")

    # Validate log level and predictor
    valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    if log_level not in valid_log_levels:
        raise ValueError(
            f"Invalid log level: {log_level}. Must be one of {valid_log_levels}."
        )

    if predictor not in VALID_PREDICTORS:
        raise ValueError(
            f"Invalid predictor: {predictor}. Must be one of {VALID_PREDICTORS}."
        )

    # Set up logging
    setup_logging(log_level=log_level)
    # Create the Flask app
    app = create_app(predictor=predictor)

    # Run the app
    app.run(debug=debug_mode)


if __name__ == "__main__":
    main()
