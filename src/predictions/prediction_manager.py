"""
prediction_manager.py

This module orchestrates the prediction generation process.
It consists of functions to:
- Determine the proper predictor.
- Make pre-game predictions.
- Make current predictions.

Functions:
- determine_predictor_class(predictor_name): Determines the predictor class based on the provided predictor name.
- make_pre_game_predictions(game_ids, predictor_name=None, save=True): Generates pre-game predictions for the given game IDs using the specified predictor.
- make_current_predictions(game_ids, predictor_name=None): Generates current predictions for the given game IDs using the specified predictor.
- main(): Main function to handle command-line arguments and orchestrate the prediction process.

Usage:
- Typically run as part of a larger data processing pipeline.
- Script can be run directly from the command line (project root) to generate and optionally save predictions.
    Example: python -m src.predictions.prediction_manager --save --game_ids=0042300401,0022300649 --log_level=DEBUG --predictor=Linear
- Successful execution will display logs for prediction generation and optionally save pre-game predictions to the database.
"""

import argparse
import json
import logging
import sqlite3

import numpy as np
import pandas as pd

from src.config import config
from src.logging_config import setup_logging
from src.predictions.prediction_engines.baseline_predictor import BaselinePredictor
from src.predictions.prediction_engines.ensemble_predictor import EnsemblePredictor
from src.predictions.prediction_engines.linear_predictor import LinearPredictor
from src.predictions.prediction_engines.mlp_predictor import MLPPredictor
from src.predictions.prediction_engines.tree_predictor import TreePredictor
from src.utils import log_execution_time

# Configuration
DB_PATH = config["database"]["path"]
DEFAULT_PREDICTOR = config["default_predictor"]
PREDICTORS_CONFIG = config["predictors"]

# Define the PREDICTOR_MAP with actual class references
PREDICTOR_MAP = {
    "Baseline": BaselinePredictor,
    "Linear": LinearPredictor,
    "Tree": TreePredictor,
    "MLP": MLPPredictor,
    "Ensemble": EnsemblePredictor,
}


def determine_predictor_class(predictor_name):
    if predictor_name is None:
        predictor_name = DEFAULT_PREDICTOR

    if predictor_name not in PREDICTOR_MAP:
        raise ValueError(
            f"Predictor '{predictor_name}' not found in PREDICTOR_MAP. Current options include: {PREDICTOR_MAP.keys()}"
        )

    return PREDICTOR_MAP[predictor_name], predictor_name


@log_execution_time(average_over="game_ids")
def make_pre_game_predictions(game_ids, predictor_name=None, save=True):
    # Determine the predictor class based on the provided name
    predictor_class, predictor_name = determine_predictor_class(predictor_name)

    logging.info(
        f"Generating pre-game predictions for {len(game_ids)} games using predictor '{predictor_name}'."
    )

    # Get the model paths from the configuration
    model_paths = PREDICTORS_CONFIG.get(predictor_name, {}).get("model_paths", [])

    # Instantiate the predictor class
    predictor_instance = predictor_class(model_paths=model_paths)

    # Create the predictions
    pre_game_predictions = predictor_instance.make_pre_game_predictions(game_ids)

    logging.info(
        f"Pre-game predictions generated successfully for {len(pre_game_predictions)} games using predictor '{predictor_name}'."
    )
    logging.debug(f"Pre-Game Predictions: {pre_game_predictions}")

    # Optionally, save the predictions
    if save:
        save_predictions(pre_game_predictions, predictor_name)

    return pre_game_predictions


@log_execution_time(average_over="game_ids")
def make_current_predictions(game_ids, predictor_name=None):
    """
    Generate current predictions by blending pre-game predictions with current game state.

    This function loads pre-game predictions from the database and blends them with
    the current game state (score, clock, period) using a weighted averaging formula.

    For games in progress:
    - Blends pre-game prediction with extrapolated current score based on pace
    - Weight shifts from pre-game prediction to current score as game progresses

    For completed games:
    - Returns actual final scores as prediction

    Args:
        game_ids (list): List of game IDs to generate predictions for.
        predictor_name (str, optional): Name of predictor (used to load pre-game predictions).
                                       Defaults to DEFAULT_PREDICTOR.

    Returns:
        dict: Dictionary mapping game_id to current prediction dict.

    Note:
        This function does NOT use ML models - it uses formula-based blending regardless
        of predictor type. ML models are only used for pre-game predictions.
    """
    from src.predictions.prediction_utils import (
        load_current_game_data,
        update_predictions,
    )

    # Determine predictor name
    if predictor_name is None:
        predictor_name = DEFAULT_PREDICTOR

    if predictor_name not in PREDICTOR_MAP:
        raise ValueError(
            f"Predictor '{predictor_name}' not found in PREDICTOR_MAP. Current options include: {PREDICTOR_MAP.keys()}"
        )

    logging.info(
        f"Generating current predictions for {len(game_ids)} games using predictor '{predictor_name}'."
    )

    # Load pre-game predictions from DB and current game state
    if not game_ids:
        return {}

    games = load_current_game_data(game_ids, predictor_name)

    # Blend pre-game predictions with current state using formula
    current_predictions = update_predictions(games)

    logging.info(
        f"Current predictions generated successfully for {len(current_predictions)} games using predictor '{predictor_name}'."
    )
    logging.debug(f"Current Predictions: {current_predictions}")

    return current_predictions


@log_execution_time(average_over="predictions")
def save_predictions(predictions, predictor_name, db_path=DB_PATH):
    """
    Save predictions to the Predictions table.
    Validates that predictions are made before game start time.

    Parameters:
    predictions (dict): The predictions to save.
    predictor_name (str): The name of the predictor.
    db_path (str): The path to the SQLite database file. Defaults to DB_PATH from config.

    Returns:
    None

    Raises:
    ValueError: If prediction is made after game start time.
    """
    if not predictions:
        logging.info("No predictions to save.")
        return

    logging.info(
        f"Saving {len(predictions)} predictions for predictor '{predictor_name}'..."
    )
    prediction_datetime = pd.Timestamp.now(tz="UTC")
    prediction_datetime_str = prediction_datetime.strftime("%Y-%m-%d %H:%M:%S")

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()

        # Validate prediction times against game start times
        game_ids = list(predictions.keys())
        placeholders = ",".join("?" * len(game_ids))
        cursor.execute(
            f"SELECT game_id, date_time_est FROM Games WHERE game_id IN ({placeholders})",
            game_ids,
        )
        game_times = {
            row[0]: pd.to_datetime(row[1], utc=True) for row in cursor.fetchall()
        }

        # Check each game
        for game_id in game_ids:
            if game_id not in game_times:
                logging.warning(
                    f"Game {game_id} not found in database - skipping time validation"
                )
                continue

            game_time = game_times[game_id]
            time_until_game = (game_time - prediction_datetime).total_seconds() / 60

            if time_until_game < 0:
                # Allow predictions for past games (for historical analysis)
                # but log a warning
                logging.debug(
                    f"Saving prediction for completed game {game_id}: prediction time "
                    f"({prediction_datetime_str}) is after game start time ({game_time})."
                )

            elif time_until_game < 10:
                logging.warning(
                    f"Prediction for game {game_id} made within {time_until_game:.1f} minutes "
                    f"of game start - very tight window!"
                )

        data = [
            (
                game_id,
                predictor_name,
                prediction_datetime_str,
                json.dumps(
                    {
                        k: (
                            float(v)
                            if isinstance(v, (np.float32, np.float64, np.int64))
                            else v
                        )
                        for k, v in predictions[game_id].items()
                    }
                ),
            )
            for game_id in predictions.keys()
        ]

        cursor.executemany(
            """
            INSERT OR REPLACE INTO Predictions (game_id, predictor, prediction_datetime, prediction_set)
            VALUES (?, ?, ?, ?)
            """,
            data,
        )

        conn.commit()

    logging.info("Predictions saved successfully.")
    if data:
        logging.debug(f"Example record: {data[0]}")


def main():
    """
    Main function to handle command-line arguments and orchestrate the prediction process.
    """
    parser = argparse.ArgumentParser(
        description="Generate predictions for NBA games using various predictive models."
    )
    parser.add_argument(
        "--game_ids", type=str, help="Comma-separated list of game IDs to process"
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        help="The logging level. Default is INFO. DEBUG provides more details.",
    )
    parser.add_argument(
        "--save", action="store_true", help="Save pre-game predictions to database."
    )
    parser.add_argument(
        "--predictor",
        type=str,
        help="The predictor to use for predictions.",
    )

    args = parser.parse_args()
    log_level = args.log_level.upper()
    setup_logging(log_level=log_level)

    game_ids = args.game_ids.split(",") if args.game_ids else []

    # Generate predictions using the specified predictor
    pre_game_predictions = make_pre_game_predictions(
        game_ids, args.predictor, save=args.save  # Explicitly set save to args.save
    )

    # Create predictions based on the current game state
    current_predictions = make_current_predictions(game_ids, args.predictor)


if __name__ == "__main__":
    main()
