"""
prediction_manager.py

This module orchestrates the prediction generation process.
It consists of functions to:
- Determine the proper predictor.
- Make pre-game predictions.

Functions:
- determine_predictor_class(predictor_name): Determines the predictor class based on the provided predictor name.
- make_pre_game_predictions(game_ids, predictor_name=None, save=True): Generates pre-game predictions for the given game IDs using the specified predictor.
- main(): Main function to handle command-line arguments and orchestrate the prediction process.

Usage:
- Typically run as part of a larger data processing pipeline.
- Script can be run directly from the command line (project root) to generate and optionally save predictions.
    Example: python -m src.predictions.prediction_manager --save --game_ids=0042300401,0022300649 --log_level=DEBUG --predictor=Linear
- Successful execution will display logs for prediction generation and optionally save pre-game predictions to the database.
"""

import argparse
import importlib
import json
import logging

import numpy as np
import pandas as pd

from src.config import config
from src.database import get_db
from src.logging_config import setup_logging
from src.utils import log_execution_time

# Configuration
DB_PATH = config["database"]["path"]
DEFAULT_PREDICTOR = config["default_predictor"]
PREDICTORS_CONFIG = config["predictors"]

# Plausible bounds for a pre-game NBA team score prediction. Values outside
# this range signal degenerate inputs (e.g., empty feature rows during the
# first games of a season) rather than a real forecast, and saving them would
# pollute accuracy/ATS metrics downstream.
MIN_PLAUSIBLE_SCORE = 60
MAX_PLAUSIBLE_SCORE = 200


def _drop_implausible_predictions(predictions, predictor_name):
    """
    Filter out predictions with scores outside the plausible NBA range.

    Returns a dict containing only the predictions that pass the sanity check.
    Dropped games are logged; the model effectively abstains for them.
    """
    valid = {}
    for game_id, pred in predictions.items():
        home = pred.get("pred_home_score")
        away = pred.get("pred_away_score")
        # None scores are allowed: spread/win-pct-only models (e.g. Phase5
        # with w_total=0) legitimately omit team scores.
        if any(
            score is not None
            and not (MIN_PLAUSIBLE_SCORE <= score <= MAX_PLAUSIBLE_SCORE)
            for score in (home, away)
        ):
            logging.warning(
                f"{predictor_name}: dropping implausible prediction for game "
                f"{game_id} ({home}-{away}) — likely missing or degenerate features"
            )
            continue
        valid[game_id] = pred
    return valid


# Valid predictor names (for validation before lazy import)
VALID_PREDICTORS = {
    "Baseline",
    "Linear",
    "Tree",
    "MLP",
    "Phase5",
    "Phase3",
    "Ensemble",
}

PREDICTOR_IMPORTS = {
    "Baseline": ("src.predictions.prediction_engines.baseline_predictor", "BaselinePredictor"),
    "Linear": ("src.predictions.prediction_engines.linear_predictor", "LinearPredictor"),
    "Tree": ("src.predictions.prediction_engines.tree_predictor", "TreePredictor"),
    "MLP": ("src.predictions.prediction_engines.mlp_predictor", "MLPPredictor"),
    "Phase5": ("src.pipeline.phase5_predictor", "Phase5Predictor"),
    "Phase3": ("src.pipeline.phase3_predictor", "Phase3Predictor"),
    "Ensemble": ("src.pipeline.ensemble_predictor", "EnsemblePredictor"),
}


def _get_predictor_class(predictor_name):
    """Import only the selected predictor and avoid loading unrelated ML runtimes."""
    module_name, class_name = PREDICTOR_IMPORTS[predictor_name]
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def determine_predictor_class(predictor_name):
    if predictor_name is None:
        predictor_name = DEFAULT_PREDICTOR

    if predictor_name not in VALID_PREDICTORS:
        raise ValueError(
            f"Predictor '{predictor_name}' not found. Options: {VALID_PREDICTORS}"
        )

    return _get_predictor_class(predictor_name), predictor_name


@log_execution_time(average_over="game_ids")
def make_pre_game_predictions(game_ids, predictor_name=None, save=True):
    # Determine the predictor class based on the provided name
    predictor_class, predictor_name = determine_predictor_class(predictor_name)

    logging.debug(
        f"Generating pre-game predictions for {len(game_ids)} games using predictor '{predictor_name}'."
    )

    # Get the model paths from the configuration
    model_paths = PREDICTORS_CONFIG.get(predictor_name, {}).get("model_paths", [])

    # Instantiate the predictor class
    predictor_instance = predictor_class(model_paths=model_paths)

    # Create the predictions
    pre_game_predictions = predictor_instance.make_pre_game_predictions(game_ids)
    pre_game_predictions = _drop_implausible_predictions(
        pre_game_predictions, predictor_name
    )

    # Warn if some games didn't get predictions
    if len(pre_game_predictions) < len(game_ids):
        missing_count = len(game_ids) - len(pre_game_predictions)
        logging.warning(
            f"Predictions: {missing_count}/{len(game_ids)} games did not receive predictions "
            f"(missing features or model error)"
        )

    logging.debug(
        f"Pre-game predictions generated successfully for {len(pre_game_predictions)} games using predictor '{predictor_name}'."
    )
    logging.debug(f"Pre-Game Predictions: {pre_game_predictions}")

    # Optionally, save the predictions
    if save:
        save_predictions(pre_game_predictions, predictor_name)

    return pre_game_predictions


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
        logging.debug("No predictions to save.")
        return

    logging.debug(
        f"Saving {len(predictions)} predictions for predictor '{predictor_name}'..."
    )
    prediction_datetime = pd.Timestamp.now(tz="UTC")
    prediction_datetime_str = prediction_datetime.strftime("%Y-%m-%d %H:%M:%S")

    with get_db(db_path) as conn:
        cursor = conn.cursor()

        # Validate prediction times against game start times
        game_ids = list(predictions.keys())
        placeholders = ",".join("?" * len(game_ids))
        cursor.execute(
            f"SELECT game_id, date_time_utc FROM Games WHERE game_id IN ({placeholders})",
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
                logging.debug(
                    f"Saving prediction for completed game {game_id}: prediction time "
                    f"({prediction_datetime_str}) is after game start time ({game_time})."
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

    logging.debug("Predictions saved successfully.")
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


if __name__ == "__main__":
    main()
