"""
api.py

This module provides API endpoints for retrieving NBA game data based on game IDs or a specific date.
It supports detailed validation and error handling for various input parameters to ensure the integrity and accuracy of the data retrieved.

Endpoints:
- `/games`: Retrieves game data either by a list of game IDs or a specific date.

Required Query Parameters:
- `game_ids` (string): Comma-separated list of game IDs to retrieve data for. (e.g., "0042300401,0022300649")
  OR
- `date` (string): Date to retrieve games for, in the format "YYYY-MM-DD".

Optional Query Parameters:
- `predictor` (optional, string): Specifies the predictive model to use. Must be one of the valid predictors defined in the config file. Defaults to default predictor set in the config.

Functions:
- games(): The main endpoint function that handles the retrieval of game data based on the provided query parameters.

Error Handling:
- The API returns appropriate error messages and HTTP status codes for invalid inputs, missing parameters, or unexpected server errors.
"""

import logging

from flask import Blueprint, jsonify, request

from src.config import config
from src.games_api.games import get_games, get_games_for_date
from src.utils import (
    date_to_season,
    determine_current_season,
    game_id_to_season,
    validate_date_format,
    validate_game_ids,
)

# Configuration
VALID_PREDICTORS = list(config["predictors"].keys())
CONFIGURED_SEASONS = config["api"]["valid_seasons"]
MAX_GAME_IDS = config["api"]["max_game_ids"]

api = Blueprint("api", __name__)


def get_valid_seasons():
    """
    Returns the seasons the /games endpoint will serve.

    The list in config.yaml acts as a floor of older seasons to keep serving.
    The current season and the one before it are always added, so the endpoint
    keeps working when a new season tips off instead of rejecting every date in
    it until someone edits config.yaml.

    Evaluated per request rather than once at import, so a server left running
    across the June 30th season boundary picks up the new season on its own.

    Returns:
        list: Seasons in 'XXXX-XXXX' format, sorted ascending.
    """
    seasons = set(CONFIGURED_SEASONS)
    current_season = determine_current_season()
    previous_start_year = int(current_season.split("-")[0]) - 1

    seasons.add(current_season)
    seasons.add(f"{previous_start_year}-{previous_start_year + 1}")

    return sorted(seasons)


@api.route("/games", methods=["GET"])
def games():
    """
    Retrieve game data based on game IDs or a specific date.

    This endpoint accepts either a list of game IDs or a specific date to fetch game data. It ensures that the input parameters are validated and returns detailed information based on the requested detail level and predictive model.

    Query Parameters:
    - game_ids (str): Comma-separated list of game IDs to retrieve data for. (e.g., "0042300401,0022300649"). Maximum 20 IDs allowed.
    - date (str): Date to retrieve games for, in the format "YYYY-MM-DD". Must fall within a valid season (see get_valid_seasons).
    - predictor (str, optional): Predictive model to use. Defaults to the default predictor set in the config.

    Returns:
    - JSON response containing game data, or an error message if inputs are invalid.

    Raises:
    - ValueError: For invalid game IDs, date format, or detail level.
    - Exception: For any unexpected server errors.
    """
    try:
        game_ids = request.args.get("game_ids")
        date = request.args.get("date")
        predictor = request.args.get("predictor")

        # Validate that only one of game_ids or date is provided
        if game_ids and date:
            return (
                jsonify({"error": "Provide either 'game_ids' or 'date', not both."}),
                400,
            )

        # Validate predictor if provided
        if predictor and predictor not in VALID_PREDICTORS:
            return (
                jsonify(
                    {
                        "error": f"Invalid predictor. Must be one of: {', '.join(VALID_PREDICTORS)}"
                    }
                ),
                400,
            )

        if game_ids:
            # Split and validate game_ids
            game_ids_list = game_ids.split(",")
            if len(game_ids_list) > MAX_GAME_IDS:
                return (
                    jsonify(
                        {
                            "error": f"Too many game IDs provided. Maximum allowed is {MAX_GAME_IDS}."
                        }
                    ),
                    400,
                )

            try:
                validate_game_ids(game_ids_list)
            except ValueError as ve:
                logging.warning("Invalid game_ids: %s", ve)
                return jsonify({"error": "Invalid game IDs."}), 400

            # Ensure all game_ids belong to the valid seasons
            valid_seasons = get_valid_seasons()
            seasons = {game_id_to_season(game_id) for game_id in game_ids_list}
            if not seasons.issubset(valid_seasons):
                return (
                    jsonify(
                        {
                            "error": f"All game IDs must belong to the valid seasons: {', '.join(valid_seasons)}"
                        }
                    ),
                    400,
                )

            data = get_games(
                game_ids_list,
                predictor=predictor,
            )
        elif date:
            try:
                validate_date_format(date)
            except ValueError as ve:
                logging.warning("Invalid date format: %s", ve)
                return (
                    jsonify({"error": "Invalid date format. Expected YYYY-MM-DD."}),
                    400,
                )

            # Ensure the date belongs to the valid seasons
            valid_seasons = get_valid_seasons()
            if date_to_season(date) not in valid_seasons:
                return (
                    jsonify(
                        {
                            "error": f"Date must be within the valid seasons: {', '.join(valid_seasons)}"
                        }
                    ),
                    400,
                )

            data = get_games_for_date(
                date,
                predictor=predictor,
            )
        else:
            return (
                jsonify({"error": "Either 'game_ids' or 'date' must be provided."}),
                400,
            )

        return jsonify(data)
    except ValueError as ve:
        logging.warning("Bad API request: %s", ve)
        return jsonify({"error": "Invalid request parameters"}), 400
    except Exception:
        logging.exception("API error")
        return jsonify({"error": "Internal server error"}), 500
