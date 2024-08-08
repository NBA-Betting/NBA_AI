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
- `predictor` (optional, string): Specifies the predictive model to use. Must be one of the valid predictors defined in the config file. Defaults to "Best".
- `detail_level` (optional, string): Specifies the level of detail for the data. Must be "Basic" or "Normal". Defaults to "Basic".
- `update_predictions` (optional, string): Indicates whether to update predictions. Must be "True" or "False". Defaults to "True".

Functions:
- games(): The main endpoint function that handles the retrieval of game data based on the provided query parameters.

Error Handling:
- The API returns appropriate error messages and HTTP status codes for invalid inputs, missing parameters, or unexpected server errors.
"""

from flask import Blueprint, jsonify, request

from src.config import config
from src.games_api.games import get_games, get_games_for_date
from src.utils import (
    date_to_season,
    game_id_to_season,
    validate_date_format,
    validate_game_ids,
)

# Configuration
VALID_PREDICTORS = list(config["predictors"].keys())
VALID_SEASONS = config["api"]["valid_seasons"]
MAX_GAME_IDS = config["api"]["max_game_ids"]

api = Blueprint("api", __name__)


@api.route("/games", methods=["GET"])
def games():
    """
    Retrieve game data based on game IDs or a specific date.

    This endpoint accepts either a list of game IDs or a specific date to fetch game data. It ensures that the input parameters are validated and returns detailed information based on the requested detail level and predictive model.

    Query Parameters:
    - game_ids (str): Comma-separated list of game IDs to retrieve data for. (e.g., "0042300401,0022300649"). Maximum 20 IDs allowed.
    - date (str): Date to retrieve games for, in the format "YYYY-MM-DD". Only the 2023-2024 or 2024-2025 season is allowed.
    - predictor (str, optional): Predictive model to use. Defaults to "Best".
    - detail_level (str, optional): Level of detail for the data. Can be "Basic" or "Normal". Defaults to "Basic".
    - update_predictions (str, optional): Whether to update predictions. Must be "True" or "False". Defaults to "True".

    Returns:
    - JSON response containing game data, or an error message if inputs are invalid.

    Raises:
    - ValueError: For invalid game IDs, date format, or detail level.
    - Exception: For any unexpected server errors.
    """
    try:
        game_ids = request.args.get("game_ids")
        date = request.args.get("date")
        predictor = request.args.get("predictor", "Best")
        detail_level = request.args.get("detail_level", "Basic").lower()
        update_predictions_str = request.args.get("update_predictions", "True").lower()

        # Validate that only one of game_ids or date is provided
        if game_ids and date:
            return (
                jsonify({"error": "Provide either 'game_ids' or 'date', not both."}),
                400,
            )

        # Validate detail_level
        if detail_level not in ["basic", "normal"]:
            return (
                jsonify(
                    {"error": "Invalid detail_level. Must be 'Basic' or 'Normal'."}
                ),
                400,
            )

        # Validate predictor
        if predictor not in VALID_PREDICTORS:
            return (
                jsonify(
                    {
                        "error": f"Invalid predictor. Must be one of: {', '.join(VALID_PREDICTORS)}"
                    }
                ),
                400,
            )

        # Validate update_predictions
        if update_predictions_str not in ["true", "false"]:
            return (
                jsonify(
                    {"error": "Invalid update_predictions. Must be 'True' or 'False'."}
                ),
                400,
            )
        update_predictions = update_predictions_str == "true"

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
                return jsonify({"error": str(ve)}), 400

            # Ensure all game_ids belong to the valid seasons
            seasons = {game_id_to_season(game_id) for game_id in game_ids_list}
            if not seasons.issubset(VALID_SEASONS):
                return (
                    jsonify(
                        {
                            "error": f"All game IDs must belong to the valid seasons: {', '.join(VALID_SEASONS)}"
                        }
                    ),
                    400,
                )

            data = get_games(
                game_ids_list,
                predictor=predictor,
                detail_level=detail_level.capitalize(),  # Normalize to "Basic" or "Normal"
                update_predictions=update_predictions,
            )
        elif date:
            try:
                validate_date_format(date)
            except ValueError as ve:
                return jsonify({"error": str(ve)}), 400

            # Ensure the date belongs to the valid seasons
            if date_to_season(date) not in VALID_SEASONS:
                return (
                    jsonify(
                        {
                            "error": f"Date must be within the valid seasons: {', '.join(VALID_SEASONS)}"
                        }
                    ),
                    400,
                )

            data = get_games_for_date(
                date,
                predictor=predictor,
                detail_level=detail_level.capitalize(),  # Normalize to "Basic" or "Normal"
                update_predictions=update_predictions,
            )
        else:
            return (
                jsonify({"error": "Either 'game_ids' or 'date' must be provided."}),
                400,
            )

        return jsonify(data)
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500
