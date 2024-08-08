"""
games.py

This module provides functionality to fetch and process NBA game data from a SQLite database.
It consists of functions to:
- Retrieve basic or detailed game data based on game IDs.
- Retrieve game data for a specific date.
- Update predictions using various predictive models.

Functions:
- get_basic_data(conn, game_ids, predictor_name): Fetch basic game data for specified game IDs.
- get_normal_data(conn, game_ids, predictor_name): Fetch detailed game data including play-by-play logs for specified game IDs.
- get_games(game_ids, predictor="Best", detail_level="Normal", update_predictions=True): Retrieve game data for specified game IDs.
- get_games_for_date(date, predictor="Best", detail_level="Normal", update_predictions=True): Retrieve game data for games on a specific date.
- main(): Main function to handle command-line arguments and invoke appropriate data fetching and output functions.

Usage:
- This script can be run directly to fetch game data and optionally update predictions.
- The output can be directed to a file, printed to the screen, or both, depending on the command-line arguments provided.

Example:
    python -m src.games --date="2024-04-01" --predictor="Best" --output="file" --log_level="DEBUG"
"""

import argparse
import json
import logging
import sqlite3

from src.config import config
from src.database_updater.database_update_manager import update_database
from src.database_updater.predictions import make_current_predictions
from src.logging_config import setup_logging
from src.utils import (
    date_to_season,
    game_id_to_season,
    log_execution_time,
    validate_date_format,
    validate_game_ids,
)

# Configurations
DB_PATH = config["database"]["path"]
VALID_PREDICTORS = list(config["predictors"].keys()) + [None]


def get_basic_data(conn, game_ids, predictor_name):
    """
    Fetch basic game data for the given game IDs.

    Args:
        conn (sqlite3.Connection): Database connection object.
        game_ids (list): List of game IDs to fetch data for.
        predictor_name (str): Name of the predictor to use.

    Returns:
        dict: Dictionary containing game data including the latest game state and pre-game predictions.
    """
    query = """
    WITH LatestGameStates AS (
        SELECT
            s.game_id, s.play_id, s.game_date, s.home, s.away, s.clock, s.period, s.home_score, s.away_score, s.total, s.home_margin, s.is_final_state, s.players_data
        FROM GameStates s
        WHERE s.play_id = (
            SELECT MAX(inner_s.play_id)
            FROM GameStates inner_s
            WHERE inner_s.game_id = s.game_id
        )
    )
    SELECT
        g.game_id, g.date_time_est, g.home_team, g.away_team, g.status, g.season, g.season_type, g.pre_game_data_finalized, g.game_data_finalized,
        s.play_id AS state_play_id, s.game_date, s.home, s.away, s.clock, s.period, s.home_score, s.away_score, s.total, s.home_margin, s.is_final_state, s.players_data,
        pr.predictor, pr.model_id, pr.prediction_datetime, pr.prediction_set
    FROM Games g
    LEFT JOIN LatestGameStates s ON g.game_id = s.game_id
    LEFT JOIN Predictions pr ON g.game_id = pr.game_id AND pr.predictor = ?
    WHERE g.game_id IN ({})
    """.format(
        ",".join("?" * len(game_ids))
    )

    cursor = conn.cursor()
    cursor.execute(query, [predictor_name] + game_ids)
    rows = cursor.fetchall()

    result = {}
    for row in rows:
        game_id = row["game_id"]
        if game_id not in result:
            result[game_id] = {
                "date_time_est": row["date_time_est"],
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "status": row["status"],
                "season": row["season"],
                "season_type": row["season_type"],
                "pre_game_data_finalized": row["pre_game_data_finalized"],
                "game_data_finalized": row["game_data_finalized"],
                "game_states": [],
                "predictions": {"pre_game": {}},
            }

        # Adding the latest game state (only one per game)
        if not result[game_id]["game_states"]:
            game_state = {
                "play_id": row["state_play_id"],
                "game_date": row["game_date"],
                "home": row["home"],
                "away": row["away"],
                "clock": row["clock"],
                "period": row["period"],
                "home_score": row["home_score"],
                "away_score": row["away_score"],
                "total": row["total"],
                "home_margin": row["home_margin"],
                "is_final_state": row["is_final_state"],
                "players_data": (
                    json.loads(row["players_data"]) if row["players_data"] else {}
                ),
            }
            result[game_id]["game_states"].append(game_state)

        # Adding prediction data for the specified predictor
        if row["predictor"] == predictor_name:
            result[game_id]["predictions"]["pre_game"] = {
                "model_id": row["model_id"],
                "prediction_datetime": row["prediction_datetime"],
                "prediction_set": json.loads(row["prediction_set"]),
            }

    return result


def get_normal_data(conn, game_ids, predictor_name):
    """
    Fetch detailed game data, including play-by-play logs, for the given game IDs.

    Args:
        conn (sqlite3.Connection): Database connection object.
        game_ids (list): List of game IDs to fetch data for.
        predictor_name (str): Name of the predictor to use.

    Returns:
        dict: Dictionary containing detailed game data including play-by-play logs, game states, and predictions.
    """
    query = """
    WITH LatestGameStates AS (
        SELECT
            s.game_id, s.play_id, s.game_date, s.home, s.away, s.clock, s.period, s.home_score, s.away_score, s.total, s.home_margin, s.is_final_state, s.players_data
        FROM GameStates s
        WHERE s.play_id = (
            SELECT MAX(inner_s.play_id)
            FROM GameStates inner_s
            WHERE inner_s.game_id = s.game_id
        )
    )
    SELECT
        g.game_id, g.date_time_est, g.home_team, g.away_team, g.status, g.season, g.season_type, g.pre_game_data_finalized, g.game_data_finalized,
        p.play_id, p.log_data,
        s.play_id AS state_play_id, s.game_date, s.home, s.away, s.clock, s.period, s.home_score, s.away_score, s.total, s.home_margin, s.is_final_state, s.players_data,
        pr.predictor, pr.model_id, pr.prediction_datetime, pr.prediction_set
    FROM Games g
    LEFT JOIN PbP_Logs p ON g.game_id = p.game_id
    LEFT JOIN LatestGameStates s ON g.game_id = s.game_id
    LEFT JOIN Predictions pr ON g.game_id = pr.game_id AND pr.predictor = ?
    WHERE g.game_id IN ({})
    ORDER BY p.play_id
    """.format(
        ",".join("?" * len(game_ids))
    )

    cursor = conn.cursor()
    cursor.execute(query, [predictor_name] + game_ids)
    rows = cursor.fetchall()

    result = {}
    for row in rows:
        game_id = row["game_id"]
        if game_id not in result:
            result[game_id] = {
                "date_time_est": row["date_time_est"],
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "status": row["status"],
                "season": row["season"],
                "season_type": row["season_type"],
                "pre_game_data_finalized": row["pre_game_data_finalized"],
                "game_data_finalized": row["game_data_finalized"],
                "play_by_play": [],
                "game_states": [],
                "predictions": {"pre_game": {}},
            }

        # Extracting specific fields from log_data
        if row["log_data"]:
            log_data = json.loads(row["log_data"])
            play_log = {
                "play_id": row["play_id"],
                "period": log_data.get("period"),
                "clock": log_data.get("clock"),
                "scoreHome": log_data.get("scoreHome"),
                "scoreAway": log_data.get("scoreAway"),
                "description": log_data.get("description"),
            }
            result[game_id]["play_by_play"].append(play_log)

        # Adding the latest game state (only one per game)
        if not result[game_id]["game_states"]:
            game_state = {
                "play_id": row["state_play_id"],
                "game_date": row["game_date"],
                "home": row["home"],
                "away": row["away"],
                "clock": row["clock"],
                "period": row["period"],
                "home_score": row["home_score"],
                "away_score": row["away_score"],
                "total": row["total"],
                "home_margin": row["home_margin"],
                "is_final_state": row["is_final_state"],
                "players_data": (
                    json.loads(row["players_data"]) if row["players_data"] else {}
                ),
            }
            result[game_id]["game_states"].append(game_state)

        # Adding prediction data for the specified predictor
        if row["predictor"] == predictor_name:
            result[game_id]["predictions"]["pre_game"] = {
                "model_id": row["model_id"],
                "prediction_datetime": row["prediction_datetime"],
                "prediction_set": json.loads(row["prediction_set"]),
            }

    return result


@log_execution_time(average_over="game_ids")
def get_games(
    game_ids, predictor="Best", detail_level="Normal", update_predictions=True
):
    """
    Retrieve game data for the specified game IDs.

    Args:
        game_ids (list): List of game IDs to fetch data for.
        predictor (str): Name of the predictor to use.
        detail_level (str): Level of detail for the data ("Basic" or "Normal").
        update_predictions (bool): Whether to update the predictions.

    Returns:
        dict: Dictionary containing game data including predictions and game states.
    """
    logging.info(f"Getting game info for {len(game_ids)} games.")
    logging.debug(f"Game IDs: {game_ids}")

    # Validate inputs
    validate_game_ids(game_ids)
    if detail_level not in ["Basic", "Normal"]:
        raise ValueError(f"Invalid detail level: {detail_level}")
    if predictor not in VALID_PREDICTORS:
        raise ValueError(f"Invalid predictor: {predictor}")

    # Update the database
    seasons = set(game_id_to_season(game_id) for game_id in game_ids)
    for season in seasons:
        update_database(season, predictor, DB_PATH)

    # Use context manager to connect to the database
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row

        if detail_level == "Basic":
            data = get_basic_data(conn, game_ids, predictor_name=predictor)
        else:  # detail_level == "Normal"
            data = get_normal_data(conn, game_ids, predictor_name=predictor)

    # Prepare data for updating predictions if required
    if update_predictions:
        games_for_update = {}
        for game_id, game_data in data.items():
            if game_data["predictions"]["pre_game"] and game_data["game_states"]:
                games_for_update[game_id] = {
                    "pre_game_predictions": game_data["predictions"]["pre_game"],
                    "current_game_state": game_data["game_states"][0],
                }

        # Call update_predictions and integrate the current predictions
        if games_for_update:
            current_predictions = make_current_predictions(games_for_update, predictor)
            for game_id, current_prediction_dict in current_predictions.items():
                if game_id in data:
                    data[game_id]["predictions"]["current"] = current_prediction_dict

    logging.info(f"Game info retrieval complete for {len(data)} games.")

    return data


@log_execution_time(average_over="output")
def get_games_for_date(
    date, predictor="Best", detail_level="Normal", update_predictions=True
):
    """
    Retrieve game data for games on a specific date.

    Args:
        date (str): The date to fetch games for (YYYY-MM-DD).
        predictor (str): Name of the predictor to use.
        detail_level (str): Level of detail for the data ("Basic" or "Normal").
        update_predictions (bool): Whether to update the predictions.

    Returns:
        dict: Dictionary containing game data for the specified date.
    """
    logging.info(f"Getting games for date: {date}")

    # Validate inputs
    validate_date_format(date)
    if detail_level not in ["Basic", "Normal"]:
        raise ValueError(f"Invalid detail level: {detail_level}")
    if predictor not in VALID_PREDICTORS:
        raise ValueError(f"Invalid predictor: {predictor}")

    # Update the database
    season = date_to_season(date)
    update_database(season, predictor, DB_PATH)

    # Get game_ids for the given date
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT game_id FROM games WHERE date(date_time_est) = ?",
            (date,),
        )
        game_ids = [row[0] for row in cursor.fetchall()]

    logging.info(f"Found {len(game_ids)} games for date: {date}. Fetching game info...")
    logging.debug(f"Game IDs: {game_ids}")

    # Use the get_games function to get the games
    games = get_games(
        game_ids,
        predictor=predictor,
        detail_level=detail_level,
        update_predictions=update_predictions,
    )

    logging.info(f"Game retrieval complete for {len(games)} games from date: {date}.")

    return games


def main():
    """
    Main function to demonstrate the usage of the get_games and get_games_for_date functions.
    Handles command-line arguments and invokes the appropriate data fetching and output functions.
    """
    parser = argparse.ArgumentParser(
        description="Get games for a list of game IDs or for a specific date."
    )
    parser.add_argument(
        "--game_ids", type=str, help="Comma-separated list of game IDs to process"
    )
    parser.add_argument("--date", type=str, help="The date to get games for.")
    parser.add_argument(
        "--detail_level",
        type=str,
        default="Normal",
        help="The detail level for the data.",
    )
    parser.add_argument(
        "--update_predictions",
        type=bool,
        default=True,
        help="Whether to update the predictions.",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        help="The logging level. Default is INFO. DEBUG provides more details.",
    )
    parser.add_argument(
        "--predictor",
        default="Best",
        type=str,
        help="The predictor to use for predictions.",
    )
    parser.add_argument(
        "--output",
        choices=["file", "screen", "both"],
        default="file",
        help="Where to output the results: 'file' (default), 'screen', or 'both'.",
    )
    parser.add_argument(
        "--output_file", type=str, help="The output file to save the data."
    )

    args = parser.parse_args()
    log_level = args.log_level.upper()
    setup_logging(log_level=log_level)

    game_ids = args.game_ids.split(",") if args.game_ids else None
    date = args.date
    detail_level = args.detail_level
    update_predictions = args.update_predictions
    predictor = args.predictor
    output_choice = args.output

    # Argument validation: Only one of game_ids or date should be provided
    if game_ids and date:
        parser.error("Please provide either --game_ids or --date, but not both.")
    elif not game_ids and not date:
        parser.error("Please provide either --game_ids or --date.")

    if game_ids:
        output_file = args.output_file if args.output_file else "games.json"
        games = get_games(
            game_ids,
            predictor=predictor,
            detail_level=detail_level,
            update_predictions=update_predictions,
        )
    elif date:
        output_file = args.output_file if args.output_file else f"games_{date}.json"
        games = get_games_for_date(
            date,
            predictor=predictor,
            detail_level=detail_level,
            update_predictions=update_predictions,
        )

    # Handle output based on the user's choice
    if output_choice in ["screen", "both"]:
        print(json.dumps(games, indent=4))

    if output_choice in ["file", "both"]:
        with open(output_file, "w") as json_file:
            json.dump(games, json_file, indent=4)


if __name__ == "__main__":
    main()
