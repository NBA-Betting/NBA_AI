"""
games.py

This module provides functionality to fetch and process NBA game data from a SQLite database.
It consists of functions to:
- Retrieve detailed game data based on game IDs.
- Retrieve game data for a specific date.

Functions:
- get_normal_data(conn, game_ids, predictor_name): Fetch detailed game data including play-by-play logs for specified game IDs.
- get_games(game_ids, predictor=DEFAULT_PREDICTOR): Retrieve game data for specified game IDs.
- get_games_for_date(date, predictor=DEFAULT_PREDICTOR): Retrieve game data for games on a specific date.
- main(): Main function to handle command-line arguments and invoke appropriate data fetching and output functions.

Usage:
- This script can be run directly to fetch game data.
- The output can be directed to a file, printed to the screen, or both, depending on the command-line arguments provided.

Example:
    python -m src.games_api.games --date="2024-04-01" --predictor="Baseline" --output="file" --log_level="DEBUG"
"""

import argparse
import json
import logging
import sqlite3

from src.config import config
from src.database import get_db
from src.logging_config import setup_logging
from src.utils import (
    log_execution_time,
    validate_date_format,
    validate_game_ids,
)

# Configurations
VALID_PREDICTORS = list(config["predictors"].keys()) + [None]
DEFAULT_PREDICTOR = config["default_predictor"]


@log_execution_time(average_over="game_ids")
def get_normal_data(conn, game_ids, predictor_name, pbp_limit=50):
    """
    Fetch detailed game data, including play-by-play logs, for the given game IDs.

    Uses optimized queries:
    - ROW_NUMBER() pattern for latest game states (avoids correlated subquery)
    - Separate PBP query with configurable limit (avoids row multiplication)

    Args:
        conn (sqlite3.Connection): Database connection object.
        game_ids (list): List of game IDs to fetch data for.
        predictor_name (str): Name of the predictor to use.
        pbp_limit (int): Maximum number of PBP entries per game (default 50).

    Returns:
        dict: Dictionary containing detailed game data including play-by-play logs, game states, and predictions.
    """
    placeholders = ",".join("?" * len(game_ids))

    # Query 1: Games, LatestGameStates, and Predictions (no PBP join)
    # Uses ROW_NUMBER() instead of correlated subquery for better performance
    main_query = f"""
    WITH LatestGameStates AS (
        SELECT
            game_id, play_id, game_date, home, away, clock, period,
            home_score, away_score, total, home_margin, is_final_state, players_data,
            ROW_NUMBER() OVER (PARTITION BY game_id ORDER BY play_id DESC) AS rn
        FROM GameStates
        WHERE game_id IN ({placeholders})
    )
    SELECT
        g.game_id, g.date_time_utc, g.home_team, g.away_team, g.status, g.status_text, g.season,
        g.season_type, g.pre_game_data_finalized, g.game_data_finalized,
        s.play_id AS state_play_id, s.game_date, s.home, s.away, s.clock, s.period,
        s.home_score, s.away_score, s.total, s.home_margin, s.is_final_state, s.players_data,
        pr.predictor, pr.prediction_datetime, pr.prediction_set,
        b.espn_opening_spread
    FROM Games g
    LEFT JOIN LatestGameStates s ON g.game_id = s.game_id AND s.rn = 1
    LEFT JOIN Predictions pr ON g.game_id = pr.game_id AND pr.predictor = ?
    LEFT JOIN Betting b ON g.game_id = b.game_id
    WHERE g.game_id IN ({placeholders})
    """

    cursor = conn.cursor()
    cursor.execute(main_query, game_ids + [predictor_name] + game_ids)
    rows = cursor.fetchall()

    result = {}
    for row in rows:
        game_id = row["game_id"]
        result[game_id] = {
            "date_time_utc": row["date_time_utc"],
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "status": row["status"],
            "status_text": row["status_text"],
            "season": row["season"],
            "season_type": row["season_type"],
            "pre_game_data_finalized": row["pre_game_data_finalized"],
            "game_data_finalized": row["game_data_finalized"],
            "play_by_play": [],
            "game_states": [],
            "predictions": {"pre_game": {}},
            "opening_spread": row["espn_opening_spread"],
        }

        # Add the latest game state (only one per game from CTE)
        if row["state_play_id"] is not None:
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

        # Add prediction data for the specified predictor
        if row["predictor"] == predictor_name and row["prediction_set"] is not None:
            result[game_id]["predictions"]["pre_game"] = {
                "prediction_datetime": row["prediction_datetime"],
                "prediction_set": json.loads(row["prediction_set"]),
            }

    # Query 2: PBP logs (separate query, limited per game)
    # Uses ROW_NUMBER() to get only the most recent N plays per game
    pbp_query = f"""
    WITH RankedPbP AS (
        SELECT
            game_id, play_id, log_data,
            ROW_NUMBER() OVER (PARTITION BY game_id ORDER BY play_id DESC) AS rn
        FROM PbP_Logs
        WHERE game_id IN ({placeholders})
    )
    SELECT game_id, play_id, log_data
    FROM RankedPbP
    WHERE rn <= ?
    ORDER BY game_id, play_id DESC
    """

    cursor.execute(pbp_query, game_ids + [pbp_limit])
    pbp_rows = cursor.fetchall()

    for row in pbp_rows:
        game_id = row["game_id"]
        if game_id in result and row["log_data"]:
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

    return result


@log_execution_time(average_over="game_ids")
def get_games(
    game_ids,
    predictor=DEFAULT_PREDICTOR,
):
    """
    Retrieve game data for the specified game IDs.

    Args:
        game_ids (list): List of game IDs to fetch data for.
        predictor (str): Name of the predictor to use.

    Returns:
        dict: Dictionary containing game data including predictions and game states.
    """
    logging.debug(f"Getting game info for {len(game_ids)} games.")
    logging.debug(f"Game IDs: {game_ids}")

    # Validate inputs
    validate_game_ids(game_ids)
    if predictor not in VALID_PREDICTORS:
        raise ValueError(f"Invalid predictor: {predictor}")

    # Use context manager to connect to the database
    with get_db() as conn:
        conn.row_factory = sqlite3.Row

        data = get_normal_data(conn, game_ids, predictor_name=predictor)

    logging.debug(f"Game info retrieval complete for {len(data)} games.")

    return data


@log_execution_time(average_over="output")
def get_games_for_date(date, predictor=DEFAULT_PREDICTOR):
    """
    Retrieve game data for games on a specific date.

    Args:
        date (str): The date to fetch games for (YYYY-MM-DD).
        predictor (str): Name of the predictor to use.

    Returns:
        dict: Dictionary containing game data for the specified date.
    """
    logging.debug(f"Getting games for date: {date}")

    # Validate inputs
    validate_date_format(date)
    if predictor not in VALID_PREDICTORS:
        raise ValueError(f"Invalid predictor: {predictor}")

    # Get game_ids for the given date
    # The date parameter represents the NBA schedule date (which uses Eastern Time)
    # We need to find all games scheduled for that date in ET
    # Convert the date to a UTC datetime range covering the entire ET day
    from datetime import datetime, timedelta, timezone

    import pytz

    eastern = pytz.timezone("US/Eastern")

    # Start of day in ET (00:00:00 ET)
    start_of_day_et = eastern.localize(datetime.strptime(date, "%Y-%m-%d"))
    # End of day in ET (next day at 00:00:00 ET)
    end_of_day_et = start_of_day_et + timedelta(days=1)

    # Convert to UTC for database query (use ISO format with T separator to match DB format)
    start_utc = start_of_day_et.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    end_utc = end_of_day_et.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")

    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT game_id FROM games WHERE date_time_utc >= ? AND date_time_utc < ?",
            (start_utc, end_utc),
        )
        game_ids = [row[0] for row in cursor.fetchall()]

    logging.debug(
        f"Found {len(game_ids)} games for date: {date}. Fetching game info..."
    )
    logging.debug(f"Game IDs: {game_ids}")

    # Use the get_games function to get the games
    games = get_games(
        game_ids,
        predictor=predictor,
    )

    logging.debug(f"Game retrieval complete for {len(games)} games from date: {date}.")

    return games


def get_game_counts_by_date(month):
    """
    Counts the games on each date of a month, for the calendar date picker.

    Games are bucketed by their Eastern Time date, matching how
    get_games_for_date selects games, so the calendar can never disagree with
    the table it navigates to.

    Args:
        month (str): The month to count games for, as 'YYYY-MM'.

    Returns:
        dict: Maps 'YYYY-MM-DD' to the number of games on that date. Dates with
              no games are omitted.

    Raises:
        ValueError: If the month is not in 'YYYY-MM' format.
    """
    from datetime import datetime, timedelta, timezone

    import pytz

    try:
        year_str, month_str = month.split("-")
        year, month_number = int(year_str), int(month_str)
        first_day = datetime(year, month_number, 1)
    except (AttributeError, ValueError):
        raise ValueError("Invalid month format. Please use YYYY-MM format.")

    eastern = pytz.timezone("US/Eastern")

    # The ET day boundaries of the month, converted to the UTC range stored in the DB.
    start_of_month_et = eastern.localize(first_day)
    if month_number == 12:
        next_month = datetime(year + 1, 1, 1)
    else:
        next_month = datetime(year, month_number + 1, 1)
    end_of_month_et = eastern.localize(next_month)

    start_utc = start_of_month_et.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    end_utc = end_of_month_et.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")

    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT date_time_utc FROM Games WHERE date_time_utc >= ? AND date_time_utc < ?",
            (start_utc, end_utc),
        )
        rows = cursor.fetchall()

    counts = {}
    for (date_time_utc,) in rows:
        # Convert each game's UTC tip-off back to the ET date it belongs to.
        parsed = datetime.strptime(date_time_utc[:19], "%Y-%m-%dT%H:%M:%S")
        game_date = parsed.replace(tzinfo=timezone.utc).astimezone(eastern).date()
        date_key = game_date.strftime("%Y-%m-%d")
        counts[date_key] = counts.get(date_key, 0) + 1

    return counts


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
        "--log_level",
        type=str,
        default="INFO",
        help="The logging level. Default is INFO. DEBUG provides more details.",
    )
    parser.add_argument(
        "--predictor",
        default=DEFAULT_PREDICTOR,
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
        )
    elif date:
        output_file = args.output_file if args.output_file else f"games_{date}.json"
        games = get_games_for_date(
            date,
            predictor=predictor,
        )

    # Handle output based on the user's choice
    if output_choice in ["screen", "both"]:
        print(json.dumps(games, indent=4))

    if output_choice in ["file", "both"]:
        with open(output_file, "w") as json_file:
            json.dump(games, json_file, indent=4)


if __name__ == "__main__":
    main()
