"""
game_states.py

This module processes NBA play-by-play data to create and save game states.
It consists of functions to:
- Create game states from play-by-play logs for multiple games.
- Save the created game states to a SQLite database.
- Ensure data integrity by handling errors and logging the status of operations.

Functions:
- create_game_states(games_info): Creates a dictionary of game states from play-by-play logs for multiple games.
- save_game_states(game_states, db_path): Saves the game states to the database and updates game data status.
- main(): Handles command-line arguments to fetch, save, and create game states, with optional timing.

Usage:
- Typically run as part of a larger data processing pipeline.
- Script can be run directly from the command line to fetch and save NBA play-by-play data, and create game states:
    python -m src.game_states --save --game_ids=0042300401,0022300649 --log_level=DEBUG
- Successful execution will print the number of games processed, the number of game states created, and the first and last state of each game.
"""

import argparse
import json
import logging
import re
import sqlite3
from copy import deepcopy

from src.config import config
from src.database_updater.pbp import get_pbp, save_pbp
from src.logging_config import setup_logging
from src.utils import (
    log_execution_time,
    lookup_basic_game_info,
    validate_date_format,
    validate_game_ids,
)

# Configuration values
DB_PATH = config["database"]["path"]


@log_execution_time(average_over="games_info")
def create_game_states(games_info):
    """
    Create a dictionary of game states from play-by-play logs for multiple games.

    Parameters:
    games_info (dict): A dictionary where keys are game IDs and values are dictionaries with the keys:
                       - 'home': Home team's tricode
                       - 'away': Away team's tricode
                       - 'date_time_est': Game date in 'YYYY-MM-DDTHH:MM:SS' format
                       - 'pbp_logs': List of dictionaries representing play-by-play logs for the game

    Returns:
    dict: A dictionary where the keys are game IDs and the values are lists of dictionaries representing the game states.
          If an error occurs, an empty dictionary is returned.

    The function handles two types of play-by-play log sources:
    1. Logs from the live endpoint (contains 'orderNumber')
    2. Logs from the stats endpoint on NBA.com (contains 'actionId')
    """

    def duration_to_seconds(duration_str):
        minutes = int(duration_str.split("M")[0][2:])
        seconds = float(duration_str.split("M")[1][:-1])
        return minutes * 60 + seconds

    logging.info(f"Creating game states for {len(games_info)} games")

    game_states = {}

    try:
        for game_id, game_info in games_info.items():
            home = game_info["home"]
            away = game_info["away"]
            game_date = game_info["date_time_est"].split("T")[0]
            logs = game_info["pbp_logs"]

            validate_game_ids(game_id)
            validate_date_format(game_date)

            if not logs:
                logging.warning(f"No play-by-play logs found for game ID {game_id}")
                game_states[game_id] = []
                continue

            # Sort play-by-play logs by period, remaining time (clock), and play ID
            logs = sorted(
                logs,
                key=lambda x: (
                    x["period"],
                    -duration_to_seconds(x["clock"]),
                    x.get("orderNumber", x.get("actionId")),
                ),
            )

            # Filter out logs where 'description' is not a key
            logs = [log for log in logs if "description" in log]

            game_states[game_id] = []
            players = {"home": {}, "away": {}}

            # Check if the first log has orderNumber to decide which logic to use
            if "orderNumber" in logs[0]:
                # Logic for logs from the live endpoint (includes orderNumber)
                for row in logs:
                    if (
                        row.get("personId") is not None
                        and row.get("playerNameI") is not None
                    ):
                        team = "home" if row["teamTricode"] == home else "away"
                        player_id = row["personId"]
                        player_name = row["playerNameI"]

                        if player_id not in players[team]:
                            players[team][player_id] = {
                                "name": player_name,
                                "points": 0,
                            }

                        if row.get("pointsTotal") is not None:
                            points = int(row["pointsTotal"])
                            players[team][player_id]["points"] = points

                    current_game_state = {
                        "game_id": game_id,
                        "play_id": int(row["orderNumber"]),
                        "game_date": game_date,
                        "home": home,
                        "away": away,
                        "clock": row["clock"],
                        "period": int(row["period"]),
                        "home_score": int(row["scoreHome"]),
                        "away_score": int(row["scoreAway"]),
                        "total": int(row["scoreHome"]) + int(row["scoreAway"]),
                        "home_margin": int(row["scoreHome"]) - int(row["scoreAway"]),
                        "is_final_state": row["description"] == "Game End",
                        "players_data": deepcopy(players),
                    }

                    game_states[game_id].append(current_game_state)
            else:
                # Logic for logs from the stats endpoint on NBA.com (includes actionId)
                current_home_score = 0
                current_away_score = 0

                for i, row in enumerate(logs):
                    if row.get("personId") and row.get("playerNameI"):
                        team = "home" if row["teamTricode"] == home else "away"
                        player_id = row["personId"]
                        player_name = row["playerNameI"]

                        if player_id not in players[team]:
                            players[team][player_id] = {
                                "name": player_name,
                                "points": 0,
                            }

                        match = re.search(r"\((\d+) PTS\)", row.get("description", ""))
                        if match:
                            points = int(match.group(1))
                            players[team][player_id]["points"] = points

                    if row.get("scoreHome"):
                        current_home_score = int(row["scoreHome"])
                    if row.get("scoreAway"):
                        current_away_score = int(row["scoreAway"])

                    current_game_state = {
                        "game_id": game_id,
                        "play_id": int(row["actionId"]),
                        "game_date": game_date,
                        "home": home,
                        "away": away,
                        "clock": row["clock"],
                        "period": int(row["period"]),
                        "home_score": current_home_score,
                        "away_score": current_away_score,
                        "total": current_home_score + current_away_score,
                        "home_margin": current_home_score - current_away_score,
                        "is_final_state": (i == len(logs) - 1)
                        and (row.get("subType") == "end"),
                        "players_data": deepcopy(players),
                    }

                    game_states[game_id].append(current_game_state)

    except Exception as e:
        logging.error(f"An error occurred while creating game states: {e}")
        return {}

    logging.info(f"Game states created for {len(game_states)} games")

    for game in game_states:
        logging.debug(f"Game ID: {game} - Number of states: {len(game_states[game])}")
        if game_states[game]:
            logging.debug(f"Game ID: {game} - First state: {game_states[game][0]}")
            logging.debug(f"Game ID: {game} - Last state: {game_states[game][-1]}")

    return game_states


@log_execution_time(average_over="game_states")
def save_game_states(game_states, db_path=DB_PATH):
    """
    Saves the game states to the database and updates Games.game_data_finalized to True if any state has is_final_state = True.
    Each game_id is processed in a separate transaction to ensure all-or-nothing behavior.

    Parameters:
    game_states (dict): A dictionary with game IDs as keys and lists of dictionaries (game states) as values.
    db_path (str): The path to the SQLite database file. Default to DB_PATH from config.

    Returns:
    bool: True if the operation was successful for all game IDs, False otherwise.
    """
    logging.info(f"Saving game states to database: {db_path}")
    overall_success = True

    try:
        with sqlite3.connect(db_path) as conn:
            for game_id, states in game_states.items():
                if not states:
                    logging.info(
                        f"Game ID {game_id} - No game states to save. Skipping."
                    )
                    continue

                try:
                    conn.execute("BEGIN")

                    # Delete existing game states for the game_id
                    conn.execute("DELETE FROM GameStates WHERE game_id = ?", (game_id,))

                    # Insert new game states for the game_id
                    data_to_insert = [
                        (
                            game_id,
                            state["play_id"],
                            state["game_date"],
                            state["home"],
                            state["away"],
                            state["clock"],
                            state["period"],
                            state["home_score"],
                            state["away_score"],
                            state["total"],
                            state["home_margin"],
                            state["is_final_state"],
                            json.dumps(state["players_data"]),
                        )
                        for state in states
                    ]

                    conn.executemany(
                        """
                        INSERT INTO GameStates (game_id, play_id, game_date, home, away, clock, period, home_score, away_score, total, home_margin, is_final_state, players_data)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        data_to_insert,
                    )

                    # Check if any state has is_final_state = True and update Games table accordingly
                    if any(state["is_final_state"] for state in states):
                        conn.execute(
                            """
                            UPDATE Games SET game_data_finalized = 1 WHERE game_id = ?
                            """,
                            (game_id,),
                        )

                    conn.commit()
                except Exception as e:
                    conn.rollback()  # Roll back the transaction if an error occurred
                    logging.error(f"Game ID {game_id} - Error saving game states: {e}")
                    overall_success = False

    except Exception as e:
        logging.error(f"Database connection error: {e}")
        return False

    if overall_success:
        logging.info("Game states saved successfully")
    else:
        logging.error("Some game states were not saved successfully")

    if game_states:
        logging.debug(f"Example record (First): {data_to_insert[0]}")
        logging.debug(f"Example record (Last): {data_to_insert[-1]}")

    return overall_success


def main():
    """
    Main function to handle command-line arguments and orchestrate fetching, saving, and creating game states.
    """
    parser = argparse.ArgumentParser(
        description="Fetch and save NBA play-by-play data and create game states."
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save play-by-play data and game states to database",
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

    args = parser.parse_args()
    log_level = args.log_level.upper()
    setup_logging(log_level=log_level)

    game_ids = args.game_ids.split(",") if args.game_ids else []

    pbp_data = get_pbp(game_ids)
    if args.save:
        save_pbp(pbp_data)

    basic_game_info = lookup_basic_game_info(list(pbp_data.keys()))

    game_state_inputs = {}
    for game_id, game_info in pbp_data.items():
        game_state_inputs[game_id] = {
            "home": basic_game_info[game_id]["home"],
            "away": basic_game_info[game_id]["away"],
            "date_time_est": basic_game_info[game_id]["date_time_est"],
            "pbp_logs": game_info,
        }

    game_states = create_game_states(game_state_inputs)
    if args.save:
        save_game_states(game_states)


if __name__ == "__main__":
    main()
