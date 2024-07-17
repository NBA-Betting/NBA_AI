"""
pbp.py

This module fetches and saves NBA play-by-play data for specified game IDs.
It consists of functions to:
- Fetch play-by-play data from the NBA API.
- Validate and save the data to a SQLite database.
- Ensure data integrity by checking for empty or corrupted data before updating the database.

Functions:
- fetch_game_data(session, base_url, headers, game_id): Fetches play-by-play data for a specific game.
- get_pbp(game_ids): Fetches play-by-play data for a list of games concurrently.
- save_pbp(pbp_data, db_path): Saves the fetched play-by-play data to the database.
- main(): Handles command-line arguments to fetch and/or save play-by-play data, with optional timing.

Usage:
- Typically run as part of a larger data collection pipeline.
- Script can be run directly from the command line (project root) to fetch and save NBA play-by-play data:
    python -m src.pbp --save --game_ids=0042300401,0022300649 --log_level=DEBUG
- Successful execution will print the number of games fetched, the number of actions in each game, and the first and last actions in each game.
"""

import argparse
import json
import logging
import os
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

from src.config import config
from src.logging_config import setup_logging
from src.utils import log_execution_time, requests_retry_session, validate_game_ids

# Configuration values
DB_PATH = config["database"]["path"]
NBA_API_BASE_URL = config["nba_api"]["pbp_endpoint"]
NBA_API_HEADERS = config["nba_api"]["pbp_headers"]


def fetch_game_data(session, base_url, headers, game_id):
    """
    Fetches game data from a given URL and sorts the actions based on the action number.

    Parameters:
    session (requests.Session): The session to use for making the HTTP request.
    base_url (str): The base URL to fetch the game data from. Should contain a placeholder for the game_id.
    headers (dict): The headers to include in the HTTP request.
    game_id (str): The ID of the game to fetch data for.

    Returns:
    tuple: A tuple containing the game_id and a list of sorted actions. If an error occurs, the list of actions will be empty.
    """

    def duration_to_seconds(duration_str):
        minutes = int(duration_str.split("M")[0][2:])
        seconds = float(duration_str.split("M")[1][:-1])
        return minutes * 60 + seconds

    try:
        response = session.get(base_url.format(game_id), headers=headers, timeout=10)
        response.raise_for_status()

        data = response.json()
        actions = data.get("game", {}).get("actions", [])

        actions_sorted = sorted(
            actions,
            key=lambda x: (
                x["period"],
                -duration_to_seconds(x["clock"]),
                x["orderNumber"],
            ),
        )

        return game_id, actions_sorted

    except requests.exceptions.HTTPError as http_err:
        logging.info(f"Game ID {game_id} - HTTP error: {http_err}")
    except Exception as e:
        logging.warning(f"Game ID {game_id} - API call error: {e}")

    return game_id, []


@log_execution_time(average_over="game_ids")
def get_pbp(game_ids):
    """
    Fetches play-by-play data for a list of games concurrently.

    Parameters:
    game_ids (list or str): A list of game IDs to fetch data for, or a single game ID.

    Returns:
    dict: A dictionary mapping game IDs to lists of sorted actions. If an error occurs when fetching data for a game, the list of actions will be empty.
    """
    logging.info(f"Fetching play-by-play data for {len(game_ids)} games.")
    if isinstance(game_ids, str):
        game_ids = [game_ids]

    validate_game_ids(game_ids)

    thread_pool_size = min(32, os.cpu_count() * 5)
    results = {}

    with requests_retry_session() as session:
        with ThreadPoolExecutor(max_workers=thread_pool_size) as executor:
            futures = [
                executor.submit(
                    fetch_game_data, session, NBA_API_BASE_URL, NBA_API_HEADERS, game_id
                )
                for game_id in game_ids
            ]
            for future in as_completed(futures):
                game_id, actions_sorted = future.result()
                results[game_id] = actions_sorted if actions_sorted else []

    logging.info(f"Fetched play-by-play data for {len(results)} games.")
    for game in results:
        first_action = results[game][0] if results[game] else "No actions"
        last_action = results[game][-1] if results[game] else "No actions"
        logging.debug(f"Game ID: {game} - Number of actions: {len(results[game])}")
        logging.debug(f"Game ID: {game} - First action: {first_action}")
        logging.debug(f"Game ID: {game} - Last action: {last_action}")

    return results


@log_execution_time(average_over="pbp_data")
def save_pbp(pbp_data, db_path=DB_PATH):
    """
    Saves the play-by-play logs to the database. Each game_id is processed in a separate transaction to ensure all-or-nothing behavior.

    Parameters:
    pbp_data (dict): A dictionary with game IDs as keys and sorted lists of play-by-play logs as values.
    db_path (str): The path to the SQLite database file.

    Returns:
    bool: True if the operation was successful for all game IDs, False otherwise.
    """
    logging.info(f"Saving play-by-play logs to database: {db_path}")
    overall_success = True

    try:
        with sqlite3.connect(db_path) as conn:
            for game_id, pbp_logs_sorted in pbp_data.items():
                if not pbp_logs_sorted:
                    logging.info(f"Game ID {game_id} - No logs to save.")
                    continue

                try:
                    conn.execute("BEGIN")
                    data_to_insert = [
                        (game_id, log["orderNumber"], json.dumps(log))
                        for log in pbp_logs_sorted
                    ]
                    conn.executemany(
                        "INSERT OR REPLACE INTO PbP_Logs (game_id, play_id, log_data) VALUES (?, ?, ?)",
                        data_to_insert,
                    )
                    conn.commit()
                except sqlite3.Error as e:
                    conn.rollback()
                    logging.error(f"Game ID {game_id} - DB error: {e}")
                    overall_success = False
                    continue

    except sqlite3.Error as e:
        logging.error(f"Database connection error: {e}")
        return False

    if overall_success:
        logging.info("Play-by-play logs saved successfully.")
    else:
        logging.warning("Some play-by-play logs were not saved successfully.")
    return overall_success


def main():
    """
    Main function to handle command-line arguments and orchestrate fetching and saving NBA play-by-play data.
    """

    parser = argparse.ArgumentParser(
        description="Fetch and save NBA play-by-play data."
    )
    parser.add_argument(
        "--save", action="store_true", help="Save play-by-play data to database"
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


if __name__ == "__main__":
    main()
