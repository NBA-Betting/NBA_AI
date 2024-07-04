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
    python -m src.pbp --fetch --save --game_ids=0042300401,0022300649 --timing
- Successful execution will print the number of games fetched, the number of actions in each game, and the first and last actions in each game.
"""

import json
import logging
import os
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

from src.config import config
from src.utils import requests_retry_session, validate_game_ids

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s",
)

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
    try:
        response = session.get(base_url.format(game_id), headers=headers, timeout=10)
        response.raise_for_status()  # Raises an HTTPError for bad responses

        data = response.json()
        actions = data.get("game", {}).get("actions", [])

        def duration_to_seconds(duration_str):
            minutes = int(duration_str.split("M")[0][2:])
            seconds = float(duration_str.split("M")[1][:-1])
            return minutes * 60 + seconds

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
        if http_err.response.status_code == 403:
            logging.info(
                f"Game Id: {game_id} - HTTP error occurred: {http_err}. Game may not have started yet or is from the distant past."
            )
        else:
            logging.warning(f"Game Id: {game_id} - HTTP error occurred: {http_err}.")
    except Exception as e:
        logging.warning(f"Game Id: {game_id} - API call error: {str(e)}.")

    return game_id, []


def get_pbp(game_ids):
    """
    Fetches play-by-play data for a list of games concurrently.

    Parameters:
    game_ids (list or str): A list of game IDs to fetch data for, or a single game ID.

    Returns:
    dict: A dictionary mapping game IDs to lists of sorted actions. If an error occurs when fetching data for a game, the list of actions will be empty.
    """
    if isinstance(game_ids, str):
        game_ids = [game_ids]

    validate_game_ids(game_ids)

    thread_pool_size = os.cpu_count() * 10

    results = {}
    with requests_retry_session() as session:
        with ThreadPoolExecutor(max_workers=thread_pool_size) as executor:
            future_to_game_id = {
                executor.submit(
                    fetch_game_data, session, NBA_API_BASE_URL, NBA_API_HEADERS, game_id
                ): game_id
                for game_id in game_ids
            }

            for future in as_completed(future_to_game_id):
                game_id = future_to_game_id[future]
                try:
                    game_id, actions_sorted = future.result()
                    results[game_id] = actions_sorted
                except Exception as exc:
                    logging.warning(f"Game ID {game_id} generated an exception: {exc}")
                    results[game_id] = []

    return results


def save_pbp(pbp_data, db_path=DB_PATH):
    """
    Saves the play-by-play logs to the database. Each game_id is processed in a separate transaction to ensure all-or-nothing behavior.

    Parameters:
    pbp_data (dict): A dictionary with game IDs as keys and sorted lists of play-by-play logs as values.
    db_path (str): The path to the SQLite database file.

    Returns:
    bool: True if the operation was successful for all game IDs, False otherwise.
    """
    overall_success = True

    try:
        with sqlite3.connect(db_path) as conn:
            for game_id, pbp_logs_sorted in pbp_data.items():
                if not pbp_logs_sorted:
                    logging.info(
                        f"Game Id {game_id} - No play-by-play logs to save. Skipping."
                    )
                    continue

                try:
                    conn.execute("BEGIN")
                    data_to_insert = [
                        (game_id, log["orderNumber"], json.dumps(log))
                        for log in pbp_logs_sorted
                    ]

                    conn.executemany(
                        """
                        INSERT OR REPLACE INTO PbP_Logs (game_id, play_id, log_data)
                        VALUES (?, ?, ?)
                        """,
                        data_to_insert,
                    )
                    conn.commit()
                except Exception as e:
                    conn.rollback()
                    logging.error(f"Game Id {game_id} - Error saving PbP Logs. {e}")
                    overall_success = False

    except Exception as e:
        logging.error(f"Database connection error: {e}")
        return False

    return overall_success


def main():
    """
    Main function to handle command-line arguments and orchestrate fetching and saving NBA play-by-play data.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Fetch and save NBA play-by-play data."
    )
    parser.add_argument(
        "--fetch", action="store_true", help="Fetch play-by-play data from API"
    )
    parser.add_argument(
        "--save", action="store_true", help="Save play-by-play data to database"
    )
    parser.add_argument(
        "--game_ids", type=str, help="Comma-separated list of game IDs to process"
    )
    parser.add_argument("--timing", action="store_true", help="Measure execution time")

    args = parser.parse_args()

    if not args.fetch and not args.save:
        parser.error("No action requested, add --fetch or --save")

    game_ids = args.game_ids.split(",") if args.game_ids else []

    pbp_data = None

    if args.fetch:
        overall_start_time = time.time()
        pbp_data = get_pbp(game_ids)
        print(f"Number of games: {len(pbp_data)}")
        for game_id in game_ids:
            actions = pbp_data[game_id]
            num_actions = len(actions)
            first_action = actions[0] if num_actions > 0 else "No actions"
            last_action = actions[-1] if num_actions > 0 else "No actions"
            print(
                f"Game ID: {game_id}\nNumber of actions: {num_actions}\nFirst action: {first_action}\nLast action: {last_action}"
            )

        if args.timing:
            overall_elapsed_time = time.time() - overall_start_time
            avg_time_per_item = overall_elapsed_time / len(game_ids) if game_ids else 0
            logging.info(f"Fetching data took {overall_elapsed_time:.2f} seconds.")
            logging.info(f"Average time per game_id: {avg_time_per_item:.2f} seconds.")

    if args.save:
        if pbp_data is None:
            logging.error(
                "No data to save. Ensure --fetch is used or data is provided."
            )
        else:
            overall_start_time = time.time()
            success = save_pbp(pbp_data)
            if args.timing:
                overall_elapsed_time = time.time() - overall_start_time
                avg_time_per_item = (
                    overall_elapsed_time / len(pbp_data) if pbp_data else 0
                )
                logging.info(f"Saving data took {overall_elapsed_time:.2f} seconds.")
                logging.info(
                    f"Average time per game_id: {avg_time_per_item:.2f} seconds."
                )
            if success:
                logging.info("Play-by-play logs saved successfully.")
            else:
                logging.error("Failed to save some or all play-by-play logs.")


if __name__ == "__main__":
    main()
