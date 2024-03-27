import json
import logging
import os
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

from .utils import validate_game_ids

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s",
)


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
        # Make the HTTP request
        response = session.get(base_url.format(game_id), headers=headers, timeout=30)
        response.raise_for_status()  # Raises an HTTPError for bad responses

        # Parse the JSON response
        data = response.json()

        # Extract the actions and sort them by action number
        actions = data.get("game", {}).get("actions", [])
        actions_sorted = sorted(actions, key=lambda x: x["actionNumber"])

        return game_id, actions_sorted
    except requests.exceptions.HTTPError as http_err:
        # Handle HTTP errors
        if http_err.response.status_code == 403:
            logging.info(
                f"Game Id: {game_id} - HTTP error occurred: {http_err}. Game may not have started yet or is from the distant past."
            )
        else:
            logging.warning(f"Game Id: {game_id} - HTTP error occurred: {http_err}.")
    except Exception as e:
        # Handle other exceptions
        logging.warning(f"Game Id: {game_id} - API call error: {str(e)}.")

    # Return the game_id and an empty list if an error occurred
    return game_id, []


def get_pbp(game_ids):
    """
    Fetches play-by-play data for a list of games concurrently.

    Parameters:
    game_ids (list or str): A list of game IDs to fetch data for, or a single game ID.

    Returns:
    dict: A dictionary mapping game IDs to lists of sorted actions. If an error occurs when fetching data for a game, the list of actions will be empty.
    """
    # If a single game ID was provided, convert it to a list
    if isinstance(game_ids, str):
        game_ids = [game_ids]

    validate_game_ids(game_ids)

    # Set up headers for the HTTP requests
    headers = {
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Accept-Language": "en-US,en;q=0.9",
        "Cache-Control": "max-age=0",
        "Connection": "keep-alive",
        "Host": "cdn.nba.com",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36",
    }

    # Set up the base URL for the HTTP requests
    base_url = "https://cdn.nba.com/static/json/liveData/playbyplay/playbyplay_{}.json"

    # Determine the size of the thread pool
    thread_pool_size = os.cpu_count() * 10

    results = {}
    with requests.Session() as session:
        with ThreadPoolExecutor(max_workers=thread_pool_size) as executor:
            # Create a future for each game ID
            future_to_game_id = {
                executor.submit(
                    fetch_game_data, session, base_url, headers, game_id
                ): game_id
                for game_id in game_ids
            }

            # As each future completes, collect results
            for future in as_completed(future_to_game_id):
                game_id = future_to_game_id[future]
                try:
                    game_id, actions_sorted = future.result()
                    results[game_id] = actions_sorted
                except Exception as exc:
                    logging.warning(f"Game ID {game_id} generated an exception: {exc}")
                    results[game_id] = []

    return results


def save_pbp(pbp_data, db_path):
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
                if not pbp_logs_sorted:  # Skip if there are no logs for this game ID
                    logging.info(
                        f"Game Id {game_id} - No play-by-play logs to save. Skipping."
                    )
                    continue

                try:
                    # Begin a new transaction for each game_id
                    conn.execute("BEGIN")
                    data_to_insert = [
                        (game_id, log["orderNumber"], json.dumps(log))
                        for log in pbp_logs_sorted
                    ]

                    # Use executemany to insert or replace data in a single operation
                    conn.executemany(
                        """
                        INSERT OR REPLACE INTO PbP_Logs (game_id, orderNumber, log_data)
                        VALUES (?, ?, ?)
                        """,
                        data_to_insert,
                    )
                    conn.commit()  # Commit the transaction if no errors occurred
                except Exception as e:
                    conn.rollback()  # Roll back the transaction if an error occurred
                    logging.error(f"Game Id {game_id} - Error saving PbP Logs. {e}")
                    overall_success = False  # Mark the overall operation as failed, but continue processing other game_ids

    except Exception as e:
        logging.error(f"Database connection error: {e}")
        return False  # Return False immediately if a database connection error occurred

    return overall_success  # Return True if the operation was successful for all game_ids, False otherwise
