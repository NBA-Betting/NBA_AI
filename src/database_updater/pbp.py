"""
pbp.py

This module fetches and saves NBA play-by-play data for specified game IDs.
It consists of functions to:
- Fetch play-by-play data from the NBA API.
- Validate and save the data to a SQLite database.
- Ensure data integrity by checking for empty or corrupted data before updating the database.

Functions:
- fetch_game_data(session, base_url, headers, game_id): Fetches play-by-play data for a specific game.
- get_pbp(game_ids, pbp_endpoint): Fetches play-by-play data for a list of games concurrently.
- save_pbp(pbp_data, db_path): Saves the fetched play-by-play data to the database.
- main(): Handles command-line arguments to fetch and/or save play-by-play data, with optional timing.

Usage:
- Typically run as part of a larger data collection pipeline.
- Script can be run directly from the command line (project root) to fetch and save NBA play-by-play data:
    python -m src.database_updater.pbp --save --game_ids=0042300401,0022300649 --log_level=DEBUG --pbp_endpoint=both
- Successful execution will print the number of games fetched, the number of actions in each game, and the first and last actions in each game.
"""

import argparse
import json
import logging
import os
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from tqdm import tqdm

from src.config import config
from src.database import get_db
from src.logging_config import setup_logging
from src.utils import (
    StageLogger,
    log_execution_time,
    requests_retry_session,
    validate_game_ids,
)

# Configuration values
DB_PATH = config["database"]["path"]
NBA_API_LIVE_URL = config["nba_api"]["pbp_live_endpoint"]
NBA_API_LIVE_HEADERS = config["nba_api"]["pbp_live_headers"]
NBA_API_STATS_URL = config["nba_api"]["pbp_stats_endpoint"]
NBA_API_STATS_HEADERS = config["nba_api"]["pbp_stats_headers"]


def fetch_game_data(
    session,
    primary_base_url,
    fallback_base_url,
    primary_headers,
    fallback_headers,
    game_id,
):
    """
    Fetches game data from the primary or fallback endpoint and sorts the actions based on the action number.

    Parameters:
    session (requests.Session): The session to use for making the HTTP request.
    primary_base_url (str): The primary base URL to fetch the game data from. Should contain a placeholder for the game_id.
    fallback_base_url (str): The fallback base URL to fetch the game data from. Should contain a placeholder for the game_id.
    primary_headers (dict): The headers to include in the HTTP request for the primary endpoint.
    fallback_headers (dict): The headers to include in the HTTP request for the fallback endpoint.
    game_id (str): The ID of the game to fetch data for.

    Returns:
    tuple: A tuple containing the game_id and a list of sorted actions. If an error occurs, the list of actions will be empty.
    """

    def duration_to_seconds(duration_str):
        minutes = int(duration_str.split("M")[0][2:])
        seconds = float(duration_str.split("M")[1][:-1])
        return minutes * 60 + seconds

    def parse_and_sort_data(data, source):
        actions = data.get("game", {}).get("actions", [])
        sort_key = "orderNumber" if source == "live" else "actionId"
        actions_sorted = sorted(
            actions,
            key=lambda x: (
                x["period"],
                -duration_to_seconds(x["clock"]),
                x[sort_key],
            ),
        )
        return actions_sorted

    try:
        # Attempt to fetch data from the primary endpoint
        response = session.get(
            primary_base_url.format(game_id), headers=primary_headers, timeout=(10, 30)
        )
        response.raise_for_status()
        data = response.json()
        actions_sorted = parse_and_sort_data(
            data, "live" if "live" in primary_base_url else "stats"
        )
        return game_id, actions_sorted

    # Any failure (HTTP error, timeout, bad JSON, missing keys) falls back; one
    # escaping here would abort every other game in the batch
    except (requests.exceptions.RequestException, ValueError, KeyError):
        if fallback_base_url:
            try:
                # Fallback to the secondary endpoint if the primary endpoint fails
                response = session.get(
                    fallback_base_url.format(game_id),
                    headers=fallback_headers,
                    timeout=(10, 30),
                )
                response.raise_for_status()
                data = response.json()
                actions_sorted = parse_and_sort_data(
                    data, "stats" if "stats" in fallback_base_url else "live"
                )
                return game_id, actions_sorted

            except requests.exceptions.HTTPError as http_err:
                logging.warning(f"Game ID {game_id} - HTTP error: {http_err}")
            except Exception as e:
                logging.warning(f"Game ID {game_id} - API call error: {e}")

    return game_id, []


@log_execution_time(average_over="game_ids")
def get_pbp(game_ids, pbp_endpoint="both", stage_logger=None):
    """
    Fetches play-by-play data for a list of games concurrently.

    Parameters:
    game_ids (list or str): A list of game IDs to fetch data for, or a single game ID.
    pbp_endpoint (str): The endpoint to use for fetching play-by-play data ("both", "live", "stats").
    stage_logger (StageLogger): Optional logger for tracking API calls.

    Returns:
    dict: A dictionary mapping game IDs to lists of sorted actions. If an error occurs when fetching data for a game, the list of actions will be empty.
    """
    logging.debug(
        f"Fetching play-by-play data for {len(game_ids)} games using pbp_endpoint: {pbp_endpoint}."
    )
    if isinstance(game_ids, str):
        game_ids = [game_ids]

    validate_game_ids(game_ids)

    n = len(game_ids)
    thread_pool_size = min(3, os.cpu_count() or 4)
    results = {}

    endpoint_settings = {
        "live": {"base_url": NBA_API_LIVE_URL, "headers": NBA_API_LIVE_HEADERS},
        "stats": {"base_url": NBA_API_STATS_URL, "headers": NBA_API_STATS_HEADERS},
    }

    def get_endpoint_priority(pbp_endpoint):
        if pbp_endpoint == "both":
            return ("live", "stats")
        elif pbp_endpoint == "live":
            return ("live",)
        elif pbp_endpoint == "stats":
            return ("stats",)
        else:
            raise ValueError(
                "Invalid pbp_endpoint value. Choose from 'both', 'live', 'stats'."
            )

    endpoint_priority = get_endpoint_priority(pbp_endpoint)

    def _fetch_one(session, game_id):
        return fetch_game_data(
            session,
            endpoint_settings[endpoint_priority[0]]["base_url"],
            (
                endpoint_settings[endpoint_priority[1]]["base_url"]
                if len(endpoint_priority) > 1
                else None
            ),
            endpoint_settings[endpoint_priority[0]]["headers"],
            (
                endpoint_settings[endpoint_priority[1]]["headers"]
                if len(endpoint_priority) > 1
                else None
            ),
            game_id,
        )

    # Daily updates (<=15 games): use threading for speed
    # Catch-up mode (>15 games): process in bursts with cooldown pauses
    if n <= 15:
        with requests_retry_session() as session:
            with ThreadPoolExecutor(max_workers=thread_pool_size) as executor:
                futures = [
                    executor.submit(_fetch_one, session, gid) for gid in game_ids
                ]
                with tqdm(
                    total=len(futures), desc="Fetching PBP", unit="game", leave=False
                ) as pbar:
                    for future in as_completed(futures):
                        game_id, actions_sorted = future.result()
                        results[game_id] = actions_sorted if actions_sorted else []
                        if stage_logger:
                            stage_logger.log_api_call()
                        pbar.update(1)
    else:
        # Sequential with burst pauses to avoid NBA API throttling.
        burst_size = 10
        cooldown = 10  # seconds between bursts
        logging.info(
            f"PBP catch-up mode: {n} games, fetching {burst_size} at a time "
            f"with {cooldown}s cooldown"
        )
        failures = 0
        with requests_retry_session() as session:
            with tqdm(total=n, desc="Fetching PBP", unit="game", leave=False) as pbar:
                for i, game_id in enumerate(game_ids):
                    game_id, actions_sorted = _fetch_one(session, game_id)
                    results[game_id] = actions_sorted if actions_sorted else []
                    if stage_logger:
                        stage_logger.log_api_call()
                    pbar.update(1)
                    # Consecutive failures mean the API has started throttling, which
                    # lasts 30-60+ minutes; stop instead of hanging on timeouts. Games
                    # left out of the results are fetched on the next run.
                    failures = 0 if actions_sorted else failures + 1
                    if failures == 3:
                        logging.warning(
                            "PBP catch-up stopped after 3 consecutive failures "
                            "(likely NBA API throttling)"
                        )
                        break
                    # Cooldown between bursts
                    if (i + 1) % burst_size == 0 and i + 1 < n:
                        time.sleep(cooldown)

    logging.debug(f"Fetched play-by-play data for {len(results)} games.")

    # Track and warn about empty responses
    empty_count = sum(1 for game_id, actions in results.items() if not actions)
    if empty_count > 0:
        logging.warning(
            f"PBP collection: {empty_count}/{len(results)} games returned 0 actions from API"
        )

    for game in results:
        if results[game]:
            logging.debug(f"Game ID: {game} - {len(results[game])} actions")

    return results


@log_execution_time(average_over="pbp_data")
def save_pbp(pbp_data, db_path=DB_PATH):
    """
    Saves the play-by-play logs to the database and updates pbp_last_fetched_at timestamp.
    Each game_id is processed in a separate transaction to ensure all-or-nothing behavior.

    Parameters:
    pbp_data (dict): A dictionary with game IDs as keys and sorted lists of play-by-play logs as values.
    db_path (str): The path to the SQLite database file.

    Returns:
    dict: Dictionary with counts of added/updated/unchanged games.
    """
    logging.debug(f"Saving play-by-play logs to database")

    added_count = 0
    updated_count = 0
    unchanged_count = 0
    failed_count = 0

    try:
        with get_db(db_path) as conn:
            cursor = conn.cursor()

            for game_id, pbp_logs_sorted in pbp_data.items():
                if not pbp_logs_sorted:
                    logging.debug(f"Game ID {game_id} - No logs to save.")
                    unchanged_count += 1
                    continue

                try:
                    conn.execute("BEGIN")

                    # Check if game already has PBP data
                    cursor.execute(
                        "SELECT COUNT(*) FROM PbP_Logs WHERE game_id = ?", (game_id,)
                    )
                    existing_count = cursor.fetchone()[0]

                    # Never replace a game's PBP with a shorter copy of the same
                    # feed: it only grows during a game, so fewer actions means a
                    # truncated fetch. Counts across feeds don't compare (the
                    # stats feed is terser), so a newer fetch from the other
                    # feed (e.g. the final stats copy after a mid-game live
                    # one) replaces it.
                    stored = cursor.execute(
                        "SELECT log_data FROM PbP_Logs WHERE game_id = ? LIMIT 1",
                        (game_id,),
                    ).fetchone()
                    same_feed = stored is None or (
                        "timeActual" in json.loads(stored[0])
                    ) == ("timeActual" in pbp_logs_sorted[0])
                    if same_feed and existing_count > len(pbp_logs_sorted):
                        conn.execute("ROLLBACK")
                        logging.warning(
                            f"Game ID {game_id} - refetch has {len(pbp_logs_sorted)} "
                            f"actions, {existing_count} stored; keeping stored PBP"
                        )
                        unchanged_count += 1
                        continue

                    # Delete existing logs for the game_id
                    conn.execute("DELETE FROM PbP_Logs WHERE game_id = ?", (game_id,))

                    # Insert new logs for the game_id
                    data_to_insert = [
                        (
                            game_id,
                            log.get("orderNumber", log.get("actionId")),
                            json.dumps(log),
                        )
                        for log in pbp_logs_sorted
                    ]
                    conn.executemany(
                        "INSERT INTO PbP_Logs (game_id, play_id, log_data) VALUES (?, ?, ?)",
                        data_to_insert,
                    )

                    # Update pbp_last_fetched_at timestamp
                    cursor.execute(
                        "UPDATE Games SET pbp_last_fetched_at = datetime('now') WHERE game_id = ?",
                        (game_id,),
                    )

                    conn.commit()

                    # Track whether this was an add or update
                    if existing_count == 0:
                        added_count += 1
                        logging.debug(
                            f"Game ID {game_id} - Added {len(pbp_logs_sorted)} plays"
                        )
                    else:
                        updated_count += 1
                        logging.debug(
                            f"Game ID {game_id} - Updated from {existing_count} to {len(pbp_logs_sorted)} plays"
                        )

                except sqlite3.Error as e:
                    conn.rollback()
                    logging.error(f"Game ID {game_id} - DB error: {e}")
                    failed_count += 1
                    continue

    except sqlite3.Error as e:
        logging.error(f"Database connection error: {e}")
        return {"added": 0, "updated": 0, "unchanged": 0, "failed": len(pbp_data)}

    total_success = added_count + updated_count + unchanged_count
    if total_success == len(pbp_data):
        logging.debug(
            f"PBP saved: {added_count} added, {updated_count} updated, {unchanged_count} unchanged"
        )
    else:
        logging.warning(
            f"Some play-by-play logs failed: {failed_count} failures out of {len(pbp_data)} games"
        )

    return {
        "added": added_count,
        "updated": updated_count,
        "unchanged": unchanged_count,
        "failed": failed_count,
    }


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
    parser.add_argument(
        "--pbp_endpoint",
        type=str,
        default="both",
        help="The endpoint to use for fetching play-by-play data ('both', 'live', 'stats'). Default is 'both'.",
    )

    args = parser.parse_args()
    log_level = args.log_level.upper()
    setup_logging(log_level=log_level)

    game_ids = args.game_ids.split(",") if args.game_ids else []

    pbp_data = get_pbp(game_ids, pbp_endpoint=args.pbp_endpoint)

    if args.save:
        save_pbp(pbp_data)


if __name__ == "__main__":
    main()
