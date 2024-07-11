"""
prior_states.py

This module provides functionality to determine and load prior game states for NBA games.
It consists of functions to:
- Determine game IDs for previous games played by the home and away teams.
- Load prior states from the GameStates table in the database.
- Main function to handle command-line arguments and orchestrate the process.

Functions:
- determine_prior_states_needed(game_ids, db_path=DB_PATH): Determines game IDs for previous games played by the home and away teams.
- load_prior_states(game_ids_dict, db_path=DB_PATH): Loads prior states from the GameStates table.
- main(): Main function to handle command-line arguments and orchestrate determining and loading prior game states.

Usage:
- Typically run as part of a larger data processing pipeline.
- Script can be run directly from the command line (project root) to determine and load prior game states:
    python -m src.prior_states --game_ids=0042300401,0022300649 --timing --debug
- Successful execution will print loaded and missing prior states along with execution timing if specified.
"""

import argparse
import logging
import sqlite3
import time

from src.config import config
from src.games import lookup_basic_game_info

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s",
)

DB_PATH = config["database"]["path"]


def determine_prior_states_needed(game_ids, db_path=DB_PATH):
    """
    Determines game IDs for previous games played by the home and away teams,
    restricting to Regular Season and Post Season games from the same season.

    Parameters:
    game_ids (list): A list of IDs for the games to determine prior states for.
    db_path (str): The path to the SQLite database file. Defaults to the DB_PATH from config.

    Returns:
    dict: A dictionary where each key is a game ID from the input list and each value is a tuple containing
          two lists. The first list contains the IDs of previous games played by the home team, and the second
          list contains the IDs of previous games played by the away team. Both lists are restricted to games
          from the same season (Regular Season and Post Season). The lists are ordered by date and time.
    """
    necessary_prior_states = {}

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()

        # Get basic game info for all game_ids
        games_info = lookup_basic_game_info(game_ids, db_path)

        for game_info in games_info:
            game_id = game_info["game_id"]
            game_datetime = game_info["date_time_est"]
            home = game_info["home_team"]
            away = game_info["away_team"]
            season = game_info["season"]

            home_game_ids = []
            away_game_ids = []

            # Query for prior games of the home team
            base_query = """
                SELECT game_id FROM Games
                WHERE date_time_est < ? AND (home_team = ? OR away_team = ?) 
                AND season = ? AND (season_type = 'Regular Season' OR season_type = 'Post Season')
            """
            cursor.execute(base_query, (game_datetime, home, home, season))
            home_game_ids = [row[0] for row in cursor.fetchall()]

            # Query for prior games of the away team
            cursor.execute(base_query, (game_datetime, away, away, season))
            away_game_ids = [row[0] for row in cursor.fetchall()]

            # Store the lists of game IDs in the results dictionary
            necessary_prior_states[game_id] = (home_game_ids, away_game_ids)

    return necessary_prior_states


def load_prior_states(game_ids_dict, db_path=DB_PATH):
    """
    Loads and orders by date the prior states for lists of home and away game IDs
    from the GameStates table in the database, retrieving all columns for each state and
    storing each state as a dictionary within a list.

    Parameters:
    game_ids_dict (dict): A dictionary where keys are game IDs and values are tuples of lists.
                          Each list contains game IDs for the home and away team's prior games.
    db_path (str): The path to the SQLite database file. Defaults to the DB_PATH from config.

    Returns:
    tuple: A dictionary of prior states and a dictionary of missing prior states.
           The keys are game IDs, and values are lists of final state information for each home and away game,
           ordered by game date. The missing prior states dictionary contains game IDs with no prior states.
    """
    prior_states = {game_id: [[], []] for game_id in game_ids_dict.keys()}

    # Get all unique game IDs from the dictionary
    all_game_ids = list(
        set(game_id for ids in game_ids_dict.values() for game_id in ids[0] + ids[1])
    )

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row  # Use sqlite3.Row for dictionary-like row access
        cursor = conn.cursor()

        # Load prior states for all games
        if all_game_ids:
            placeholders = ", ".join(["?"] * len(all_game_ids))
            cursor.execute(
                f"""
                SELECT * FROM GameStates
                WHERE game_id IN ({placeholders}) AND is_final_state = 1
                ORDER BY game_date ASC
            """,
                all_game_ids,
            )
            all_prior_states = [dict(row) for row in cursor.fetchall()]

            # Create a dictionary mapping game IDs to their states
            states_dict = {state["game_id"]: state for state in all_prior_states}

            # Separate the states into their respective high-level game_ids and lower-level home/away buckets
            for game_id, (home_game_ids, away_game_ids) in game_ids_dict.items():
                prior_states[game_id][0] = [
                    states_dict[id] for id in home_game_ids if id in states_dict
                ]
                prior_states[game_id][1] = [
                    states_dict[id] for id in away_game_ids if id in states_dict
                ]

    missing_prior_states = {}
    for game_id, (home_game_ids, away_game_ids) in game_ids_dict.items():
        if not prior_states[game_id][0] or not prior_states[game_id][1]:
            missing_prior_states[game_id] = (home_game_ids, away_game_ids)

    return prior_states, missing_prior_states


def main():
    """
    Main function to handle command-line arguments and orchestrate determining and loading prior game states.
    """
    parser = argparse.ArgumentParser(
        description="Determine and load prior states for NBA games."
    )
    parser.add_argument(
        "--game_ids", type=str, help="Comma-separated list of game IDs to process"
    )
    parser.add_argument("--timing", action="store_true", help="Measure execution time")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()

    if not args.game_ids:
        parser.error("No game IDs provided")

    game_ids = args.game_ids.split(",")

    overall_start_time = time.time()

    # Determine the prior states needed for the games
    logging.info("Determining prior states needed for games...")
    start_determine_time = time.time()
    prior_games_dict = determine_prior_states_needed(game_ids, DB_PATH)
    determine_time = time.time() - start_determine_time

    # Load the prior states for the games
    logging.info("Loading prior states...")
    start_load_time = time.time()
    prior_states, missing_prior_states = load_prior_states(prior_games_dict, DB_PATH)
    load_time = time.time() - start_load_time

    overall_time = time.time() - overall_start_time

    logging.info(f"Prior states loaded for {len(prior_states)} games.")
    if missing_prior_states:
        logging.warning(f"Missing prior states for {len(missing_prior_states)} games.")

    if args.debug:
        logging.info("Prior States Loaded:")
        for game_id, states in prior_states.items():
            logging.info(f"Game ID {game_id}:")
            logging.info("  Home States:")
            for state in states[0]:
                logging.info(f"    {state}")
            logging.info("  Away States:")
            for state in states[1]:
                logging.info(f"    {state}")

        logging.info("Missing Prior States:")
        for game_id, (home_game_ids, away_game_ids) in missing_prior_states.items():
            logging.info(f"Game ID {game_id}:")
            logging.info("  Missing Home States:")
            for home_id in home_game_ids:
                logging.info(f"    {home_id}")
            logging.info("  Missing Away States:")
            for away_id in away_game_ids:
                logging.info(f"    {away_id}")

    if args.timing:
        logging.info(
            f"Overall execution took {overall_time:.2f} seconds. Average time per game: {overall_time / len(prior_states):.2f} seconds."
        )
        logging.info(
            f"Determining prior states took {determine_time:.2f} seconds. Average: {determine_time / len(prior_states):.2f} seconds."
        )
        logging.info(
            f"Loading prior states took {load_time:.2f} seconds. Average: {load_time / len(prior_states):.2f} seconds."
        )


if __name__ == "__main__":
    main()
