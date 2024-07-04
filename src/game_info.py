"""
game_info.py

This module combines functionality from the play-by-play (PBP) and game states modules to retrieve current game information.
It consists of functions to:
- Lookup basic game information from the database.
- Fetch play-by-play logs and create game states.
- Save the retrieved information to the database.

Functions:
- lookup_basic_game_info(game_ids, db_path): Looks up basic game information from the Games table.
- get_current_game_info(game_ids, db_path, save_to_db=True): Retrieves current game information for given game IDs.
- main(): Main function to handle command-line arguments and orchestrate fetching and saving NBA game data.

Usage:
- Typically run as part of a larger data collection pipeline.
- Script can be run directly from the command line (project root) to fetch and save NBA game data:
    python -m src.game_info --fetch --save --game_ids=0042300401,0022300649 --timing
- Successful execution will print the number of game IDs input and the number of game info outputs along with the game information.
"""

import argparse
import logging
import sqlite3
import time

from src.config import config
from src.game_states import create_game_states, save_game_states
from src.pbp import get_pbp, save_pbp
from src.utils import print_game_info, validate_game_ids

# Configuration
DB_PATH = config["database"]["path"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s",
)


def lookup_basic_game_info(game_ids, db_path):
    """
    Looks up basic game information given a game_id or a list of game_ids from the Games table in the SQLite database.

    Parameters:
    game_ids (str or list): The ID of the game or a list of game IDs to look up.
    db_path (str): The path to the SQLite database.

    Returns:
    list: A list of dictionaries, each representing a game. Each dictionary contains the game ID, home team, away team, date/time, status, season, and season type.
    """
    if not isinstance(game_ids, list):
        game_ids = [game_ids]

    validate_game_ids(game_ids)

    sql = f"""
    SELECT game_id, home_team, away_team, date_time_est, status, season, season_type
    FROM Games
    WHERE game_id IN ({','.join(['?'] * len(game_ids))})
    """

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(sql, game_ids)
        games = cursor.fetchall()

    game_ids_set = set(game_ids)
    game_info_list = []
    for game_id, home, away, date_time_est, status, season, season_type in games:
        game_ids_set.remove(game_id)
        game_info_list.append(
            {
                "game_id": game_id,
                "home_team": home,
                "away_team": away,
                "date_time_est": date_time_est,
                "status": status,
                "season": season,
                "season_type": season_type,
            }
        )

    if game_ids_set:
        logging.warning(f"Game IDs not found in the database: {game_ids_set}")

    return game_info_list


def get_current_game_info(game_ids, db_path, save_to_db=True):
    """
    Retrieves the current game information given a game ID or a list of game IDs.

    Parameters:
    game_ids (str or list): The ID(s) of the game(s) to retrieve the information for.
    db_path (str): The path to the SQLite database file.
    save_to_db (bool): If True, saves the play-by-play logs and game states to the database.

    Returns:
    list: A list of dictionaries, each containing the game information for a game ID, including the game ID, game date, game time (EST), home team, away team, game status, play-by-play logs, game states, and final state.
    """
    validate_game_ids(game_ids)
    games = lookup_basic_game_info(game_ids, db_path)

    try:
        pbp_logs = get_pbp(game_ids)
        if save_to_db:
            save_pbp(pbp_logs, db_path)
    except Exception as e:
        logging.error(f"Error fetching or saving PBP logs: {e}")
        return []

    game_states = {}
    try:
        for game in games:
            game_id = game["game_id"]
            game_states[game_id] = create_game_states(
                pbp_logs[game_id],
                game["home_team"],
                game["away_team"],
                game_id,
                game["date_time_est"].split("T")[0],
            )
        if save_to_db:
            save_game_states(game_states, db_path)
    except Exception as e:
        logging.error(f"Error creating or saving game states: {e}")
        return []

    game_info_list = []
    for game in games:
        game_info_dict = {
            "game_id": game["game_id"],
            "game_date": game["date_time_est"].split("T")[0],
            "game_time_est": game["date_time_est"].split("T")[1].rstrip("Z"),
            "home": game["home_team"],
            "away": game["away_team"],
            "game_status": game["status"],
            "season": game["season"],
            "season_type": game["season_type"],
            "pbp_logs": pbp_logs.get(game["game_id"], []),
            "game_states": game_states.get(game["game_id"], []),
        }
        game_info_list.append(game_info_dict)

    return game_info_list


def main():
    parser = argparse.ArgumentParser(description="Fetch and save NBA game data.")
    parser.add_argument("--fetch", action="store_true", help="Fetch game data from API")
    parser.add_argument(
        "--save", action="store_true", help="Save fetched data to database"
    )
    parser.add_argument(
        "--game_ids", type=str, help="Comma-separated list of game IDs to process"
    )
    parser.add_argument("--timing", action="store_true", help="Measure execution time")

    args = parser.parse_args()

    if not args.fetch and not args.save:
        parser.error("No action requested, add --fetch or --save")

    game_ids = args.game_ids.split(",") if args.game_ids else []
    db_path = DB_PATH

    if args.fetch:
        overall_start_time = time.time()
        save_to_db = args.save
        game_info_list = get_current_game_info(game_ids, db_path, save_to_db=save_to_db)
        print(f"Number of game IDs input: {len(game_ids)}")
        print(f"Number of game info outputs: {len(game_info_list)}")

        for game_info in game_info_list:
            print_game_info(game_info)

        if args.timing:
            overall_elapsed_time = time.time() - overall_start_time
            avg_time_per_item = overall_elapsed_time / len(game_ids) if game_ids else 0
            logging.info(
                f"Fetching, parsing, and saving data took {overall_elapsed_time:.2f} seconds."
            )
            logging.info(f"Average time per game_id: {avg_time_per_item:.2f} seconds.")


if __name__ == "__main__":
    main()
