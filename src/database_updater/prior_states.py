"""
prior_states.py

This module processes NBA play-by-play data to determine and load prior game states.
It consists of functions to:
- Determine prior game states needed for specified games.
- Load prior game states from a SQLite database, including handling of missing states.
- Log detailed information about the process and any missing data.

Functions:
- determine_prior_states_needed(game_ids, db_path=DB_PATH): Determines the game IDs for previous games played by the home and away teams, restricted to regular season and post-season games from the same season.
- load_prior_states(game_ids_dict, db_path=DB_PATH): Loads and orders by date the prior states for lists of home and away game IDs from the GameStates table in the database.
- main(): Handles command-line arguments to determine and load prior game states, with optional logging.

Usage:
- Typically run as part of a larger data processing pipeline.
- Script can be run directly from the command line to determine and load prior game states:
    python -m src.prior_states --game_ids=0042300401,0022300649 --log_level=DEBUG
- Successful execution will log detailed information about the prior states loaded and any missing data.
"""

import argparse
import logging
import sqlite3

from src.config import config
from src.logging_config import setup_logging
from src.utils import log_execution_time, lookup_basic_game_info

# Configuration values
DB_PATH = config["database"]["path"]


@log_execution_time(average_over="game_ids")
def determine_prior_states_needed(game_ids, db_path=DB_PATH):
    """
    Determines game IDs for previous games played by the home and away teams,
    restricting to Regular Season and Post Season games from the same season.

    Parameters:
    game_ids (list): A list of IDs for the games to determine prior states for.
    db_path (str): The path to the SQLite database file. Defaults to the DB_PATH from config.

    Returns:
    dict: A dictionary where each key is a game ID from the input list and each value is a dictionary containing
          two keys 'home' and 'away'. The value of each key is a list of IDs of previous games played by the respective team.
          Both lists are restricted to games from the same season (Regular Season and Post Season). The lists are ordered by date and time.
    """
    logging.info(f"Determining prior states needed for {len(game_ids)} games...")
    necessary_prior_states = {}

    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()

            # Get basic game info for all game_ids
            games_info = lookup_basic_game_info(game_ids, db_path)

            for game_id, game_info in games_info.items():
                game_datetime = game_info["date_time_est"]
                home = game_info["home"]
                away = game_info["away"]
                season = game_info["season"]

                home_game_ids = []
                away_game_ids = []

                # Query for prior games of the home team
                base_query = """
                    SELECT game_id FROM Games
                    WHERE date_time_est < ? AND (home_team = ? OR away_team = ?) 
                    AND season = ? AND (season_type = 'Regular Season' OR season_type = 'Post Season')
                    ORDER BY date_time_est
                """
                cursor.execute(base_query, (game_datetime, home, home, season))
                home_game_ids = [row[0] for row in cursor.fetchall()]

                # Query for prior games of the away team
                cursor.execute(base_query, (game_datetime, away, away, season))
                away_game_ids = [row[0] for row in cursor.fetchall()]

                # Store the lists of game IDs in the results dictionary
                necessary_prior_states[game_id] = {
                    "home": home_game_ids,
                    "away": away_game_ids,
                }

            logging.info("Prior states determined.")
            for game_id, prior_games in necessary_prior_states.items():
                logging.debug(
                    f"Game ID: {game_id} - Home Team Prior Game Count: {len(prior_games['home'])}"
                )
                logging.debug(
                    f"Game ID: {game_id} - Away Team Prior Game Count: {len(prior_games['away'])}"
                )
                logging.debug(
                    f"Game ID: {game_id} - Home Team Prior Games: {prior_games['home']}"
                )
                logging.debug(
                    f"Game ID: {game_id} - Away Team Prior Games: {prior_games['away']}"
                )

    except sqlite3.Error as e:
        logging.error(f"Database error: {e}")
    except Exception as e:
        logging.error(f"Error: {e}")

    return necessary_prior_states


@log_execution_time(average_over="game_ids_dict")
def load_prior_states(game_ids_dict, db_path=DB_PATH):
    """
    Loads and orders by date the prior states for lists of home and away game IDs
    from the GameStates table in the database, retrieving all columns for each state and
    storing each state as a dictionary within a list.

    Parameters:
    game_ids_dict (dict): A dictionary where keys are game IDs and values are dictionaries containing
                          'home' and 'away' lists of game IDs for the home and away team's prior games.
    db_path (str): The path to the SQLite database file. Defaults to the DB_PATH from config.

    Returns:
    dict: A dictionary where each key is a game ID and each value is another dictionary containing
          'home_prior_states', 'away_prior_states', and 'missing_prior_states'.
          'home_prior_states' and 'away_prior_states' are lists of final state information for each home and away game,
          ordered by game date. 'missing_prior_states' is a dictionary containing 'home' and 'away' lists of missing game IDs.
    """
    logging.info(f"Loading prior states for {len(game_ids_dict)} games...")
    prior_states_dict = {
        game_id: {
            "home_prior_states": [],
            "away_prior_states": [],
            "missing_prior_states": {"home": [], "away": []},
        }
        for game_id in game_ids_dict.keys()
    }

    all_game_ids = list(
        set(
            game_id
            for ids in game_ids_dict.values()
            for game_id in ids["home"] + ids["away"]
        )
    )

    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

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

                states_dict = {state["game_id"]: state for state in all_prior_states}

                for game_id, teams_game_ids in game_ids_dict.items():
                    home_game_ids = teams_game_ids["home"]
                    away_game_ids = teams_game_ids["away"]

                    prior_states_dict[game_id]["home_prior_states"] = [
                        states_dict[id] for id in home_game_ids if id in states_dict
                    ]
                    prior_states_dict[game_id]["away_prior_states"] = [
                        states_dict[id] for id in away_game_ids if id in states_dict
                    ]

                    if not prior_states_dict[game_id]["home_prior_states"]:
                        prior_states_dict[game_id]["missing_prior_states"][
                            "home"
                        ] = home_game_ids

                    if not prior_states_dict[game_id]["away_prior_states"]:
                        prior_states_dict[game_id]["missing_prior_states"][
                            "away"
                        ] = away_game_ids

        logging.info(f"Prior states loaded for {len(prior_states_dict)} games.")
        missing_count = sum(
            1
            for states in prior_states_dict.values()
            if states["missing_prior_states"]["home"]
            or states["missing_prior_states"]["away"]
        )
        if missing_count:
            logging.info(f"Missing prior states for {missing_count} games.")

        for game_id, states in prior_states_dict.items():
            logging.debug(
                f"Game ID: {game_id} - Home Team - Prior States Count: {len(states['home_prior_states']) if states['home_prior_states'] else 'No prior states'}"
            )
            logging.debug(
                f"Game ID: {game_id} - Home Team - First Prior State: {states['home_prior_states'][0] if states['home_prior_states'] else 'No prior states'}"
            )
            logging.debug(
                f"Game ID: {game_id} - Home Team - Last Prior State: {states['home_prior_states'][-1] if states['home_prior_states'] else 'No prior states'}"
            )
            logging.debug(
                f"Game ID: {game_id} - Home Team - Missing Count: {len(states['missing_prior_states']['home']) if states['missing_prior_states']['home'] else 0}"
            )
            logging.debug(
                f"Game ID: {game_id} - Home Team - Missing IDs: {states['missing_prior_states']['home']}"
            )
            logging.debug(
                f"Game ID: {game_id} - Away Team - Prior States Count: {len(states['away_prior_states']) if states['away_prior_states'] else 'No prior states'}"
            )
            logging.debug(
                f"Game ID: {game_id} - Away Team - First Prior State: {states['away_prior_states'][0] if states['away_prior_states'] else 'No prior states'}"
            )
            logging.debug(
                f"Game ID: {game_id} - Away Team - Last Prior State: {states['away_prior_states'][-1] if states['away_prior_states'] else 'No prior states'}"
            )
            logging.debug(
                f"Game ID: {game_id} - Away Team - Missing Count: {len(states['missing_prior_states']['away']) if states['missing_prior_states']['away'] else 0}"
            )
            logging.debug(
                f"Game ID: {game_id} - Away Team - Missing IDs: {states['missing_prior_states']['away']}"
            )

    except sqlite3.Error as e:
        logging.error(f"Database error: {e}")
    except Exception as e:
        logging.error(f"Error: {e}")

    return prior_states_dict


def main():
    """
    Main function to handle command-line arguments and orchestrate the process of determining and loading prior game states.
    """
    parser = argparse.ArgumentParser(
        description="Determine and load prior states for NBA games."
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

    prior_states_needed = determine_prior_states_needed(game_ids)
    prior_states_dict = load_prior_states(prior_states_needed)


if __name__ == "__main__":
    main()
