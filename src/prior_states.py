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

    return necessary_prior_states


@log_execution_time(average_over="game_ids_dict")
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
    logging.info(f"Loading prior states for {len(game_ids_dict)} games...")
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

    logging.info(f"Prior states loaded for {len(prior_states)} games.")
    if missing_prior_states:
        logging.info(f"Missing prior states for {len(missing_prior_states)} games.")

    for game in game_ids_dict.keys():
        logging.debug(
            f"Game ID: {game} - Home Team - Prior States Count: {len(prior_states[game][0]) if prior_states[game][0] else 'No prior states'}"
        )
        logging.debug(
            f"Game ID: {game} - Home Team - First Prior State: {prior_states[game][0][0] if prior_states[game][0] else 'No prior states'}"
        )
        logging.debug(
            f"Game ID: {game} - Away Team - Last Prior State: {prior_states[game][1][-1] if prior_states[game][1] else 'No prior states'}"
        )
        logging.debug(
            f"Game ID: {game} - Home Team - Missing Count: {len(missing_prior_states[game][0]) if game in missing_prior_states else 0}"
        )
        logging.debug(
            f"Game ID: {game} - Home Team - Missing IDs: {missing_prior_states[game][0]}"
        )
        logging.debug(
            f"Game ID: {game} - Away Team - Prior States Count: {len(prior_states[game][1])}"
        )
        logging.debug(
            f"Game ID: {game} - Away Team - First Prior State: {prior_states[game][1][0] if prior_states[game][1] else 'No prior states'}"
        )
        logging.debug(
            f"Game ID: {game} - Home Team - Last Prior State: {prior_states[game][0][-1] if prior_states[game][0] else 'No prior states'}"
        )
        logging.debug(
            f"Game ID: {game} - Away Team - Missing Count: {len(missing_prior_states[game][1])}"
        )
        logging.debug(
            f"Game ID: {game} - Away Team - Missing IDs: {missing_prior_states[game][1]}"
        )

    return prior_states, missing_prior_states


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
    prior_states, missing_prior_states = load_prior_states(prior_states_needed)


if __name__ == "__main__":
    main()
