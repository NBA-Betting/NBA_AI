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
    python -m src.database_updater.prior_states --game_ids=0042300401,0022300649 --log_level=DEBUG
- Successful execution will log detailed information about the prior states loaded and any missing data.
"""

import argparse
import json
import logging
import sqlite3

from src.config import config
from src.database import get_db
from src.logging_config import setup_logging
from src.utils import log_execution_time, lookup_basic_game_info

# Configuration values
DB_PATH = config["database"]["path"]


@log_execution_time(average_over="game_ids")
def determine_prior_states_needed(game_ids, db_path=DB_PATH):
    """
    Determines game IDs for previous games played by the home and away teams,
    restricting to Regular Season and Post Season games from the same season.

    Uses a batch query approach: loads all season games once, then derives
    prior games in Python. This is ~3.4x faster than per-game queries.

    Parameters:
    game_ids (list): A list of IDs for the games to determine prior states for.
    db_path (str): The path to the SQLite database file. Defaults to the DB_PATH from config.

    Returns:
    dict: A dictionary where each key is a game ID from the input list and each value is a dictionary containing
          two keys 'home' and 'away'. The value of each key is a list of IDs of previous games played by the respective team.
          Both lists are restricted to games from the same season (Regular Season and Post Season). The lists are ordered by date and time.
    """
    logging.debug(f"Determining prior states needed for {len(game_ids)} games...")
    necessary_prior_states = {}

    try:
        with get_db(db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Get basic game info for all target game_ids
            games_info = lookup_basic_game_info(game_ids, db_path)

            # Collect unique seasons from target games
            seasons = set(info["season"] for info in games_info.values())

            # Batch load ALL games for relevant seasons (ordered by date)
            # This is much faster than querying per-game
            season_games = {}  # season -> list of (game_id, home, away, date_time)
            for season in seasons:
                cursor.execute(
                    """
                    SELECT game_id, home_team, away_team, date_time_utc
                    FROM Games
                    WHERE season = ?
                    AND season_type IN ('Regular Season', 'Post Season')
                    ORDER BY date_time_utc
                    """,
                    (season,),
                )
                season_games[season] = [
                    {
                        "game_id": row["game_id"],
                        "home": row["home_team"],
                        "away": row["away_team"],
                        "date_time": row["date_time_utc"],
                    }
                    for row in cursor.fetchall()
                ]

            # Derive prior games from the cached season data
            for game_id, game_info in games_info.items():
                game_datetime = game_info["date_time_utc"]
                home = game_info["home"]
                away = game_info["away"]
                season = game_info["season"]

                home_game_ids = []
                away_game_ids = []

                # Iterate through season games (already ordered by date)
                for g in season_games.get(season, []):
                    if g["date_time"] >= game_datetime:
                        break  # Games are ordered, so we can stop early

                    # Check if home team played in this game
                    if g["home"] == home or g["away"] == home:
                        home_game_ids.append(g["game_id"])

                    # Check if away team played in this game
                    if g["home"] == away or g["away"] == away:
                        away_game_ids.append(g["game_id"])

                # Store the lists of game IDs in the results dictionary
                necessary_prior_states[game_id] = {
                    "home": home_game_ids,
                    "away": away_game_ids,
                }

            logging.debug("Prior states determined.")
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
def load_prior_states(game_ids_dict, db_path=DB_PATH, parse_players_data=False):
    """
    Loads and orders by date the prior states for lists of home and away game IDs
    from the GameStates table in the database, retrieving all columns for each state and
    storing each state as a dictionary within a list.

    Loads all columns (SELECT *) to support future GenAI engine needs that may require
    additional fields like players_data, clock, period, etc.

    Parameters:
    game_ids_dict (dict): A dictionary where keys are game IDs and values are dictionaries containing
                          'home' and 'away' lists of game IDs for the home and away team's prior games.
    db_path (str): The path to the SQLite database file. Defaults to the DB_PATH from config.
    parse_players_data (bool): If True, parse the players_data JSON column. Defaults to False
                               for performance (current features don't use it). Set to True when
                               GenAI engine needs player-level data.

    Returns:
    dict: A dictionary where each key is a game ID and each value is another dictionary containing
          'home_prior_states', 'away_prior_states', and 'missing_prior_states'.
          'home_prior_states' and 'away_prior_states' are lists of final state information for each home and away game,
          ordered by game date. A team with no current-season final states uses
          the preceding season and is marked with '<side>_prior_season'.
          'missing_prior_states' retains every missing current-season game ID.
    """
    logging.debug(f"Loading prior states for {len(game_ids_dict)} games...")
    prior_states_dict = {
        game_id: {
            "home_prior_states": [],
            "away_prior_states": [],
            "missing_prior_states": {"home": [], "away": []},
        }
        for game_id in game_ids_dict.keys()
    }
    games_info = lookup_basic_game_info(list(game_ids_dict), db_path)

    all_game_ids = list(
        set(
            game_id
            for ids in game_ids_dict.values()
            for game_id in ids["home"] + ids["away"]
        )
    )

    try:
        with get_db(db_path) as conn:
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

                # Only parse players_data JSON if explicitly requested
                # (deferred for performance - current features don't use it)
                if parse_players_data:
                    for state in all_prior_states:
                        state["players_data"] = json.loads(state["players_data"])

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

                    for side, ids in (("home", home_game_ids), ("away", away_game_ids)):
                        prior_states_dict[game_id]["missing_prior_states"][side] = [
                            gid for gid in ids if gid not in states_dict
                        ]

            # Seed a team's first games from the preceding season, only until
            # current-season final states exist. Keep missing current states
            # visible so the updater retries instead of finalizing stale data.
            for game_id, states in prior_states_dict.items():
                info = games_info.get(game_id)
                if not info:
                    continue
                start_year = int(info["season"].split("-")[0])
                previous_season = f"{start_year - 1}-{start_year}"
                for side in ("home", "away"):
                    if states[f"{side}_prior_states"]:
                        continue
                    cursor.execute(
                        """
                        SELECT gs.* FROM GameStates gs
                        JOIN Games g ON g.game_id = gs.game_id
                        WHERE g.season = ?
                        AND g.season_type IN ('Regular Season', 'Post Season')
                        AND g.status = 3 AND gs.is_final_state = 1
                        AND (g.home_team = ? OR g.away_team = ?)
                        AND g.date_time_utc < ?
                        ORDER BY g.date_time_utc
                        """,
                        (previous_season, info[side], info[side], info["date_time_utc"]),
                    )
                    fallback = [dict(row) for row in cursor.fetchall()]
                    if parse_players_data:
                        for state in fallback:
                            state["players_data"] = json.loads(state["players_data"])
                    states[f"{side}_prior_states"] = fallback
                    states[f"{side}_prior_season"] = bool(fallback)

        logging.debug(f"Prior states loaded for {len(prior_states_dict)} games.")
        missing_count = sum(
            1
            for states in prior_states_dict.values()
            if states["missing_prior_states"]["home"]
            or states["missing_prior_states"]["away"]
        )
        if missing_count:
            logging.debug(f"Missing prior states for {missing_count} games.")

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
