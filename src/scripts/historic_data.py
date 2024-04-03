import json
import logging
import os
import re
import sqlite3
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from nba_api.stats.endpoints import playbyplayv3
from tqdm import tqdm

# Add the parent directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ..prior_states import get_prior_states
from ..utils import lookup_basic_game_info, update_scheduled_games

load_dotenv()
PROJECT_ROOT = os.getenv("PROJECT_ROOT")


def get_ordered_game_ids(season):
    """
    This function retrieves game IDs for a given season from the 'games' table in the database.
    The games are ordered by their date and time (in Eastern Standard Time).

    Args:
        season (str): The season for which to retrieve game IDs.

    Returns:
        list: A list of tuples, each containing game_id, home_team, away_team, and date_time_est.
    """
    # Connect to the SQLite database
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()

        # Execute the SQL query to fetch game IDs for the given season
        cursor.execute(
            f"""
            SELECT game_id, home_team, away_team, date_time_est
            FROM games
            WHERE season = "{season}"
            AND season_type IN ("Regular Season", "Post Season")
            ORDER BY date_time_est ASC
            """
        )

        # Fetch all the rows returned by the query
        game_ids = cursor.fetchall()

    # Return the fetched game IDs
    return game_ids


def delete_old_records(game_id, db_path):
    """
    This function deletes old records for a given game ID from the 'PbP_Logs', 'GameStates', and 'PriorStates' tables in the database.

    Args:
        game_id (str): The game ID for which to delete old records.
        db_path (str): The path to the SQLite database.

    Returns:
        bool: True if the deletion was successful, False otherwise.
    """
    success = True

    try:
        # Connect to the SQLite database
        with sqlite3.connect(db_path) as conn:
            try:
                # Start a new transaction
                conn.execute("BEGIN")

                # Execute the SQL query to delete records from the 'PbP_Logs' table
                conn.execute(
                    """
                    DELETE FROM PbP_Logs
                    WHERE game_id = ?
                    """,
                    (game_id,),
                )

                # Execute the SQL query to delete records from the 'GameStates' table
                conn.execute(
                    """
                    DELETE FROM GameStates
                    WHERE game_id = ?
                    """,
                    (game_id,),
                )

                # Execute the SQL query to delete records from the 'PriorStates' table
                conn.execute(
                    """
                    DELETE FROM PriorStates
                    WHERE game_id = ?
                    """,
                    (game_id,),
                )

                # Commit the transaction
                conn.commit()
            except Exception as e:
                # Rollback the transaction in case of an error
                conn.rollback()

                # Log the error
                logging.error(f"Error deleting old records for game {game_id}. {e}")
                success = False

    except Exception as e:
        # Log the database connection error
        logging.error(f"Database connection error: {e}")
        return False

    # Return the success status
    return success


def get_pbp_stats_endpoint(game_id):
    """
    This function retrieves the play-by-play logs for a given game ID from the NBA's stats endpoint.
    It also checks if the action IDs in the logs are continuously increasing.

    Args:
        game_id (str): The game ID for which to retrieve play-by-play logs.

    Returns:
        tuple: A tuple containing the sorted play-by-play logs and any errors encountered.
    """
    # Initialize a dictionary to store any errors encountered
    errors = {"actionIds_not_continuous": [], "api_call_error": []}

    try:
        # Use the PlayByPlay class to retrieve the play-by-play logs for the game
        pbp = playbyplayv3.PlayByPlayV3(game_id=game_id).get_dict()["game"]["actions"]

        # Define a helper function to parse the duration from a string
        def parse_duration(duration):
            minutes = int(duration[2:4])
            seconds = float(duration[5:-1])
            return minutes * 60 + seconds

        # Sort the play-by-play logs by period, clock, and action ID
        pbp_logs_sorted = sorted(
            pbp, key=lambda x: (x["period"], -parse_duration(x["clock"]), x["actionId"])
        )

        # Check if the action IDs are continuously increasing
        action_ids = [log["actionId"] for log in pbp_logs_sorted]
        if any(action_ids[i] <= action_ids[i - 1] for i in range(1, len(action_ids))):
            print(
                f"Game Id: {game_id} - The actionIds are not continuously increasing."
            )
            errors["actionIds_not_continuous"].append(game_id)
            logs = []
        else:
            logs = pbp_logs_sorted
    except Exception as e:
        print(f"Game Id: {game_id} - PBP API call error: {str(e)}.")
        logs = []
        errors["api_call_error"].append(game_id)

    # Return the sorted play-by-play logs and any errors encountered
    return logs, errors


def save_pbp_stats_endpoint(game_id, pbp_logs_sorted, db_path):
    """
    This function saves the play-by-play logs for a given game ID to the 'PbP_Logs' table in the database.

    Args:
        game_id (str): The game ID for which to save play-by-play logs.
        pbp_logs_sorted (list): The sorted play-by-play logs to save.
        db_path (str): The path to the SQLite database.

    Returns:
        bool: True if the save was successful, False otherwise.
    """
    overall_success = True

    try:
        # Connect to the SQLite database
        with sqlite3.connect(db_path) as conn:
            # If there are no play-by-play logs to save, skip this game
            if not pbp_logs_sorted:
                print("No play-by-play logs to save. Skipping.")
                return False

            try:
                # Start a new transaction
                conn.execute("BEGIN")

                # Prepare the data to insert into the 'PbP_Logs' table
                data_to_insert = [
                    (game_id, log["actionId"], json.dumps(log))
                    for log in pbp_logs_sorted
                ]

                # Execute the SQL query to insert the data into the 'PbP_Logs' table
                conn.executemany(
                    """
                    INSERT OR REPLACE INTO PbP_Logs (game_id, play_id, log_data)
                    VALUES (?, ?, ?)
                    """,
                    data_to_insert,
                )

                # Commit the transaction
                conn.commit()
            except Exception as e:
                # Rollback the transaction in case of an error
                conn.rollback()

                # Log the error
                print(f"Error saving PbP Logs. {e}")
                overall_success = False

    except Exception as e:
        # Log the database connection error
        print(f"Database connection error: {e}")
        return False

    # Return the success status
    return overall_success


def create_game_states_stats_endpoint(pbp_logs, home, away, game_id, game_date):
    """
    This function creates game states for a given game ID using the play-by-play logs.
    It also checks for any errors during the creation of game states.

    Args:
        pbp_logs (list): The play-by-play logs for the game.
        home (str): The home team's tricode.
        away (str): The away team's tricode.
        game_id (str): The game ID for which to create game states.
        game_date (str): The date of the game.

    Returns:
        tuple: A tuple containing the created game states and any errors encountered.
    """
    # Initialize a dictionary to store any errors encountered
    errors = {"no_pbp_for_game_states": [], "game_state_creation_error": []}
    # Initialize a list to store the created game states
    game_states = []

    try:
        # If there are no play-by-play logs, return an error
        if not pbp_logs:
            errors["no_pbp_for_game_states"].append(game_id)
            return [], errors

        # Initialize a dictionary to store the players' data
        players = {"home": {}, "away": {}}

        # Initialize the current scores
        current_home_score = 0
        current_away_score = 0

        # Iterate over the play-by-play logs
        for i, row in enumerate(pbp_logs):
            # If the row contains a player's data, update the players' data
            if row.get("personId") and row.get("playerNameI"):
                team = "home" if row["teamTricode"] == home else "away"
                player_id = row["personId"]
                player_name = row["playerNameI"]

                if player_id not in players[team]:
                    players[team][player_id] = {"name": player_name, "points": 0}

                # Extract the player's points from the description
                match = re.search(r"\((\d+) PTS\)", row.get("description", ""))
                if match:
                    points = int(match.group(1))
                    players[team][player_id]["points"] = points

            # Update the current scores if new scores are available
            if row.get("scoreHome"):
                current_home_score = int(row["scoreHome"])
            if row.get("scoreAway"):
                current_away_score = int(row["scoreAway"])

            # Create the current game state
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
                "is_final_state": i == len(pbp_logs) - 1,
                "players_data": deepcopy(players),
            }

            # Add the current game state to the list of game states
            game_states.append(current_game_state)

        # Return the created game states and any errors encountered
        return game_states, errors

    except Exception as e:
        # Log the error and return it
        logging.error(f"Game Id {game_id} - Failed to create game states. {e}")
        errors["game_state_creation_error"].append(game_id)
        return [], errors


def save_game_states_stats_endpoint(game_id, states, db_path):
    """
    This function saves the game states for a given game ID to the 'GameStates' table in the database.

    Args:
        game_id (str): The game ID for which to save game states.
        states (list): The game states to save.
        db_path (str): The path to the SQLite database.

    Returns:
        bool: True if the save was successful, False otherwise.
    """
    success = True

    try:
        # Connect to the SQLite database
        with sqlite3.connect(db_path) as conn:
            try:
                # Begin a new transaction
                conn.execute("BEGIN")

                # Prepare the data to insert into the 'GameStates' table
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

                # Use executemany to insert or replace data in a single operation
                conn.executemany(
                    """
                    INSERT OR REPLACE INTO GameStates (game_id, play_id, game_date, home, away, clock, period, home_score, away_score, total, home_margin, is_final_state, players_data)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    data_to_insert,
                )

                # Commit the transaction if no errors occurred
                conn.commit()
            except Exception as e:
                # Roll back the transaction if an error occurred
                conn.rollback()

                # Log the error
                logging.error(f"Game Id {game_id} - Error saving game states. {e}")
                success = False  # Mark the operation as failed, but continue processing

    except Exception as e:
        # Log the database connection error
        logging.error(f"Database connection error: {e}")
        return False  # Return False immediately if a database connection error occurred

    # Return True if the operation was successful, False otherwise
    return success


def process_seasons(db_path, seasons):
    """
    This function processes the data for a list of seasons. It updates the schedule, gets the ordered game IDs,
    deletes old records, gets and saves play-by-play stats, creates and saves game states, and updates prior states.

    Args:
        db_path (str): The path to the SQLite database.
        seasons (list): The list of seasons to process.

    Returns:
        None
    """
    try:
        all_errors = {}
        # Loop through seasons
        for season in seasons:
            print("--------------------------------------------------")
            print(f"{season} - Start")
            # Update Schedule
            print(f"{season} - Updating Schedule")
            update_scheduled_games(season, db_path)
            # Get Ordered Game Ids
            print(f"{season} - Getting Ordered Game Ids")
            games = get_ordered_game_ids(season)
            print(f"{season} - Game Ids: {len(games)}")

            # Initialize a dictionary for the season if it doesn't exist
            all_errors.setdefault(season, {})

            print(f"{season} - Looping through games")
            # Loop through game ids
            for game in tqdm(
                games,
                desc=f"{season} - Processing Games",
                unit="game",
                dynamic_ncols=True,
            ):
                # Basic Game Info
                game_id = game[0]
                home = game[1]
                away = game[2]
                game_date = game[3].split("T")[0]

                # Delete Old Records
                is_success = delete_old_records(game_id, db_path)
                if not is_success:
                    print(f"Game Id: {game_id} - Error deleting old records.")
                    all_errors[season].setdefault(
                        "delete_old_records_error", []
                    ).append(game_id)
                    continue

                # Get PBP Stats
                pbp, errors = get_pbp_stats_endpoint(game_id)
                for error_type, game_ids in errors.items():
                    all_errors[season].setdefault(error_type, []).extend(game_ids)
                if not pbp:
                    print(f"Game Id: {game_id} - Error getting PbP Logs.")
                    continue

                # Save PBP Stats
                is_success = save_pbp_stats_endpoint(game_id, pbp, db_path)
                if not is_success:
                    print(f"Game Id: {game_id} - Error saving PbP Logs.")
                    all_errors[season].setdefault("pbp_save_error", []).append(game_id)
                    continue

                # Create Game States
                game_states, errors = create_game_states_stats_endpoint(
                    pbp, home, away, game_id, game_date
                )
                for error_type, game_ids in errors.items():
                    all_errors[season].setdefault(error_type, []).extend(game_ids)
                if not game_states:
                    print(f"Game Id: {game_id} - Error creating Game States.")
                    continue

                # Save Game States
                is_success = save_game_states_stats_endpoint(
                    game_id, game_states, db_path
                )
                if not is_success:
                    print(f"Game Id: {game_id} - Error saving Game States.")
                    all_errors[season].setdefault("game_states_save_error", []).append(
                        game_id
                    )
                    continue

                # Update Prior States
                is_success = get_prior_states(game_id, db_path)
                if not is_success:
                    print(f"Game Id: {game_id} - Error updating Prior States.")
                    all_errors[season].setdefault("prior_states_error", []).append(
                        game_id
                    )
                    continue

            print(f"{season} - End")
            print("--------------------------------------------------")

    except KeyboardInterrupt:
        print("Interrupted by user. Saving errors...")
        log_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f"errors_updating_historic_games_{log_time}.json", "w") as f:
            json.dump(all_errors, f, indent=4)
        print("All errors have been saved to all_errors.json.")
        raise  # re-raise the exception after saving the errors

    except Exception as e:
        print(f"An error occurred: {e}")
        log_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f"errors_updating_historic_games_{log_time}.json", "w") as f:
            json.dump(all_errors, f, indent=4)
        print("All errors have been saved to all_errors.json.")
        raise  # re-raise the exception after saving the errors

    else:
        # If no exception was raised, save the errors after the loop
        log_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f"errors_updating_historic_games_{log_time}.json", "w") as f:
            json.dump(all_errors, f, indent=4)
        print("All errors have been saved to all_errors.json.")


def process_games(game_ids, db_path):
    """
    This function processes a list of games. For each game, it deletes old records, gets and saves play-by-play stats,
    creates and saves game states, and updates prior states.

    Args:
        game_ids (list): The list of game IDs to process.
        db_path (str): The path to the SQLite database.

    Returns:
        None
    """
    # Loop through each game ID
    for game_id in tqdm(
        game_ids, desc="Processing Games", unit="game", dynamic_ncols=True
    ):
        # Get basic game info
        game = lookup_basic_game_info(game_id, db_path)[0]
        home = game["home_team"]
        away = game["away_team"]
        game_date = game["date_time_est"].split("T")[0]

        # Delete old records for the game
        is_success = delete_old_records(game_id, db_path)
        if not is_success:
            print(f"Game Id: {game_id} - Error deleting old records.")
            continue

        # Get play-by-play stats for the game
        pbp, errors = get_pbp_stats_endpoint(game_id)
        if not pbp:
            print(f"Game Id: {game_id} - Error getting PbP Logs.")
            continue

        # Save play-by-play stats for the game
        is_success = save_pbp_stats_endpoint(game_id, pbp, db_path)
        if not is_success:
            print(f"Game Id: {game_id} - Error saving PbP Logs.")
            continue

        # Create game states for the game
        game_states, errors = create_game_states_stats_endpoint(
            pbp, home, away, game_id, game_date
        )
        if not game_states:
            print(f"Game Id: {game_id} - Error creating Game States.")
            continue

        # Save game states for the game
        is_success = save_game_states_stats_endpoint(game_id, game_states, db_path)
        if not is_success:
            print(f"Game Id: {game_id} - Error saving Game States.")
            continue

        # Update prior states for the game
        is_success = get_prior_states(game_id, db_path)
        if not is_success:
            print(f"Game Id: {game_id} - Error updating Prior States.")
            continue


if __name__ == "__main__":
    db_path = f"{PROJECT_ROOT}/data/NBA_AI.sqlite"

    # ----------------------------------------------------
    # Uncomment the code below to process seasons
    # ----------------------------------------------------

    # seasons = [
    #     "2023-2024",
    #     "2022-2023",
    #     "2021-2022",
    #     "2020-2021",
    #     "2019-2020",
    #     "2018-2019",
    #     "2017-2018",
    #     "2016-2017",
    #     "2015-2016",
    #     "2014-2015",
    #     "2013-2014",
    #     "2012-2013",
    #     "2011-2012",
    #     "2010-2011",
    #     "2009-2010",
    #     "2008-2009",
    #     "2007-2008",
    #     "2006-2007",
    #     "2005-2006",
    #     "2004-2005",
    #     "2003-2004",
    #     "2002-2003",
    #     "2001-2002",
    #     "2000-2001",
    # ]

    # process_seasons(db_path, seasons)

    # ----------------------------------------------------
    # Uncomment the code below to process specific games
    # ----------------------------------------------------

    # game_ids = []

    # process_games(game_ids, db_path)
