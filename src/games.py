import os
import sqlite3
import time
from datetime import datetime

import pytz
from dotenv import load_dotenv
from tqdm import tqdm

from .game_states import get_current_games_info
from .prior_states import get_prior_states
from .utils import update_scheduled_games, validate_game_ids, validate_season_format

load_dotenv()
PROJECT_ROOT = os.getenv("PROJECT_ROOT")


def get_games_info(game_ids, db_path, include_prior_states, save_to_database=True):
    """
    Fetches game information for given game IDs. Optionally includes prior states.

    Args:
        game_ids (list of str): The IDs of the games to fetch information for.
        include_prior_states (bool): Whether to include prior states in the returned information.

    Returns:
        list of dict: A list of dictionaries, each containing the fetched game information for a game ID.
    """
    # Validate the provided game IDs
    validate_game_ids(game_ids)

    start = time.time()
    # Fetch the current game information for all game IDs
    games_info = get_current_games_info(game_ids, db_path, save_to_db=save_to_database)
    print(f"Time taken to fetch current game info: {time.time() - start:.2f} seconds")

    start = time.time()
    # If requested, fetch and include prior states for each game
    if include_prior_states:
        game_ids = [game_info["game_id"] for game_info in games_info]
        prior_states_dict = get_prior_states(game_ids, db_path)

        for game_info in games_info:
            game_info["prior_states"] = prior_states_dict[game_info["game_id"]]
    print(f"Time taken to fetch prior states: {time.time() - start:.2f} seconds")

    # Return the fetched game information
    return games_info


def update_full_season(season, db_path, force=False):
    """
    Updates the full season of games in the database.

    Args:
        season (str): The season to update in 'YYYY-YYYY' format.
        db_path (str): The path to the SQLite database.
        force (bool, optional): If True, updates all games in the season. If False, only updates games that are not completed or do not have a final state in the GameStates table or a finalized prior state in the PriorStates table and are at most 1 day into the future. Defaults to False.
    """
    # Validate the season format
    validate_season_format(season, abbreviated=False)

    # Update the scheduled games for the season
    update_scheduled_games(season, db_path)

    # Connect to the SQLite database
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()

        # If force is True, select all game IDs from the Games table for the season
        if force:
            cursor.execute(
                """
                SELECT game_id
                FROM Games 
                WHERE season = ? AND season_type IN ('Regular Season', 'Post Season')
                ORDER BY date_time_est ASC
                """,
                (season,),
            )
        # If force is False, select game IDs from the Games table for the season that are not completed or do not have a final state in the GameStates table or a finalized prior state in the PriorStates table and are at most 1 day in the future
        else:
            # Get the current date and time in EST
            current_datetime_est = datetime.now(pytz.timezone("US/Eastern"))

            cursor.execute(
                """
                SELECT g.game_id
                FROM Games g
                LEFT JOIN GameStates gs ON g.game_id = gs.game_id AND gs.is_final_state = 1
                LEFT JOIN PriorStates ps ON g.game_id = ps.game_id AND ps.are_prior_states_finalized = 1
                WHERE g.season = ?
                AND g.season_type IN ('Regular Season', 'Post Season')
                AND NOT (
                    g.status = 'Completed'
                    AND gs.game_id IS NOT NULL
                    AND ps.game_id IS NOT NULL
                )
                AND julianday(g.date_time_est) - julianday(?) <= 1
                ORDER BY g.date_time_est ASC
                """,
                (season, current_datetime_est.isoformat()),
            )

        # Fetch all the rows from the query result and extract the game ID from each row to get a list of game IDs
        game_ids = [row[0] for row in cursor.fetchall()]

    # Loop over each game ID
    for game_id in tqdm(
        game_ids, desc=f"Updating {season}", unit="game", dynamic_ncols=True
    ):
        # Fetch and store game information for each game ID
        get_games_info(
            game_id,
            db_path,
            include_prior_states=True,
            save_to_database=True,
        )


def update_multiple_games(game_ids, db_path, include_prior_states=True):
    """
    Updates the database with game information for multiple games.

    Args:
        game_ids (list): A list of game IDs to fetch information for.
        db_path (str): The path to the database.
        include_prior_states (bool, optional): If True, the function will include prior states in the fetched information. Defaults to True.
                                                Including prior states will update all prior games for the home and away team of the game_id (within the same season).
                                                This will also update and save a feature set based off of the prior states.

    Returns:
        None. The function updates the database directly with the fetched game information.
    """

    # Loop over each game ID
    for game_id in tqdm(
        game_ids, desc="Updating database", unit="game", dynamic_ncols=True
    ):
        # Fetch and store game information for each game ID
        get_games_info(
            game_id,
            db_path,
            include_prior_states=include_prior_states,
            save_to_database=True,
        )
