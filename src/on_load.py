import logging
import sqlite3

from src.config import config
from src.features import create_feature_sets, save_feature_sets
from src.game_states import create_game_states, save_game_states
from src.games import lookup_basic_game_info
from src.pbp import get_pbp, save_pbp
from src.prior_states import determine_prior_states_needed, load_prior_states
from src.schedule import determine_current_season, update_schedule
from src.utils import validate_game_ids, validate_season_format

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s",
)

# Configuration
DB_PATH = config["database"]["path"]


def update_database(season="Current", db_path=DB_PATH, model_id=None):
    # STEP 1: Update Schedule
    if season == "Current":
        season = determine_current_season()
    else:
        validate_season_format(season)

    update_schedule(season=season)

    # STEP 2: Update Game Data
    game_ids = get_games_needing_game_state_update(season, db_path)
    update_game_data(game_ids, db_path)

    # STEP 3: Update Pre Game Data
    game_ids = get_games_with_incomplete_pre_game_data(season, db_path)
    update_pre_game_data(game_ids, db_path)

    # STEP 4: Update Predictions
    if model_id:
        pass


def update_game_data(game_ids, db_path=DB_PATH):
    validate_game_ids(game_ids)
    games = lookup_basic_game_info(game_ids, db_path)
    pbp_logs = get_pbp(game_ids)
    save_pbp(pbp_logs, db_path)
    game_states = {}
    for game in games:
        game_id = game["game_id"]
        game_states[game_id] = create_game_states(
            pbp_logs[game_id],
            game["home_team"],
            game["away_team"],
            game_id,
            game["date_time_est"].split("T")[0],
        )
    save_game_states(game_states, db_path)


def update_pre_game_data(game_ids, db_path=DB_PATH):
    validate_game_ids(game_ids)
    prior_states_needed = determine_prior_states_needed(game_ids, db_path)
    prior_states, missing_prior_states = load_prior_states(prior_states_needed, db_path)
    feature_sets = create_feature_sets(game_ids, prior_states, db_path)
    save_feature_sets(feature_sets, db_path)
    finalized_game_ids = [
        game_id for game_id in game_ids if game_id not in missing_prior_states
    ]
    # Update pre_game_data_finalized flag
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.executemany(
            """
            UPDATE Games
            SET pre_game_data_finalized = 1
            WHERE game_id = ?
        """,
            [(game_id,) for game_id in finalized_game_ids],
        )
        conn.commit()


def get_games_needing_game_state_update(season, db_path=DB_PATH):
    """
    Identify games that need to have their data updated for a given season.

    This function filters games by the specified season and checks for
    completed games that have not yet been marked as fully updated in terms
    of play-by-play logs and game states.

    Args:
        season: The season to filter games by (e.g., '2023-2024').
        db_path: The file path to the SQLite database. Default to DB_PATH from config.

    Returns:
        A list of game_ids for games that need to be updated.
    """
    with sqlite3.connect(db_path) as db_connection:
        cursor = db_connection.cursor()
        # Query to identify games needing updates, including 'In Progress' games
        cursor.execute(
            """
            SELECT game_id 
            FROM Games 
            WHERE season = ? 
              AND (status = 'Completed' OR status = 'In Progress')
              AND game_data_finalized = False;
        """,
            (season,),
        )

        games_to_update = cursor.fetchall()

    # Return a list of game_ids
    return [game_id for (game_id,) in games_to_update]


def get_games_with_incomplete_pre_game_data(season, db_path=DB_PATH):
    """
    Retrieves game_ids for games that should have their pre_game_data_finalized flag set to True but do not.

    The function checks:
    1. Games in the specified season with status 'Completed' or 'In Progress' and pre_game_data_finalized = 0.
    2. Games in the specified season with status 'Not Started' where all prior games involving the teams
       (home team or away team) have game_data_finalized = True.

    Parameters:
    season (str): The season to filter games by.
    db_path (str): Path to the SQLite database file.

    Returns:
    list: List of game_ids that need to have their pre_game_data_finalized flag updated.
    """

    query = """
    SELECT game_id
    FROM Games
    WHERE season = ?
      AND pre_game_data_finalized = 0
      AND (status = 'Completed' OR status = 'In Progress')
    
    UNION

    SELECT g1.game_id
    FROM Games g1
    WHERE g1.season = ?
      AND g1.pre_game_data_finalized = 0
      AND g1.status = 'Not Started'
      AND NOT EXISTS (
          SELECT 1
          FROM Games g2
          WHERE g2.season = ?
            AND g2.date_time_est < g1.date_time_est
            AND (g2.home_team = g1.home_team OR g2.away_team = g1.home_team OR g2.home_team = g1.away_team OR g2.away_team = g1.away_team)
            AND g2.game_data_finalized = 0
      )
    """

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(query, (season, season, season))
        results = cursor.fetchall()

    return [row[0] for row in results]


def get_games_for_prediction_update(season, model_id, db_path=DB_PATH):
    pass
