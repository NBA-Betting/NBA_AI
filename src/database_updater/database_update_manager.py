"""
database_update_manager.py

Description:
This module handles the core process of updating the database with game data, including schedule updates, play-by-play logs,
game states, prior states, feature sets, and predictions.
It consists of functions to:
- Update the schedule for a given season.
- Update play-by-play logs and game states for games needing updates.
- Update prior states and feature sets for games with incomplete pre-game data.
- Update predictions for games needing updated predictions.

Functions:
    - update_database(season="Current", predictor=None, db_path=DB_PATH): Orchestrates the full update process for the specified season.
    - update_game_data(season, db_path=DB_PATH): Updates play-by-play logs and game states for games needing updates.
    - update_pre_game_data(season, db_path=DB_PATH): Updates prior states and feature sets for games with incomplete pre-game data.
    - update_prediction_data(season, predictor, db_path=DB_PATH): Generates and saves predictions for upcoming games.
    - get_games_needing_game_state_update(season, db_path=DB_PATH): Retrieves game_ids for games needing game state updates.
    - get_games_with_incomplete_pre_game_data(season, db_path=DB_PATH): Retrieves game_ids for games with incomplete pre-game data.
    - get_games_for_prediction_update(season, predictor, db_path=DB_PATH): Retrieves game_ids for games needing updated predictions.
    - main(): Main function to handle command-line arguments and orchestrate the update process.

Usage:
- Typically run as part of a larger data processing pipeline.
- Script can be run directly from the command line (project root) to update the database with the latest game data and predictions.
    python -m src.database_updater.database_update_manager --log_level=DEBUG --season=2023-2024 --predictor=Linear
- Successful execution will update the database with the latest game data and predictions for the specified season.
"""

import argparse
import logging
import sqlite3

from tqdm import tqdm

from src.config import config
from src.database_updater.boxscores import get_boxscores, save_boxscores
from src.database_updater.game_states import create_game_states, save_game_states
from src.database_updater.pbp import get_pbp, save_pbp
from src.database_updater.players import update_players
from src.database_updater.prior_states import (
    determine_prior_states_needed,
    load_prior_states,
)
from src.database_updater.schedule import update_schedule
from src.logging_config import setup_logging
from src.predictions.features import create_feature_sets, save_feature_sets
from src.predictions.prediction_manager import (
    make_pre_game_predictions,
    save_predictions,
)
from src.utils import log_execution_time, lookup_basic_game_info

# Configuration
DB_PATH = config["database"]["path"]


@log_execution_time()
def update_database(
    season="Current", predictor=None, db_path=DB_PATH, skip_players=False
):
    """
    Orchestrates the full update process for the specified season.

    Parameters:
        season (str): The season to update (default is "Current").
        predictor: The prediction model to use (default is None).
        db_path (str): The path to the database (default is from config).
        skip_players (bool): If True, skip player enrichment (default is False).

    Returns:
        None
    """
    # STEP 1: Update Schedule
    update_schedule(season)
    # STEP 2: Update Players List (optional - can be slow on first run)
    if not skip_players:
        update_players(db_path)
    else:
        logging.info("Skipping player enrichment (--skip-players flag set)")
    # STEP 3: Update Game Data (Play-by-Play Logs, Game States, Boxscores)
    update_game_data(season, db_path)
    # STEP 3.5: Update Live Game Data (for In Progress games)
    update_live_game_data(season, db_path)
    # STEP 4: Update Pre Game Data (Prior States, Feature Sets)
    update_pre_game_data(season, db_path)
    # STEP 5: Update Predictions
    if predictor:
        update_prediction_data(season, predictor, db_path)


@log_execution_time()
def update_game_data(season, db_path=DB_PATH, chunk_size=100):
    """
    Updates play-by-play logs and game states for games needing updates.

    Parameters:
        season (str): The season to update.
        db_path (str): The path to the database (default is from config).
        chunk_size (int): Number of games to process at a time (default is 100).

    Returns:
        None
    """

    game_ids = get_games_needing_game_state_update(season, db_path)

    total_games = len(game_ids)
    total_chunks = (
        total_games + chunk_size - 1
    ) // chunk_size  # Ceiling division to calculate total chunks

    # Only log chunk information if there will be more than 1 chunk
    if total_chunks > 1:
        logging.info(f"Processing {total_games} games in {total_chunks} chunks.")

    # Process the games in chunks
    chunk_iterator = range(0, total_games, chunk_size)
    pbar = (
        tqdm(total=total_chunks, desc="Game data chunks", unit="chunk")
        if total_chunks > 1
        else None
    )

    for i in chunk_iterator:
        chunk_game_ids = game_ids[i : i + chunk_size]

        try:
            basic_game_info = lookup_basic_game_info(chunk_game_ids, db_path)
            pbp_data = get_pbp(chunk_game_ids)
            save_pbp(pbp_data, db_path)

            game_state_inputs = {
                game_id: {
                    "home": basic_game_info[game_id]["home"],
                    "away": basic_game_info[game_id]["away"],
                    "date_time_est": basic_game_info[game_id]["date_time_est"],
                    "pbp_logs": game_info,
                }
                for game_id, game_info in pbp_data.items()
            }

            game_states = create_game_states(game_state_inputs)
            save_game_states(game_states)

            # Collect boxscore data for completed games
            boxscore_data = get_boxscores(chunk_game_ids)
            save_boxscores(boxscore_data, db_path)

            # Update progress bar
            if pbar:
                pbar.update(1)

        except Exception as e:
            logging.error(f"Error processing chunk starting at index {i}: {str(e)}")
            if pbar:
                pbar.update(1)
            continue

    if pbar:
        pbar.close()

    # ADDITIONAL CHECK: Collect boxscores for games that have PBP but no PlayerBox
    # This handles cases where boxscores were added to the pipeline after initial collection
    missing_boxscores = get_games_needing_boxscores_only(season, db_path)

    if missing_boxscores:
        total_missing = len(missing_boxscores)
        total_chunks = (total_missing + chunk_size - 1) // chunk_size

        logging.info(
            f"Found {total_missing} games with PBP but missing PlayerBox. Collecting boxscores..."
        )

        pbar = (
            tqdm(total=total_chunks, desc="Boxscore backfill chunks", unit="chunk")
            if total_chunks > 1
            else None
        )

        for i in range(0, total_missing, chunk_size):
            chunk = missing_boxscores[i : i + chunk_size]

            try:
                boxscore_data = get_boxscores(chunk, check_game_status=False)
                save_boxscores(boxscore_data, db_path)

                if pbar:
                    pbar.update(1)
            except Exception as e:
                logging.error(f"Error collecting boxscores for chunk: {e}")
                if pbar:
                    pbar.update(1)
                continue

        if pbar:
            pbar.close()


@log_execution_time()
def update_live_game_data(season, db_path=DB_PATH):
    """
    Collects live play-by-play and boxscore data for in-progress games.

    This function specifically targets games with status='In Progress' to collect
    real-time game data using the live NBA API endpoints. This ensures the web app
    can display current scores and statistics for ongoing games.

    Parameters:
        season (str): The season to update.
        db_path (str): The path to the database (default is from config).

    Returns:
        None
    """
    game_ids = get_games_in_progress(season, db_path)

    if not game_ids:
        logging.debug("No in-progress games found.")
        return

    logging.info(f"Collecting live data for {len(game_ids)} in-progress games...")

    try:
        # Collect live play-by-play data
        pbp_data = get_pbp(game_ids, pbp_endpoint="live")
        save_pbp(pbp_data)

        # Get basic game info for game states
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            basic_game_info = {}
            for game_id in game_ids:
                cursor.execute(
                    "SELECT date_time_est FROM Games WHERE game_id = ?", (game_id,)
                )
                row = cursor.fetchone()
                if row:
                    basic_game_info[game_id] = {"date_time_est": row[0]}

        # Create game states from live PBP
        game_state_inputs = {
            game_id: {
                "date_time_est": basic_game_info[game_id]["date_time_est"],
                "pbp_logs": game_info,
            }
            for game_id, game_info in pbp_data.items()
        }

        game_states = create_game_states(game_state_inputs)
        save_game_states(game_states)

        # Collect live boxscores (with check_game_status=True to use live endpoint)
        boxscore_data = get_boxscores(game_ids, check_game_status=True, db_path=db_path)
        save_boxscores(boxscore_data, db_path)

        logging.info(f"Live data collection complete for {len(game_ids)} games.")

    except Exception as e:
        logging.error(f"Error collecting live game data: {e}")


@log_execution_time()
def update_pre_game_data(season, db_path=DB_PATH, chunk_size=100):
    """
    Updates prior states and feature sets for games with incomplete pre-game data.

    Parameters:
        season (str): The season to update.
        db_path (str): The path to the database (default is from config).
        chunk_size (int): Number of games to process at a time (default is 100).

    Returns:
        None
    """
    game_ids = get_games_with_incomplete_pre_game_data(season, db_path)

    total_games = len(game_ids)
    total_chunks = (total_games + chunk_size - 1) // chunk_size

    # Only log chunk information if there will be more than 1 chunk
    if total_chunks > 1:
        logging.info(
            f"Processing {total_games} games for pre-game data in {total_chunks} chunks."
        )

    # Process the games in chunks
    chunk_iterator = range(0, total_games, chunk_size)
    pbar = (
        tqdm(total=total_chunks, desc="Pre-game data chunks", unit="chunk")
        if total_chunks > 1
        else None
    )

    for i in chunk_iterator:
        chunk_game_ids = game_ids[i : i + chunk_size]

        try:
            prior_states_needed = determine_prior_states_needed(chunk_game_ids, db_path)
            prior_states_dict = load_prior_states(prior_states_needed, db_path)
            feature_sets = create_feature_sets(prior_states_dict, db_path)
            save_feature_sets(feature_sets, db_path)

            # Categorize games and prepare data for database update
            games_update_data = []
            for game_id, states in prior_states_dict.items():
                pre_game_data_finalized = int(
                    not states["missing_prior_states"]["home"]
                    and not states["missing_prior_states"]["away"]
                )
                games_update_data.append((pre_game_data_finalized, game_id))

            # Update database in a single transaction
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                cursor.executemany(
                    """
                    UPDATE Games
                    SET pre_game_data_finalized = ?
                    WHERE game_id = ?
                """,
                    games_update_data,
                )
                conn.commit()

            # Update progress bar
            if pbar:
                pbar.update(1)

        except Exception as e:
            logging.error(
                f"Error processing pre-game data chunk starting at index {i}: {str(e)}"
            )
            if pbar:
                pbar.update(1)
            continue

    if pbar:
        pbar.close()


@log_execution_time()
def update_prediction_data(season, predictor, db_path=DB_PATH):
    """
    Generates and saves predictions for upcoming games.

    Parameters:
        season (str): The season to update.
        predictor: The prediction model to use.
        db_path (str): The path to the database (default is from config).

    Returns:
        None
    """
    # Get game_ids for games needing updated predictions
    game_ids = get_games_for_prediction_update(season, predictor, db_path)

    # Generate and save predictions
    predictions = make_pre_game_predictions(game_ids, predictor)


@log_execution_time()
def get_games_needing_game_state_update(season, db_path=DB_PATH):
    """
    Retrieves game_ids for games needing game state updates.

    Parameters:
        season (str): The season to filter games by.
        db_path (str): The path to the database (default is from config).

    Returns:
        list: A list of game_ids for games that need to be updated.
    """
    with sqlite3.connect(db_path) as db_connection:
        cursor = db_connection.cursor()
        # Query to identify completed games needing data collection
        # Note: Only collect from finalized games to avoid incomplete data
        cursor.execute(
            """
            SELECT game_id 
            FROM Games 
            WHERE season = ?
              AND season_type IN ('Regular Season', 'Post Season') 
              AND status IN ('Completed', 'Final')
              AND game_data_finalized = False;
        """,
            (season,),
        )

        games_to_update = cursor.fetchall()

    # Return a list of game_ids
    return [game_id for (game_id,) in games_to_update]


@log_execution_time()
def get_games_in_progress(season, db_path=DB_PATH):
    """
    Retrieves game_ids for games currently in progress.

    Parameters:
        season (str): The season to filter games by.
        db_path (str): The path to the database (default is from config).

    Returns:
        list: A list of game_ids for games with status='In Progress'.
    """
    with sqlite3.connect(db_path) as db_connection:
        cursor = db_connection.cursor()
        cursor.execute(
            """
            SELECT game_id 
            FROM Games 
            WHERE season = ?
              AND season_type IN ('Regular Season', 'Post Season') 
              AND status = 'In Progress';
        """,
            (season,),
        )

        games_in_progress = cursor.fetchall()

    # Return a list of game_ids
    return [game_id for (game_id,) in games_in_progress]


@log_execution_time()
def get_games_needing_boxscores_only(season, db_path=DB_PATH):
    """
    Retrieves game_ids for games that have PBP data but are missing PlayerBox records.

    This handles cases where boxscores were added to the pipeline after initial collection.

    Parameters:
        season (str): The season to check.
        db_path (str): The path to the database (default is from config).

    Returns:
        list: A list of game_ids that need boxscore collection.
    """
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT g.game_id
            FROM Games g
            WHERE g.season = ?
            AND g.season_type IN ('Regular Season', 'Post Season')
            AND g.status IN ('Completed', 'Final')
            AND g.game_data_finalized = 1
            AND NOT EXISTS (
                SELECT 1 FROM PlayerBox pb WHERE pb.game_id = g.game_id
            )
            ORDER BY g.date_time_est
        """,
            (season,),
        )
        return [row[0] for row in cursor.fetchall()]


@log_execution_time()
def get_games_with_incomplete_pre_game_data(season, db_path=DB_PATH):
    """
    Retrieves game_ids for games with incomplete pre-game data.

    Parameters:
        season (str): The season to filter games by.
        db_path (str): The path to the database (default is from config).

    Returns:
        list: A list of game_ids that need to have their pre_game_data_finalized flag updated.
    """
    query = """
    SELECT game_id
    FROM Games
    WHERE season = ?
      AND season_type IN ("Regular Season", "Post Season")
      AND pre_game_data_finalized = 0
      AND (status = 'Completed' OR status = 'In Progress')
    
    UNION

    SELECT g1.game_id
    FROM Games g1
    WHERE g1.season = ?
      AND g1.season_type IN ("Regular Season", "Post Season")
      AND g1.pre_game_data_finalized = 0
      AND g1.status = 'Not Started'
      AND NOT EXISTS (
          SELECT 1
          FROM Games g2
          WHERE g2.season = ?
            AND g2.season_type IN ("Regular Season", "Post Season")
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


@log_execution_time()
def get_games_for_prediction_update(season, predictor, db_path=DB_PATH):
    """
    Retrieves game_ids for games needing updated predictions.

    Returns games that have pre_game_data_finalized = 1 but no predictions yet.
    This allows predictions to be generated for past games on-demand when the web
    app requests them.

    Parameters:
        season (str): The season to update.
        predictor (str): The predictor to check for existing predictions.
        db_path (str): The path to the database (default is from config).

    Returns:
        list: A list of game_ids that need updated predictions.
    """
    query = """
        SELECT g.game_id
        FROM Games g
        LEFT JOIN Predictions p ON g.game_id = p.game_id AND p.predictor = ?
        WHERE g.season = ?
            AND g.season_type IN ("Regular Season", "Post Season")
            AND g.pre_game_data_finalized = 1
            AND p.game_id IS NULL
        """

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(query, (predictor, season))
        result = cursor.fetchall()

    game_ids = [row[0] for row in result]

    return game_ids


def main():
    """
    Main function to handle command-line arguments and orchestrate the update process.
    """
    parser = argparse.ArgumentParser(
        description="Update the database with the latest game data and predictions."
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        help="The logging level. Default is INFO. DEBUG provides more details.",
    )
    parser.add_argument(
        "--season",
        default="Current",
        type=str,
        help="The season to update. Default is 'Current'.",
    )
    parser.add_argument(
        "--predictor",
        default=None,
        type=str,
        help="The predictor to use for predictions.",
    )
    parser.add_argument(
        "--skip-players",
        action="store_true",
        help="Skip player enrichment (useful for quick data collection).",
    )

    args = parser.parse_args()
    log_level = args.log_level.upper()
    setup_logging(log_level=log_level)

    update_database(args.season, args.predictor, skip_players=args.skip_players)


if __name__ == "__main__":
    main()
