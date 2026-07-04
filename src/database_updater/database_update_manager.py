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
    python -m src.database_updater.database_update_manager --log_level=DEBUG --season=2025-2026 --predictor=Linear
- Successful execution will update the database with the latest game data and predictions for the specified season.
"""

import argparse
import logging
from datetime import datetime

from tqdm import tqdm

from src.config import config
from src.database import get_db
from src.database_updater.betting import update_betting_data
from src.database_updater.boxscores import get_boxscores, save_boxscores
from src.database_updater.game_states import create_game_states, save_game_states
from src.database_updater.nba_official_injuries import update_nba_official_injuries
from src.database_updater.pbp import get_pbp, save_pbp
from src.database_updater.players import update_players
from src.database_updater.prior_states import (
    determine_prior_states_needed,
    load_prior_states,
)
from src.database_updater.schedule import update_schedule
from src.logging_config import setup_logging
from src.predictions.features import create_feature_sets, save_feature_sets
from src.predictions.prediction_manager import make_pre_game_predictions
from src.utils import log_execution_time, lookup_basic_game_info

# Configuration
DB_PATH = config["database"]["path"]


def _validate_pbp(game_ids, db_path=DB_PATH, suppress_no_final_state=False):
    """
    Validate PBP data after collection.

    Checks for missing PBP, low play counts, stale data, duplicate plays.
    Critical issues are logged but don't block pipeline (data may be refetched).

    Args:
        game_ids: List of game IDs to validate
        db_path: Database path
        suppress_no_final_state: If True, skip NO_FINAL_STATE check (used during pipeline
            when GameStates haven't been created yet)
    """
    from src.database_updater.validators import PbPValidator

    if not game_ids:
        return

    validator = PbPValidator()

    with get_db(db_path) as conn:
        cursor = conn.cursor()
        result = validator.validate(game_ids, cursor)

        # Filter out NO_FINAL_STATE if suppressed (will be created in next stage)
        if suppress_no_final_state:
            result.issues = [i for i in result.issues if i.check_id != "NO_FINAL_STATE"]

        if result.has_critical_issues:
            logging.error(f"PBP validation failed:\n{result.summary()}")
        elif result.has_warnings:
            logging.warning(f"PBP validation warnings:\n{result.summary()}")
        else:
            logging.debug(f"PBP validation: PASS ({len(game_ids)} games)")


def _validate_game_states(game_ids, db_path=DB_PATH):
    """
    Validate GameStates data after creation.

    Checks for missing final states, low state counts, invalid scores, duplicates.
    Critical issues indicate broken parsing and may require regeneration.
    """
    from src.database_updater.validators import GameStatesValidator

    if not game_ids:
        return

    validator = GameStatesValidator()

    with get_db(db_path) as conn:
        cursor = conn.cursor()
        result = validator.validate(game_ids, cursor)

        if result.has_critical_issues:
            logging.error(f"GameStates validation failed:\n{result.summary()}")
        elif result.has_warnings:
            logging.warning(f"GameStates validation warnings:\n{result.summary()}")
        else:
            logging.debug(f"GameStates validation: PASS ({len(game_ids)} games)")


@log_execution_time()
def update_database(
    season="Current",
    predictor=None,
    db_path=DB_PATH,
):
    """
    Orchestrates the full update process for the specified season.

    Parameters:
        season (str): The season to update (default is "Current").
        predictor: The prediction model to use (default is None).
        db_path (str): The path to the database (default is from config).

    Returns:
        None
    """
    from src.utils import determine_current_season

    # Resolve "Current" to actual season name once at start
    if season == "Current":
        season = determine_current_season()

    logging.info(f"=== Updating database for {season} ===")

    # Check for large backfill and warn user
    try:
        n_needing_pbp = len(get_games_needing_pbp_update(season, db_path))
        n_needing_box = len(get_games_needing_boxscores(season, db_path))
        n_pending = max(n_needing_pbp, n_needing_box)
        if n_pending > 20:
            est_minutes = (n_pending // 10) * 0.5 + 2  # rough estimate
            logging.info(
                f"  Backfill detected: ~{n_pending} games need updating. "
                f"This involves many NBA API calls with rate-limit pauses "
                f"and may take ~{est_minutes:.0f}+ minutes."
            )
    except Exception:
        pass  # Don't let the estimate check block the actual update

    # STEP 1: Update Schedule
    update_schedule(season)

    # STEP 2: Update Players List
    update_players(db_path)

    # STEP 3: Update Injury Data
    update_injury_data(season, db_path)

    # STEP 4: Update Betting Data
    update_betting_lines(season, db_path)

    # STEP 5: Update Play-by-Play Data
    update_pbp_data(season, db_path)

    # STEP 6: Update Game States (parsed from PbP)
    update_game_state_data(season, db_path)

    # STEP 7: Update Boxscore Data
    update_boxscore_data(season, db_path)

    # STEP 8: Update Pre Game Data (Prior States, Feature Sets)
    update_pre_game_data(season, db_path)

    # STEP 9: Update Predictions
    if predictor:
        update_prediction_data(season, predictor, db_path)


@log_execution_time()
def update_pbp_data(season, db_path=DB_PATH, chunk_size=100):
    """
    Collects play-by-play logs for games needing updates.

    This is Step 5 of the pipeline - fetches raw PBP data from NBA API.
    Automatically uses live endpoint for in-progress games.
    Does NOT parse into GameStates (that's Step 6).

    Parameters:
        season (str): The season to update.
        db_path (str): The path to the database (default is from config).
        chunk_size (int): Number of games to process at a time (default is 100).

    Returns:
        None
    """
    from src.utils import StageLogger

    stage_logger = StageLogger("PBP")

    game_ids = get_games_needing_pbp_update(season, db_path)

    if not game_ids:
        stage_logger.set_counts(added=0, updated=0, removed=0)
        stage_logger.log_complete()
        return

    total_games = len(game_ids)
    total_chunks = (total_games + chunk_size - 1) // chunk_size

    if total_chunks > 1:
        logging.debug(f"Processing {total_games} PBP games in {total_chunks} chunks.")

    pbar = (
        tqdm(total=total_chunks, desc="PBP chunks", unit="chunk", leave=False)
        if total_chunks > 1
        else None
    )

    total_added = 0
    total_updated = 0
    total_unchanged = 0

    for i in range(0, total_games, chunk_size):
        chunk_game_ids = game_ids[i : i + chunk_size]

        try:
            pbp_data = get_pbp(
                chunk_game_ids, pbp_endpoint="both", stage_logger=stage_logger
            )
            counts = save_pbp(pbp_data, db_path)

            total_added += counts["added"]
            total_updated += counts["updated"]
            total_unchanged += counts["unchanged"]

            if pbar:
                pbar.update(1)
        except Exception as e:
            logging.error(f"Error processing PBP chunk starting at index {i}: {str(e)}")
            if pbar:
                pbar.update(1)
            continue

    if pbar:
        pbar.close()

    stage_logger.set_counts(
        added=total_added, updated=total_updated, removed=0, total=total_games
    )
    stage_logger.log_complete()

    # Validate PBP data (suppress NO_FINAL_STATE since GameStates haven't been created yet)
    _validate_pbp(game_ids, db_path, suppress_no_final_state=True)


@log_execution_time()
def update_game_state_data(season, db_path=DB_PATH, chunk_size=100):
    """
    Parses play-by-play logs into structured GameStates.

    This is Step 6 of the pipeline - converts raw PBP into game state snapshots.
    Handles both completed and in-progress games.
    Requires PBP data to already exist (Step 5 must run first).

    Parameters:
        season (str): The season to update.
        db_path (str): The path to the database (default is from config).
        chunk_size (int): Number of games to process at a time (default is 100).

    Returns:
        None
    """
    from src.utils import StageLogger

    stage_logger = StageLogger("GameStates")

    game_ids = get_games_needing_game_state_update(season, db_path)

    if not game_ids:
        stage_logger.set_counts(added=0, updated=0, removed=0)
        stage_logger.log_complete()
        return

    total_games = len(game_ids)
    total_chunks = (total_games + chunk_size - 1) // chunk_size

    if total_chunks > 1:
        logging.debug(
            f"Processing {total_games} GameState games in {total_chunks} chunks."
        )

    pbar = (
        tqdm(total=total_chunks, desc="GameState chunks", unit="chunk", leave=False)
        if total_chunks > 1
        else None
    )

    total_created = 0
    total_updated = 0

    for i in range(0, total_games, chunk_size):
        chunk_game_ids = game_ids[i : i + chunk_size]

        try:
            basic_game_info = lookup_basic_game_info(chunk_game_ids, db_path)

            # Load PBP data for parsing
            pbp_data = {}
            with get_db(db_path) as conn:
                cursor = conn.cursor()
                for game_id in chunk_game_ids:
                    # Check if GameStates already exist for this game
                    cursor.execute(
                        "SELECT COUNT(*) FROM GameStates WHERE game_id = ?", (game_id,)
                    )
                    existing_count = cursor.fetchone()[0]

                    cursor.execute(
                        "SELECT log_data FROM PbP_Logs WHERE game_id = ?", (game_id,)
                    )
                    rows = cursor.fetchall()
                    if rows:
                        import json

                        pbp_data[game_id] = {
                            "logs": [json.loads(row[0]) for row in rows],
                            "had_existing": existing_count > 0,
                        }

            # Create GameStates from PBP
            game_state_inputs = {
                game_id: {
                    "home": basic_game_info[game_id]["home"],
                    "away": basic_game_info[game_id]["away"],
                    "date_time_utc": basic_game_info[game_id]["date_time_utc"],
                    "status": basic_game_info[game_id]["status"],
                    "pbp_logs": game_info["logs"],
                }
                for game_id, game_info in pbp_data.items()
            }

            game_states = create_game_states(game_state_inputs)
            save_game_states(game_states)

            # Track added vs updated
            for game_id in game_states:
                if pbp_data.get(game_id, {}).get("had_existing"):
                    total_updated += 1
                else:
                    total_created += 1

            # Set game_data_finalized flag for games with complete PBP/GameStates
            pbp_finalized = _mark_pbp_games_finalized(chunk_game_ids, db_path)
            if pbp_finalized:
                logging.debug(
                    f"Marked {len(pbp_finalized)} games with PBP/GameStates finalized."
                )

            if pbar:
                pbar.update(1)
        except Exception as e:
            logging.error(
                f"Error processing GameState chunk starting at index {i}: {str(e)}"
            )
            if pbar:
                pbar.update(1)
            continue

    if pbar:
        pbar.close()

    stage_logger.set_counts(
        added=total_created, updated=total_updated, removed=0, total=total_games
    )
    stage_logger.log_complete()

    # Validate GameStates data
    _validate_game_states(game_ids, db_path)


@log_execution_time()
def update_boxscore_data(season, db_path=DB_PATH, chunk_size=30):
    """
    Collects boxscore data (PlayerBox, TeamBox) for games needing updates.

    This is Step 7 of the pipeline - fetches boxscore statistics from NBA API.
    Independent of PBP/GameStates (can run in parallel if needed).

    Parameters:
        season (str): The season to update.
        db_path (str): The path to the database (default is from config).
        chunk_size (int): Number of games to process at a time (default is 100).

    Returns:
        None
    """
    from src.database_updater.validators import BoxscoresValidator
    from src.utils import StageLogger

    stage_logger = StageLogger("Boxscores")
    validator = BoxscoresValidator()

    game_ids = get_games_needing_boxscores(season, db_path)

    if not game_ids:
        stage_logger.set_counts(added=0, updated=0, removed=0)
        stage_logger.log_complete()
        return

    total_games = len(game_ids)
    total_chunks = (total_games + chunk_size - 1) // chunk_size

    if total_chunks > 1:
        logging.debug(
            f"Processing {total_games} boxscore games in {total_chunks} chunks."
        )

    pbar = (
        tqdm(total=total_chunks, desc="Boxscore chunks", unit="chunk", leave=False)
        if total_chunks > 1
        else None
    )

    total_added = 0
    total_updated = 0

    for i in range(0, total_games, chunk_size):
        chunk_game_ids = game_ids[i : i + chunk_size]

        try:
            boxscore_data = get_boxscores(
                chunk_game_ids,
                check_game_status=True,
                stage_logger=stage_logger,
                db_path=db_path,
            )
            counts = save_boxscores(boxscore_data, db_path)

            total_added += counts["added"]
            total_updated += counts["updated"]

            # Set boxscore_data_finalized flag for games with complete boxscore data
            boxscore_finalized = _mark_boxscore_games_finalized(chunk_game_ids, db_path)
            if boxscore_finalized:
                logging.debug(
                    f"Marked {len(boxscore_finalized)} games with boxscores finalized."
                )

            if pbar:
                pbar.update(1)
        except Exception as e:
            logging.error(
                f"Error processing boxscore chunk starting at index {i}: {str(e)}"
            )
            if pbar:
                pbar.update(1)
            continue

    if pbar:
        pbar.close()

    # Validate all processed games
    with get_db(db_path) as conn:
        cursor = conn.cursor()
        validation_result = validator.validate(game_ids, cursor)
        stage_logger.set_validation(validation_result)

    stage_logger.set_counts(
        added=total_added, updated=total_updated, removed=0, total=total_games
    )
    stage_logger.log_complete()


@log_execution_time()
def update_game_data(season, db_path=DB_PATH, chunk_size=100):
    """
    DEPRECATED: Legacy function that combines PBP, GameStates, and Boxscores.

    Use the individual functions instead:
    - update_pbp_data() for play-by-play collection
    - update_game_state_data() for parsing PBP into states
    - update_boxscore_data() for boxscore collection

    This function is kept for backwards compatibility but will call
    the new modular functions internally.

    Parameters:
        season (str): The season to update.
        db_path (str): The path to the database (default is from config).
        chunk_size (int): Number of games to process at a time (default is 100).

    Returns:
        None
    """
    logging.warning(
        "update_game_data() is deprecated. Use update_pbp_data(), update_game_state_data(), update_boxscore_data() instead."
    )

    # Call new modular functions
    update_pbp_data(season, db_path, chunk_size)
    update_game_state_data(season, db_path, chunk_size)
    update_boxscore_data(season, db_path, chunk_size)


# Helper functions for determining which games need updates


def get_games_needing_pbp_update(season, db_path):
    """
    Get game IDs that need PBP data updates.

    Logic:
    - In-progress games: status = 2 (In Progress) AND (no PBP OR last fetch >5 min ago)
    - Completed but not finalized: status = 3 AND game_data_finalized = 0 AND (no PBP OR last fetch >5 min ago)
    - Completed games missing PBP: status = 3 AND no PBP (ANY regular/postseason game)

    NOTE: Ensures complete coverage for current season regular/postseason games.

    Parameters:
        season (str): The season to check (e.g., "2024-2025" or "Current").
        db_path (str): Path to database.

    Returns:
        list: Game IDs needing PBP updates.
    """
    from src.utils import determine_current_season

    # Convert "Current" to actual season
    if season == "Current":
        season = determine_current_season()

    with get_db(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT game_id FROM Games
            WHERE season = ?
            AND season_type IN ('Regular Season', 'Post Season')
            AND (
                -- In-progress games (refetch every 5 minutes)
                (status = 2
                 AND (pbp_last_fetched_at IS NULL 
                      OR pbp_last_fetched_at < datetime('now', '-5 minutes')))
                
                OR
                
                -- Completed but not finalized (refetch every 5 minutes until finalized)
                (status = 3
                 AND game_data_finalized = 0
                 AND pbp_last_fetched_at IS NOT NULL
                 AND pbp_last_fetched_at < datetime('now', '-5 minutes'))
                
                OR
                
                -- ALL completed games missing PBP (no time window restriction)
                (status = 3
                 AND game_data_finalized = 0
                 AND NOT EXISTS (SELECT 1 FROM PbP_Logs WHERE PbP_Logs.game_id = Games.game_id))
            )
            ORDER BY date_time_utc
            """,
            (season,),
        )
        return [row[0] for row in cursor.fetchall()]


def update_pbp_and_gamestates(season, db_path, chunk_size):
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
                    "date_time_utc": basic_game_info[game_id]["date_time_utc"],
                    "status": basic_game_info[game_id]["status"],
                    "pbp_logs": game_info,
                }
                for game_id, game_info in pbp_data.items()
            }

            game_states = create_game_states(game_state_inputs)
            save_game_states(game_states)

            # Set game_data_finalized flag for games with complete PBP/GameStates
            pbp_finalized = _mark_pbp_games_finalized(chunk_game_ids, db_path)
            if pbp_finalized:
                logging.debug(
                    f"Marked {len(pbp_finalized)} games with PBP/GameStates finalized."
                )

            # Collect boxscore data for completed games
            boxscore_data = get_boxscores(chunk_game_ids)
            save_boxscores(boxscore_data, db_path)

            # Set boxscore_data_finalized flag for games with complete boxscore data
            boxscore_finalized = _mark_boxscore_games_finalized(chunk_game_ids, db_path)
            if boxscore_finalized:
                logging.debug(
                    f"Marked {len(boxscore_finalized)} games with boxscores finalized."
                )

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

    logging.info(
        f"Stage 3 complete. Updated game_data_finalized and boxscore_data_finalized flags."
    )

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

                # Mark games as finalized now that boxscores are backfilled
                boxscore_finalized = _mark_boxscore_games_finalized(chunk, db_path)
                if boxscore_finalized:
                    logging.debug(
                        f"Marked {len(boxscore_finalized)} backfilled games as finalized."
                    )

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
def update_injury_data(season, db_path=DB_PATH):
    """
    Collects NBA Official injury reports for the specified season.

    Smart fetching strategy:
    - Current season: Fetch ALL missing dates from season start to today (gap filling)
    - Historical seasons: Fetch ALL missing dates from season start to end (one-time backfill)
    - Subsequent runs: Skip dates already in database

    Parameters:
        season (str): The season to update (e.g., "2024-2025" or "Current").
        db_path (str): The path to the database (default is from config).

    Returns:
        None
    """
    from src.database_updater.nba_official_injuries import update_nba_official_injuries
    from src.database_updater.validators import InjuryValidator
    from src.utils import StageLogger, determine_current_season

    stage_logger = StageLogger("Injuries")

    current_season = determine_current_season()
    actual_season = current_season if season == "Current" else season

    try:
        # Fetch injury reports with season-wide gap filling
        counts = update_nba_official_injuries(
            season=actual_season, db_path=db_path, stage_logger=stage_logger
        )

        # Validate injury data
        if counts["added"] > 0 or counts["updated"] > 0:
            from datetime import datetime

            with get_db(db_path) as conn:
                cursor = conn.cursor()

                # Get date range for validation
                season_start_year = int(actual_season.split("-")[0])
                season_start = f"{season_start_year}-10-15"
                season_end = datetime.now().strftime("%Y-%m-%d")

                # Validate the data
                validator = InjuryValidator()
                validation_result = validator.validate(
                    (season_start, season_end), cursor
                )

                # Set validation in logger
                stage_logger.set_validation(validation_result)

                # Log validation issues
                if validation_result.has_critical_issues:
                    logging.error(
                        f"Critical validation issues: {validation_result.summary()}"
                    )
                elif validation_result.has_warnings:
                    logging.warning(
                        f"Validation warnings: {validation_result.summary()}"
                    )

        # Set counts and log completion
        stage_logger.set_counts(
            added=counts["added"], updated=counts["updated"], total=counts["total"]
        )
        stage_logger.log_complete()

    except Exception as e:
        logging.error(f"Error collecting NBA Official injury data: {e}")
        stage_logger.log_complete()


@log_execution_time()
def update_betting_lines(season, db_path=DB_PATH):
    """
    Collects betting data (spreads, totals, moneylines) from ESPN/Covers.

    3-tier fetching strategy:
    - Tier 1: ESPN API (games -7 to +2 days) - primary source with full odds
    - Tier 2: Covers matchups (completed games >7 days) - finalizes older games
    - Tier 3: Covers team schedules (historical backfill via CLI only)

    ESPN provides DraftKings odds 1-2 days before games and retains 5-7 days after.
    Covers provides closing lines for games outside ESPN window.

    Parameters:
        season (str): The season to update (e.g., "2024-2025" or "Current").
        db_path (str): The path to the database (default is from config).

    Returns:
        None
    """
    from src.utils import StageLogger, determine_current_season

    # Convert "Current" to actual season
    if season == "Current":
        season = determine_current_season()

    stage_logger = StageLogger("Betting")

    try:
        stats = update_betting_data(
            season=season, use_covers=True, stage_logger=stage_logger
        )

        # Inline warnings for collection issues
        # Any API errors should be visible (0 failure rate principle)
        if stats.get("errors", 0) > 0:
            logging.warning(
                f"Betting collection had {stats['errors']} API errors "
                f"(ESPN or Covers failures)"
            )

        # Warn if we processed games but got very few successful fetches
        # Note: cached games are a valid outcome (not a failure), so exclude from attempted
        total_attempted = (
            stats.get("espn_fetched", 0)
            + stats.get("covers_fetched", 0)
            + stats.get("errors", 0)
        )
        total_success = stats.get("espn_fetched", 0) + stats.get("covers_fetched", 0)
        # Only warn if we actually tried to fetch (not all cached) and got 0 successes
        if total_attempted > 10 and total_success == 0:
            logging.warning(
                f"Betting collection: 0 games fetched out of {total_attempted} attempted "
                f"(errors={stats.get('errors', 0)})"
            )

        # Get total count (any closing lines from ESPN or Covers)
        with get_db(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT COUNT(*) FROM Betting WHERE espn_closing_spread IS NOT NULL OR covers_closing_spread IS NOT NULL"
            )
            total = cursor.fetchone()[0]

        # Set counts and log completion
        stage_logger.set_counts(
            added=stats["saved"],
            updated=0,  # Betting uses INSERT OR REPLACE, can't distinguish
            total=total,
        )
        stage_logger.log_complete()

    except Exception as e:
        logging.error(f"Error collecting betting data: {e}")
        raise


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
    from src.database_updater.validators import FeaturesValidator
    from src.utils import StageLogger

    stage_logger = StageLogger("Features")
    validator = FeaturesValidator()

    game_ids = get_games_with_incomplete_pre_game_data(season, db_path)

    if not game_ids:
        stage_logger.set_counts(added=0, updated=0, removed=0)
        stage_logger.log_complete()
        return

    total_games = len(game_ids)
    total_chunks = (total_games + chunk_size - 1) // chunk_size

    # Only log chunk information if there will be more than 1 chunk
    if total_chunks > 1:
        logging.debug(
            f"Processing {total_games} games for pre-game data in {total_chunks} chunks."
        )

    # Process the games in chunks
    chunk_iterator = range(0, total_games, chunk_size)
    pbar = (
        tqdm(total=total_chunks, desc="Features chunks", unit="chunk", leave=False)
        if total_chunks > 1
        else None
    )

    total_added = 0
    total_updated = 0
    total_missing_priors = 0

    # Check which games already have features (for added vs updated tracking)
    existing_features = set()
    with get_db(db_path) as conn:
        cursor = conn.cursor()
        placeholders = ",".join(["?"] * len(game_ids))
        cursor.execute(
            f"SELECT game_id FROM Features WHERE game_id IN ({placeholders})", game_ids
        )
        existing_features = {row[0] for row in cursor.fetchall()}

    for i in chunk_iterator:
        chunk_game_ids = game_ids[i : i + chunk_size]

        try:
            prior_states_needed = determine_prior_states_needed(chunk_game_ids, db_path)
            prior_states_dict = load_prior_states(prior_states_needed, db_path)
            feature_sets = create_feature_sets(prior_states_dict, db_path)
            save_feature_sets(feature_sets, db_path)

            # Track added vs updated (only for games with actual features)
            for game_id, features in feature_sets.items():
                if features:  # Only count if features were actually created
                    if game_id in existing_features:
                        total_updated += 1
                    else:
                        total_added += 1

            # Categorize games and prepare data for database update
            games_update_data = []
            for game_id, states in prior_states_dict.items():
                # pre_game_data_finalized=1 means we've collected all AVAILABLE prior data
                # (even if that's 0 games for opening night). It's finalized if there are
                # no missing states that we tried to load but couldn't find.
                has_no_missing_home = not states["missing_prior_states"]["home"]
                has_no_missing_away = not states["missing_prior_states"]["away"]
                pre_game_data_finalized = int(
                    has_no_missing_home and has_no_missing_away
                )
                games_update_data.append((pre_game_data_finalized, game_id))

                if not pre_game_data_finalized:
                    total_missing_priors += 1

            # Update database in a single transaction
            with get_db(db_path) as conn:
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

    # Validate the processed games
    with get_db(db_path) as conn:
        cursor = conn.cursor()
        validation_result = validator.validate(game_ids, cursor)
        stage_logger.set_validation(validation_result)

    # Log summary with additional context about data loading issues
    stage_logger.set_counts(
        added=total_added, updated=total_updated, removed=0, total=total_games
    )
    if total_missing_priors > 0:
        stage_logger.set_extra_info(f"({total_missing_priors} incomplete)")
    stage_logger.log_complete()


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
    from src.database_updater.validators import PredictionsValidator
    from src.utils import StageLogger

    stage_logger = StageLogger(f"Predictions ({predictor})")
    validator = PredictionsValidator()

    # Get game_ids for games needing updated predictions
    game_ids = get_games_for_prediction_update(season, predictor, db_path)

    if not game_ids:
        stage_logger.set_counts(added=0, updated=0, removed=0)
        stage_logger.log_complete()
        return

    # Generate and save predictions
    predictions = make_pre_game_predictions(game_ids, predictor, save=True)

    # Track counts
    total_added = len(predictions) if predictions else 0

    # Validate the predictions
    with get_db(db_path) as conn:
        cursor = conn.cursor()
        validation_result = validator.validate(game_ids, cursor, predictor)
        stage_logger.set_validation(validation_result)

    stage_logger.set_counts(
        added=total_added, updated=0, removed=0, total=len(game_ids)
    )
    stage_logger.log_complete()


def get_games_needing_boxscores(season, db_path):
    """
    Get game IDs that need boxscore data updates.

    Logic:
    - In-progress games: status = 2 (In Progress) AND (no boxscores OR last fetch >5 min ago)
    - Completed but not finalized: status = 3 AND boxscore_data_finalized = 0 AND (no boxscores OR last fetch >5 min ago)
    - Completed games missing boxscores: status = 3 AND no boxscores (ANY regular/postseason game)

    NOTE: Ensures complete coverage for current season regular/postseason games.

    Parameters:
        season (str): The season to check (e.g., "2024-2025" or "Current").
        db_path (str): Path to database.

    Returns:
        list: Game IDs needing boxscore updates.
    """
    from src.utils import determine_current_season

    # Convert "Current" to actual season
    if season == "Current":
        season = determine_current_season()

    with get_db(db_path) as conn:
        cursor = conn.cursor()

        # Check if boxscore_last_fetched_at column exists
        cursor.execute("PRAGMA table_info(Games)")
        columns = [row[1] for row in cursor.fetchall()]
        has_timestamp_column = "boxscore_last_fetched_at" in columns

        if has_timestamp_column:
            # Use timestamp-based caching if column exists
            cursor.execute(
                """
                SELECT game_id FROM Games
                WHERE season = ?
                AND season_type IN ('Regular Season', 'Post Season')
                AND (
                    -- In-progress games (refetch every 5 minutes)
                    (status = 2
                     AND (boxscore_last_fetched_at IS NULL 
                          OR boxscore_last_fetched_at < datetime('now', '-5 minutes')))
                    
                    OR
                    
                    -- Completed but not finalized (refetch every 5 minutes until finalized)
                    (status = 3
                     AND boxscore_data_finalized = 0
                     AND boxscore_last_fetched_at IS NOT NULL
                     AND boxscore_last_fetched_at < datetime('now', '-5 minutes'))
                    
                    OR
                    
                    -- ALL completed games missing boxscores (no time window restriction)
                    (status = 3
                     AND boxscore_data_finalized = 0
                     AND NOT EXISTS (SELECT 1 FROM PlayerBox WHERE PlayerBox.game_id = Games.game_id))
                )
                ORDER BY date_time_utc
                """,
                (season,),
            )
        else:
            # Fallback to simple logic if column doesn't exist (first run)
            cursor.execute(
                """
                SELECT game_id FROM Games
                WHERE season = ?
                AND season_type IN ('Regular Season', 'Post Season')
                AND status IN (2, 3)  -- In Progress or Final
                AND boxscore_data_finalized = 0
                AND NOT EXISTS (SELECT 1 FROM PlayerBox WHERE PlayerBox.game_id = Games.game_id)
                ORDER BY date_time_utc
                """,
                (season,),
            )

        return [row[0] for row in cursor.fetchall()]


@log_execution_time()
def get_games_needing_game_state_update(season, db_path=DB_PATH):
    """
    Retrieves game_ids for games needing GameState creation/update from PBP logs.

    GameStates should always catch up to PBP. This function returns games where:
    1. PBP exists but no GameStates exist
    2. PBP was fetched more recently than GameStates were created
    3. Completed games without a final state marker

    Parameters:
        season (str): The season to filter games by (e.g., "2024-2025" or "Current").
        db_path (str): The path to the database (default is from config).

    Returns:
        list: A list of game_ids for games that need GameState parsing.
    """
    from src.utils import determine_current_season

    # Convert "Current" to actual season
    if season == "Current":
        season = determine_current_season()

    with get_db(db_path) as db_connection:
        cursor = db_connection.cursor()
        cursor.execute(
            """
            SELECT game_id 
            FROM Games 
            WHERE season = ?
              AND season_type IN ('Regular Season', 'Post Season') 
              AND status IN (2, 3)  -- In Progress or Final
              AND EXISTS (SELECT 1 FROM PbP_Logs WHERE PbP_Logs.game_id = Games.game_id)
              AND (
                  -- Case 1: No GameStates exist at all
                  NOT EXISTS (SELECT 1 FROM GameStates WHERE GameStates.game_id = Games.game_id)
                  
                  -- Case 2: PBP was fetched after GameStates were created (stale GameStates)
                  OR (gamestates_last_created_at IS NOT NULL 
                      AND pbp_last_fetched_at > gamestates_last_created_at)
                  
                  -- Case 3: Completed game but no final state marker (incomplete parsing)
                  OR (status = 3 
                      AND NOT EXISTS (
                          SELECT 1 FROM GameStates gs 
                          WHERE gs.game_id = Games.game_id AND gs.is_final_state = 1
                      ))
              )
            ORDER BY date_time_utc;
        """,
            (season,),
        )

        games_to_update = cursor.fetchall()

    return [game_id for (game_id,) in games_to_update]


@log_execution_time()
def get_games_needing_boxscores_only(season, db_path=DB_PATH):
    """
    Retrieves game_ids for games that have PBP data but are missing boxscores.

    This handles cases where PBP/GameStates succeeded but boxscore collection failed.

    Parameters:
        season (str): The season to check (e.g., "2024-2025" or "Current").
        db_path (str): The path to the database (default is from config).

    Returns:
        list: A list of game_ids that need boxscore collection.
    """
    from src.utils import determine_current_season

    # Convert "Current" to actual season
    if season == "Current":
        season = determine_current_season()

    with get_db(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT g.game_id
            FROM Games g
            WHERE g.season = ?
            AND g.season_type IN ('Regular Season', 'Post Season')
            AND g.status = 3  -- Final
            AND g.game_data_finalized = 1
            AND g.boxscore_data_finalized = 0
            ORDER BY g.date_time_utc
        """,
            (season,),
        )
        return [row[0] for row in cursor.fetchall()]


@log_execution_time()
def get_games_with_incomplete_pre_game_data(season, db_path=DB_PATH):
    """
    Retrieves game_ids for games with incomplete pre-game data.

    Excludes postponed games (status_text = 'PPD') since they will never be played
    and should not have features or predictions generated.

    Parameters:
        season (str): The season to filter games by (e.g., "2024-2025" or "Current").
        db_path (str): The path to the database (default is from config).

    Returns:
        list: A list of game_ids that need to have their pre_game_data_finalized flag updated.
    """
    from src.utils import determine_current_season

    # Convert "Current" to actual season
    if season == "Current":
        season = determine_current_season()

    query = """
    SELECT game_id
    FROM Games
    WHERE season = ?
      AND season_type IN ("Regular Season", "Post Season")
      AND pre_game_data_finalized = 0
      AND game_data_finalized = 1
      AND status IN (2, 3)  -- In Progress or Final
      AND status_text != 'PPD'  -- Exclude postponed games

    UNION

    SELECT g1.game_id
    FROM Games g1
    WHERE g1.season = ?
      AND g1.season_type IN ("Regular Season", "Post Season")
      AND g1.pre_game_data_finalized = 0
      AND g1.status = 1  -- Not Started
      AND g1.status_text != 'PPD'  -- Exclude postponed games
      AND NOT EXISTS (
          SELECT 1
          FROM Games g2
          WHERE g2.season = ?
            AND g2.season_type IN ("Regular Season", "Post Season")
            AND g2.date_time_utc < g1.date_time_utc
            AND (g2.home_team = g1.home_team OR g2.away_team = g1.home_team OR g2.home_team = g1.away_team OR g2.away_team = g1.away_team)
            AND (g2.game_data_finalized = 0 OR g2.boxscore_data_finalized = 0)
            AND g2.status_text != 'PPD'  -- Ignore postponed games when checking for blocking
      )
    """

    with get_db(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(query, (season, season, season))
        results = cursor.fetchall()

    return [row[0] for row in results]


def _mark_pbp_games_finalized(game_ids, db_path=DB_PATH):
    """
    Marks games as having finalized PBP/GameStates data:
    - PBP_Logs (at least one play)
    - GameStates (with is_final_state=1)

    This is the core play-by-play data that can succeed independently of boxscores.

    Parameters:
        game_ids (list): List of game IDs to check.
        db_path (str): Path to database.

    Returns:
        list: Game IDs that were marked as finalized.
    """
    finalized = []

    with get_db(db_path) as conn:
        cursor = conn.cursor()

        for game_id in game_ids:
            # Check if PBP and GameStates exist
            cursor.execute(
                """
                SELECT 
                    (SELECT COUNT(*) FROM PbP_Logs WHERE game_id = ?) > 0 as has_pbp,
                    (SELECT COUNT(*) FROM GameStates WHERE game_id = ? AND is_final_state = 1) > 0 as has_final_state
                """,
                (game_id, game_id),
            )

            row = cursor.fetchone()
            has_pbp, has_final_state = row

            if has_pbp and has_final_state:
                cursor.execute(
                    "UPDATE Games SET game_data_finalized = 1 WHERE game_id = ?",
                    (game_id,),
                )
                finalized.append(game_id)

        conn.commit()

    return finalized


def _mark_boxscore_games_finalized(game_ids, db_path=DB_PATH):
    """
    Marks games as having finalized boxscore data based on minutes played and game status:
    - PlayerBox has sufficient data (>=16 players total)
    - TeamBox has both teams (exactly 2 records)
    - Total team minutes >= 240 per team (indicates complete game)
    - Game status = 3 (Final) in Games table

    Uses minutes-based finalization since NBA Stats API doesn't include gameStatus.
    Works for both Live and Stats endpoint data collection.

    Parameters:
        game_ids (list): List of game IDs to check.
        db_path (str): Path to database.

    Returns:
        list: Game IDs that were marked as finalized.
    """
    finalized = []

    with get_db(db_path) as conn:
        cursor = conn.cursor()

        for game_id in game_ids:
            # Check basic data requirements and game status
            cursor.execute(
                """
                SELECT 
                    (SELECT COUNT(*) FROM PlayerBox WHERE game_id = ?) as player_count,
                    (SELECT COUNT(*) FROM TeamBox WHERE game_id = ?) as team_count,
                    (SELECT status FROM Games WHERE game_id = ?) as game_status
                """,
                (game_id, game_id, game_id),
            )

            row = cursor.fetchone()
            if not row:
                continue

            player_count, team_count, game_status = row

            # Basic requirements: players exist, both teams, game is final
            if player_count < 16 or team_count != 2 or game_status != 3:
                continue

            # Check if both teams have sufficient minutes (239+ indicates complete game)
            # Note: Using 239 instead of 240 to account for floating-point precision issues
            cursor.execute(
                """
                SELECT team_id, SUM(COALESCE(min, 0)) as total_minutes
                FROM PlayerBox 
                WHERE game_id = ? AND min IS NOT NULL
                GROUP BY team_id
                """,
                (game_id,),
            )

            team_minutes = cursor.fetchall()

            # Must have exactly 2 teams with 239+ minutes each
            if len(team_minutes) != 2:
                continue

            both_teams_complete = all(minutes >= 239 for _, minutes in team_minutes)

            if both_teams_complete:
                cursor.execute(
                    "UPDATE Games SET boxscore_data_finalized = 1 WHERE game_id = ?",
                    (game_id,),
                )
                finalized.append(game_id)

        conn.commit()

    return finalized


@log_execution_time()
def get_games_for_prediction_update(season, predictor, db_path=DB_PATH):
    """
    Retrieves game_ids for games needing updated predictions.

    Returns games that have:
    - pre_game_data_finalized = 1 (prior states collected)
    - Valid features (non-empty feature_set)
    - No existing predictions for this predictor
    - Not postponed (status_text != 'PPD')

    This excludes opening night games with no prior season data and postponed games.

    Parameters:
        season (str): The season to update (e.g., "2024-2025" or "Current").
        predictor (str): The predictor to check for existing predictions.
        db_path (str): The path to the database (default is from config).

    Returns:
        list: A list of game_ids that need updated predictions.
    """
    from src.utils import determine_current_season

    # Convert "Current" to actual season
    if season == "Current":
        season = determine_current_season()

    # Phase5/Phase3 predictors are expensive (~2s/game). Only predict upcoming
    # games (status=1) on-demand. Historical backtesting should be done
    # deliberately via the orchestrator, not triggered by web app page loads.
    expensive_predictors = {"Phase5", "Phase3"}
    status_filter = "AND g.status = 1" if predictor in expensive_predictors else ""

    query = f"""
        SELECT g.game_id
        FROM Games g
        JOIN Features f ON g.game_id = f.game_id
        LEFT JOIN Predictions p ON g.game_id = p.game_id AND p.predictor = ?
        WHERE g.season = ?
            AND g.season_type IN ("Regular Season", "Post Season")
            AND g.pre_game_data_finalized = 1
            AND g.status_text != 'PPD'
            AND LENGTH(f.feature_set) > 10
            AND p.game_id IS NULL
            {status_filter}
        """

    with get_db(db_path) as conn:
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

    args = parser.parse_args()
    log_level = args.log_level.upper()
    setup_logging(log_level=log_level)

    update_database(
        args.season,
        args.predictor,
    )


if __name__ == "__main__":
    main()
