"""
schedule.py

Overview:
This module fetches and saves NBA schedule data for a given season. It consists of functions to:
- Fetch the schedule from the NBA API.
- Validate and save the schedule to a SQLite database.
- Ensure data integrity by checking for empty or corrupted data before updating the database.

Functions:
- update_schedule(season, db_path): Orchestrates fetching and saving the schedule.
- fetch_schedule(season): Fetches the NBA schedule for a specified season.
- save_schedule(games, season, db_path): Saves the fetched schedule to the database.
- main(): Handles command-line arguments to update the schedule, with optional logging level.

Usage:
- Typically run as part of a larger data collection pipeline.
- Script can be run directly from the command line (project root) to fetch and save NBA schedule data:
    python -m src.database_updater.schedule --season=2025-2026 --log_level=DEBUG
- Successful execution will print the number of games fetched and saved along with logging information.
"""

import argparse
import logging
import sqlite3

import pandas as pd
import requests

from src.config import config
from src.database import get_db
from src.database_updater.validators import ScheduleValidator
from src.logging_config import setup_logging
from src.utils import (
    StageLogger,
    determine_current_season,
    log_execution_time,
    requests_retry_session,
    validate_season_format,
)

# Configuration values
DB_PATH = config["database"]["path"]
NBA_API_BASE_URL = config["nba_api"]["schedule_endpoint"]
NBA_API_HEADERS = config["nba_api"]["schedule_headers"]
SCHEDULE_CACHE_CURRENT_MINUTES = (
    5  # Cache duration for current season (games change frequently)
)


def _get_schedule_cache_info(season, db_path):
    """
    Get the cache info for a season (last update time and finalized status).

    Parameters:
    season (str): The season to check.
    db_path (str): The path to the SQLite database file.

    Returns:
    tuple: (last_update_datetime, schedule_finalized) or (None, False) if not cached.
    """
    try:
        with get_db(db_path) as conn:
            cursor = conn.cursor()
            # Check if schedule_finalized column exists
            cursor.execute("PRAGMA table_info(ScheduleCache)")
            columns = [row[1] for row in cursor.fetchall()]

            if "schedule_finalized" in columns:
                cursor.execute(
                    "SELECT last_update_datetime, schedule_finalized FROM ScheduleCache WHERE season = ?",
                    (season,),
                )
            else:
                cursor.execute(
                    "SELECT last_update_datetime FROM ScheduleCache WHERE season = ?",
                    (season,),
                )
            result = cursor.fetchone()
            if result:
                last_update = pd.to_datetime(result[0], utc=True)
                finalized = result[1] if len(result) > 1 else False
                return last_update, bool(finalized)
            return None, False
    except sqlite3.OperationalError:
        # Table doesn't exist yet - will be created by migration
        return None, False


def _should_update_schedule(season, db_path):
    """
    Determine if schedule should be updated based on cache and season status.

    Cache strategy:
    - Current season: 5-minute cache (games change frequently)
    - Historical seasons: Once finalized (all games Final), never refetch

    Parameters:
    season (str): The season to check.
    db_path (str): The path to the SQLite database file.

    Returns:
    bool: True if schedule should be updated, False otherwise.
    """
    current_season = determine_current_season()
    is_current = season == current_season

    # Check cache
    last_update, is_finalized = _get_schedule_cache_info(season, db_path)

    if last_update is None:
        logging.debug(f"No cache entry for season {season} - updating")
        return True

    # Historical seasons: If finalized, never refetch
    if not is_current and is_finalized:
        logging.debug(f"Season {season} schedule is finalized - skipping update")
        return False

    # Current season: 5-minute cache
    minutes_since_update = (
        pd.Timestamp.now(tz="UTC") - last_update
    ).total_seconds() / 60

    if is_current:
        cache_threshold_minutes = SCHEDULE_CACHE_CURRENT_MINUTES
        if minutes_since_update > cache_threshold_minutes:
            logging.debug(
                f"Cache expired for season {season} ({minutes_since_update:.1f} minutes old, "
                f"threshold: {cache_threshold_minutes:.0f} minutes)"
            )
            return True
    else:
        # Historical season not yet finalized - check if all games are Final
        # We'll do one update to potentially finalize it
        logging.debug(
            f"Historical season {season} not finalized - checking for updates"
        )
        return True

    logging.debug(
        f"Using cached schedule for season {season} (updated {minutes_since_update:.1f} minutes ago)"
    )
    return False


def _update_schedule_cache(season, db_path, check_finalized=True):
    """
    Update the schedule cache with current timestamp and check if season should be finalized.

    A season is finalized when all games have status=3 (Final).

    Parameters:
    season (str): The season to update cache for.
    db_path (str): The path to the SQLite database file.
    check_finalized (bool): Whether to check if season should be marked finalized.
    """
    try:
        with get_db(db_path) as conn:
            cursor = conn.cursor()
            update_time = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M:%S")

            # Check if we should finalize this season
            should_finalize = False
            if check_finalized:
                current_season = determine_current_season()
                if season != current_season:
                    # Historical season - check if all games are Final
                    cursor.execute(
                        """
                        SELECT COUNT(*) FROM Games 
                        WHERE season = ? 
                        AND season_type IN ('Regular Season', 'Post Season')
                        AND status != 3
                        """,
                        (season,),
                    )
                    non_final_count = cursor.fetchone()[0]
                    should_finalize = non_final_count == 0
                    if should_finalize:
                        logging.info(
                            f"Season {season} has all games finalized - marking schedule as finalized"
                        )

            # Ensure schedule_finalized column exists
            cursor.execute("PRAGMA table_info(ScheduleCache)")
            columns = [row[1] for row in cursor.fetchall()]
            if "schedule_finalized" not in columns:
                cursor.execute(
                    "ALTER TABLE ScheduleCache ADD COLUMN schedule_finalized INTEGER DEFAULT 0"
                )

            cursor.execute(
                """
                INSERT INTO ScheduleCache (season, last_update_datetime, schedule_finalized)
                VALUES (?, ?, ?)
                ON CONFLICT(season) DO UPDATE SET 
                    last_update_datetime = excluded.last_update_datetime,
                    schedule_finalized = CASE 
                        WHEN excluded.schedule_finalized = 1 THEN 1 
                        ELSE schedule_finalized 
                    END
                """,
                (season, update_time, 1 if should_finalize else 0),
            )
            conn.commit()
            logging.debug(
                f"Updated schedule cache for season {season} (finalized={should_finalize})"
            )
    except sqlite3.OperationalError as e:
        # Table doesn't exist - this should only happen on fresh databases
        logging.warning(f"ScheduleCache table not found: {e}")


def _validate_schedule(season: str, db_path: str = DB_PATH):
    """
    Validate schedule data after update.

    Checks for NULL critical fields, TBD teams, invalid status values.
    Logs warnings for issues but doesn't block pipeline.
    """
    validator = ScheduleValidator()

    with get_db(db_path) as conn:
        cursor = conn.cursor()

        # Get all game_ids for this season
        cursor.execute("SELECT game_id FROM Games WHERE season = ?", (season,))
        game_ids = [row[0] for row in cursor.fetchall()]

        if game_ids:
            result = validator.validate(game_ids, cursor)

            if result.has_critical_issues or result.has_warnings:
                logging.warning(f"Schedule validation issues:\n{result.summary()}")
            else:
                logging.debug(f"Schedule validation: PASS ({len(game_ids)} games)")


def sync_live_game_status(db_path=DB_PATH):
    """
    Sync game status from the live scoreboard API for today's games.

    The Schedule API doesn't update game status in real-time. This function
    checks the live scoreboard and updates the status for any in-progress
    or recently completed games.

    Parameters:
    db_path (str): The path to the SQLite database file.

    Returns:
    int: Number of games updated.
    """
    try:
        from nba_api.live.nba.endpoints import scoreboard

        board = scoreboard.ScoreBoard()
        games_data = board.get_dict()
        live_games = games_data.get("scoreboard", {}).get("games", [])

        if not live_games:
            return 0

        updates = []
        for game in live_games:
            game_id = game.get("gameId")
            status = game.get("gameStatus")  # 1=Not Started, 2=In Progress, 3=Final
            status_text = game.get("gameStatusText", "")

            if game_id and status:
                updates.append((status, status_text, game_id))

        if not updates:
            return 0

        with get_db(db_path) as conn:
            cursor = conn.cursor()
            cursor.executemany(
                "UPDATE Games SET status = ?, status_text = ? WHERE game_id = ?",
                updates,
            )
            updated_count = cursor.rowcount
            conn.commit()

        if updated_count > 0:
            logging.debug(
                f"Synced live status for {updated_count} games from scoreboard"
            )

        return updated_count

    except Exception as e:
        logging.debug(f"Could not sync live scoreboard: {e}")
        return 0


@log_execution_time()
def update_schedule(season="Current", db_path=DB_PATH, force=False):
    """
    Fetches and updates the NBA schedule for a given season in the database.
    Uses caching to avoid redundant NBA API calls for historical seasons.

    Parameters:
    season (str): The season to fetch and update the schedule for. Defaults to "Current".
    db_path (str): The path to the SQLite database file. Defaults to the configured database path.
    force (bool): If True, bypass cache and force update. Defaults to False.
    """
    if season == "Current":
        season = determine_current_season()
    else:
        validate_season_format(season, abbreviated=False)

    stage_logger = StageLogger("Schedule")

    # Always sync live game status for current season (fast, no caching issues)
    current_season = determine_current_season()
    if season == current_season:
        sync_live_game_status(db_path)

    # Check if update is needed (unless forced)
    if not force and not _should_update_schedule(season, db_path):
        last_update, is_finalized = _get_schedule_cache_info(season, db_path)
        if last_update:
            minutes_ago = (
                pd.Timestamp.now(tz="UTC") - last_update
            ).total_seconds() / 60
            stage_logger.log_cache_hit(season, minutes_ago)
        return

    games = fetch_schedule(season, stage_logger)
    if save_schedule(games, season, db_path, stage_logger):
        # Update cache on successful save
        _update_schedule_cache(season, db_path)

        # Validate schedule data
        _validate_schedule(season, db_path)


@log_execution_time()
def fetch_schedule(season, stage_logger=None):
    """
    Fetches the NBA schedule for a given season.

    Parameters:
    season (str): The season to fetch the schedule for, formatted as 'XXXX-XXXX' (e.g., '2020-2021').
    stage_logger (StageLogger): Optional logger to track API calls.

    Returns:
    list: A list of dictionaries, each containing details of a game. If the request fails or the data is corrupted, an empty list is returned.
    """
    api_season = season[:5] + season[-2:]
    endpoint = NBA_API_BASE_URL.format(season=api_season)

    try:
        session = requests_retry_session()
        response = session.get(endpoint, headers=NBA_API_HEADERS)
        response.raise_for_status()

        # Track API call
        if stage_logger:
            stage_logger.log_api_call()

    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching schedule for {season}: {e}")
        return []

    try:
        game_dates = response.json()["leagueSchedule"]["gameDates"]

        if not game_dates:
            logging.error(f"No games found for season {season}")
            return []

        all_games = [game for date in game_dates for game in date["games"]]

        keys_needed = [
            "gameId",
            "gameStatus",
            "gameStatusText",  # Human-readable status from NBA API
            "gameDateTimeUTC",  # UTC time - store this instead of EST
            "homeTeam",
            "awayTeam",
        ]

        all_games = [
            {key: game.get(key, "") for key in keys_needed} for game in all_games
        ]

        for game in all_games:
            game["homeTeam"] = game["homeTeam"]["teamTricode"]
            game["awayTeam"] = game["awayTeam"]["teamTricode"]

        season_type_codes = {
            "001": "Pre Season",
            "002": "Regular Season",
            "003": "All-Star",
            "004": "Post Season",
            "005": "Post Season",  # Play-In
            "006": "NBA Cup Final",  # championship game; not part of regular-season stats
        }

        for game in all_games:
            game["seasonType"] = season_type_codes.get(game["gameId"][:3], "Unknown")
            # Keep numeric gameStatus (1, 2, 3) as-is - no mapping needed
            # gameStatusText is human-readable ("Final", "5:00 pm ET", etc.)
            game["season"] = season

        return all_games

    except (KeyError, TypeError) as e:
        logging.error(f"Error processing schedule data for {season}: {e}")
        return []


@log_execution_time()
def save_schedule(games, season, db_path=DB_PATH, stage_logger=None):
    """
    Saves the NBA schedule to the database. This function first checks the validity of the data,
    then updates the database by adding new records, updating existing ones, and removing obsolete records.

    Parameters:
    games (list): A list of game dictionaries to be saved.
    season (str): The season to which the games belong.
    db_path (str): The path to the SQLite database file.
    stage_logger (StageLogger): Optional logger for unified output.

    Returns:
    bool: True if the operation was successful, False otherwise.
    """
    if not games:
        logging.error("No games fetched. Skipping database update.")
        return False

    game_ids = [game["gameId"] for game in games]

    with get_db(db_path) as conn:
        cursor = conn.cursor()

        # Check for data corruption or unexpected issues in the new data
        if any("gameId" not in game or "gameDateTimeUTC" not in game for game in games):
            logging.error(
                "Fetched schedule data is corrupted. Skipping database update to avoid data loss."
            )
            return False

        # Check if all games belong to the same season
        if any(game["season"] != season for game in games):
            logging.error(
                "Inconsistent season data. All games must belong to the same season."
            )
            return False

        # Start transaction
        cursor.execute("BEGIN TRANSACTION")

        try:
            # Count existing games and games to be deleted
            cursor.execute(
                f"SELECT COUNT(*) FROM Games WHERE season = ?",
                [season],
            )
            existing_count = cursor.fetchone()[0]

            cursor.execute(
                f"SELECT COUNT(*) FROM Games WHERE season = ? AND game_id NOT IN ({','.join('?' * len(game_ids))})",
                [season] + game_ids,
            )
            removed_count = cursor.fetchone()[0]

            # Safety check: Prevent accidental mass deletion from incomplete API data
            # If we would delete more than 50% of existing games, abort the update
            if existing_count > 0 and removed_count > existing_count * 0.5:
                logging.error(
                    f"Safety check failed for {season}: Would delete {removed_count}/{existing_count} games "
                    f"({removed_count/existing_count*100:.1f}%). This likely indicates incomplete API data. "
                    f"Fetched only {len(game_ids)} games. Aborting schedule update to prevent data loss."
                )
                cursor.execute("ROLLBACK")
                return False

            # Delete records not in the new data (e.g., postponed/cancelled games)
            if removed_count > 0:
                cursor.execute(
                    f"DELETE FROM Games WHERE season = ? AND game_id NOT IN ({','.join('?' * len(game_ids))})",
                    [season] + game_ids,
                )
                logging.info(f"Removed {removed_count} obsolete games for {season}")

            added_count = 0
            updated_count = 0

            # Check existing records
            existing_games = {}
            cursor.execute(
                f"SELECT game_id, date_time_utc, home_team, away_team, status, status_text, season, season_type FROM Games WHERE game_id IN ({','.join('?' * len(game_ids))})",
                game_ids,
            )
            for row in cursor.fetchall():
                existing_games[row[0]] = row

            # Insert or replace new and updated game records
            # Note: status uses MAX() to never decrease - this preserves live status updates
            # (e.g., if sync_live_game_status set status=2, don't let stale Schedule API reset to 1)
            insert_sql = """
            INSERT INTO Games (game_id, date_time_utc, home_team, away_team, status, status_text, season, season_type, 
                pre_game_data_finalized, game_data_finalized, boxscore_data_finalized)
            VALUES (:game_id, :date_time_utc, :home_team, :away_team, :status, :status_text, :season, :season_type,
                COALESCE((SELECT pre_game_data_finalized FROM Games WHERE game_id = :game_id), 0),
                COALESCE((SELECT game_data_finalized FROM Games WHERE game_id = :game_id), 0),
                COALESCE((SELECT boxscore_data_finalized FROM Games WHERE game_id = :game_id), 0))
            ON CONFLICT(game_id) DO UPDATE SET
                date_time_utc=excluded.date_time_utc,
                home_team=excluded.home_team,
                away_team=excluded.away_team,
                status=MAX(Games.status, excluded.status),
                status_text=excluded.status_text,
                season=excluded.season,
                season_type=excluded.season_type
            """

            for game in games:
                game_id = game["gameId"]
                if game_id not in existing_games:
                    added_count += 1
                else:
                    existing_game = existing_games[game_id]
                    if (
                        game["gameDateTimeUTC"] != existing_game[1]
                        or game["homeTeam"] != existing_game[2]
                        or game["awayTeam"] != existing_game[3]
                        or game["gameStatus"] != existing_game[4]
                        or game.get("gameStatusText", "") != existing_game[5]
                        or game["season"] != existing_game[6]
                        or game["seasonType"] != existing_game[7]
                    ):
                        updated_count += 1

                params = {
                    "game_id": game["gameId"],
                    "date_time_utc": game["gameDateTimeUTC"],  # Actual UTC from NBA API
                    "home_team": game["homeTeam"],
                    "away_team": game["awayTeam"],
                    "status": game["gameStatus"],  # Numeric: 1, 2, or 3
                    "status_text": game.get(
                        "gameStatusText", ""
                    ),  # Text: "Final", "5:00 pm ET", etc.
                    "season": game["season"],
                    "season_type": game["seasonType"],
                }

                # Skip games with None values (e.g., Cup/All-Star games with TBD teams)
                if any(v is None for v in params.values()):
                    logging.debug(f"Skipping game {game_id} with TBD teams: {params}")
                    continue

                cursor.execute(insert_sql, params)

            # Commit transaction
            conn.commit()

            # Validate saved data
            validator = ScheduleValidator()
            validation_result = validator.validate(game_ids, cursor)

            # Use StageLogger for unified one-line output
            if stage_logger:
                stage_logger.set_counts(
                    added=added_count,
                    updated=updated_count,
                    removed=removed_count,
                    total=len(games),
                )
                stage_logger.set_validation(validation_result)
                stage_logger.log_complete(season)

            # Log detailed validation warnings separately if issues found
            if validation_result.has_critical_issues or validation_result.has_warnings:
                for issue in validation_result.issues:
                    if issue.severity.value == "CRITICAL":
                        logging.error(f"  {issue}")
                    elif issue.severity.value == "WARNING":
                        logging.warning(f"  {issue}")

            return True
        except Exception as e:
            # Rollback transaction on error
            conn.rollback()
            logging.error(f"Error saving schedule: {e}")
            raise e


def main():
    """
    Main function to handle command-line arguments and orchestrate updating the schedule.
    """
    parser = argparse.ArgumentParser(description="Update NBA schedule data.")
    parser.add_argument(
        "--season",
        type=str,
        default="Current",
        help="The season to fetch the schedule for. Format: 'XXXX-XXXX'. Default is the current season.",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        help="The logging level. Default is INFO. DEBUG provides more details.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force schedule update, bypassing cache.",
    )

    args = parser.parse_args()
    log_level = args.log_level.upper()
    setup_logging(log_level=log_level)

    update_schedule(season=args.season, force=args.force)


if __name__ == "__main__":
    main()
