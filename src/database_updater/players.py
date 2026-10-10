"""
players.py

Fetches basic player metadata from NBA API to update Players table.
Tracks: person_id, names, from_year, to_year, roster_status, team

This is called as part of the database update pipeline to keep player
data current for linking with PlayerBox, InjuryReports, etc.

Functions:
- update_players(db_path=DB_PATH): Orchestrates the process of updating the players data.
- fetch_players(): Fetches the players data from the NBA API and processes it.
- save_players(players_data, db_path=DB_PATH): Saves the fetched players data to the database.
- main(): Main function to handle command-line arguments and update the players data.

Usage:
- Typically, run as part of the database update process.
- Can be run independently (from project root) to update the players data in the database using the command:
    python -m src.database_updater.players --log_level=INFO
- Successful execution will log the number of players fetched and saved.
"""

import argparse
import logging
import sqlite3

import requests

from src.config import config
from src.database import get_db
from src.database_updater.validators import PlayerValidator
from src.logging_config import setup_logging
from src.utils import (
    NBATeamConverter,
    StageLogger,
    determine_current_season,
    log_execution_time,
    requests_retry_session,
)

# Configuration
DB_PATH = config["database"]["path"]
NBA_API_PLAYERS_ENDPOINT = config["nba_api"]["players_endpoint"]
NBA_API_STATS_HEADERS = config["nba_api"]["pbp_stats_headers"]

# Cache configuration
PLAYERS_CACHE_CURRENT_MINUTES = 60  # Cache duration for current season (1 hour)
PLAYERS_CACHE_HISTORICAL_DAYS = 365  # Cache duration for historical seasons (1 year)


def _ensure_players_cache_table(db_path):
    """Create PlayersCache table if it doesn't exist."""
    with get_db(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS PlayersCache (
                season TEXT PRIMARY KEY,
                last_update_datetime TEXT NOT NULL
            )
        """
        )
        conn.commit()


def _get_last_players_update(db_path):
    """
    Get the last update datetime from the cache.

    Returns:
        datetime or None: The last update datetime, or None if not cached.
    """
    try:
        with get_db(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT last_update_datetime FROM PlayersCache ORDER BY last_update_datetime DESC LIMIT 1"
            )
            result = cursor.fetchone()
            if result:
                from datetime import datetime

                return datetime.fromisoformat(result[0])
            return None
    except sqlite3.OperationalError:
        # Table doesn't exist yet
        return None


def _should_update_players(db_path):
    """
    Determine if players should be updated based on cache.

    Cache strategy:
    - Current season: 60-minute cache (rosters change less frequently than schedule)
    - Historical seasons: 365-day cache (rosters essentially immutable)

    Returns:
        bool: True if players should be updated, False otherwise.
    """
    current_season = determine_current_season()

    # Check cache
    last_update = _get_last_players_update(db_path)
    if last_update is None:
        logging.debug("No cache entry for players - updating")
        return True

    # Calculate time since last update (use UTC for consistency)
    from datetime import datetime, timedelta, timezone

    now_utc = datetime.now(timezone.utc)
    # Ensure last_update is timezone-aware (assume UTC if naive)
    if last_update.tzinfo is None:
        last_update = last_update.replace(tzinfo=timezone.utc)
    time_since_update = now_utc - last_update
    minutes_since_update = time_since_update.total_seconds() / 60

    # Check if cache expired (always use current season threshold since we fetch all players)
    cache_threshold_minutes = PLAYERS_CACHE_CURRENT_MINUTES

    if minutes_since_update > cache_threshold_minutes:
        logging.debug(
            f"Cache expired for players ({minutes_since_update:.1f} minutes old, "
            f"threshold: {cache_threshold_minutes:.0f} minutes)"
        )
        return True

    logging.debug(
        f"Using cached players (updated {minutes_since_update:.1f} minutes ago)"
    )
    return False


def _update_players_cache(db_path):
    """
    Update the players cache with current UTC timestamp.
    """
    from datetime import datetime, timezone

    _ensure_players_cache_table(db_path)

    current_season = determine_current_season()
    with get_db(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO PlayersCache (season, last_update_datetime)
            VALUES (?, ?)
        """,
            (current_season, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()


def _get_player_count(db_path):
    """Get total player count from database."""
    with get_db(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM Players")
        return cursor.fetchone()[0]


@log_execution_time()
def update_players(db_path=DB_PATH):
    """
    Orchestrates the process of updating the players data by fetching the latest data
    from the NBA API and saving it to the SQLite database.

    Only updates players with changes (from_year, to_year, roster_status, team)
    or new players not in the database.

    Returns:
        dict: {"added": int, "updated": int, "total": int}
    """
    stage_logger = StageLogger("Players")

    # Check cache before fetching
    if not _should_update_players(db_path):
        total_count = _get_player_count(db_path)
        stage_logger.set_counts(added=0, updated=0, total=total_count)
        stage_logger.log_complete()
        return {"added": 0, "updated": 0, "total": total_count}

    # Fetch from NBA API
    logging.debug("Fetching player data from NBA API...")
    players_data = fetch_players(stage_logger)

    if not players_data:
        logging.warning("No player data fetched")
        stage_logger.log_complete()
        return {"added": 0, "updated": 0, "total": 0}

    # Save to database and get counts
    counts = save_players(players_data, db_path, stage_logger)

    # Log completion
    stage_logger.set_counts(
        added=counts["added"], updated=counts["updated"], total=counts["total"]
    )
    stage_logger.log_complete()

    return counts


@log_execution_time(average_over="output")
def fetch_players(stage_logger: StageLogger):
    """
    Fetches the players data from the NBA API and processes it into a list.

    Args:
        stage_logger: StageLogger instance for tracking API calls

    Returns:
        list: List of player dictionaries with keys: person_id, first_name, last_name,
              full_name, from_year, to_year, roster_status, team
    """
    current_season = determine_current_season()
    # Convert "2025-2026" to "2025-26" for NBA API
    api_season = current_season[:5] + current_season[-2:]

    # Construct the request URL
    url = NBA_API_PLAYERS_ENDPOINT.format(season=api_season)

    # Fetch the data with retry logic
    try:
        session = requests_retry_session()
        response = session.get(url, headers=NBA_API_STATS_HEADERS, timeout=30)
        response.raise_for_status()
        stage_logger.log_api_call()  # Track API call
    except Exception as e:
        logging.warning(f"Players API request failed: {e}")
        return []

    try:
        data = response.json()
    except Exception as e:
        logging.warning(f"Players API returned invalid JSON: {e}")
        return []

    # Extract the player data from the response
    headers = data.get("resultSets", [{}])[0].get("headers", [])
    rows = data.get("resultSets", [{}])[0].get("rowSet", [])

    # Build a mapping from header names to indices
    header_indices = {header: idx for idx, header in enumerate(headers)}

    # Define the required fields
    required_fields = [
        "PERSON_ID",
        "DISPLAY_LAST_COMMA_FIRST",
        "FROM_YEAR",
        "TO_YEAR",
        "ROSTERSTATUS",
        "TEAM_ABBREVIATION",
    ]

    # Check if all required fields are present
    missing_fields = [field for field in required_fields if field not in header_indices]
    if missing_fields:
        logging.error(f"Missing required fields in API response: {missing_fields}")
        return []

    # Process each row into a player dictionary
    players_list = []
    for row in rows:
        try:
            team_abbr = row[header_indices["TEAM_ABBREVIATION"]]
            # Convert NBA team abbreviations to our standard format
            team = NBATeamConverter.get_abbreviation(team_abbr) if team_abbr else None

            # Parse name from "Last, First" format
            full_name = row[header_indices["DISPLAY_LAST_COMMA_FIRST"]]
            name_parts = full_name.split(", ")
            if len(name_parts) == 2:
                last_name, first_name = name_parts
            else:
                # Fallback for unusual name formats
                name_parts = full_name.split()
                if len(name_parts) > 1:
                    last_name, first_name = name_parts[-1], " ".join(name_parts[:-1])
                else:
                    last_name, first_name = name_parts[0], ""

            player = {
                "person_id": row[header_indices["PERSON_ID"]],
                "first_name": first_name,
                "last_name": last_name,
                "full_name": full_name,
                "from_year": (
                    int(row[header_indices["FROM_YEAR"]])
                    if row[header_indices["FROM_YEAR"]]
                    else None
                ),
                "to_year": (
                    int(row[header_indices["TO_YEAR"]])
                    if row[header_indices["TO_YEAR"]]
                    else None
                ),
                "roster_status": row[header_indices["ROSTERSTATUS"]],
                "team": team,
            }
            players_list.append(player)
        except (KeyError, IndexError) as e:
            logging.error(f"Error processing player row: {e}")
            continue

    logging.debug(f"Fetched {len(players_list)} players from NBA API")

    # Sanity check: NBA has ~500+ active players
    if len(players_list) < 100:
        logging.warning(
            f"NBA API returned only {len(players_list)} players (expected 500+) "
            f"- may indicate API issue"
        )

    return players_list


@log_execution_time(average_over="players_data")
def save_players(players_data, db_path=DB_PATH, stage_logger=None):
    """
    Saves player data to database with selective updates and validation.

    Only updates players that are new or have changed fields.

    Args:
        players_data: List of player dictionaries
        db_path: Path to database
        stage_logger: Optional StageLogger for tracking counts

    Returns:
        dict: {"added": int, "updated": int, "total": int}
    """
    if not players_data:
        return {"added": 0, "updated": 0, "total": 0}

    with get_db(db_path) as conn:
        cursor = conn.cursor()

        # Get existing players to compare
        person_ids = [p["person_id"] for p in players_data]
        placeholders = ",".join("?" * len(person_ids))
        cursor.execute(
            f"""
            SELECT person_id, first_name, last_name, full_name, from_year, to_year, roster_status, team
            FROM Players WHERE person_id IN ({placeholders})
        """,
            person_ids,
        )

        # Build lookup of existing players
        existing_players = {
            row[0]: {
                "first_name": row[1],
                "last_name": row[2],
                "full_name": row[3],
                "from_year": row[4],
                "to_year": row[5],
                "roster_status": row[6],
                "team": row[7],
            }
            for row in cursor.fetchall()
        }

        # Filter to only changed or new players
        players_to_update = []
        for player in players_data:
            person_id = player["person_id"]
            if person_id not in existing_players:
                # New player
                players_to_update.append(player)
            else:
                # Check if any fields changed
                existing = existing_players[person_id]
                if (
                    player["first_name"] != existing["first_name"]
                    or player["last_name"] != existing["last_name"]
                    or player["full_name"] != existing["full_name"]
                    or player["from_year"] != existing["from_year"]
                    or player["to_year"] != existing["to_year"]
                    or player["roster_status"] != existing["roster_status"]
                    or player["team"] != existing["team"]
                ):
                    players_to_update.append(player)

        # Count changes
        added_count = sum(
            1 for p in players_to_update if p["person_id"] not in existing_players
        )
        updated_count = len(players_to_update) - added_count

        # Only execute UPDATE if there are changes
        if players_to_update:
            # Prepare data for bulk insert
            players_tuples = [
                (
                    player["person_id"],
                    player["first_name"],
                    player["last_name"],
                    player["full_name"],
                    player["from_year"],
                    player["to_year"],
                    player["roster_status"],
                    player["team"],
                )
                for player in players_to_update
            ]

            # Insert or replace records
            cursor.executemany(
                """
                INSERT OR REPLACE INTO Players (
                    person_id, first_name, last_name, full_name, from_year, to_year, 
                    roster_status, team
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                players_tuples,
            )

            logging.debug(
                f"Updated {len(players_to_update)} players (out of {len(players_data)} fetched)"
            )
        else:
            logging.debug(
                f"No player changes detected (checked {len(players_data)} players)"
            )

        # Players absent from the league-wide list (inserted by other paths) are
        # never updated above; clear their roster flag so they aren't projected.
        if len(players_data) >= 1000:
            cursor.execute(
                f"""
                UPDATE Players SET roster_status = 0
                WHERE roster_status = 1 AND person_id NOT IN ({placeholders})
            """,
                person_ids,
            )
            updated_count += cursor.rowcount

        conn.commit()

        # Validate saved data
        validator = PlayerValidator()
        validation_result = validator.validate(person_ids, cursor)

        # Set validation suffix in stage logger
        if stage_logger:
            stage_logger.set_validation(validation_result)

        # Log validation issues
        if validation_result.has_critical_issues:
            logging.error(f"Critical validation issues: {validation_result.summary()}")
        elif validation_result.has_warnings:
            logging.warning(f"Validation warnings: {validation_result.summary()}")

        # Get total count from DB
        cursor.execute("SELECT COUNT(*) FROM Players")
        total_count = cursor.fetchone()[0]

        logging.debug(
            f"Saved players: +{added_count} ~{updated_count} (total: {total_count})"
        )

    # Update cache after successful save
    _update_players_cache(db_path)

    return {"added": added_count, "updated": updated_count, "total": total_count}


def main():
    """
    Main function to handle command-line arguments and orchestrate updating the players data.

    This function sets up logging based on the provided log level and then calls the
    `update_players` function to fetch and save the latest players data.
    """
    parser = argparse.ArgumentParser(description="Update players data.")
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        help="The logging level. Default is INFO. DEBUG provides more details.",
    )

    args = parser.parse_args()
    log_level = args.log_level.upper()
    setup_logging(log_level=log_level)

    update_players()


if __name__ == "__main__":
    main()
