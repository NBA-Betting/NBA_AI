"""
games.py

This module manages loading and updating NBA game data from the NBA API into a SQLite database.

Key Features:
- Initial data load from the start of the 2000-2001 season through the end of the current NBA season.
- Loading data for a specific date.
- Updating incomplete or recently changed games.

Progress and Logging:
- Uses tqdm progress bars for season and date-level tracking during the initial load.
- Logs detailed issue counts (no games, errors) every 30 dates and at the end of each season.
- Sleeps a fixed 0.05 seconds between processing each date to respect rate limits.

Command Line Interface (CLI):
    python -m src.database_updater_CM.games --task=initial_load --save_to_db
    python -m src.database_updater_CM.games --task=load_date --game_date=MM/DD/YYYY --save_to_db
    python -m src.database_updater_CM.games --task=update --save_to_db

Arguments:
- --task: The operation to perform (initial_load, load_date, update).
- --game_date: Required if task=load_date. In MM/DD/YYYY format.
- --save_to_db: If passed, saves data to the database.

Example:
    python -m src.database_updater_CM.games --task=initial_load --save_to_db
"""

import argparse
import logging
import sqlite3
import time
from datetime import date, datetime, timedelta

from nba_api.stats.endpoints import ScoreboardV2
from tqdm import tqdm

from src.config import config
from src.logging_config_CM import setup_logging

# Configuration
DB_PATH = config["database"]["path"]

# Set up logging using the centralized configuration
setup_logging(
    log_level="INFO",
    log_file="custom_model.log",
    structured=True,
)

logger = logging.getLogger("games")

season_type_codes = {
    "001": "Pre Season",
    "002": "Regular Season",
    "003": "All-Star",
    "004": "Post Season",
    "005": "Play In",
}

status_map = {
    1: "Not Started",
    2: "In Progress",
    3: "Completed",
}

# Fixed settings
SLEEP_DURATION = 1  # seconds between API calls
BATCH_SIZE = (
    30  # Number of dates after which to log issue summaries during initial load
)


def transform_season(season_str):
    """
    Transform a season year (e.g. "2024") into "2024-2025".

    Args:
        season_str (str): A string representing the starting year of the season.

    Returns:
        str: The season in the format "YYYY-YYYY+1".
    """
    if not season_str.isdigit():
        return season_str
    start_year = int(season_str)
    end_year = start_year + 1
    return f"{season_str}-{end_year}"


def date_to_season(date_str):
    """
    Converts a date to the NBA season in YYYY-YYYY format, considering special cases.

    Special cases:
        - 2011-2012: 2011-07-01 to 2012-06-30
        - 2019-2020: 2019-07-01 to 2020-10-11
        - 2020-2021: 2020-10-12 to 2021-07-20

    Args:
        date_str (str): The date in "YYYY-MM-DD" format.

    Returns:
        str: The season in YYYY-YYYY format.
    """
    date_obj = datetime.strptime(date_str, "%Y-%m-%d")

    special_cases = [
        ("2011-2012", datetime(2011, 7, 1), datetime(2012, 6, 30)),
        ("2019-2020", datetime(2019, 7, 1), datetime(2020, 10, 11)),
        ("2020-2021", datetime(2020, 10, 12), datetime(2021, 7, 20)),
    ]

    for season, start, end in special_cases:
        if start <= date_obj <= end:
            return season

    year = date_obj.year
    # If after June 30 => next season
    if date_obj.month > 6 or (date_obj.month == 6 and date_obj.day > 30):
        return f"{year}-{year+1}"
    else:
        return f"{year-1}-{year}"


def get_season_end_date(season_str):
    """
    Given a season string (e.g. "2024-2025"), return the end date of that season.

    Special cases:
    - "2011-2012": ends 2012-06-30
    - "2019-2020": ends 2020-10-11
    - "2020-2021": ends 2021-07-20

    Args:
        season_str (str): Season in "YYYY-YYYY" format.

    Returns:
        date: The end date of the season.
    """
    special_cases_end = {
        "2011-2012": datetime(2012, 6, 30),
        "2019-2020": datetime(2020, 10, 11),
        "2020-2021": datetime(2021, 7, 20),
    }

    if season_str in special_cases_end:
        return special_cases_end[season_str].date()

    # Normal case: season ends June 30 of second year
    start_year = int(season_str.split("-")[0])
    end_year = start_year + 1
    return date(end_year, 6, 30)


def get_current_season():
    """
    Determine the current season based on today's date.

    Returns:
        str: Current season in "YYYY-YYYY" format.
    """
    today_str = date.today().strftime("%Y-%m-%d")
    return date_to_season(today_str)


def get_initial_load_date_range():
    """
    Get the start and end dates for the initial load.

    Start date: 2000-10-01 (start of the 2000-2001 season)
    End date: end of current season determined by get_current_season() and get_season_end_date().

    Returns:
        (date, date): start_date, end_date
    """
    start_date = date(2000, 10, 1)
    current_season = get_current_season()
    end_date = get_season_end_date(current_season)
    return start_date, end_date


def load_games_for_date(conn, game_date, save_to_db=True):
    """
    Load NBA games data for a specific date into the database.

    Args:
        conn (sqlite3.Connection): The database connection.
        game_date (str): Date in MM/DD/YYYY format to fetch games for.
        save_to_db (bool): Whether to save fetched data to the database.

    Returns:
        int: Number of records inserted/updated.
    """
    logger.debug(f"Fetching games for date: {game_date}")
    try:
        scoreboard = ScoreboardV2(
            game_date=game_date, league_id="00", day_offset=0
        ).get_dict()
    except Exception as e:
        logger.error(f"Error fetching games for date {game_date}: {e}")
        time.sleep(2)
        try:
            scoreboard = ScoreboardV2(
                game_date=game_date, league_id="00", day_offset=0
            ).get_dict()
        except Exception as e:
            logger.error(f"Retry failed for fetching games for date {game_date}: {e}")
            raise

    if not scoreboard["resultSets"]:
        logger.debug("No result sets returned from API.")
        return 0

    game_data = scoreboard["resultSets"][0]["rowSet"]
    headers = scoreboard["resultSets"][0]["headers"]
    header_index = {h: i for i, h in enumerate(headers)}

    if not game_data:
        logger.debug(f"No games found for {game_date}.")
        return 0

    upsert_sql = """
    INSERT INTO Games (
        game_id, gamecode, date_time_est, home_team_id, away_team_id,
        status, season, season_type, boxscores_finalized, game_states_finalized
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(game_id) DO UPDATE SET
        gamecode=excluded.gamecode,
        date_time_est=excluded.date_time_est,
        home_team_id=excluded.home_team_id,
        away_team_id=excluded.away_team_id,
        status=excluded.status,
        season=excluded.season,
        season_type=excluded.season_type,
        boxscores_finalized=excluded.boxscores_finalized,
        game_states_finalized=excluded.game_states_finalized
    """

    records_to_upsert = []
    for g in game_data:
        game_id = g[header_index["GAME_ID"]]
        gamecode = g[header_index["GAMECODE"]]
        date_time_est = g[header_index["GAME_DATE_EST"]]
        home_team_id = g[header_index["HOME_TEAM_ID"]]
        away_team_id = g[header_index["VISITOR_TEAM_ID"]]
        game_status_id = g[header_index["GAME_STATUS_ID"]]
        season_raw = g[header_index["SEASON"]]
        season = transform_season(season_raw)
        prefix = game_id[:3]
        season_type = season_type_codes.get(prefix, "Unknown")
        status = status_map.get(game_status_id, "Unknown")

        record = (
            game_id,
            gamecode,
            date_time_est,
            home_team_id,
            away_team_id,
            status,
            season,
            season_type,
            0,
            0,
        )
        records_to_upsert.append(record)

    if save_to_db and records_to_upsert:
        try:
            conn.executemany(upsert_sql, records_to_upsert)
            conn.commit()
            inserted_count = len(records_to_upsert)
            logger.debug(f"Upserted {inserted_count} game records for {game_date}.")
            return inserted_count
        except Exception as e:
            logger.error(
                f"Error upserting game records into database for {game_date}: {e}"
            )
            time.sleep(2)
            try:
                conn.executemany(upsert_sql, records_to_upsert)
                conn.commit()
                inserted_count = len(records_to_upsert)
                logger.debug(
                    f"Retry succeeded: Upserted {inserted_count} game records for {game_date}."
                )
                return inserted_count
            except Exception as e:
                logger.error(
                    f"Retry failed for upserting game records into database for {game_date}: {e}"
                )
                raise

    return len(records_to_upsert)


def load_initial_games(conn, save_to_db=True):
    """
    Load all NBA games starting from the full 2000-2001 season until the end of the current season.

    Progress:
        - Seasons are identified dynamically.
        - Each season uses its own tqdm progress bar.
        - Logs intermediate stats every 30 dates.
        - Tracks issues (no games, errors) for each season and logs at season end.

    Args:
        conn (sqlite3.Connection): The database connection.
        save_to_db (bool): Whether to commit data to the database.

    Returns:
        None
    """
    start_date, end_date = get_initial_load_date_range()

    # Determine all seasons in this range
    seasons = {}
    current_date = start_date
    while current_date <= end_date:
        s = date_to_season(current_date.strftime("%Y-%m-%d"))
        if s not in seasons:
            seasons[s] = []
        seasons[s].append(current_date)
        current_date += timedelta(days=1)

    all_season_keys = sorted(seasons.keys(), reverse=True)  # Process newest to oldest
    total_seasons = len(all_season_keys)

    overall_games_inserted = 0

    for idx, season_str in enumerate(all_season_keys, start=1):
        logger.info(f"Processing season {season_str} ({idx}/{total_seasons})")

        # Per-season counters
        season_games_inserted = 0
        no_games_count = 0
        errors_count = 0
        dates_processed = 0
        error_dates = []

        season_dates = seasons[season_str]

        with tqdm(
            total=len(season_dates), desc=f"Dates in {season_str}", unit="day"
        ) as pbar:
            for day_date in season_dates:
                game_date_str = day_date.strftime("%m/%d/%Y")
                pbar.set_postfix({"date": game_date_str})
                try:
                    inserted = load_games_for_date(
                        conn, game_date_str, save_to_db=save_to_db
                    )
                    if inserted == 0:
                        no_games_count += 1
                    season_games_inserted += inserted
                except Exception:
                    errors_count += 1
                    error_dates.append(game_date_str)

                dates_processed += 1

                # Every BATCH_SIZE dates, log a summary
                if dates_processed % BATCH_SIZE == 0:
                    logger.info(
                        f"Season {season_str}: {dates_processed} dates processed so far. "
                        f"No games: {no_games_count}, Errors: {errors_count}"
                    )

                pbar.update(1)
                time.sleep(SLEEP_DURATION)  # rate limiting

        # End of the season, log a final summary
        logger.info(
            f"Completed loading season {season_str}. Inserted/Upserted {season_games_inserted} games. "
            f"No games: {no_games_count}, Errors: {errors_count}"
        )
        if error_dates:
            logger.error(
                f"Dates with errors in season {season_str}: {', '.join(error_dates)}"
            )

        overall_games_inserted += season_games_inserted

    logger.info(
        f"Initial load complete. Total games inserted/updated: {overall_games_inserted}"
    )


def update_games_data(conn, save_to_db=True):
    """
    Update the Games table with the latest NBA game data.

    Updates:
    - All incomplete games dated up to today.
    - Today's and tomorrow's dates for potential schedule changes.

    Args:
        conn (sqlite3.Connection): The database connection.
        save_to_db (bool): Whether to commit data to the database.

    Returns:
        None
    """
    logger.info("Starting update of incomplete games...")

    today = date.today()
    tomorrow = today + timedelta(days=1)

    cursor = conn.execute(
        """
        SELECT date_time_est FROM Games 
        WHERE (status != 'Completed' AND status != 'Unknown')
          AND date(date_time_est) <= date(?)
    """,
        (today.isoformat(),),
    )
    rows = cursor.fetchall()

    unique_dates = set()
    for r in rows:
        dt_str = r[0]
        dt_part = dt_str.split("T")[0] if "T" in dt_str else dt_str
        try:
            dt_obj = datetime.strptime(dt_part, "%Y-%m-%d")
            game_date_str = dt_obj.strftime("%m/%d/%Y")
            unique_dates.add(game_date_str)
        except ValueError:
            logger.error(f"Invalid date format in DB record: {dt_str}")

    # Add today's and tomorrow's date
    today_str = today.strftime("%m/%d/%Y")
    tomorrow_str = tomorrow.strftime("%m/%d/%Y")
    unique_dates.add(today_str)
    unique_dates.add(tomorrow_str)

    if not unique_dates:
        logger.info(
            "No incomplete games require updating and no current/future dates to update."
        )
        return

    logger.info(
        f"Updating {len(unique_dates)} unique dates (including today and tomorrow)."
    )

    error_dates = []

    with tqdm(total=len(unique_dates), desc="Updating dates") as pbar:
        for g_date in sorted(unique_dates):
            pbar.set_postfix({"date": g_date})
            try:
                load_games_for_date(conn, g_date, save_to_db=save_to_db)
            except Exception:
                logger.error(f"Error updating games for date {g_date}")
                error_dates.append(g_date)
            pbar.update(1)
            time.sleep(SLEEP_DURATION)  # rate limit pacing if needed

    if error_dates:
        logger.error(f"Dates with errors during update: {', '.join(error_dates)}")

    logger.info("Update of incomplete games completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Game data management")
    parser.add_argument(
        "--task",
        choices=["initial_load", "load_date", "update"],
        help="The task to complete: 'initial_load' for full data, 'load_date' for a given date, 'update' for incomplete games.",
        required=True,
    )
    parser.add_argument(
        "--game_date",
        type=str,
        help="Date in MM/DD/YYYY format (required if task=load_date).",
    )
    parser.add_argument(
        "--save_to_db",
        action="store_true",
        help="If set, commits data to the database.",
    )
    args = parser.parse_args()

    with sqlite3.connect(DB_PATH) as conn:
        if args.task == "initial_load":
            load_initial_games(conn, save_to_db=args.save_to_db)
        elif args.task == "load_date":
            if not args.game_date:
                raise ValueError("--game_date argument is required when task=load_date")
            load_games_for_date(conn, args.game_date, save_to_db=args.save_to_db)
        elif args.task == "update":
            update_games_data(conn, save_to_db=args.save_to_db)

    logger.info("Game data management tasks completed.")
