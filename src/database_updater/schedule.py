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
    python -m src.database_updater.schedule --season=2023-2024 --log_level=DEBUG
- Successful execution will print the number of games fetched and saved along with logging information.
"""

import argparse
import logging
import sqlite3

import requests

from src.config import config
from src.logging_config import setup_logging
from src.utils import (
    determine_current_season,
    log_execution_time,
    requests_retry_session,
    validate_season_format,
)

# Configuration values
DB_PATH = config["database"]["path"]
NBA_API_BASE_URL = config["nba_api"]["schedule_endpoint"]
NBA_API_HEADERS = config["nba_api"]["schedule_headers"]


@log_execution_time()
def update_schedule(season="Current", db_path=DB_PATH):
    """
    Fetches and updates the NBA schedule for a given season in the database.

    Parameters:
    season (str): The season to fetch and update the schedule for. Defaults to "Current".
    db_path (str): The path to the SQLite database file. Defaults to the configured database path.
    """
    if season == "Current":
        season = determine_current_season()
    else:
        validate_season_format(season, abbreviated=False)

    games = fetch_schedule(season)
    save_schedule(games, season, db_path)


@log_execution_time()
def fetch_schedule(season):
    """
    Fetches the NBA schedule for a given season.

    Parameters:
    season (str): The season to fetch the schedule for, formatted as 'XXXX-XXXX' (e.g., '2020-2021').

    Returns:
    list: A list of dictionaries, each containing details of a game. If the request fails or the data is corrupted, an empty list is returned.
    """
    logging.info(f"Fetching schedule for season: {season}")
    api_season = season[:5] + season[-2:]

    endpoint = NBA_API_BASE_URL.format(season=api_season)
    logging.debug(f"Endpoint URL: {endpoint}")

    try:
        session = requests_retry_session(timeout=10)
        response = session.get(endpoint, headers=NBA_API_HEADERS)
        logging.debug(f"Response status code: {response.status_code}")
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        logging.error(f"Error occurred while fetching the schedule for {season}: {e}")
        return []

    try:
        game_dates = response.json()["leagueSchedule"]["gameDates"]

        if not game_dates:
            logging.error(f"No games found for the season {season}.")
            return []

        all_games = [game for date in game_dates for game in date["games"]]

        keys_needed = [
            "gameId",
            "gameStatus",
            "gameDateTimeEst",
            "homeTeam",
            "awayTeam",
        ]

        all_games = [{key: game[key] for key in keys_needed} for game in all_games]

        for game in all_games:
            game["homeTeam"] = game["homeTeam"]["teamTricode"]
            game["awayTeam"] = game["awayTeam"]["teamTricode"]

        season_type_codes = {
            "001": "Pre Season",
            "002": "Regular Season",
            "003": "All-Star",
            "004": "Post Season",
            "005": "Post Season",  # Play-In
        }

        game_status_codes = {
            1: "Not Started",
            2: "In Progress",
            3: "Completed",
        }

        for game in all_games:
            game["seasonType"] = season_type_codes.get(game["gameId"][:3], "Unknown")
            game["gameStatus"] = game_status_codes.get(game["gameStatus"], "Unknown")
            game["season"] = season

        logging.info(f"Total Games Fetched: {len(all_games)}")
        logging.debug(f"Sample games fetched:\n{all_games[:2]}\n{all_games[-2:]}")
        return all_games

    except (KeyError, TypeError) as e:
        logging.error(f"Error processing the schedule data for {season}: {e}")
        return []


@log_execution_time()
def save_schedule(games, season, db_path=DB_PATH):
    """
    Saves the NBA schedule to the database. This function first checks the validity of the data,
    then updates the database by adding new records, updating existing ones, and removing obsolete records.

    Parameters:
    games (list): A list of game dictionaries to be saved.
    season (str): The season to which the games belong.
    db_path (str): The path to the SQLite database file.

    Returns:
    bool: True if the operation was successful, False otherwise.
    """
    logging.info(f"Saving schedule for season: {season}")
    if not games:
        logging.error("No games fetched. Skipping database update to avoid data loss.")
        return False

    game_ids = [game["gameId"] for game in games]

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()

        # Check for data corruption or unexpected issues in the new data
        if any("gameId" not in game or "gameDateTimeEst" not in game for game in games):
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
            # Count records to be deleted
            cursor.execute(
                f"SELECT COUNT(*) FROM Games WHERE season = ? AND game_id NOT IN ({','.join('?' * len(game_ids))})",
                [season] + game_ids,
            )
            removed_count = cursor.fetchone()[0]

            # Delete records not in the new data
            cursor.execute(
                f"DELETE FROM Games WHERE season = ? AND game_id NOT IN ({','.join('?' * len(game_ids))})",
                [season] + game_ids,
            )

            added_count = 0
            updated_count = 0

            # Check existing records
            existing_games = {}
            cursor.execute(
                f"SELECT game_id, date_time_est, home_team, away_team, status, season, season_type FROM Games WHERE game_id IN ({','.join('?' * len(game_ids))})",
                game_ids,
            )
            for row in cursor.fetchall():
                existing_games[row[0]] = row

            # Insert or replace new and updated game records
            insert_sql = """
            INSERT INTO Games (game_id, date_time_est, home_team, away_team, status, season, season_type, pre_game_data_finalized, game_data_finalized)
            VALUES (:gameId, :gameDateTimeEst, :homeTeam, :awayTeam, :gameStatus, :season, :seasonType,
                COALESCE((SELECT pre_game_data_finalized FROM Games WHERE game_id = :gameId), 0),
                COALESCE((SELECT game_data_finalized FROM Games WHERE game_id = :gameId), 0))
            ON CONFLICT(game_id) DO UPDATE SET
                date_time_est=excluded.date_time_est,
                home_team=excluded.home_team,
                away_team=excluded.away_team,
                status=excluded.status,
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
                        game["gameDateTimeEst"] != existing_game[1]
                        or game["homeTeam"] != existing_game[2]
                        or game["awayTeam"] != existing_game[3]
                        or game["gameStatus"] != existing_game[4]
                        or game["season"] != existing_game[5]
                        or game["seasonType"] != existing_game[6]
                    ):
                        updated_count += 1

                cursor.execute(
                    insert_sql,
                    {
                        "gameId": game["gameId"],
                        "gameDateTimeEst": game["gameDateTimeEst"],
                        "homeTeam": game["homeTeam"],
                        "awayTeam": game["awayTeam"],
                        "gameStatus": game["gameStatus"],
                        "season": game["season"],
                        "seasonType": game["seasonType"],
                    },
                )

            # Commit transaction
            conn.commit()

            logging.info(
                f"Successfully processed {len(games)} games for season {season}."
            )
            logging.info(
                f"Records added: {added_count}, updated: {updated_count}, removed: {removed_count}."
            )
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

    args = parser.parse_args()
    log_level = args.log_level.upper()
    setup_logging(log_level=log_level)

    update_schedule(season=args.season)


if __name__ == "__main__":
    main()
