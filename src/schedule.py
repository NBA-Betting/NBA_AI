"""
schedule.py

This module fetches and saves NBA schedule data for a given season.
It consists of functions to:
- Fetch the schedule from the NBA API.
- Validate and save the schedule to a SQLite database.
- Ensure data integrity by checking for empty or corrupted data before updating the database.

Functions:
- fetch_schedule(season): Fetches the NBA schedule for a specified season.
- save_schedule(games, season, db_path): Saves the fetched schedule to the database.
- update_schedule(season, db_path): Orchestrates fetching and saving the schedule.
- main(): Handles command-line arguments to fetch and/or save the schedule, with optional timing.

Usage:
- Typically run as part of a larger data collection pipeline.
- Script can be run directly from the command line (project root) to fetch and save NBA schedule data:
    python -m src.schedule --fetch --save --season=2023-2024 --timing
- Successful execution will print the number of games fetched along with the first and last games fetched.
"""

import logging
import sqlite3

import requests

from src.config import config
from src.utils import requests_retry_session, validate_season_format

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s",
)

# Configuration values
DB_PATH = config["database"]["path"]
NBA_API_BASE_URL = config["nba_api"]["schedule_endpoint"]
NBA_API_HEADERS = config["nba_api"]["schedule_headers"]


def fetch_schedule(season):
    """
    Fetches the NBA schedule for a given season.

    Parameters:
    season (str): The season to fetch the schedule for, formatted as 'XXXX-XXXX' (e.g., '2020-2021').

    Returns:
    list: A list of dictionaries, each containing details of a game. If the request fails or the data is corrupted, an empty list is returned.
    """
    validate_season_format(season, abbreviated=False)
    api_season = season[:5] + season[-2:]

    endpoint = NBA_API_BASE_URL.format(season=api_season)

    try:
        session = requests_retry_session(timeout=10)
        response = session.get(endpoint, headers=NBA_API_HEADERS)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        logging.error(f"Error occurred while fetching the schedule for {season}.")
        return []

    try:
        game_dates = response.json()["leagueSchedule"]["gameDates"]
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

        return all_games

    except (KeyError, TypeError) as e:
        logging.error(f"Error processing the schedule data for {season}: {e}")
        return []


def save_schedule(games, season, db_path):
    """
    Saves the NBA schedule to the database. This function first checks the validity of the data,
    then performs a DELETE operation to remove old records for the given season,
    followed by an INSERT operation for the new games.

    Parameters:
    games (list): A list of game dictionaries to be saved.
    season (str): The season to which the games belong.
    db_path (str): The path to the SQLite database file.

    Returns:
    bool: True if the operation was successful, False otherwise.
    """
    if not games:
        logging.error("No games fetched. Skipping database update to avoid data loss.")
        return False

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()

        # Check for data corruption or unexpected issues in the new data
        if any("gameId" not in game or "gameDateTimeEst" not in game for game in games):
            logging.error(
                "Fetched schedule data is corrupted. Skipping database update to avoid data loss."
            )
            return False

        # Delete old records for the given season
        delete_sql = "DELETE FROM Games WHERE season = ?"
        cursor.execute(delete_sql, (season,))

        # Insert new game records
        insert_sql = """
        INSERT INTO Games
        (game_id, date_time_est, home_team, away_team, status, season, season_type)
        VALUES (:gameId, :gameDateTimeEst, :homeTeam, :awayTeam, :gameStatus, :season, :seasonType)
        """

        for game in games:
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

        conn.commit()
        logging.info("Schedule updated successfully.")
        return True


def update_schedule(season, db_path):
    """
    Fetches and updates the NBA schedule for a given season in the database.

    Parameters:
    season (str): The season to fetch and update the schedule for.
    db_path (str): The path to the SQLite database file.
    """
    games = fetch_schedule(season)
    save_schedule(games, season, db_path)


def main():
    """
    Main function to handle command-line arguments and orchestrate fetching and saving NBA schedule data.
    """
    import argparse
    import time

    parser = argparse.ArgumentParser(description="Fetch and save NBA schedule data.")
    parser.add_argument("--fetch", action="store_true", help="Fetch NBA schedule data")
    parser.add_argument(
        "--save", action="store_true", help="Save NBA schedule data to database"
    )
    parser.add_argument(
        "--season", type=str, help="Season to fetch/update (format 'XXXX-XXXX')"
    )
    parser.add_argument("--timing", action="store_true", help="Measure execution time")

    args = parser.parse_args()

    if not args.fetch and not args.save:
        parser.error("No action requested, add --fetch or --save")

    season = args.season

    if args.fetch:
        start_time = time.time()
        games = fetch_schedule(season)
        print(games[:5])
        print(games[-5:])
        print(f"Total games fetched: {len(games)}")
        if args.timing:
            elapsed_time = time.time() - start_time
            logging.info(f"Fetching data took {elapsed_time:.2f} seconds.")

    if args.save:
        if "games" not in locals():
            logging.error(
                "No data to save. Ensure --fetch is used or data is provided."
            )
        else:
            start_time = time.time()
            save_schedule(games, season, DB_PATH)
            if args.timing:
                elapsed_time = time.time() - start_time
                logging.info(f"Saving data took {elapsed_time:.2f} seconds.")


if __name__ == "__main__":
    main()
