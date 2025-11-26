"""
players.py

This module fetches the latest players data from the NBA API and
uses it to update the Players table in the SQLite database.

Enhanced with features from database_updater_CM:
- Height parsing ("6-9" → 81.0 inches)
- Age calculation from birthdate
- Progress bars with tqdm
- Retry logic with detailed error handling

Functions:
- update_players(db_path=DB_PATH): Orchestrates the process of updating the players data.
- fetch_players(): Fetches the players data from the NBA API and processes it.
- save_players(players_data, db_path=DB_PATH): Saves the fetched players data to the database.
- parse_height(height_str): Converts height strings to total inches.
- calculate_age(birthdate_str): Calculates age from birthdate.
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
import time
from datetime import date, datetime

import requests
from nba_api.stats.endpoints import commonallplayers, commonplayerinfo
from tqdm import tqdm

from src.config import config
from src.logging_config import setup_logging
from src.utils import (
    NBATeamConverter,
    determine_current_season,
    log_execution_time,
    requests_retry_session,
)

# Configuration
DB_PATH = config["database"]["path"]
NBA_API_PLAYERS_ENDPOINT = config["nba_api"]["players_endpoint"]
NBA_API_STATS_HEADERS = config["nba_api"]["pbp_stats_headers"]

# CM enhancements
BATCH_SIZE = 10  # Players to process per batch
SLEEP_DURATION = 0.6  # Seconds between batches to respect API limits


def parse_height(height_str):
    """
    Convert a height string like "6-9" to a float representing total inches.

    Args:
        height_str (str): Height in the format "feet-inches" (e.g., "6-9").

    Returns:
        float: Height converted to total inches, or None if invalid.
    """
    if not height_str:
        return None
    parts = height_str.split("-")
    if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
        feet = int(parts[0])
        inches = int(parts[1])
        total_inches = feet * 12 + inches
        return float(total_inches)
    return None


def calculate_age(birthdate_str):
    """
    Calculate the age in years from a birthdate string.

    Args:
        birthdate_str (str): Birthdate in format "YYYY-MM-DDT00:00:00".

    Returns:
        int: Age in years, or None if invalid.
    """
    if not birthdate_str:
        return None
    try:
        birthdate = datetime.strptime(birthdate_str.split("T")[0], "%Y-%m-%d").date()
        today = date.today()
        age = (
            today.year
            - birthdate.year
            - ((today.month, today.day) < (birthdate.month, birthdate.day))
        )
        return age
    except (ValueError, AttributeError):
        return None


@log_execution_time()
def update_players(db_path=DB_PATH):
    """
    Orchestrates the process of updating the players data by fetching the latest data
    from the NBA API and saving it to the SQLite database.

    Enhanced with:
    - Fetches detailed player info (height, weight, position, age) from commonplayerinfo
    - Progress bars showing batch processing
    - Only updates players with changes (from_year, to_year, roster_status, team)

    Args:
        db_path (str): Path to the SQLite database file. Defaults to the path specified
                       in the configuration file.
    """
    logging.info("Starting player data update process...")

    # Get existing players from database
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT person_id, from_year, to_year, roster_status, team FROM Players"
        )
        db_players = {
            row[0]: {
                "from_year": row[1],
                "to_year": row[2],
                "roster_status": row[3],
                "team": row[4],
            }
            for row in cursor.fetchall()
        }

    # Fetch latest player metadata from NBA API
    players_data = fetch_players()

    # Determine which players need updating
    players_to_update = []
    for player in players_data:
        person_id = player["person_id"]
        if person_id not in db_players:
            # New player
            players_to_update.append(player)
        else:
            # Check if any fields changed
            db_data = db_players[person_id]
            if (
                player["from_year"] != db_data["from_year"]
                or player["to_year"] != db_data["to_year"]
                or player["roster_status"] != db_data["roster_status"]
                or player["team"] != db_data["team"]
            ):
                players_to_update.append(player)

    if not players_to_update:
        logging.info("No players require updates.")
        return

    logging.info(f"{len(players_to_update)} players to update (new or changed).")

    # Fetch detailed info for players needing updates
    enriched_players = fetch_detailed_player_info(players_to_update)

    # Save to database
    save_players(enriched_players, db_path)

    logging.info("Player data update complete.")


def fetch_detailed_player_info(players_list):
    """
    Fetch detailed player information (height, weight, position, age) for a list of players.

    Args:
        players_list (list): List of player dictionaries from fetch_players().

    Returns:
        list: List of enriched player dictionaries with height, weight, position, age.
    """
    enriched_players = []

    def chunk_list(lst, chunk_size):
        """Yield successive chunks from a list."""
        for i in range(0, len(lst), chunk_size):
            yield lst[i : i + chunk_size]

    total_players = len(players_list)
    with tqdm(
        total=total_players, desc="Enriching player data", unit="player"
    ) as progress_bar:
        for batch in chunk_list(players_list, BATCH_SIZE):
            for player in batch:
                person_id = player["person_id"]

                # Fetch detailed info from commonplayerinfo
                try:
                    info = commonplayerinfo.CommonPlayerInfo(
                        player_id=person_id
                    ).get_dict()
                    result_row = info["resultSets"][0]["rowSet"]
                except Exception as e:
                    logging.debug(
                        f"Error fetching detailed info for player_id {person_id}: {e}"
                    )
                    result_row = []

                # Parse detailed info or use None
                if not result_row:
                    position = None
                    height_val = None
                    weight_val = None
                    age = None
                else:
                    pinfo = result_row[0]
                    position = pinfo[15] if pinfo[15] != "" else None  # POSITION
                    height_str = pinfo[11] if pinfo[11] != "" else None  # HEIGHT
                    weight_val = pinfo[12] if pinfo[12] != "" else None  # WEIGHT
                    birthdate_str = pinfo[7] if pinfo[7] != "" else None  # BIRTHDATE

                    height_val = parse_height(height_str)
                    age = calculate_age(birthdate_str)

                # Add enriched fields to player dict
                enriched_player = {
                    **player,  # All original fields
                    "position": position,
                    "height": height_val,
                    "weight": weight_val,
                    "age": age,
                }
                enriched_players.append(enriched_player)

            progress_bar.update(len(batch))
            time.sleep(SLEEP_DURATION)  # Rate limiting

    return enriched_players


@log_execution_time(average_over="output")
def fetch_players():
    """
    Fetches the players data from the NBA API and processes it into a list of dictionaries.

    Returns:
        list: A list of dictionaries, where each dictionary contains information about
              a player. If an error occurs during the API request or data processing,
              an empty list is returned.
    """
    logging.info(f"Fetching players data from the NBA API...")

    endpoint = NBA_API_PLAYERS_ENDPOINT
    logging.debug(f"Endpoint URL: {endpoint}")

    # Determine the current NBA season
    current_season = determine_current_season()
    api_season = (
        current_season[:5] + current_season[-2:]
    )  # Format the season for API request

    # Format the endpoint with the current season
    endpoint = NBA_API_PLAYERS_ENDPOINT.format(season=api_season)

    try:
        # Retry session setup with timeout
        session = requests_retry_session(timeout=10)
        response = session.get(endpoint, headers=NBA_API_STATS_HEADERS)
        logging.debug(f"Response status code: {response.status_code}")
        response.raise_for_status()  # Raise an error for bad status codes
    except requests.exceptions.RequestException as e:
        logging.error(f"Error occurred while fetching the players data: {e}")
        return []

    try:
        data = response.json()  # Parse the response JSON
        players_data = data["resultSets"][0][
            "rowSet"
        ]  # Extract the relevant part of the data

        # Process the data to create a list of player dictionaries
        players_list = []
        for player in players_data:
            try:
                person_id = player[0]
                name_parts = player[1].split(", ")

                # Handle different name formats
                if len(name_parts) > 2:
                    last_name, first_name = name_parts[0], name_parts[1]
                elif len(name_parts) == 2:
                    last_name, first_name = name_parts
                else:
                    name_parts = player[1].split(" ")
                    if len(name_parts) > 1:
                        last_name, first_name = name_parts[-1], " ".join(
                            name_parts[:-1]
                        )
                    else:
                        last_name, first_name = name_parts[0], ""

                full_name = player[2]
                roster_status = player[3]
                from_year = player[4]
                to_year = player[5]
                team_name = player[11]

                # Convert team name to abbreviation
                if team_name:
                    team_abbr = NBATeamConverter.get_abbreviation(team_name)
                else:
                    team_abbr = None

                # Create a dictionary for the player
                player_dict = {
                    "person_id": person_id,
                    "first_name": first_name,
                    "last_name": last_name,
                    "full_name": full_name,
                    "from_year": from_year,
                    "to_year": to_year,
                    "roster_status": roster_status,
                    "team": team_abbr,
                }
                players_list.append(player_dict)
            except (KeyError, TypeError, ValueError) as e:
                logging.error(f"Error processing the player record {player} for: {e}")

    except (KeyError, TypeError) as e:
        logging.error(f"Error processing the players data for: {e}")
        return []

    logging.info(f"Successfully fetched players data for {len(players_list)} players.")
    logging.debug(f"Sample player data: {players_list[0]}")

    return players_list


@log_execution_time(average_over="players_data")
def save_players(players_data, db_path=DB_PATH):
    """
    Saves the fetched players data into the SQLite database.

    Enhanced to save height, weight, position, and age fields.

    Args:
        players_data (list): A list of dictionaries containing player information.
        db_path (str): Path to the SQLite database file. Defaults to the path specified
                       in the configuration file.
    """
    logging.info(f"Saving {len(players_data)} players to the database...")

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()

        # Prepare the data for bulk insert as tuples
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
                player.get("position"),
                player.get("height"),
                player.get("weight"),
                player.get("age"),
            )
            for player in players_data
        ]

        # Insert or replace records based on person_id
        cursor.executemany(
            """
            INSERT OR REPLACE INTO Players (
                person_id, first_name, last_name, full_name, from_year, to_year, 
                roster_status, team, position, height, weight, age
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            players_tuples,
        )

    logging.info(f"Successfully saved {len(players_data)} players to the database.")


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
