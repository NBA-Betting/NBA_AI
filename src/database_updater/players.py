"""
players.py

This module fetches the latest players data from the NBA API and 
uses it to update the Players table in the SQLite database.

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


@log_execution_time()
def update_players(db_path=DB_PATH):
    """
    Orchestrates the process of updating the players data by fetching the latest data
    from the NBA API and saving it to the SQLite database.

    Args:
        db_path (str): Path to the SQLite database file. Defaults to the path specified
                       in the configuration file.
    """
    players_data = fetch_players()
    save_players(players_data, db_path)


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
            )
            for player in players_data
        ]

        # Insert or replace records based on person_id
        cursor.executemany(
            """
            INSERT OR REPLACE INTO Players (
                person_id, first_name, last_name, full_name, from_year, to_year, roster_status, team
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
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
