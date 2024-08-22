import argparse
import json
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


def update_players(db_path=DB_PATH):
    players_data = fetch_players()
    save_players(players_data, db_path)


@log_execution_time()
def fetch_players():
    """
    Fetches the players data from the NBA API.

    Returns:
    ???
    """
    logging.info(f"Fetching players data from the NBA API...")

    endpoint = NBA_API_PLAYERS_ENDPOINT
    logging.debug(f"Endpoint URL: {endpoint}")

    current_season = determine_current_season()
    api_season = current_season[:5] + current_season[-2:]

    endpoint = NBA_API_PLAYERS_ENDPOINT.format(season=api_season)

    try:
        session = requests_retry_session(timeout=10)
        response = session.get(endpoint, headers=NBA_API_STATS_HEADERS)
        logging.debug(f"Response status code: {response.status_code}")
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        logging.error(f"Error occurred while fetching the players data: {e}")
        return []

    try:
        data = response.json()
        # Extracting the relevant part of the data
        players_data = data["resultSets"][0]["rowSet"]

        # Process the data to create a list of dictionaries
        players_list = []
        for player in players_data:
            person_id = player[0]
            last_name, first_name = player[1].split(", ")
            full_name = player[2]
            roster_status = player[3]
            from_year = player[4]
            to_year = player[5]
            team_name = player[11]

            # Convert team name to abbreviation
            team_abbr = NBATeamConverter.get_abbreviation(team_name)

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

    except (KeyError, TypeError) as e:
        logging.error(f"Error processing the players data for: {e}")
        return []


def save_players(players_data, db_path=DB_PATH):
    pass


def main():
    """
    Main function to handle command-line arguments and orchestrate updating the players data.
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
