"""
players.py

Overview:
This module handles the loading and updating of NBA player data into a SQLite database. It consists of functions to:
- Fetch player data from the NBA API.

- Process and save player data to the database.
- Update existing player records with the latest information.

Functions:
- load_initial_players(conn, batch_size=10, sleep_duration=1, save_to_db=True): Loads all player data into the database.
- update_player_data(conn, batch_size=10, sleep_duration=1, save_to_db=True): Updates player data in the database.
- parse_height(height_str): Converts height strings to total inches.
- calculate_age(birthdate_str): Calculates age from birthdate strings.
- chunk_list(lst, chunk_size): Splits a list into chunks of specified size.

Usage:
- Typically run as part of a larger data collection pipeline.
- Script can be run directly from the command line to load or update player data:
    python -m src.database_updater_CM.players --task=initial_load --save_to_db
    python -m src.database_updater_CM.players --task=update --save_to_db
"""

import argparse
import logging
import sqlite3
import time
from datetime import date, datetime

from nba_api.stats.endpoints import commonallplayers, commonplayerinfo
from tqdm import tqdm

from src.config import config
from src.logging_config_CM import setup_logging

# Configuration
DB_PATH = config["database"]["path"]

# Set up logging using the centralized configuration
setup_logging(
    log_level="INFO",
    log_file="custom_model.log",
    structured=True,  # Use structured JSON logs for easier parsing
)

logger = logging.getLogger("players")


def parse_height(height_str):
    """
    Convert a height string like "6-9" to a float representing total inches.

    Args:
        height_str (str): Height in the format "feet-inches" (e.g., "6-9").

    Returns:
        float: Height converted to total inches.
        None: If the input is invalid or empty.

    The function splits the input string on the hyphen, converts feet and inches to integers,
    and calculates the total height in inches by multiplying feet by 12 and adding the inches.
    """
    # Convert a height string like "6-9" to a float (in inches)
    # If height_str is None or empty, return None
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
        birthdate_str (str): Birthdate in the format "YYYY-MM-DDT00:00:00" (e.g., "1984-12-30T00:00:00").

    Returns:
        int: Age in years.
        None: If the input is invalid or empty.

    The function extracts the date portion from the birthdate string, converts it to a date object,
    and computes the age by comparing it with the current date, accounting for leap years and whether
    the birthday has occurred yet this year.
    """
    # birthdate_str is usually in the format YYYY-MM-DDT00:00:00
    # Example: "1984-12-30T00:00:00"
    # Strip time and parse
    if not birthdate_str:
        return None
    birthdate = datetime.strptime(birthdate_str.split("T")[0], "%Y-%m-%d").date()
    today = date.today()
    age = (
        today.year
        - birthdate.year
        - ((today.month, today.day) < (birthdate.month, birthdate.day))
    )
    return age


def chunk_list(lst, chunk_size):
    """
    Yield successive chunks from a list of specified size.

    Args:
        lst (list): The list to be divided into chunks.
        chunk_size (int): The size of each chunk.

    Yields:
        list: Slices of the original list with length up to 'chunk_size'.

    This generator function is useful for iterating over a list in fixed-size batches,
    which is helpful for processing large datasets without loading everything into memory.
    """
    for i in range(0, len(lst), chunk_size):
        yield lst[i : i + chunk_size]


def load_initial_players(conn, batch_size=10, sleep_duration=1, save_to_db=True):
    """
    Load initial NBA player data into the database.

    Args:
        conn (sqlite3.Connection): SQLite database connection object.
        batch_size (int, optional): Number of players to process in each batch. Defaults to 10.
        sleep_duration (int, optional): Time in seconds to sleep between batches to respect API rate limits. Defaults to 1.
        save_to_db (bool, optional): Whether to save the fetched data to the database. Defaults to True.

    This function fetches all player data from the NBA API, processes each player's information,
    and inserts the data into the 'Players' table in the database. It handles missing data,
    logs progress using a progress bar and logging messages, and respects API rate limits by sleeping
    between batches.
    """
    all_players = commonallplayers.CommonAllPlayers(is_only_current_season=0).get_dict()
    player_data = all_players["resultSets"][0]["rowSet"]
    total_players = len(player_data)

    players_info = []
    for row in player_data:
        player_id = row[0]  # PERSON_ID
        full_name = row[2]  # DISPLAY_FIRST_LAST
        roster_status = row[3]  # ROSTERSTATUS
        from_year = row[4]  # FROM_YEAR
        to_year = row[5]  # TO_YEAR
        team_id = row[8] if row[8] != 0 else None  # TEAM_ID

        name_parts = full_name.split(" ")
        first_name = name_parts[0]
        last_name = " ".join(name_parts[1:]) if len(name_parts) > 1 else ""

        players_info.append(
            {
                "player_id": player_id,
                "first_name": first_name,
                "last_name": last_name,
                "full_name": full_name,
                "from_year": from_year,
                "to_year": to_year,
                "roster_status": roster_status,
                "team_id": team_id,
            }
        )

    insert_sql = """
    INSERT INTO Players (
        player_id, first_name, last_name, full_name,
        from_year, to_year, roster_status, team_id,
        height, weight, position, age
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    empty_records_count = 0
    total_records_processed = 0

    with tqdm(
        total=total_players, desc="Loading Players", unit="player"
    ) as progress_bar:
        for batch in chunk_list(players_info, batch_size):
            all_inserts = []
            for player_record in batch:
                player_id = player_record["player_id"]
                try:
                    info = commonplayerinfo.CommonPlayerInfo(
                        player_id=player_id
                    ).get_dict()
                    result_row = info["resultSets"][0]["rowSet"]
                except Exception as e:
                    logger.error(
                        f"Error fetching player info for player_id {player_id}: {e}"
                    )
                    result_row = []

                if not result_row:
                    position = None
                    height_val = None
                    weight_val = None
                    age = None
                    empty_records_count += 1
                else:
                    pinfo = result_row[0]
                    position = pinfo[15] if pinfo[15] != "" else None  # POSITION
                    height_str = pinfo[11] if pinfo[11] != "" else None  # HEIGHT
                    weight_val = pinfo[12] if pinfo[12] != "" else None  # WEIGHT
                    birthdate_str = pinfo[7] if pinfo[7] != "" else None  # BIRTHDATE

                    height_val = parse_height(height_str)
                    age = calculate_age(birthdate_str)

                all_inserts.append(
                    (
                        player_record["player_id"],
                        player_record["first_name"],
                        player_record["last_name"],
                        player_record["full_name"],
                        player_record["from_year"],
                        player_record["to_year"],
                        player_record["roster_status"],
                        player_record["team_id"],
                        height_val,
                        weight_val,
                        position,
                        age,
                    )
                )

            total_records_processed += len(batch)

            if save_to_db:
                try:
                    conn.executemany(insert_sql, all_inserts)
                    conn.commit()  # Ensure the transaction is committed
                except Exception as e:
                    logger.error(f"Error inserting batch into database: {e}")

            # Log one record per batch for debugging
            logger.info(f"Batch processed. Sample record: {all_inserts[0]}")
            logger.info(f"Running empty records count: {empty_records_count}")
            logger.info(f"Total records processed: {total_records_processed}")

            # Update progress bar for each player
            progress_bar.update(len(batch))

            # Sleep after processing a batch regardless of saving to the database
            time.sleep(sleep_duration)

    if save_to_db:
        logger.info(
            f"Initial player load complete. Empty records count: {empty_records_count}"
        )
    else:
        logger.info("Data loaded but not saved to the database.")
        logger.info(f"Empty records count: {empty_records_count}")
        logger.info(f"Total records processed: {total_records_processed}")


def update_player_data(conn, batch_size=10, sleep_duration=1, save_to_db=True):
    """
    Update the Players table with the latest NBA player data.

    Args:
        conn (sqlite3.Connection): SQLite database connection object.
        batch_size (int, optional): Number of players to process in each batch. Defaults to 10.
        sleep_duration (int, optional): Time in seconds to sleep between batches to respect API rate limits. Defaults to 1.
        save_to_db (bool, optional): Whether to save the updates to the database. Defaults to True.

    This function compares existing player records in the database with the latest data from the NBA API.
    It identifies new players or players with updated attributes, fetches detailed information for those players,
    and updates the database accordingly. The function minimizes API calls by only fetching data for players
    that require updates and handles batch processing and error logging throughout the process.
    """
    logger.info("Starting player data update process...")

    # 1. Fetch all players currently in the database
    db_players = {}
    cursor = conn.execute(
        "SELECT player_id, from_year, to_year, roster_status, team_id FROM Players"
    )
    for row in cursor.fetchall():
        player_id = row[0]
        db_players[player_id] = {
            "from_year": row[1],
            "to_year": row[2],
            "roster_status": row[3],
            "team_id": row[4],
        }

    # 2. Fetch the latest player metadata from the NBA API
    all_players = commonallplayers.CommonAllPlayers(is_only_current_season=0).get_dict()
    player_data = all_players["resultSets"][0]["rowSet"]

    # Process the incoming data into a dict for easy comparison
    incoming_players = {}
    for row in player_data:
        player_id = row[0]  # PERSON_ID
        full_name = row[2]  # DISPLAY_FIRST_LAST
        roster_status = row[3]  # ROSTERSTATUS
        from_year = row[4]  # FROM_YEAR
        to_year = row[5]  # TO_YEAR
        team_id = row[8] if row[8] != 0 else None  # TEAM_ID

        # Split full name into first and last
        name_parts = full_name.split(" ")
        first_name = name_parts[0]
        last_name = " ".join(name_parts[1:]) if len(name_parts) > 1 else ""

        incoming_players[player_id] = {
            "first_name": first_name,
            "last_name": last_name,
            "full_name": full_name,
            "from_year": from_year,
            "to_year": to_year,
            "roster_status": roster_status,
            "team_id": team_id,
        }

    # 3. Determine which players need updating
    # Conditions:
    # - Player not in DB => New player, needs to be fetched and inserted
    # - Player in DB but any key fields differ => Needs update
    players_to_update = []
    for player_id, p_data in incoming_players.items():
        if player_id not in db_players:
            # New player
            players_to_update.append(player_id)
        else:
            db_data = db_players[player_id]
            if (
                int(p_data["from_year"]) != db_data["from_year"]
                or int(p_data["to_year"]) != db_data["to_year"]
                or p_data["roster_status"] != db_data["roster_status"]
                or p_data["team_id"] != db_data["team_id"]
            ):
                players_to_update.append(player_id)

    if not players_to_update:
        logger.info("No players require updates.")
        return

    # 4. Prepare SQL for upsert
    insert_update_sql = """
    INSERT INTO Players (
        player_id, first_name, last_name, full_name,
        from_year, to_year, roster_status, team_id,
        height, weight, position, age
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(player_id) DO UPDATE SET
        first_name=excluded.first_name,
        last_name=excluded.last_name,
        full_name=excluded.full_name,
        from_year=excluded.from_year,
        to_year=excluded.to_year,
        roster_status=excluded.roster_status,
        team_id=excluded.team_id,
        height=excluded.height,
        weight=excluded.weight,
        position=excluded.position,
        age=excluded.age
    """

    # 5. Fetch detailed info and update in batches
    logger.info(f"{len(players_to_update)} players to update.")
    with tqdm(
        total=len(players_to_update), desc="Updating Players", unit="player"
    ) as progress_bar:
        for batch in chunk_list(players_to_update, batch_size):
            records_to_save = []

            for player_id in batch:
                base_data = incoming_players[player_id]

                # Fetch player detail from API
                try:
                    info = commonplayerinfo.CommonPlayerInfo(
                        player_id=player_id
                    ).get_dict()
                    result_row = info["resultSets"][0]["rowSet"]
                except Exception as e:
                    logger.error(
                        f"Error fetching player info for player_id {player_id}: {e}"
                    )
                    result_row = []

                if not result_row:
                    # If no info is available, mark as inactive and leave other fields None
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

                record = (
                    player_id,
                    base_data["first_name"],
                    base_data["last_name"],
                    base_data["full_name"],
                    base_data["from_year"],
                    base_data["to_year"],
                    base_data["roster_status"],
                    base_data["team_id"],
                    height_val,
                    weight_val,
                    position,
                    age,
                )
                records_to_save.append(record)

            if save_to_db:
                try:
                    conn.executemany(insert_update_sql, records_to_save)
                    conn.commit()
                except Exception as e:
                    logger.error(f"Error updating batch in the database: {e}")

            # Log a sample record
            if records_to_save:
                logger.info(f"Batch processed. Sample record: {records_to_save[0]}")

            progress_bar.update(len(batch))
            time.sleep(sleep_duration)

    logger.info("Player updates completed successfully.")


# Main execution for running specific tasks
if __name__ == "__main__":

    # Set up argument parser to allow running specific functions
    parser = argparse.ArgumentParser(description="Player data management")
    parser.add_argument(
        "--task",
        choices=["initial_load", "update"],
        help="The task to complete. 'initial_load' loads all players into the database. 'update' updates player data.",
        required=True,
    )
    parser.add_argument(
        "--save_to_db",
        action="store_true",
        help="Flag to save data to the database. If not set, data is loaded but not saved.",
    )
    args = parser.parse_args()

    # Execute the requested task
    if args.task == "initial_load":
        with sqlite3.connect(DB_PATH) as conn:
            load_initial_players(conn, save_to_db=args.save_to_db)
    elif args.task == "update":
        with sqlite3.connect(DB_PATH) as conn:
            update_player_data(conn, save_to_db=args.save_to_db)

    logger.info("Player data management tasks completed.")
