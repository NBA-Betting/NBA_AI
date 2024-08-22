"""
utils.py

This module provides utility functions and classes for managing and processing NBA data, including 
database interactions, HTTP request handling, and data validation. It includes functions for looking 
up game information, validating game IDs and dates, and converting between different NBA team identifiers.

Core Functions:
- lookup_basic_game_info(game_ids, db_path=DB_PATH): Retrieves basic game information for given game IDs from the database.
- log_execution_time(average_over=None): A decorator to log the execution time of functions.
- requests_retry_session(retries=3, backoff_factor=0.3, status_forcelist=(500, 502, 504), session=None, timeout=10): Creates an HTTP session with retry logic for handling transient errors.
- game_id_to_season(game_id, abbreviate=False): Converts a game ID to a season string.
- validate_game_ids(game_ids): Validates game IDs.
- validate_date_format(date): Validates that a date string is in the format "YYYY-MM-DD".
- validate_season_format(season, abbreviated=False): Validates the format of a season string.
- date_to_season(date_str): Converts a date to the corresponding NBA season.
- determine_current_season(): Determines the current NBA season based on the current date.
- get_player_image(player_id): Retrieves a player's image from the NBA website or a local cache.

Classes:
- NBATeamConverter: A class for converting between various identifiers of NBA teams such as team ID, abbreviation, short name, and full name.

Usage:
- This module can be used to support data validation and transformation tasks in an NBA data pipeline.
- Functions are typically called to validate inputs, fetch data from the database, or format data for display.
"""

import logging
import os
import re
import sqlite3
import time
from datetime import datetime
from functools import wraps
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

from src.config import config

# Configuration values
DB_PATH = config["database"]["path"]
PROJECT_ROOT = Path(config["project"]["root"])


def lookup_basic_game_info(game_ids, db_path=DB_PATH):
    """
    Looks up basic game information given a game_id or a list of game_ids from the Games table in the SQLite database.

    Args:
        game_ids (str or list): The ID of the game or a list of game IDs to look up.
        db_path (str): The path to the SQLite database. Defaults to the value in the config file.

    Returns:
        dict: A dictionary with game IDs as keys and each value being a dictionary representing a game.
              Each game dictionary contains the home team, away team, date/time, status, season, and season type.
    """
    if not isinstance(game_ids, list):
        game_ids = [game_ids]

    validate_game_ids(game_ids)

    sql = f"""
    SELECT game_id, home_team, away_team, date_time_est, status, season, season_type
    FROM Games
    WHERE game_id IN ({','.join(['?'] * len(game_ids))})
    """

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(sql, game_ids)
        games = cursor.fetchall()

    game_ids_set = set(game_ids)
    game_info_dict = {}
    for game_id, home, away, date_time_est, status, season, season_type in games:
        game_ids_set.remove(game_id)
        game_info_dict[game_id] = {
            "home": home,
            "away": away,
            "date_time_est": date_time_est,
            "status": status,
            "season": season,
            "season_type": season_type,
        }

    if game_ids_set:
        logging.warning(f"Game IDs not found in the database: {game_ids_set}")

    return game_info_dict


def determine_current_season():
    """
    Determines the current NBA season based on the current date.
    Returns the current NBA season in 'XXXX-XXXX' format.
    """

    current_date = datetime.now()
    current_year = current_date.year

    # Determine the season based on the league year cutoff (June 30th)
    league_year_cutoff = datetime(current_year, 6, 30)

    if current_date > league_year_cutoff:
        season = f"{current_year}-{current_year + 1}"
    else:
        season = f"{current_year - 1}-{current_year}"

    return season


def get_player_image(player_id):
    """
    Gets the player's image by checking locally first, then attempting to download it,
    and finally falling back to a default image if the first two steps fail.

    Args:
        player_id (str): The ID of the player whose image is to be retrieved.

    Returns:
        str: The relative path to the player's image from the static directory.
    """
    # Define paths using Path objects based on the PROJECT_ROOT
    player_images_dir = PROJECT_ROOT / "src/web_app/static/img/player_images"
    player_image_file = player_images_dir / f"{player_id}.png"
    default_image = PROJECT_ROOT / "src/web_app/static/img/basketball_player.png"

    # Check if the image exists locally
    if player_image_file.exists():
        return f"static/img/player_images/{player_id}.png"

    # Attempt to download the image if it doesn't exist locally
    try:
        url = f"https://cdn.nba.com/headshots/nba/latest/260x190/{player_id}.png"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        }
        response = requests.get(url, headers=headers, timeout=10)

        if response.status_code == 200:
            # Save the image locally
            with player_image_file.open("wb") as f:
                f.write(response.content)
            return f"static/img/player_images/{player_id}.png"
        else:
            print(f"Image not found at {url}, status code: {response.status_code}")
    except requests.RequestException as e:
        print(f"Failed to download the image for player {player_id}: {e}")

    # If all else fails, return the default image
    return str(default_image.relative_to(PROJECT_ROOT / "src/web_app/static"))


def log_execution_time(average_over=None):
    """
    Decorator that logs the execution time of a function and optionally averages the time over the output or a specified input.

    Args:
        average_over (str or None): Specifies what to average over. Can be None, "output", or the name of an input argument.

    Returns:
        function: The wrapped function with added logging for execution time.
    """
    if average_over not in (None, "output") and not isinstance(average_over, str):
        raise ValueError(
            "average_over must be None, 'output', or a string representing an input argument name."
        )

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Validate that if average_over is specified as an argument name, it exists among the function's arguments.
            if average_over and average_over != "output" and average_over not in kwargs:
                arg_names = func.__code__.co_varnames[: func.__code__.co_argcount]
                if average_over not in arg_names:
                    raise ValueError(
                        f"The specified average_over argument '{average_over}' does not exist in the function '{func.__name__}'."
                    )

            start_time = time.time()
            logging.info(f"Starting {func.__name__}...")

            result = func(*args, **kwargs)

            duration = time.time() - start_time

            if average_over:
                items_to_average = None
                if average_over == "output":
                    if isinstance(result, (list, tuple)):
                        items_to_average = result
                    elif isinstance(result, dict):
                        items_to_average = result.keys()
                elif average_over in kwargs:
                    arg = kwargs[average_over]
                    if isinstance(arg, (list, tuple)):
                        items_to_average = arg
                    elif isinstance(arg, dict):
                        items_to_average = arg.keys()
                else:
                    for arg in args:
                        if isinstance(arg, (list, tuple, dict)):
                            items_to_average = (
                                arg if not isinstance(arg, dict) else arg.keys()
                            )
                            break

                if items_to_average is not None and len(items_to_average) > 0:
                    avg_time_per_item = duration / len(items_to_average)
                else:
                    avg_time_per_item = 0

            if average_over:
                logging.info(
                    f"{func.__name__} execution time: {duration:.2f} seconds. Average per item: {avg_time_per_item:.2f} seconds."
                )
            else:
                logging.info(f"{func.__name__} execution time: {duration:.2f} seconds.")

            return result

        return wrapper

    return decorator


def requests_retry_session(
    retries=3,
    backoff_factor=0.3,
    status_forcelist=(500, 502, 504),
    session=None,
    timeout=10,
):
    """
    Creates a session with retry logic for handling transient HTTP errors.

    Args:
        retries (int): The number of retry attempts.
        backoff_factor (float): The backoff factor for retries.
        status_forcelist (tuple): A set of HTTP status codes to trigger a retry.
        session (requests.Session): An existing session to use, or None to create a new one.
        timeout (int): The timeout for the request.

    Returns:
        requests.Session: A session configured with retry logic.
    """
    session = session or requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.timeout = timeout
    return session


def game_id_to_season(game_id, abbreviate=False):
    """
    Converts a game ID to a season.

    The season is determined based on the third and fourth characters of the game ID.
    If these characters represent a number less than 40, the season is in the 2000s, otherwise it's in the 1900s.

    Args:
        game_id (str): The game ID to convert.
        abbreviate (bool): Whether to abbreviate the second year of the season.

    Returns:
        str: The season corresponding to the game ID.
    """
    # Validate the game ID
    validate_game_ids(game_id)

    # Extract the season from the game ID
    season = game_id[3:5]

    # Determine the prefix based on the season
    prefix = "20" if int(season) < 40 else "19"

    # Construct the years for the season
    year1 = prefix + season
    year2 = str(int(year1) + 1)

    # Return the season in the appropriate format
    if abbreviate:
        return year1 + "-" + year2[2:]
    return year1 + "-" + year2


def validate_game_ids(game_ids):
    """
    Validates a game ID or a list of game IDs.

    Each game ID must be a 10-character string that starts with '00'.

    Args:
        game_ids (str or list): The game ID(s) to validate.

    Raises:
        ValueError: If any game ID is not valid.
    """
    # Ensure game_ids is a list
    if isinstance(game_ids, str):
        game_ids = [game_ids]

    invalid_game_ids = []

    # Validate the game_ids
    for game_id in game_ids:
        if not (
            game_id
            and isinstance(game_id, str)
            and len(game_id) == 10
            and game_id.startswith("00")
        ):
            invalid_game_ids.append(game_id)
            logging.warning(
                f"Invalid game ID {game_id}. Game ID must be a 10-digit string starting with '00'. Example: '0022100001'."
            )

    if invalid_game_ids:
        raise ValueError(
            f"Invalid game IDs: {invalid_game_ids}. Each game ID must be a 10-digit string starting with '00'. Example: '0022100001'."
        )


def validate_date_format(date):
    """
    Validates if the given date is in the format "YYYY-MM-DD".

    Args:
        date (str): The date string to validate.

    Raises:
        ValueError: If the date is not in the correct format or if the month or day is not valid.
    """
    # Check the overall format
    if len(date) != 10 or date[4] != "-" or date[7] != "-":
        raise ValueError("Invalid date format. Please use YYYY-MM-DD format.")

    year, month, day = date.split("-")

    # Check if year, month and day are all digits
    if not year.isdigit() or not month.isdigit() or not day.isdigit():
        raise ValueError("Invalid date format. Please use YYYY-MM-DD format.")

    year, month, day = int(year), int(month), int(day)

    # Check if month is between 1 and 12
    if month < 1 or month > 12:
        raise ValueError(
            "Invalid month. Please use MM format with a value between 01 and 12."
        )

    # Check if day is between 1 and the maximum day of the month
    if month in [4, 6, 9, 11] and day > 30:
        raise ValueError(
            "Invalid day. Please use DD format with a value between 01 and 30 for this month."
        )
    elif month == 2 and day > 29:
        raise ValueError(
            "Invalid day. Please use DD format with a value between 01 and 29 for this month."
        )
    elif day < 1 or day > 31:
        raise ValueError(
            "Invalid day. Please use DD format with a value between 01 and 31."
        )


def validate_season_format(season, abbreviated=False):
    """
    Validates the format of a season string.

    Args:
        season (str): The season string to validate, formatted as 'XXXX-XX' or 'XXXX-XXXX'.
        abbreviated (bool): Whether the second year in the season string is abbreviated.

    Raises:
        ValueError: If the season string does not match the required format or if the second year does not logically follow the first year.
    """
    FULL_SEASON_PATTERN = r"^(\d{4})-(\d{4})$"
    ABBREVIATED_SEASON_PATTERN = r"^(\d{4})-(\d{2})$"

    # Define the regex pattern based on abbreviated flag
    pattern = ABBREVIATED_SEASON_PATTERN if abbreviated else FULL_SEASON_PATTERN

    # Attempt to match the pattern to the season string
    match = re.match(pattern, season)
    if not match:
        raise ValueError(
            "Season does not match the required format. Please use 'XXXX-XX' or 'XXXX-XXXX'."
        )

    year1, year2_suffix = map(int, match.groups())

    # Handle the year2 based on whether it's abbreviated or not
    year2 = year2_suffix if not abbreviated else year1 // 100 * 100 + year2_suffix

    # Check if year2 logically follows year1
    if year1 + 1 != year2:
        raise ValueError(
            f"Second year {year2} does not logically follow the first year {year1}."
        )

    # Check if years are within a valid range
    if year1 < 1900 or year2 > 2100:
        raise ValueError(
            f"Season years must be between 1900 and 2100. {year1}-{year2} is not a valid season."
        )


def date_to_season(date_str):
    """
    Converts a date to the NBA season.

    The typical cutoff date between seasons is June 30th.
    Special cases are handled for seasons affected by lockouts and COVID-19.

    Args:
        date_str (str): The date in YYYY-MM-DD format.

    Returns:
        str: The season in YYYY-YYYY format.
    """
    # Validate the date format
    validate_date_format(date_str)

    date = datetime.strptime(date_str, "%Y-%m-%d")

    # Special cases for lockout and COVID-19 seasons (full league year)
    special_cases = [
        ("2011-2012", datetime(2011, 7, 1), datetime(2012, 6, 30)),
        ("2019-2020", datetime(2019, 7, 1), datetime(2020, 10, 11)),
        ("2020-2021", datetime(2020, 10, 12), datetime(2021, 7, 20)),
    ]

    for season, start, end in special_cases:
        if start <= date <= end:
            return season

    # General case
    year = date.year
    if date.month > 6 or (
        date.month == 6 and date.day > 30
    ):  # After June 30th, it's the next season
        return f"{year}-{year + 1}"
    else:  # Before July 1st, it's the previous season
        return f"{year - 1}-{year}"


class NBATeamConverter:
    """
    A class to convert between various identifiers of NBA teams such as team ID,
    abbreviation, short name, and full name along with any historical identifiers.
    """

    project_root = config["project"]["root"]
    relative_db_path = config["database"]["path"]
    absolute_db_path = os.path.join(project_root, relative_db_path)

    @staticmethod
    def __get_team_id(identifier):
        """
        Get the team ID corresponding to the given identifier.
        If the identifier is unknown, raise a ValueError.

        Args:
            identifier (str): The identifier of the team.

        Returns:
            int: The team ID corresponding to the identifier.

        Raises:
            ValueError: If the identifier is unknown.
        """
        # Normalize the identifier
        identifier_normalized = str(identifier).lower().replace("-", " ")

        # Open a new database connection
        with sqlite3.connect(NBATeamConverter.absolute_db_path) as conn:
            cursor = conn.cursor()

            # Execute the SQL query
            cursor.execute(
                """
                SELECT team_id FROM Teams
                WHERE abbreviation_normalized = ? OR full_name_normalized = ? OR short_name_normalized = ? OR
                json_extract(alternatives_normalized, '$') LIKE ?
                """,
                (
                    identifier_normalized,
                    identifier_normalized,
                    identifier_normalized,
                    f'%"{identifier_normalized}"%',
                ),
            )

            # Fetch the result of the query
            result = cursor.fetchone()

            # If the result is None, raise a ValueError
            if result is None:
                raise ValueError(f"Unknown team identifier: {identifier}")

            # Return the team ID
            return result[0]

    @staticmethod
    def get_abbreviation(identifier):
        """
        Get the abbreviation of the team corresponding to the given identifier.

        Args:
            identifier (str): The identifier of the team.

        Returns:
            str: The abbreviation of the team.
        """
        # Get the team ID corresponding to the identifier
        team_id = NBATeamConverter.__get_team_id(identifier)

        # Open a new database connection
        with sqlite3.connect(NBATeamConverter.absolute_db_path) as conn:
            cursor = conn.cursor()

            # Execute the SQL query
            cursor.execute(
                "SELECT abbreviation FROM Teams WHERE team_id = ?", (team_id,)
            )

            # Return the abbreviation of the team
            return cursor.fetchone()[0].upper()

    @staticmethod
    def get_short_name(identifier):
        """
        Get the short name of the team corresponding to the given identifier.

        Args:
            identifier (str): The identifier of the team.

        Returns:
            str: The short name of the team.
        """
        # Get the team ID corresponding to the identifier
        team_id = NBATeamConverter.__get_team_id(identifier)

        # Open a new database connection
        with sqlite3.connect(NBATeamConverter.absolute_db_path) as conn:
            cursor = conn.cursor()

            # Execute the SQL query
            cursor.execute("SELECT short_name FROM Teams WHERE team_id = ?", (team_id,))

            # Return the short name of the team
            return cursor.fetchone()[0].title()

    @staticmethod
    def get_full_name(identifier):
        """
        Get the full name of the team corresponding to the given identifier.

        Args:
            identifier (str): The identifier of the team.

        Returns:
            str: The full name of the team.
        """
        # Get the team ID corresponding to the identifier
        team_id = NBATeamConverter.__get_team_id(identifier)

        # Open a new database connection
        with sqlite3.connect(NBATeamConverter.absolute_db_path) as conn:
            cursor = conn.cursor()

            # Execute the SQL query
            cursor.execute("SELECT full_name FROM Teams WHERE team_id = ?", (team_id,))

            # Return the full name of the team
            return cursor.fetchone()[0].title()
