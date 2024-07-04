import logging
import os
import re
import sqlite3

import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

from src.config import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s",
)


def requests_retry_session(
    retries=3,
    backoff_factor=0.3,
    status_forcelist=(500, 502, 504),
    session=None,
    timeout=10,
):
    """
    Creates a session with retry logic for handling transient HTTP errors.

    Parameters:
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


def print_game_info(game_info):
    are_game_states_finalized = any(
        state["is_final_state"] for state in reversed(game_info["game_states"])
    )
    print()
    print("-" * 50)
    print("Game ID:", game_info["game_id"])
    print("Game Date:", game_info["game_date"])
    print("Game Time (EST):", game_info["game_time_est"])
    print("Home Team:", game_info["home"])
    print("Away Team:", game_info["away"])
    print("Game Status:", game_info["game_status"])
    print("Play-by-Play Log Count:", len(game_info["pbp_logs"]))
    print("Game States Count:", len(game_info["game_states"]))
    print("Are Game States Finalized:", are_game_states_finalized)
    if game_info.get("prior_states"):
        if game_info["prior_states"]:
            print(
                f"Prior States Count: Home-{len(game_info['prior_states']['home_prior_states'])} Away-{len(game_info['prior_states']['away_prior_states'])}"
            )
            print(
                "Are Prior States Finalized:",
                game_info["prior_states"]["are_prior_states_finalized"],
            )
    if game_info.get("game_states"):
        if game_info["game_states"]:
            most_recent_state = game_info["game_states"][-1]
            print("Most Recent State:")
            print("  Remaining Time:", most_recent_state["clock"])
            print("  Period:", most_recent_state["period"])
            print("  Home Score:", most_recent_state["home_score"])
            print("  Away Score:", most_recent_state["away_score"])
            print("  Total Score:", most_recent_state["total"])
            print("  Home Margin:", most_recent_state["home_margin"])
            print("Players:")
            for team, players in most_recent_state["players_data"].items():
                print(f"  {team.title()} Team:")
                sorted_players = sorted(
                    players.items(), key=lambda x: x[1]["points"], reverse=True
                )
                for player_id, player_info in sorted_players:
                    print(f"    {player_info['name']}: {player_info['points']} points")
        else:
            print("No game states available.")
    print("-" * 50)
    print()


def get_games_for_date(date, db_path):
    """
    Fetches the NBA games for a given date from the Games table in the SQLite database.

    Parameters:
    date (str): The date to fetch the games for, formatted as 'YYYY-MM-DD'.
    db_path (str): The path to the SQLite database.

    Returns:
    list: A list of dictionaries, each representing a game. Each dictionary contains the game ID, status, date/time, and the home and away teams.
    """

    # Validate the date format
    validate_date_format(date)

    # Prepare the SQL statement to fetch the games
    sql = """
    SELECT game_id, home_team, away_team, date_time_est, status
    FROM Games
    WHERE date(date_time_est) = :date
    """

    # Use a context manager to handle the database connection
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()

        # Execute the SQL statement with the date
        cursor.execute(sql, {"date": date})

        # Fetch the games from the database
        games = cursor.fetchall()

    # Convert the games to a list of dictionaries
    games_on_date = [
        {
            "game_id": game[0],
            "home_team": game[1],
            "away_team": game[2],
            "date_time_est": game[3],
            "status": game[4],
        }
        for game in games
    ]

    return games_on_date


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
                f"Invalid game ID {game_id}. Game ID must be a 10-digit string starting with '00'. Example: '0022100001'. Offical NBA.com Game ID"
            )

    if invalid_game_ids:
        raise ValueError(
            f"Invalid game IDs: {invalid_game_ids}. Each game ID must be a 10-digit string starting with '00'. Example: '0022100001'. Offical NBA.com Game ID"
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

    Parameters:
    season (str): The season string to validate, formatted as 'XXXX-XX' or 'XXXX-XXXX'.
    abbreviated (bool): Whether the second year in the season string is abbreviated.

    Raises:
    ValueError: If the season string does not match the required format or if the second year does not logically follow the first year.
    """
    # Define the regex pattern based on abbreviated flag
    pattern = r"^(\d{4})-(\d{2})$" if abbreviated else r"^(\d{4})-(\d{4})$"

    # Attempt to match the pattern to the season string
    match = re.match(pattern, season)
    if not match:
        raise ValueError("Season does not match the required format.")

    year1, year2_suffix = map(int, match.groups())

    # Handle the year2 based on whether it's abbreviated or not
    year2 = year2_suffix if not abbreviated else year1 // 100 * 100 + year2_suffix

    # Check if year2 logically follows year1
    if year1 + 1 != year2:
        raise ValueError("Second year does not logically follow the first year.")


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
