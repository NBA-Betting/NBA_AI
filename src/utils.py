import logging
import os
import re
import sqlite3

import requests
from dotenv import load_dotenv

load_dotenv()
PROJECT_ROOT = os.getenv("PROJECT_ROOT")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s",
)


def lookup_basic_game_info(game_ids, db_path):
    """
    This function looks up basic game information given a game_id or a list of game_ids from the Games table in the SQLite database.

    Parameters:
    game_ids (str or list): The ID of the game or a list of game IDs to look up.
    db_path (str): The path to the SQLite database.

    Returns:
    list: A list of dictionaries, each representing a game. Each dictionary contains the game ID, home team, away team, date/time, status, season, and season type.
    """

    # Ensure game_ids is a list
    if not isinstance(game_ids, list):
        game_ids = [game_ids]

    # Validate the game_ids
    validate_game_ids(game_ids)

    # Prepare the SQL statement to fetch the games
    sql = f"""
    SELECT game_id, home_team, away_team, date_time_est, status, season, season_type
    FROM Games
    WHERE game_id IN ({','.join(['?']*len(game_ids))})
    """

    # Use a context manager to handle the database connection
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()

        # Execute the SQL statement with the game_ids
        cursor.execute(sql, game_ids)

        # Fetch the games from the database
        games = cursor.fetchall()

    # Create a set of all game_ids
    game_ids_set = set(game_ids)

    # Process each game
    game_info_list = []
    for game_id, home, away, date_time_est, status, season, season_type in games:
        # Remove the game_id from the set
        game_ids_set.remove(game_id)

        # Add the game information to the list
        game_info_list.append(
            {
                "game_id": game_id,
                "home_team": home,
                "away_team": away,
                "date_time_est": date_time_est,
                "status": status,
                "season": season,
                "season_type": season_type,
            }
        )

    # Log any game_ids that were not found
    if game_ids_set:
        logging.warning(f"Game IDs not found in the database: {game_ids_set}")

    # Return the game information
    return game_info_list


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


def update_scheduled_games(season, db_path):
    """
    Fetches the NBA schedule for a given season and updates the Games database.
    This function performs an UPSERT operation (UPDATE or INSERT).

    Parameters:
    season (str): The season to fetch the schedule for, formatted as 'XXXX-XXXX' (e.g., '2020-2021').
    db_path (str): The path to the SQLite database file.
    """
    # Validate the format of the season string
    validate_season_format(season, abbreviated=False)
    # Convert the season string to the format used by the API
    api_season = season[:5] + season[-2:]

    # Define the URL for the API endpoint, including the season
    endpoint = (
        f"https://stats.nba.com/stats/scheduleleaguev2?Season={api_season}&LeagueID=00"
    )

    # Define the headers to be sent with the request
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3",
        "Referer": "https://stats.nba.com/schedule/",
        "Accept-Language": "en-US,en;q=0.9",
    }

    # Try to send the request and get the response
    try:
        response = requests.get(endpoint, headers=headers)
        # If the response indicates an error, raise an exception
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        logging.error(f"Error occurred while fetching the schedule for {season}.")
        raise e

    # Parse the JSON response to get the game dates
    game_dates = response.json()["leagueSchedule"]["gameDates"]

    # Extract all games from the response
    all_games = [game for date in game_dates for game in date["games"]]

    # Define the keys to be kept in the game dictionaries
    keys_needed = [
        "gameId",
        "gameStatus",
        "gameDateTimeEst",
        "homeTeam",
        "awayTeam",
    ]

    # Filter the game dictionaries to only include the needed keys
    all_games = [{key: game[key] for key in keys_needed} for game in all_games]

    # Replace the homeTeam and awayTeam dictionaries with their teamTricode values
    for game in all_games:
        game["homeTeam"] = game["homeTeam"]["teamTricode"]
        game["awayTeam"] = game["awayTeam"]["teamTricode"]

    # Define the season type codes
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

    # Use a context manager to handle the database connection
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()

        # Update or insert the games in the database
        for game in all_games:
            # Determine the season type based on the game ID
            season_type = season_type_codes.get(game["gameId"][:3], "Unknown")
            status = game_status_codes.get(game["gameStatus"], "Unknown")

            # Prepare the SQL statement for UPSERT operation
            sql = """
            INSERT OR REPLACE INTO Games
            (game_id, date_time_est, home_team, away_team, status, season, season_type)
            VALUES (:gameId, :gameDateTimeEst, :homeTeam, :awayTeam, :gameStatus, :season, :seasonType)
            """

            # Execute the SQL statement with the game data
            cursor.execute(
                sql,
                {
                    "gameId": game["gameId"],
                    "gameDateTimeEst": game["gameDateTimeEst"],
                    "homeTeam": game["homeTeam"],
                    "awayTeam": game["awayTeam"],
                    "gameStatus": status,
                    "season": season,
                    "seasonType": season_type,
                },
            )

        # Commit the changes to the database
        conn.commit()


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

    # Path to the SQLite database file
    DATABASE_PATH = os.path.join(PROJECT_ROOT, "data", "NBA_AI.sqlite")

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
        with sqlite3.connect(NBATeamConverter.DATABASE_PATH) as conn:
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
        with sqlite3.connect(NBATeamConverter.DATABASE_PATH) as conn:
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
        with sqlite3.connect(NBATeamConverter.DATABASE_PATH) as conn:
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
        with sqlite3.connect(NBATeamConverter.DATABASE_PATH) as conn:
            cursor = conn.cursor()

            # Execute the SQL query
            cursor.execute("SELECT full_name FROM Teams WHERE team_id = ?", (team_id,))

            # Return the full name of the team
            return cursor.fetchone()[0].title()
