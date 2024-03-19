import json
import os
import re

import pandas as pd
import requests
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()
PROJECT_ROOT = os.getenv("PROJECT_ROOT")


def lookup_basic_game_info(game_id):
    """
    This function looks up basic game information given a game_id.

    Parameters:
    game_id (str): The ID of the game to look up.

    Returns:
    dict: A dictionary containing the game_id, home team, away team, game date, game time, and game status.

    Raises:
    ValueError: If multiple games are found for the given game_id or if no game is found for the given game_id.
    """

    # Validate the game_id
    validate_game_id(game_id)

    # Convert the game_id to a season string
    season = game_id_to_season(game_id, abbreviate=True)

    # Get the schedule for the extracted season
    schedule = get_schedule(season)

    # Find the game in the schedule using the game_id
    game = [g for g in schedule if g["gameId"] == game_id]

    # Ensure a single game is returned
    if len(game) > 1:
        raise ValueError(f"""Multiple games found for Game ID {game_id}.""")

    # Raise an error if the game is not found
    if not game:
        raise ValueError(f"""Game ID {game_id} not found in the schedule.""")

    # Extract the first (and only) game from the list
    game = game[0]

    # Extract the home team, away team, game date, and game time from the game dictionary
    home = game["homeTeam"]
    away = game["awayTeam"]
    game_date = game["gameDateTimeEst"][:10]
    game_time_est = game["gameDateTimeEst"][11:19]
    game_status_id = game["gameStatus"]

    # Return the game information as a dictionary
    return {
        "game_id": game_id,
        "home": home,
        "away": away,
        "game_date": game_date,
        "game_time_est": game_time_est,
        "game_status_id": game_status_id,
    }


def get_games_for_date(date, season="2023-24"):
    """
    Fetches the NBA games for a given date and season.

    Parameters:
    date (str): The date to fetch the games for, formatted as 'YYYY-MM-DD'.
    season (str): The season to fetch the games for, formatted as 'XXXX-XX'.

    Returns:
    list: A list of dictionaries, each representing a game. Each dictionary contains the game ID, status, date/time, and the home and away teams.
    """
    # Validate the date and season formats
    validate_date_format(date)
    validate_season_format(season, abbreviated=True)

    # Fetch the schedule for the entire season
    season_schedule = get_schedule(season)

    # Filter the schedule to only include games on the specified date
    # Note: This assumes that game["gameDateTimeEst"] is a string starting with the date in 'YYYY-MM-DD' format
    games_on_date = [
        game for game in season_schedule if game["gameDateTimeEst"].startswith(date)
    ]

    return games_on_date


def get_schedule(season, season_type="All"):
    """
    Fetches the NBA schedule for a given season.

    Parameters:
    season (str): The season to fetch the schedule for, formatted as 'XXXX-XX' (e.g., '2020-21').
    season_type (str): The type of season to fetch the schedule for. Defaults to 'All'.

    Returns:
    list: A list of dictionaries, each representing a game. Each dictionary contains the game ID, status, date/time, and the home and away teams.
    """
    # Check if the season format is correct
    validate_season_format(season, abbreviated=True)

    season_type_codes = {
        "Pre Season": "001",
        "Regular Season": "002",
        "All-Star": "003",
        "Post Season": "004",
    }
    if season_type not in [
        "Pre Season",
        "Regular Season",
        "All-Star",
        "Post Season",
        "All",
    ]:
        raise ValueError(
            "Invalid season type. Please use one of 'Pre Season', 'Regular Season', 'All-Star', 'Post Season', or 'All'."
        )

    # Define the endpoint URL, including the season
    endpoint = (
        f"https://stats.nba.com/stats/scheduleleaguev2?Season={season}&LeagueID=00"
    )

    # Define the headers to be sent with the request
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3",
        "Referer": "https://stats.nba.com/schedule/",
        "Accept-Language": "en-US,en;q=0.9",
    }

    try:
        # Send the request and get the response
        response = requests.get(endpoint, headers=headers)
        response.raise_for_status()  # Raise an exception if the response indicates an error
    except requests.exceptions.RequestException as e:
        print(f"Error occurred: {e}")
        raise

    # Parse the JSON response
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

    # Filter for season type
    if season_type != "All":
        season_code = season_type_codes[season_type]
        all_games = [game for game in all_games if game["gameId"][:3] == season_code]

    return all_games


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
    validate_game_id(game_id)

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


def validate_game_id(game_id):
    """
    Validates a game ID.

    The game ID must be a 10-character string that starts with '00'.

    Args:
        game_id (str): The game ID to validate.

    Raises:
        ValueError: If the game ID is not valid.
    """
    if not (
        game_id
        and isinstance(game_id, str)
        and len(game_id) == 10
        and game_id.startswith("00")
    ):
        raise ValueError(
            """Invalid game ID.
            Game ID must be a 10-digit string starting with '00'.
            Example: '0022100001'. 
            Offical NBA.com Game ID"""
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

    __lookup_dict = None

    @classmethod
    def __generate_lookup_dict(cls):
        """
        Generate a lookup dictionary from teams_data. The dictionary maps each identifier
        (team ID, abbreviation, full name, short name, and alternatives) to the corresponding team ID.
        """
        lookup_dict = {}
        for team_id, details in cls.teams_data.items():
            # Directly map the team's ID
            lookup_dict[team_id] = team_id

            # Map other identifiers, normalized to lowercase
            identifiers = [
                details["abbreviation"],
                details["full_name"],
                details["short_name"],
            ] + details["alternatives"]

            for identifier in identifiers:
                normalized_identifier = identifier.lower().replace(" ", "-")
                lookup_dict[normalized_identifier] = team_id
        return lookup_dict

    @classmethod
    def __get_team_id(cls, identifier):
        """
        Get the team ID corresponding to the given identifier.
        If the identifier is unknown, raise a ValueError.
        """
        identifier_normalized = str(identifier).lower().replace(" ", "-")
        if cls.__lookup_dict is None:
            cls.__lookup_dict = cls.__generate_lookup_dict()
        if identifier_normalized not in cls.__lookup_dict:
            raise ValueError(f"Unknown team identifier: {identifier}")
        return cls.__lookup_dict[identifier_normalized]

    @classmethod
    def get_abbreviation(cls, identifier):
        """
        Get the abbreviation of the team corresponding to the given identifier.
        """
        team_id = cls.__get_team_id(identifier)
        return cls.teams_data[team_id]["abbreviation"].upper()

    @classmethod
    def get_short_name(cls, identifier):
        """
        Get the short name of the team corresponding to the given identifier.
        """
        team_id = cls.__get_team_id(identifier)
        return cls.teams_data[team_id]["short_name"].title()

    @classmethod
    def get_full_name(cls, identifier):
        """
        Get the full name of the team corresponding to the given identifier.
        """
        team_id = cls.__get_team_id(identifier)
        return cls.teams_data[team_id]["full_name"].title()

    # Team data with details for each team. Including abbreviation, full name, short name, and alternatives.
    teams_data = {
        "1610612737": {
            "abbreviation": "ATL",
            "full_name": "Atlanta Hawks",
            "short_name": "Hawks",
            "alternatives": [
                "STL",
                "Tri-Cities Blackhawks",
                "MLH",
                "TRI",
                "Milwaukee Hawks",
                "St. Louis Hawks",
            ],
        },
        "1610612751": {
            "abbreviation": "BKN",
            "full_name": "Brooklyn Nets",
            "short_name": "Nets",
            "alternatives": [
                "NJA",
                "BK",
                "NYN",
                "NJN",
                "New York Nets",
                "BRK",
                "New Jersey Americans",
                "New Jersey Nets",
            ],
        },
        "1610612738": {
            "abbreviation": "BOS",
            "full_name": "Boston Celtics",
            "short_name": "Celtics",
            "alternatives": [],
        },
        "1610612766": {
            "abbreviation": "CHA",
            "full_name": "Charlotte Hornets",
            "short_name": "Hornets",
            "alternatives": [
                "Charlotte Bobcats",
                "CHH",
                "CHO",
            ],
        },
        "1610612739": {
            "abbreviation": "CLE",
            "full_name": "Cleveland Cavaliers",
            "short_name": "Cavaliers",
            "alternatives": [],
        },
        "1610612741": {
            "abbreviation": "CHI",
            "full_name": "Chicago Bulls",
            "short_name": "Bulls",
            "alternatives": [],
        },
        "1610612742": {
            "abbreviation": "DAL",
            "full_name": "Dallas Mavericks",
            "short_name": "Mavericks",
            "alternatives": [],
        },
        "1610612743": {
            "abbreviation": "DEN",
            "full_name": "Denver Nuggets",
            "short_name": "Nuggets",
            "alternatives": ["Denver Rockets", "DNR"],
        },
        "1610612765": {
            "abbreviation": "DET",
            "full_name": "Detroit Pistons",
            "short_name": "Pistons",
            "alternatives": ["FTW", "Fort Wayne Pistons"],
        },
        "1610612744": {
            "abbreviation": "GSW",
            "full_name": "Golden State Warriors",
            "short_name": "Warriors",
            "alternatives": [
                "PHW",
                "GS",
                "Philadelphia Warriors",
                "San Francisco Warriors",
                "SFW",
            ],
        },
        "1610612745": {
            "abbreviation": "HOU",
            "full_name": "Houston Rockets",
            "short_name": "Rockets",
            "alternatives": ["SDR", "San Diego Rockets"],
        },
        "1610612754": {
            "abbreviation": "IND",
            "full_name": "Indiana Pacers",
            "short_name": "Pacers",
            "alternatives": [],
        },
        "1610612746": {
            "abbreviation": "LAC",
            "full_name": "Los Angeles Clippers",
            "short_name": "Clippers",
            "alternatives": [
                "SDC",
                "Buffalo Braves",
                "San Diego Clippers",
                "LA Clippers",
                "BUF",
            ],
        },
        "1610612747": {
            "abbreviation": "LAL",
            "full_name": "Los Angeles Lakers",
            "short_name": "Lakers",
            "alternatives": ["MNL", "Minneapolis Lakers"],
        },
        "1610612763": {
            "abbreviation": "MEM",
            "full_name": "Memphis Grizzlies",
            "short_name": "Grizzlies",
            "alternatives": ["VAN", "Vancouver Grizzlies"],
        },
        "1610612748": {
            "abbreviation": "MIA",
            "full_name": "Miami Heat",
            "short_name": "Heat",
            "alternatives": [],
        },
        "1610612749": {
            "abbreviation": "MIL",
            "full_name": "Milwaukee Bucks",
            "short_name": "Bucks",
            "alternatives": [],
        },
        "1610612750": {
            "abbreviation": "MIN",
            "full_name": "Minnesota Timberwolves",
            "short_name": "Timberwolves",
            "alternatives": [],
        },
        "1610612752": {
            "abbreviation": "NYK",
            "full_name": "New York Knicks",
            "short_name": "Knicks",
            "alternatives": ["NY"],
        },
        "1610612740": {
            "abbreviation": "NOP",
            "full_name": "New Orleans Pelicans",
            "short_name": "Pelicans",
            "alternatives": [
                "NOH",
                "New Orleans Hornets",
                "NOK",
                "NO",
                "New Orleans/Oklahoma City Hornets",
            ],
        },
        "1610612760": {
            "abbreviation": "OKC",
            "full_name": "Oklahoma City Thunder",
            "short_name": "Thunder",
            "alternatives": ["Seattle SuperSonics", "SEA"],
        },
        "1610612753": {
            "abbreviation": "ORL",
            "full_name": "Orlando Magic",
            "short_name": "Magic",
            "alternatives": [],
        },
        "1610612755": {
            "abbreviation": "PHI",
            "full_name": "Philadelphia 76ers",
            "short_name": "76ers",
            "alternatives": [
                "PHL",
                "Syracuse Nationals",
                "SYR",
            ],
        },
        "1610612756": {
            "abbreviation": "PHX",
            "full_name": "Phoenix Suns",
            "short_name": "Suns",
            "alternatives": ["PHO"],
        },
        "1610612757": {
            "abbreviation": "POR",
            "full_name": "Portland Trail Blazers",
            "short_name": "Trail Blazers",
            "alternatives": [],
        },
        "1610612759": {
            "abbreviation": "SAS",
            "full_name": "San Antonio Spurs",
            "short_name": "Spurs",
            "alternatives": [
                "DLC",
                "SAN",
                "Dallas Chaparrals",
                "SA",
            ],
        },
        "1610612758": {
            "abbreviation": "SAC",
            "full_name": "Sacramento Kings",
            "short_name": "Kings",
            "alternatives": [
                "KCO",
                "KCK",
                "Kansas City-Omaha Kings",
                "Kansas City Kings",
                "Cincinnati Royals",
                "CIN",
                "Rochester Royals",
                "ROC",
            ],
        },
        "1610612761": {
            "abbreviation": "TOR",
            "full_name": "Toronto Raptors",
            "short_name": "Raptors",
            "alternatives": [],
        },
        "1610612762": {
            "abbreviation": "UTA",
            "full_name": "Utah Jazz",
            "short_name": "Jazz",
            "alternatives": ["NOJ", "New Orleans Jazz", "UTAH"],
        },
        "1610612764": {
            "abbreviation": "WAS",
            "full_name": "Washington Wizards",
            "short_name": "Wizards",
            "alternatives": [
                "WSH",
                "Washington Bullets",
                "CAP",
                "BAL",
                "Baltimore Bullets",
                "CHP",
                "CHZ",
                "Chicago Packers",
                "WSB",
                "Capital Bullets",
                "Chicago Zephyrs",
            ],
        },
    }


# --------------- MODELING UTILS -----------------


def load_featurized_modeling_data(seasons):
    """
    This function loads featurized modeling data for a list of NBA seasons.

    Parameters:
    seasons (list): A list of seasons for which to load the featurized modeling data.

    Returns:
    pd.DataFrame: A DataFrame of the featurized modeling data for the given seasons.
    """
    # Initialize an empty list to store the DataFrames
    dfs = []

    # Iterate over all seasons in the list
    for season in seasons:
        # Load the CSV file for the current season into a DataFrame and append it to the list
        dfs.append(pd.read_csv(f"{PROJECT_ROOT}/data/featurized_NBAStats/{season}.csv"))

    # Concatenate all DataFrames in the list into a single DataFrame
    return pd.concat(dfs)


def create_featurized_training_data_csv(season):
    """
    This function creates a CSV file of featurized data for a given NBA season.
    Saves time loading in data from core data files for basic featurized modeling.

    Parameters:
    season (str): The season for which to create the featurized data CSV.

    Returns:
    pd.DataFrame: A DataFrame of the featurized data for the given season.
    """
    # Validate the season format
    validate_season_format(season, abbreviated=False)

    # Define the path to the season folder
    season_folder = f"{PROJECT_ROOT}/data/NBAStats/{season}"

    # Initialize an empty list to store the season data
    season_data = []

    # Get only the directories from the season folder
    date_folders = [entry for entry in os.scandir(season_folder) if entry.is_dir()]

    # Iterate over all date subfolders in the season folder
    for date_folder in tqdm(
        date_folders,
        total=len(date_folders),
        desc="Loading season data",
        dynamic_ncols=True,
    ):
        # Iterate over all JSON files in the date folder
        for game_file in os.scandir(date_folder.path):
            try:
                # Open and load the JSON file
                with open(game_file.path, "r") as f:
                    game = json.load(f)

                    # Check if the game has a feature set
                    if game["prior_states"]["feature_set"]:
                        # Extract the game ID, date, feature set, and home margin
                        game_id = game["game_id"]
                        game_date = game["game_date"]
                        feature_set = game["prior_states"]["feature_set"]
                        home_score = game["final_state"]["home_score"]
                        away_score = game["final_state"]["away_score"]
                        home_margin = game["final_state"]["home_margin"]
                        total_score = home_score + away_score

                        # Append the extracted data to the season data list
                        season_data.append(
                            {
                                "game_id": game_id,
                                "game_date": game_date,
                                "home_score": home_score,
                                "away_score": away_score,
                                "home_margin": home_margin,
                                "total_score": total_score,
                                **feature_set,
                            }
                        )
            except Exception as e:
                # Print an error message if an exception occurs
                print(f"Error processing file: {game_file.path}")
                print(f"Error: {str(e)}")

    # Convert the list of dictionaries to a DataFrame
    season_data = pd.DataFrame(season_data)

    # Write the DataFrame to a CSV file
    season_data.to_csv(
        f"{PROJECT_ROOT}/data/featurized_NBAStats/{season}.csv", index=False
    )

    # Return the DataFrame
    return season_data


if __name__ == "__main__":
    seasons = ["2020-2021", "2021-2022", "2022-2023", "2023-2024"]

    for season in seasons:
        create_featurized_training_data_csv(season)
        print(f"Created featurized training data for {season}.")
