import os
from datetime import datetime, timedelta

import joblib
import pytz
import torch
import xgboost
from dotenv import load_dotenv
from tzlocal import get_localzone

from src.utils import NBATeamConverter

load_dotenv()
PROJECT_ROOT = os.getenv("PROJECT_ROOT")


def load_models(prediction_engine="Random"):
    """
    Loads a prediction model from disk.

    Parameters:
    prediction_engine (str): The prediction engine to use. Defaults to "Random".

    Returns:
    The loaded model, or None if the prediction engine is "Random".
    """
    model_folder = os.path.join(PROJECT_ROOT, "models")

    # Map prediction engine names to model filenames
    model_filenames = {
        "LinearModel": "Ridge Regression_2024-03-14T02:49:44.817782.joblib",
        "TreeModel": "xgboost_model.xgb",
        "MLPModel": "pytorch_model.pt",
        # Add new models here
    }

    if prediction_engine == "Random":
        return None
    elif prediction_engine in model_filenames:
        model_filename = model_filenames[prediction_engine]
        model_path = os.path.join(model_folder, model_filename)
        try:
            if prediction_engine == "LinearModel":
                model = joblib.load(model_path)
            elif prediction_engine == "TreeModel":
                model = xgboost.Booster()
                model.load_model(model_path)
            elif prediction_engine == "MLPModel":
                model = torch.load(model_path)
            return model
        except Exception as e:
            raise ValueError(f"Error loading model: {prediction_engine}. {e}")
    else:
        raise ValueError(
            f"Invalid prediction engine: {prediction_engine}. Please choose from: {list(model_filenames.keys())}"
        )


def get_user_datetime(as_eastern_tz=False):
    """
    This function returns the current date and time in the user's local timezone or in Eastern Time Zone (ET), taking into account daylight saving time.

    Args:
        as_eastern_tz (bool, optional): If True, return the date and time in ET. Otherwise, in the user's local timezone.

    Returns:
        datetime: The current date and time in the specified timezone.
    """
    # Fetch the current UTC time
    utc_now = datetime.now(pytz.utc)

    # If requested in Eastern Time Zone
    if as_eastern_tz:
        eastern_timezone = pytz.timezone("US/Eastern")
        return utc_now.astimezone(eastern_timezone)

    # For user's local timezone
    user_timezone = get_localzone()
    return utc_now.astimezone(user_timezone)


def process_games_data(games_info, predictions_list):
    """
    Process game data for display.

    This function takes a list of game information and a list of predictions, processes them, and returns a list
    of dictionaries with the processed data. The processing includes formatting team names, adding team logo URLs, formatting
    date and time for display, adding current score data, adding condensed play-by-play logs, and adding sorted
    players and predictions if available.

    Args:
        games_info (list of dict): List of dictionaries with game information.
        predictions_list (list of dict): List of dictionaries with game predictions.

    Returns:
        list of dict: List of dictionaries with the processed game data.
    """
    games_data = []
    predictions_dict = {pred["game_id"]: pred for pred in predictions_list}

    for game_info in games_info:
        # Initialize game_data with basic game information
        game_data = {
            "game_id": game_info["game_id"],
            "game_date": game_info["game_date"],
            "game_time_est": game_info["game_time_est"],
            "home": game_info["home"],
            "away": game_info["away"],
            "game_status": game_info["game_status"],
        }

        # Process team names, add team logo URLs, and format date and time for display
        game_data.update(_process_team_names(game_data))
        game_data.update(_add_team_logo_urls(game_data))
        game_data.update(_format_date_time_display(game_data, game_info))

        # Add current score data if available, otherwise set the scores to empty strings
        if game_info["game_states"]:
            game_data["home_score"] = game_info["game_states"][-1]["home_score"]
            game_data["away_score"] = game_info["game_states"][-1]["away_score"]
        else:
            game_data["home_score"] = ""
            game_data["away_score"] = ""

        # Add condensed play-by-play logs if available
        if game_info["pbp_logs"]:
            game_data.update(_get_condensed_pbp(game_info))
        else:
            game_data["condensed_pbp"] = []

        # Add sorted players and predictions if available
        predictions = predictions_dict.get(game_info["game_id"])
        if predictions:
            game_data.update(_get_sorted_players(predictions))
            game_data["predictions"] = predictions

        games_data.append(game_data)

    return games_data


def _process_team_names(game_data):
    """
    Format team data for display.

    This function takes a dictionary of game data, retrieves the full team names using the NBATeamConverter,
    and formats the team names for display.

    Args:
        game_data (dict): A dictionary containing game data. It should have a structure like:
            {
                "home": "Home Team Abbreviation",
                "away": "Away Team Abbreviation",
            }

    Returns:
        dict: The updated game data dictionary with added keys for full team names and formatted team names.
    """

    # Retrieve the full team names using the NBATeamConverter
    home_full_name = NBATeamConverter.get_full_name(game_data["home"])
    away_full_name = NBATeamConverter.get_full_name(game_data["away"])

    def format_team_name(full_name):
        """
        Format a team name for display.

        This function takes a full team name and formats it for display.
        If the team name contains "Trail Blazers", it is split at " Trail ".
        Otherwise, it is split at the last space.

        Args:
            full_name (str): The full name of the team.

        Returns:
            str: The formatted team name.
        """
        # If the team name contains "Trail Blazers", split it at " Trail "
        if "Trail Blazers" in full_name:
            city, team = full_name.split(" Trail ")
            return f"{city}<br>Trail {team}"
        # Otherwise, split the team name at the last space
        else:
            city, team = full_name.rsplit(" ", 1)
            return f"{city}<br>{team}"

    # Format the team names for display
    home_team_display = format_team_name(home_full_name)
    away_team_display = format_team_name(away_full_name)

    # Return the updated game data with the full and formatted team names
    return {
        "home_full_name": home_full_name,
        "away_full_name": away_full_name,
        "home_team_display": home_team_display,
        "away_team_display": away_team_display,
    }


def _add_team_logo_urls(game_data):
    """
    Add team logo URLs to game data.

    This function takes a dictionary of game data, generates the logo URLs for the home and away teams,
    and adds them to the game data dictionary.

    Args:
        game_data (dict): A dictionary containing game data. It should have a structure like:
            {
                "home_full_name": "Home Team Name",
                "away_full_name": "Away Team Name",
            }

    Returns:
        dict: The updated game data dictionary with added keys for home and away team logo URLs.
    """

    def generate_logo_url(team_name):
        """
        Generate a logo URL for a team.

        This function takes a team name, formats it to match the naming convention of the logo files,
        and generates a URL for the logo.

        Args:
            team_name (str): The name of the team.

        Returns:
            str: The URL for the team's logo.
        """
        # Format the team name to match the naming convention of the logo files
        # Example inbound team_name: "Phoenix Suns"
        # Example formatted_team_name: "phoenix-suns"
        formatted_team_name = team_name.lower().replace(" ", "-")

        # Generate the URL for the logo
        # Example url: "static/img/team_logos/nba-phoenix-suns-logo.png"

        logo_url = f"static/img/team_logos/nba-{formatted_team_name}-logo.png"

        return logo_url

    # Generate the logo URLs for the home and away teams
    home_logo_url = generate_logo_url(game_data["home_full_name"])
    away_logo_url = generate_logo_url(game_data["away_full_name"])

    # Return the updated game data with the logo URLs
    return {"home_logo_url": home_logo_url, "away_logo_url": away_logo_url}


def _format_date_time_display(game_data, game_info_api):
    """
    This function formats the date and time display for a game.
    It shows the time and period/quarter/overtime if the game is in progress,
    or the date and start time if the game is not in progress.

    Args:
        game_data (dict): A dictionary containing game data. It should have a structure like:
            {
                "game_status": "In Progress" or "Not Started" or "Completed",
                "game_date": "YYYY-MM-DD",
                "game_time_est": "HH:MM",
            }
        game_info_api (dict): A dictionary containing game info data. It should have a structure like:
            {
                "game_states": [
                    {
                        "period": period,
                        "remaining_time": "PTMMSS.SS",
                    },
                    ...
                ]
            }

    Returns:
        dict: A dictionary containing the formatted date and time display.
    """
    # If the game is in progress and game_states is available
    if (
        game_info_api
        and game_info_api["game_states"]
        and game_data["game_status"] == "In Progress"
    ):
        # Get the current period and remaining time
        period = game_info_api["game_states"][-1]["period"]
        time_remaining = game_info_api["game_states"][-1]["remaining_time"]

        # Parse the remaining time into minutes and seconds
        minutes, seconds = time_remaining.lstrip("PT").rstrip("S").split("M")
        minutes = int(minutes)
        seconds = int(seconds.split(".")[0])

        # Format the remaining time
        time_remaining = f"{minutes}:{seconds:02}"

        # Map the period number to a display string
        period_display_dict = {
            1: "1st Quarter",
            2: "2nd Quarter",
            3: "3rd Quarter",
            4: "4th Quarter",
            5: "Overtime",
            6: "2nd Overtime",
            7: "3rd Overtime",
            8: "4th Overtime",
            9: "5th Overtime",
            10: "Crazy Overtime",
        }
        period_display = period_display_dict[period]

        # Combine the remaining time and period into a single display string
        datetime_display = f"{time_remaining} - {period_display}"
        return {"datetime_display": datetime_display}

    # If the game is not in progress, format the game date and time
    game_date_est = datetime.strptime(game_data["game_date"], "%Y-%m-%d")
    game_time_est = datetime.strptime(game_data["game_time_est"], "%H:%M:%S")
    game_date_time_est = datetime.combine(game_date_est, game_time_est.time())
    user_timezone = get_localzone()
    game_date_time_local = game_date_time_est.astimezone(user_timezone)

    # Determine the display date based on the game date
    game_date = game_date_time_local.date()
    current_date = datetime.now().date()
    next_date = current_date + timedelta(days=1)
    previous_date = current_date - timedelta(days=1)

    if game_date == current_date:
        date_display = "Today"
    elif game_date == next_date:
        date_display = "Tomorrow"
    elif game_date == previous_date:
        date_display = "Yesterday"
    else:
        date_display = game_date.strftime("%b %d")

    # Format the game time
    time_display = game_date_time_local.strftime("%I:%M %p").lstrip("0")

    # Combine the date and time into a single display string
    if game_data["game_status"] == "Not Started" or (
        game_data["game_status"] == "In Progress"
        and (not game_info_api or not game_info_api.get("game_states"))
    ):
        datetime_display = f"{date_display} - {time_display}"
    elif game_data["game_status"] == "Completed":
        datetime_display = f"{date_display} - Final"

    return {"datetime_display": datetime_display}


def _get_condensed_pbp(game_info_api):
    """
    This function condenses the play-by-play logs from a game info API response.
    It sorts the logs in reverse order, parses the clock values, and combines the clock and period into a single item.

    Args:
        game_info_api (dict): A dictionary containing game info data. It should have a structure like:
            {
                "pbp_logs": [
                    {
                        "orderNumber": orderNumber,
                        "clock": "PTMMSS.SS",
                        "period": period,
                        "scoreHome": scoreHome,
                        "scoreAway": scoreAway,
                        "description": "description",
                    },
                    ...
                ]
            }

    Returns:
        dict: A dictionary containing the condensed play-by-play logs.
    """
    # Sort the play-by-play logs in reverse order
    pbp = sorted(
        game_info_api["pbp_logs"], key=lambda x: x["orderNumber"], reverse=True
    )

    # Initialize the list to store the condensed logs
    condensed_pbp = []

    # Iterate over each play in the logs
    for play in pbp:
        # Parse the clock value into minutes and seconds
        time_remaining = play["clock"]
        minutes, seconds = time_remaining.lstrip("PT").rstrip("S").split("M")
        minutes = int(minutes)  # Remove leading zero from minutes
        seconds = int(seconds.split(".")[0])  # Only take the whole seconds

        # Combine the clock and period into a single item
        time_info = f"{minutes}:{seconds:02} Q{play['period']}"

        # Add the play to condensed_pbp with the new time_info key
        condensed_pbp.append(
            {
                "time_info": time_info,
                "home_score": play["scoreHome"],
                "away_score": play["scoreAway"],
                "description": play["description"],
            }
        )

    # Return the condensed play-by-play logs
    return {"condensed_pbp": condensed_pbp}


def _get_sorted_players(predictions):
    """
    This function sorts players based on their predicted points in descending order.
    It also assigns a headshot image to each player.

    Args:
        predictions (dict): A dictionary containing player data. It should have a structure like:
            {
                "players": {
                    "home": {
                        "player_id1": {"name": "name1", "points": points1},
                        "player_id2": {"name": "name2", "points": points2},
                        ...
                    },
                    "away": {
                        "player_id3": {"name": "name3", "points": points3},
                        "player_id4": {"name": "name4", "points": points4},
                        ...
                    }
                }
            }

    Returns:
        dict: A dictionary containing sorted home and away players.
    """
    # Initialize lists to store player data
    home_players = []
    away_players = []

    # If predictions is None or doesn't contain player data, return empty lists
    if predictions is None or not predictions.get("players"):
        return {"home_players": home_players, "away_players": away_players}

    # Get a list of all image files in the directory
    files = os.listdir("web_app/static/img/player_images")

    # Iterate over home and away teams
    for team in ["home", "away"]:
        # Iterate over each player in the team
        for player_id, player_data in predictions["players"][team].items():
            # Find the image file that corresponds to the player
            player_image_file = next(
                (f for f in files if f.endswith(f"_{player_id}.png")), None
            )

            # If an image file was found, use it. Otherwise, use a default image
            if player_image_file:
                player_headshot_url = f"static/img/player_images/{player_image_file}"
            else:
                player_headshot_url = "static/img/basketball-player.png"

            # Create a dictionary with the player's data
            player = {
                "player_id": player_id,
                "name": player_data["name"],
                "player_headshot_url": player_headshot_url,
                "pred_points": player_data["points"],
            }

            # Add the player to the appropriate list
            if team == "home":
                home_players.append(player)
            else:
                away_players.append(player)

    # Sort the players by predicted points in descending order
    home_players = sorted(home_players, key=lambda x: x["pred_points"], reverse=True)
    away_players = sorted(away_players, key=lambda x: x["pred_points"], reverse=True)

    # Return the sorted lists of players
    return {"home_players": home_players, "away_players": away_players}
