import os
from datetime import datetime, timedelta

import joblib
import pytz
import torch
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
        "LinearModel": "Ridge_Regression_2024-04-12T01:17:47.777432.joblib",
        "TreeModel": "XGBoost_Regression_2024-04-12T01:18:04.876947.joblib",
        "MLPModel": "MLP_Regression_2024-04-12T01:17:04.677772.pth",
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
                model = joblib.load(model_path)
            elif prediction_engine == "MLPModel":
                model = torch.load(model_path)
        except Exception as e:
            raise ValueError(f"Error loading model: {prediction_engine}. {e}")
    else:
        raise ValueError(
            f"Invalid prediction engine: {prediction_engine}. Please choose from: {list(model_filenames.keys())}"
        )

    return model


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


def process_games_data(games, predictions):
    """
    Process game data for display.

    This function takes a list of game information and a list of predictions, processes them, and returns a list
    of dictionaries with the processed data. The processing includes formatting team names, adding team logo URLs, formatting
    date and time for display, adding current score data, adding condensed play-by-play logs, and adding sorted
    players and predictions if available.

    Args:
        games (list of dict): List of dictionaries with game information.
        predictions (list of dict): List of dictionaries with game predictions.

    Returns:
        list of dict: List of dictionaries with the processed game data.

    Output Game Dictionary Structure:
    {
        "game_id": "game_id",
        "game_date": "YYYY-MM-DD",
        "game_time_est": "HH:MM:SS",
        "home": "Home Team Abbreviation",
        "away": "Away Team Abbreviation",
        "game_status": "In Progress" or "Not Started" or "Completed",
        "home_full_name": "Home Team Name",
        "away_full_name": "Away Team Name",
        "home_team_display": "Home Team Name",
        "away_team_display": "Away Team Name",
        "home_logo_url": "URL",
        "away_logo_url": "URL",
        "home_score": "Home Team Score",
        "away_score": "Away Team Score",
        "datetime_display": "Date - Time" or "Date - Final" or "Time - Period",
        "condensed_pbp": [{time_info, home_score, away_score, description}, ...],
        "pred_home_score": "Predicted Home Score",
        "pred_away_score": "Predicted Away Score",
        "pred_winner": "Predicted Winning Team",
        "pred_win_pct": "Predicted Win Probability",
        "home_players": [{player_id, player_name, player_headshot_url, points, pred_points}, ...],
        "away_players": [{player_id, player_name, player_headshot_url, points, pred_points}, ...],
    }
    """
    # Initialize an empty list to store the processed game data to be returned
    outbound_games = []
    # Reorganize the inbound prediction data into a dictionary for easier access
    if predictions:
        predictions_dict = {pred["game_id"]: pred for pred in predictions}
    else:
        predictions_dict = {}

    for game in games:
        # Get the predictions for the current game
        game_predictions = predictions_dict.get(game["game_id"], {})
        # Initialize game_data with basic game information
        outbound_game_data = {
            "game_id": game["game_id"],
            "game_date": game["game_date"],
            "game_time_est": game["game_time_est"],
            "home": game["home"],
            "away": game["away"],
            "game_status": game["game_status"],
        }

        # Add current score data if available, otherwise set the scores to empty strings
        if game["game_states"]:
            outbound_game_data["home_score"] = game["game_states"][-1]["home_score"]
            outbound_game_data["away_score"] = game["game_states"][-1]["away_score"]
        else:
            outbound_game_data["home_score"] = ""
            outbound_game_data["away_score"] = ""

        # Process team names
        outbound_game_data.update(_process_team_names(game))

        # Generate logo URLs for the home and away teams
        outbound_game_data["home_logo_url"] = generate_logo_url(
            outbound_game_data["home_full_name"]
        )
        outbound_game_data["away_logo_url"] = generate_logo_url(
            outbound_game_data["away_full_name"]
        )

        # Format the date and time display
        outbound_game_data.update(_format_date_time_display(game))

        # Add condensed play-by-play logs if available
        if game["pbp_logs"]:
            outbound_game_data.update(_get_condensed_pbp(game))
        else:
            outbound_game_data["condensed_pbp"] = []

        # Add predicted score and winner if available, otherwise set the values to empty strings
        outbound_game_data["pred_home_score"] = game_predictions.get(
            "pred_home_score", ""
        )
        outbound_game_data["pred_away_score"] = game_predictions.get(
            "pred_away_score", ""
        )
        outbound_game_data["pred_winner"] = game_predictions.get("pred_winner", "")
        outbound_game_data["pred_win_pct"] = game_predictions.get("pred_win_pct", "")

        # Add sorted players and predictions if available
        outbound_game_data.update(_get_sorted_players(game, game_predictions))

        # Append the processed game data to the list
        outbound_games.append(outbound_game_data)

    return outbound_games


def _process_team_names(game_info):
    """
    Format team data for display.

    This function takes a dictionary of game info, retrieves the full team names using the NBATeamConverter,
    and formats the team names for display.

    Args:
        game_info (dict): A dictionary containing game data. It should have a structure including the home and away team abbreviations like:
            {
                "home": "Home Team Abbreviation",
                "away": "Away Team Abbreviation",
            }

    Returns:
        dict: A dictionary containing the full and formatted team names.
    """

    # Retrieve the full team names using the NBATeamConverter
    home_full_name = NBATeamConverter.get_full_name(game_info["home"])
    away_full_name = NBATeamConverter.get_full_name(game_info["away"])

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


def _format_date_time_display(game_info):
    """
    This function formats the date and time display for a game.
    It shows the time and period/quarter/overtime if the game is in progress,
    or the date and start time if the game is not in progress.

    Args:
        game_info (dict): A dictionary containing game data. It should have a structure including the game date and time like:
            {
                "game_date": "YYYY-MM-DD",
                "game_time_est": "HH:MM:SS",
                "game_status": "In Progress" or "Not Started" or "Completed",
                "game_states": [{"period": period, "clock": "PTMMSS.SS", ...}, ...]
            }

    Returns:
        dict: A dictionary containing the formatted date and time display.
    """
    # If the game is in progress and game_states is available
    if game_info["game_status"] == "In Progress" or (
        game_info["game_states"]
        and not game_info["game_states"][-1].get("is_final_state", False)
    ):
        # Get the current period and remaining time
        period = game_info["game_states"][-1]["period"]
        time_remaining = game_info["game_states"][-1]["clock"]

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
    # ISSUE: Unsure of database game timezone. Seems to be EDT.
    game_date_est = datetime.strptime(game_info["game_date"], "%Y-%m-%d")
    game_time_est = datetime.strptime(game_info["game_time_est"], "%H:%M:%S")
    game_date_time_est = datetime.combine(
        game_date_est, game_time_est.time(), tzinfo=pytz.timezone("Etc/GMT+4")
    )
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
    if game_info["game_status"] == "Completed":
        datetime_display = f"{date_display} - Final"
    else:
        # Typically, this will be game status: "Not Started"
        datetime_display = f"{date_display} - {time_display}"

    return {"datetime_display": datetime_display}


def _get_condensed_pbp(game_info):
    """
    This function condenses the play-by-play logs from a game info API response.
    It sorts the logs in reverse order, parses the clock values, and combines the clock and period into a single item.

    Args:
        game_info (dict): A dictionary containing game info data. It should have a structure like:
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
    pbp = sorted(game_info["pbp_logs"], key=lambda x: x["orderNumber"], reverse=True)

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
        if play["period"] > 4:
            time_info = f"{minutes}:{seconds:02} OT{play['period'] - 4}"
        else:
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


def _get_sorted_players(game_info, predictions):
    """
    This function combines player data from the current game state and predictions, assigns a headshot image to each player,
    sorts the players based on their predicted points in descending order, and returns a dictionary with the sorted lists of players
    for both the home and away teams.

    Args:
        game_info (dict): A dictionary containing the current game state. It should have a structure like:
            {
                "game_states": [
                    {
                        "players_data": {
                            "home": {
                                "player_id1": {"player_name": "name1", "points": points1},
                                "player_id2": {"player_name": "name2", "points": points2},
                                ...
                            },
                            "away": {
                                "player_id3": {"player_name": "name3", "points": points3},
                                "player_id4": {"player_name": "name4", "points": points4},
                                ...
                            }
                        }
                    },
                    ...
                ]
            }

        predictions (dict): A dictionary containing player data. It should have a structure like:
            {
                "players": {
                    "home": {
                        "player_id1": {"pred_points": pred_points1},
                        "player_id2": {"pred_points": pred_points2},
                        ...
                    },
                    "away": {
                        "player_id3": {"pred_points": pred_points3},
                        "player_id4": {"pred_points": pred_points4},
                        ...
                    }
                }
            }

    Returns:
        dict: A dictionary containing sorted home and away players. Each player is represented as a dictionary with the following keys:
            - player_id: The player's ID.
            - player_name: The player's name.
            - player_headshot_url: The URL of the player's headshot image.
            - points: The player's points from the current game state.
            - pred_points: The player's predicted points.
    """

    def get_player_image(player_id, files):
        player_image_file = next(
            (f for f in files if f.endswith(f"_{player_id}.png")), None
        )
        if player_image_file:
            return f"static/img/player_images/{player_image_file}"
        else:
            return "static/img/basketball-player.png"

    # Initialize dictionaries to store player data
    players = {"home_players": [], "away_players": []}

    # Get a list of all image files in the directory
    files = os.listdir("web_app/static/img/player_images")

    # Iterate over home and away teams
    for team in ["home", "away"]:
        # Get current game's team player data
        team_players = (
            game_info["game_states"][-1]["players_data"][team]
            if game_info["game_states"]
            else {}
        )
        # Get predicted team player data
        team_predictions = predictions.get("pred_players", {}).get(team, {})

        # Combine all player ids from both sources
        all_player_ids = set(list(team_players.keys()) + list(team_predictions.keys()))

        # Iterate over each player in the team
        for player_id in all_player_ids:
            player_data = team_players.get(player_id, {})
            player_prediction = team_predictions.get(player_id, {})

            # Get player image
            player_headshot_url = get_player_image(player_id, files)

            # Create a dictionary with the player's data
            player = {
                "player_id": player_id,
                "player_name": player_data.get("name", ""),
                "player_headshot_url": player_headshot_url,
                "points": player_data.get("points", 0),
                "pred_points": player_prediction.get("pred_points", 0),
            }

            # Add the player to the appropriate list
            players[f"{team}_players"].append(player)

        # Sort the players by predicted points in descending order
        players[f"{team}_players"] = sorted(
            players[f"{team}_players"], key=lambda x: x["pred_points"], reverse=True
        )

    # Return the sorted lists of players
    return players
