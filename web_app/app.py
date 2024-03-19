import logging
import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytz
from dotenv import load_dotenv
from flask import Flask, flash, jsonify, render_template, request
from tzlocal import get_localzone

# Add the parent directory to sys.path to allow imports from there
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from src.NBAStats_game_info import get_game_info
from src.predictions import get_predictions
from src.utils import NBATeamConverter, get_games_for_date, validate_date_format

load_dotenv()
PROJECT_ROOT = os.getenv("PROJECT_ROOT")
WEB_APP_SECRET_KEY = os.getenv("WEB_APP_SECRET_KEY")

app = Flask(__name__)
app.secret_key = WEB_APP_SECRET_KEY


@app.route("/")
def home():
    """
    Home route handler. It handles the display of games based on the date provided in the query parameters.
    If no date is provided, it defaults to the current date. If an invalid date is provided, it flashes an error message
    and defaults to the current date. It also calculates the previous and next dates for navigation links.
    """

    # Get the current date in Eastern Time Zone
    current_date_et = get_user_datetime(as_eastern_tz=True)
    # Format the current date as a string
    current_date_str = current_date_et.strftime("%Y-%m-%d")
    # Get the date from the query parameters, default to the current date if not provided
    query_date_str = request.args.get("date", current_date_str)

    # Attempt to validate the date provided in the query parameters
    try:
        validate_date_format(query_date_str)
        # If the date is valid, parse it into a datetime object
        query_date = datetime.strptime(query_date_str, "%Y-%m-%d")
    except Exception as e:
        # If the date is invalid, log a warning with the invalid date and the error
        logging.warning(
            f"Date validation error for input date string: {query_date_str}. Error: {e}"
        )
        # Flash an error message to the user
        flash("Invalid date format. Showing games for today.", "error")
        # Default to the current date
        query_date_str = current_date_str
        query_date = current_date_et

    # Format the query date as a string for display
    query_date_display_str = query_date.strftime("%b %d")

    # Calculate the previous and next dates for the navigation links
    next_date = query_date + timedelta(days=1)
    prev_date = query_date - timedelta(days=1)
    # Format the previous and next dates as strings
    next_date_str = next_date.strftime("%Y-%m-%d")
    prev_date_str = prev_date.strftime("%Y-%m-%d")

    # Render the index.html template, passing the necessary data for initial page setup
    return render_template(
        "index.html",
        query_date_str=query_date_str,
        query_date_display_str=query_date_display_str,
        prev_date=prev_date_str,
        next_date=next_date_str,
    )


@app.route("/get-games")
def get_games():
    """
    Fetch and process games for a given date.

    This function fetches games for a given date, gets game info and predictions for each game,
    and processes the game data. If any of these steps fail, it logs an error and includes an error
    message in the response.

    The date is provided as a 'date' parameter in the request. If no date is provided, it defaults
    to the current date.

    Returns:
        A JSON response containing the processed game data for each game.
    """
    # Get the current date in Eastern Time Zone
    current_date_est = get_user_datetime(as_eastern_tz=True)

    # Fetch the date parameter from the request, default to current date if not provided
    inbound_query_date_str = request.args.get("date")
    if inbound_query_date_str is None or inbound_query_date_str == "":
        query_date_str = current_date_est.strftime("%Y-%m-%d")
    else:
        query_date_str = inbound_query_date_str

    # Before loading games, validate the query date format
    try:
        validate_date_format(query_date_str)
    except Exception as e:  # Adjust to catch specific exceptions as needed
        # Log an error if the date format is invalid
        logging.error(f"Invalid date format provided for get-games: {e}")
        return jsonify({"error": "Invalid date format provided"}), 400

    # Load Games
    try:
        scheduled_games = get_games_for_date(query_date_str)
    except Exception as e:
        # Log an error if loading games fails
        logging.error(f"Error loading games for date {query_date_str}: {e}")
        return jsonify({"error": f"Error loading games for date {query_date_str}"}), 500

    #!!!!!! RESTRICTED TO 2 GAMES FOR TESTING
    scheduled_games = scheduled_games[0:2]

    # Initialize an empty list to store the processed game data
    outbound_game_data = []

    # Process each game in the scheduled games
    for game_info_schedule in scheduled_games:
        # Try to get the game info
        try:
            game_info_api = get_game_info(
                game_info_schedule["gameId"],
                include_prior_states=False,
                save_to_database=False,
                force_update_prior_states=False,
            )
        except Exception as e:
            # Log an error if getting the game info fails
            logging.error(
                f"Error getting game info for gameId {game_info_schedule['gameId']}: {e}"
            )
            game_info_api = None

        # Try to get predictions
        try:
            predictions = get_predictions(game_info_api, "random")
        except Exception as e:
            # Log an error if making predictions fails
            logging.error(
                f"Error making predictions for gameId {game_info_schedule['gameId']}: {e}"
            )
            predictions = None

        # Try to process the game data
        try:
            processed_game_data = process_game_data(
                game_info_api, predictions, game_info_schedule
            )
            # Add the processed game data to the outbound game data
            outbound_game_data.append(processed_game_data)
        except Exception as e:
            # Log an error if processing the game data fails
            logging.error(
                f"Error processing game data for gameId {game_info_schedule['gameId']}: {e}"
            )
            # Construct an error message using keys from the schedule
            error_message = {
                "gameId": game_info_schedule["gameId"],
                "homeTeam": game_info_schedule["homeTeam"],
                "awayTeam": game_info_schedule["awayTeam"],
                "errorMessage": f"Error creating game data for {game_info_schedule['gameId']} - {game_info_schedule['awayTeam']}@{game_info_schedule['homeTeam']}.",
            }
            # Add the error message to the outbound game data
            outbound_game_data.append(error_message)

    # Return the outbound game data as a JSON response
    return jsonify(outbound_game_data)


@app.route("/game-details/<game_id>")
def game_details(game_id):
    """
    Endpoint to get game details.

    This function is mapped to the "/game-details/<game_id>" route. It loads game data, creates predictions,
    processes the game data, and returns it as a JSON response.

    Args:
        game_id (str): The ID of the game.

    Returns:
        Response: A response object containing the game data in JSON format.
    """
    # Load game data using the get_game_info function
    # The game data is not saved to the database and prior states are not included or updated
    game_info_api = get_game_info(
        game_id,
        include_prior_states=False,
        save_to_database=False,
        force_update_prior_states=False,
    )

    # Create predictions for the game using the get_predictions function
    # The predictions are created using a "random" model
    predictions = get_predictions(game_info_api, "random")

    # Process the game data using the process_game_data function
    # The processed game data includes the game info and predictions
    outbound_game_data = process_game_data(game_info_api, predictions, None)

    # Return the processed game data as a JSON response
    return jsonify(outbound_game_data)


@app.after_request
def add_header(response):
    response.headers["Cache-Control"] = "no-store"
    return response


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


def process_game_data(game_info_api, predictions, game_info_schedule):
    """
    Process game data for display.

    This function takes game information from an API or a schedule, processes it, and returns a dictionary
    with the processed data. The processing includes formatting team names, adding team logo URLs, formatting
    date and time for display, adding current score data, adding condensed play-by-play logs, and adding sorted
    players and predictions if available.

    Args:
        game_info_api (dict): Dictionary with game information from an API.
        predictions (dict): Dictionary with game predictions.
        game_info_schedule (dict): Dictionary with game information from a schedule.

    Returns:
        dict: Dictionary with the processed game data.
    """
    # Map game status IDs to their descriptions
    game_status_map = {1: "Not Started", 2: "In Progress", 3: "Completed"}

    # If game_info_api is provided, use it. Otherwise, use game_info_schedule
    if game_info_api:
        game_data = {
            "game_id": game_info_api["game_id"],
            "game_date": game_info_api["game_date"],
            "game_time_est": game_info_api["game_time_est"],
            "home": game_info_api["home"],
            "away": game_info_api["away"],
            "game_status": game_info_api["game_status"],
        }
    else:
        game_data = {
            "game_id": game_info_schedule["gameId"],
            "game_date": game_info_schedule["gameDateTimeEst"][:10],
            "game_time_est": game_info_schedule["gameDateTimeEst"][11:19],
            "home": game_info_schedule["homeTeam"],
            "away": game_info_schedule["awayTeam"],
            "game_status": game_status_map[game_info_schedule["gameStatusId"]],
        }

    # Process team names and update game_data
    game_data.update(_process_team_names(game_data))

    # Add team logo URLs to game_data
    game_data.update(_add_team_logo_urls(game_data))

    # Format date and time for display and update game_data
    game_data.update(_format_date_time_display(game_data, game_info_api))

    # If game_info_api is provided and it contains "game_states", add current score data to game_data
    # Otherwise, set the scores to empty strings
    if game_info_api and game_info_api["game_states"]:
        game_data["home_score"] = game_info_api["game_states"][-1]["home_score"]
        game_data["away_score"] = game_info_api["game_states"][-1]["away_score"]
    else:
        game_data["home_score"] = ""
        game_data["away_score"] = ""

    # If game_info_api is provided and it contains "pbp_logs", add condensed play-by-play logs to game_data
    if game_info_api and game_info_api["pbp_logs"]:
        game_data.update(_get_condensed_pbp(game_info_api))

    # If predictions are provided, add sorted players to game_data
    if predictions:
        game_data.update(_get_sorted_players(predictions))

    # If predictions are provided, add them to game_data
    if predictions:
        game_data["predictions"] = predictions

    return game_data


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
    files = os.listdir("static/img/player_images")

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


if __name__ == "__main__":
    app.run(debug=True)
