import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytz
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from tzlocal import get_localzone

# Add the parent directory to sys.path to allow imports from there
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from src.NBAStats_game_states import get_current_game_info
from src.predictions import get_predictions
from src.utils import NBATeamConverter, get_games_for_date

load_dotenv()
PROJECT_ROOT = os.getenv("PROJECT_ROOT")

app = Flask(__name__)


def process_game_data(game, current_date_et):
    game_data = {
        "game_id": game["game_id"],
        "game_date": game["game_date"],
        "home": game["home"],
        "away": game["away"],
    }

    # Full team names
    home_full_name = NBATeamConverter.get_full_name(game_data["home"])
    away_full_name = NBATeamConverter.get_full_name(game_data["away"])
    game_data["home_full_name"] = home_full_name
    game_data["away_full_name"] = away_full_name

    def format_team_name(full_name):
        if "Trail Blazers" in full_name:
            city, team = full_name.split(" Trail ")
            return f"{city}<br>Trail {team}"
        else:
            city, team = full_name.rsplit(" ", 1)
            return f"{city}<br>{team}"

    game_data["home_team_display"] = format_team_name(home_full_name)
    game_data["away_team_display"] = format_team_name(away_full_name)

    # Team Logo URLs
    def generate_logo_url(team_name):
        # Example inbound team_name: "Phoenix Suns"
        # Example url: web_app/static/img/team_logos/nba-phoenix-suns-logo.png
        formatted_team_name = team_name.lower().replace(" ", "-")
        logo_url = f"static/img/team_logos/nba-{formatted_team_name}-logo.png"
        return logo_url

    game_data["home_logo_url"] = generate_logo_url(home_full_name)
    game_data["away_logo_url"] = generate_logo_url(away_full_name)

    # Time Display
    # Add a time_display field to show the time and period/quarter/overtime
    # or the date and start time if the game is not in progress
    if game["game_status"] == "In Progress":
        period = game["game_states"][-1]["period"]
        # Parse the time_remaining string
        # Parse the time_remaining string
        time_remaining = game["game_states"][-1]["remaining_time"]
        minutes, seconds = time_remaining.lstrip("PT").rstrip("S").split("M")
        minutes = int(minutes)  # Remove leading zero from minutes
        seconds = int(seconds.split(".")[0])  # Only take the whole seconds

        # Format the time_remaining string
        time_remaining = f"{minutes}:{seconds:02}"

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
        game_data["time_display"] = f"{time_remaining} - {period_display}"
    elif game["game_status"] == "Completed":
        game_date = datetime.strptime(game["game_date"], "%Y-%m-%d")
        if game_date.date() == current_date_et.date():
            date_display = "Today"
        elif game_date.date() == (current_date_et - timedelta(days=1)).date():
            date_display = "Yesterday"
        else:
            date_display = game_date.strftime("%b %d")
        game_data["time_display"] = f"{date_display} - Final"
    elif game["game_status"] == "Not Started":
        game_date = datetime.strptime(game["game_date"], "%Y-%m-%d")
        if game_date.date() == current_date_et.date():
            date_display = "Today"
        elif game_date.date() == (current_date_et + timedelta(days=1)).date():
            date_display = "Tomorrow"
        elif game_date.date() == (current_date_et - timedelta(days=1)).date():
            date_display = "Yesterday"
        else:
            date_display = game_date.strftime("%b %d")

        # Parse the game date and time as Eastern Time
        eastern = pytz.timezone("US/Eastern")
        game_date = datetime.strptime(game["game_date"], "%Y-%m-%d").date()
        game_hour, game_minute = map(int, game["game_time_et"].split(":"))
        game_time_et = datetime(
            game_date.year, game_date.month, game_date.day, game_hour, game_minute
        )
        game_time_et = eastern.localize(game_time_et)

        # Convert to the user's local timezone
        user_timezone = get_localzone()
        game_time_local = game_time_et.astimezone(user_timezone)

        # Format the local game time as a string
        game_time_local_str = game_time_local.strftime("%I:%M %p").lstrip("0")

        # Update the time display
        game_data["time_display"] = f"{date_display} - {game_time_local_str}"

    # Score Display
    if game["game_states"]:
        game_data["home_score"] = game["game_states"][-1]["home_score"]
        game_data["away_score"] = game["game_states"][-1]["away_score"]
    else:
        game_data["home_score"] = ""
        game_data["away_score"] = ""

    # Play by Play Logs
    pbp = sorted(game["pbp_logs"], key=lambda x: x["orderNumber"], reverse=True)
    condensed_pbp = []
    for play in pbp:
        # Parse the clock value
        time_remaining = play["clock"]
        minutes, seconds = time_remaining.lstrip("PT").rstrip("S").split("M")
        minutes = int(minutes)  # Remove leading zero from minutes
        seconds = int(seconds.split(".")[0])  # Only take the whole seconds

        # Combine clock and period into a single item
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

    game_data["pbp"] = condensed_pbp

    # Pass through predictions
    game_data["predictions"] = game["predictions"]

    # Players to show
    home_players = []
    away_players = []

    # Iterate over home players
    for player_id, player_data in game["predictions"]["players"]["home"].items():
        # Get all files in the directory
        files = os.listdir("static/img/player_images")

        # Find the file that ends with {player_id}.png
        player_image_file = next(
            (f for f in files if f.endswith(f"_{player_id}.png")), None
        )

        if player_image_file:
            player_headshot_url = f"static/img/player_images/{player_image_file}"
        else:
            player_headshot_url = "static/img/basketball-player.png"  # Replace with your default image URL

        player = {
            "player_id": player_id,
            "name": player_data["name"],
            "player_headshot_url": player_headshot_url,
            "predicted_points": player_data["points"],
        }
        home_players.append(player)

    # Iterate over away players
    for player_id, player_data in game["predictions"]["players"]["away"].items():
        # Get all files in the directory
        files = os.listdir("static/img/player_images")

        # Find the file that ends with {player_id}.png
        player_image_file = next(
            (f for f in files if f.endswith(f"_{player_id}.png")), None
        )

        if player_image_file:
            player_headshot_url = f"static/img/player_images/{player_image_file}"
        else:
            player_headshot_url = "static/img/basketball-player.png"  # Replace with your default image URL

        player = {
            "player_id": player_id,
            "name": player_data["name"],
            "player_headshot_url": player_headshot_url,
            "predicted_points": player_data["points"],
        }
        away_players.append(player)

    # Sort players by predicted points in descending order
    home_players = sorted(
        home_players, key=lambda x: x["predicted_points"], reverse=True
    )
    away_players = sorted(
        away_players, key=lambda x: x["predicted_points"], reverse=True
    )

    game_data["home_players"] = home_players
    game_data["away_players"] = away_players

    return game_data


def get_user_datetime(as_eastern_tz=False):
    user_timezone = get_localzone()
    user_datetime = datetime.now(user_timezone)
    if as_eastern_tz:
        eastern_timezone = pytz.timezone("US/Eastern")
        user_datetime_eastern = user_datetime.astimezone(eastern_timezone)
        return user_datetime_eastern
    return user_datetime


@app.route("/")
def home():
    # Set the current date, or use the date provided in the query parameters
    current_date_et = get_user_datetime(as_eastern_tz=True)
    # current_date = datetime.now()
    # current_date = datetime(2024, 2, 25)
    current_date_str = current_date_et.strftime("%Y-%m-%d")
    query_date_str = request.args.get("date", current_date_str)
    query_date = datetime.strptime(query_date_str, "%Y-%m-%d")
    query_date_display_str = query_date.strftime("%b %d")

    # Calculate previous and next dates for navigation links
    next_date = query_date + timedelta(days=1)
    prev_date = query_date - timedelta(days=1)
    next_date_str = next_date.strftime("%Y-%m-%d")
    prev_date_str = prev_date.strftime("%Y-%m-%d")

    # Render the template, passing necessary data for initial page setup
    return render_template(
        "index.html",
        query_date_str=query_date_str,
        query_date_display_str=query_date_display_str,
        prev_date=prev_date_str,
        next_date=next_date_str,
    )


@app.route("/get-games")
def get_games():
    current_date_et = get_user_datetime(as_eastern_tz=True)
    # Fetch the date parameter from the request, default to current date if not provided
    query_date_str = request.args.get("date")
    if query_date_str is None or query_date_str == "":
        query_date_str = current_date_et.strftime("%Y-%m-%d")

    # Load Games
    games = get_games_for_date(query_date_str)[
        0:2
    ]  # Limit to 2 games for now for testing
    games = [get_current_game_info(game["game_id"]) for game in games]

    # Create Predictions
    games = [get_predictions(game) for game in games]

    outbound_game_data = []

    for game in games:
        game_data = process_game_data(game, current_date_et)
        outbound_game_data.append(game_data)

    return jsonify(outbound_game_data)


@app.route("/game-details/<game_id>")
def game_details(game_id):
    # Load game data
    game = get_current_game_info(game_id)

    # Create Predictions
    game = get_predictions(game)

    # Process game data
    current_date_et = get_user_datetime(as_eastern_tz=True)
    outbound_game_data = process_game_data(game, current_date_et)

    return jsonify(outbound_game_data)


@app.after_request
def add_header(response):
    response.headers["Cache-Control"] = "no-store"
    return response


if __name__ == "__main__":
    app.run(debug=True)
