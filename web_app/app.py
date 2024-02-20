import math
import os
import sys
from datetime import datetime, timedelta

import pandas as pd
from flask import Flask, jsonify, render_template, request

# Add the parent directory to sys.path to allow imports from there
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from team_mapper import NBATeamConverter

app = Flask(__name__)


game_records = pd.read_csv("../data/test_game_records.csv").to_dict(orient="records")

play_by_play_records = pd.read_csv("../data/test_play_by_play.csv").to_dict(
    orient="records"
)


@app.route("/")
def home():
    # Set the current date, or use the date provided in the query parameters
    # current_date = datetime.now()
    current_date = datetime(2024, 2, 25)
    current_date_str = current_date.strftime("%m/%d/%Y")
    query_date_str = request.args.get("date", current_date_str)
    query_date = datetime.strptime(query_date_str, "%m/%d/%Y")
    query_date_display_str = query_date.strftime("%b %d")

    # Calculate previous and next dates for navigation links
    next_date = query_date + timedelta(days=1)
    prev_date = query_date - timedelta(days=1)
    next_date_str = next_date.strftime("%m/%d/%Y")
    prev_date_str = prev_date.strftime("%m/%d/%Y")

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
    current_date = datetime(2024, 2, 25)
    # current_date = datetime.now()
    # Fetch the date parameter from the request, default to current date if not provided
    query_date_str = request.args.get("date")
    if query_date_str is None or query_date_str == "":
        query_date_str = current_date.strftime("%m/%d/%Y")
    query_date = datetime.strptime(query_date_str, "%m/%d/%Y")

    # Filter game_records for games on the requested date
    games_on_date = [game for game in game_records if game["date"] == query_date_str]

    # Sort games by time
    games_on_date_sorted = sorted(
        games_on_date, key=lambda x: datetime.strptime(x["time"], "%I:%M %p")
    )

    def process_game_data(sorted_games):
        # Helper function to format team names
        def format_team_name(team_name):
            if team_name == "Portland Trail Blazers":
                return "Portland<br>Trail Blazers"
            else:
                parts = team_name.rsplit(" ", 1)
                return f"{parts[0]}<br>{parts[1]}" if len(parts) == 2 else team_name

        # Select the current date for testing or use the current date
        # current_date = datetime.now()
        current_date = datetime(2024, 2, 25)
        for game in sorted_games:
            # Add a time_display field to show the time and period/quarter/overtime
            # or the date and start time if the game is not in progress
            if game["game_state"] == "In Progress":
                period = game["period"]
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
                game["time_display"] = f"{game['time_remaining']} - {period_display}"
            elif game["game_state"] == "Completed":
                game_date = datetime.strptime(game["date"], "%m/%d/%Y")
                if game_date.date() == current_date.date():
                    date_display = "Today"
                elif game_date.date() == (current_date - timedelta(days=1)).date():
                    date_display = "Yesterday"
                else:
                    date_display = game_date.strftime("%b %d")
                game["time_display"] = f"{date_display} - Final"
            elif game["game_state"] == "Not Started":
                game_date = datetime.strptime(game["date"], "%m/%d/%Y")
                if game_date.date() == current_date.date():
                    date_display = "Today"
                elif game_date.date() == (current_date + timedelta(days=1)).date():
                    date_display = "Tomorrow"
                elif game_date.date() == (current_date - timedelta(days=1)).date():
                    date_display = "Yesterday"
                game["time_display"] = f"{date_display} - {game['time']}"

            # Abbreviate the predicted_winner field
            game["predicted_winner_abbr"] = NBATeamConverter.get_abbreviation(
                game["predicted_winner"]
            )

            # Convert predicted_win_pct to a rounded percentage string
            game["predicted_win_pct_str"] = f"{round(game['predicted_win_pct'] * 100)}%"

            # Format team names
            game["home_team_display"] = format_team_name(game["home"])
            game["away_team_display"] = format_team_name(game["away"])

            def generate_logo_url(team_name):
                # Example inbound team_name: "Phoenix Suns"
                # Example url: web_app/static/img/team_logos/nba-phoenix-suns-logo.png
                formatted_team_name = team_name.lower().replace(" ", "-")
                logo_url = f"static/img/team_logos/nba-{formatted_team_name}-logo.png"
                return logo_url

            game["home_logo_url"] = generate_logo_url(game["home"])
            game["away_logo_url"] = generate_logo_url(game["away"])

            def generate_headshot_url(player_id, player_name):
                # Example inbound player_id: 203078
                # Example url: "static/img/player_images/Bradley Beal_203078.png"
                headshot_url = f"static/img/player_images/{player_name}_{player_id}.png"
                return headshot_url

            for player in range(1, 6):
                game[f"player_{player}H_headshot_url"] = generate_headshot_url(
                    game[f"player_{player}H_id"], game[f"player_{player}H_name"]
                )
                game[f"player_{player}R_headshot_url"] = generate_headshot_url(
                    game[f"player_{player}R_id"], game[f"player_{player}R_name"]
                )

        return sorted_games

    games = process_game_data(games_on_date_sorted)

    def clean_data(games):
        for game in games:
            for key, value in game.items():
                if isinstance(value, float) and math.isnan(value):
                    game[key] = None  # Replace NaN with None
        return games

    cleaned_games = clean_data(games)  # Clean the data
    return jsonify(cleaned_games)


@app.route("/game-details/<int:game_id>")
def game_details(game_id):
    # Find the game by game_id
    game_detail = next(
        (game for game in game_records if game["game_id"] == game_id), None
    )

    # Check if the game was found
    if game_detail:
        # Sort play_by_play_records by play_id descending
        sorted_play_by_play_records = sorted(
            play_by_play_records, key=lambda x: x["play_id"], reverse=True
        )

        def clean_play_by_play_records(play_by_play_records):
            for record in play_by_play_records:
                for key, value in record.items():
                    if isinstance(value, float) and math.isnan(value):
                        record[key] = None
            return play_by_play_records

        sorted_play_by_play_records = clean_play_by_play_records(
            sorted_play_by_play_records
        )

        # Return the game details and sorted play by play records as JSON
        return jsonify(
            {
                "game_detail": game_detail,
                "play_by_play": sorted_play_by_play_records,
            }
        )
    else:
        # If no game was found, return an error message and a 404 status code
        return jsonify({"error": "Game not found"}), 404


if __name__ == "__main__":
    app.run(debug=True)
