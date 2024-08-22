"""
game_data_processor.py

This module provides functionality to process and prepare NBA game data for display, including team names, logos, 
game status, scores, predictions, and player statistics. Converts api data into a format suitable for rendering in a web application.
Core Functions:
- get_user_datetime(as_eastern_tz=False): Fetch the current date and time in the user's local timezone or in Eastern Time Zone (ET).
- process_game_data(games): Process game data to include team names, scores, logos, predictions, and player stats.

Helper Functions:
- _process_team_names(game): Format team data for display, including full and display names.
- _generate_logo_url(team_name): Generate a URL for the team's logo based on the team name.
- _format_date_time_display(game): Format the date and time for display based on game status.
- _get_sorted_players(game, predictions): Compile and sort player data, including headshots and predicted points.
- _get_condensed_pbp(game): Condense the play-by-play logs into a simplified format.

Usage:
- Typically integrated into web applications or dashboards for displaying NBA game information.
"""

import os
from datetime import datetime, timedelta

import pytz
from tzlocal import get_localzone

from src.utils import NBATeamConverter, get_player_image


def get_user_datetime(as_eastern_tz=False):
    """
    Returns the current date and time in the user's local timezone or in Eastern Time Zone (ET),
    taking into account daylight saving time.

    Args:
        as_eastern_tz (bool, optional): If True, returns the date and time in ET.
                                        Otherwise, in the user's local timezone.

    Returns:
        datetime: The current date and time in the specified timezone.
    """
    # Fetch the current UTC time
    utc_now = datetime.now(pytz.utc)

    if as_eastern_tz:
        # Convert to Eastern Time Zone if requested
        eastern_timezone = pytz.timezone("US/Eastern")
        return utc_now.astimezone(eastern_timezone)

    # Convert to user's local timezone
    user_timezone = get_localzone()
    return utc_now.astimezone(user_timezone)


def process_game_data(games):
    """
    Processes game data for display, including team names, logos, date and time display,
    condensed play-by-play logs, predictions, and player data.

    Args:
        games (dict): A dictionary containing game data.

    Returns:
        list of dict: List of dictionaries with the processed game data.
    """
    outbound_games = []

    for game_id, game in games.items():
        # Basic game information
        outbound_game_data = {
            "game_id": game_id,
            "game_date": game["date_time_est"].split("T")[0],
            "game_time_est": game["date_time_est"].split("T")[1],
            "home": game["home_team"],
            "away": game["away_team"],
            "game_status": game["status"],
        }

        # Current scores if available
        game_state = game.get("game_states", [{}])[0]
        outbound_game_data["home_score"] = game_state.get("home_score", "")
        outbound_game_data["away_score"] = game_state.get("away_score", "")

        # Process team names and generate logo URLs
        outbound_game_data.update(
            _process_team_names({"home": game["home_team"], "away": game["away_team"]})
        )
        outbound_game_data["home_logo_url"] = _generate_logo_url(
            outbound_game_data["home_full_name"]
        )
        outbound_game_data["away_logo_url"] = _generate_logo_url(
            outbound_game_data["away_full_name"]
        )

        # Format date and time for display
        outbound_game_data.update(_format_date_time_display(game))

        # Extract predictions
        predictions = game.get("predictions", {})
        current_predictions = predictions.get("current", {})
        pre_game_predictions = predictions.get("pre_game", {})

        pred_home_score = current_predictions.get(
            "pred_home_score", pre_game_predictions.get("pred_home_score", "")
        )
        pred_away_score = current_predictions.get(
            "pred_away_score", pre_game_predictions.get("pred_away_score", "")
        )
        pred_home_win_pct = current_predictions.get(
            "pred_home_win_pct", pre_game_predictions.get("pred_home_win_pct", "")
        )

        # Determine the predicted winner and win probability
        if pred_home_win_pct != "":
            if pred_home_win_pct >= 0.5:
                pred_winner = outbound_game_data["home"]
                pred_win_pct = pred_home_win_pct
            else:
                pred_winner = outbound_game_data["away"]
                pred_win_pct = 1 - pred_home_win_pct
        else:
            pred_winner = ""
            pred_win_pct = ""

        outbound_game_data["pred_home_score"] = pred_home_score
        outbound_game_data["pred_away_score"] = pred_away_score
        outbound_game_data["pred_winner"] = pred_winner

        # Format predicted win percentage
        if pred_win_pct == 1:
            pred_win_pct_str = "100%"
        elif pred_win_pct >= 0.995:
            pred_win_pct_str = ">99%"
        elif pred_win_pct < 0.995:
            pred_win_pct_str = "99%"
        else:
            pred_win_pct_str = ""

        outbound_game_data["pred_win_pct"] = pred_win_pct_str

        # Add sorted players and condensed play-by-play logs if available
        outbound_game_data.update(_get_sorted_players(game, predictions))

        if "play_by_play" in game and game["play_by_play"]:
            outbound_game_data.update(_get_condensed_pbp(game))
        else:
            outbound_game_data["condensed_pbp"] = []

        # Add the processed game data to the list
        outbound_games.append(outbound_game_data)

    return outbound_games


def _process_team_names(game):
    """
    Formats team data for display.

    Args:
        game (dict): A dictionary containing game data.

    Returns:
        dict: A dictionary containing the full and formatted team names.
    """
    # Retrieve full team names
    home_full_name = NBATeamConverter.get_full_name(game["home"])
    away_full_name = NBATeamConverter.get_full_name(game["away"])

    def format_team_name(full_name):
        # Special formatting for Trail Blazers
        if "Trail Blazers" in full_name:
            city, team = full_name.split(" Trail ")
            return f"{city}<br>Trail {team}"
        else:
            city, team = full_name.rsplit(" ", 1)
            return f"{city}<br>{team}"

    # Format team names for display
    home_team_display = format_team_name(home_full_name)
    away_team_display = format_team_name(away_full_name)

    return {
        "home_full_name": home_full_name,
        "away_full_name": away_full_name,
        "home_team_display": home_team_display,
        "away_team_display": away_team_display,
    }


def _generate_logo_url(team_name):
    """
    Generates a logo URL for a team.

    Args:
        team_name (str): The name of the team.

    Returns:
        str: The URL for the team's logo.
    """
    # Format the team name for URL
    formatted_team_name = team_name.lower().replace(" ", "-")
    logo_url = f"static/img/team_logos/nba-{formatted_team_name}-logo.png"
    return logo_url


def _format_date_time_display(game):
    """
    Formats the date and time display for a game.

    Args:
        game (dict): A dictionary containing game data.

    Returns:
        dict: A dictionary containing the formatted date and time display.
    """
    if game["status"] == "In Progress" or (
        game["game_states"] and not game["game_states"][-1].get("is_final_state", False)
    ):
        period = game["game_states"][-1]["period"]
        time_remaining = game["game_states"][-1]["clock"]
        minutes, seconds = time_remaining.lstrip("PT").rstrip("S").split("M")
        minutes = int(minutes)
        seconds = int(seconds.split(".")[0])
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
        datetime_display = f"{time_remaining} - {period_display}"
        return {"datetime_display": datetime_display}

    # Handle cases for not started or completed games
    game_date_time_est = game["date_time_est"]
    game_date_time_est = datetime.strptime(game_date_time_est, "%Y-%m-%dT%H:%M:%SZ")
    user_timezone = get_localzone()
    game_date_time_local = game_date_time_est.astimezone(user_timezone)

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

    time_display = game_date_time_local.strftime("%I:%M %p").lstrip("0")

    if game["status"] == "Completed":
        datetime_display = f"{date_display} - Final"
    else:
        datetime_display = f"{date_display} - {time_display}"

    return {"datetime_display": datetime_display}


def _get_sorted_players(game, predictions):
    """
    This function combines player data from the current game state and predictions, assigns a headshot image to each player,
    sorts the players based on their predicted points in descending order, and returns a dictionary with the sorted lists of players
    for both the home and away teams.

    Args:
        game (dict): A dictionary containing the current game state.
        predictions (dict): A dictionary containing player data predictions.

    Returns:
        dict: A dictionary containing sorted home and away players.
    """

    players = {"home_players": [], "away_players": []}

    for team in ["home", "away"]:
        team_players = (
            game.get("game_states", [{}])[-1].get("players_data", {}).get(team, {})
            if game.get("game_states")
            else {}
        )
        current_team_predictions = (
            predictions.get("current", {}).get("pred_players", {}).get(team, {})
        )
        pre_game_team_predictions = (
            predictions.get("pre_game", {}).get("pred_players", {}).get(team, {})
        )

        all_player_ids = set(team_players.keys()).union(
            current_team_predictions.keys(), pre_game_team_predictions.keys()
        )

        for player_id in all_player_ids:
            player_data = team_players.get(player_id, {})
            player_prediction = current_team_predictions.get(
                player_id, current_team_predictions.get(player_id, {})
            )

            player_headshot_url = get_player_image(player_id)

            player = {
                "player_id": player_id,
                "player_name": player_data.get("name", ""),
                "player_headshot_url": player_headshot_url,
                "points": player_data.get("points", 0),
                "pred_points": player_prediction.get("pred_points", 0),
            }

            players[f"{team}_players"].append(player)

        players[f"{team}_players"] = sorted(
            players[f"{team}_players"], key=lambda x: x["pred_points"], reverse=True
        )

    return players


def _get_condensed_pbp(game):
    """
    Condense the play-by-play logs from a game info API response.

    Args:
        game (dict): A dictionary containing game info data.

    Returns:
        dict: A dictionary containing the condensed play-by-play logs.
    """
    pbp = sorted(game["play_by_play"], key=lambda x: x["play_id"], reverse=True)

    condensed_pbp = []

    for play in pbp:
        time_remaining = play["clock"]
        minutes, seconds = time_remaining.lstrip("PT").rstrip("S").split("M")
        minutes = int(minutes)
        seconds = int(seconds.split(".")[0])
        if play["period"] > 4:
            time_info = f"{minutes}:{seconds:02} OT{play['period'] - 4}"
        else:
            time_info = f"{minutes}:{seconds:02} Q{play['period']}"

        condensed_pbp.append(
            {
                "time_info": time_info,
                "home_score": play["scoreHome"],
                "away_score": play["scoreAway"],
                "description": play["description"],
            }
        )

    return {"condensed_pbp": condensed_pbp}
