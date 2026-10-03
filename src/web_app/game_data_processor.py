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

from datetime import datetime, timedelta

import pytz
from tzlocal import get_localzone

from src.utils import NBATeamConverter, get_player_image, log_execution_time


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


@log_execution_time(average_over="games")
def process_game_data(games, user_tz=None):
    """
    Processes game data for display, including team names, logos, date and time display,
    condensed play-by-play logs, predictions, and player data.

    Args:
        games (dict): A dictionary containing game data.
        user_tz (str, optional): User's timezone in IANA format (e.g., "America/New_York").
                                 If None, falls back to server's local timezone.

    Returns:
        list of dict: List of dictionaries with the processed game data.
    """
    outbound_games = []

    for game_id, game in games.items():
        # Parse UTC datetime and convert to user's timezone for display
        from src.utils import parse_utc_datetime, utc_to_user_tz

        utc_dt = parse_utc_datetime(game["date_time_utc"])
        local_dt = utc_to_user_tz(utc_dt, user_tz)

        # Basic game information
        outbound_game_data = {
            "game_id": game_id,
            "game_date": local_dt.strftime("%Y-%m-%d"),
            "game_time_local": local_dt.strftime("%H:%M:%S"),
            "home": game["home_team"],
            "away": game["away_team"],
            "game_status": game.get("status_text", ""),  # Human-readable status
            "game_status_code": game["status"],  # Numeric code (1, 2, 3)
            "season_type": game.get("season_type", ""),
            "prediction_unavailable_reason": (
                "Preseason predictions unavailable"
                if game.get("season_type") == "Pre Season"
                else ""
            ),
        }

        # Current scores if available
        game_states = game.get("game_states", [])
        if game_states:
            game_state = game_states[0]
            outbound_game_data["home_score"] = game_state.get("home_score", "")
            outbound_game_data["away_score"] = game_state.get("away_score", "")
        else:
            outbound_game_data["home_score"] = ""
            outbound_game_data["away_score"] = ""

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

        # Format date and time for display (pass user_tz for Today/Tomorrow logic)
        outbound_game_data.update(_format_date_time_display(game, user_tz))

        # Extract predictions (pre-game only — no in-game blending)
        # Preseason games are listed, but the prediction pipeline covers only
        # regular-season and postseason games. Hide any older preseason forecasts.
        predictions = (
            {} if game.get("season_type") == "Pre Season" else game.get("predictions", {})
        )
        pre_game_predictions = predictions.get("pre_game", {}).get("prediction_set", {})

        pred_home_win_pct = pre_game_predictions.get("pred_home_win_pct", "")
        pred_home_score = pre_game_predictions.get("pred_home_score", "")
        pred_away_score = pre_game_predictions.get("pred_away_score", "")

        # Predicted spread: use pred_spread if available (Phase5/Phase3),
        # otherwise derive from predicted scores (legacy models)
        pred_spread = pre_game_predictions.get("pred_spread", "")
        if pred_spread == "" and pred_home_score != "" and pred_away_score != "":
            pred_spread = pred_home_score - pred_away_score

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

        outbound_game_data["pred_winner"] = pred_winner
        outbound_game_data["pred_home_score"] = (
            round(pred_home_score) if pred_home_score != "" else ""
        )
        outbound_game_data["pred_away_score"] = (
            round(pred_away_score) if pred_away_score != "" else ""
        )

        # Format predicted win percentage
        if pred_win_pct == "":
            pred_win_pct_str = ""
        elif pred_win_pct == 1:
            pred_win_pct_str = "100%"
        elif pred_win_pct >= 0.995:
            pred_win_pct_str = ">99%"
        elif pred_win_pct < 0.995:
            pred_win_pct_str = f"{pred_win_pct:.0%}"
        else:
            pred_win_pct_str = ""

        outbound_game_data["pred_win_pct"] = pred_win_pct_str

        # Format spreads as "TEAM by X" (e.g., "BOS by 6.5")
        home_abbr = outbound_game_data["home"]
        away_abbr = outbound_game_data["away"]

        def _spread_display(spread_val, home, away):
            """Format spread as 'TEAM by X'. Positive = home favored."""
            if spread_val is None or spread_val == "":
                return ""
            v = float(spread_val)
            if v > 0:
                return f"{home} by {v:.1f}"
            elif v < 0:
                return f"{away} by {abs(v):.1f}"
            return "Pick"

        # Model's pred_spread: positive = home advantage
        outbound_game_data["pred_spread"] = (
            _spread_display(pred_spread, home_abbr, away_abbr)
            if pred_spread != ""
            else ""
        )

        # Vegas opening spread: negative = home favored → negate for our convention
        opening_spread = game.get("opening_spread")
        outbound_game_data["opening_spread"] = (
            _spread_display(-opening_spread, home_abbr, away_abbr)
            if opening_spread is not None
            else ""
        )

        # Determine if predicted winner was correct (for completed games)
        if (
            game["status"] == 3
            and pred_winner
            and outbound_game_data["home_score"] != ""
            and outbound_game_data["away_score"] != ""
        ):
            home_score = outbound_game_data["home_score"]
            away_score = outbound_game_data["away_score"]
            actual_winner = (
                outbound_game_data["home"]
                if home_score > away_score
                else outbound_game_data["away"]
            )
            outbound_game_data["pred_winner_correct"] = pred_winner == actual_winner
        else:
            outbound_game_data["pred_winner_correct"] = None

        # Determine if our predicted spread was closer to actual margin than Vegas
        if (
            game["status"] == 3
            and pred_spread != ""
            and opening_spread is not None
            and outbound_game_data["home_score"] != ""
            and outbound_game_data["away_score"] != ""
        ):
            home_score = outbound_game_data["home_score"]
            away_score = outbound_game_data["away_score"]
            # Actual margin in Vegas convention: negative = home won by that amount
            actual_margin = -(home_score - away_score)
            our_spread_val = (
                -pred_spread
            )  # pred_spread is model's home advantage; negate for Vegas convention
            our_error = abs(our_spread_val - actual_margin)
            vegas_error = abs(opening_spread - actual_margin)
            outbound_game_data["spread_closer_than_vegas"] = our_error < vegas_error
        else:
            outbound_game_data["spread_closer_than_vegas"] = None

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


def _format_date_time_display(game, user_tz=None):
    """
    Formats the date and time display for a game.

    Args:
        game (dict): A dictionary containing game data.
        user_tz (str, optional): User's timezone in IANA format (e.g., "America/New_York").
                                 If None, falls back to server's local timezone.

    Returns:
        dict: A dictionary containing the formatted date and time display.
    """

    # Only show time remaining if we have game states and the game is in progress or not finalized
    has_game_states = game.get("game_states") and len(game["game_states"]) > 0

    if has_game_states and (
        game["status"] == 2  # In Progress
        or not game["game_states"][-1].get("is_final_state", False)
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
    # Data is now stored as actual UTC - convert to user's timezone
    from src.utils import get_utc_now, parse_utc_datetime, utc_to_user_tz

    game_date_time_utc = game["date_time_utc"]
    utc_dt = parse_utc_datetime(game_date_time_utc)
    # Convert to user's timezone for display
    game_date_time_local = utc_to_user_tz(utc_dt, user_tz)

    game_date = game_date_time_local.date()

    # Get current date in user's timezone (not server's local time)
    # This ensures "Today/Tomorrow/Yesterday" is correct for the user
    current_datetime_user = utc_to_user_tz(get_utc_now(), user_tz)
    current_date = current_datetime_user.date()
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

    if game["status"] == 3:  # Final
        datetime_display = f"{date_display} - Final"
    else:
        datetime_display = f"{date_display} - {time_display}"

    return {"datetime_display": datetime_display}


def _get_sorted_players(game, predictions):
    """
    Combines player data from the current game state and predictions, assigns a headshot image
    to each player, and sorts the players for display.

    For completed/in-progress games: shows actual stats, sorted by actual points scored.
    For upcoming games with predictions: shows predicted stats, sorted by predicted points.
    For upcoming games without predictions: shows roster from game state (if available).

    Args:
        game (dict): A dictionary containing the current game state.
        predictions (dict): A dictionary containing player data predictions.

    Returns:
        dict: A dictionary containing sorted home and away players.
    """

    game_status = game.get("status", 1)
    has_actual_stats = game_status in (2, 3)  # In-progress or Final

    players = {"home_players": [], "away_players": []}

    for team in ["home", "away"]:
        team_players = (
            game.get("game_states", [{}])[-1].get("players_data", {}).get(team, {})
            if game.get("game_states")
            else {}
        )
        pre_game_team_predictions = (
            predictions.get("pre_game", {})
            .get("prediction_set", {})
            .get("pred_players", {})
            .get(team, {})
        )

        all_player_ids = set(team_players.keys()).union(
            pre_game_team_predictions.keys()
        )

        has_pred_players = bool(pre_game_team_predictions)

        for player_id in all_player_ids:
            player_data = team_players.get(player_id, {})
            player_prediction = pre_game_team_predictions.get(player_id, {})

            player_headshot_url = get_player_image(player_id)

            # Actual points from game state (available for in-progress/completed games)
            actual_points = player_data.get("points", None)
            # Predicted points from predictor (may not be available for Phase 5)
            pred_points = player_prediction.get("pred_points", None)

            player = {
                "player_id": player_id,
                "player_name": player_data.get("name", ""),
                "player_headshot_url": player_headshot_url,
                "points": actual_points,
                "pred_points": pred_points,
            }

            players[f"{team}_players"].append(player)

        # Sort by actual points for completed/in-progress games, pred_points for upcoming
        if has_actual_stats:
            players[f"{team}_players"] = sorted(
                players[f"{team}_players"],
                key=lambda x: x["points"] if x["points"] is not None else -1,
                reverse=True,
            )
        elif has_pred_players:
            players[f"{team}_players"] = sorted(
                players[f"{team}_players"],
                key=lambda x: x["pred_points"] if x["pred_points"] is not None else -1,
                reverse=True,
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
