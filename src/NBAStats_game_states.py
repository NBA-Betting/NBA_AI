import os
from copy import deepcopy

import pandas as pd
from dotenv import load_dotenv
from nba_api.live.nba.endpoints import playbyplay

try:
    from src.utils import lookup_basic_game_info, validate_game_id
except ModuleNotFoundError:
    from utils import lookup_basic_game_info, validate_game_id

load_dotenv()
PROJECT_ROOT = os.getenv("PROJECT_ROOT")

pd.set_option("display.max_columns", None)


def get_pbp(game_id):
    """
    This function retrieves the play-by-play logs for a given game ID.

    Parameters:
    game_id (str): The ID of the game to retrieve the play-by-play logs for.

    Returns:
    list: A list of dictionaries where each dictionary represents a play-by-play log, sorted by order number.
    """

    # Validate the game_id
    validate_game_id(game_id)

    try:
        # Use the PlayByPlay class to retrieve the play-by-play logs for the game
        pbp = playbyplay.PlayByPlay(game_id=game_id)
    except Exception as e:
        # If an API call error occurs (e.g., the game ID is not found), print the error and return an empty list
        print(
            f"API call error occurred for game ID {game_id}. Has game started yet? \n {e}"
        )
        return []

    try:
        # Convert the PlayByPlay object to a dictionary
        pbp = pbp.get_dict()
        # Extract the play-by-play logs from the dictionary
        pbp_logs = pbp["game"]["actions"]
        # Sort the play-by-play logs by order number
        pbp_logs_sorted = sorted(pbp_logs, key=lambda x: x["orderNumber"])
    except Exception as e:
        # If an API response format error occurs, print the error and return an empty list
        print(f"API response format error occurred for game ID {game_id}: {e}")
        return []

    # Return the sorted play-by-play logs
    return pbp_logs_sorted


def _create_game_states_and_final_state(pbp_logs, home, away, game_id, game_date):
    """
    This function creates game states from play-by-play logs.

    Parameters:
    pbp_logs (list): A list of dictionaries where each dictionary represents a play-by-play log.
    home (str): The tricode of the home team.
    away (str): The tricode of the away team.
    game_id (str): The ID of the game.
    game_date (str): The date of the game.

    Returns:
    list, dict: A list of dictionaries where each dictionary represents a game state at a specific point in time,
                and a dictionary representing the final game state.
    """

    # If there are no play-by-play logs, return an empty list and dictionary
    if not pbp_logs:
        return [], {}

    # Sort the play-by-play logs by order number
    pbp_logs = sorted(pbp_logs, key=lambda x: x["orderNumber"])

    # Filter out logs where 'description' is not a key
    pbp_logs = [log for log in pbp_logs if "description" in log]

    # Initialize the list of game states and the players dictionary
    game_states = []
    players = {"home": {}, "away": {}}

    # Iterate over the play-by-play logs
    for row in pbp_logs:
        # If the play involves a player, update the player's points
        if row.get("personId") is not None and row.get("playerNameI") is not None:
            # Determine the team of the player
            team = "home" if row["teamTricode"] == home else "away"

            # Extract the player's ID and name
            player_id = row["personId"]
            player_name = row["playerNameI"]

            # If the player is not already in the game state, add them
            if player_id not in players[team]:
                players[team][player_id] = {"name": player_name, "points": 0}

            # If the play resulted in points, update the player's points
            if row.get("pointsTotal") is not None:
                points = int(row["pointsTotal"])
                players[team][player_id]["points"] = points

        # Create the current game state
        current_game_state = {
            "game_id": game_id,
            "game_date": game_date,
            "home": home,
            "away": away,
            "play_id": int(row["orderNumber"]),
            "remaining_time": row["clock"],
            "period": int(row["period"]),
            "home_score": int(row["scoreHome"]),
            "away_score": int(row["scoreAway"]),
            "total": int(row["scoreHome"]) + int(row["scoreAway"]),
            "home_margin": int(row["scoreHome"]) - int(row["scoreAway"]),
            "players": deepcopy(
                players
            ),  # Create a deep copy of the players dictionary
        }

        # Add the current game state to the list of game states
        game_states.append(current_game_state)

    # If the last play was the end of the game, set the final state to the last game state
    if pbp_logs[-1]["description"] == "Game End":
        final_state = game_states[-1]
    else:
        final_state = {}

    return game_states, final_state


def get_current_game_info(game_id):
    """
    This function retrieves the current game information given a game ID.

    Parameters:
    game_id (str): The ID of the game to retrieve the information for.

    Returns:
    dict: A dictionary containing the game information, including the game ID, game date, game time (EST), home team, away team, game status, play-by-play logs, game states, and final state.
    """

    # Validate the game_id
    validate_game_id(game_id)

    # Look up the basic game information
    game_info = lookup_basic_game_info(game_id)

    # Extract the game date, game time, home team, and away team from the game information
    game_date = game_info["game_date"]
    game_time_est = game_info["game_time_est"]
    home = game_info["home"]
    away = game_info["away"]
    game_status_id = int(game_info["game_status_id"])

    # Retrieve the play-by-play logs for the game
    pbp_logs = get_pbp(game_id)

    # Create the game states from the play-by-play logs
    game_states, final_state = _create_game_states_and_final_state(
        pbp_logs, home, away, game_id, game_date
    )

    # Determine the game status based on the game status ID and the presence of final state and play-by-play logs
    if final_state and game_status_id == 3:
        game_status = "Completed"
    elif pbp_logs and game_status_id == 2:
        game_status = "In Progress"
    elif game_status_id == 1:
        game_status = "Not Started"
    else:
        game_status = "Unknown"

    # Create the base dictionary with the game information
    game_info = {
        "game_id": game_id,
        "game_date": game_date,
        "game_time_est": game_time_est,
        "home": home,
        "away": away,
        "game_status": game_status,
        "pbp_logs": pbp_logs,
        "game_states": game_states,
        "final_state": final_state,
    }

    return game_info


if __name__ == "__main__":

    def print_current_game_info(game_info):
        print()
        print("Game ID:", game_info["game_id"])
        print("Game Date:", game_info["game_date"])
        print("Game Time (EST):", game_info["game_time_est"])
        print("Home Team:", game_info["home"])
        print("Away Team:", game_info["away"])
        print("Game Status:", game_info["game_status"])
        print("Play-by-Play Log Count:", len(game_info["pbp_logs"]))
        print("Game States Count:", len(game_info["game_states"]))
        if game_info["game_status"] != "Not Started":
            if game_info["final_state"] == game_info["game_states"][-1]:
                print("Final State matches the last game state.")

            most_recent_state = game_info["game_states"][-1]
            print("Most Recent State:")
            print("  Remaining Time:", most_recent_state["remaining_time"])
            print("  Period:", most_recent_state["period"])
            print("  Home Score:", most_recent_state["home_score"])
            print("  Away Score:", most_recent_state["away_score"])
            print("  Total Score:", most_recent_state["total"])
            print("  Home Margin:", most_recent_state["home_margin"])
            print("  Players:", most_recent_state["players"])
        print()

    # First Day of Season Game
    c1 = get_current_game_info("0022200001")
    print_current_game_info(c1)

    # Late Season Game
    c2 = get_current_game_info("0022200919")
    print_current_game_info(c2)

    # Future Game
    c3 = get_current_game_info("0022301170")
    print_current_game_info(c3)
