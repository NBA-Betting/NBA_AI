import json
import os
from copy import deepcopy

import pandas as pd
from dotenv import load_dotenv
from nba_api.live.nba.endpoints import playbyplay
from tqdm import tqdm

try:
    from src.NBAStats_prior_states import get_prior_states
    from src.utils import (
        game_id_to_season,
        get_schedule,
        lookup_basic_game_info,
        validate_game_id,
    )
except ModuleNotFoundError:
    from NBAStats_prior_states import get_prior_states
    from utils import (
        game_id_to_season,
        get_schedule,
        lookup_basic_game_info,
        validate_game_id,
    )

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
        # Convert the PlayByPlay object to a dictionary
        pbp = pbp.get_dict()
    except Exception as e:
        # If an API call error occurs (e.g., the game ID is not found), print the error and return an empty list
        print(f"API call error occurred for game ID {game_id}: {e}")
        return []

    try:
        # Extract the play-by-play logs from the dictionary
        pbp_logs = pbp["game"]["actions"]
        # Sort the play-by-play logs by order number
        pbp_logs_sorted = sorted(pbp_logs, key=lambda x: x["orderNumber"])
    except KeyError as e:
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


def get_current_game_info(game_id, include_prior_states=False):
    """
    This function retrieves the current game information given a game ID and a flag indicating whether to include prior states.

    Parameters:
    game_id (str): The ID of the game to retrieve the information for.
    include_prior_states (bool): A flag indicating whether to include prior states. Defaults to False.

    Returns:
    dict: A dictionary containing the game information, including the game ID, game date, game time (EST), home team, away team, game status, play-by-play logs, game states, final state, and prior states (if requested).
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

    # If the include_prior_states flag is set to True, retrieve the prior states and handle any exceptions
    if include_prior_states:
        try:
            game_info["prior_states"] = get_prior_states(game_id)
        except Exception as e:
            print(f"Failed to get prior states for game {game_id}. Error: {e}")

    return game_info


def update_database(game_data_dict, print_updated=True):
    game_id = game_data_dict["game_id"]
    game_date = game_data_dict["game_date"]
    home = game_data_dict["home"]
    away = game_data_dict["away"]
    season = game_id_to_season(game_id)

    directory = f"{PROJECT_ROOT}/data/NBAStats/{season}/{game_date}"
    filename = f"{directory}/{game_id}_{home}_{away}.json"

    # Create the directory if it does not exist
    os.makedirs(directory, exist_ok=True)

    # If the file exists, load the existing data and update it
    if os.path.exists(filename):
        with open(filename, "r") as json_file:
            existing_data = json.load(json_file)
        existing_data.update(game_data_dict)
    else:
        existing_data = game_data_dict

    # Write the updated data back to the file
    with open(filename, "w") as json_file:
        json.dump(existing_data, json_file, indent=4)

    if print_updated:
        print(f"Updated: {filename}")


def update_full_season_database(
    season,
    season_type="Regular Season",
    include_prior_states=False,
):
    schedule = get_schedule(season)
    season_type_codes = {
        "001": "Pre Season",
        "002": "Regular Season",
        "003": "All-Star",
        "004": "Post Season",
    }

    # Filter the game_ids to only include games that match the season type
    games = [
        game
        for game in schedule
        if season_type_codes.get(game["gameId"][:3]) == season_type
    ]

    # Sort the games by 'gameDateTimeEst' from least recent to most recent
    games_sorted = sorted(games, key=lambda game: game["gameDateTimeEst"])

    # Extract the game_ids from the sorted games
    game_ids = [game["gameId"] for game in games_sorted]

    for game_id in tqdm(game_ids, desc="Updating database"):
        try:
            # Update the database for this game
            if include_prior_states:
                game_data_dict = get_current_game_info(
                    game_id, include_prior_states=True
                )
            else:
                game_data_dict = get_current_game_info(game_id)
            update_database(game_data_dict, print_updated=False)
        except Exception as e:
            # Find the game that caused the exception
            failed_game = next(
                game for game in games_sorted if game["gameId"] == game_id
            )
            print(
                f"Failed to update game with ID: {failed_game['gameId']} and DateTime: {failed_game['gameDateTimeEst']}"
            )
            print(f"Error: {str(e)}")


if __name__ == "__main__":
    # game_id = "0022300001"
    # game_info = get_current_game_info(game_id, include_prior_states=True)
    # keys = ["game_id", "game_date", "game_time_est", "home", "away", "game_status"]

    # for key in keys:
    #     if key in game_info:
    #         print(f"{key}: {game_info[key]}")
    # print(game_info["game_states"][-1] == game_info["final_state"])
    # print(game_info["final_state"])
    # print(game_info["prior_states"])

    # update_database(game_info)

    update_full_season_database("2020-21", include_prior_states=True)
