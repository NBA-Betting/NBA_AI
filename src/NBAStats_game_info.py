import json
import os

from dotenv import load_dotenv
from tqdm import tqdm

try:
    from src.NBAStats_game_states import get_current_game_info
    from src.NBAStats_prior_states import get_prior_states
    from src.utils import (
        game_id_to_season,
        get_schedule,
        validate_game_id,
        validate_season_format,
    )
except ModuleNotFoundError:
    from NBAStats_game_states import get_current_game_info
    from NBAStats_prior_states import get_prior_states
    from utils import (
        game_id_to_season,
        get_schedule,
        validate_game_id,
        validate_season_format,
    )

load_dotenv()
PROJECT_ROOT = os.getenv("PROJECT_ROOT")


def get_game_info(
    game_id, include_prior_states, save_to_database, force_update_prior_states
):
    """
    Fetches game information for a given game ID. Optionally includes prior states,
    saves the information to a database, and forces an update of prior states.

    Args:
        game_id (str): The ID of the game to fetch information for.
        include_prior_states (bool): Whether to include prior states in the returned information.
        save_to_database (bool): Whether to save the fetched information to a database.
        force_update_prior_states (bool): Whether to force an update of prior states.

    Returns:
        dict: A dictionary containing the fetched game information.
    """
    # Validate the provided game ID
    validate_game_id(game_id)

    # Fetch the current game information
    game_info = get_current_game_info(game_id)

    # If requested, fetch and include prior states
    if include_prior_states:
        prior_states = get_prior_states(
            game_id,
            game_info["game_date"],
            game_info["home"],
            game_info["away"],
            force_update=force_update_prior_states,
        )
        game_info["prior_states"] = prior_states

    # If requested, save the fetched information to a database
    if save_to_database:
        update_database(game_info, print_updated=False)

    # Return the fetched game information
    return game_info


def update_full_season_database(
    season,
    season_type="Regular Season",
    include_prior_states=True,
    force_update_prior_states=False,
):
    """
    Updates the database with game information for a full season.

    Args:
        season (str): The season to fetch information for, in abbreviated format (e.g., '2021').
        season_type (str, optional): The type of the season. Defaults to "Regular Season".
        include_prior_states (bool, optional): Whether to include prior states in the fetched information. Defaults to True.
        force_update_prior_states (bool, optional): Whether to force an update of prior states. Defaults to False.
    """
    # Validate the provided season format
    validate_season_format(season, abbreviated=True)

    # Fetch the schedule for the specified season and season type
    schedule = get_schedule(season, season_type)

    # Sort the games by 'gameDateTimeEst' from least recent to most recent
    games_sorted = sorted(schedule, key=lambda game: game["gameDateTimeEst"])

    # Extract the game_ids from the sorted games
    game_ids = [game["gameId"] for game in games_sorted]

    # Loop over each game ID, fetching and storing game information
    for game_id in tqdm(
        game_ids, desc="Updating database", unit="game", dynamic_ncols=True
    ):
        get_game_info(
            game_id,
            include_prior_states=include_prior_states,
            force_update_prior_states=force_update_prior_states,
            save_to_database=True,
        )


def update_multiple_games(
    game_ids, include_prior_states=True, force_update_prior_states=False
):
    """
    This function updates the database with game information for multiple games.

    Args:
        game_ids (list): A list of game IDs to fetch information for.
        include_prior_states (bool, optional): If True, the function will include prior states in the fetched information. Defaults to True.
        force_update_prior_states (bool, optional): If True, the function will force an update of prior states, regardless of whether they have been updated recently. Defaults to False.

    Returns:
        None. The function updates the database directly with the fetched game information.
    """
    # Loop over each game ID
    for game_id in tqdm(
        game_ids, desc="Updating database", unit="game", dynamic_ncols=True
    ):
        # Fetch and store game information for each game ID
        get_game_info(
            game_id,
            include_prior_states=include_prior_states,
            force_update_prior_states=force_update_prior_states,
            save_to_database=True,  # The fetched game information is saved directly to the database
        )


def update_and_return_season_info(season, season_type="Regular Season"):
    """
    This function updates and returns the season information.

    Args:
        season (str): The season to fetch information for.
        season_type (str, optional): The type of the season. Defaults to "Regular Season".

    Returns:
        dict: A dictionary containing the season information.
    """
    # Validate season format
    validate_season_format(season, abbreviated=False)

    # Get the schedule for the season
    abbreviated_season = season[:5] + season[-2:]
    schedule = get_schedule(abbreviated_season, season_type)

    all_game_ids = {game["gameId"] for game in schedule}
    finalized_game_ids = set()

    # Loop over each game in the schedule
    for game in tqdm(schedule, unit="game", dynamic_ncols=True):
        # Create the game file path
        game_filepath = f"{PROJECT_ROOT}/data/NBAStats/{season}/{game['gameDateTimeEst'][:10]}/{game['gameId']}_{game['homeTeam']}_{game['awayTeam']}.json"
        # Check if the game file exists
        if os.path.exists(game_filepath):
            with open(game_filepath, "r") as json_file:
                game_info = json.load(json_file)

            # Game finalization checks
            game_status_check = game_info.get("game_status") == "Completed"
            # Check if the final state is the last game state
            final_state_check = False
            if game_info.get("game_states") and len(game_info.get("game_states")) > 0:
                final_state_check = (
                    game_info.get("final_state") == game_info["game_states"][-1]
                )
            # Check if the prior states are finalized
            prior_states_check = game_info.get("prior_states", {}).get(
                "prior_states_finalized", False
            )
            # If all checks pass, add the game id to the finalized game ids
            if game_status_check and final_state_check and prior_states_check:
                finalized_game_ids.add(game["gameId"])

    # Create the season info dictionary
    season_info = {
        "season": season,
        "season_finalized": finalized_game_ids == all_game_ids,
        "unfinalized_game_ids": list(all_game_ids - finalized_game_ids),
        "all_game_ids": list(all_game_ids),
        "finalized_game_ids": list(finalized_game_ids),
    }

    # Save the season info to a file
    season_info_filepath = f"{PROJECT_ROOT}/data/NBAStats/{season}/season_info.json"
    with open(season_info_filepath, "w") as json_file:
        json.dump(season_info, json_file)

    return season_info


def update_database(game_data_dict, print_updated=True):
    """
    Update the game data in the database.

    If the file for the game data already exists, this function will update the existing data with game_data_dict.
    If the file does not exist, this function will create a new file with game_data_dict.
    If the existing file contains invalid JSON, it will be replaced with game_data_dict.

    Parameters:
    game_data_dict (dict): The game data to update. Must contain the keys "game_id", "game_date", "home", and "away".
    print_updated (bool): Whether to print a message indicating the file that was updated. Defaults to True.

    Returns:
    None
    """

    # Extract necessary information from game_data_dict
    game_id = game_data_dict["game_id"]
    game_date = game_data_dict["game_date"]
    home = game_data_dict["home"]
    away = game_data_dict["away"]

    # Determine the season based on the game_id
    season = game_id_to_season(game_id)

    # Construct the directory path and filename
    directory = f"{PROJECT_ROOT}/data/NBAStats/{season}/{game_date}"
    filename = f"{directory}/{game_id}_{home}_{away}.json"

    # Create the directory if it does not exist
    os.makedirs(directory, exist_ok=True)

    # If the file exists, try to load the existing data and update it
    if os.path.exists(filename):
        try:
            with open(filename, "r") as json_file:
                existing_data = json.load(json_file)
            # Update the existing data with game_data_dict
            existing_data.update(game_data_dict)
        except json.JSONDecodeError:
            # If the existing JSON is invalid, replace it with game_data_dict
            existing_data = game_data_dict
    else:
        # If the file does not exist, use game_data_dict as the data to write
        existing_data = game_data_dict

    # Write the updated data back to the file
    with open(filename, "w") as json_file:
        json.dump(existing_data, json_file, indent=4)

    # Print a message indicating the file that was updated, if print_updated is True
    if print_updated:
        print(f"Updated: {filename}")


if __name__ == "__main__":
    pass
    # get_game_info(
    #     "0022200919",
    #     include_prior_states=True,
    #     save_to_database=True,
    #     force_update=True,
    # )

    # update_full_season_database("2023-24")

    # check_season_info = update_and_return_season_info("2023-2024")
