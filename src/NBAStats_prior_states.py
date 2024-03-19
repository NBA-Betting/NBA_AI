import json
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytz
from dotenv import load_dotenv

try:
    from src.NBAStats_game_states import get_current_game_info
    from src.utils import (
        game_id_to_season,
        get_schedule,
        validate_date_format,
        validate_game_id,
    )
except ModuleNotFoundError:
    from NBAStats_game_states import get_current_game_info
    from utils import (
        game_id_to_season,
        get_schedule,
        validate_date_format,
        validate_game_id,
    )

load_dotenv()
PROJECT_ROOT = os.getenv("PROJECT_ROOT")

pd.set_option("display.max_columns", None)


def get_prior_states(game_id, game_date, home, away, force_update=False):
    """
    Get the prior states for a given game.

    This function first validates the game_id and game_date. If force_update is True, it updates the prior states
    regardless of whether they are already finalized. Otherwise, it loads the prior states from file and updates them.

    Parameters:
    game_id (str): The ID of the game.
    game_date (str): The date of the game in the format "YYYY-MM-DD".
    home (str): The home team.
    away (str): The away team.
    force_update (bool): Whether to force an update of the prior states.

    Returns:
    dict: The updated prior states.
    """

    # Validate the game_id and game_date
    validate_game_id(game_id)
    validate_date_format(game_date)

    # If force_update is True, update the prior states regardless of whether they are already finalized
    if force_update:
        return update_prior_states(game_id, game_date, home, away, {}, force_update)
    else:
        # Load the prior states from file
        prior_states_from_file = load_prior_states(game_id, game_date, home, away)

        # Update the prior states
        updated_prior_states = update_prior_states(
            game_id, game_date, home, away, prior_states_from_file
        )

        return updated_prior_states


def load_prior_states(game_id, game_date, home, away):
    """
    Load prior states from a JSON file.

    Parameters:
    game_id (str): The game ID.
    game_date (str): The game date.
    home (str): The home team.
    away (str): The away team.

    Returns:
    dict: The prior states if the file exists and contains "prior_states", otherwise an empty dictionary.
    """

    # Convert game ID to season
    season = game_id_to_season(game_id)

    # Construct the file path
    filepath = os.path.join(
        PROJECT_ROOT,
        "data",
        "NBAStats",
        season,
        game_date,
        f"{game_id}_{home}_{away}.json",
    )

    # Check if the file exists
    if not os.path.exists(filepath):
        return {}

    try:
        # Open and read the JSON file
        with open(filepath, "r") as json_file:
            data = json.load(json_file)

            # Get the "prior_states" from the data
            prior_states = data.get("prior_states", {})

            return prior_states
    except Exception:
        # If any error occurs while reading the file, return an empty dictionary
        return {}


def update_prior_states(
    game_id, game_date, home, away, prior_states_from_file, force_update=False
):
    """Update the prior states for a given game.

    Parameters:
    game_id (str): The ID of the game.
    game_date (str): The date of the game.
    home (str): The home team.
    away (str): The away team.
    prior_states_from_file (dict): The prior states from the file.
    force_update (bool, optional): Whether to force an update of the prior states. Defaults to False.

    Returns:
    dict: The updated prior states.
    """

    # If prior states are finalized and we're not forcing an update, no update is needed
    if prior_states_from_file.get("prior_states_finalized") and not force_update:
        return prior_states_from_file

    # Determine the season based on the game ID
    season = game_id_to_season(game_id, abbreviate=True)
    # Get the schedule for the season
    schedule = get_schedule(season, season_type="Regular Season")

    # Get the schedules for the home and away teams
    home_schedule = get_team_schedule(home, schedule)
    away_schedule = get_team_schedule(away, schedule)

    # Determine which games to add to the prior states
    if force_update:
        # If we're forcing an update, add all games
        home_prior_games_to_add = filter_team_schedule(
            game_date, home_schedule, "1900-01-01"
        )
        away_prior_games_to_add = filter_team_schedule(
            game_date, away_schedule, "1900-01-01"
        )
    else:
        # Otherwise, only add games that haven't been added yet
        home_start_date_str = prior_states_from_file.get(
            "home_prior_states_updated_to_date", "1900-01-01"
        )
        away_start_date_str = prior_states_from_file.get(
            "away_prior_states_updated_to_date", "1900-01-01"
        )

        # Convert the start dates to datetime objects
        home_start_date = datetime.strptime(home_start_date_str, "%Y-%m-%d")
        away_start_date = datetime.strptime(away_start_date_str, "%Y-%m-%d")

        # Add one day to the start dates
        home_start_date += timedelta(days=1)
        away_start_date += timedelta(days=1)

        # Convert the start dates back to strings
        home_start_date = home_start_date.strftime("%Y-%m-%d")
        away_start_date = away_start_date.strftime("%Y-%m-%d")

        # Filter the team schedules for games to add
        home_prior_games_to_add = filter_team_schedule(
            game_date,
            home_schedule,
            home_start_date,
        )
        away_prior_games_to_add = filter_team_schedule(
            game_date,
            away_schedule,
            away_start_date,
        )

    # Collect the prior states for the games to add
    file_system_season = game_id_to_season(game_id)
    home_new_prior_states = collect_prior_states(
        home_prior_games_to_add, file_system_season
    )
    away_new_prior_states = collect_prior_states(
        away_prior_games_to_add, file_system_season
    )

    # Combine the new prior states with the prior states from the file
    home_updated_prior_states = combine_prior_states(
        home_new_prior_states, prior_states_from_file.get("home_prior_final_states", [])
    )
    away_updated_prior_states = combine_prior_states(
        away_new_prior_states, prior_states_from_file.get("away_prior_final_states", [])
    )

    # Sort the updated prior states by game_date from least recent to most recent
    home_updated_prior_states = sorted(
        home_updated_prior_states, key=lambda game: game["game_date"]
    )
    away_updated_prior_states = sorted(
        away_updated_prior_states, key=lambda game: game["game_date"]
    )

    # Find the most recent prior final state date for the home and away teams
    home_prior_states_updated_to_date = (
        max(state["game_date"] for state in home_updated_prior_states)
        if home_updated_prior_states
        else None
    )
    away_prior_states_updated_to_date = (
        max(state["game_date"] for state in away_updated_prior_states)
        if away_updated_prior_states
        else None
    )

    # Check if all prior states are finalized
    all_prior_states_finalized = are_prior_states_finalized(
        home_updated_prior_states,
        away_updated_prior_states,
        home_schedule,
        away_schedule,
        game_date,
    )

    # Generate the feature set if all prior states are finalized
    if all_prior_states_finalized:
        feature_set = _generate_feature_set(
            home_updated_prior_states,
            away_updated_prior_states,
            home,
            away,
            game_date,
        )
    else:
        feature_set = {}

    # Return the updated prior states
    updated_prior_states = {
        "prior_states_finalized": all_prior_states_finalized,
        "home_prior_states_updated_to_date": home_prior_states_updated_to_date,
        "away_prior_states_updated_to_date": away_prior_states_updated_to_date,
        "feature_set": feature_set,
        "home_prior_final_states": home_updated_prior_states,
        "away_prior_final_states": away_updated_prior_states,
    }

    return updated_prior_states


def get_team_schedule(team, league_schedule):
    """
    Get the entire schedule for a single team.

    Parameters:
    team (str): The team for which to get the schedule.
    schedule (list): The schedule of games.

    Returns:
    list: The schedule for the team, sorted by gameDateTimeEst in ascending order.
    """

    # Convert the schedule list to a DataFrame
    schedule_df = pd.DataFrame(league_schedule)

    # Filter the games to include only those involving the team
    team_schedule_df = schedule_df[
        (schedule_df["homeTeam"] == team) | (schedule_df["awayTeam"] == team)
    ]

    # Sort the games by gameDateTimeEst in ascending order
    team_schedule_df = team_schedule_df.sort_values(by="gameDateTimeEst")

    # Convert the DataFrame back to a list of dictionaries
    team_schedule = team_schedule_df.to_dict("records")

    return team_schedule


def filter_team_schedule(game_date, team_schedule, start_date):
    """
    Filter the team schedule to only include games that are between the start_date (inclusive) and the game_date (exclusive),
    and are on or before the current date.

    Parameters:
    game_date (str): The date of the game, in the format "YYYY-MM-DD".
    team_schedule (list): The team's schedule. Each game is a dictionary that includes a "gameDateTimeEst" key.
    start_date (str): The start date, in the format "YYYY-MM-DD".

    Returns:
    list: The filtered team schedule.
    """

    # Convert game_date and start_date to datetime objects for comparison
    game_date = datetime.strptime(game_date, "%Y-%m-%d").date()
    start_date = datetime.strptime(start_date, "%Y-%m-%d").date()

    # Get the current date in EST
    now = datetime.now(pytz.timezone("US/Eastern")).date()

    # Create a new list of games that are between the start_date (inclusive) and the game_date (exclusive), and have already happened
    prior_games = [
        game
        for game in team_schedule
        if start_date
        <= datetime.strptime(game["gameDateTimeEst"][:10], "%Y-%m-%d").date()
        < game_date
        and datetime.strptime(game["gameDateTimeEst"][:10], "%Y-%m-%d").date() <= now
    ]

    return prior_games


def collect_prior_states(games, season):
    """
    This function collects the prior states of NBA games.

    Parameters:
    games (list): A list of dictionaries, each representing a game.
    season (str): The season in which the games were played.

    Returns:
    list: A list of prior states for each game.
    """
    collected_prior_states = []
    # Iterate over each game
    for game in games:
        # Extract the date from the game's datetime
        date_str = game["gameDateTimeEst"][:10]
        try:
            # Construct the filepath where the game's data is stored
            filepath = os.path.join(
                PROJECT_ROOT,
                "data",
                "NBAStats",
                season,
                date_str,
                f"{game['gameId']}_{game['homeTeam']}_{game['awayTeam']}.json",
            )

            # If the file doesn't exist, fetch the current game info
            if not os.path.exists(filepath):
                game_data = get_current_game_info(game["gameId"])
            else:
                # If the file exists, load the game data from the file
                with open(filepath, "r") as f:
                    game_data = json.load(f)

            # If the file is empty or the final_state is an empty dict, fetch the current game info
            if not game_data or not game_data.get("final_state"):
                game_data = get_current_game_info(game["gameId"])

            # If final_state is still empty, print a warning
            if not game_data.get("final_state"):
                # Convert date_str to a datetime object
                game_date = datetime.strptime(date_str, "%Y-%m-%d").date()

                # Get the current date in EST
                current_date = datetime.now(pytz.timezone("US/Eastern")).date()

                # If the game date is 2 or more days before the current date, print a warning
                if game_date <= current_date - timedelta(days=2):
                    print(
                        f"Warning: No final state for game {game['gameId']} from {date_str}."
                    )
            else:
                # If final_state is not empty, add it to the collected_prior_states list
                collected_prior_states.append(game_data["final_state"])
        except Exception as e:
            # If there's an issue with a game, print an error message
            print(f"Issue with game {game['gameId']} on {date_str}. Error: {e}")

    # Return the list of collected prior states
    return collected_prior_states


def combine_prior_states(new_prior_states, prior_states_from_file):
    """
    Combine two lists of game states, removing duplicates based on the 'game_id' key.
    If a 'game_id' is found in both lists, the state from 'new_prior_states' takes precedence.

    Parameters:
    new_prior_states (list): A list of dictionaries representing game states.
                             Each dictionary has a 'game_id' key.
    prior_states_from_file (list): A list of dictionaries representing game states.
                                   Each dictionary has a 'game_id' key.

    Returns:
    list: A combined list of dictionaries from 'new_prior_states' and 'prior_states_from_file',
          with no duplicates based on the 'game_id' key.
    """

    # Initialize an empty dictionary to store the combined states
    combined_states = {}

    # Iterate over the new_prior_states list
    for state in new_prior_states:
        # If the game_id is already in combined_states, print a warning
        if state["game_id"] in combined_states:
            print(
                f"Warning: Duplicate game_id {state['game_id']} found in new_prior_states"
            )
        # Add the state to combined_states, using the game_id as the key
        combined_states[state["game_id"]] = state

    # Iterate over the prior_states_from_file list
    for state in prior_states_from_file:
        # If the game_id is already in combined_states, print a warning
        if state["game_id"] in combined_states:
            print(
                f"Warning: Duplicate game_id {state['game_id']} found in prior_states_from_file"
            )
        # If the game_id is not in combined_states, add the state
        else:
            combined_states[state["game_id"]] = state

    # Convert combined_states back to a list and return it
    return list(combined_states.values())


def are_prior_states_finalized(
    home_updated_prior_states,
    away_updated_prior_states,
    home_schedule,
    away_schedule,
    game_date,
):
    """
    Determine if prior states are finalized.

    Parameters:
    home_updated_prior_states (list): List of updated prior states for the home team.
    away_updated_prior_states (list): List of updated prior states for the away team.
    home_schedule (list): List of scheduled games for the home team.
    away_schedule (list): List of scheduled games for the away team.
    game_date (str): The date of the game.

    Returns:
    bool: True if all prior states are finalized, False otherwise.
    """

    # Convert game_date to datetime object
    game_date = datetime.strptime(game_date, "%Y-%m-%d")

    # Extract game IDs from prior states and schedules
    home_prior_states_game_ids = set(
        [game["game_id"] for game in home_updated_prior_states]
    )
    away_prior_states_game_ids = set(
        [game["game_id"] for game in away_updated_prior_states]
    )
    home_schedule_game_ids = set(
        [
            game["gameId"]
            for game in home_schedule
            if datetime.strptime(game["gameDateTimeEst"][:10], "%Y-%m-%d") < game_date
        ]
    )
    away_schedule_game_ids = set(
        [
            game["gameId"]
            for game in away_schedule
            if datetime.strptime(game["gameDateTimeEst"][:10], "%Y-%m-%d") < game_date
        ]
    )

    # Compare game IDs
    home_diff = home_schedule_game_ids.difference(home_prior_states_game_ids)
    away_diff = away_schedule_game_ids.difference(away_prior_states_game_ids)

    # Determine if prior states are finalized
    home_prior_states_finalized = len(home_diff) == 0
    away_prior_states_finalized = len(away_diff) == 0
    all_prior_states_finalized = (
        home_prior_states_finalized and away_prior_states_finalized
    )

    return all_prior_states_finalized


def _generate_feature_set(
    home_prior_states, away_prior_states, home_team, away_team, game_date
):
    """
    Generate a set of features for a given game.

    Parameters:
    home_prior_states (list): List of prior states for the home team.
    away_prior_states (list): List of prior states for the away team.
    home_team (str): The home team's name.
    away_team (str): The away team's name.
    game_date (str): The date of the game.

    Returns:
    dict: A dictionary containing the generated features.
    """

    # Convert the prior states of home and away teams into DataFrames
    home_prior_states_df = pd.DataFrame(home_prior_states)
    away_prior_states_df = pd.DataFrame(away_prior_states)

    # If either DataFrame is empty, return an empty dictionary
    if home_prior_states_df.empty or away_prior_states_df.empty:
        return {}

    # Create basic features using the home and away teams' prior states
    basic_features_df = _create_basic_features(
        home_prior_states_df, away_prior_states_df, home_team, away_team
    )

    # Create contextual features using the home and away teams' prior states
    contextual_features_df = _create_contextual_features(
        home_prior_states_df, away_prior_states_df, home_team, away_team
    )

    # Create time decay features using the home and away teams' prior states
    time_decay_features_df = _create_time_decay_features(
        home_prior_states_df,
        away_prior_states_df,
        home_team,
        away_team,
        game_date,
        half_life=10,
    )

    # Create rest days and day of season features using the home and away teams' prior states
    rest_and_day_of_season_features_df = _create_rest_and_season_features(
        home_prior_states_df, away_prior_states_df, game_date
    )

    # Concatenate all the features into a single DataFrame
    features_df = pd.concat(
        [
            basic_features_df,
            contextual_features_df,
            time_decay_features_df,
            rest_and_day_of_season_features_df,
        ],
        axis=1,
    )

    # Convert NaN values to None
    features_df = features_df.where(pd.notnull(features_df), None)

    # Convert the DataFrame to a dictionary and return it
    features_dict = features_df.to_dict(orient="records")[0]

    return features_dict


def _create_basic_features(home_df, away_df, home_team_abbr, away_team_abbr):
    """
    Function to create basic features for a basketball game matchup.

    Parameters:
    home_df (pd.DataFrame): DataFrame containing the home team's game data.
    away_df (pd.DataFrame): DataFrame containing the away team's game data.
    home_team_abbr (str): Abbreviation of the home team's name.
    away_team_abbr (str): Abbreviation of the away team's name.

    Returns:
    pd.DataFrame: A DataFrame containing the calculated features for the matchup.
    """

    def calculate_team_features(df, team_abbr):
        """
        Helper function to calculate team features.

        Parameters:
        df (pd.DataFrame): DataFrame containing the team's game data.
        team_abbr (str): Abbreviation of the team's name.

        Returns:
        dict: A dictionary containing the calculated team features.
        """

        # Determine which games were home games
        is_home_game = df["home"] == team_abbr

        # Calculate team's score and opponent's score for each game
        team_score = np.where(is_home_game, df["home_score"], df["away_score"])
        opponent_score = np.where(is_home_game, df["away_score"], df["home_score"])

        # Determine which games were wins
        win = team_score > opponent_score

        # Calculate number of wins and games
        wins = win.sum()
        games = win.size

        # Calculate winning percentage
        winning_percentage = wins / games if games > 0 else 0

        # Calculate average points per game (PPG) and opponents' PPG
        ppg = team_score.mean()
        opp_ppg = opponent_score.mean()

        # Calculate net PPG (difference between PPG and opponents' PPG)
        net_ppg = ppg - opp_ppg

        return {
            "Win_Pct": winning_percentage,
            "PPG": ppg,
            "OPP_PPG": opp_ppg,
            "Net_PPG": net_ppg,
        }

    # Calculate team features for both teams
    home_features = calculate_team_features(home_df, home_team_abbr)
    away_features = calculate_team_features(away_df, away_team_abbr)

    # Combine features into a single DataFrame for the matchup
    matchup_features = {
        "Home_Win_Pct": home_features["Win_Pct"],
        "Home_PPG": home_features["PPG"],
        "Home_OPP_PPG": home_features["OPP_PPG"],
        "Home_Net_PPG": home_features["Net_PPG"],
        "Away_Win_Pct": away_features["Win_Pct"],
        "Away_PPG": away_features["PPG"],
        "Away_OPP_PPG": away_features["OPP_PPG"],
        "Away_Net_PPG": away_features["Net_PPG"],
    }

    # Calculate differences in features between home and away teams
    matchup_features.update(
        {
            "Win_Pct_Diff": matchup_features["Home_Win_Pct"]
            - matchup_features["Away_Win_Pct"],
            "PPG_Diff": matchup_features["Home_PPG"] - matchup_features["Away_PPG"],
            "OPP_PPG_Diff": matchup_features["Home_OPP_PPG"]
            - matchup_features["Away_OPP_PPG"],
            "Net_PPG_Diff": matchup_features["Home_Net_PPG"]
            - matchup_features["Away_Net_PPG"],
        }
    )

    return pd.DataFrame([matchup_features])


def _create_contextual_features(home_df, away_df, home_team, away_team):
    """
    Function to create contextual features for NBA games.

    Parameters:
    home_df (DataFrame): DataFrame containing home team data
    away_df (DataFrame): DataFrame containing away team data
    home_team (str): Home team abbreviation
    away_team (str): Away team abbreviation

    Returns:
    DataFrame: A DataFrame containing the calculated contextual features
    """

    def calculate_contextual_features(df, team_abbr, context):
        """
        Helper function to calculate contextual features for a given team and context.

        Parameters:
        df (DataFrame): DataFrame containing team data
        team_abbr (str): Team abbreviation
        context (str): Context ('home' or 'away')

        Returns:
        dict: A dictionary containing the calculated contextual features
        """

        # Filter DataFrame based on context
        df_filtered = df[df[context] == team_abbr]

        # Define team_score and opponent_score based on the context
        score_col = f"{context}_score"
        opp_context = "away" if context == "home" else "home"
        opp_score_col = f"{opp_context}_score"

        # Calculate team and opponent scores
        team_score = df_filtered[score_col]
        opponent_score = df_filtered[opp_score_col]

        # Calculate wins, games, winning percentage, points per game (ppg), opponent's ppg, and net ppg
        win = team_score > opponent_score
        wins = win.sum()
        games = len(df_filtered)
        winning_percentage = wins / games if games > 0 else 0
        ppg = team_score.mean()
        opp_ppg = opponent_score.mean()
        net_ppg = ppg - opp_ppg

        # Return calculated features as a dictionary
        return {
            "Win_Pct": winning_percentage,
            "PPG": ppg,
            "OPP_PPG": opp_ppg,
            "Net_PPG": net_ppg,
        }

    # Calculate contextual features for home and away teams in their respective contexts
    home_features_home_context = calculate_contextual_features(
        home_df, home_team, "home"
    )
    away_features_away_context = calculate_contextual_features(
        away_df, away_team, "away"
    )

    # Combine the calculated features into a dictionary
    contextual_features = {
        "Home_Win_Pct_Home": home_features_home_context["Win_Pct"],
        "Home_PPG_Home": home_features_home_context["PPG"],
        "Home_OPP_PPG_Home": home_features_home_context["OPP_PPG"],
        "Home_Net_PPG_Home": home_features_home_context["Net_PPG"],
        "Away_Win_Pct_Away": away_features_away_context["Win_Pct"],
        "Away_PPG_Away": away_features_away_context["PPG"],
        "Away_OPP_PPG_Away": away_features_away_context["OPP_PPG"],
        "Away_Net_PPG_Away": away_features_away_context["Net_PPG"],
        "Win_Pct_Home_Away_Diff": home_features_home_context["Win_Pct"]
        - away_features_away_context["Win_Pct"],
        "PPG_Home_Away_Diff": home_features_home_context["PPG"]
        - away_features_away_context["PPG"],
        "OPP_PPG_Home_Away_Diff": home_features_home_context["OPP_PPG"]
        - away_features_away_context["OPP_PPG"],
        "Net_PPG_Home_Away_Diff": home_features_home_context["Net_PPG"]
        - away_features_away_context["Net_PPG"],
    }

    # Return the contextual features as a single-row DataFrame
    return pd.DataFrame([contextual_features])


def _create_time_decay_features(
    home_df, away_df, home_team, away_team, game_date, half_life=10
):
    """
    Create time decay features for NBA statistics.

    Parameters:
    home_df (DataFrame): DataFrame containing home team data.
    away_df (DataFrame): DataFrame containing away team data.
    home_team (str): Home team abbreviation.
    away_team (str): Away team abbreviation.
    game_date (str): Date of the game.
    half_life (int, optional): Half-life for time decay. Defaults to 10.

    Returns:
    DataFrame: A DataFrame containing time decay features.
    """

    def calculate_time_decayed_features(df, team_abbr, target_date, half_life):
        """
        Calculate time decayed features for a given team.

        Parameters:
        df (DataFrame): DataFrame containing team data.
        team_abbr (str): Team abbreviation.
        target_date (str): Date of the game.
        half_life (int): Half-life for time decay.

        Returns:
        dict: A dictionary containing time decayed features.
        """

        # Convert game_date to datetime and calculate days before the game
        game_date = pd.to_datetime(df["game_date"])
        target_date = pd.to_datetime(target_date)
        days_before_game = (target_date - game_date).dt.days

        # Calculate decay rate from half-life
        lambda_decay = np.log(2) / half_life

        # Calculate decay weights using the half-life
        decay_weight = np.exp(-lambda_decay * days_before_game)

        # Assign team and opponent scores based on home/away status
        is_home_game = df["home"] == team_abbr
        team_score = np.where(is_home_game, df["home_score"], df["away_score"])
        opponent_score = np.where(is_home_game, df["away_score"], df["home_score"])

        # Calculate win/loss and apply weights
        win = team_score > opponent_score
        weighted_wins = (win * decay_weight).sum()
        total_weight = decay_weight.sum()

        # Calculate time-decayed metrics
        time_decayed_winning_percentage = (
            weighted_wins / total_weight if total_weight > 0 else 0
        )
        time_decayed_ppg = (
            (team_score * decay_weight).sum() / total_weight if total_weight > 0 else 0
        )
        time_decayed_opp_ppg = (
            (opponent_score * decay_weight).sum() / total_weight
            if total_weight > 0
            else 0
        )
        time_decayed_ppg_diff = time_decayed_ppg - time_decayed_opp_ppg

        return {
            "Win_Pct": time_decayed_winning_percentage,
            "PPG": time_decayed_ppg,
            "OPP_PPG": time_decayed_opp_ppg,
            "Net_PPG": time_decayed_ppg_diff,
        }

    # Calculate time-decayed features for both teams
    home_features_time_decay = calculate_time_decayed_features(
        home_df, home_team, game_date, half_life
    )
    away_features_time_decay = calculate_time_decayed_features(
        away_df, away_team, game_date, half_life
    )

    # Construct the features DataFrame by combining features
    time_decay_features = {
        "Time_Decay_Home_Win_Pct": home_features_time_decay["Win_Pct"],
        "Time_Decay_Home_PPG": home_features_time_decay["PPG"],
        "Time_Decay_Home_OPP_PPG": home_features_time_decay["OPP_PPG"],
        "Time_Decay_Home_Net_PPG": home_features_time_decay["Net_PPG"],
        "Time_Decay_Away_Win_Pct": away_features_time_decay["Win_Pct"],
        "Time_Decay_Away_PPG": away_features_time_decay["PPG"],
        "Time_Decay_Away_OPP_PPG": away_features_time_decay["OPP_PPG"],
        "Time_Decay_Away_Net_PPG": away_features_time_decay["Net_PPG"],
        "Time_Decay_Win_Pct_Diff": home_features_time_decay["Win_Pct"]
        - away_features_time_decay["Win_Pct"],
        "Time_Decay_PPG_Diff": home_features_time_decay["PPG"]
        - away_features_time_decay["PPG"],
        "Time_Decay_OPP_PPG_Diff": home_features_time_decay["OPP_PPG"]
        - away_features_time_decay["OPP_PPG"],
        "Time_Decay_Net_PPG_Diff": home_features_time_decay["Net_PPG"]
        - away_features_time_decay["Net_PPG"],
    }

    return pd.DataFrame([time_decay_features])


def _create_rest_and_season_features(home_df, away_df, game_date):
    """
    Create features related to rest days and season day for both home and away teams.

    Parameters:
    home_df (pd.DataFrame): DataFrame containing the home team's game data.
    away_df (pd.DataFrame): DataFrame containing the away team's game data.
    game_date (str): The date of the game.

    Returns:
    pd.DataFrame: A DataFrame containing the calculated features.
    """

    def calculate_rest_and_season_day(df, target_date):
        """
        Calculate rest days, day of season, and average game frequency for a team.

        Parameters:
        df (pd.DataFrame): DataFrame containing the team's game data.
        target_date (str): The date of the game.

        Returns:
        tuple: A tuple containing rest days, day of season, and average game frequency.
        """
        # Convert target_date to datetime
        target_date = pd.to_datetime(target_date)

        # Find the start date of the season
        team_season_start = pd.to_datetime(df["game_date"].min())

        # Filter out games that happened after the target_date
        previous_games = df[pd.to_datetime(df["game_date"]) < target_date]

        # Calculate rest days
        if previous_games.empty:
            last_game_date = team_season_start
            rest_days = (
                (target_date - last_game_date).days
                if target_date != team_season_start
                else 0
            )
        else:
            last_game_date = pd.to_datetime(previous_games["game_date"].max())
            rest_days = (target_date - last_game_date).days

        # Calculate day of season
        day_of_season = (target_date - team_season_start).days

        # Calculate average game frequency over the last 5, 10, and 30 days
        rest_play_counts = []
        for days in [5, 10, 30]:
            start_date = max(target_date - pd.Timedelta(days=days), team_season_start)
            date_range = pd.date_range(
                start=start_date, end=target_date - pd.Timedelta(days=1)
            )
            rest_play_count = 0
            for day in date_range:
                rest_play_count += (
                    1
                    if day in pd.to_datetime(previous_games["game_date"]).values
                    else -1
                )
            rest_play_counts.append(rest_play_count)

        avg_rest_play_count = np.mean(rest_play_counts)

        return rest_days, day_of_season, avg_rest_play_count

    # Calculate features for home team
    home_rest_days, home_day_of_season, home_avg_rest_play_count = (
        calculate_rest_and_season_day(home_df, game_date)
    )

    # Calculate features for away team
    away_rest_days, away_day_of_season, away_avg_rest_play_count = (
        calculate_rest_and_season_day(away_df, game_date)
    )

    # Combine features into a dictionary
    rest_and_day_of_season_features = {
        "Day_of_Season": (home_day_of_season + away_day_of_season) / 2,
        "Home_Rest_Days": home_rest_days,
        "Home_Game_Freq": home_avg_rest_play_count,
        "Away_Rest_Days": away_rest_days,
        "Away_Game_Freq": away_avg_rest_play_count,
        "Rest_Days_Diff": home_rest_days - away_rest_days,
        "Game_Freq_Diff": home_avg_rest_play_count - away_avg_rest_play_count,
    }

    # Convert dictionary to DataFrame and return
    return pd.DataFrame([rest_and_day_of_season_features])


if __name__ == "__main__":

    def print_prior_states_info(prior_states):
        print()
        print("Finalized:", prior_states["prior_states_finalized"])
        print(
            "Home Prior States Updated To Date:",
            prior_states["home_prior_states_updated_to_date"],
        )
        print(
            "Away Prior States Updated To Date:",
            prior_states["away_prior_states_updated_to_date"],
        )
        print(
            "Home Prior Final States Length:",
            len(prior_states["home_prior_final_states"]),
        )
        print(
            "Away Prior Final States Length:",
            len(prior_states["away_prior_final_states"]),
        )
        print("----- Feature Set -----")
        if not prior_states["feature_set"]:
            print("No feature set generated.")
        for k, v in prior_states["feature_set"].items():
            print(f"{k}:{v}")
        print()

    # First Day of Season Game
    p1 = get_prior_states("0022200001", "2022-10-18", "BOS", "PHI", force_update=True)
    print_prior_states_info(p1)

    # Late Season Game
    p2 = get_prior_states("0022200919", "2023-02-27", "CHA", "DET", force_update=True)
    print_prior_states_info(p2)

    # Future Game
    p3 = get_prior_states("0022301170", "2024-04-11", "SAC", "NOP", force_update=True)
    print_prior_states_info(p3)
