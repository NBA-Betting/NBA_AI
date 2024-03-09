import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from utils import (
    game_id_to_season,
    get_schedule,
    lookup_basic_game_info,
    validate_date_format,
    validate_game_id,
)

load_dotenv()
PROJECT_ROOT = os.getenv("PROJECT_ROOT")

pd.set_option("display.max_columns", None)


def get_prior_states(game_id):
    validate_game_id(game_id)
    game_info = lookup_basic_game_info(game_id)
    game_date = game_info["game_date"]
    home = game_info["home"]
    away = game_info["away"]
    original_prior_states = load_prior_states(game_id, game_date, home, away)
    current_prior_states = update_prior_states(
        game_id, game_date, home, away, original_prior_states
    )
    return current_prior_states


def load_prior_states(game_id, game_date, home, away):
    validate_game_id(game_id)
    validate_date_format(game_date)

    season = game_id_to_season(game_id)

    filepath = f"{PROJECT_ROOT}/data/NBAStats/{season}/{game_date}/{game_id}_{home}_{away}.json"

    try:
        with open(filepath, "r") as json_file:
            data = json.load(json_file)
            prior_states = data.get("prior_states", None)

            if prior_states is not None:
                return prior_states
    except Exception as e:
        pass

    return {}


def update_prior_states(game_id, game_date, home, away, prior_states):
    validate_game_id(game_id)
    validate_date_format(game_date)

    # basic prior_states dict structure
    if prior_states == {}:
        prior_states = {
            "prior_states_finalized": False,  # True if game is completed, False otherwise
            "home_prior_states_updated_to_date": None,  # date of which prior games are included. Not a date of last update.
            "away_prior_states_updated_to_date": None,  # date of which prior games are included. Not a date of last update.
            "home_prior_final_states": [],
            "away_prior_final_states": [],
            "feature_set": [],
        }
    # elif prior_states["prior_states_finalized"]:
    #     return prior_states  # No update needed if prior states are finalized

    # Determine games to include in prior states
    season = game_id_to_season(game_id, abbreviate=True)
    file_system_season = game_id_to_season(game_id, abbreviate=False)
    schedule = get_schedule(season)
    home_prior_games, away_prior_games = _get_scheduled_prior_games(
        game_date, home, away, schedule
    )

    # Filter prior games to include only games that are past the prior_states_updated_to_date
    home_prior_games_to_add = _filter_prior_games(
        home_prior_games, prior_states["home_prior_states_updated_to_date"]
    )
    away_prior_games_to_add = _filter_prior_games(
        away_prior_games, prior_states["away_prior_states_updated_to_date"]
    )

    # Load and add prior states from home_prior_games_to_add and away_prior_games_to_add
    if home_prior_games_to_add:
        prior_states = _add_prior_states(
            home_prior_games_to_add, "home", prior_states, file_system_season
        )
        # Find the most recent prior final state date for home
        prior_states["home_prior_states_updated_to_date"] = prior_states[
            "home_prior_final_states"
        ][0]["game_date"]
        # Extract the gameIds from home_prior_games
        needed_home_game_ids = [game["gameId"] for game in home_prior_games]
        # Extract the game_ids from home_prior_states
        existing_home_game_ids = [
            game["game_id"] for game in prior_states["home_prior_final_states"]
        ]
        # Check if all needed home game ids are in existing_home_game_ids
        home_prior_states_finalized = all(
            game_id in existing_home_game_ids for game_id in needed_home_game_ids
        )
    else:
        home_prior_states_finalized = True

    if away_prior_games_to_add:
        prior_states = _add_prior_states(
            away_prior_games_to_add, "away", prior_states, file_system_season
        )
        # Find the most recent prior final state date for away
        prior_states["away_prior_states_updated_to_date"] = prior_states[
            "away_prior_final_states"
        ][0]["game_date"]
        # Extract the gameIds from away_prior_games
        needed_away_game_ids = [game["gameId"] for game in away_prior_games]
        # Extract the game_ids from away_prior_states
        existing_away_game_ids = [
            game["game_id"] for game in prior_states["away_prior_final_states"]
        ]
        # Check if all needed away game ids are in existing_away_game_ids
        away_prior_states_finalized = all(
            game_id in existing_away_game_ids for game_id in needed_away_game_ids
        )
    else:
        away_prior_states_finalized = True

    # Check if both home and away prior states are finalized
    prior_states["prior_states_finalized"] = (
        home_prior_states_finalized and away_prior_states_finalized
    )

    # Generate feature set
    if prior_states["prior_states_finalized"] and not prior_states["feature_set"]:
        prior_states["feature_set"] = _generate_feature_set(
            prior_states, home, away, game_date
        )

    return prior_states


def _get_scheduled_prior_games(
    game_date, home, away, schedule, season_type="Regular Season"
):
    """
    This function returns the prior games for the home and away teams based on the season type.

    Parameters:
    game_date (str): The date of the game in the format "YYYY-MM-DD".
    home (str): The name of the home team.
    away (str): The name of the away team.
    schedule (list): The list of games. Each game is a dictionary with keys "gameDateTimeEst", "homeTeam", "awayTeam", and "GameId".
    season_type (str): The type of the season. It can be "Pre Season", "Regular Season", "All-Star", or "Post Season". Default is "Regular Season".

    Returns:
    tuple: A tuple containing two lists. The first list contains the prior games of the home team and the second list contains the prior games of the away team.
    """

    # Convert the game_date from string to datetime object
    game_date = datetime.strptime(game_date, "%Y-%m-%d")
    home_prior_games = []
    away_prior_games = []

    # Mapping of season type codes to season types
    season_type_codes = {
        "001": "Pre Season",
        "002": "Regular Season",
        "003": "All-Star",
        "004": "Post Season",
    }

    # Loop through each game in the schedule
    for game in schedule:
        # Convert the game date from string to datetime object
        game_datetime = datetime.strptime(game["gameDateTimeEst"][:10], "%Y-%m-%d")

        # Check if the game date is before the input game_date
        if game_datetime < game_date:
            # Check if the game's season type matches the input season_type
            if season_type_codes.get(game["gameId"][:3]) == season_type:
                # If the home team played in this game, add it to home_prior_games
                if home in (game["homeTeam"], game["awayTeam"]):
                    home_prior_games.append(game)
                # If the away team played in this game, add it to away_prior_games
                if away in (game["homeTeam"], game["awayTeam"]):
                    away_prior_games.append(game)

    # Return the prior games of the home and away teams
    return home_prior_games, away_prior_games


def _filter_prior_games(prior_games, prior_states_updated_to_date_str):
    # Convert prior_states_updated_to_date to datetime for comparison
    prior_states_updated_to_date = None
    if prior_states_updated_to_date_str:
        prior_states_updated_to_date = datetime.strptime(
            prior_states_updated_to_date_str, "%Y-%m-%d"
        )

    # Only include games that are past the prior_states_updated_to_date
    prior_games_to_add = [
        game
        for game in prior_games
        if not prior_states_updated_to_date
        or datetime.strptime(game["gameDateTimeEst"][:10], "%Y-%m-%d")
        > prior_states_updated_to_date
    ]

    return prior_games_to_add


def _add_prior_states(games_to_add, team_type, prior_states, season):
    # Add the final state from the filepath to the prior_final_states
    for game in games_to_add:
        date_str = game["gameDateTimeEst"][:10]
        filepath = f"{PROJECT_ROOT}/data/NBAStats/{season}/{date_str}/{game['gameId']}_{game['homeTeam']}_{game['awayTeam']}.json"

        # Check if the file exists
        if not os.path.exists(filepath):
            continue

        with open(filepath, "r") as f:
            game_data = json.load(f)

        # Check if the file is empty or the final_state is an empty dict
        if not game_data or not game_data.get("final_state"):
            continue

        final_state = game_data["final_state"]

        prior_states[f"{team_type}_prior_final_states"].append(final_state)

    # Sort prior states by date most recent to least recent
    prior_states[f"{team_type}_prior_final_states"].sort(
        key=lambda x: datetime.strptime(x["game_date"], "%Y-%m-%d"), reverse=True
    )

    return prior_states


def _generate_feature_set(prior_states, home_team, away_team, game_date):
    # Create DataFrames from prior_states
    home_df = pd.DataFrame(prior_states["home_prior_final_states"])
    away_df = pd.DataFrame(prior_states["away_prior_final_states"])

    if home_df.empty or away_df.empty:
        return []

    # Sort the datasets by 'game_date' to ensure chronological order
    home_df_sorted = home_df.sort_values(by="game_date")
    away_df_sorted = away_df.sort_values(by="game_date")

    # Basic Features #######
    home_features_basic = calculate_team_features(home_df_sorted, home_team)
    away_features_basic = calculate_team_features(away_df_sorted, away_team)

    # Mapping the recalculated features to the new structure for the upcoming game's DataFrame
    basic_features_df = pd.DataFrame(
        {
            "Home_Winning_Percentage": [home_features_basic["Winning_Percentage"]],
            "Home_PPG": [home_features_basic["PPG"]],
            "Home_OPP_PPG": [home_features_basic["OPP_PPG"]],
            "Home_PPG_Diff": [home_features_basic["PPG_Diff"]],
            "Home_WL_Streak": [home_features_basic["WL_Streak"]],
            "Away_Winning_Percentage": [away_features_basic["Winning_Percentage"]],
            "Away_PPG": [away_features_basic["PPG"]],
            "Away_OPP_PPG": [away_features_basic["OPP_PPG"]],
            "Away_PPG_Diff": [away_features_basic["PPG_Diff"]],
            "Away_WL_Streak": [away_features_basic["WL_Streak"]],
            "Winning_Percentage_Diff": [
                home_features_basic["Winning_Percentage"]
                - away_features_basic["Winning_Percentage"]
            ],
            "PPG_Diff": [home_features_basic["PPG"] - away_features_basic["PPG"]],
            "OPP_PPG_Diff": [
                home_features_basic["OPP_PPG"] - away_features_basic["OPP_PPG"]
            ],
            "PPG_Diff_Diff": [
                home_features_basic["PPG_Diff"] - away_features_basic["PPG_Diff"]
            ],
            "WL_Streak_Diff": [
                home_features_basic["WL_Streak"] - away_features_basic["WL_Streak"]
            ],
        }
    )

    # Context-Specific Features #######
    home_features_home_context = calculate_context_specific_features(
        home_df_sorted, home_team, "home"
    )
    away_features_away_context = calculate_context_specific_features(
        away_df_sorted, away_team, "away"
    )

    # Construct the features DataFrame
    contextual_features_df = pd.DataFrame(
        {
            "Home_Winning_Percentage_Home": [
                home_features_home_context["Winning_Percentage"]
            ],
            "Home_PPG_Home": [home_features_home_context["PPG"]],
            "Home_OPP_PPG_Home": [home_features_home_context["OPP_PPG"]],
            "Home_PPG_Diff_Home": [home_features_home_context["PPG_Diff"]],
            "Home_WL_Streak_Home": [home_features_home_context["WL_Streak"]],
            "Away_Winning_Percentage_Away": [
                away_features_away_context["Winning_Percentage"]
            ],
            "Away_PPG_Away": [away_features_away_context["PPG"]],
            "Away_OPP_PPG_Away": [away_features_away_context["OPP_PPG"]],
            "Away_PPG_Diff_Away": [away_features_away_context["PPG_Diff"]],
            "Away_WL_Streak_Away": [away_features_away_context["WL_Streak"]],
            "Winning_Percentage_Home_Away_Diff": [
                home_features_home_context["Winning_Percentage"]
                - away_features_away_context["Winning_Percentage"]
            ],
            "PPG_Home_Away_Diff": [
                home_features_home_context["PPG"] - away_features_away_context["PPG"]
            ],
            "OPP_PPG_Home_Away_Diff": [
                home_features_home_context["OPP_PPG"]
                - away_features_away_context["OPP_PPG"]
            ],
            "PPG_Diff_Home_Away_Diff": [
                home_features_home_context["PPG_Diff"]
                - away_features_away_context["PPG_Diff"]
            ],
            "WL_Streak_Home_Away_Diff": [
                home_features_home_context["WL_Streak"]
                - away_features_away_context["WL_Streak"]
            ],
        }
    )

    # Time Decay Features #######
    home_features_time_decay = calculate_time_decayed_features(
        home_df_sorted, home_team, game_date
    )
    away_features_time_decay = calculate_time_decayed_features(
        away_df_sorted, away_team, game_date
    )

    # Construct the features DataFrame
    time_decay_features_df = pd.DataFrame(
        {
            "Home_Time_Decayed_Winning_Percentage": [
                home_features_time_decay["Time_Decayed_Winning_Percentage"]
            ],
            "Home_Time_Decayed_PPG": [home_features_time_decay["Time_Decayed_PPG"]],
            "Home_Time_Decayed_OPP_PPG": [
                home_features_time_decay["Time_Decayed_OPP_PPG"]
            ],
            "Home_Time_Decayed_PPG_Diff": [
                home_features_time_decay["Time_Decayed_PPG_Diff"]
            ],
            "Away_Time_Decayed_Winning_Percentage": [
                away_features_time_decay["Time_Decayed_Winning_Percentage"]
            ],
            "Away_Time_Decayed_PPG": [away_features_time_decay["Time_Decayed_PPG"]],
            "Away_Time_Decayed_OPP_PPG": [
                away_features_time_decay["Time_Decayed_OPP_PPG"]
            ],
            "Away_Time_Decayed_PPG_Diff": [
                away_features_time_decay["Time_Decayed_PPG_Diff"]
            ],
            "Time_Decayed_Winning_Percentage_Diff": [
                home_features_time_decay["Time_Decayed_Winning_Percentage"]
                - away_features_time_decay["Time_Decayed_Winning_Percentage"]
            ],
            "Time_Decayed_PPG_Diff": [
                home_features_time_decay["Time_Decayed_PPG"]
                - away_features_time_decay["Time_Decayed_PPG"]
            ],
            "Time_Decayed_OPP_PPG_Diff": [
                home_features_time_decay["Time_Decayed_OPP_PPG"]
                - away_features_time_decay["Time_Decayed_OPP_PPG"]
            ],
            "Time_Decayed_PPG_Diff_Diff": [
                home_features_time_decay["Time_Decayed_PPG_Diff"]
                - away_features_time_decay["Time_Decayed_PPG_Diff"]
            ],
        }
    )

    # Rest Days and Day of Season Features #######

    home_rest_days, home_day_of_season, home_avg_rest_play_count = (
        calculate_rest_and_season_day(home_df_sorted, game_date)
    )

    away_rest_days, away_day_of_season, away_avg_rest_play_count = (
        calculate_rest_and_season_day(away_df_sorted, game_date)
    )

    rest_and_day_of_season_features_df = pd.DataFrame(
        {
            "Home_Rest_Days": [home_rest_days],
            "Home_Day_of_Season": [home_day_of_season],
            "Home_Avg_Rest_Play_Count": [home_avg_rest_play_count],
            "Away_Rest_Days": [away_rest_days],
            "Away_Day_of_Season": [away_day_of_season],
            "Away_Avg_Rest_Play_Count": [away_avg_rest_play_count],
            "Rest_Days_Diff": [home_rest_days - away_rest_days],
            "Avg_Rest_Play_Count_Diff": home_avg_rest_play_count
            - away_avg_rest_play_count,
        }
    )

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

    # Convert DataFrame to dictionary
    features_dict = features_df.to_dict(orient="records")

    return features_dict


def calculate_team_features(df, team_abbr):
    # Determine if the team was playing at home or away, then calculate wins, losses, and scores
    df["team_score"] = df.apply(
        lambda x: x["home_score"] if x["home"] == team_abbr else x["away_score"], axis=1
    )
    df["opponent_score"] = df.apply(
        lambda x: x["away_score"] if x["home"] == team_abbr else x["home_score"], axis=1
    )
    df["win"] = df["team_score"] > df["opponent_score"]
    df["loss"] = df["team_score"] < df["opponent_score"]

    # Calculate features
    wins = df["win"].sum()
    losses = df["loss"].sum()
    winning_percentage = wins / (wins + losses) if (wins + losses) > 0 else 0
    ppg = df["team_score"].mean()
    opp_ppg = df["opponent_score"].mean()
    ppg_diff = ppg - opp_ppg

    # Calculate WL_Streak
    streak = 0
    for s in df["win"]:
        if s:
            streak = streak + 1 if streak > 0 else 1
        else:
            streak = streak - 1 if streak < 0 else -1
    wl_streak = streak

    return {
        "Wins": wins,
        "Losses": losses,
        "Winning_Percentage": winning_percentage,
        "PPG": ppg,
        "OPP_PPG": opp_ppg,
        "PPG_Diff": ppg_diff,
        "WL_Streak": wl_streak,
    }


def calculate_context_specific_features(df, team_abbr, context):
    # Filter the DataFrame based on the context (home/away games)
    if context == "home":
        df_filtered = df[df["home"] == team_abbr].copy()
    elif context == "away":
        df_filtered = df[df["away"] == team_abbr].copy()
    else:
        raise ValueError("Context must be 'home' or 'away'")

    # Define team_score and opponent_score based on the context after filtering
    if context == "home":
        df_filtered["team_score"] = df_filtered["home_score"]
        df_filtered["opponent_score"] = df_filtered["away_score"]
    else:  # context == 'away'
        df_filtered["team_score"] = df_filtered["away_score"]
        df_filtered["opponent_score"] = df_filtered["home_score"]

    # Proceed with calculations
    df_filtered["win"] = df_filtered["team_score"] > df_filtered["opponent_score"]
    df_filtered["loss"] = df_filtered["team_score"] < df_filtered["opponent_score"]
    wins = df_filtered["win"].sum()
    losses = df_filtered["loss"].sum()
    winning_percentage = wins / (wins + losses) if (wins + losses) > 0 else 0
    ppg = df_filtered["team_score"].mean()
    opp_ppg = df_filtered["opponent_score"].mean()
    ppg_diff = ppg - opp_ppg

    # Calculate streak with context-specific games
    streak = 0
    for s in df_filtered["win"]:
        if s:
            streak = streak + 1 if streak > 0 else 1
        else:
            streak = streak - 1 if streak < 0 else -1
    wl_streak = streak

    return {
        "Winning_Percentage": winning_percentage,
        "PPG": ppg,
        "OPP_PPG": opp_ppg,
        "PPG_Diff": ppg_diff,
        "WL_Streak": wl_streak,
    }


def calculate_time_decayed_features(df, team_abbr, game_date, half_life=10):
    # Convert game_date to datetime and calculate days before the game
    df["game_date"] = pd.to_datetime(df["game_date"])
    target_date = pd.to_datetime(game_date)
    df["days_before_game"] = (target_date - df["game_date"]).dt.days

    # Calculate decay rate from half-life
    lambda_decay = np.log(2) / half_life

    # Calculate decay weights using the half-life
    df["decay_weight"] = np.exp(-lambda_decay * df["days_before_game"])

    # Assign team and opponent scores based on home/away status
    df["team_score"] = df.apply(
        lambda x: x["home_score"] if x["home"] == team_abbr else x["away_score"], axis=1
    )
    df["opponent_score"] = df.apply(
        lambda x: x["away_score"] if x["home"] == team_abbr else x["home_score"], axis=1
    )

    # Calculate win/loss and apply weights
    df["win"] = df["team_score"] > df["opponent_score"]
    weighted_wins = (df["win"] * df["decay_weight"]).sum()
    total_weight = df["decay_weight"].sum()

    # Calculate time-decayed metrics
    time_decayed_winning_percentage = (
        weighted_wins / total_weight if total_weight > 0 else 0
    )
    time_decayed_ppg = (
        (df["team_score"] * df["decay_weight"]).sum() / total_weight
        if total_weight > 0
        else 0
    )
    time_decayed_opp_ppg = (
        (df["opponent_score"] * df["decay_weight"]).sum() / total_weight
        if total_weight > 0
        else 0
    )
    time_decayed_ppg_diff = time_decayed_ppg - time_decayed_opp_ppg

    return {
        "Time_Decayed_Winning_Percentage": time_decayed_winning_percentage,
        "Time_Decayed_PPG": time_decayed_ppg,
        "Time_Decayed_OPP_PPG": time_decayed_opp_ppg,
        "Time_Decayed_PPG_Diff": time_decayed_ppg_diff,
    }


def calculate_rest_and_season_day(df, game_date):
    # Ensure 'game_date' is in datetime format
    df["game_date"] = pd.to_datetime(df["game_date"])
    target_date = pd.to_datetime(game_date)

    # Sort the dataframe by game_date to ensure chronological order
    df_sorted = df.sort_values(by="game_date")

    # Find the team's first game of the season and last game before the game_date
    team_season_start = df_sorted["game_date"].min()
    previous_games = df_sorted[df_sorted["game_date"] < target_date]

    if previous_games.empty:
        # If no previous games, use the season start as the last game (implies first game of the season)
        last_game_date = team_season_start
        rest_days = (
            (target_date - last_game_date).days
            if target_date != team_season_start
            else 0
        )
    else:
        # Calculate rest days using the last game date
        last_game_date = previous_games["game_date"].max()
        rest_days = (target_date - last_game_date).days

    # Calculate the day of the season
    day_of_season = (target_date - team_season_start).days

    # Calculate rest/play count for the last 5, 10, and 30 days
    rest_play_counts = []
    for days in [5, 10, 30]:
        start_date = max(target_date - pd.Timedelta(days=days), team_season_start)
        date_range = pd.date_range(
            start=start_date, end=target_date - pd.Timedelta(days=1)
        )
        rest_play_count = 0
        for day in date_range:
            if day in previous_games["game_date"].values:
                rest_play_count += 1  # Add 1 for game days
            else:
                rest_play_count -= 1  # Subtract 1 for rest days
        rest_play_counts.append(rest_play_count)

    # Average the rest/play counts
    avg_rest_play_count = sum(rest_play_counts) / len(rest_play_counts)

    return rest_days, day_of_season, avg_rest_play_count


if __name__ == "__main__":
    game_id = "0022300122"
    prior_states = get_prior_states(game_id)
    for key, value in prior_states.items():
        print(key, value)
