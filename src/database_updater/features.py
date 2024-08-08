"""
features.py

This module provides functionality to generate feature sets for NBA games from the final game states from prior games.
It consists of functions to:
- Create feature sets for multiple games based on prior states.
- Save feature sets to the database.
- Load feature sets from the database.

Core Functions:
- create_feature_sets(prior_states, db_path=DB_PATH): Generate a set of features for each game in the list.
- save_feature_sets(feature_sets, db_path): Save feature sets to the database for multiple games.
- load_feature_sets(game_ids, db_path=DB_PATH): Load feature sets from the database for a list of game_ids.
- main(): Main function to handle command-line arguments and orchestrate the feature generation process.

Helper Functions for Feature Creation:
- _create_basic_features(home_df, away_df, home_team_abbr, away_team_abbr): Creates basic game features like winning percentage, points per game (PPG), opponents' PPG, and net PPG for a matchup.
- _create_contextual_features(home_df, away_df, home_team, away_team): Creates contextual (home,away) features.
- _create_time_decay_features(home_df, away_df, home_team, away_team, game_date, half_life=10): Creates time-decayed features for a matchup between two teams.
- _create_rest_and_season_features(home_df, away_df, game_date): Creates features related to rest days and days into season for both home and away teams.

Usage:
- Typically run as part of a larger data processing pipeline.
- Script can be run directly from the command line (project root) to generate and save feature sets for specific games.
    python -m src.features --save --game_ids=0042300401,0022300649 --log_level=DEBUG
- Successful execution will log the number of games processed and the time taken to generate the feature sets.
"""

import argparse
import datetime
import json
import logging
import sqlite3

import numpy as np
import pandas as pd

from src.config import config
from src.database_updater.prior_states import (
    determine_prior_states_needed,
    load_prior_states,
)
from src.logging_config import setup_logging
from src.utils import log_execution_time, lookup_basic_game_info

# Configuration
DB_PATH = config["database"]["path"]


@log_execution_time(average_over="prior_states_dict")
def create_feature_sets(prior_states_dict, db_path=DB_PATH):
    """
    Generate a set of features for each game in the list. Feature categories include basic, contextual, time decay, rest days, and day of season.

    Parameters:
    prior_states_dict (dict): A dictionary where each key is a game_id and each value is another dictionary containing
                              'home_prior_states', 'away_prior_states', and 'missing_prior_states'.
    db_path (str): The path to the SQLite database file. Defaults to DB_PATH from config.

    Returns:
    dict: A dictionary where each key is a game_id and each value is a dictionary containing the generated features for that game.
    """

    logging.info(f"Creating feature sets for {len(prior_states_dict)} games...")
    game_ids = list(prior_states_dict.keys())
    game_info = lookup_basic_game_info(game_ids, db_path)

    # Initialize an empty dictionary to store the features for each game
    features_dict = {}

    # Iterate over the list of game_ids
    for game_id, game_info in game_info.items():
        home_team = game_info["home"]
        away_team = game_info["away"]
        game_date = game_info["date_time_est"][:10]

        # Get the prior states for the home and away teams
        home_prior_states = prior_states_dict[game_id]["home_prior_states"]
        away_prior_states = prior_states_dict[game_id]["away_prior_states"]

        # Convert the prior states of home and away teams into DataFrames
        home_prior_states_df = pd.DataFrame(home_prior_states)
        away_prior_states_df = pd.DataFrame(away_prior_states)

        # If either DataFrame is empty, continue to the next game
        if home_prior_states_df.empty or away_prior_states_df.empty:
            features_dict[game_id] = {}
            continue

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

        # Convert the DataFrame to a dictionary and store it in the features_dict
        features_dict[game_id] = features_df.to_dict(orient="records")[0]

    # Calculate the number of successful and unsuccessful feature set creations
    total_games = len(prior_states_dict)
    successful_games = sum(1 for features in features_dict.values() if features)
    no_feature_games_count = total_games - successful_games

    # Log the results
    logging.info(f"Feature sets created successfully for {successful_games} games.")
    logging.info(
        f"No feature sets were created for {no_feature_games_count} games due to insufficient prior states."
    )

    if successful_games > 0:
        example_game_id, example_features = next(
            (game_id, features)
            for game_id, features in features_dict.items()
            if features
        )
        logging.debug(
            f"Example feature set - Game Id {example_game_id}: {example_features}"
        )

    return features_dict


@log_execution_time(average_over="feature_sets")
def save_feature_sets(feature_sets, db_path=DB_PATH):
    """
    Save feature sets to the database for multiple games, including game_id, current datetime as save datetime, and feature set.

    Args:
        feature_sets (dict): The feature sets to save. Keys are game_ids and values are the feature set dict.
        db_path (str): The path to the SQLite database file. Defaults to DB_PATH from config.

    Returns:
        None. The function updates the database directly with the provided information.
    """
    logging.info(f"Saving feature sets for {len(feature_sets)} games...")

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()

        # Prepare the data for the parameterized query
        current_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        data = [
            (
                game_id,
                current_datetime,  # Use the current datetime for all entries
                json.dumps(feature_set),  # Convert feature set to JSON
            )
            for game_id, feature_set in feature_sets.items()
        ]

        # Insert new records or replace the existing ones
        cursor.executemany(
            "INSERT OR REPLACE INTO Features (game_id, save_datetime, feature_set) VALUES (?, ?, ?)",
            data,
        )

        # Commit the changes to the database
        conn.commit()

    # Count the number of games with empty feature sets
    empty_feature_sets_count = sum(
        1 for feature_set in feature_sets.values() if not feature_set
    )
    non_empty_feature_sets_count = len(feature_sets) - empty_feature_sets_count

    logging.info(
        f"Feature sets saved successfully for {non_empty_feature_sets_count} of {len(feature_sets)} games. {empty_feature_sets_count} were empty."
    )
    if data:
        logging.debug(f"Example record: {data[0]}")


@log_execution_time(average_over="game_ids")
def load_feature_sets(game_ids, db_path=DB_PATH):
    """
    Load feature sets from the database for a list of game_ids.

    Args:
        game_ids (list): A list of game_ids to load feature sets for.
        db_path (str): The path to the SQLite database file. Defaults to DB_PATH from config.

    Returns:
        dict: A dictionary where each key is a game_id and each value is the corresponding feature set.
    """
    logging.info(f"Loading feature sets for {len(game_ids)} games...")
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()

        # Query the database for the feature sets for the specified game_ids
        cursor.execute(
            """
            SELECT game_id, feature_set
            FROM Features
            WHERE game_id IN ({})
        """.format(
                ",".join("?" * len(game_ids))
            ),
            game_ids,
        )

        # Fetch the results and construct the dictionary of feature sets
        feature_sets = {
            game_id: json.loads(feature_set) for game_id, feature_set in cursor
        }

    non_empty_feature_sets_count = len([fs for fs in feature_sets.values() if fs])
    empty_feature_sets_count = len(feature_sets) - non_empty_feature_sets_count

    logging.info(
        f"Feature sets loaded successfully for {non_empty_feature_sets_count} of {len(game_ids)} games. {empty_feature_sets_count} were empty."
    )
    if feature_sets:
        example_game_id, example_features = next(iter(feature_sets.items()))
        logging.debug(
            f"Example feature set - Game Id {example_game_id}: {example_features}"
        )

    return feature_sets


def _create_basic_features(home_df, away_df, home_team_abbr, away_team_abbr):
    """
    Creates basic game features like winning percentage, points per game (PPG), opponents' PPG, and net PPG for a matchup.

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
    Creates contextual features for a matchup between two teams based on home and away status.
    Contextual features include winning percentage, points per game (PPG), opponents' PPG, and net PPG.

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
    Creates time-decayed (more recent games weighted higher) features for a matchup between two teams based.
    Time-decayed features include winning percentage, points per game (PPG), opponents' PPG, and net PPG.

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
    Creates features related to rest days and days into season for both home and away teams.

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


def main():
    """
    Main function to handle command-line arguments and orchestrate the feature generation process.
    """
    parser = argparse.ArgumentParser(
        description="Generate feature sets for NBA games from the final game states from prior games."
    )
    parser.add_argument(
        "--game_ids", type=str, help="Comma-separated list of game IDs to process"
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        help="The logging level. Default is INFO. DEBUG provides more details.",
    )
    parser.add_argument(
        "--save", action="store_true", help="Save feature sets to the database."
    )

    args = parser.parse_args()
    log_level = args.log_level.upper()
    setup_logging(log_level=log_level)

    game_ids = args.game_ids.split(",") if args.game_ids else []

    prior_states_needed = determine_prior_states_needed(game_ids)
    prior_states_dict = load_prior_states(prior_states_needed)

    feature_sets = create_feature_sets(prior_states_dict)
    if args.save:
        save_feature_sets(feature_sets)


if __name__ == "__main__":
    main()
