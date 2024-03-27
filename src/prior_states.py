import json
import logging
import sqlite3

import numpy as np
import pandas as pd

from .game_states import get_current_games_info
from .utils import lookup_basic_game_info, validate_game_ids

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s",
)


def get_prior_states(game_id, db_path):
    """
    Get the prior states for a given game.

    Parameters:
    game_id (str): The ID of the game.
    db_path (str): The path to the database.

    Returns:
    dict: A dictionary containing the prior states and the feature set. If an error occurs, an empty dictionary is returned.
    """
    try:
        # Validate the game_id
        validate_game_ids(game_id)

        # Determine necessary prior states for home and away
        home_game_ids_statuses, away_game_ids_statuses = determine_prior_states_needed(
            game_id, db_path
        )

        # Filter the lists to include only game IDs where the status is "Completed"
        home_game_ids = [
            game_id
            for game_id, status in home_game_ids_statuses
            if status == "Completed"
        ]
        away_game_ids = [
            game_id
            for game_id, status in away_game_ids_statuses
            if status == "Completed"
        ]

        # Find missing prior states
        missing_home_game_ids, missing_away_game_ids = find_missing_prior_states(
            home_game_ids, away_game_ids, db_path
        )
        # Attempt to create missing prior states
        for game_id in missing_home_game_ids:
            get_current_games_info(
                game_id, db_path, save_to_db=True
            )  # Fetch and save current game info for home team
        for game_id in missing_away_game_ids:
            get_current_games_info(
                game_id, db_path, save_to_db=True
            )  # Fetch and save current game info for away team

        # Load the prior states from database
        home_prior_states, away_prior_states = load_prior_states(
            home_game_ids, away_game_ids, db_path
        )

        # Warn for missing prior states
        count_missing_completed, count_missing_not_completed = (
            _warn_for_missing_prior_states(
                game_id,
                home_game_ids_statuses,
                home_prior_states,
                away_game_ids_statuses,
                away_prior_states,
            )
        )

        # Determine if the prior states are finalized
        are_prior_states_finalized = False
        if count_missing_completed + count_missing_not_completed == 0:
            are_prior_states_finalized = True

        # Generate the feature set
        feature_set = {}
        # Warn if the feature set is likely to be suboptimal
        if count_missing_completed + count_missing_not_completed != 0:
            logging.warning(
                f"""Game Id {game_id} - Feature set is missing {count_missing_completed} completed games and {count_missing_not_completed} non-completed games. Likely to be suboptimal."""
            )
        # Create the feature set and save it to the database
        feature_set = create_feature_set(
            game_id, db_path, home_prior_states, away_prior_states
        )

        # Save the feature set and prior states flag to the database
        save_prior_states_info(
            game_id, db_path, are_prior_states_finalized, feature_set
        )

        # Create prior states dict
        prior_states = {
            "home_prior_states": home_prior_states,
            "away_prior_states": away_prior_states,
            "feature_set": feature_set,
            "are_prior_states_finalized": are_prior_states_finalized,
        }

        return prior_states

    except Exception as e:
        logging.error(f"Game Id {game_id} - Error getting prior states. {e}")
        return {}


def determine_prior_states_needed(game_id, db_path):
    """
    Determines game IDs and statuses for previous games played by the home and away teams,
    restricting to Regular Season and Post Season games from the same season.

    Parameters:
    db_path (str): The path to the SQLite database file. This database should contain a table
                   named 'Games' with columns for 'game_id', 'date_time_est', 'home_team',
                   'away_team', 'status', 'season', and 'season_type'.
    game_id (str): The ID of the current game. This ID should correspond to a game in the 'Games'
                   table of the database.

    Returns:
    tuple: A tuple containing two lists of tuples. Each tuple contains a game ID and its status.
           The first list contains the IDs and statuses of previous games played by the home team,
           and the second list contains the IDs and statuses of previous games played by the away team.
           Both lists are restricted to games from the same season (Regular Season and Post Season).
           The lists are ordered by date and time.
    """
    # Validate the game_id
    validate_game_ids(game_id)
    # Initialize empty lists to store the game IDs and statuses
    home_game_ids_statuses = []
    away_game_ids_statuses = []

    # Assume the existence of a function to lookup basic game info
    game_info = lookup_basic_game_info(game_id, db_path)[0]
    game_datetime = game_info["date_time_est"]
    home = game_info["home_team"]
    away = game_info["away_team"]
    season = game_info["season"]

    # Connect to the SQLite database
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()

        # Construct the base query with conditions that are always true
        base_query = """
            SELECT game_id, status FROM Games
            WHERE date_time_est < ? AND (home_team = ? OR away_team = ?) 
            AND season = ? AND (season_type = 'Regular Season' OR season_type = 'Post Season')
        """

        # Query for home team's prior games based on the constructed query
        cursor.execute(base_query, (game_datetime, home, home, season))
        home_game_ids_statuses = cursor.fetchall()

        # Repeat the process for the away team
        cursor.execute(base_query, (game_datetime, away, away, season))
        away_game_ids_statuses = cursor.fetchall()

    # Return the lists of game IDs and statuses
    return home_game_ids_statuses, away_game_ids_statuses


def find_missing_prior_states(home_game_ids, away_game_ids, db_path):
    """
    Optimized version to find which game IDs from the provided lists for home and away teams
    do not have a final state in the GameStates table by reducing the number of queries.

    Parameters:
    home_game_ids (list): A list of game IDs for the home team's prior games.
    away_game_ids (list): A list of game IDs for the away team's prior games.
    db_path (str): The path to the SQLite database file.

    Returns:
    tuple: A tuple containing two lists of game IDs. The first list contains the IDs of the home
           team's games that are missing a final state, and the second list contains the IDs of
           the away team's games that are missing a final state.
    """
    missing_home_game_ids = []
    missing_away_game_ids = []

    # Connect to the SQLite database
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()

        # Query for final states for home team games
        placeholders = ", ".join(["?"] * len(home_game_ids))
        cursor.execute(
            f"""
            SELECT game_id FROM GameStates 
            WHERE game_id IN ({placeholders}) AND is_final_state = 1
        """,
            home_game_ids,
        )
        final_states_home = {row[0] for row in cursor.fetchall()}
        missing_home_game_ids = [
            game_id for game_id in home_game_ids if game_id not in final_states_home
        ]

        # Query for final states for away team games
        placeholders = ", ".join(["?"] * len(away_game_ids))
        cursor.execute(
            f"""
            SELECT game_id FROM GameStates 
            WHERE game_id IN ({placeholders}) AND is_final_state = 1
        """,
            away_game_ids,
        )
        final_states_away = {row[0] for row in cursor.fetchall()}
        missing_away_game_ids = [
            game_id for game_id in away_game_ids if game_id not in final_states_away
        ]

    return missing_home_game_ids, missing_away_game_ids


def load_prior_states(home_game_ids, away_game_ids, db_path):
    """
    Loads and orders by date (oldest first) the prior states for lists of home and away game IDs
    from the GameStates table in the database, retrieving all columns for each state and
    storing each state as a dictionary within a list.

    Parameters:
    home_game_ids (list): A list of game IDs for the home team's prior games.
    away_game_ids (list): A list of game IDs for the away team's prior games.
    db_path (str): The path to the SQLite database file.

    Returns:
    tuple: A tuple containing two lists. The first list contains dictionaries of state information
           for each home game_id, ordered by date_time_est, and the second list contains dictionaries
           of state information for each away game_id, also ordered by date_time_est.
    """
    home_prior_states = []
    away_prior_states = []

    # Connect to the SQLite database
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row  # Use sqlite3.Row for dictionary-like row access

        # Load prior states for home games
        if home_game_ids:
            placeholders = ", ".join(["?"] * len(home_game_ids))
            cursor = conn.cursor()
            cursor.execute(
                f"""
                SELECT * FROM GameStates
                WHERE game_id IN ({placeholders}) AND is_final_state = 1
                ORDER BY game_date ASC
            """,
                home_game_ids,
            )
            home_prior_states = [dict(row) for row in cursor.fetchall()]

        # Load prior states for away games
        if away_game_ids:
            placeholders = ", ".join(["?"] * len(away_game_ids))
            cursor = conn.cursor()
            cursor.execute(
                f"""
                SELECT * FROM GameStates
                WHERE game_id IN ({placeholders}) AND is_final_state = 1
                ORDER BY game_date ASC
            """,
                away_game_ids,
            )
            away_prior_states = [dict(row) for row in cursor.fetchall()]

    return home_prior_states, away_prior_states


def create_feature_set(game_id, db_path, home_prior_states, away_prior_states):
    """
    Generate a set of features for a given game.

    Parameters:
    game_id (str): The ID of the game.
    db_path (str): The path to the SQLite database file.
    home_prior_states (list): List of prior states for the home team.
    away_prior_states (list): List of prior states for the away team.

    Returns:
    dict: A dictionary containing the generated features.
    """
    # Validate the game_id
    validate_game_ids(game_id)
    game_info = lookup_basic_game_info(game_id, db_path)[0]
    home_team = game_info["home_team"]
    away_team = game_info["away_team"]
    game_date = game_info["date_time_est"][:10]

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


def save_prior_states_info(game_id, db_path, are_prior_states_finalized, feature_set):
    """
    Save prior states information to the database.

    Args:
        game_id (str): The ID of the game.
        db_path (str): The path to the database.
        are_prior_states_finalized (bool): Whether the prior states are finalized.
        feature_set (dict): The feature set to save.

    Returns:
        None. The function updates the database directly with the provided information.
    """

    # Validate the game_id
    validate_game_ids(game_id)

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()

        # Insert a new record or replace the existing one
        cursor.execute(
            "INSERT OR REPLACE INTO PriorStates (game_id, are_prior_states_finalized, feature_set) VALUES (?, ?, ?)",
            (game_id, are_prior_states_finalized, json.dumps(feature_set)),
        )

        # Commit the changes to the database
        conn.commit()


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


def _warn_for_missing_prior_states(
    game_id,
    home_game_ids_statuses,
    home_prior_states,
    away_game_ids_statuses,
    away_prior_states,
):
    """
    Logs a warning if there are any game IDs in home_game_ids_statuses or away_game_ids_statuses that are not in home_prior_states or away_prior_states and have status "Completed".
    Also logs a warning for game IDs that are not in home_prior_states or away_prior_states and have status other than "Completed".

    Parameters:
    game_id (str): The ID of the game.
    home_game_ids_statuses (list): A list of tuples, each containing a game ID and its status.
    home_prior_states (list): A list of dictionaries, each containing a game ID and its prior state.
    away_game_ids_statuses (list): A list of tuples, each containing a game ID and its status.
    away_prior_states (list): A list of dictionaries, each containing a game ID and its prior state.

    Returns:
    None
    """

    # Convert home_prior_states and away_prior_states to sets of game_ids
    home_prior_states_ids = set(game["game_id"] for game in home_prior_states)
    away_prior_states_ids = set(game["game_id"] for game in away_prior_states)

    # Filter the lists to include only game IDs where the status is "Completed"
    home_game_ids_completed = [
        game_id for game_id, status in home_game_ids_statuses if status == "Completed"
    ]
    away_game_ids_completed = [
        game_id for game_id, status in away_game_ids_statuses if status == "Completed"
    ]

    # Filter the lists to include only game IDs where the status is not "Completed"
    home_game_ids_not_completed = [
        game_id for game_id, status in home_game_ids_statuses if status != "Completed"
    ]
    away_game_ids_not_completed = [
        game_id for game_id, status in away_game_ids_statuses if status != "Completed"
    ]

    # Find game_ids that are in home_game_ids but not in home_prior_states_ids
    missing_home_game_ids_completed = [
        game_id
        for game_id in home_game_ids_completed
        if game_id not in home_prior_states_ids
    ]
    missing_away_game_ids_completed = [
        game_id
        for game_id in away_game_ids_completed
        if game_id not in away_prior_states_ids
    ]

    # Find game_ids that are in home_game_ids_not_completed but not in home_prior_states_ids
    missing_home_game_ids_not_completed = [
        game_id
        for game_id in home_game_ids_not_completed
        if game_id not in home_prior_states_ids
    ]
    missing_away_game_ids_not_completed = [
        game_id
        for game_id in away_game_ids_not_completed
        if game_id not in away_prior_states_ids
    ]

    # Log a warning message if there are any missing game_ids
    if missing_home_game_ids_completed:
        logging.warning(
            f"Game Id {game_id} - {len(missing_home_game_ids_completed)} home team games are missing prior states: {missing_home_game_ids_completed}"
        )
    if missing_away_game_ids_completed:
        logging.warning(
            f"Game Id {game_id} - {len(missing_away_game_ids_completed)} away team games are missing prior states: {missing_away_game_ids_completed}"
        )

    # Log a warning message if there are any missing game_ids that are not "Completed"
    if missing_home_game_ids_not_completed:
        logging.warning(
            f"Game Id {game_id} - {len(missing_home_game_ids_not_completed)} home team games are not completed yet and missing prior states."
        )
    if missing_away_game_ids_not_completed:
        logging.warning(
            f"Game Id {game_id} - {len(missing_away_game_ids_not_completed)} away team games are not completed yet and missing prior states."
        )

    count_missing_completed = len(missing_home_game_ids_completed) + len(
        missing_away_game_ids_completed
    )
    count_missing_not_completed = len(missing_home_game_ids_not_completed) + len(
        missing_away_game_ids_not_completed
    )

    return count_missing_completed, count_missing_not_completed
