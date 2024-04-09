import json
import logging
import sqlite3
import time
import traceback

import numpy as np
import pandas as pd

from .game_states import get_current_games_info
from .utils import lookup_basic_game_info, validate_game_ids

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s",
)


def get_prior_states(game_ids, db_path):
    """
    Get the prior states for given games.

    Parameters:
    game_ids (list): A list of game IDs for which prior states are required.
    db_path (str): The path to the SQLite database file.

    Returns:
    dict: A dictionary where keys are game IDs and values are dictionaries containing the prior states, feature set, and a flag indicating whether prior states are finalized.
    """
    try:
        # Validate the game_ids
        validate_game_ids(game_ids)

        # Determine necessary prior states for home and away teams for completed games
        start = time.time()
        completed_prior_states = _determine_prior_states_needed(
            game_ids, db_path, completed_only=True
        )
        print(
            f"Time taken to determine completed prior states: {time.time() - start:.2f} seconds"
        )

        # Find missing prior states for completed games
        start = time.time()
        missing_completed_prior_states = _find_missing_prior_states(
            completed_prior_states, db_path
        )
        print(
            f"Time taken to find missing completed prior states: {time.time() - start:.2f} seconds"
        )

        # Identify games that need to be updated
        games_to_attempt_update = list(
            set(
                game_id
                for game_ids in missing_completed_prior_states.values()
                for sublist in game_ids
                for game_id in sublist
            )
        )

        # Attempt to create missing prior states
        start = time.time()
        get_current_games_info(games_to_attempt_update, db_path, save_to_db=True)
        print(
            f"Time taken to attempt update of missing prior states: {time.time() - start:.2f} seconds"
        )

        # Determine all required prior states for home and away teams
        start = time.time()
        all_required_prior_states = _determine_prior_states_needed(
            game_ids, db_path, completed_only=False
        )
        print(
            f"Time taken to determine all required prior states: {time.time() - start:.2f} seconds"
        )

        # Load the prior states from the database
        start = time.time()
        prior_states_from_database = _load_prior_states(
            all_required_prior_states, db_path
        )
        print(
            f"Time taken to load prior states from database: {time.time() - start:.2f} seconds"
        )

        # Count the missing prior states
        missing_prior_state_counts = _count_missing_prior_states(
            all_required_prior_states, prior_states_from_database
        )

        # Create the feature set for each game
        start = time.time()
        feature_sets = _create_feature_sets(
            game_ids, prior_states_from_database, db_path
        )
        print(f"Time taken to create feature sets: {time.time() - start:.2f} seconds")

        # Save the feature set and prior states flag to the database
        start = time.time()
        _save_prior_states_info(
            game_ids, missing_prior_state_counts, feature_sets, db_path
        )
        print(
            f"Time taken to save prior states info: {time.time() - start:.2f} seconds"
        )

        # Prepare the final dictionary to return
        prior_states = {}
        for game_id in game_ids:
            prior_states[game_id] = {
                "home_prior_states": prior_states_from_database[game_id][0],
                "away_prior_states": prior_states_from_database[game_id][1],
                "feature_set": feature_sets[game_id],
                "are_prior_states_finalized": sum(missing_prior_state_counts[game_id])
                == 0,
            }

        return prior_states

    except Exception as e:
        logging.error(f"Error getting prior states. {traceback.format_exc()}")
        return {}


def _determine_prior_states_needed(game_ids, db_path, completed_only=False):
    """
    Determines game IDs for previous games played by the home and away teams,
    restricting to Regular Season and Post Season games from the same season.

    Parameters:
    db_path (str): The path to the SQLite database file. This database should contain a table
                   named 'Games' with columns for 'game_id', 'date_time_est', 'home_team',
                   'away_team', 'status', 'season', and 'season_type'.
    game_ids (list): A list of IDs of the current games. These IDs should correspond to games in the 'Games'
                     table of the database.
    completed_only (bool): If True, only return prior states for games that are completed. If False,
                           return prior states for all games.

    Returns:
    dict: A dictionary where each key is a game ID from the input list and each value is a tuple containing
          two lists. The first list contains the IDs of previous games played by the home team, and the second
          list contains the IDs of previous games played by the away team. Both lists are restricted to games
          from the same season (Regular Season and Post Season). If completed_only is True, the lists only
          include games with status "Completed". The lists are ordered by date and time.
    """
    # Initialize an empty dictionary to store the results
    necessary_prior_states = {}

    # Connect to the SQLite database
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()

        # Get basic game info for all game_ids
        games_info = lookup_basic_game_info(game_ids, db_path)

        # Iterate over the list of game info
        for game_info in games_info:
            game_id = game_info["game_id"]
            game_datetime = game_info["date_time_est"]
            home = game_info["home_team"]
            away = game_info["away_team"]
            season = game_info["season"]

            # Initialize empty lists to store the game IDs
            home_game_ids = []
            away_game_ids = []

            # Construct the base query with conditions that are always true
            base_query = """
                SELECT game_id FROM Games
                WHERE date_time_est < ? AND (home_team = ? OR away_team = ?) 
                AND season = ? AND (season_type = 'Regular Season' OR season_type = 'Post Season')
            """

            # Add the condition for completed games if completed_only is True
            if completed_only:
                base_query += " AND status = 'Completed'"

            # Query for home team's prior games based on the constructed query
            cursor.execute(base_query, (game_datetime, home, home, season))
            home_game_ids = [row[0] for row in cursor.fetchall()]

            # Repeat the process for the away team
            cursor.execute(base_query, (game_datetime, away, away, season))
            away_game_ids = [row[0] for row in cursor.fetchall()]

            # Store the lists of game IDs in the results dictionary
            necessary_prior_states[game_id] = (home_game_ids, away_game_ids)

    # Return the results dictionary
    return necessary_prior_states


def _find_missing_prior_states(filtered_prior_states, db_path):
    """
    Finds the game IDs for the home and away teams that are missing final states in the database.

    Parameters:
    filtered_prior_states (dict): A dictionary where keys are game IDs and values are tuples of lists.
                                  Each list contains game IDs for the home and away team's prior games.
    db_path (str): The path to the SQLite database file.

    Returns:
    dict: A dictionary where keys are game IDs and values are tuples of lists. Each list contains the IDs
          of the home and away team's games that are missing a final state.
    """
    missing_prior_states = {}

    # Connect to the SQLite database
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()

        # Get all game IDs from the filtered_prior_states dictionary
        all_game_ids = [
            game_id
            for game_ids in filtered_prior_states.values()
            for game_id in game_ids[0] + game_ids[1]
        ]

        # Query for final states for all game IDs
        placeholders = ", ".join(["?"] * len(all_game_ids))
        cursor.execute(
            f"""
            SELECT game_id FROM GameStates 
            WHERE game_id IN ({placeholders}) AND is_final_state = 1
        """,
            all_game_ids,
        )
        final_states = {row[0] for row in cursor.fetchall()}

        # Iterate over the filtered_prior_states dictionary
        for game_id, (home_game_ids, away_game_ids) in filtered_prior_states.items():
            missing_home_game_ids = [
                id for id in home_game_ids if id not in final_states
            ]
            missing_away_game_ids = [
                id for id in away_game_ids if id not in final_states
            ]

            # Add the missing game IDs to the missing_prior_states dictionary
            missing_prior_states[game_id] = (
                missing_home_game_ids,
                missing_away_game_ids,
            )

    return missing_prior_states


def _load_prior_states(game_ids_dict, db_path):
    """
    Loads and orders by date (oldest first) the prior states for lists of home and away game IDs
    from the GameStates table in the database, retrieving all columns for each state and
    storing each state as a dictionary within a list.

    Parameters:
    game_ids_dict (dict): A dictionary where keys are game IDs and values are tuples of lists.
                          Each list contains game IDs for the home and away team's prior games.
    db_path (str): The path to the SQLite database file.

    Returns:
    dict: A dictionary where keys are game IDs and values are lists of lists. Each internal list contains dictionaries
          of final state information for each home and away game_id, ordered by date_time_est.
    """
    prior_states = {game_id: [[], []] for game_id in game_ids_dict.keys()}

    # Get all unique game IDs
    all_game_ids = list(
        set(game_id for ids in game_ids_dict.values() for game_id in ids[0] + ids[1])
    )

    # Connect to the SQLite database
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row  # Use sqlite3.Row for dictionary-like row access
        cursor = conn.cursor()

        # Load prior states for all games
        if all_game_ids:
            placeholders = ", ".join(["?"] * len(all_game_ids))
            cursor.execute(
                f"""
                SELECT * FROM GameStates
                WHERE game_id IN ({placeholders}) AND is_final_state = 1
                ORDER BY game_date ASC
            """,
                all_game_ids,
            )
            all_prior_states = [dict(row) for row in cursor.fetchall()]

            # Create a dictionary mapping game IDs to their states
            states_dict = {state["game_id"]: state for state in all_prior_states}

            # Separate the states into their respective high-level game_ids and lower-level home/away buckets
            for game_id, (home_game_ids, away_game_ids) in game_ids_dict.items():
                prior_states[game_id][0] = [
                    states_dict[id] for id in home_game_ids if id in states_dict
                ]
                prior_states[game_id][1] = [
                    states_dict[id] for id in away_game_ids if id in states_dict
                ]

    return prior_states


def _count_missing_prior_states(
    all_required_prior_states, prior_states_from_database, warn=True
):
    """
    Count the number of low-level game_ids that are in all_required_prior_states but not in prior_states_from_database.

    Parameters:
    all_required_prior_states (dict): A dictionary with the game_ids being keys and the values being tuples of lists of other game_ids.
    prior_states_from_database (dict): A dictionary with the game_ids being keys and the values being tuples of lists of dicts.
                                        Within these low level dicts is a key "game_id".
    warn (bool): If True, log a warning indicating the game_ids that have total counts greater than 0.

    Returns:
    dict: A dictionary where keys are high-level game_ids and values are the counts of missing low-level game_ids.
    """
    missing_counts = {}

    for game_id, (
        required_home_game_ids,
        required_away_game_ids,
    ) in all_required_prior_states.items():
        loaded_home_game_ids = [
            state["game_id"] for state in prior_states_from_database[game_id][0]
        ]
        loaded_away_game_ids = [
            state["game_id"] for state in prior_states_from_database[game_id][1]
        ]

        missing_home_count = len(
            set(required_home_game_ids) - set(loaded_home_game_ids)
        )
        missing_away_count = len(
            set(required_away_game_ids) - set(loaded_away_game_ids)
        )

        missing_counts[game_id] = (missing_home_count, missing_away_count)

        # Log a warning if the total count is greater than 0 and warn is True
        if warn and (missing_home_count + missing_away_count) != 0:
            logging.warning(
                f"Game {game_id} has missing prior states. This may affect the feature set and downstream predictions."
            )

    return missing_counts


def _save_prior_states_info(
    game_ids, missing_prior_state_counts, feature_sets, db_path
):
    """
    Save prior states information to the database for multiple games.

    Args:
        game_ids (list of str): The IDs of the games.
        missing_prior_state_counts (dict): A dictionary with keys being game_ids and values being tuples of ints which are the counts of missing prior states for home and away.
        feature_sets (dict): The feature sets to save. Keys are game_ids and values are the feature set dict.
        db_path (str): The path to the SQLite database file.

    Returns:
        None. The function updates the database directly with the provided information.
    """
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()

        # Prepare the data for the parameterized query
        data = [
            (
                game_id,
                sum(missing_prior_state_counts[game_id]) == 0,
                json.dumps(feature_sets[game_id]),
            )
            for game_id in game_ids
        ]

        # Insert new records or replace the existing ones
        cursor.executemany(
            "INSERT OR REPLACE INTO PriorStates (game_id, are_prior_states_finalized, feature_set) VALUES (?, ?, ?)",
            data,
        )

        # Commit the changes to the database
        conn.commit()


def _create_feature_sets(game_ids, prior_states_from_database, db_path):
    """
    Generate a set of features for each game in the list.

    Parameters:
    game_ids (list of str): The IDs of the games.
    db_path (str): The path to the SQLite database file.
    prior_states_from_database (dict): A dictionary with the keys being game_ids and the values being tuples of home_prior_states, away_prior_states.

    Returns:
    dict: A dictionary where each key is a game_id and each value is a dictionary containing the generated features for that game.
    """
    games_info = lookup_basic_game_info(game_ids, db_path)

    # Initialize an empty dictionary to store the features for each game
    features_dict = {}

    # Iterate over the list of game_ids
    for game_info in games_info:
        game_id = game_info["game_id"]
        home_team = game_info["home_team"]
        away_team = game_info["away_team"]
        game_date = game_info["date_time_est"][:10]

        # Get the prior states for the home and away teams
        home_prior_states, away_prior_states = prior_states_from_database[game_id]

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
