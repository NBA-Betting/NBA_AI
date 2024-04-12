import logging
import re
import traceback

import numpy as np
import pandas as pd
import torch

from .modeling.mlp_model import MLP

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s",
)

pd.set_option("display.max_columns", None)


def get_predictions(games, predictor, model=None):
    """
    Get predictions for a game using a specified predictor.

    Parameters:
    games (list): A list of dictionaries containing game information.
    predictor (str): The name of the predictor to use.
    model (object): The model object to use for making predictions.

    Returns:
    list: A list of dictionaries containing the predictions for each game.
    """
    try:
        predictor_dict = {
            "Random": random_predictions,
            "LinearModel": pregame_model_predictor,
            "TreeModel": pregame_model_predictor,
            "MLPModel": pregame_model_predictor,
        }
        if predictor in predictor_dict:
            return predictor_dict[predictor](games, model)
        else:
            raise ValueError(f"Predictor '{predictor}' not found.")
    except Exception as e:
        logging.error(f"Failed to get predictions. Error: {e}")
        return []


def pregame_model_predictor(games, model):
    """
    Function to predict game outcomes using a pre-trained model.

    Parameters:
    games (list): A list of dictionaries, each representing a game with its features.
    model (object): A pre-trained model object with a predict method.

    Returns:
    list: A list of dictionaries, each representing a game with its predicted outcomes.
    """

    # Initialize list to hold game information
    games_info = []

    # Collect game information in a structured way for all games
    for game in games:
        # Extract basic game information
        game_info = {
            key: game[key] for key in ["game_id", "home", "away", "game_status"]
        }

        # Initialize game state keys with None
        state_keys = ["period", "clock", "home_score", "away_score", "players_data"]
        game_info.update({key: None for key in state_keys})

        # If there are game states, extract the latest state information
        if game["game_states"]:
            state_info = {key: game["game_states"][-1][key] for key in state_keys}
            game_info.update(state_info)

        # Add feature set information to game info
        game_info.update(game["prior_states"]["feature_set"])
        games_info.append(game_info)

    # Convert collected game information to DataFrame
    games_df = pd.DataFrame(games_info)

    # Extract feature column names
    feature_cols = list(games[0]["prior_states"]["feature_set"].keys()) if games else []
    features_df = games_df[feature_cols]

    try:
        if isinstance(model, dict):
            # Extract the model state dict, mean, and std from the dictionary
            model_state_dict = model["model_state_dict"]
            mean = model["mean"]
            std = model["std"]

            # Create a new instance of your model class and load the state dict
            mlp_model = MLP(input_size=features_df.shape[1])
            mlp_model.load_state_dict(model_state_dict)
            mlp_model.eval()

            # Convert features to a tensor and normalize
            features_tensor = torch.tensor(features_df.values, dtype=torch.float32)
            features_tensor = (features_tensor - mean) / std

            # Predict scores for MLP model
            with torch.no_grad():
                scores = mlp_model(features_tensor).numpy()

        else:
            # Predict scores for linear and tree models
            scores = model.predict(features_df.values)
        games_df[["pregame_pred_home_score", "pregame_pred_away_score"]] = scores
    except Exception as e:
        # Log error and return empty list if prediction fails
        logging.error(f"Error creating pregame model prediction: {e}")
        return []

    try:
        # Update score predictions based on the current game state
        games_df[["pred_home_score", "pred_away_score"]] = games_df.apply(
            lambda row: update_score_predictions(
                row["period"],
                row["clock"],
                row["pregame_pred_home_score"],
                row["pregame_pred_away_score"],
                row["home_score"],
                row["away_score"],
                pred_importance=0.5,
            ),
            axis=1,
            result_type="expand",
        )
    except Exception as e:
        # Log error and return empty list if updating predictions fails
        logging.error(f"Error updating score predictions: {e}")
        return []

    try:
        # Update win probability based on the updated score predictions
        games_df["home_win_prob"] = games_df.apply(
            lambda row: create_win_prob(
                row["period"],
                row["clock"],
                row["pred_home_score"],
                row["pred_away_score"],
            ),
            axis=1,
        )
    except Exception as e:
        # Log error and return empty list if updating win probability fails
        logging.error(f"Error creating win probability: {e}\n{traceback.format_exc()}")
        return []

    # Determine the predicted winner and winning percentage
    games_df[["pred_winner", "pred_win_pct"]] = games_df.apply(
        lambda row: pd.Series(
            [row["home"], row["home_win_prob"]]
            if row["home_win_prob"] >= 0.5
            else [row["away"], 1 - row["home_win_prob"]]
        ),
        axis=1,
    )

    # Format predictions
    games_df["pred_win_pct"] = games_df["pred_win_pct"].apply(
        lambda x: "100%" if x == 1 else (">99%" if x > 0.99 else f"{x:.0%}")
    )
    games_df["pred_home_score"] = games_df["pred_home_score"].round().astype(int)
    games_df["pred_away_score"] = games_df["pred_away_score"].round().astype(int)

    # Convert predictions to a list of dictionaries
    predictions = games_df[
        ["game_id", "pred_home_score", "pred_away_score", "pred_winner", "pred_win_pct"]
    ].to_dict(orient="records")

    return predictions


def update_score_predictions(
    period,
    clock,
    pregame_pred_home_score,
    pregame_pred_away_score,
    current_home_score,
    current_away_score,
    pred_importance,
):
    """
    Updates NBA game score predictions based on the current score and pre-game predictions.

    Parameters:
    period (int): Current period of the game.
    clock (str): Current time left in the period.
    pregame_pred_home_score (float): Pre-game predicted score for the home team.
    pregame_pred_away_score (float): Pre-game predicted score for the away team.
    current_home_score (int): Current score for the home team.
    current_away_score (int): Current score for the away team.
    pred_importance (float): Importance of the pre-game prediction in the updated prediction.

    Returns:
    tuple: Updated predicted scores for the home and away teams.
    """
    # If the game hasn't started, return pregame predictions
    if (
        pd.isnull(period)
        or pd.isnull(clock)
        or pd.isnull(current_home_score)
        or pd.isnull(current_away_score)
    ):
        return pregame_pred_home_score, pregame_pred_away_score

    # Parse clock string to extract minutes and seconds
    minutes, seconds = map(float, re.findall(r"PT(\d+)M(\d+\.\d+)S", clock)[0])

    # Calculate remaining time in the current period
    remaining_time_in_current_period = minutes + seconds / 60

    # Calculate total expected game time based on period
    total_expected_game_time = 48 if period <= 4 else 48 + (period - 4) * 5

    # Calculate elapsed time in the game
    total_elapsed_time = (
        (period - 1) * 12 + 12 - remaining_time_in_current_period
        if period <= 4
        else 48 + (period - 5) * 5 + 5 - remaining_time_in_current_period
    )

    # Calculate fraction of the game remaining
    fraction_of_game_remaining = 1 - total_elapsed_time / total_expected_game_time

    # If the game hasn't started, use pregame predictions
    if fraction_of_game_remaining == 1:
        return pregame_pred_home_score, pregame_pred_away_score

    # If the game has ended, use current scores
    if fraction_of_game_remaining == 0:
        return current_home_score, current_away_score

    # Adjust game progress for non-linear transition based on pred_importance
    adjusted_progress = (
        np.power(fraction_of_game_remaining, 2)
        if pred_importance < 0.5
        else np.sqrt(fraction_of_game_remaining)
    )
    if pred_importance == 0.5:
        adjusted_progress = fraction_of_game_remaining

    # Calculate weights for current and predicted scores
    weight_for_current = 1 - adjusted_progress
    weight_for_pred = adjusted_progress

    # Extrapolate current scores to a full game estimation
    extrapolated_home_score = current_home_score / (1 - fraction_of_game_remaining)
    extrapolated_away_score = current_away_score / (1 - fraction_of_game_remaining)

    # Calculate updated predictions by combining the extrapolated scores and pregame predictions
    updated_home_score = (
        weight_for_current * extrapolated_home_score
        + weight_for_pred * pregame_pred_home_score
    )
    updated_away_score = (
        weight_for_current * extrapolated_away_score
        + weight_for_pred * pregame_pred_away_score
    )

    return updated_home_score, updated_away_score


def create_win_prob(period, clock, updated_home_score, updated_away_score):
    """
    Calculate the win probability for the home team using an updated logistic function
    that dynamically adjusts based on the time remaining in the game.

    Parameters:
    period (int): The current period of the game.
    clock (str): The remaining time in the current period in ISO 8601 format (e.g., "PT12M34.567S").
    updated_home_score (float): The current score of the home team.
    updated_away_score (float): The current score of the away team.

    Returns:
    float: The win probability for the home team.
    """
    # Base parameters for the logistic function
    a = -0.2504
    b = 0.1949

    # Calculate the predicted score difference for the home team
    score_diff = updated_home_score - updated_away_score

    def determine_minutes_remaining(period, clock):
        """
        Calculate the total minutes remaining in the game.

        Parameters:
        period (int): The current period of the game.
        clock (str): The remaining time in the current period in ISO 8601 duration format.

        Returns:
        float: The total minutes remaining in the game.
        """
        try:
            minutes, seconds = map(float, re.findall(r"PT(\d+)M(\d+\.\d+)S", clock)[0])
        except IndexError:
            # Default to 0 if parsing fails
            minutes, seconds = 0, 0
        remaining_in_period = minutes + seconds / 60

        # Adjust for periods beyond the 4th
        if period <= 4:
            minutes_remaining = (4 - period) * 12 + remaining_in_period
        else:
            minutes_remaining = remaining_in_period

        return minutes_remaining

    # If period or clock is missing, assume pregame scenario
    if pd.isnull(period) or pd.isnull(clock):
        minutes_remaining = 48
    else:
        minutes_remaining = determine_minutes_remaining(period, clock)

    # Game over scenario
    if minutes_remaining == 0:
        return 1 if score_diff > 0 else 0

    # Dynamic adjustment based on the time remaining
    if minutes_remaining > 0:  # Avoid division by zero
        time_factor = np.log(48 / (minutes_remaining + 1))  # Logarithmic adjustment
        adjusted_b = b * (1 + time_factor)  # Adjust b dynamically
    else:
        adjusted_b = b

    # Calculate the win probability using the dynamically adjusted logistic function
    win_prob = 1 / (1 + np.exp(-(a + adjusted_b * score_diff)))

    return win_prob


def random_predictions(games, model=None):
    """
    Generate random predictions for a list of games.

    This function generates random scores for the home and away teams, determines the winning team and winning
    percentage, and generates random points for each player in the home and away teams. The predictions are
    returned as a list of dictionaries.

    Args:
        games (list): A list of dictionaries, each containing game information.
        model (object, optional): The model object to use for making predictions. Defaults to None.

    Returns:
        list: A list of dictionaries, each containing the generated predictions for a game.
    """
    predictions_list = []
    for game in games:
        # Generate random home and away scores from a normal distribution
        # The scores are constrained to be between 80 and 140
        home_score = int(max(80, min(140, abs(np.random.normal(110, 15)))))
        away_score = int(max(80, min(140, abs(np.random.normal(110, 15)))))

        # Determine the winning team and winning percentage
        # If the scores are equal, the away team is considered the winning team
        if home_score > away_score:
            winning_team = game["home"]
            winning_team_pct = (home_score + (home_score - away_score)) / (
                home_score + away_score
            )
        else:
            winning_team = game["away"]
            winning_team_pct = (away_score + (away_score - home_score)) / (
                home_score + away_score
            )

        # Format the winning percentage as a string with a percent sign
        winning_team_pct = f"{winning_team_pct:.0%}"

        # Initialize the predictions dictionary
        predictions = {
            "game_id": game["game_id"],
            "pred_home_score": home_score,
            "pred_away_score": away_score,
            "pred_winner": winning_team,
            "pred_win_pct": winning_team_pct,
            "pred_players": {"home": {}, "away": {}},
        }

        # Generate random player predictions if game["game_states"] is not empty
        if game["game_states"]:
            # Get the current players from the last game state
            current_players = game["game_states"][-1]["players_data"]

            # Generate random points for each home player
            # The points are constrained to be between 0 and 40
            for player in current_players["home"]:
                predictions["pred_players"]["home"][player] = {
                    "name": current_players["home"][player]["name"],
                    "pred_points": int(max(0, min(40, abs(np.random.normal(20, 5))))),
                }

            # Generate random points for each away player
            # The points are constrained to be between 0 and 40
            for player in current_players["away"]:
                predictions["pred_players"]["away"][player] = {
                    "name": current_players["away"][player]["name"],
                    "pred_points": int(max(0, min(40, abs(np.random.normal(20, 5))))),
                }

        predictions_list.append(predictions)

    return predictions_list
