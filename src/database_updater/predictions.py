"""
predictions.py

This module provides functionality to generate predictions for NBA games using various predictive models.
It consists of classes and functions to:
- Load models and make predictions for multiple games.
- Save predictions to a database.
- Dynamically select predictor classes based on configuration.

Classes:
- BasePredictor: Abstract base class defining the predictor interface.
- RandomPredictor: Predictor generating random scores.
- LinearPredictor: Predictor using a linear regression model.
- TreePredictor: Predictor using a decision tree model.
- MLPPredictor: Predictor using a multi-layer perceptron model.

Functions:
- make_pre_game_predictions(games, predictor_name): Generate predictions using the specified predictor based on historical data.
- make_current_predictions(games, predictor_name): Update predictions using the specified predictor based on the pre game predictions and current game state.
- save_predictions(predictions, predictor_name, db_path=DB_PATH): Save predictions to the database.
- calculate_updated_scores(scores, fraction_of_game_remaining, method="weighted", logistic_params=None): Calculate updated home and away scores based on pre-game predictions, current scores, and game progress.
- calculate_game_progress(period, clock): Calculate the fraction of the game completed and the total minutes remaining.
- calculate_home_win_prob(home_score, away_score, minutes_remaining=None, adjustment_type="logarithmic"): Calculate the win probability for the home team using a logistic function based on the score difference and, optionally, the time remaining.
- get_predictor_class(class_name): Dynamically import and return the predictor class.
- main(): Main function to handle command-line arguments and orchestrate the prediction process. Specific to pre game predictions.

Usage:
- Typically run as part of a larger data processing pipeline.
- Script can be run directly from the command line to generate and save predictions.
    python -m src.predictions --save --game_ids=0042300401,0022300649 --log_level=DEBUG --predictor=Linear
- Successful execution will log the generated predictions and save them to the database.
"""

import argparse
import importlib
import json
import logging
import re
import sqlite3
from abc import ABC, abstractmethod

import joblib
import numpy as np
import pandas as pd
import torch

from src.config import config
from src.database_updater.features import load_feature_sets
from src.logging_config import setup_logging
from src.model_training.mlp_model import MLP
from src.utils import log_execution_time

# Configuration
DB_PATH = config["database"]["path"]
PREDICTORS = config["predictors"]


def calculate_updated_scores(
    scores, fraction_of_game_remaining, method="weighted", logistic_params=None
):
    """
    Calculate updated home and away scores based on pre-game predictions, current scores, and game progress.

    This function updates predictions for the final scores of a basketball game by integrating pre-game predictions
    with current game scores. The update accounts for the progression of the game, adjusting the weight of current
    and predicted scores based on the time remaining. The adjustment method can be chosen to reflect different
    levels of certainty as the game progresses.

    Rationale:
    As the game progresses, the actual game state becomes increasingly meaningful for predicting the final outcome,
    while pre-game predictions become less relevant. This is because there is less time for significant changes
    to occur, making the current scores a more reliable indicator of the final results. The function provides
    various methods to adjust the influence of pre-game predictions versus current scores as the game progresses.

    Parameters:
    scores (dict): A dictionary containing the following keys:
                   - 'pregame_pred_home_score' (float): The pre-game predicted score for the home team.
                   - 'pregame_pred_away_score' (float): The pre-game predicted score for the away team.
                   - 'current_home_score' (float): The current actual score for the home team.
                   - 'current_away_score' (float): The current actual score for the away team.
    fraction_of_game_remaining (float): The fraction of the game remaining, ranging from 0 to 1.
    method (str): The method to use for weighting the scores:
                  - 'simple': A simple average of pre-game and extrapolated current scores.
                  - 'weighted': A weighted average based on the fraction of the game completed.
                  - 'logistic': Uses a logistic function to dynamically adjust the weighting.
    logistic_params (tuple, optional): Parameters for the logistic function, with default values (0.5, 10).
                                       The tuple consists of:
                                       - x0: The midpoint of the logistic curve.
                                       - k: The steepness of the logistic curve.

    Returns:
    tuple: The updated scores (home, away) as floats.
    """
    pregame_pred_home_score = scores["pregame_pred_home_score"]
    pregame_pred_away_score = scores["pregame_pred_away_score"]
    current_home_score = scores["current_home_score"]
    current_away_score = scores["current_away_score"]

    # Calculate the fraction of the game completed
    fraction_of_game_completed = 1 - fraction_of_game_remaining

    # Extrapolate current scores to estimate full game scores
    if fraction_of_game_remaining == 0:
        # Avoid division by zero when the game is complete
        extrapolated_home_score = current_home_score
        extrapolated_away_score = current_away_score
    else:
        extrapolated_home_score = current_home_score / fraction_of_game_completed
        extrapolated_away_score = current_away_score / fraction_of_game_completed

    # Cap the extrapolated scores to avoid unrealistic predictions
    # This is to ensure that the extrapolated scores don't exceed reasonable bounds
    MAX_POINTS_PER_TEAM = 150
    MIN_POINTS_PER_TEAM = 70
    extrapolated_home_score = max(
        min(extrapolated_home_score, MAX_POINTS_PER_TEAM), MIN_POINTS_PER_TEAM
    )
    extrapolated_away_score = max(
        min(extrapolated_away_score, MAX_POINTS_PER_TEAM), MIN_POINTS_PER_TEAM
    )

    if method == "simple":
        # Simple average of the pre-game predictions and extrapolated current scores
        updated_home_score = (extrapolated_home_score + pregame_pred_home_score) / 2
        updated_away_score = (extrapolated_away_score + pregame_pred_away_score) / 2
    elif method == "weighted":
        # Weighted average, with more weight on current scores as the game progresses
        # This method reflects the increasing reliability of current scores as more of the game is played
        weight_for_current = fraction_of_game_completed
        weight_for_pred = fraction_of_game_remaining

        updated_home_score = (
            weight_for_current * extrapolated_home_score
            + weight_for_pred * pregame_pred_home_score
        )
        updated_away_score = (
            weight_for_current * extrapolated_away_score
            + weight_for_pred * pregame_pred_away_score
        )
    elif method == "logistic":
        # Logistic function for dynamic weighting based on game progress
        # The logistic function can model a smooth transition from less to more certainty
        x0, k = logistic_params if logistic_params else (0.5, 10)
        weight_for_current = 1 / (1 + np.exp(-k * (fraction_of_game_completed - x0)))
        weight_for_pred = 1 - weight_for_current

        updated_home_score = (
            weight_for_current * extrapolated_home_score
            + weight_for_pred * pregame_pred_home_score
        )
        updated_away_score = (
            weight_for_current * extrapolated_away_score
            + weight_for_pred * pregame_pred_away_score
        )
    else:
        raise ValueError(
            "Invalid method specified. Use 'simple', 'weighted', or 'logistic'."
        )

    return updated_home_score, updated_away_score


def calculate_game_progress(period, clock):
    """
    Calculate the fraction of the game completed and the total minutes remaining.

    This function calculates how much of the game has been completed and how many minutes are left,
    based on the current period and the remaining time in the period.

    Parameters:
    period (int): The current period of the game. Typically 1-4 for regulation and higher for overtime periods.
    clock (str): The remaining time in the current period, formatted as an ISO 8601 duration (e.g., "PT12M34.567S").

    Returns:
    tuple: A tuple containing:
           - fraction_of_game_completed (float): The fraction of the game that has been completed, ranging from 0 to 1.
           - minutes_remaining (float): The total number of minutes remaining in the game.
    """
    try:
        # Parse the clock string to extract minutes and seconds
        minutes, seconds = map(float, re.findall(r"PT(\d+)M(\d+\.\d+)S", clock)[0])
    except IndexError:
        # Default to 0 if parsing fails
        minutes, seconds = 0, 0

    # Calculate remaining time in the current period
    remaining_time_in_current_period = minutes + seconds / 60

    # Determine total expected game time in minutes
    total_expected_game_time = 48 if period <= 4 else 48 + (period - 4) * 5

    # Calculate elapsed time in the game
    if period <= 4:
        # Regulation time calculation
        total_elapsed_time = (period - 1) * 12 + (12 - remaining_time_in_current_period)
    else:
        # Overtime period calculation
        total_elapsed_time = (
            48 + (period - 5) * 5 + (5 - remaining_time_in_current_period)
        )

    # Calculate fraction of the game completed
    fraction_of_game_completed = total_elapsed_time / total_expected_game_time

    # Calculate total minutes remaining in the game
    minutes_remaining = total_expected_game_time - total_elapsed_time

    return fraction_of_game_completed, minutes_remaining


def calculate_home_win_prob(
    home_score, away_score, minutes_remaining=None, adjustment_type="logarithmic"
):
    """
    Calculate the win probability for the home team using a logistic function
    based on the score difference and, optionally, the time remaining.

    This function computes the probability that the home team will win the game based on the
    current or predicted score difference between the home and away teams. The calculation can
    account for the time remaining in the game, reflecting increased certainty as the game progresses.

    Rationale:
    As a game progresses and the remaining time decreases, the likelihood of a comeback diminishes.
    Thus, the same score difference becomes more indicative of the final outcome when less time is left.
    This function adjusts the win probability calculation to reflect this increasing certainty.

    Parameters:
    home_score (float): The predicted or current score of the home team.
    away_score (float): The predicted or current score of the away team.
    minutes_remaining (float, optional): The minutes remaining in the game. If None, assume a pre-game scenario.
    adjustment_type (str): The type of adjustment to use for in-game calculation:
                           - 'linear': A linear adjustment factor that increases certainty as time decreases.
                           - 'logarithmic': A logarithmic adjustment, providing more sensitivity near the end of the game.

    Returns:
    float: The win probability for the home team, ranging from 0 to 1.
    """
    # Base parameters for the logistic function
    base_a = (
        -0.2504
    )  # Intercept parameter, establishing baseline probability without score difference
    base_b = 0.1949  # Coefficient for score difference, defining the slope of the logistic curve

    # Calculate the score difference, a key factor in determining win probability
    score_diff = home_score - away_score

    # Pre-game scenario: use the base logistic parameters without adjustment
    if minutes_remaining is None:
        win_prob = 1 / (1 + np.exp(-(base_a + base_b * score_diff)))
    else:
        # In-game scenario: Adjust the logistic function based on time remaining
        # Linear and logarithmic adjustments increase certainty as time decreases

        if adjustment_type == "linear":
            # Linear adjustment: certainty increases steadily as minutes_remaining decreases
            time_factor = 48 / (minutes_remaining + 1)
        elif adjustment_type == "logarithmic":
            # Logarithmic adjustment: certainty increases more sharply near the end of the game
            time_factor = np.log(48 / (minutes_remaining + 1))
        else:
            raise ValueError(
                "Invalid adjustment type. Choose 'linear' or 'logarithmic'."
            )

        # Adjust the coefficient 'base_b' to reflect increased certainty
        adjusted_b = base_b * (1 + time_factor)

        # Calculate the win probability using the adjusted logistic function
        win_prob = 1 / (1 + np.exp(-(base_a + adjusted_b * score_diff)))

    return win_prob


class BasePredictor(ABC):
    """
    The BasePredictor serves as an abstract base class for different predictive models.
    It defines the core interface and basic structure that all subclasses must implement.

    This class provides a foundation for loading models, generating pre-game predictions,
    and updating those predictions as the game progresses based on real-time data.

    Subclasses must implement specific methods to handle the model loading and prediction
    generation based on their unique algorithms and data requirements.
    """

    def __init__(self, model_paths=None):
        """
        Initialize the BasePredictor with model paths.

        Parameters:
        model_paths (list of str): Paths to the model files. These paths are used to load the
                                   predictive models necessary for generating predictions.
        """
        self.model_paths = model_paths or []
        self.models = []

    @abstractmethod
    def load_models(self):
        """
        Load the predictive models. Must be implemented by subclasses.

        This method is responsible for loading the machine learning or statistical models that will be used
        to make predictions. The specific implementation will vary depending on the type of model used
        (e.g., linear regression, decision trees, neural networks).

        Subclasses should define the specific model loading logic, ensuring that the models are
        correctly initialized and ready for making predictions.
        """
        pass

    @abstractmethod
    def make_pre_game_predictions(self, games):
        """
        Generate pre-game predictions for the given games. Must be implemented by subclasses.

        This method generates predictions for the outcomes of games before they start, based on
        historical data and pre-game analysis. It leverages the loaded models to forecast various
        aspects of the game, such as scores and player performance.

        Parameters:
        games (dict): The game data to make predictions for, typically including historical data,
                      team statistics, and other relevant features.

        Returns:
        dict: A dictionary containing the predicted scores and other relevant predictions for each game.
        """
        pass

    def update_predictions(self, games):
        """
        Update predictions based on the current state of the games.

        This method updates the predictions as the game progresses, using the current scores and game
        dynamics. It adjusts the pre-game predictions based on the real-time data, providing a more
        accurate forecast of the final game outcomes.

        Parameters:
        games (dict): The game data including current game state. Each game's data should include
                      both the pre-game predictions and the current state of the game.

        Returns:
        dict: A dictionary containing the updated predictions for each game, considering the current
              game state.
        """
        updated_predictions = {}
        for game_id, game_data in games.items():
            pre_game_pred = game_data.get("pre_game_predictions", {})
            current_state = game_data.get("current_game_state", {})

            if not current_state:
                updated_predictions[game_id] = pre_game_pred
                continue

            if current_state.get("is_final_state"):
                # Create the new dictionary item in updated_predictions
                updated_predictions[game_id] = {
                    "pred_home_score": current_state["home_score"],
                    "pred_away_score": current_state["away_score"],
                    "pred_home_win_pct": (
                        1.0
                        if current_state["home_score"] > current_state["away_score"]
                        else 0.0
                    ),
                    "pred_players": current_state.get(
                        "players_data", {"home": {}, "away": {}}
                    ),
                }

                # Modify the pred_players within the new dictionary item
                pred_players = updated_predictions[game_id]["pred_players"]
                for team in ["home", "away"]:
                    for player_id, player_stats in pred_players[team].items():
                        if "points" in player_stats:
                            player_stats["pred_points"] = player_stats.pop("points")
                continue

            pre_home = pre_game_pred["pred_home_score"]
            pre_away = pre_game_pred["pred_away_score"]
            curr_home = current_state["home_score"]
            curr_away = current_state["away_score"]
            pre_players = pre_game_pred["pred_players"]
            curr_players = current_state["players_data"]

            # Use calculate_game_progress to get fraction completed and minutes remaining
            fraction_completed, minutes_remaining = calculate_game_progress(
                current_state["period"], current_state["clock"]
            )

            # Create a dictionary for scores
            scores = {
                "pregame_pred_home_score": pre_home,
                "pregame_pred_away_score": pre_away,
                "current_home_score": curr_home,
                "current_away_score": curr_away,
            }

            # Call the refactored calculate_updated_scores function
            updated_home, updated_away = calculate_updated_scores(
                scores=scores, fraction_of_game_remaining=(1 - fraction_completed)
            )

            updated_win_prob = calculate_home_win_prob(
                updated_home, updated_away, minutes_remaining=minutes_remaining
            )

            updated_pred_players = (
                pre_players  # Pass through for now; to be implemented in the future
            )

            updated_predictions[game_id] = {
                "pred_home_score": updated_home,
                "pred_away_score": updated_away,
                "pred_home_win_pct": updated_win_prob,
                "pred_players": updated_pred_players,
            }

        return updated_predictions


class BaselinePredictor(BasePredictor):
    """
    Predictor that uses a simple baseline model to generate predictions for NBA games.

    This class provides a simple baseline model that predicts the final scores of NBA games
    based on the average scores of the home and away teams in the training data.
    """

    def load_models(self):
        """
        Load the baseline model (not applicable for baseline predictor).

        Since the baseline predictor uses a simple average, no model loading is necessary.
        """
        pass

    def make_pre_game_predictions(self, games):
        """
        Generate pre-game predictions using specific features from each game's data.

        Parameters:
        games (dict): A dictionary containing game data, with each game having associated features.

        Returns:
        dict: A dictionary of predictions, including predicted scores and win probabilities for each game.
        """
        predictions = {}

        for game_id, game_data in games.items():
            # Ensure all required features are present
            if (
                "Home_PPG" in game_data
                and "Home_OPP_PPG" in game_data
                and "Away_PPG" in game_data
                and "Away_OPP_PPG" in game_data
            ):

                # Extract the relevant features
                home_ppg = game_data["Home_PPG"]
                home_opp_ppg = game_data["Home_OPP_PPG"]
                away_ppg = game_data["Away_PPG"]
                away_opp_ppg = game_data["Away_OPP_PPG"]

                # Calculate the predicted scores
                pred_home_score = (home_ppg + away_opp_ppg) / 2
                pred_away_score = (away_ppg + home_opp_ppg) / 2

                # Calculate the predicted win probability for the home team
                pred_home_win_pct = calculate_home_win_prob(
                    pred_home_score, pred_away_score
                )

                # Store predictions for the current game
                predictions[game_id] = {
                    "pred_home_score": pred_home_score,
                    "pred_away_score": pred_away_score,
                    "pred_home_win_pct": pred_home_win_pct,
                    "pred_players": game_data.get(
                        "pred_players", {"home": {}, "away": {}}
                    ),
                }

            else:
                # Skip games with missing data and optionally log the issue
                print(f"Skipping game {game_id} due to missing data")

        return predictions

    def update_predictions(self, games):
        """
        Update predictions based on the current state of the games (baseline predictor uses base logic).

        Parameters:
        games (dict): A dictionary containing current game states and pre-game predictions.

        Returns:
        dict: A dictionary of updated predictions with baseline values.
        """
        return super().update_predictions(games)


class LinearPredictor(BasePredictor):
    """
    Predictor that uses a linear regression model to generate predictions for NBA games.

    This class loads a linear regression model to make pre-game predictions and update them
    based on the current game state.
    """

    def load_models(self):
        """
        Load the linear regression model from the specified path.

        This method initializes the linear regression model using a pre-trained model file.
        """
        self.model = joblib.load(self.model_paths[0])

    def make_pre_game_predictions(self, games):
        """
        Generate pre-game predictions using the linear regression model.

        Parameters:
        games (dict): A dictionary containing game data, with each game having associated features.

        Returns:
        dict: A dictionary of predictions, including predicted scores and win probabilities for each game.
        """
        game_ids = list(games.keys())
        features = [games[game_id] for game_id in game_ids]
        features_df = pd.DataFrame(features).fillna(0)

        scores = self.model.predict(features_df.values)
        home_scores, away_scores = scores[:, 0], scores[:, 1]

        # Generate predictions dictionary
        predictions = {}
        for game_id, home_score, away_score in zip(game_ids, home_scores, away_scores):
            home_win_prob = calculate_home_win_prob(home_score, away_score)
            predictions[game_id] = {
                "pred_home_score": home_score,
                "pred_away_score": away_score,
                "pred_home_win_pct": home_win_prob,
                "pred_players": games[game_id].get(
                    "pred_players", {"home": {}, "away": {}}
                ),
            }
        return predictions

    def update_predictions(self, games):
        """
        Update predictions based on the current state of the games.

        Parameters:
        games (dict): A dictionary containing current game states and pre-game predictions.

        Returns:
        dict: A dictionary of updated predictions based on real-time game data.
        """
        return super().update_predictions(games)


class TreePredictor(BasePredictor):
    """
    Predictor that uses a decision tree model to generate predictions for NBA games.

    This class loads a decision tree model to make pre-game predictions and update them
    based on the current game state.
    """

    def load_models(self):
        """
        Load the decision tree model from the specified path.

        This method initializes the decision tree model using a pre-trained model file.
        """
        self.model = joblib.load(self.model_paths[0])

    def make_pre_game_predictions(self, games):
        """
        Generate pre-game predictions using the decision tree model.

        Parameters:
        games (dict): A dictionary containing game data, with each game having associated features.

        Returns:
        dict: A dictionary of predictions, including predicted scores and win probabilities for each game.
        """
        game_ids = list(games.keys())
        features = [games[game_id] for game_id in game_ids]
        features_df = pd.DataFrame(features).fillna(0)

        scores = self.model.predict(features_df.values)
        home_scores, away_scores = scores[:, 0], scores[:, 1]

        # Generate predictions dictionary
        predictions = {}
        for game_id, home_score, away_score in zip(game_ids, home_scores, away_scores):
            home_win_prob = calculate_home_win_prob(home_score, away_score)
            predictions[game_id] = {
                "pred_home_score": home_score,
                "pred_away_score": away_score,
                "pred_home_win_pct": home_win_prob,
                "pred_players": games[game_id].get(
                    "pred_players", {"home": {}, "away": {}}
                ),
            }
        return predictions

    def update_predictions(self, games):
        """
        Update predictions based on the current state of the games.

        Parameters:
        games (dict): A dictionary containing current game states and pre-game predictions.

        Returns:
        dict: A dictionary of updated predictions based on real-time game data.
        """
        return super().update_predictions(games)


class MLPPredictor(BasePredictor):
    """
    Predictor that uses a multi-layer perceptron (MLP) model to generate predictions for NBA games.

    This class loads an MLP model to make pre-game predictions and update them based on the current game state.
    """

    def load_models(self):
        """
        Load the MLP model from the specified path and set up normalization parameters.

        This method initializes the MLP model using a pre-trained model checkpoint file. It also sets up
        normalization parameters required for the model's inputs.
        """
        checkpoint = torch.load(self.model_paths[0])
        self.model = MLP(input_size=checkpoint["input_size"])
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.mean = checkpoint["mean"]
        self.std = checkpoint["std"]

    def make_pre_game_predictions(self, games):
        """
        Generate pre-game predictions using the MLP model.

        Parameters:
        games (dict): A dictionary containing game data, with each game having associated features.

        Returns:
        dict: A dictionary of predictions, including predicted scores and win probabilities for each game.
        """
        game_ids = list(games.keys())
        features = [games[game_id] for game_id in game_ids]
        features_df = pd.DataFrame(features).fillna(0)  # Handle NaN values

        self.model.eval()
        with torch.no_grad():
            features_tensor = torch.tensor(features_df.values, dtype=torch.float32)
            features_normalized = (features_tensor - self.mean) / self.std
            scores = self.model(features_normalized).numpy()
            home_scores, away_scores = scores[:, 0], scores[:, 1]

        # Generate predictions dictionary
        predictions = {}
        for game_id, home_score, away_score in zip(game_ids, home_scores, away_scores):
            home_win_prob = calculate_home_win_prob(home_score, away_score)
            predictions[game_id] = {
                "pred_home_score": home_score,
                "pred_away_score": away_score,
                "pred_home_win_pct": home_win_prob,
                "pred_players": games[game_id].get(
                    "pred_players", {"home": {}, "away": {}}
                ),
            }
        return predictions

    def update_predictions(self, games):
        """
        Update predictions based on the current state of the games.

        Parameters:
        games (dict): A dictionary containing current game states and pre-game predictions.

        Returns:
        dict: A dictionary of updated predictions based on real-time game data.
        """
        return super().update_predictions(games)


class RandomPredictor(BasePredictor):
    """
    Predictor that generates random predictions for NBA games.

    This class provides a simple model that generates random scores for demonstration purposes or as a baseline.
    """

    def load_models(self):
        """
        Load models (not applicable for random predictor).

        Since this predictor generates random values, no model loading is necessary.
        """
        pass

    def make_pre_game_predictions(self, games):
        """
        Generate random pre-game predictions for the given games.

        Parameters:
        games (dict): A dictionary containing game data.

        Returns:
        dict: A dictionary of random predictions, including scores and win probabilities for each game.
        """
        game_ids = list(games.keys())
        home_scores = np.random.randint(80, 131, size=len(game_ids))
        away_scores = np.random.randint(80, 131, size=len(game_ids))

        predictions = {}
        for game_id, home_score, away_score in zip(game_ids, home_scores, away_scores):
            home_win_prob = calculate_home_win_prob(home_score, away_score)
            predictions[game_id] = {
                "pred_home_score": home_score,
                "pred_away_score": away_score,
                "pred_home_win_pct": home_win_prob,
                "pred_players": games[game_id].get(
                    "pred_players", {"home": {}, "away": {}}
                ),
            }
        return predictions

    def update_predictions(self, games):
        """
        Update predictions based on the current state of the games (random predictor uses base logic).

        Parameters:
        games (dict): A dictionary containing current game states and pre-game predictions.

        Returns:
        dict: A dictionary of updated predictions with random values.
        """
        return super().update_predictions(games)


def get_predictor_class(class_name):
    """
    Dynamically import and return the predictor class.

    Parameters:
    class_name (str): The name of the predictor class.

    Returns:
    class: The predictor class.
    """
    module = importlib.import_module(__name__)
    predictor_class = getattr(module, class_name)
    return predictor_class


@log_execution_time(average_over="games")
def make_pre_game_predictions(games, predictor_name):
    """
    Generate pre-game predictions using the specified predictor based on historical data.

    Parameters:
    games (dict): The game data to make predictions for. Feature sets for each game from the database.
    predictor_name (str): The name of the predictor.

    Returns:
    dict: The generated pre-game predictions.
    """
    # Handle the "Best" option by mapping it to the actual best predictor
    if predictor_name == "Best":
        predictor_name = PREDICTORS["Best"]

    total_games = len(games)

    # Filter out games with empty feature sets
    games_to_predict = {
        game_id: features for game_id, features in games.items() if features
    }
    skipped_games_count = total_games - len(games_to_predict)

    if not games_to_predict:
        logging.warning(
            f"No games with sufficient data to predict. Skipping {skipped_games_count} games due to missing or empty feature sets."
        )
        return {}

    logging.info(
        f"Generating predictions for {len(games_to_predict)} games using predictor '{predictor_name}'."
        f" Skipping {skipped_games_count} games due to missing or empty feature sets."
    )

    predictor_cfg = PREDICTORS[predictor_name]
    class_name = predictor_cfg["class"]
    model_paths = predictor_cfg.get("model_paths", [])

    predictor_class = get_predictor_class(class_name)
    predictor = predictor_class(model_paths)
    predictor.load_models()
    pre_game_predictions = predictor.make_pre_game_predictions(games_to_predict)

    logging.info(
        f"Predictions generated successfully for {len(pre_game_predictions)} games using predictor '{predictor_name}'."
    )
    logging.debug(f"Pre Game Predictions: {pre_game_predictions}")

    return pre_game_predictions


@log_execution_time(average_over="games")
def make_current_predictions(games, predictor_name):
    """
    Update predictions using the specified predictor based on the pre-game predictions and current game state.

    This function updates the predictions for a set of games using a chosen predictive model. It integrates
    real-time game data with pre-game predictions to provide a more accurate forecast of final scores and outcomes.

    Parameters:
    games (dict): A dictionary containing data for multiple games. Each game entry should include:
                  - pre_game_predictions: The initial predictions made before the game started.
                  - current_game_state: The real-time data capturing the current state of the game, including scores,
                    time remaining, and game status.
    predictor_name (str): The name of the predictor to use for updating predictions. This should match one of the
                          available predictor configurations.

    Returns:
    dict: A dictionary containing the updated predictions for each game. The predictions include updated scores,
          win probabilities, and player-specific data if applicable.

    Raises:
    KeyError: If the specified predictor_name does not exist in the PREDICTORS configuration.
    """
    # Handle the "Best" option by mapping it to the actual best predictor
    if predictor_name == "Best":
        predictor_name = PREDICTORS["Best"]

    logging.info(
        f"Updating {len(games)} predictions using predictor '{predictor_name}'..."
    )

    # Retrieve the predictor configuration based on the provided name
    predictor_cfg = PREDICTORS[predictor_name]
    class_name = predictor_cfg["class"]

    # Dynamically load the predictor class
    predictor_class = get_predictor_class(class_name)
    predictor = predictor_class()

    # Generate the updated predictions using the current game data
    current_predictions = predictor.update_predictions(games)

    logging.info(
        f"Predictions updated successfully for {len(games)} games using predictor '{predictor_name}'."
    )
    logging.debug(f"Updated Predictions: {current_predictions}")

    return current_predictions


@log_execution_time(average_over="predictions")
def save_predictions(predictions, predictor_name, db_path=DB_PATH):
    """
    Save predictions to the Predictions table.

    Parameters:
    predictions (dict): The predictions to save.
    predictor_name (str): The name of the predictor.
    db_path (str): The path to the SQLite database file. Defaults to DB_PATH from config.

    Returns:
    None
    """
    if not predictions:
        logging.info("No predictions to save.")
        return

    if predictor_name == "Best":
        predictor_name = PREDICTORS["Best"]

    logging.info(
        f"Saving {len(predictions)} predictions for predictor '{predictor_name}'..."
    )
    prediction_datetime = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    if predictor_name == "Random":
        model_id = "Random"
    elif predictor_name == "Baseline":
        model_id = "Baseline"
    else:
        model_id = (
            PREDICTORS[predictor_name]["model_paths"][0].split("/")[-1].split(".")[0]
        )

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()

        data = [
            (
                game_id,
                predictor_name,
                model_id,
                prediction_datetime,
                json.dumps(
                    {
                        k: (
                            float(v)
                            if isinstance(v, (np.float32, np.float64, np.int64))
                            else v
                        )
                        for k, v in predictions[game_id].items()
                    }
                ),
            )
            for game_id in predictions.keys()
        ]

        cursor.executemany(
            """
            INSERT OR REPLACE INTO Predictions (game_id, predictor, model_id, prediction_datetime, prediction_set)
            VALUES (?, ?, ?, ?, ?)
            """,
            data,
        )

        conn.commit()

    logging.info("Predictions saved successfully.")
    if data:
        logging.debug(f"Example record: {data[0]}")


def main():
    """
    Main function to handle command-line arguments and orchestrate the prediction process.
    """
    parser = argparse.ArgumentParser(
        description="Generate predictions for NBA games using various predictive models."
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
    parser.add_argument(
        "--predictor",
        default="Best",
        type=str,
        help="The predictor to use for predictions.",
    )

    args = parser.parse_args()
    log_level = args.log_level.upper()
    setup_logging(log_level=log_level)

    game_ids = args.game_ids.split(",") if args.game_ids else []

    # Load feature sets from the database
    feature_sets = load_feature_sets(game_ids=game_ids)

    # Generate predictions using the specified predictor
    predictions = make_pre_game_predictions(args.predictor, feature_sets)
    if args.save:
        save_predictions(predictions, args.predictor)


if __name__ == "__main__":
    main()
