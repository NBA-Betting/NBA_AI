"""
predictions.py

This module provides functionality to generate predictions for NBA games using various predictive models.
It consists of classes and functions to:
- Load models and make predictions for multiple games.
- Save predictions to a database.
- Dynamically select predictor classes based on configuration.

Core Classes:
- BasePredictor: Abstract base class defining the predictor interface.
- RandomPredictor: Predictor generating random scores.
- LinearPredictor: Predictor using a linear regression model.
- TreePredictor: Predictor using a decision tree model.
- MLPPredictor: Predictor using a multi-layer perceptron model.

Core Functions:
- make_predictions(predictor_name, games): Generate predictions using the specified predictor.
- save_predictions(predictions, predictor, db_path=DB_PATH): Save predictions to the database.

Helper Functions:
- calculate_pred_home_win_pct(home_scores, away_scores): Calculate home team win probabilities.
- get_predictor_class(class_name): Dynamically import and return the predictor class.

Usage:
- Typically run as part of a larger data processing pipeline.
"""

import importlib
import json
import logging
import sqlite3
from abc import ABC, abstractmethod

import joblib
import numpy as np
import pandas as pd
import torch

from src.config import config
from src.modeling.mlp_model import MLP

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s",
)

# Configuration
DB_PATH = config["database"]["path"]
PREDICTORS = config["predictors"]

# Ensure all columns are displayed in DataFrame outputs
pd.set_option("display.max_columns", None)


def calculate_pred_home_win_pct(home_scores, away_scores):
    """
    Calculate the win probability for the home team using a logistic function
    based on the score difference.

    Parameters:
    home_scores (array): The predicted scores of the home teams.
    away_scores (array): The predicted scores of the away teams.

    Returns:
    array: The win probabilities for the home teams.
    """
    # Base parameters for the logistic function
    a = -0.2504
    b = 0.1949

    # Calculate the predicted score difference for the home teams
    score_diffs = home_scores - away_scores

    # Calculate the win probabilities using the logistic function
    win_probs = 1 / (1 + np.exp(-(a + b * score_diffs)))

    return win_probs


class BasePredictor(ABC):
    def __init__(self, model_paths=None):
        """
        Initialize the BasePredictor with model paths.

        Parameters:
        model_paths (list of str): Paths to the model files.
        """
        self.model_paths = model_paths or []
        self.models = []

    @abstractmethod
    def load_models(self):
        """
        Load the predictive models. Must be implemented by subclasses.
        """
        pass

    def make_prediction_dict(self, game_ids, home_scores, away_scores):
        """
        Create a dictionary of predictions for given games.

        Parameters:
        game_ids (list of str): The IDs of the games.
        home_scores (array): The predicted scores of the home teams.
        away_scores (array): The predicted scores of the away teams.

        Returns:
        dict: A dictionary containing the predictions.
        """
        home_win_pcts = calculate_pred_home_win_pct(home_scores, away_scores)

        predictions = {}
        for game_id, home_score, away_score, home_win_pct in zip(
            game_ids, home_scores, away_scores, home_win_pcts
        ):
            predictions[game_id] = {
                "pred_home_score": home_score,
                "pred_away_score": away_score,
                "pred_home_win_pct": home_win_pct,
                "pred_players": {
                    "home": {},  # Placeholder for player predictions
                    "away": {},  # Placeholder for player predictions
                },
            }
        return predictions

    @abstractmethod
    def make_predictions(self, games):
        """
        Generate predictions for the given games. Must be implemented by subclasses.

        Parameters:
        games (dict): The game data to make predictions for.

        Returns:
        dict: The generated predictions.
        """
        pass


class LinearPredictor(BasePredictor):
    def load_models(self):
        """Load the linear regression model from the specified path."""
        self.model = joblib.load(self.model_paths[0])

    def make_predictions(self, games):
        """Generate predictions using the linear regression model."""
        game_ids = list(games.keys())
        features = [games[game_id] for game_id in game_ids]
        features_df = pd.DataFrame(features).fillna(0)

        scores = self.model.predict(features_df.values)
        home_scores, away_scores = scores[:, 0], scores[:, 1]

        return self.make_prediction_dict(game_ids, home_scores, away_scores)


class TreePredictor(BasePredictor):
    def load_models(self):
        """Load the decision tree model from the specified path."""
        self.model = joblib.load(self.model_paths[0])

    def make_predictions(self, games):
        """Generate predictions using the decision tree model."""
        game_ids = list(games.keys())
        features = [games[game_id] for game_id in game_ids]
        features_df = pd.DataFrame(features).fillna(0)

        scores = self.model.predict(features_df.values)
        home_scores, away_scores = scores[:, 0], scores[:, 1]

        return self.make_prediction_dict(game_ids, home_scores, away_scores)


class MLPPredictor(BasePredictor):
    def load_models(self):
        """Load the MLP model from the specified path and set up normalization parameters."""
        checkpoint = torch.load(self.model_paths[0])
        self.model = MLP(input_size=checkpoint["input_size"])
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.mean = checkpoint["mean"]
        self.std = checkpoint["std"]

    def make_predictions(self, games):
        """Generate predictions using the MLP model."""
        game_ids = list(games.keys())
        features = [games[game_id] for game_id in game_ids]
        features_df = pd.DataFrame(features).fillna(0)  # Handle NaN values

        self.model.eval()
        with torch.no_grad():
            features_tensor = torch.tensor(features_df.values, dtype=torch.float32)
            features_normalized = (features_tensor - self.mean) / self.std
            scores = self.model(features_normalized).numpy()
            home_scores, away_scores = scores[:, 0], scores[:, 1]

        return self.make_prediction_dict(game_ids, home_scores, away_scores)


class RandomPredictor(BasePredictor):
    def load_models(self):
        """Random predictor does not load any models."""
        pass

    def make_predictions(self, games):
        """Generate random predictions for the given games."""
        game_ids = list(games.keys())
        home_scores = np.random.randint(80, 131, size=len(game_ids))
        away_scores = np.random.randint(80, 131, size=len(game_ids))

        return self.make_prediction_dict(game_ids, home_scores, away_scores)


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


def make_predictions(predictor_name, games):
    """
    Generate predictions using the specified predictor.

    Parameters:
    predictor_name (str): The name of the predictor.
    games (dict): The game data to make predictions for.

    Returns:
    dict: The generated predictions.
    """
    # Handle the "Best" option by mapping it to the actual best predictor
    if predictor_name == "Best":
        predictor_name = PREDICTORS["Best"]

    predictor_cfg = PREDICTORS[predictor_name]
    class_name = predictor_cfg["class"]
    model_paths = predictor_cfg.get("model_paths", [])

    predictor_class = get_predictor_class(class_name)
    predictor = predictor_class(model_paths)
    predictor.load_models()
    predictions = predictor.make_predictions(games)
    return predictions


def save_predictions(predictions, predictor, db_path=DB_PATH):
    """
    Save predictions to the Predictions table.

    Parameters:
    predictions (dict): The predictions to save.
    predictor (str): The name of the predictor.
    db_path (str): The path to the SQLite database file. Defaults to DB_PATH from config.

    Returns:
    None
    """
    prediction_datetime = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    model_id = PREDICTORS[predictor]["model_paths"][0].split("/")[-1].split(".")[0]

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()

        data = [
            (
                game_id,
                predictor,
                model_id,
                prediction_datetime,
                json.dumps(
                    {
                        k: float(v) if isinstance(v, (np.float32, np.float64)) else v
                        for k, v in predictions[game_id].items()
                    }
                ),
            )
            for game_id in predictions.keys()
        ]

        print(data)  # Debugging print statement

        cursor.executemany(
            """
            INSERT INTO Predictions (game_id, predictor, model_id, prediction_datetime, prediction_set)
            VALUES (?, ?, ?, ?, ?)
            """,
            data,
        )

        conn.commit()
