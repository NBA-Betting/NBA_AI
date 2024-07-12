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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s",
)

# Configuration
DB_PATH = config["database"]["path"]
PREDICTORS = config["predictors"]

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
        self.model_paths = model_paths or []
        self.models = []

    @abstractmethod
    def load_models(self):
        pass

    def make_prediction_dict(self, game_ids, home_scores, away_scores):
        """
        Creates a dictionary of predictions for given games.

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
        pass


class LinearPredictor(BasePredictor):
    def load_models(self):
        self.model = joblib.load(self.model_paths[0])

    def make_predictions(self, games):
        game_ids = list(games.keys())
        features = [games[game_id] for game_id in game_ids]
        features_df = pd.DataFrame(features).fillna(0)

        scores = self.model.predict(features_df.values)
        home_scores, away_scores = scores[:, 0], scores[:, 1]

        return self.make_prediction_dict(game_ids, home_scores, away_scores)


class TreePredictor(BasePredictor):
    def load_models(self):
        self.model = joblib.load(self.model_paths[0])

    def make_predictions(self, games):
        game_ids = list(games.keys())
        features = [games[game_id] for game_id in game_ids]
        features_df = pd.DataFrame(features).fillna(0)

        scores = self.model.predict(features_df.values)
        home_scores, away_scores = scores[:, 0], scores[:, 1]

        return self.make_prediction_dict(game_ids, home_scores, away_scores)


class MLPPredictor(BasePredictor):
    def load_models(self):
        checkpoint = torch.load(self.model_paths[0])
        self.model = MLP(input_size=checkpoint["input_size"])
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.mean = checkpoint["mean"]
        self.std = checkpoint["std"]

    def make_predictions(self, games):
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
        pass

    def make_predictions(self, games):
        game_ids = list(games.keys())
        home_scores = np.random.randint(80, 131, size=len(game_ids))
        away_scores = np.random.randint(80, 131, size=len(game_ids))

        return self.make_prediction_dict(game_ids, home_scores, away_scores)


def get_predictor_class(class_name):
    module = importlib.import_module(__name__)
    predictor_class = getattr(module, class_name)
    return predictor_class


def make_predictions(predictor_name, games):
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
                json.dumps(predictions[game_id]),
            )
            for game_id in predictions.keys()
        ]

        cursor.executemany(
            """
            INSERT INTO Predictions (game_id, predictor, model_id, prediction_datetime, prediction_set)
            VALUES (?, ?, ?, ?, ?)
            """,
            data,
        )

        conn.commit()


# Example usage
if __name__ == "__main__":
    games = {
        "game_1": {
            "Home_Win_Pct": 0.5,
            "Home_PPG": 117.5,
            "Home_OPP_PPG": 114.0,
            "Home_Net_PPG": 3.5,
            "Away_Win_Pct": 0.5,
            "Away_PPG": 118.0,
            "Away_OPP_PPG": 117.0,
            "Away_Net_PPG": 1.0,
            "Win_Pct_Diff": 0.0,
            "PPG_Diff": -0.5,
            "OPP_PPG_Diff": -3.0,
            "Net_PPG_Diff": 2.5,
            "Home_Win_Pct_Home": 0,
            "Home_PPG_Home": np.nan,
            "Home_OPP_PPG_Home": np.nan,
            "Home_Net_PPG_Home": np.nan,
            "Away_Win_Pct_Away": 0.0,
            "Away_PPG_Away": 106.0,
            "Away_OPP_PPG_Away": 112.0,
            "Away_Net_PPG_Away": -6.0,
            "Win_Pct_Home_Away_Diff": 0.0,
            "PPG_Home_Away_Diff": np.nan,
            "OPP_PPG_Home_Away_Diff": np.nan,
            "Net_PPG_Home_Away_Diff": np.nan,
            "Time_Decay_Home_Win_Pct": 0.5346019613807635,
            "Time_Decay_Home_PPG": 116.98097057928855,
            "Time_Decay_Home_OPP_PPG": 113.16955292686168,
            "Time_Decay_Home_Net_PPG": 3.8114176524268686,
            "Time_Decay_Away_Win_Pct": 0.4482004813398909,
            "Time_Decay_Away_PPG": 116.75681155215736,
            "Time_Decay_Away_OPP_PPG": 116.4820048133989,
            "Time_Decay_Away_Net_PPG": 0.27480673875845696,
            "Time_Decay_Win_Pct_Diff": 0.08640148004087267,
            "Time_Decay_PPG_Diff": 0.2241590271311935,
            "Time_Decay_OPP_PPG_Diff": -3.312451886537218,
            "Time_Decay_Net_PPG_Diff": 3.5366109136684116,
            "Day_of_Season": 3.5,
            "Home_Rest_Days": 1,
            "Home_Game_Freq": 1.0,
            "Away_Rest_Days": 1,
            "Away_Game_Freq": 0.0,
            "Rest_Days_Diff": 0,
            "Game_Freq_Diff": 1.0,
        },
        "game_2": {
            "Home_Win_Pct": 0.6,
            "Home_PPG": 120.0,
            "Home_OPP_PPG": 110.0,
            "Home_Net_PPG": 10.0,
            "Away_Win_Pct": 0.4,
            "Away_PPG": 115.0,
            "Away_OPP_PPG": 113.0,
            "Away_Net_PPG": 2.0,
            "Win_Pct_Diff": 0.2,
            "PPG_Diff": 5.0,
            "OPP_PPG_Diff": -3.0,
            "Net_PPG_Diff": 8.0,
            "Home_Win_Pct_Home": 0.5,
            "Home_PPG_Home": 118.0,
            "Home_OPP_PPG_Home": 112.0,
            "Home_Net_PPG_Home": 6.0,
            "Away_Win_Pct_Away": 0.3,
            "Away_PPG_Away": 110.0,
            "Away_OPP_PPG_Away": 115.0,
            "Away_Net_PPG_Away": -5.0,
            "Win_Pct_Home_Away_Diff": 0.2,
            "PPG_Home_Away_Diff": 8.0,
            "OPP_PPG_Home_Away_Diff": -3.0,
            "Net_PPG_Home_Away_Diff": 11.0,
            "Time_Decay_Home_Win_Pct": 0.6,
            "Time_Decay_Home_PPG": 118.0,
            "Time_Decay_Home_OPP_PPG": 110.0,
            "Time_Decay_Home_Net_PPG": 8.0,
            "Time_Decay_Away_Win_Pct": 0.4,
            "Time_Decay_Away_PPG": 115.0,
            "Time_Decay_Away_OPP_PPG": 113.0,
            "Time_Decay_Away_Net_PPG": 2.0,
            "Time_Decay_Win_Pct_Diff": 0.2,
            "Time_Decay_PPG_Diff": 3.0,
            "Time_Decay_OPP_PPG_Diff": -3.0,
            "Time_Decay_Net_PPG_Diff": 6.0,
            "Day_of_Season": 5.0,
            "Home_Rest_Days": 2,
            "Home_Game_Freq": 1.0,
            "Away_Rest_Days": 1,
            "Away_Game_Freq": 0.5,
            "Rest_Days_Diff": 1,
            "Game_Freq_Diff": 0.5,
        },
    }  # Example games dictionary
    predictor_name = (
        "Tree"  # This could be 'Tree', 'Linear', 'MLP', 'Random', or 'Best'
    )
    predictions = make_predictions(predictor_name, games)
    print(predictions)
