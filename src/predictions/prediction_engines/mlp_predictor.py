"""
mlp_predictor.py

This module provides a predictor that uses a multi-layer perceptron (MLP) model to generate predictions for NBA games.
It consists of a class to:
- Generate pre-game predictions.
- Generate current predictions.

Classes:
- MLPPredictor: Predictor that uses an MLP model to generate predictions.

Methods:
- load_models(): Loads the MLP models from the specified paths and sets up normalization parameters.
- make_pre_game_predictions(game_ids): Generates pre-game predictions for the given game IDs.
- load_pre_game_data(game_ids): Loads pre-game data for the given game IDs.
- make_current_predictions(game_ids): Generates current predictions for the given game IDs.
- load_current_game_data(game_ids): Loads current game data for the given game IDs.

Usage:
- Typically used as part of the prediction generation process in the prediction_manager module.
- Can be instantiated and used to generate predictions for specified game IDs.

Example:
    predictor = MLPPredictor(model_paths=["path/to/model1.pth", "path/to/model2.pth"])
    pre_game_predictions = predictor.make_pre_game_predictions(game_ids)
    current_predictions = predictor.make_current_predictions(game_ids)
"""

import pandas as pd
import torch

from src.model_training.mlp_model import MLP
from src.predictions.features import load_feature_sets
from src.predictions.prediction_utils import (
    calculate_home_win_prob,
    load_current_game_data,
    update_predictions,
)


class MLPPredictor:
    """
    Predictor that uses a multi-layer perceptron (MLP) model to generate predictions for NBA games.

    This class loads an MLP model to make pre-game predictions and update them based on the current game state.
    """

    def __init__(self, model_paths=None):
        self.model_paths = model_paths or []
        self.models = []
        self.load_models()

    def load_models(self):
        """
        Load the MLP models from the specified paths and set up normalization parameters.

        This method initializes the MLP models using pre-trained model checkpoint files. It also sets up
        normalization parameters required for the model's inputs.
        """
        for model_path in self.model_paths:
            checkpoint = torch.load(model_path)
            model = MLP(input_size=checkpoint["input_size"])
            model.load_state_dict(checkpoint["model_state_dict"])
            model.mean = checkpoint["mean"]
            model.std = checkpoint["std"]
            self.models.append(model)

    def make_pre_game_predictions(self, game_ids):
        predictions = {}
        games = self.load_pre_game_data(game_ids)

        features = [games[game_id] for game_id in game_ids]
        features_df = pd.DataFrame(features).fillna(0)

        # Use the first model for predictions (modify as needed for multiple models)
        model = self.models[0]
        model.eval()
        with torch.no_grad():
            features_tensor = torch.tensor(features_df.values, dtype=torch.float32)
            features_normalized = (features_tensor - model.mean) / model.std
            scores = model(features_normalized).numpy()
            home_scores, away_scores = scores[:, 0], scores[:, 1]

        for game_id, home_score, away_score in zip(game_ids, home_scores, away_scores):
            home_win_prob = calculate_home_win_prob(home_score, away_score)
            predictions[game_id] = {
                "pred_home_score": float(home_score),
                "pred_away_score": float(away_score),
                "pred_home_win_pct": float(home_win_prob),
                "pred_players": games[game_id].get(
                    "pred_players", {"home": {}, "away": {}}
                ),
            }
        return predictions

    def load_pre_game_data(self, game_ids):
        feature_sets = load_feature_sets(game_ids)
        return feature_sets

    def make_current_predictions(self, game_ids):
        games = self.load_current_game_data(game_ids)
        current_predictions = update_predictions(games)
        return current_predictions

    def load_current_game_data(self, game_ids):
        return load_current_game_data(game_ids, predictor_name="MLP")
