"""
mlp_predictor.py

This module provides a PyTorch MLP predictor for NBA games.

Classes:
- MLPPredictor: Uses PyTorch neural network to generate predictions.

Model:
- Multi-layer perceptron trained on 34 features from FeatureSets.
- Outputs [home_score, away_score] predictions.
- Includes normalization parameters (mean, std) in checkpoint.

Usage:
    predictor = MLPPredictor(model_paths=["path/to/mlp_model.pth"])
    pre_game_predictions = predictor.make_pre_game_predictions(game_ids)
"""

import pandas as pd
import torch

from src.model_training.mlp_model import MLP
from src.predictions.prediction_engines.base_predictor import BaseMLPredictor
from src.predictions.prediction_utils import calculate_home_win_prob


class MLPPredictor(BaseMLPredictor):
    """
    PyTorch MLP predictor for NBA game scores.

    Loads pre-trained PyTorch model(s) from .pth checkpoint files.
    Uses first model in list for predictions.
    """

    def load_models(self):
        """
        Load PyTorch MLP models from .pth checkpoint files.

        Checkpoint must contain:
        - input_size: Number of input features
        - model_state_dict: Model weights
        - mean: Feature normalization mean
        - std: Feature normalization std

        Raises:
            ValueError: If checkpoint files cannot be loaded.
        """
        for model_path in self.model_paths:
            checkpoint = torch.load(model_path)
            model = MLP(input_size=checkpoint["input_size"])
            model.load_state_dict(checkpoint["model_state_dict"])
            model.mean = checkpoint["mean"]
            model.std = checkpoint["std"]
            self.models.append(model)

    def make_pre_game_predictions(self, game_ids):
        """
        Generate predictions using PyTorch MLP model.

        Args:
            game_ids (list): List of game IDs to predict.

        Returns:
            dict: Predictions for each game.

        Raises:
            ValueError: If models are not loaded.
        """
        if not game_ids:
            return {}
        if not self.models:
            raise ValueError(
                "Models are not loaded. Please load the models before making predictions."
            )

        predictions = {}
        games = self.load_pre_game_data(game_ids)

        features = [games[game_id] for game_id in game_ids]
        features_df = pd.DataFrame(features).fillna(0)

        # Use the first model for predictions
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
