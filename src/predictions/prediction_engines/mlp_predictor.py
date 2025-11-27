"""
mlp_predictor.py

This module provides a PyTorch MLP predictor for NBA games.

Classes:
- MLPPredictor: Uses PyTorch neural network to generate predictions.

Model:
- Multi-layer perceptron trained on 43 features from FeatureSets.
- Outputs [home_score, away_score] predictions.
- Includes normalization parameters in checkpoint.

Usage:
    predictor = MLPPredictor(model_paths=["path/to/mlp_model.pth"])
    pre_game_predictions = predictor.make_pre_game_predictions(game_ids)
"""

import pandas as pd
import torch

from src.model_training.models import MLP
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
        - scaler_mean: Feature normalization mean
        - scaler_scale: Feature normalization scale
        - y_mean: Target normalization mean (optional)
        - y_std: Target normalization std (optional)

        Raises:
            ValueError: If checkpoint files cannot be loaded.
        """
        for model_path in self.model_paths:
            checkpoint = torch.load(model_path, weights_only=False)

            # Handle both old and new checkpoint formats
            hidden_sizes = checkpoint.get("hidden_sizes", [64, 32])
            dropout = checkpoint.get("dropout", 0.2)

            model = MLP(
                input_size=checkpoint["input_size"],
                hidden_sizes=hidden_sizes,
                dropout=dropout,
            )
            model.load_state_dict(checkpoint["model_state_dict"])

            # Store normalization params
            model.scaler_mean = checkpoint.get("scaler_mean")
            model.scaler_scale = checkpoint.get("scaler_scale")
            model.y_mean = checkpoint.get("y_mean")
            model.y_std = checkpoint.get("y_std")

            # Handle legacy format (mean/std instead of scaler_mean/scaler_scale)
            if model.scaler_mean is None and "mean" in checkpoint:
                model.scaler_mean = checkpoint["mean"]
                model.scaler_scale = checkpoint["std"]

            model.eval()
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
        with torch.no_grad():
            features_tensor = torch.tensor(features_df.values, dtype=torch.float32)

            # Normalize features
            if model.scaler_mean is not None:
                features_normalized = (
                    features_tensor - model.scaler_mean
                ) / model.scaler_scale
            else:
                features_normalized = features_tensor

            # Get predictions
            pred_norm = model(features_normalized).numpy()

            # Denormalize predictions if y normalization was used
            if model.y_mean is not None:
                y_mean = (
                    model.y_mean.numpy()
                    if isinstance(model.y_mean, torch.Tensor)
                    else model.y_mean
                )
                y_std = (
                    model.y_std.numpy()
                    if isinstance(model.y_std, torch.Tensor)
                    else model.y_std
                )
                scores = pred_norm * y_std + y_mean
            else:
                scores = pred_norm

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
