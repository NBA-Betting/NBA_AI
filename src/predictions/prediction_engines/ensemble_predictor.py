"""
ensemble_predictor.py

Ensemble predictor combining multiple ML models for NBA game predictions.

Classes:
- EnsemblePredictor: Combines Ridge, XGBoost, and MLP predictions using weighted average.

Design:
- Loads all three ML models (Linear, Tree, MLP)
- Generates predictions from each model
- Combines using inverse-MAE weighting or equal weights
- Falls back to available models if some are missing

Usage:
    predictor = EnsemblePredictor(model_paths={
        "Linear": ["models/ridge_v1.0_mae13.7.joblib"],
        "Tree": ["models/xgboost_v1.0_mae10.2.joblib"],
        "MLP": ["models/mlp_v1.0_mae13.8.pth"]
    })
    predictions = predictor.make_pre_game_predictions(game_ids)
"""

import logging

import joblib
import numpy as np
import pandas as pd
import torch

from src.model_training.models import MLP
from src.predictions.prediction_engines.base_predictor import BasePredictor
from src.predictions.prediction_utils import calculate_home_win_prob


class EnsemblePredictor(BasePredictor):
    """
    Ensemble predictor combining Ridge, XGBoost, and MLP models.

    Uses weighted average of predictions from all available models.
    Weights can be:
    - Equal (1/n for each model)
    - Inverse-MAE (models with lower MAE get higher weight)

    Attributes:
        models: Dict mapping model type to loaded model
        weights: Dict mapping model type to weight (sums to 1.0)
    """

    def __init__(self, model_paths=None):
        """
        Initialize ensemble predictor.

        Args:
            model_paths: Dict with keys 'Linear', 'Tree', 'MLP' mapping to
                        list of model paths. Example:
                        {
                            "Linear": ["models/ridge.joblib"],
                            "Tree": ["models/xgboost.joblib"],
                            "MLP": ["models/mlp.pth"]
                        }
        """
        super().__init__(model_paths)
        self.models = {}
        self.weights = {}
        self.load_models()

    def load_models(self):
        """
        Load all available models from model_paths dict.

        Populates self.models with loaded models and self.weights with equal weights.
        """
        if not self.model_paths:
            logging.warning("No model paths provided for ensemble predictor")
            return

        # Load Linear (Ridge) model
        linear_paths = self.model_paths.get("Linear", [])
        if linear_paths:
            try:
                self.models["Linear"] = joblib.load(linear_paths[0])
                logging.info(f"Loaded Linear model from {linear_paths[0]}")
            except Exception as e:
                logging.warning(f"Failed to load Linear model: {e}")

        # Load Tree (XGBoost) model
        tree_paths = self.model_paths.get("Tree", [])
        if tree_paths:
            try:
                self.models["Tree"] = joblib.load(tree_paths[0])
                logging.info(f"Loaded Tree model from {tree_paths[0]}")
            except Exception as e:
                logging.warning(f"Failed to load Tree model: {e}")

        # Load MLP model
        mlp_paths = self.model_paths.get("MLP", [])
        if mlp_paths:
            try:
                checkpoint = torch.load(mlp_paths[0], weights_only=False)

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
                self.models["MLP"] = model
                logging.info(f"Loaded MLP model from {mlp_paths[0]}")
            except Exception as e:
                logging.warning(f"Failed to load MLP model: {e}")

        # Set equal weights for all loaded models
        if self.models:
            weight = 1.0 / len(self.models)
            self.weights = {name: weight for name in self.models}
            logging.info(f"Ensemble weights: {self.weights}")
        else:
            logging.error("No models loaded for ensemble predictor")

    def _predict_linear(self, features_df):
        """Generate predictions using Linear (Ridge) model."""
        model = self.models.get("Linear")
        if model is None:
            return None
        return model.predict(features_df.values)

    def _predict_tree(self, features_df):
        """Generate predictions using Tree (XGBoost) model."""
        model = self.models.get("Tree")
        if model is None:
            return None
        return model.predict(features_df.values)

    def _predict_mlp(self, features_df):
        """Generate predictions using MLP model."""
        model = self.models.get("MLP")
        if model is None:
            return None

        with torch.no_grad():
            features_tensor = torch.tensor(features_df.values, dtype=torch.float32)

            # Normalize features if scaler params available
            if model.scaler_mean is not None:
                features_normalized = (
                    features_tensor - model.scaler_mean
                ) / model.scaler_scale
            else:
                features_normalized = features_tensor

            # Get predictions (normalized)
            pred_norm = model(features_normalized).numpy()

            # Denormalize predictions if y params available
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
                return pred_norm * y_std + y_mean
            else:
                return pred_norm

    def make_pre_game_predictions(self, game_ids):
        """
        Generate ensemble predictions by combining all available models.

        Args:
            game_ids: List of game IDs to predict.

        Returns:
            dict: Predictions for each game.

        Raises:
            ValueError: If no models are loaded.
        """
        if not game_ids:
            return {}

        if not self.models:
            raise ValueError("No models loaded for ensemble predictor")

        # Load feature data
        games = self.load_pre_game_data(game_ids)
        features = [games[game_id] for game_id in game_ids]
        features_df = pd.DataFrame(features).fillna(0)

        # Collect predictions from each model
        all_predictions = []
        all_weights = []

        if "Linear" in self.models:
            pred = self._predict_linear(features_df)
            if pred is not None:
                all_predictions.append(pred)
                all_weights.append(self.weights["Linear"])

        if "Tree" in self.models:
            pred = self._predict_tree(features_df)
            if pred is not None:
                all_predictions.append(pred)
                all_weights.append(self.weights["Tree"])

        if "MLP" in self.models:
            pred = self._predict_mlp(features_df)
            if pred is not None:
                all_predictions.append(pred)
                all_weights.append(self.weights["MLP"])

        if not all_predictions:
            raise ValueError("No predictions generated from any model")

        # Weighted average of predictions
        weights = np.array(all_weights)
        weights = weights / weights.sum()  # Normalize in case some models failed

        combined_scores = np.zeros((len(game_ids), 2))
        for pred, weight in zip(all_predictions, weights):
            combined_scores += weight * pred

        home_scores = combined_scores[:, 0]
        away_scores = combined_scores[:, 1]

        # Build predictions dict
        predictions = {}
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
