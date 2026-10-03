"""
linear_predictor.py

This module provides a Ridge Regression predictor for NBA games.

Classes:
- LinearPredictor: Uses scikit-learn Ridge Regression to generate predictions.

Model:
- Ridge Regression trained on 43 standardized features from Features table.
- Outputs [home_score, away_score] predictions.
- Requires scaler.json alongside model file for feature standardization.

Usage:
    predictor = LinearPredictor(model_paths=["path/to/ridge_model.joblib"])
    pre_game_predictions = predictor.make_pre_game_predictions(game_ids)
"""

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.predictions.prediction_engines.base_predictor import BaseMLPredictor
from src.predictions.prediction_utils import calculate_home_win_prob


class LinearPredictor(BaseMLPredictor):
    """
    Ridge Regression predictor for NBA game scores.

    Loads pre-trained Ridge Regression model(s) from .joblib files.
    Uses first model in list for predictions.

    The model was trained on standardized features (zero mean, unit variance),
    so raw features must be scaled using the saved scaler parameters before
    prediction.
    """

    def __init__(self, model_paths=None):
        self.scaler_mean = None
        self.scaler_std = None
        super().__init__(model_paths)

    def load_models(self):
        """
        Load Ridge Regression models from .joblib files and the
        corresponding scaler parameters from scaler.json.

        Raises:
            ValueError: If model files cannot be loaded.
        """
        for model_path in self.model_paths:
            self.models.append(joblib.load(model_path))

            # Load scaler from same directory as model file
            scaler_path = Path(model_path).parent / "scaler.json"
            if scaler_path.exists():
                with open(scaler_path) as f:
                    scaler = json.load(f)
                self.scaler_mean = np.array(scaler["mean"])
                self.scaler_std = np.array(scaler["std"])
                logging.debug(f"LinearPredictor: loaded scaler from {scaler_path}")
            else:
                logging.warning(
                    f"LinearPredictor: scaler.json not found at {scaler_path}. "
                    f"Predictions will use unscaled features and may be inaccurate."
                )

    def make_pre_game_predictions(self, game_ids):
        """
        Generate predictions using Ridge Regression model.

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
        game_ids = list(games)
        if not game_ids:
            return {}

        features = [games[game_id] for game_id in game_ids]
        features_df = pd.DataFrame(features).fillna(0)

        # Standardize features using training-set mean/std
        features_values = features_df.values
        if self.scaler_mean is not None:
            features_values = (features_values - self.scaler_mean) / self.scaler_std

        # Use the first model for predictions
        scores = self.models[0].predict(features_values)
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
