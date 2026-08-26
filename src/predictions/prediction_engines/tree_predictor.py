"""
tree_predictor.py

This module provides an XGBoost predictor for NBA games.

Classes:
- TreePredictor: Uses XGBoost (gradient boosted trees) to generate predictions.

Model:
- XGBoost trained on 34 standardized features from Features table.
- Outputs [home_score, away_score] predictions.
- Requires scaler.json alongside model file for feature standardization.

Usage:
    predictor = TreePredictor(model_paths=["path/to/xgboost_model.joblib"])
    pre_game_predictions = predictor.make_pre_game_predictions(game_ids)
"""

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from src.predictions.prediction_engines.base_predictor import BaseMLPredictor
from src.predictions.prediction_utils import calculate_home_win_prob


class TreePredictor(BaseMLPredictor):
    """
    XGBoost predictor for NBA game scores.

    Loads pre-trained XGBoost model(s) from .joblib files.
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
        Load XGBoost models from .joblib files and the
        corresponding scaler parameters from scaler.json.

        Raises:
            ValueError: If model files cannot be loaded.
        """
        model_paths = [Path(model_path) for model_path in self.model_paths]
        if model_paths and all(path.suffix == ".json" for path in model_paths):
            if len(model_paths) != 2:
                raise ValueError(
                    "TreePredictor requires exactly two JSON models: home and away."
                )

            home_model, away_model = XGBRegressor(), XGBRegressor()
            home_model.load_model(model_paths[0])
            away_model.load_model(model_paths[1])
            self.models.append((home_model, away_model))
        else:
            for model_path in model_paths:
                self.models.append(joblib.load(model_path))

        # All model files share a scaler in their directory.
        scaler_path = model_paths[0].parent / "scaler.json" if model_paths else None
        if scaler_path and scaler_path.exists():
            with open(scaler_path) as f:
                scaler = json.load(f)
            self.scaler_mean = np.array(scaler["mean"])
            self.scaler_std = np.array(scaler["std"])
            logging.debug(f"TreePredictor: loaded scaler from {scaler_path}")
        else:
            logging.warning(
                "TreePredictor: scaler.json not found. Predictions will use "
                "unscaled features and may be inaccurate."
            )

    def make_pre_game_predictions(self, game_ids):
        """
        Generate predictions using XGBoost model.

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

        # Standardize features using training-set mean/std
        features_values = features_df.values
        if self.scaler_mean is not None:
            features_values = (features_values - self.scaler_mean) / self.scaler_std

        # Use the first model for predictions
        # v0.5 saves a tuple of (home_model, away_model); v0.4 saves a single multi-output model
        model = self.models[0]
        if isinstance(model, tuple):
            home_model, away_model = model
            home_scores = home_model.predict(features_values)
            away_scores = away_model.predict(features_values)
        else:
            scores = model.predict(features_values)
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
