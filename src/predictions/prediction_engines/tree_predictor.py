"""
tree_predictor.py

This module provides an XGBoost predictor for NBA games.

Classes:
- TreePredictor: Uses XGBoost (gradient boosted trees) to generate predictions.

Model:
- XGBoost trained on 34 features from FeatureSets.
- Outputs [home_score, away_score] predictions.

Usage:
    predictor = TreePredictor(model_paths=["path/to/xgboost_model.joblib"])
    pre_game_predictions = predictor.make_pre_game_predictions(game_ids)
"""

import joblib
import pandas as pd

from src.predictions.prediction_engines.base_predictor import BaseMLPredictor
from src.predictions.prediction_utils import calculate_home_win_prob


class TreePredictor(BaseMLPredictor):
    """
    XGBoost predictor for NBA game scores.

    Loads pre-trained XGBoost model(s) from .joblib files.
    Uses first model in list for predictions.
    """

    def load_models(self):
        """
        Load XGBoost models from .joblib files.

        Raises:
            ValueError: If model files cannot be loaded.
        """
        for model_path in self.model_paths:
            self.models.append(joblib.load(model_path))

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

        # Use the first model for predictions
        scores = self.models[0].predict(features_df.values)
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
