"""
ensemble_predictor.py

This module provides a predictor that combines multiple prediction engines to generate ensemble predictions for NBA games.
It consists of a class to:
- Generate pre-game predictions by averaging predictions from multiple predictors.
- Generate current predictions by averaging predictions from multiple predictors.

Classes:
- EnsemblePredictor: Predictor that combines multiple prediction engines to generate ensemble predictions.

Methods:
- load_models(): Loads the component predictors.
- make_pre_game_predictions(game_ids): Generates pre-game predictions by averaging component predictor outputs.
- load_pre_game_data(game_ids): Loads pre-game data for the given game IDs.
- make_current_predictions(game_ids): Generates current predictions by averaging component predictor outputs.
- load_current_game_data(game_ids): Loads current game data for the given game IDs.

Usage:
- Typically used as part of the prediction generation process in the prediction_manager module.
- Can be instantiated and used to generate predictions for specified game IDs.

Example:
    predictor = EnsemblePredictor()
    pre_game_predictions = predictor.make_pre_game_predictions(game_ids)
    current_predictions = predictor.make_current_predictions(game_ids)
"""

import numpy as np

from src.config import config
from src.predictions.prediction_engines.linear_predictor import LinearPredictor
from src.predictions.prediction_engines.mlp_predictor import MLPPredictor
from src.predictions.prediction_engines.tree_predictor import TreePredictor
from src.predictions.prediction_utils import (
    calculate_home_win_prob,
    load_current_game_data,
    update_predictions,
)

# Configuration
PREDICTORS_CONFIG = config["predictors"]


class EnsemblePredictor:
    """
    Predictor that combines multiple prediction engines to generate ensemble predictions for NBA games.

    This class combines predictions from Linear, Tree, and MLP predictors using simple averaging
    to produce more robust predictions than any single model.
    """

    def __init__(self, model_paths=None):
        self.model_paths = model_paths or []
        self.component_predictors = []
        self.load_models()

    def load_models(self):
        """
        Load the component predictors (Linear, Tree, and MLP).

        This method initializes all component predictors that will be used in the ensemble.
        """
        # Initialize component predictors with their respective model paths from config
        predictor_configs = [
            ("Linear", LinearPredictor),
            ("Tree", TreePredictor),
            ("MLP", MLPPredictor),
        ]

        for predictor_name, predictor_class in predictor_configs:
            model_paths = PREDICTORS_CONFIG.get(predictor_name, {}).get(
                "model_paths", []
            )
            try:
                predictor = predictor_class(model_paths=model_paths)
                self.component_predictors.append((predictor_name, predictor))
            except Exception as e:
                # Log warning but continue with other predictors
                print(
                    f"Warning: Could not load {predictor_name} predictor: {e}. Skipping."
                )

        if not self.component_predictors:
            raise ValueError(
                "No component predictors could be loaded for the ensemble."
            )

    def make_pre_game_predictions(self, game_ids):
        """
        Generate pre-game predictions by averaging predictions from all component predictors.

        Parameters:
        game_ids (list): A list of game IDs to generate predictions for.

        Returns:
        dict: A dictionary of ensemble predictions, including averaged scores and win probabilities.
        """
        if not game_ids:
            return {}

        # Collect predictions from all component predictors
        all_predictions = {}
        for predictor_name, predictor in self.component_predictors:
            try:
                predictions = predictor.make_pre_game_predictions(game_ids)
                all_predictions[predictor_name] = predictions
            except Exception as e:
                print(
                    f"Warning: {predictor_name} predictor failed: {e}. Excluding from ensemble."
                )

        if not all_predictions:
            raise ValueError("All component predictors failed to generate predictions.")

        # Average the predictions across all successful predictors
        ensemble_predictions = {}
        for game_id in game_ids:
            home_scores = []
            away_scores = []
            valid_predictions = 0

            for predictor_name, predictions in all_predictions.items():
                if game_id in predictions:
                    pred = predictions[game_id]
                    home_scores.append(pred["pred_home_score"])
                    away_scores.append(pred["pred_away_score"])
                    valid_predictions += 1

            if valid_predictions > 0:
                avg_home_score = float(np.mean(home_scores))
                avg_away_score = float(np.mean(away_scores))
                avg_win_prob = calculate_home_win_prob(avg_home_score, avg_away_score)

                # Get pred_players from first available prediction
                pred_players = {"home": {}, "away": {}}
                for predictor_name, predictions in all_predictions.items():
                    if game_id in predictions:
                        pred_players = predictions[game_id].get(
                            "pred_players", {"home": {}, "away": {}}
                        )
                        break

                ensemble_predictions[game_id] = {
                    "pred_home_score": avg_home_score,
                    "pred_away_score": avg_away_score,
                    "pred_home_win_pct": avg_win_prob,
                    "pred_players": pred_players,
                }

        return ensemble_predictions

    def load_pre_game_data(self, game_ids):
        """
        Load pre-game data for the given game IDs.

        This method delegates to the first available component predictor.
        """
        if self.component_predictors:
            return self.component_predictors[0][1].load_pre_game_data(game_ids)
        return {}

    def make_current_predictions(self, game_ids):
        """
        Generate current predictions by averaging predictions from all component predictors.

        Parameters:
        game_ids (list): A list of game IDs to generate predictions for.

        Returns:
        dict: A dictionary of ensemble predictions for the current game state.
        """
        if not game_ids:
            return {}

        games = self.load_current_game_data(game_ids)
        current_predictions = update_predictions(games)
        return current_predictions

    def load_current_game_data(self, game_ids):
        """
        Load current game data for the given game IDs.

        This method loads game data using the Ensemble predictor name.
        """
        return load_current_game_data(game_ids, predictor_name="Ensemble")
