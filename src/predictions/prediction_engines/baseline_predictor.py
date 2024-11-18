"""
baseline_predictor.py

This module provides a predictor that uses a baseline model to generate predictions for NBA games.
It consists of a class to:
- Generate pre-game predictions.
- Generate current predictions.

Classes:
- BaselinePredictor: Predictor that uses a baseline model to generate predictions.

Methods:
- load_models(): Loads the baseline models (no-op for baseline predictor).
- make_pre_game_predictions(game_ids): Generates pre-game predictions for the given game IDs.
- load_pre_game_data(game_ids): Loads pre-game data for the given game IDs.
- make_current_predictions(game_ids): Generates current predictions for the given game IDs.
- load_current_game_data(game_ids): Loads current game data for the given game IDs.

Usage:
- Typically used as part of the prediction generation process in the prediction_manager module.
- Can be instantiated and used to generate predictions for specified game IDs.

Example:
    predictor = BaselinePredictor()
    pre_game_predictions = predictor.make_pre_game_predictions(game_ids)
    current_predictions = predictor.make_current_predictions(game_ids)
"""

from src.predictions.features import load_feature_sets
from src.predictions.prediction_utils import (
    calculate_home_win_prob,
    load_current_game_data,
    update_predictions,
)


class BaselinePredictor:
    def __init__(self, model_paths=None):
        self.model_paths = model_paths or []
        self.models = []
        self.load_models()

    def load_models(self):
        pass

    def make_pre_game_predictions(self, game_ids):
        predictions = {}

        games = self.load_pre_game_data(game_ids)

        for game_id, game_data in games.items():
            # Ensure all required features are present
            if (
                "Home_PPG" in game_data
                and "Home_OPP_PPG" in game_data
                and "Away_PPG" in game_data
                and "Away_OPP_PPG" in game_data
            ):

                # Extract the relevant features
                home_ppg = game_data["Home_PPG"]
                home_opp_ppg = game_data["Home_OPP_PPG"]
                away_ppg = game_data["Away_PPG"]
                away_opp_ppg = game_data["Away_OPP_PPG"]

                # Calculate the predicted scores
                pred_home_score = (home_ppg + away_opp_ppg) / 2
                pred_away_score = (away_ppg + home_opp_ppg) / 2

                # Calculate the predicted win probability for the home team
                pred_home_win_pct = calculate_home_win_prob(
                    pred_home_score, pred_away_score
                )

                # Store predictions for the current game
                predictions[game_id] = {
                    "pred_home_score": float(pred_home_score),
                    "pred_away_score": float(pred_away_score),
                    "pred_home_win_pct": float(pred_home_win_pct),
                    "pred_players": game_data.get(
                        "pred_players", {"home": {}, "away": {}}
                    ),
                }

            else:
                # Skip games with missing data and optionally log the issue
                print(f"Skipping game {game_id} due to missing data")

        return predictions

    def load_pre_game_data(self, game_ids):
        feature_sets = load_feature_sets(game_ids)
        return feature_sets

    def make_current_predictions(self, game_ids):
        games = self.load_current_game_data(game_ids)
        current_predictions = update_predictions(games)
        return current_predictions

    def load_current_game_data(self, game_ids):
        return load_current_game_data(game_ids, predictor_name="Baseline")
