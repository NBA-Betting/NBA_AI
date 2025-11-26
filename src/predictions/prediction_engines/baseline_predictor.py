"""
baseline_predictor.py

This module provides a formula-based baseline predictor for NBA games.

Classes:
- BaselinePredictor: Uses simple averaging formula to generate predictions.

Formula:
- pred_home_score = (home_ppg + away_opp_ppg) / 2
- pred_away_score = (away_ppg + home_opp_ppg) / 2

Usage:
    predictor = BaselinePredictor()
    pre_game_predictions = predictor.make_pre_game_predictions(game_ids)
"""

from src.predictions.prediction_engines.base_predictor import BasePredictor
from src.predictions.prediction_utils import calculate_home_win_prob


class BaselinePredictor(BasePredictor):
    """
    Formula-based predictor that averages team offensive and defensive stats.

    Uses 4 features from FeatureSets:
    - Home_PPG: Home team points per game
    - Home_OPP_PPG: Home team opponent points per game (defense)
    - Away_PPG: Away team points per game
    - Away_OPP_PPG: Away team opponent points per game (defense)
    """

    def make_pre_game_predictions(self, game_ids):
        """
        Generate predictions using baseline averaging formula.

        Args:
            game_ids (list): List of game IDs to predict.

        Returns:
            dict: Predictions for each game.
        """
        if not game_ids:
            return {}

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
