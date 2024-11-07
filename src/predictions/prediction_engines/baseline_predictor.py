from src.predictions.features import load_feature_sets
from src.predictions.prediction_utils import calculate_home_win_prob


class BaselinePredictor:
    def make_pre_game_predictions(self, game_ids):
        predictions = {}

        games = self.load_data_pre_game(game_ids)

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
                    "pred_home_score": pred_home_score,
                    "pred_away_score": pred_away_score,
                    "pred_home_win_pct": pred_home_win_pct,
                    "pred_players": game_data.get(
                        "pred_players", {"home": {}, "away": {}}
                    ),
                }

            else:
                # Skip games with missing data and optionally log the issue
                print(f"Skipping game {game_id} due to missing data")

        return predictions

    def make_current_predictions(self, game_ids):
        pass

    def load_data_pre_game(self, game_ids):
        feature_sets = load_feature_sets(game_ids)
        return feature_sets

    def load_data_current(self, game_ids):
        pass
