class LinearPredictor(BasePredictor):
    """
    Predictor that uses a linear regression model to generate predictions for NBA games.

    This class loads a linear regression model to make pre-game predictions and update them
    based on the current game state.
    """

    def load_models(self):
        """
        Load the linear regression model from the specified path.

        This method initializes the linear regression model using a pre-trained model file.
        """
        self.model = joblib.load(self.model_paths[0])

    def make_pre_game_predictions(self, games):
        """
        Generate pre-game predictions using the linear regression model.

        Parameters:
        games (dict): A dictionary containing game data, with each game having associated features.

        Returns:
        dict: A dictionary of predictions, including predicted scores and win probabilities for each game.
        """
        game_ids = list(games.keys())
        features = [games[game_id] for game_id in game_ids]
        features_df = pd.DataFrame(features).fillna(0)

        scores = self.model.predict(features_df.values)
        home_scores, away_scores = scores[:, 0], scores[:, 1]

        # Generate predictions dictionary
        predictions = {}
        for game_id, home_score, away_score in zip(game_ids, home_scores, away_scores):
            home_win_prob = calculate_home_win_prob(home_score, away_score)
            predictions[game_id] = {
                "pred_home_score": home_score,
                "pred_away_score": away_score,
                "pred_home_win_pct": home_win_prob,
                "pred_players": games[game_id].get(
                    "pred_players", {"home": {}, "away": {}}
                ),
            }
        return predictions

    def update_predictions(self, games):
        """
        Update predictions based on the current state of the games.

        Parameters:
        games (dict): A dictionary containing current game states and pre-game predictions.

        Returns:
        dict: A dictionary of updated predictions based on real-time game data.
        """
        return super().update_predictions(games)
