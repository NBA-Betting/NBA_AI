class MLPPredictor(BasePredictor):
    """
    Predictor that uses a multi-layer perceptron (MLP) model to generate predictions for NBA games.

    This class loads an MLP model to make pre-game predictions and update them based on the current game state.
    """

    def load_models(self):
        """
        Load the MLP model from the specified path and set up normalization parameters.

        This method initializes the MLP model using a pre-trained model checkpoint file. It also sets up
        normalization parameters required for the model's inputs.
        """
        checkpoint = torch.load(self.model_paths[0])
        self.model = MLP(input_size=checkpoint["input_size"])
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.mean = checkpoint["mean"]
        self.std = checkpoint["std"]

    def make_pre_game_predictions(self, games):
        """
        Generate pre-game predictions using the MLP model.

        Parameters:
        games (dict): A dictionary containing game data, with each game having associated features.

        Returns:
        dict: A dictionary of predictions, including predicted scores and win probabilities for each game.
        """
        game_ids = list(games.keys())
        features = [games[game_id] for game_id in game_ids]
        features_df = pd.DataFrame(features).fillna(0)  # Handle NaN values

        self.model.eval()
        with torch.no_grad():
            features_tensor = torch.tensor(features_df.values, dtype=torch.float32)
            features_normalized = (features_tensor - self.mean) / self.std
            scores = self.model(features_normalized).numpy()
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
