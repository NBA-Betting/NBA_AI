import logging

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s",
)


def get_predictions(games, predictor, model=None):
    """
    Get predictions for a game using a specified predictor.

    Parameters:
    games (list): A list of dictionaries containing game information.
    predictor (str): The name of the predictor to use.
    model (object): The model object to use for making predictions.

    Returns:
    list: A list of dictionaries containing the predictions for each game.
    """
    try:
        predictor_dict = {"Random": random_predictions}
        if predictor in predictor_dict:
            return predictor_dict[predictor](games, model)
        else:
            raise ValueError(f"Predictor '{predictor}' not found.")
    except Exception as e:
        logging.error(f"Failed to get predictions. Error: {e}")
        return []


def random_predictions(games, model=None):
    """
    Generate random predictions for a list of games.

    This function generates random scores for the home and away teams, determines the winning team and winning
    percentage, and generates random points for each player in the home and away teams. The predictions are
    returned as a list of dictionaries.

    Args:
        games (list): A list of dictionaries, each containing game information.
        model (object, optional): The model object to use for making predictions. Defaults to None.

    Returns:
        list: A list of dictionaries, each containing the generated predictions for a game.
    """
    predictions_list = []
    for game in games:
        # Generate random home and away scores from a normal distribution
        # The scores are constrained to be between 80 and 140
        home_score = int(max(80, min(140, abs(np.random.normal(110, 15)))))
        away_score = int(max(80, min(140, abs(np.random.normal(110, 15)))))

        # Determine the winning team and winning percentage
        # If the scores are equal, the away team is considered the winning team
        if home_score > away_score:
            winning_team = game["home"]
            winning_team_pct = (home_score + (home_score - away_score)) / (
                home_score + away_score
            )
        else:
            winning_team = game["away"]
            winning_team_pct = (away_score + (away_score - home_score)) / (
                home_score + away_score
            )

        # Format the winning percentage as a string with a percent sign
        winning_team_pct = f"{winning_team_pct:.0%}"

        # Initialize the predictions dictionary
        predictions = {
            "game_id": game["game_id"],
            "pred_home_score": home_score,
            "pred_away_score": away_score,
            "pred_winner": winning_team,
            "pred_win_pct": winning_team_pct,
            "pred_players": {"home": {}, "away": {}},
        }

        # Generate random player predictions if game["game_states"] is not empty
        if game["game_states"]:
            # Get the current players from the last game state
            current_players = game["game_states"][-1]["players_data"]

            # Generate random points for each home player
            # The points are constrained to be between 0 and 40
            for player in current_players["home"]:
                predictions["pred_players"]["home"][player] = {
                    "name": current_players["home"][player]["name"],
                    "pred_points": int(max(0, min(40, abs(np.random.normal(20, 5))))),
                }

            # Generate random points for each away player
            # The points are constrained to be between 0 and 40
            for player in current_players["away"]:
                predictions["pred_players"]["away"][player] = {
                    "name": current_players["away"][player]["name"],
                    "pred_points": int(max(0, min(40, abs(np.random.normal(20, 5))))),
                }

        predictions_list.append(predictions)

    return predictions_list
