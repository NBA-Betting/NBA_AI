import numpy as np


def get_predictions(game, predictor):
    predictor_dict = {"random": random_predictions}
    if predictor in predictor_dict:
        return predictor_dict[predictor](game)
    else:
        raise ValueError(f"Predictor '{predictor}' not found.")


def random_predictions(game):
    """
    Generate random predictions for a game.

    This function generates random scores for the home and away teams, determines the winning team and winning
    percentage, and generates random points for each player in the home and away teams. The predictions are
    returned as a dictionary.

    Args:
        game (dict): A dictionary containing game information.

    Returns:
        dict: A dictionary containing the generated predictions.
    """
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
        "home_score": home_score,
        "away_score": away_score,
        "winning_team": winning_team,
        "winning_team_pct": winning_team_pct,
        "players": {"home": {}, "away": {}},
    }

    # Generate random player predictions if game["game_states"] is not empty
    if game["game_states"]:
        # Get the current players from the last game state
        current_players = game["game_states"][-1]["players"]

        # Generate random points for each home player
        # The points are constrained to be between 0 and 40
        for player in current_players["home"]:
            predictions["players"]["home"][player] = {
                "name": current_players["home"][player]["name"],
                "points": int(max(0, min(40, abs(np.random.normal(20, 5))))),
            }

        # Generate random points for each away player
        # The points are constrained to be between 0 and 40
        for player in current_players["away"]:
            predictions["players"]["away"][player] = {
                "name": current_players["away"][player]["name"],
                "points": int(max(0, min(40, abs(np.random.normal(20, 5))))),
            }

    return predictions
