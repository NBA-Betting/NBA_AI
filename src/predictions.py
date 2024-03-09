import numpy as np


def random_predictions(game):
    output_game = game.copy()

    # Generate random home and away scores from a normal distribution
    home_score = int(max(80, min(140, abs(np.random.normal(110, 15)))))
    away_score = int(max(80, min(140, abs(np.random.normal(110, 15)))))

    # Determine the winning team and winning percentage
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

    winning_team_pct = f"{winning_team_pct:.0%}"

    predictions = {
        "home_score": home_score,
        "away_score": away_score,
        "winning_team": winning_team,
        "winning_team_pct": winning_team_pct,
        "players": {"home": {}, "away": {}},
    }

    # Generate random player predictions
    if game["game_states"]:
        current_players = game["game_states"][-1]["players"]
        for player in current_players["home"]:
            predictions["players"]["home"][player] = {
                "name": current_players["home"][player]["name"],
                "points": int(max(0, min(40, abs(np.random.normal(20, 5))))),
            }
        # Add this loop for away team players
        for player in current_players["away"]:
            predictions["players"]["away"][player] = {
                "name": current_players["away"][player]["name"],
                "points": int(max(0, min(40, abs(np.random.normal(20, 5))))),
            }

    output_game["predictions"] = predictions

    return output_game
