import numpy as np

# Pre-game logistic constants, refit July 2026 on 10,410 predicted-margin vs
# outcome pairs (2023-24 + 2024-25 seasons, all legacy predictors pooled).
# The original constants (a=-0.2504, b=0.1949) had roughly twice the correct
# slope — a predicted margin maps to a much less certain outcome than an
# in-game score difference does — and imposed an unexplained away lean at
# margin 0. Improved live Brier for Baseline/MLP/Tree, neutral for Linear.
PREGAME_A = 0.0278
PREGAME_B = 0.0984


def calculate_home_win_prob(
    home_score, away_score, minutes_remaining=None, adjustment_type="logarithmic"
):
    """
    Calculate the win probability for the home team using a logistic function
    based on the score difference and, optionally, the time remaining.

    This function computes the probability that the home team will win the game based on the
    current or predicted score difference between the home and away teams. The calculation can
    account for the time remaining in the game, reflecting increased certainty as the game progresses.

    Rationale:
    As a game progresses and the remaining time decreases, the likelihood of a comeback diminishes.
    Thus, the same score difference becomes more indicative of the final outcome when less time is left.
    This function adjusts the win probability calculation to reflect this increasing certainty.

    Parameters:
    home_score (float): The predicted or current score of the home team.
    away_score (float): The predicted or current score of the away team.
    minutes_remaining (float, optional): The minutes remaining in the game. If None, assume a pre-game scenario.
    adjustment_type (str): The type of adjustment to use for in-game calculation:
                           - 'linear': A linear adjustment factor that increases certainty as time decreases.
                           - 'logarithmic': A logarithmic adjustment, providing more sensitivity near the end of the game.

    Returns:
    float: The win probability for the home team, ranging from 0 to 1.
    """
    # Base parameters for the logistic function
    base_a = (
        -0.2504
    )  # Intercept parameter, establishing baseline probability without score difference
    base_b = 0.1949  # Coefficient for score difference, defining the slope of the logistic curve

    # Calculate the score difference, a key factor in determining win probability
    score_diff = home_score - away_score

    # Pre-game scenario: use the refit pre-game constants (see module header).
    # The in-game path below keeps the original, steeper curve — an actual
    # score lead is far more informative than a predicted margin.
    if minutes_remaining is None:
        win_prob = float(1 / (1 + np.exp(-(PREGAME_A + PREGAME_B * score_diff))))
    else:
        # In-game scenario: Adjust the logistic function based on time remaining
        # Linear and logarithmic adjustments increase certainty as time decreases

        if adjustment_type == "linear":
            # Linear adjustment: certainty increases steadily as minutes_remaining decreases
            time_factor = 48 / (minutes_remaining + 1)
        elif adjustment_type == "logarithmic":
            # Logarithmic adjustment: certainty increases more sharply near the end of the game
            time_factor = np.log(48 / (minutes_remaining + 1))
        else:
            raise ValueError(
                "Invalid adjustment type. Choose 'linear' or 'logarithmic'."
            )

        # Adjust the coefficient 'base_b' to reflect increased certainty
        adjusted_b = base_b * (1 + time_factor)

        # Calculate the win probability using the adjusted logistic function
        win_prob = float(1 / (1 + np.exp(-(base_a + adjusted_b * score_diff))))

    return win_prob
