import re

import numpy as np


def calculate_updated_scores(
    scores, fraction_of_game_remaining, method="weighted", logistic_params=None
):
    """
    Calculate updated home and away scores based on pre-game predictions, current scores, and game progress.

    This function updates predictions for the final scores of a basketball game by integrating pre-game predictions
    with current game scores. The update accounts for the progression of the game, adjusting the weight of current
    and predicted scores based on the time remaining. The adjustment method can be chosen to reflect different
    levels of certainty as the game progresses.

    Rationale:
    As the game progresses, the actual game state becomes increasingly meaningful for predicting the final outcome,
    while pre-game predictions become less relevant. This is because there is less time for significant changes
    to occur, making the current scores a more reliable indicator of the final results. The function provides
    various methods to adjust the influence of pre-game predictions versus current scores as the game progresses.

    Parameters:
    scores (dict): A dictionary containing the following keys:
                   - 'pregame_pred_home_score' (float): The pre-game predicted score for the home team.
                   - 'pregame_pred_away_score' (float): The pre-game predicted score for the away team.
                   - 'current_home_score' (float): The current actual score for the home team.
                   - 'current_away_score' (float): The current actual score for the away team.
    fraction_of_game_remaining (float): The fraction of the game remaining, ranging from 0 to 1.
    method (str): The method to use for weighting the scores:
                  - 'simple': A simple average of pre-game and extrapolated current scores.
                  - 'weighted': A weighted average based on the fraction of the game completed.
                  - 'logistic': Uses a logistic function to dynamically adjust the weighting.
    logistic_params (tuple, optional): Parameters for the logistic function, with default values (0.5, 10).
                                       The tuple consists of:
                                       - x0: The midpoint of the logistic curve.
                                       - k: The steepness of the logistic curve.

    Returns:
    tuple: The updated scores (home, away) as floats.
    """
    pregame_pred_home_score = scores["pregame_pred_home_score"]
    pregame_pred_away_score = scores["pregame_pred_away_score"]
    current_home_score = scores["current_home_score"]
    current_away_score = scores["current_away_score"]

    # Calculate the fraction of the game completed
    fraction_of_game_completed = 1 - fraction_of_game_remaining

    # Extrapolate current scores to estimate full game scores
    if fraction_of_game_remaining == 0:
        # Avoid division by zero when the game is complete
        extrapolated_home_score = current_home_score
        extrapolated_away_score = current_away_score
    else:
        extrapolated_home_score = current_home_score / fraction_of_game_completed
        extrapolated_away_score = current_away_score / fraction_of_game_completed

    # Cap the extrapolated scores to avoid unrealistic predictions
    # This is to ensure that the extrapolated scores don't exceed reasonable bounds
    MAX_POINTS_PER_TEAM = 150
    MIN_POINTS_PER_TEAM = 70
    extrapolated_home_score = max(
        min(extrapolated_home_score, MAX_POINTS_PER_TEAM), MIN_POINTS_PER_TEAM
    )
    extrapolated_away_score = max(
        min(extrapolated_away_score, MAX_POINTS_PER_TEAM), MIN_POINTS_PER_TEAM
    )

    if method == "simple":
        # Simple average of the pre-game predictions and extrapolated current scores
        updated_home_score = (extrapolated_home_score + pregame_pred_home_score) / 2
        updated_away_score = (extrapolated_away_score + pregame_pred_away_score) / 2
    elif method == "weighted":
        # Weighted average, with more weight on current scores as the game progresses
        # This method reflects the increasing reliability of current scores as more of the game is played
        weight_for_current = fraction_of_game_completed
        weight_for_pred = fraction_of_game_remaining

        updated_home_score = (
            weight_for_current * extrapolated_home_score
            + weight_for_pred * pregame_pred_home_score
        )
        updated_away_score = (
            weight_for_current * extrapolated_away_score
            + weight_for_pred * pregame_pred_away_score
        )
    elif method == "logistic":
        # Logistic function for dynamic weighting based on game progress
        # The logistic function can model a smooth transition from less to more certainty
        x0, k = logistic_params if logistic_params else (0.5, 10)
        weight_for_current = 1 / (1 + np.exp(-k * (fraction_of_game_completed - x0)))
        weight_for_pred = 1 - weight_for_current

        updated_home_score = (
            weight_for_current * extrapolated_home_score
            + weight_for_pred * pregame_pred_home_score
        )
        updated_away_score = (
            weight_for_current * extrapolated_away_score
            + weight_for_pred * pregame_pred_away_score
        )
    else:
        raise ValueError(
            "Invalid method specified. Use 'simple', 'weighted', or 'logistic'."
        )

    return updated_home_score, updated_away_score


def calculate_game_progress(period, clock):
    """
    Calculate the fraction of the game completed and the total minutes remaining.

    This function calculates how much of the game has been completed and how many minutes are left,
    based on the current period and the remaining time in the period.

    Parameters:
    period (int): The current period of the game. Typically 1-4 for regulation and higher for overtime periods.
    clock (str): The remaining time in the current period, formatted as an ISO 8601 duration (e.g., "PT12M34.567S").

    Returns:
    tuple: A tuple containing:
           - fraction_of_game_completed (float): The fraction of the game that has been completed, ranging from 0 to 1.
           - minutes_remaining (float): The total number of minutes remaining in the game.
    """
    try:
        # Parse the clock string to extract minutes and seconds
        minutes, seconds = map(float, re.findall(r"PT(\d+)M(\d+\.\d+)S", clock)[0])
    except IndexError:
        # Default to 0 if parsing fails
        minutes, seconds = 0, 0

    # Calculate remaining time in the current period
    remaining_time_in_current_period = minutes + seconds / 60

    # Determine total expected game time in minutes
    total_expected_game_time = 48 if period <= 4 else 48 + (period - 4) * 5

    # Calculate elapsed time in the game
    if period <= 4:
        # Regulation time calculation
        total_elapsed_time = (period - 1) * 12 + (12 - remaining_time_in_current_period)
    else:
        # Overtime period calculation
        total_elapsed_time = (
            48 + (period - 5) * 5 + (5 - remaining_time_in_current_period)
        )

    # Calculate fraction of the game completed
    fraction_of_game_completed = total_elapsed_time / total_expected_game_time

    # Calculate total minutes remaining in the game
    minutes_remaining = total_expected_game_time - total_elapsed_time

    return fraction_of_game_completed, minutes_remaining


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

    # Pre-game scenario: use the base logistic parameters without adjustment
    if minutes_remaining is None:
        win_prob = float(1 / (1 + np.exp(-(base_a + base_b * score_diff))))
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


def update_predictions(games):
    """
    Update predictions based on the current state of the games.

    This method updates the predictions as the game progresses, using the current scores and game
    dynamics. It adjusts the pre-game predictions based on the real-time data, providing a more
    accurate forecast of the final game outcomes.

    Parameters:
    games (dict): The game data including current game state. Each game's data should include
                    both the pre-game predictions and the current state of the game.

    Returns:
    dict: A dictionary containing the updated predictions for each game, considering the current
            game state.
    """
    updated_predictions = {}
    for game_id, game_data in games.items():
        pre_game_pred = game_data.get("pre_game_predictions", {})
        current_state = game_data.get("current_game_state", {})

        if not current_state:
            updated_predictions[game_id] = pre_game_pred
            continue

        if current_state.get("is_final_state"):
            # Create the new dictionary item in updated_predictions
            updated_predictions[game_id] = {
                "pred_home_score": current_state["home_score"],
                "pred_away_score": current_state["away_score"],
                "pred_home_win_pct": (
                    1.0
                    if current_state["home_score"] > current_state["away_score"]
                    else 0.0
                ),
                "pred_players": current_state.get(
                    "players_data", {"home": {}, "away": {}}
                ),
            }

            # Modify the pred_players within the new dictionary item
            pred_players = updated_predictions[game_id]["pred_players"]
            for team in ["home", "away"]:
                for player_id, player_stats in pred_players[team].items():
                    if "points" in player_stats:
                        player_stats["pred_points"] = player_stats.pop("points")
            continue

        pre_home = pre_game_pred["pred_home_score"]
        pre_away = pre_game_pred["pred_away_score"]
        curr_home = current_state["home_score"]
        curr_away = current_state["away_score"]
        pre_players = pre_game_pred["pred_players"]
        curr_players = current_state["players_data"]

        # Use calculate_game_progress to get fraction completed and minutes remaining
        fraction_completed, minutes_remaining = calculate_game_progress(
            current_state["period"], current_state["clock"]
        )

        # Create a dictionary for scores
        scores = {
            "pregame_pred_home_score": pre_home,
            "pregame_pred_away_score": pre_away,
            "current_home_score": curr_home,
            "current_away_score": curr_away,
        }

        # Call the refactored calculate_updated_scores function
        updated_home, updated_away = calculate_updated_scores(
            scores=scores, fraction_of_game_remaining=(1 - fraction_completed)
        )

        updated_win_prob = calculate_home_win_prob(
            updated_home, updated_away, minutes_remaining=minutes_remaining
        )

        updated_pred_players = (
            pre_players  # Pass through for now; to be implemented in the future
        )

        updated_predictions[game_id] = {
            "pred_home_score": updated_home,
            "pred_away_score": updated_away,
            "pred_home_win_pct": updated_win_prob,
            "pred_players": updated_pred_players,
        }

    return updated_predictions
