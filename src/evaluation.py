import numpy as np
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


def create_evaluations(correct, predicted):
    """
    This function creates evaluations for different metrics based on the correct and predicted values.

    Parameters:
    correct (dict): A dictionary of correct values.
    predicted (dict): A dictionary of predicted values.

    Returns:
    dict: A dictionary of evaluations for different metrics.
    """
    # Initialize an empty dictionary to store the evaluations
    evaluations = {}

    # Evaluate the output completeness and update the evaluations dictionary
    evaluations.update(evaluate_output_completeness(correct, predicted))

    # If the correct values dictionary contains a home score, evaluate it
    if "home_score" in correct:
        evaluations.update(
            evaluate_regression(correct, predicted, "home_score", (0, 200))
        )

    # If the correct values dictionary contains an away score, evaluate it
    if "away_score" in correct:
        evaluations.update(
            evaluate_regression(correct, predicted, "away_score", (0, 200))
        )

    # Iterate over the suffixes for the home margin and total points metrics
    for x in ["", "_direct", "_derived"]:
        # If the correct values dictionary contains a home margin with the current suffix, evaluate it
        if f"home_margin{x}" in correct:
            evaluations.update(
                evaluate_regression(correct, predicted, f"home_margin{x}", (-100, 100))
            )

        # If the correct values dictionary contains total points with the current suffix, evaluate it
        if f"total_points{x}" in correct:
            evaluations.update(
                evaluate_regression(correct, predicted, f"total_points{x}", (0, 400))
            )

    # If the correct values dictionary contains a home win probability, evaluate it
    if "home_win_prob" in correct:
        evaluations.update(evaluate_home_win_probability(correct, predicted))

    # If the correct values dictionary contains players, skip it for now
    if "players" in correct:
        pass

    # Return the evaluations dictionary
    return evaluations


def evaluate_output_completeness(correct, predicted):
    """
    Evaluate the completeness of the predicted output.

    Parameters:
    correct (dict): The correct output.
    predicted (dict): The predicted output.

    Returns:
    dict: A dictionary with the completeness percentage, unnecessary keys percentage,
          missing key count, and unnecessary key count.
    """

    # Convert the keys of the correct and predicted outputs to sets
    correct_keys = set(correct.keys())
    predicted_keys = set(predicted.keys())

    # Calculate the missing and unnecessary keys
    missing_keys = correct_keys - predicted_keys
    unnecessary_keys = predicted_keys - correct_keys

    # Calculate the completeness and unnecessary keys percentages
    output_completeness_percentage = len(predicted_keys) / len(correct_keys) * 100
    unnecessary_keys_percentage = len(unnecessary_keys) / len(correct_keys) * 100

    # Calculate the counts of missing and unnecessary keys
    missing_key_count = len(missing_keys)
    unnecessary_key_count = len(unnecessary_keys)

    return {
        "output_completeness_percentage": output_completeness_percentage,
        "unnecessary_keys_percentage": unnecessary_keys_percentage,
        "missing_key_count": missing_key_count,
        "unnecessary_key_count": unnecessary_key_count,
    }


def evaluate_regression(correct, predicted, key, range):
    """
    Evaluate the regression model.

    Parameters:
    correct (dict): The correct output.
    predicted (dict): The predicted output.
    key (str): The key to evaluate.
    range (tuple): The valid range of values.

    Returns:
    dict: A dictionary with the availability of the key, whether the format of the value was correct,
          the percentage of logical values, and the regression metrics (MAE, R2, RMSE, MAPE, and Median AE).
    """

    # Convert the correct values to a numpy array of floats
    correct_values = np.array(correct[key]).astype(float)

    # Check if the key exists in the predicted output
    value_available = key in predicted
    if not value_available:
        return {f"{key}_availability": value_available}

    try:
        # Try to convert the predicted values to a numpy array of floats
        predicted_values = np.array(predicted[key]).astype(float)
        valid_format = True
    except ValueError:
        # If the conversion fails, return a dictionary indicating that the key is available but the format is incorrect
        return {
            f"{key}_availability": value_available,
            f"{key}_format": False,
        }

    # Calculate the percentage of predicted values that are within the valid range
    range_valid_percentage = (
        np.mean((predicted_values >= range[0]) & (predicted_values <= range[1])) * 100
    )

    # Calculate the regression metrics
    regression_metrics = calculate_regression_metrics(correct_values, predicted_values)
    mae = regression_metrics["mae"]
    r2 = regression_metrics["r2"]
    rmse = regression_metrics["rmse"]
    mape = regression_metrics["mape"]
    median_ae = regression_metrics["median_ae"]

    return {
        f"{key}_availability": value_available,
        f"{key}_format": valid_format,
        f"{key}_logical": range_valid_percentage,
        f"{key}_mae": mae,
        f"{key}_r2": r2,
        f"{key}_rmse": rmse,
        f"{key}_mape": mape,
        f"{key}_median_ae": median_ae,
    }


def evaluate_home_win_probability(correct, predicted):
    """
    Evaluate the home win probability.

    Parameters:
    correct (dict): The correct output.
    predicted (dict): The predicted output.

    Returns:
    dict: A dictionary with the availability of the home win probability, whether the format of the value was correct,
          the percentage of logical values, and the classification metrics (accuracy, precision, recall, F1 score,
          ROC AUC, log loss, Brier score, optimal threshold, and optimal threshold accuracy).
    """

    # Check if the home win probability is available in the predicted output
    home_win_prob_available = "home_win_prob" in predicted
    if not home_win_prob_available:
        return {"home_win_prob_available": home_win_prob_available}

    try:
        # Try to convert the home win probability to a numpy array of floats
        home_win_prob = np.array(predicted["home_win_prob"]).astype(float)
        valid_format = True
    except ValueError:
        # If the conversion fails, return a dictionary indicating that the home win probability is available but the format is incorrect
        return {
            "home_win_prob_available": home_win_prob_available,
            "home_win_prob_format": False,
        }

    # Calculate the percentage of home win probabilities that are within the valid range (0 to 1)
    win_probability_logical_percentage = (
        np.mean(np.logical_and(home_win_prob >= 0, home_win_prob <= 1)) * 100
    )

    # Convert the home win probabilities to binary values using the 0.5 threshold
    binary_predictions = (home_win_prob >= 0.5).astype(int)
    correct_home_win_prob = np.array(correct["home_win_prob"]).astype(int)

    # Calculate the classification metrics
    accuracy = np.mean(binary_predictions == correct_home_win_prob) * 100
    precision = precision_score(correct_home_win_prob, binary_predictions) * 100
    recall = recall_score(correct_home_win_prob, binary_predictions) * 100
    f1 = f1_score(correct_home_win_prob, binary_predictions) * 100
    roc_auc = roc_auc_score(correct_home_win_prob, home_win_prob)
    log_loss_ = log_loss(correct_home_win_prob, home_win_prob)
    brier_score = brier_score_loss(correct_home_win_prob, home_win_prob)

    # Find the optimal threshold that maximizes the accuracy
    def find_optimal_threshold(y_true, y_probs):
        thresholds = np.linspace(0, 1, 100)
        accuracies = [
            accuracy_score(y_true, y_probs > threshold) for threshold in thresholds
        ]
        optimal_index = np.argmax(accuracies)
        return thresholds[optimal_index], accuracies[optimal_index]

    optimal_threshold, optimal_threshold_accuracy = find_optimal_threshold(
        correct_home_win_prob, home_win_prob
    )

    return {
        "home_win_prob_available": home_win_prob_available,
        "home_win_prob_format": valid_format,
        "home_win_prob_logical": win_probability_logical_percentage,
        "home_win_prob_accuracy": accuracy,
        "home_win_prob_precision": precision,
        "home_win_prob_recall": recall,
        "home_win_prob_f1_score": f1,
        "home_win_prob_roc_auc": roc_auc,
        "home_win_prob_log_loss": log_loss_,
        "home_win_prob_brier_score": brier_score,
        "home_win_prob_optimal_threshold": optimal_threshold,
        "home_win_prob_optimal_threshold_accuracy": optimal_threshold_accuracy,
    }


def calculate_regression_metrics(correct, predicted):
    """
    This function evaluates the performance of a regression model.

    Args:
        correct (array-like): The correct target values.
        predicted (array-like): The predicted target values.

    Returns:
        dict: A dictionary containing the MAE, R2 score, RMSE, MAPE, and Median Absolute Error.
    """
    # Calculate the Mean Absolute Error (MAE)
    # This is the average of the absolute differences between the correct and predicted values
    mae = mean_absolute_error(correct, predicted)

    # Calculate the R2 score (coefficient of determination)
    # This is the proportion of the variance in the dependent variable that is predictable from the independent variable(s)
    r2 = r2_score(correct, predicted)

    # Calculate the Root Mean Squared Error (RMSE)
    # This is the square root of the average of the squared differences between the correct and predicted values
    rmse = np.sqrt(mean_squared_error(correct, predicted))

    # Calculate the Mean Absolute Percentage Error (MAPE)
    # This is the average of the absolute percentage differences between the correct and predicted values
    mape = np.mean(np.abs((correct - predicted) / correct)) * 100

    # Calculate the Median Absolute Error
    # This is the median of the absolute differences between the correct and predicted values
    median_ae = median_absolute_error(correct, predicted)

    # Return the calculated metrics in a dictionary
    return {
        "mae": mae,
        "r2": r2,
        "rmse": rmse,
        "mape": mape,
        "median_ae": median_ae,
    }


final_state_evaluations_to_implement = {
    "home_players_point_totals": {
        "description": "Predicted point totals for each home player",
        "examples": {"player_id_1": 20, "player_id_2": 18},
        "datatype": "dict",
        "format": "dictionary of player_id to integer points",
        "validation_checks": {
            "format_validation": "Ensure each point total is an integer and each player_id is an int or a string that converts to an int",
            "points_range": "Each player's points must be no less than 0 and no more than 100",
            "points_accuracy": "Evaluate the error, or difference, between the predicted points and the actual points for each player",
            "id_format": "Check if the player_id is an integer or a string representing an integer",
            "id_in_league": "Check if the player_id is part of the set of all player IDs in the league",
            "id_in_game": "Check if the player_id is part of the set of all player IDs in the game",
            "id_in_team": "Check if the player_id is in the set of all player IDs in the game for the home team",
        },
    },
    "away_players_point_totals": {
        "description": "Predicted point totals for each away player",
        "examples": {"player_id_3": 22, "player_id_4": 15},
        "datatype": "dict",
        "format": "dictionary of player_id to integer points",
        "validation_checks": {
            "format_validation": "Ensure each point total is an integer and each player_id is an int or a string that converts to an int",
            "points_range": "Each player's points must be no less than 0 and no more than 100",
            "points_accuracy": "Evaluate the error, or difference, between the predicted points and the actual points for each player",
            "id_format": "Check if the player_id is an integer or a string representing an integer",
            "id_in_league": "Check if the player_id is part of the set of all player IDs in the league",
            "id_in_game": "Check if the player_id is part of the set of all player IDs in the game",
            "id_in_team": "Check if the player_id is in the set of all player IDs in the game for the away team",
        },
    },
    "overall_player_identification": {
        "description": "Evaluation of player identification.",
        "evaluations": {
            "precision_evaluation": "The proportion of predicted player IDs that were correct (actual scorers).",
            "recall_evaluation": "The proportion of actual scoring players that were correctly identified by the model.",
            "F1_score_evaluation": "The harmonic mean of precision and recall, providing a single metric to assess the overall identification accuracy.",
        },
    },
    "overall_player_score_prediction": {
        "description": "Evaluation of player score predictions.",
        "evaluations": {
            "inner_join_evaluation": "Match actual scorers with predicted scorers, and calculate the metric for the matching pairs.",
            "left_join_evaluation": "Keep all actual scorers and match to predicted scorers where available. If a predicted scorer is missing, fill with 0 for the prediction. Calculate the metric for resulting pairs.",
            "right_join_evaluation": "Keep all predicted scorers and match to actual scorers where available. If an actual scorer is missing, fill with 0 for the actual score. Calculate the metric for resulting pairs.",
            "outer_join_evaluation": "Keep all actual scorers and predicted scorers and match if available. If a predicted scorer is missing, fill with 0 for the prediction. If an actual scorer is missing, fill with 0 for the actual score. Calculate the metric for resulting pairs.",
        },
    },
}
