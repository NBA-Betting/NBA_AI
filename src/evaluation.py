final_state_evaluations = {
    "winning_team": {
        "description": "Identifier of the winning team, using a tricode of three capital letters.",
        "examples": ["TOR", "GSW", "LAL", "ATL"],
        "datatype": "string",
        "format": "3 capital letter string",
        "validation_checks": {
            "format_validation": "Ensure the output is a 3 capital letter string",
            "valid_team_tricode": "Check if the tricode matches one of the 30 NBA team tricodes",
            "valid_team_tricode_for_game": "Check if the tricode matches one of the 2 teams playing in the specific game",
            "classification_accuracy": "Accuracy of the prediction in choosing the winning team out of the two",
        },
    },
    "winning_team_probability": {
        "description": "Probability of the winning team to win, expressed as a float between 0 and 1.",
        "examples": [0.75, 0.65, 0.85],
        "datatype": "float",
        "format": "float between 0 and 1",
        "validation_checks": {
            "format_validation": "Ensure the output is a float within the valid range of 0 to 1",
            "logical_check": "The probability should not be below 0.5, as it indicates the team with a higher chance of winning",
            "probability_accuracy": "Assess how closely the probability matches the outcome",
        },
    },
    "home_score": {
        "description": "Predicted score for the home team",
        "examples": [102, 110, 95],
        "datatype": "int",
        "format": "integer",
        "validation_checks": {
            "format_validation": "Ensure the output is an integer",
            "score_range": "The score must be no less than 0 and no more than 200",
            "score_accuracy": "Evaluate the error, or difference, between the predicted score and the actual score",
        },
    },
    "away_score": {
        "description": "Predicted score for the away team",
        "examples": [97, 105, 100],
        "datatype": "int",
        "format": "integer",
        "validation_checks": {
            "format_validation": "Ensure the output is an integer",
            "score_range": "The score must be no less than 0 and no more than 200",
            "score_accuracy": "Evaluate the error, or difference, between the predicted score and the actual score",
        },
    },
    "home_margin_direct": {
        "description": "Directly predicted difference in scores from the home team perspective.",
        "examples": [5, -10, 15],
        "datatype": "int",
        "format": "integer",
        "validation_checks": {
            "format_validation": "Ensure the output is an integer",
            "logical_check": "The margin must be no less than -100 and no more than 100",
            "margin_accuracy": "Evaluate the error, or difference, between the predicted margin and the actual margin",
        },
    },
    "home_margin_derived": {
        "description": "Difference in scores derived from combining home and away score predictions.",
        "examples": [5, -10, 15],
        "datatype": "int",
        "format": "integer",
        "validation_checks": {
            "format_validation": "Ensure the derived output is an integer",
            "logical_check": "The derived margin must be no less than -100 and no more than 100",
            "margin_accuracy": "Evaluate the error, or difference, between the derived margin and the actual margin",
        },
    },
    "total_points_direct": {
        "description": "Directly predicted total points scored in the game.",
        "examples": [204, 215, 190],
        "datatype": "int",
        "format": "integer",
        "validation_checks": {
            "format_validation": "Ensure the output is an integer",
            "score_range": "The total points must be no less than 0 and no more than 400",
            "total_points_accuracy": "Evaluate the error, or difference, between the predicted total points and the actual total points",
        },
    },
    "total_points_derived": {
        "description": "Total points derived from combining home and away score predictions.",
        "examples": [204, 215, 190],
        "datatype": "int",
        "format": "integer",
        "validation_checks": {
            "format_validation": "Ensure the derived output is an integer",
            "score_range": "The derived total points must be no less than 0 and no more than 400",
            "total_points_accuracy": "Evaluate the error, or difference, between the derived total points and the actual total points",
        },
    },
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
    "overall_model_output_completeness": {
        "description": "Check if the model output is complete for all required fields and if extra fields are returned.",
        "evaluations": {
            "completeness_evaluation": "Percentage of fields that are present in the model output compared to the required fields.",
            "unnecessary_fields_evaluation": "Count of fields that are present in the model output but are not required.",
        },
    },
}

game_state_evaluations = {}
