import json
import math
from typing import List

import numpy as np
from openai import OpenAI
from pydantic import BaseModel

from src.config import config

# Configuration
PROJECT_ROOT = config["project"]["root"]

client = OpenAI()

# Flag to include or exclude player-related data
INCLUDE_PLAYER_DATA = False  # Set to True to include player data, False to exclude
INCLUDE_EXPLANATION = False  # Set to True to include explanation, False to exclude

# Define the Pydantic models
if INCLUDE_PLAYER_DATA:

    class PlayerPoints(BaseModel):
        player_id: str
        points: int

    class TeamOutcome(BaseModel):
        score: int
        player_points: List[PlayerPoints]

    if INCLUDE_EXPLANATION:

        class GameOutcome(BaseModel):
            explanation: str
            home_team: TeamOutcome
            away_team: TeamOutcome

    else:

        class GameOutcome(BaseModel):
            home_team: TeamOutcome
            away_team: TeamOutcome

else:

    class TeamOutcome(BaseModel):
        score: int

    if INCLUDE_EXPLANATION:

        class GameOutcome(BaseModel):
            explanation: str
            home_team: TeamOutcome
            away_team: TeamOutcome

    else:

        class GameOutcome(BaseModel):
            home_team: TeamOutcome
            away_team: TeamOutcome


def prepare_prompt(record):
    # Prepare the system message to provide context for the generic keys
    system_message = (
        "You are an expert NBA game outcome predictor. "
        "Please predict the final outcomes for this game including the home and away team scores. "
    )
    if INCLUDE_PLAYER_DATA:
        system_message += "Please include the player ids and points for each player on the home and away teams. "
    system_message += (
        "The data below will help you make your prediction. It includes the home and away team names, "
        "the game date, and the results of the previous games for each team. "
    )
    if INCLUDE_EXPLANATION:
        system_message += "Please include your reasoning for your final score predictions before returning the results."

    # Prepare the user message
    user_message = json.dumps(record["prompt"])

    # Create the messages list
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]

    # Return the messages list
    return messages


def make_gpt4o_request(messages):
    response = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=messages,
        response_format=GameOutcome,
        logprobs=True,
        top_logprobs=20,
        n=1,
    )

    response_dict = response.to_dict()

    return response_dict


def compare_predictions_to_actual(record, response_dict):
    # Extract game_id, actual response, and predicted response
    game_id = record["game_id"]
    home_team = record["home_team"]
    away_team = record["away_team"]
    actual_response = record["response"]
    predicted_response = response_dict["choices"][0]["message"]["parsed"]

    # Extract the scores and player points from the actual response
    actual_home_score = actual_response["home_team"]["score"]
    actual_away_score = actual_response["away_team"]["score"]
    if INCLUDE_PLAYER_DATA:
        actual_home_player_points = actual_response["home_team"]["player_points"]
        actual_away_player_points = actual_response["away_team"]["player_points"]

    # Extract the scores and player points from the predicted response
    if INCLUDE_EXPLANATION:
        explained_outcome = predicted_response["explanation"]
    pred_home_score = predicted_response["home_team"]["score"]
    pred_away_score = predicted_response["away_team"]["score"]
    if INCLUDE_PLAYER_DATA:
        pred_home_player_points = {
            pp["player_id"]: pp["points"]
            for pp in predicted_response["home_team"]["player_points"]
        }
        pred_away_player_points = {
            pp["player_id"]: pp["points"]
            for pp in predicted_response["away_team"]["player_points"]
        }

    # Create a dictionary with the comparison data
    game_comparison = {
        "home_team": home_team,
        "away_team": away_team,
        "home_score": actual_home_score,
        "away_score": actual_away_score,
        "pred_home_score": pred_home_score,
        "pred_away_score": pred_away_score,
        "home_margin": actual_home_score - actual_away_score,
        "pred_home_margin": pred_home_score - pred_away_score,
        "total": actual_home_score + actual_away_score,
        "pred_total": pred_home_score + pred_away_score,
    }

    if INCLUDE_PLAYER_DATA:
        game_comparison["player_points"] = {
            home_team: actual_home_player_points,
            away_team: actual_away_player_points,
        }
        game_comparison["pred_player_points"] = {
            home_team: pred_home_player_points,
            away_team: pred_away_player_points,
        }

    if INCLUDE_EXPLANATION:
        game_comparison["explained_outcome"] = explained_outcome

    # Return the comparison dictionary with the game_id as the key
    return {game_id: game_comparison}


def evaluate_predictions(game_comparisons):
    # Initialize accumulators for MAE calculations
    total_abs_error_home_score = 0
    total_abs_error_away_score = 0
    total_abs_error_home_margin = 0
    total_abs_error_total = 0

    inclusion_rates = []
    extra_player_rates = []
    player_mae_list = []

    num_games = len(game_comparisons)

    for game_id, comparison in game_comparisons.items():
        # Calculate absolute errors for scores and margins
        abs_error_home_score = abs(
            comparison["home_score"] - comparison["pred_home_score"]
        )
        abs_error_away_score = abs(
            comparison["away_score"] - comparison["pred_away_score"]
        )
        abs_error_home_margin = abs(
            comparison["home_margin"] - comparison["pred_home_margin"]
        )
        abs_error_total = abs(comparison["total"] - comparison["pred_total"])

        # Accumulate for MAE calculation
        total_abs_error_home_score += abs_error_home_score
        total_abs_error_away_score += abs_error_away_score
        total_abs_error_home_margin += abs_error_home_margin
        total_abs_error_total += abs_error_total

        if INCLUDE_PLAYER_DATA:
            # Calculate inclusion/exclusion metrics and MAE for player points
            for team in ["home_team", "away_team"]:
                actual_player_ids = set(
                    comparison["player_points"][comparison[team]].keys()
                )
                pred_player_ids = set(
                    comparison["pred_player_points"][comparison[team]].keys()
                )

                # Inclusion Rate: % of actual player IDs that are in predicted IDs
                if actual_player_ids:
                    inclusion_rate = len(actual_player_ids & pred_player_ids) / len(
                        actual_player_ids
                    )
                else:
                    inclusion_rate = 0
                inclusion_rates.append(inclusion_rate)

                # Extra Player Rate: % of predicted player IDs not in actual IDs
                if pred_player_ids:
                    extra_player_rate = len(pred_player_ids - actual_player_ids) / len(
                        pred_player_ids
                    )
                else:
                    extra_player_rate = 0
                extra_player_rates.append(extra_player_rate)

                # Calculate MAE for matching player points
                matching_player_ids = actual_player_ids & pred_player_ids
                if matching_player_ids:
                    player_mae = np.mean(
                        [
                            abs(
                                comparison["player_points"][comparison[team]][pid]
                                - comparison["pred_player_points"][comparison[team]][
                                    pid
                                ]
                            )
                            for pid in matching_player_ids
                        ]
                    )
                else:
                    player_mae = 0
                player_mae_list.append(player_mae)

    # Calculate average MAEs and inclusion/exclusion metrics
    eval_metrics = {
        "mae_home_score": total_abs_error_home_score / num_games,
        "mae_away_score": total_abs_error_away_score / num_games,
        "mae_home_margin": total_abs_error_home_margin / num_games,
        "mae_total": total_abs_error_total / num_games,
    }

    if INCLUDE_PLAYER_DATA:
        eval_metrics.update(
            {
                "average_player_inclusion_rate": np.mean(inclusion_rates),
                "average_extra_player_rate": np.mean(extra_player_rates),
                "average_player_mae": np.mean(player_mae_list),
            }
        )

    return eval_metrics


def extract_score_probs(response, record):
    """
    Extracts predicted home and away scores along with their associated probabilities
    from the response dictionary.

    Parameters:
    response (dict): The response dictionary containing log probabilities for predicted scores.
    record (dict): The record dictionary containing the actual game outcome.

    Returns:
    dict: A dictionary with two keys, 'predicted_home_score_details' and 'predicted_away_score_details',
          each containing a list of tuples. Each tuple represents a score and its corresponding probability.
    """

    # Get actual scores from the record
    game_id = record["game_id"]
    home_team = record["home_team"]
    away_team = record["away_team"]
    actual_response = record["response"]

    # Extract the scores and player points from the actual response
    actual_home_score = actual_response["home_team"]["score"]
    actual_away_score = actual_response["away_team"]["score"]

    # Navigate to the list of token dictionaries in the response
    tokens = response.get("choices", [])[0].get("logprobs", {}).get("content", [])

    home_score_details = None
    away_score_details = None

    # Flags to track whether we're parsing home or away scores
    home_flag = False
    away_flag = False

    # Iterate through the token dictionaries in the response
    for i, token_dict in enumerate(tokens):
        token = token_dict["token"]

        # Check if the token is "home" and set the flag to parse the next score
        if token == "home":
            home_flag = True
        # Check if the token is "score" after "home" and extract the score details
        elif token == "score" and home_flag:
            # Continue searching for the next valid score token
            for j in range(i + 1, len(tokens)):
                next_token_dict = tokens[j]
                next_token = next_token_dict["token"]
                # Verify that the next token is a 2 or 3-digit integer and contains log probabilities
                if (
                    next_token.isdigit()
                    and len(next_token) in [2, 3]
                    and "top_logprobs" in next_token_dict
                ):
                    # Convert log probabilities to normal probabilities and store them with the scores
                    home_score_details = [
                        (item["token"], math.exp(item["logprob"]))
                        for item in next_token_dict["top_logprobs"]
                    ]
                    break
            # Reset the home_flag after processing the score
            home_flag = False

        # Check if the token is "away" and set the flag to parse the next score
        if token == "away":
            away_flag = True
        # Check if the token is "score" after "away" and extract the score details
        elif token == "score" and away_flag:
            # Continue searching for the next valid score token
            for j in range(i + 1, len(tokens)):
                next_token_dict = tokens[j]
                next_token = next_token_dict["token"]
                # Verify that the next token is a 2 or 3-digit integer and contains log probabilities
                if (
                    next_token.isdigit()
                    and len(next_token) in [2, 3]
                    and "top_logprobs" in next_token_dict
                ):
                    # Convert log probabilities to normal probabilities and store them with the scores
                    away_score_details = [
                        (item["token"], math.exp(item["logprob"]))
                        for item in next_token_dict["top_logprobs"]
                    ]
                    break
            # Reset the away_flag after processing the score
            away_flag = False

        # Break the loop early if both home and away scores are found
        if home_score_details and away_score_details:
            break

    # Return the extracted home and away score details
    return {
        "game_id": game_id,
        "actual_home_score": actual_home_score,
        "actual_away_score": actual_away_score,
        "predicted_home_scores": home_score_details,
        "predicted_away_scores": away_score_details,
    }


if __name__ == "__main__":
    # Set the dataset to use for testing
    dataset = "2022-2023_10H_no_player_data"

    # Set the prompt version
    prompt_version = "base_mini"

    # Load the prompt response data
    records = []
    with open(f"{PROJECT_ROOT}/data/{dataset}.jsonl", "r") as file:
        for line in file:
            records.append(json.loads(line))

    all_comparisons = {}
    alternate_scores = []

    for record in records:
        # Prepare the prompt
        messages = prepare_prompt(record)

        # Make the GPT-4o request
        response = make_gpt4o_request(messages)

        # Compare predictions to actual results
        comparison = compare_predictions_to_actual(record, response)
        all_comparisons.update(comparison)

        # Extract the predicted scores and probabilities
        alternate_scores_record = extract_score_probs(response, record)
        alternate_scores.append(alternate_scores_record)

    # Evaluate the predictions
    eval_metrics = evaluate_predictions(all_comparisons)

    # Print the evaluation metrics
    print(f"Evaluation Metrics for {dataset} {prompt_version}:")
    print(json.dumps(eval_metrics, indent=4))

    # Read the existing evaluation metrics from the file
    try:
        with open("eval_metrics.json", "r") as file:
            all_eval_metrics = json.load(file)
    except FileNotFoundError:
        all_eval_metrics = {}

    # Update the evaluation metrics for the current dataset
    all_eval_metrics[f"{dataset}_{prompt_version}"] = eval_metrics

    # Write the updated evaluation metrics back to the file
    with open("eval_metrics.json", "w") as file:
        json.dump(all_eval_metrics, file, indent=4)

    print(
        f"Evaluation Metrics for {dataset} {prompt_version} saved to 'eval_metrics.json'"
    )

    # Save all comparisons
    with open(f"pred_vs_actual_{dataset}_{prompt_version}.json", "w") as file:
        json.dump(all_comparisons, file, indent=4)

    print(
        f"Predictions vs. Actual Results saved to 'pred_vs_actual_{dataset}_{prompt_version}.json'"
    )

    # Save the alternate scores
    with open(f"score_probs_{dataset}_{prompt_version}.json", "w") as file:
        json.dump(alternate_scores, file, indent=4)

    print(
        f"Alternate Scores and Probabilities saved to 'score_probs_{dataset}_{prompt_version}.json'"
    )
