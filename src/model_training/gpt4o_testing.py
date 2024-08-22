import json
import random
from typing import List

import numpy as np
from openai import OpenAI
from pydantic import BaseModel

from src.config import config

# Configuration
PROJECT_ROOT = config["project"]["root"]

client = OpenAI()


def load_prompt_response_data(filename, num_records=10, randomize=True):
    file_path = f"{PROJECT_ROOT}/data/{filename}.jsonl"

    # First pass: count the total number of lines
    with open(file_path, "r") as file:
        total_lines = sum(1 for _ in file)

    # If random sampling is required
    if randomize:
        selected_lines = sorted(random.sample(range(total_lines), num_records))
    else:
        selected_lines = list(range(num_records))

    records = []
    with open(file_path, "r") as file:
        for i, line in enumerate(file):
            if i in selected_lines:
                record = json.loads(line.strip())
                records.append(record)
            if len(records) == num_records:
                break

    return records


# Define the Pydantic models
class PlayerPoints(BaseModel):
    player_id: str
    points: int


class TeamOutcome(BaseModel):
    score: int
    player_points: List[PlayerPoints]


class GameOutcome(BaseModel):
    home_team: TeamOutcome
    away_team: TeamOutcome


def prepare_prompt(record):
    # Extract the first key and its associated data from the record
    game_id, game_data = next(iter(record.items()))

    # Extract game metadata and determine home and away teams
    game_metadata = game_data["prompt"]["game_metadata"]
    away_team, home_team = game_metadata.split(",")[0].split("@")

    # Prepare the system message to provide context for the generic keys
    system_message = f"""You are an expert NBA game outcome predictor. Please predict the outcomes for this game. The home team is {home_team} and the away team is {away_team}. I have provided game metadata, logs for the game if it is in progress, and final states of previous games for both teams."""

    # Prepare the user message
    user_message = json.dumps(game_data["prompt"])

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
        top_logprobs=5,
        n=1,
    )

    response_dict = response.to_dict()

    return response_dict


def compare_predictions_to_actual(record, response_dict):
    # Extract game_id, actual response, and predicted response
    game_id, game_data = next(iter(record.items()))
    actual_response = game_data["response"]
    predicted_response = response_dict["choices"][0]["message"]["parsed"]

    # Extract the home and away team abbreviations from the actual response
    home_team = list(actual_response.keys())[0]
    away_team = list(actual_response.keys())[1]

    # Extract the scores and player points from the actual response
    actual_home_score = actual_response[home_team]["score"]
    actual_away_score = actual_response[away_team]["score"]
    actual_home_player_points = actual_response[home_team]["player_points"]
    actual_away_player_points = actual_response[away_team]["player_points"]

    # Extract the scores and player points from the predicted response
    pred_home_score = predicted_response["home_team"]["score"]
    pred_away_score = predicted_response["away_team"]["score"]
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
        "player_points": {
            home_team: actual_home_player_points,
            away_team: actual_away_player_points,
        },
        "pred_player_points": {
            home_team: pred_home_player_points,
            away_team: pred_away_player_points,
        },
    }

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
                            - comparison["pred_player_points"][comparison[team]][pid]
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
        "average_player_inclusion_rate": np.mean(inclusion_rates),
        "average_extra_player_rate": np.mean(extra_player_rates),
        "average_player_mae": np.mean(player_mae_list),
    }

    return eval_metrics


if __name__ == "__main__":
    # Load the prompt response data
    records = load_prompt_response_data(
        "prompt_response_data_2023-2024", num_records=20, randomize=True
    )

    all_comparisons = {}

    for record in records:
        # Prepare the prompt
        messages = prepare_prompt(record)

        # Make the GPT-4o request
        response = make_gpt4o_request(messages)

        # Compare predictions to actual results
        comparison = compare_predictions_to_actual(record, response)

        # Add the comparison to the dictionary of all comparisons
        all_comparisons.update(comparison)

    # Evaluate the predictions
    eval_metrics = evaluate_predictions(all_comparisons)
    print("Evaluation Metrics:")
    print(json.dumps(eval_metrics, indent=4))

    # Save all comparisons
    with open("all_comparisons.json", "w") as file:
        json.dump(all_comparisons, file, indent=4)

    print("Comparisons saved to all_comparisons.json")
