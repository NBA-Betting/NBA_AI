"""
gpt4o_data_prep.py

Warning:
- Using the OpenAI API will incur costs. Make sure to set usage limits and monitor usage to avoid unexpected charges.

This module handles the preparation of data for training and testing the GPT-4o model.
It consists of functions to:
- Load game data from the database.
- Fetch game and final state data.
- Process prior games.
- Generate prompts for the model.
- Count tokens in the game records.
- Create samples of game records.

Functions:
- load_game_data(season, historical_game_count, db_path=DB_PATH): Loads game data for the specified season.
- fetch_game_data(season, db_path): Fetches game data from the database.
- fetch_final_state(game_id, db_path): Fetches the final state of a game from the database.
- process_prior_games(prior_states, max_games): Processes prior games for a given set of states.
- generate_prompt(home_team, away_team, date_time_est, season_type, home_prior_games, away_prior_games): Generates a prompt for the model.
- process_game(game, prior_states_dict, db_path, historical_game_count): Processes a single game record.
- count_tokens(game_records): Counts the tokens in the game records.
- create_sample(game_records, regular_season_sample_size=None, postseason_sample_size=None, random_seed=None): Creates a random sample of game records.

Usage:
- Typically run as part of a larger data processing pipeline.
- Script can be run directly from the command line (project root) to prepare data for the GPT-4o model.
    Example: python -m src.model_training.gpt4o_data_prep
- Successful execution will load game data, count tokens, and save the data to JSON files.
"""

import json
import logging
import random
import sqlite3

import tiktoken

from src.config import config
from src.database_updater.prior_states import (
    determine_prior_states_needed,
    load_prior_states,
)
from src.logging_config import setup_logging
from src.utils import log_execution_time, validate_season_format

# Configuration
DB_PATH = config["database"]["path"]
PROJECT_ROOT = config["project"]["root"]

# Flag to include or exclude player-related data
INCLUDE_PLAYER_DATA = False  # Set to True to include player data, False to exclude


@log_execution_time(average_over="output")
def load_game_data(season, historical_game_count, db_path=DB_PATH):
    validate_season_format(season)
    logging.info(f"Loading data for season: {season}")

    games = fetch_game_data(season, db_path)
    if not games:
        logging.warning(f"No games found for season: {season}")
        return []

    game_ids = [game[0] for game in games]
    prior_states_dict = load_prior_states(determine_prior_states_needed(game_ids))

    game_records = []
    for game in games:
        record = process_game(game, prior_states_dict, db_path, historical_game_count)
        game_records.append(record)

    logging.info(f"Loaded data for {len(game_records)} games.")
    return game_records


def fetch_game_data(season, db_path):
    query_games = """
    SELECT game_id, date_time_est, home_team, away_team, season, season_type
    FROM Games
    WHERE season = ?
    AND season_type IN ('Regular Season', 'Post Season')
    AND status = 'Completed'
    """
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(query_games, (season,))
        return cursor.fetchall()


def fetch_final_state(game_id, db_path):
    query_final_state = """
    SELECT home_score, away_score, players_data
    FROM GameStates
    WHERE game_id = ?
    AND is_final_state = 1
    """
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(query_final_state, (game_id,))
        result = cursor.fetchone()
        if result:
            home_score, away_score, players_data = result
            response = {
                "home_team": {"score": home_score},
                "away_team": {"score": away_score},
            }
            if INCLUDE_PLAYER_DATA:
                response["home_team"]["player_points"] = {
                    player_id: details["points"]
                    for player_id, details in json.loads(players_data)["home"].items()
                }
                response["away_team"]["player_points"] = {
                    player_id: details["points"]
                    for player_id, details in json.loads(players_data)["away"].items()
                }
            return response
        return None


def process_prior_games(prior_states, max_games):
    sorted_games = []

    for state in prior_states:
        game_record = {
            "home_team": state["home"],
            "away_team": state["away"],
            "game_date": state["game_date"],
            "home_team_score": state["home_score"],
            "away_team_score": state["away_score"],
        }

        if INCLUDE_PLAYER_DATA:
            game_record["home_team_player_points"] = {
                player_id: details["points"]
                for player_id, details in state["players_data"]["home"].items()
            }
            game_record["away_team_player_points"] = {
                player_id: details["points"]
                for player_id, details in state["players_data"]["away"].items()
            }

        sorted_games.append(game_record)

    sorted_games.sort(key=lambda x: x["game_date"], reverse=True)

    if max_games == "all":
        return sorted_games
    else:
        return sorted_games[:max_games]


def generate_prompt(
    home_team, away_team, date_time_est, season_type, home_prior_games, away_prior_games
):
    prompt = (
        f"Home Team: {home_team}\n"
        f"Away Team: {away_team}\n"
        f"Game Date: {date_time_est[:10]}\n"
        f"Season Type: {season_type}\n"
        f"Home Teams Most Recent Previous Games: \n{home_prior_games}\n"
        f"Away Teams Most Recent Previous Games: \n{away_prior_games}"
    )
    return prompt


def process_game(game, prior_states_dict, db_path, historical_game_count):
    game_id, date_time_est, home_team, away_team, season, season_type = game

    final_state = fetch_final_state(game_id, db_path)
    if final_state is None:
        logging.error(f"Final state not found for game_id: {game_id}")
        return None

    home_prior_games = process_prior_games(
        prior_states_dict.get(game_id, {}).get("home_prior_states", []),
        max_games=historical_game_count,
    )
    away_prior_games = process_prior_games(
        prior_states_dict.get(game_id, {}).get("away_prior_states", []),
        max_games=historical_game_count,
    )

    prompt = generate_prompt(
        home_team,
        away_team,
        date_time_est,
        season_type,
        home_prior_games,
        away_prior_games,
    )

    return {
        "game_id": game_id,
        "home_team": home_team,
        "away_team": away_team,
        "game_date": date_time_est[:10],
        "season": season,
        "season_type": season_type,
        "prompt": prompt,
        "response": final_state,
    }


def count_tokens(game_records):
    # Pricing

    # gpt-4o-2024-08-06
    gpt_4o_input_cost_per_million = 2.50
    gpt_4o_output_cost_per_million = 10.00

    # gpt-4o-mini
    gpt_4o_mini_input_cost_per_million = 0.15
    gpt_4o_mini_output_cost_per_million = 0.60

    tokenizer = tiktoken.get_encoding("cl100k_base")

    game_count = len(game_records)
    total_prompt_tokens = 0
    total_response_tokens = 0

    for game in game_records:
        prompt_tokens = tokenizer.encode(json.dumps(game["prompt"]))
        total_prompt_tokens += len(prompt_tokens)

        response_tokens = tokenizer.encode(json.dumps(game.get("response", {})))
        total_response_tokens += len(response_tokens)

    average_prompt_tokens = total_prompt_tokens / game_count if game_count > 0 else 0
    average_response_tokens = (
        total_response_tokens / game_count if game_count > 0 else 0
    )

    logging.info(
        f"Token Counts:\nPrompt: {total_prompt_tokens} total. Average: {average_prompt_tokens:.2f} per game."
    )
    logging.info(
        f"Response: {total_response_tokens} total. Average: {average_response_tokens:.2f} per game."
    )

    # Calculate costs per 100 records
    prompt_tokens_per_100 = total_prompt_tokens / game_count * 100
    response_tokens_per_100 = total_response_tokens / game_count * 100

    gpt_4o_cost_per_100 = (
        prompt_tokens_per_100 / 1000000
    ) * gpt_4o_input_cost_per_million + (
        response_tokens_per_100 / 1000000
    ) * gpt_4o_output_cost_per_million

    gpt_4o_mini_cost_per_100 = (
        prompt_tokens_per_100 / 1000000
    ) * gpt_4o_mini_input_cost_per_million + (
        response_tokens_per_100 / 1000000
    ) * gpt_4o_mini_output_cost_per_million

    logging.info(
        f"Pricing per 100 records:\n"
        f"gpt-4o-2024-08-06: ${gpt_4o_cost_per_100:.2f}\n"
        f"gpt-4o-mini: ${gpt_4o_mini_cost_per_100:.2f}"
    )


def create_sample(
    game_records,
    regular_season_sample_size=None,
    postseason_sample_size=None,
    random_seed=None,
):
    """
    Creates a random sample of game records with optionality for separate Regular Season and Post Season sampling,
    and stratification by season quarters for Regular Season games.

    Parameters:
    - game_records (list of dicts): List of game records to sample from.
    - regular_season_sample_size (int, optional): Number of Regular Season records to include in the sample. Default is None.
    - postseason_sample_size (int, optional): Number of Post Season records to include in the sample. Default is None.
    - random_seed (int, optional): Seed for random number generator for reproducibility. Default is None.

    Returns:
    - list of dicts: A random sample of game records based on specified sample sizes and season quarters.
    """

    # Set the random seed for reproducibility (optional)
    if random_seed is not None:
        random.seed(random_seed)

    # Separate the records into Regular Season and Post Season
    regular_season_records = [
        record for record in game_records if record["season_type"] == "Regular Season"
    ]
    postseason_records = [
        record for record in game_records if record["season_type"] == "Post Season"
    ]

    # Sort the regular season records by date
    regular_season_records.sort(key=lambda x: x["game_date"])

    # Determine the number of games in each quarter
    total_regular_season_games = len(regular_season_records)
    quarter_size = total_regular_season_games // 4

    # Initialize quarters
    quarters = {
        "Q1": regular_season_records[:quarter_size],
        "Q2": regular_season_records[quarter_size : 2 * quarter_size],
        "Q3": regular_season_records[2 * quarter_size : 3 * quarter_size],
        "Q4": regular_season_records[3 * quarter_size :],
    }

    # Adjust if there are any leftover games due to integer division
    remaining_games = total_regular_season_games - 4 * quarter_size
    if remaining_games > 0:
        quarters["Q4"].extend(regular_season_records[-remaining_games:])

    # Initialize the sample list
    sample = []

    # Sample an equal number of games from each quarter if requested
    if regular_season_sample_size is not None:
        quarter_sample_size = regular_season_sample_size // 4

        for quarter in quarters.values():
            if quarter_sample_size > len(quarter):
                raise ValueError(
                    "Quarter sample size cannot be larger than the number of available games in that quarter."
                )
            sample += random.sample(quarter, quarter_sample_size)

    # Sample Post Season games if requested
    if postseason_sample_size is not None:
        if postseason_sample_size > len(postseason_records):
            raise ValueError(
                "Post season sample size cannot be larger than the number of available post season game records."
            )
        sample += random.sample(postseason_records, postseason_sample_size)

    return sample


if __name__ == "__main__":
    # Setup logging
    setup_logging(log_level="INFO")

    # PARAMS
    season = "2022-2023"
    historical_game_count = 10

    # Load and process the game data
    game_records = load_game_data(season, historical_game_count)

    # Count tokens
    count_tokens(game_records)

    # Save the game data to a JSON Lines file
    if not INCLUDE_PLAYER_DATA:
        full_filepath = f"{PROJECT_ROOT}/data/{season}_{historical_game_count}H_no_player_data.jsonl"
    else:
        full_filepath = f"{PROJECT_ROOT}/data/{season}_{historical_game_count}H.jsonl"

    with open(full_filepath, "w") as f:
        for game in game_records:
            f.write(json.dumps(game) + "\n")

    logging.info(f"Game data saved to: {full_filepath}")

    # Sample the data
    regular_season_sample_size = 100
    postseason_sample_size = 0
    sample = create_sample(
        game_records,
        regular_season_sample_size=regular_season_sample_size,
        postseason_sample_size=postseason_sample_size,
        random_seed=42,
    )

    # Save the sample to a JSON Lines file
    if not INCLUDE_PLAYER_DATA:
        sample_filepath = f"{PROJECT_ROOT}/data/sample_{season}_{regular_season_sample_size}R_{postseason_sample_size}P_{historical_game_count}H_no_player_data.jsonl"
    else:
        sample_filepath = f"{PROJECT_ROOT}/data/sample_{season}_{regular_season_sample_size}R_{postseason_sample_size}P_{historical_game_count}H.jsonl"

    with open(sample_filepath, "w") as f:
        for game in sample:
            f.write(json.dumps(game) + "\n")

    logging.info(f"Sample data saved to: {sample_filepath}")
