import json
import logging
import os
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


@log_execution_time(average_over="output")
def load_game_data(season, db_path=DB_PATH):
    validate_season_format(season)

    logging.info(f"Loading data for season: {season}")

    # Query strings
    query_games = """
    SELECT game_id, date_time_est, home_team, away_team, season, season_type
    FROM Games
    WHERE season = ?
    AND season_type IN ('Regular Season', 'Post Season')
    AND status = 'Completed'
    """

    query_states = """
    SELECT play_id, clock, period, home_score, away_score, players_data, is_final_state
    FROM GameStates
    WHERE game_id = ?
    ORDER BY play_id
    """

    result = {}
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(query_games, (season,))
        games = cursor.fetchall()

        game_ids = [game[0] for game in games]
        prior_states_needed = determine_prior_states_needed(game_ids)
        prior_states_dict = load_prior_states(prior_states_needed)

        for game in games:
            game_id, date_time_est, home_team, away_team, season, season_type = game

            # Get current game states
            cursor.execute(query_states, (game_id,))
            game_states = cursor.fetchall()

            # Extract only the home_prior_states and away_prior_states keys
            prior_states = prior_states_dict.get(game_id, {})
            filtered_prior_states = {
                "home_prior_states": prior_states.get("home_prior_states", []),
                "away_prior_states": prior_states.get("away_prior_states", []),
            }

            # Search for the final state by checking from the end of the list
            final_state = None
            for state in reversed(game_states):
                if state[6]:  # Check if is_final_state is True (or 1)
                    final_state = {
                        "play_id": state[0],
                        "clock": state[1],
                        "period": state[2],
                        "home_score": state[3],
                        "away_score": state[4],
                        "players_data": json.loads(state[5]),
                    }
                    break  # Stop searching once the final state is found

            result[game_id] = {
                "prompt": {
                    "date_time_est": date_time_est,
                    "home_team": home_team,
                    "away_team": away_team,
                    "season": season,
                    "season_type": season_type,
                    "game_states": [
                        {
                            "play_id": state[0],
                            "clock": state[1],
                            "period": state[2],
                            "home_score": state[3],
                            "away_score": state[4],
                            "players_data": json.loads(state[5]),
                        }
                        for state in game_states
                    ],
                    "prior_states": filtered_prior_states,
                },
                "response": final_state,
            }

    logging.info(f"Loaded data for {len(result)} games.")

    return result


@log_execution_time(average_over="inbound_data")
def parse_game_data(inbound_data):
    def parse_clock(clock_str):
        """Parse the clock from 'PT12M00.00S' to '12:00'."""
        minutes, seconds = clock_str.lstrip("PT").rstrip("S").split("M")
        minutes = int(minutes)
        seconds = int(seconds.split(".")[0])
        return f"{minutes}:{seconds:02}"

    logging.info(f"Parsing prompt-response data for {len(inbound_data)} games.")

    parsed_data = {}

    for game_id, game_info in inbound_data.items():
        prompt_info = game_info["prompt"]
        response_info = game_info["response"]

        home_team = prompt_info["home_team"]
        away_team = prompt_info["away_team"]
        season_type = "R" if prompt_info["season_type"] == "Regular Season" else "P"

        # Game Metadata
        game_metadata = f"{away_team}@{home_team}, {season_type}, {prompt_info['date_time_est'][:10]}"

        # Game States
        game_progress = []
        for state in prompt_info["game_states"]:
            game_state = {
                "play_id": state["play_id"],
                "clock": parse_clock(state["clock"]),
                "period": state["period"],
                home_team: {
                    "score": state["home_score"],
                    "player_points": {
                        pid: pdata["points"]
                        for pid, pdata in state["players_data"]["home"].items()
                    },
                },
                away_team: {
                    "score": state["away_score"],
                    "player_points": {
                        pid: pdata["points"]
                        for pid, pdata in state["players_data"]["away"].items()
                    },
                },
            }
            game_progress.append(game_state)

        # Prior States
        historical_data = {}
        for state_type, states in prompt_info["prior_states"].items():
            team = home_team if "home" in state_type else away_team
            key = f"{team}_past_games"
            historical_data[key] = []
            for state in states:
                past_game = {
                    "game_date": state["game_date"],
                    "matchup": f"{state['away']}@{state['home']}",
                    state["home"]: {
                        "score": state["home_score"],
                        "player_points": {
                            pid: pdata["points"]
                            for pid, pdata in state["players_data"]["home"].items()
                        },
                    },
                    state["away"]: {
                        "score": state["away_score"],
                        "player_points": {
                            pid: pdata["points"]
                            for pid, pdata in state["players_data"]["away"].items()
                        },
                    },
                }
                historical_data[key].append(past_game)

            # Sort past games by game_date in descending order
            historical_data[key] = sorted(
                historical_data[key], key=lambda x: x["game_date"], reverse=True
            )

        # Final State (Response)
        final_state = None
        if response_info:
            final_state = {
                home_team: {
                    "score": response_info["home_score"],
                    "player_points": {
                        pid: pdata["points"]
                        for pid, pdata in response_info["players_data"]["home"].items()
                    },
                },
                away_team: {
                    "score": response_info["away_score"],
                    "player_points": {
                        pid: pdata["points"]
                        for pid, pdata in response_info["players_data"]["away"].items()
                    },
                },
            }

        # Combine into final structure
        parsed_data[game_id] = {
            "prompt": {
                "game_metadata": game_metadata,
                "game_progress": game_progress,
                "historical_data": historical_data,
            },
            "response": final_state,
        }

    logging.info(f"Parsed data for {len(parsed_data)} games.")

    return parsed_data


def restrict_game_data(parsed_data, historical_game_count="all", num_plays="all"):
    restricted_data = {}

    historical_game_count = (
        None if historical_game_count == "all" else historical_game_count
    )
    num_plays = None if num_plays == "all" else num_plays

    for game_id, game_info in list(parsed_data.items()):
        restricted_game_progress = game_info["prompt"]["game_progress"][:num_plays]

        restricted_historical_data = {}
        for team_key, games in game_info["prompt"]["historical_data"].items():
            restricted_historical_data[team_key] = games[:historical_game_count]

        restricted_data[game_id] = {
            "prompt": {
                "game_metadata": game_info["prompt"]["game_metadata"],
                "game_progress": restricted_game_progress,
                "historical_data": restricted_historical_data,
            },
            "response": game_info["response"],
        }

    return restricted_data


@log_execution_time(average_over="inbound_data")
def finalize_game_data(inbound_data, output_filename="prompt_response_data.jsonl"):
    logging.info(f"Finalizing prompt-response data for {len(inbound_data)} games.")

    tokenizer = tiktoken.get_encoding("cl100k_base")

    total_tokens = {
        "prompt": 0,
        "response": 0,
        "game_metadata": 0,
        "game_progress": 0,
        "historical_data": 0,
    }
    game_count = len(inbound_data)

    # Add optional instructions to the top-level prompt data and count tokens
    for game_id, game_info in inbound_data.items():
        # Ensure there is a "prompt" key if not already present
        if "prompt" not in game_info:
            game_info["prompt"] = {}

        # Tokenize and count tokens for the prompt sections
        game_metadata_tokens = tokenizer.encode(
            "game_metadata: " + json.dumps(game_info["prompt"]["game_metadata"])
        )
        game_progress_tokens = tokenizer.encode(
            "game_progress: " + json.dumps(game_info["prompt"]["game_progress"])
        )
        historical_data_tokens = tokenizer.encode(
            "historical_data: " + json.dumps(game_info["prompt"]["historical_data"])
        )

        # Add to the total prompt tokens
        total_tokens["game_metadata"] += len(game_metadata_tokens)
        total_tokens["game_progress"] += len(game_progress_tokens)
        total_tokens["historical_data"] += len(historical_data_tokens)

        total_tokens["prompt"] += (
            len(game_metadata_tokens)
            + len(game_progress_tokens)
            + len(historical_data_tokens)
        )

        # Tokenize and count tokens for the response sections
        response_tokens = tokenizer.encode(json.dumps(game_info.get("response", {})))
        total_tokens["response"] += len(response_tokens)

    # Calculate averages
    average_tokens = {
        "prompt": total_tokens["prompt"] / game_count if game_count > 0 else 0,
        "game_metadata": (
            total_tokens["game_metadata"] / game_count if game_count > 0 else 0
        ),
        "game_progress": (
            total_tokens["game_progress"] / game_count if game_count > 0 else 0
        ),
        "historical_data": (
            total_tokens["historical_data"] / game_count if game_count > 0 else 0
        ),
        "response": total_tokens["response"] / game_count if game_count > 0 else 0,
    }

    logging.info(f"Finalized data for {game_count} games.")

    # Writing to a JSONL file
    with open(output_filename, "w") as outfile:
        for game_id, game_info in inbound_data.items():
            json_line = json.dumps({game_id: game_info})
            outfile.write(json_line + "\n")

    # Calculate the filesize of the output file
    file_size_bytes = os.path.getsize(output_filename)
    file_size_mb = file_size_bytes / 1024 / 1024

    logging.info(
        f"Data has been saved to {output_filename}.\nGame count: {game_count}\nFilesize: {file_size_mb:.2f} MB"
    )

    logging.info(
        f"Token Counts:\nPrompt: {total_tokens['prompt']} total. Average: {average_tokens['prompt']:.2f} per game."
    )
    logging.info(
        f"\tGame Metadata: {total_tokens['game_metadata']} total. Average: {average_tokens['game_metadata']:.2f} per game."
    )
    logging.info(
        f"\tGame Progress: {total_tokens['game_progress']} total. Average: {average_tokens['game_progress']:.2f} per game."
    )
    logging.info(
        f"\tHistorical Data: {total_tokens['historical_data']} total. Average: {average_tokens['historical_data']:.2f} per game."
    )
    logging.info(
        f"Response: {total_tokens['response']} total. Average: {average_tokens['response']:.2f} per game."
    )

    return inbound_data


if __name__ == "__main__":
    # Setup logging
    setup_logging(log_level="INFO")

    # PARAMS
    season = "2023-2024"
    historical_game_count = 10
    in_progress_game_state_count = 0

    # Load and process the game data
    d1 = load_game_data(season)
    d2 = parse_game_data(d1)
    d3 = restrict_game_data(
        d2,
        historical_game_count=historical_game_count,
        num_plays=in_progress_game_state_count,
    )
    instructions = None
    d4 = finalize_game_data(
        d3,
        instructions=instructions,
        output_filename=f"{PROJECT_ROOT}/data/prompt_response_data_{season}.jsonl",
    )

    # Take a random sample of 5 games
    sampled_game_ids = random.sample(list(d4.keys()), 5)
    sampled_game_data = {game_id: d4[game_id] for game_id in sampled_game_ids}
    # Save the sampled data to a JSON file
    with open("delme.json", "w") as f:
        json.dump(sampled_game_data, f, indent=4)
