"""
prompt_data.py

Warning:
- Using the OpenAI API will incur costs. Make sure to set usage limits and monitor usage to avoid unexpected charges.

This module handles the loading and processing of prompt data for NBA game predictions.
It consists of functions to:
- Load prompt data from the database.
- Generate prompt data for individual games.
- Format prior game states for teams.

Functions:
- load_prompt_data(game_ids, db_path=DB_PATH): Loads prompt data for a list of game IDs from the database.
- generate_game_prompt(game, prior_states): Generates prompt data for a single game.
- format_prior_games(prior_states): Formats and sorts prior games for either home or away team.
- main(): Main function to handle command-line arguments and orchestrate the prompt data loading process.

Usage:
- Typically run as part of a larger data processing pipeline.
- Script can be run directly from the command line to generate prompt data for specified game IDs.
    Example: python -m src.predictions.prompt_data --game_ids=0042300401,0022300649 --log_level=DEBUG
- Successful execution will display logs for prompt data loading and processing.
"""

import argparse
import logging
import sqlite3

from src.config import config
from src.database_updater.prior_states import (
    determine_prior_states_needed,
    load_prior_states,
)
from src.logging_config import setup_logging
from src.utils import log_execution_time

# Configuration
DB_PATH = config["database"]["path"]


@log_execution_time(average_over="game_ids")
def load_prompt_data(game_ids, db_path=DB_PATH):
    """
    Load prompt data from the database for a list of game_ids.

    Args:
        game_ids (list): A list of game_ids to load prompt data for.
        db_path (str): The path to the SQLite database file. Defaults to DB_PATH from config.

    Returns:
        dict: A dictionary where each key is a game_id and each value is the corresponding prompt data.
    """
    logging.info(f"Loading prompt data for {len(game_ids)} games...")

    # Fetch game data for the provided game_ids from the database
    query = f"""
    SELECT game_id, date_time_est, home_team, away_team, season, season_type
    FROM Games
    WHERE game_id IN ({','.join('?' for _ in game_ids)})
    """
    with sqlite3.connect(db_path) as conn:
        games = conn.execute(query, game_ids).fetchall()

    # Fetch prior state data for all games
    prior_states_dict = load_prior_states(determine_prior_states_needed(game_ids))

    # Process games to generate prompts
    prompt_dict = {
        game[0]: generate_game_prompt(game, prior_states_dict[game[0]])
        for game in games
    }

    logging.info(f"Loaded prompt data for {len(prompt_dict)} games.")
    logging.debug(f"Prompt data example: {prompt_dict.get(game_ids[0])}")

    return prompt_dict


def generate_game_prompt(game, prior_states):
    """
    Generate prompt data for a single game.

    Args:
        game (tuple): The game data tuple.
        prior_states (dict): Dictionary containing prior states data.

    Returns:
        dict: A dictionary with prompt information for the game.
    """
    _, date_time_est, home_team, away_team, season, season_type = game

    # Process prior games data for home and away teams using list comprehensions
    home_prior_games = format_prior_games(prior_states.get("home_prior_states", []))
    away_prior_games = format_prior_games(prior_states.get("away_prior_states", []))

    # Generate the textual prompt
    prompt = (
        f"Home Team: {home_team}\n"
        f"Away Team: {away_team}\n"
        f"Game Date: {date_time_est[:10]}\n"
        f"Season Type: {season_type}\n"
        f"Home Teams Most Recent Previous Games: \n{home_prior_games}\n"
        f"Away Teams Most Recent Previous Games: \n{away_prior_games}"
    )

    return {
        "home_team": home_team,
        "away_team": away_team,
        "game_date": date_time_est[:10],
        "season": season,
        "season_type": season_type,
        "prompt": prompt,
    }


def format_prior_games(prior_states):
    """
    Format and sort prior games for either home or away team.

    Args:
        prior_states (list): List of prior game states.

    Returns:
        list: A sorted and formatted list of prior games up to the maximum specified number (10).
    """
    # Format and sort games by date in descending order, limiting to 10
    return sorted(
        [
            {
                "home_team": state["home"],
                "away_team": state["away"],
                "game_date": state["game_date"],
                "home_team_score": state["home_score"],
                "away_team_score": state["away_score"],
            }
            for state in prior_states
        ],
        key=lambda x: x["game_date"],
        reverse=True,
    )[:10]


def main():
    """
    Main function to handle command-line arguments and orchestrate the prompt data loading process.
    """
    parser = argparse.ArgumentParser(
        description="Create prompt data for a list of game IDs."
    )
    parser.add_argument(
        "--game_ids", type=str, help="Comma-separated list of game IDs to process"
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        help="The logging level. Default is INFO. DEBUG provides more details.",
    )

    args = parser.parse_args()
    log_level = args.log_level.upper()
    setup_logging(log_level=log_level)

    game_ids = args.game_ids.split(",") if args.game_ids else []

    prompt_data = load_prompt_data(game_ids)
    print(prompt_data)


if __name__ == "__main__":
    main()
