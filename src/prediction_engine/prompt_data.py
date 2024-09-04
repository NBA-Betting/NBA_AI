import argparse
import logging
import sqlite3

from src.config import config
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

    #
    pass


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


if __name__ == "__main__":
    main()
