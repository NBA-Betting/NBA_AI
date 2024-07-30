"""
modeling_utils.py

This script is designed to load featurized modeling data from an SQLite database containing NBA game data. 
It extracts game information, final scores, and a set of features for each game, for specified seasons.

Overview:
The script connects to an SQLite database, runs a query to retrieve data from the Games, GameStates, and 
Features tables, and then processes this data to provide a comprehensive DataFrame. The primary use case 
is to support machine learning models for predicting NBA game outcomes.

Functions:
- load_featurized_modeling_data(seasons, db_path): Loads featurized modeling data from the database.
- main(): Main function to parse command-line arguments and initiate data loading.

Usage:
To run the script, use the following command:
    python modeling_utils.py --seasons=2021-2022,2022-2023 --log_level=INFO

Arguments:
- --seasons: Comma-separated list of seasons to load data for (e.g., 2021-2022,2022-2023).
- --log_level: Logging level (e.g., INFO, DEBUG). Defaults to INFO.
"""

import argparse
import json
import logging
import time

import pandas as pd
from sqlalchemy import create_engine

from src.config import config
from src.logging_config import setup_logging
from src.utils import log_execution_time, validate_season_format

pd.set_option("display.max_columns", None)

# Configuration
DB_PATH = config["database"]["path"]


@log_execution_time(average_over="seasons")
def load_featurized_modeling_data(seasons, db_path=DB_PATH):
    """
    Load featurized modeling data from the Games, GameStates, and Features tables.

    This function retrieves data from the specified seasons, including game details, final scores, and feature sets.
    The data is filtered to include only regular and post-season games that are finalized.

    Args:
        seasons (list of str): A list of seasons to load data for (e.g., ['2021-2022', '2022-2023']).
        db_path (str): The path to the SQLite database.

    Returns:
        pd.DataFrame: A DataFrame containing the consolidated game data, including expanded feature sets.
    """
    logging.info(f"Loading featurized modeling data for seasons: {seasons}...")

    # Validate the seasons format
    for season in seasons:
        validate_season_format(season)

    # Convert seasons list to tuple for IN clause compatibility, if not already a tuple
    seasons_tuple = tuple(seasons) if isinstance(seasons, list) else seasons

    # Timing the database loading step
    start_time = time.time()
    query = """
        SELECT g.game_id, g.date_time_est, g.home_team, g.away_team, g.season, g.season_type, 
               gs.home_score, gs.away_score, gs.total, gs.home_margin, gs.players_data, f.feature_set
        FROM (
            SELECT game_id, home_score, away_score, total, home_margin, players_data
            FROM GameStates
            WHERE is_final_state = 1
        ) AS gs
        INNER JOIN Games g ON g.game_id = gs.game_id
        INNER JOIN Features f ON g.game_id = f.game_id
        WHERE g.season IN ({placeholders})
        AND g.season_type IN ('Regular Season', 'Post Season')
        AND g.game_data_finalized = 1
        AND g.pre_game_data_finalized = 1
    """.format(
        placeholders=",".join("?" for _ in seasons_tuple)
    )

    # Load the data into a DataFrame
    engine = create_engine(f"sqlite:///{db_path}")
    df = pd.read_sql_query(query, engine, params=seasons_tuple)
    db_load_time = time.time() - start_time
    logging.info(f"Database loading time: {db_load_time:.2f} seconds")

    # Timing the JSON normalization step
    start_time = time.time()
    df_feature_set = pd.json_normalize(df["feature_set"].apply(json.loads))
    json_normalization_time = time.time() - start_time
    logging.info(f"JSON normalization time: {json_normalization_time:.2f} seconds")

    # Timing the data processing step
    start_time = time.time()
    df.drop(columns=["feature_set"], inplace=True)
    df_final = pd.concat([df, df_feature_set], axis=1)
    data_processing_time = time.time() - start_time
    logging.info(f"Data processing time: {data_processing_time:.2f} seconds")

    logging.info(f"Data loaded successfully.")
    logging.info(f"Total number of rows loaded: {len(df_final)}")

    logging.debug(f"Data Head:\n{df_final.head()}\n")
    logging.debug(f"Data Tail:\n{df_final.tail()}\n")
    logging.debug(f"Data Info:\n{df_final.info()}\n")

    return df_final


def main():
    """
    Main function to parse command-line arguments and initiate data loading.

    This function sets up logging, parses command-line arguments for seasons and logging level,
    and then calls `load_featurized_modeling_data` to load and process the data.

    Args:
        None

    Returns:
        None
    """
    parser = argparse.ArgumentParser(
        description="Load featurized modeling data from a database."
    )
    parser.add_argument(
        "--seasons",
        type=str,
        help="Comma-separated list of seasons to load data for (e.g., 2021-2022,2022-2023).",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        help="Logging level (e.g., INFO, DEBUG).",
    )

    args = parser.parse_args()
    log_level = args.log_level
    setup_logging(log_level=log_level)

    # Split the input string into a list of seasons
    seasons = args.seasons.split(",")

    # Load the data
    df = load_featurized_modeling_data(seasons)


if __name__ == "__main__":
    main()
