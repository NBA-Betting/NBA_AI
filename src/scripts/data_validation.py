import json
import os
import re
import sqlite3
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

# Add the parent directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import update_scheduled_games, validate_game_ids, validate_season_format

load_dotenv()
PROJECT_ROOT = os.getenv("PROJECT_ROOT")


def validate_pbp(game_id, db_path=None, logs=None):
    """
    This function validates the play-by-play logs for a given game.
    The logs can be provided directly or fetched from a SQLite database.

    Parameters:
    game_id (str): The ID of the game.
    db_path (str, optional): The path to the SQLite database. Defaults to None.
    logs (list, optional): The list of logs to validate. Defaults to None.

    Raises:
    ValueError: If both or none of db_path and logs are provided.
    ValueError: If the log list is empty.
    ValueError: If a log is missing required fields.
    ValueError: If the order of logs is incorrect.
    ValueError: If there is a database error.
    ValueError: If there is a JSON parsing error.

    Returns:
    bool: True if the logs are valid, False otherwise.
    """

    # Validate the game_id
    validate_game_ids([game_id])

    # Ensure that only one of db_path or logs is provided
    if (db_path is not None and logs is not None) or (db_path is None and logs is None):
        raise ValueError(
            f"Game {game_id}: Either db_path or logs should be provided, not both or none."
        )

    def validate_logs(logs):
        """
        This function validates a list of logs.

        Parameters:
        logs (list): The list of logs to validate.

        Raises:
        ValueError: If the log list is empty.
        ValueError: If a log is missing required fields.
        ValueError: If the order of logs is incorrect.

        Returns:
        bool: True if the logs are valid, False otherwise.
        """

        # Check if log list is not empty
        if not logs:
            raise ValueError(f"Game {game_id}: Log list is empty.")

        errors = []

        for i in range(len(logs)):
            log = logs[i]

            # Check if required fields exist
            required_fields = [
                "period",
                "clock",
                "description",
                "scoreHome",
                "scoreAway",
            ]
            missing_fields = [field for field in required_fields if field not in log]

            # Check if either "actionId" or "orderNumber" is in log
            if not ("actionId" in log or "orderNumber" in log):
                missing_fields.append("actionId/orderNumber")

            if missing_fields:
                errors.append(
                    f"{game_id}: A log is missing the following required fields at index {i}: {', '.join(missing_fields)}"
                )

            # If not the first log, perform order checks
            if i > 0:
                previous_log = logs[i - 1]

                # Check if the current log's actionId or orderNumber is less than the previous log's actionId or orderNumber
                if log.get("actionId", log.get("orderNumber")) < previous_log.get(
                    "actionId", previous_log.get("orderNumber")
                ):
                    errors.append(
                        f"{game_id}: Order error: 'actionId' or 'orderNumber' decreased at index {i}."
                    )

                # Check if the current log's period is less than the previous log's period
                if log["period"] < previous_log["period"]:
                    errors.append(
                        f"{game_id}: Order error: 'period' decreased at index {i}."
                    )
                # If periods are the same, check if clock value has increased
                elif (
                    log["period"] == previous_log["period"]
                    and log["clock"] > previous_log["clock"]
                ):
                    errors.append(
                        f"{game_id}: Order error: 'clock' increased within the same 'period' at index {i}."
                    )

        # If there are any errors, raise them all at once
        if errors:
            raise ValueError("\n".join(errors))

        # If no errors are raised, the logs are in the correct order
        return True

    # If logs are provided directly, validate them
    if logs is not None:
        validate_logs(logs)
    # If db_path is provided, fetch the logs from the database and validate them
    elif db_path is not None:
        try:
            # Use a context manager to handle the database connection
            with sqlite3.connect(db_path) as conn:
                # Create a cursor object
                cur = conn.cursor()
                # Execute the query to fetch logs for the given game_id
                cur.execute(
                    "SELECT log_data FROM PbP_Logs WHERE game_id=?",
                    (game_id,),
                )
                # Fetch the logs
                rows = cur.fetchall()
                # Parse the JSON strings into Python dictionaries
                logs = [json.loads(row[0]) for row in rows]
                # Validate the logs
                validate_logs(logs)
        except sqlite3.Error as e:
            raise ValueError(f"Game {game_id}: Database error - {str(e)}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Game {game_id}: JSON parsing error - {str(e)}")


def validate_game_states(game_id, db_path=None, states=None):
    """
    Validates the game states for a given game_id. The game states can be provided directly or fetched from a database.

    Parameters:
    game_id (str): The ID of the game to validate.
    db_path (str, optional): The path to the SQLite database file. If provided, the game states will be fetched from the database.
    states (list, optional): A list of game states to validate. Each game state is a dictionary with the following keys:
        - game_id: The ID of the game.
        - play_id: The ID of the play.
        - game_date: The date of the game.
        - home: The name of the home team.
        - away: The name of the away team.
        - clock: The game clock.
        - period: The game period.
        - home_score: The score of the home team.
        - away_score: The score of the away team.
        - total: The total score.
        - home_margin: The score margin of the home team.
        - is_final_state: A flag indicating whether this is the final state.
        - players_data: A dictionary with player data.

    Raises:
    ValueError: If any of the game states are invalid. This includes:
        - Missing required fields in a state.
        - Non-unique play_id.
        - play_id not in sequence.
        - Inconsistent team names.
        - Incorrect game_date format.
        - Invalid home margin or total.
        - Decreasing score, period, or clock.
        - Incorrect is_final_state flag.
        - Invalid players_data structure.
        - Removed player or decreased player points.
    sqlite3.Error: If there is a database error when fetching the states.
    json.JSONDecodeError: If there is an error parsing the players_data from JSON.

    Returns:
    None
    """
    # Validate the game_id
    validate_game_ids([game_id])

    # Ensure that only one of db_path or states is provided
    # Raise an error if both or none are provided
    if (db_path is not None and states is not None) or (
        db_path is None and states is None
    ):
        raise ValueError(
            f"Game {game_id}: Either db_path or states should be provided, not both or none."
        )

    # Define a nested function to validate the states
    def validate_states(states):
        # Raise an error if the states list is empty
        if not states:
            raise ValueError(f"{game_id}: States list is empty.")

        # Check for non-unique play_ids
        all_play_ids = [state["play_id"] for state in states]
        play_id_counts = Counter(all_play_ids)

        non_unique_play_ids = [
            play_id for play_id, count in play_id_counts.items() if count > 1
        ]
        if non_unique_play_ids:
            raise ValueError(
                f"{game_id}: play_id is not unique. Non-unique play_id(s): {non_unique_play_ids}"
            )

        # Compile a regex for date validation
        date_regex = re.compile(r"\d{4}-\d{2}-\d{2}")
        first_state = states[0]
        previous_state = None

        # List to store errors
        errors = []

        # Iterate over the states
        for i, state in enumerate(states):
            # Check if all required fields exist in the state
            # Raise an error if any required field is missing
            required_fields = [
                "game_id",
                "play_id",
                "game_date",
                "home",
                "away",
                "clock",
                "period",
                "home_score",
                "away_score",
                "total",
                "home_margin",
                "is_final_state",
                "players_data",
            ]
            missing_fields = [field for field in required_fields if field not in state]
            if missing_fields:
                errors.append(
                    f"{game_id}: A state is missing the following required fields at index {i}: {', '.join(missing_fields)}"
                )

            # Check if play_id is in sequence
            if previous_state and state["play_id"] < previous_state["play_id"]:
                errors.append(f"{game_id}: play_id is increasing at index {i}.")

            # Check if team names are consistent across states
            if (
                state["home"] != first_state["home"]
                or state["away"] != first_state["away"]
            ):
                errors.append(f"{game_id}: Team names are not consistent at index {i}.")

            # Check if game_date is in the correct format
            if not date_regex.match(state["game_date"]):
                errors.append(
                    f"{game_id}: game_date is not in the correct format at index {i}."
                )

            # Check if home_margin and total are correct
            if state["home_margin"] != state["home_score"] - state["away_score"]:
                errors.append(f"{game_id}: Home margin is invalid at index {i}.")

            if state["total"] != state["home_score"] + state["away_score"]:
                errors.append(f"{game_id}: Total is invalid at index {i}.")

            # Check if score, period, and clock are progressing correctly
            if previous_state:
                if state["period"] < previous_state["period"]:
                    errors.append(f"{game_id}: Period is decreasing at index {i}.")

                if (
                    state["period"] == previous_state["period"]
                    and state["clock"] > previous_state["clock"]
                ):
                    errors.append(
                        f"{game_id}: Clock is increasing within the same period at index {i}."
                    )

                if state["home_score"] < previous_state["home_score"]:
                    errors.append(f"{game_id}: Home score is decreasing at index {i}.")

                if state["away_score"] < previous_state["away_score"]:
                    errors.append(f"{game_id}: Away score is decreasing at index {i}.")

                if state["total"] < previous_state["total"]:
                    errors.append(f"{game_id}: Total score is decreasing at index {i}.")

            # Check if is_final_state flag is correct
            if state["is_final_state"] != (i == len(states) - 1):
                errors.append(f"{game_id}: Incorrect is_final_state flag at index {i}.")

            # Check if players_data structure is correct
            if set(state["players_data"].keys()) != {"home", "away"}:
                errors.append(
                    f"Game {game_id}: players_data should have 'home' and 'away' as keys at index {i}."
                )

            for team in state["players_data"].values():
                for player in team.values():
                    if set(player.keys()) != {"name", "points"}:
                        errors.append(
                            f"Game {game_id}: Each player should have 'name' and 'points' as keys at index {i}."
                        )
            if previous_state:
                # Check if any player is removed from the data
                for team in ["home", "away"]:
                    for player in previous_state["players_data"][team]:
                        if player not in state["players_data"][team]:
                            errors.append(
                                f"Game {game_id}: Player {player} was removed from {team} team at index {i}."
                            )

                # Check if any player's points have decreased
                for team in ["home", "away"]:
                    for player in state["players_data"][team]:
                        if (
                            player in previous_state["players_data"][team]
                            and state["players_data"][team][player]["points"]
                            < previous_state["players_data"][team][player]["points"]
                        ):
                            errors.append(
                                f"Game {game_id}: Points for player {player} in {team} team decreased at index {i}."
                            )

            # Update previous_state for the next iteration
            previous_state = state

            # Raise all errors at once
            if errors:
                raise ValueError("\n".join(errors))

    # If states are provided directly, validate them
    if states is not None:
        validate_states(states)
    # If db_path is provided, fetch the states from the database and validate them
    elif db_path is not None:
        try:
            # Use a context manager to handle the database connection
            with sqlite3.connect(db_path) as conn:
                # Create a cursor object
                cur = conn.cursor()
                # Execute the query to fetch states for the given game_id
                cur.execute(
                    """SELECT game_id,
                              play_id,
                              game_date,
                              home,
                              away,
                              clock,
                              period,
                              home_score,
                              away_score,
                              total,
                              home_margin,
                              is_final_state,
                              players_data
                       FROM GameStates
                       WHERE game_id=?""",
                    (game_id,),
                )
                # Fetch the states
                rows = cur.fetchall()
                # Get column names from cursor description
                columns = [column[0] for column in cur.description]
                # Convert list of tuples to list of dicts
                states = [dict(zip(columns, row)) for row in rows]
                # Parse players_data from string to dictionary
                for state in states:
                    state["players_data"] = json.loads(state["players_data"])
                # Validate the states
                validate_states(states)
        except sqlite3.Error as e:
            # Raise an error if there is a database error
            raise ValueError(f"Game {game_id}: Database error - {str(e)}")
        except json.JSONDecodeError as e:
            # Raise an error if there is an error parsing the players_data from JSON
            raise ValueError(f"Game {game_id}: JSON parsing error - {str(e)}")


def validate_prior_states(db_path):
    """
    Connects to a SQLite database and checks for errors in the prior states of games.

    Args:
        db_path (str): The path to the SQLite database.

    Returns:
        Tuple of Dictionaries of seasons and corresponding game_ids that have errors in their prior states.

    Prints:
        Dictionaries of seasons and corresponding game_ids that have errors in their prior states.
    """
    # Connect to the SQLite database
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()

        print("Updating scheduled games...")
        # Get seasons in the database
        cursor.execute("SELECT DISTINCT season FROM Games")
        seasons = [row[0] for row in cursor.fetchall()]
        # Update the scheduled games for each season
        for season in tqdm(
            seasons, desc="Updating scheduled games", unit="season", dynamic_ncols=True
        ):
            update_scheduled_games(season, db_path)
        print("Scheduled games updated.")

        print("\nChecking for finalized prior states that should not be finalized...")
        # Prepare a query to check for games that have finalized prior states but should not
        query = """
            SELECT g.season, GROUP_CONCAT(DISTINCT g.game_id) as error_game_ids
            FROM Games g
            INNER JOIN PriorStates ps ON ps.game_id = g.game_id
            WHERE ps.are_prior_states_finalized = 1
            AND g.season_type IN ('Regular Season', 'Post Season')
            AND EXISTS (
                SELECT 1
                FROM Games g2
                WHERE (g2.home_team = g.home_team OR g2.away_team = g.home_team OR g2.home_team = g.away_team OR g2.away_team = g.away_team)
                AND g2.date_time_est < g.date_time_est
                AND g2.season = g.season
                AND g2.season_type IN ('Regular Season', 'Post Season')
                AND g2.status != 'Completed'
            )
            GROUP BY g.season
        """
        # Execute the query
        cursor.execute(query)
        # Fetch the result
        results = cursor.fetchall()
        # Convert the results to a dictionary
        errors_1 = {row[0]: row[1].split(",") for row in results}

        print("\nChecking for non-finalized prior states that should be finalized...")
        # Prepare a query to check for games that should have finalized prior states but do not
        query = """
            SELECT g.season, GROUP_CONCAT(DISTINCT g.game_id) AS game_ids
            FROM Games g
            WHERE g.season_type IN ('Regular Season', 'Post Season')
            AND NOT EXISTS (
                SELECT 1
                FROM Games g2
                WHERE (g2.home_team = g.home_team OR g2.away_team = g.home_team OR g2.home_team = g.away_team OR g2.away_team = g.away_team)
                AND g2.date_time_est < g.date_time_est
                AND g2.season = g.season
                AND g2.season_type IN ('Regular Season', 'Post Season')
                AND g2.status != 'Completed'
            )
            AND NOT EXISTS (
                SELECT 1
                FROM PriorStates ps
                WHERE ps.game_id = g.game_id
                AND ps.are_prior_states_finalized = 1
            )
            GROUP BY g.season
        """
        # Execute the query
        cursor.execute(query)
        # Fetch the result
        results = cursor.fetchall()
        # Convert the results to a dictionary
        errors_2 = {row[0]: row[1].split(",") for row in results}

        # Print the results
        print("\nGames with finalized prior states that should not be finalized:")
        print(errors_1)

        print("\nGames with non-finalized prior states that should be finalized:")
        print(errors_2)

        return errors_1, errors_2


def validate_game(game_id, db_path):
    """
    Validates a game by its game_id.

    Args:
        game_id (str): The game_id to validate.
        db_path (str): The path to the SQLite database.

    Raises:
        ValueError: If the game is completed but missing a final state.
    """
    # Validate the game ID
    validate_game_ids([game_id])

    # Validate the play-by-play data for the game
    validate_pbp(game_id, db_path)

    # Validate the game states for the game
    validate_game_states(game_id, db_path)

    # Connect to the SQLite database
    with sqlite3.connect(db_path) as conn:
        # Prepare a query to check if there exists a game with the given game_id
        # where the game status is 'Completed', but there is no corresponding
        # game state marked as the final state
        query = """
        SELECT EXISTS (
            SELECT 1
            FROM Games g
            LEFT JOIN GameStates gs ON g.game_id = gs.game_id AND gs.is_final_state = 1
            WHERE g.status = 'Completed'
            AND gs.game_id IS NULL
            AND g.game_id = ?
        );
        """
        cursor = conn.cursor()
        # Execute the query
        cursor.execute(query, (game_id,))
        # Fetch the result
        result = cursor.fetchone()[0]
        # If the result is 1 (i.e., there is such a game), raise a ValueError
        if result:
            raise ValueError(f"Game {game_id}: Missing final state for completed game.")


def validate_season(season, db_path):
    """
    Validates all games in a given season.

    Args:
        season (str): The season to validate in 'YYYY-YYYY' format.
        db_path (str): The path to the SQLite database.

    Raises:
        ValueError: If any game in the season is invalid.
    """
    # Validate the season format
    validate_season_format(season)

    # Update the scheduled games for the given season
    update_scheduled_games(season, db_path)

    # Connect to the SQLite database
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        # Prepare a query to fetch the game IDs for the given season
        query = "SELECT game_id FROM Games WHERE season=? AND season_type IN ('Regular Season', 'Post Season')"
        # Execute the query
        cursor.execute(query, (season,))
        # Fetch the game IDs
        game_ids = [row[0] for row in cursor.fetchall()]

    # Sort the game IDs
    game_ids.sort()

    # Initialize an empty dictionary to store any errors
    errors = {}

    # Validate each game
    for game_id in tqdm(
        game_ids, desc="Validating games", unit="game", dynamic_ncols=True
    ):
        try:
            # Validate the game
            validate_game(game_id, db_path)
        except Exception as e:
            # If an error occurs, split it into separate lines and store it in the errors dictionary
            errors[game_id] = str(e).split("\n")
            print(str(e))
            continue

    # If there are any errors, write them to a JSON file
    if errors:
        # Get the current date and time
        log_time = datetime.now().strftime("%Y%m%d%H%M%S")
        # Open a file to write the errors
        with open(f"validation_errors_{season}_{log_time}.json", "w") as f:
            # Dump the errors dictionary to the file as JSON
            json.dump(errors, f, indent=4)


if __name__ == "__main__":
    db_path = os.path.join(PROJECT_ROOT, "data", "NBA_AI.sqlite")

    # -----------------------------------------------------
    # Uncomment the following code to validate by seasons
    # -----------------------------------------------------

    # seasons = [
    #     "2000-2001",
    #     "2001-2002",
    #     "2002-2003",
    #     "2003-2004",
    #     "2004-2005",
    #     "2005-2006",
    #     "2006-2007",
    #     "2007-2008",
    #     "2008-2009",
    #     "2009-2010",
    #     "2010-2011",
    #     "2011-2012",
    #     "2012-2013",
    #     "2013-2014",
    #     "2014-2015",
    #     "2015-2016",
    #     "2016-2017",
    #     "2017-2018",
    #     "2018-2019",
    #     "2019-2020",
    #     "2020-2021",
    #     "2021-2022",
    #     "2022-2023",
    #     "2023-2024",
    # ]

    # for season in seasons:
    #     validate_season(season, db_path)

    # ----------------------------------------------------------
    # Uncomment the following code to validate individual games
    # This does not automatically update the scheduled games
    # ----------------------------------------------------------

    # game_ids = []

    # for game_id in game_ids:
    #     try:
    #         validate_game(game_id, db_path)
    #         print(f"Game {game_id}: Validation successful.")
    #     except Exception as e:
    #         print(f"Game {game_id}: Validation failed - {str(e)}")
    #         continue

    # ----------------------------------------------------------
    # Uncomment the following code to validate prior states
    # ----------------------------------------------------------

    # validate_prior_states(db_path)
