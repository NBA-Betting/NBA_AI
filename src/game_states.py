import json
import logging
import sqlite3
from copy import deepcopy

from .pbp import get_pbp, save_pbp
from .utils import lookup_basic_game_info, validate_date_format, validate_game_ids

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s",
)


def create_game_states(pbp_logs, home, away, game_id, game_date):
    """
    Create a list of game states from play-by-play logs.

    Parameters:
    pbp_logs (list): A list of dictionaries representing play-by-play logs.
    home (str): The home team's tricode.
    away (str): The away team's tricode.
    game_id (str): The ID of the game.
    game_date (str): The date of the game in 'YYYY-MM-DD' format.

    Returns:
    list: A list of dictionaries representing the game states. Each dictionary contains the game ID, play ID, game date, home team, away team, remaining time, period, home score, away score, total score, home margin, whether the state is the final state, and a dictionary of player data. If an error occurs, an empty list is returned.
    """
    try:
        # Validate the game_id and game_date
        validate_game_ids(game_id)
        validate_date_format(game_date)
        # If there are no play-by-play logs, return an empty list and dictionary
        if not pbp_logs:
            return []

        # Sort the play-by-play logs by order number
        pbp_logs = sorted(pbp_logs, key=lambda x: x["orderNumber"])

        # Filter out logs where 'description' is not a key
        pbp_logs = [log for log in pbp_logs if "description" in log]

        # Initialize the list of game states and the players dictionary
        game_states = []
        players = {"home": {}, "away": {}}

        # Iterate over the play-by-play logs
        for row in pbp_logs:
            # If the play involves a player, update the player's points
            if row.get("personId") is not None and row.get("playerNameI") is not None:
                # Determine the team of the player
                team = "home" if row["teamTricode"] == home else "away"

                # Extract the player's ID and name
                player_id = row["personId"]
                player_name = row["playerNameI"]

                # If the player is not already in the game state, add them
                if player_id not in players[team]:
                    players[team][player_id] = {"name": player_name, "points": 0}

                # If the play resulted in points, update the player's points
                if row.get("pointsTotal") is not None:
                    points = int(row["pointsTotal"])
                    players[team][player_id]["points"] = points

            # Create the current game state
            current_game_state = {
                "game_id": game_id,
                "play_id": int(row["orderNumber"]),
                "game_date": game_date,
                "home": home,
                "away": away,
                "remaining_time": row["clock"],
                "period": int(row["period"]),
                "home_score": int(row["scoreHome"]),
                "away_score": int(row["scoreAway"]),
                "total": int(row["scoreHome"]) + int(row["scoreAway"]),
                "home_margin": int(row["scoreHome"]) - int(row["scoreAway"]),
                "is_final_state": row["description"] == "Game End",
                "players_data": deepcopy(
                    players
                ),  # Create a deep copy of the players dictionary
            }

            # Add the current game state to the list of game states
            game_states.append(current_game_state)

        return game_states

    except Exception as e:
        logging.error(f"Game Id {game_id} - Failed to create game states. {e}")
        return []


def save_game_states(game_states, db_path):
    """
    Saves the game states to the database. Each game_id is processed in a separate transaction to ensure all-or-nothing behavior.

    Parameters:
    game_states (dict): A dictionary with game IDs as keys and lists of dictionaries (game states) as values.
    db_path (str): The path to the SQLite database file.

    Returns:
    bool: True if the operation was successful for all game IDs, False otherwise.
    """
    overall_success = True

    try:
        with sqlite3.connect(db_path) as conn:
            for game_id, states in game_states.items():
                if not states:  # Skip if there are no game states for this game ID
                    logging.info(
                        f"Game Id {game_id} - No game states to save. Skipping."
                    )
                    continue

                try:
                    # Begin a new transaction for each game_id
                    conn.execute("BEGIN")
                    data_to_insert = [
                        (
                            game_id,
                            state["play_id"],
                            state["game_date"],
                            state["home"],
                            state["away"],
                            state["remaining_time"],
                            state["period"],
                            state["home_score"],
                            state["away_score"],
                            state["total"],
                            state["home_margin"],
                            state["is_final_state"],
                            json.dumps(state["players_data"]),
                        )
                        for state in states
                    ]

                    # Use executemany to insert or replace data in a single operation
                    conn.executemany(
                        """
                        INSERT OR REPLACE INTO GameStates (game_id, play_id, game_date, home, away, remaining_time, period, home_score, away_score, total, home_margin, is_final_state, players_data)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        data_to_insert,
                    )
                    conn.commit()  # Commit the transaction if no errors occurred
                except Exception as e:
                    conn.rollback()  # Roll back the transaction if an error occurred
                    logging.error(f"Game Id {game_id} - Error saving game states. {e}")
                    overall_success = False  # Mark the overall operation as failed, but continue processing other game_ids

    except Exception as e:
        logging.error(f"Database connection error: {e}")
        return False  # Return False immediately if a database connection error occurred

    return overall_success  # Return True if the operation was successful for all game_ids, False otherwise


def get_current_games_info(game_ids, db_path, save_to_db=True):
    """
    Retrieves the current game information given a game ID or a list of game IDs.

    Parameters:
    game_ids (str or list): The ID(s) of the game(s) to retrieve the information for.
    db_path (str): The path to the SQLite database file.
    save_to_db (bool): If True, saves the play-by-play logs and game states to the database.

    Returns:
    list: A list of dictionaries, each containing the game information for a game ID, including the game ID, game date, game time (EST), home team, away team, game status, play-by-play logs, game states, and final state.
    """

    # Validate the game_id
    validate_game_ids(game_ids)

    # Retrieve the basic game information
    games = lookup_basic_game_info(game_ids, db_path)

    # Retrieve the play-by-play logs for the game
    pbp_logs = get_pbp(game_ids)

    # If save_to_db is True, save the play-by-play logs to the database
    if save_to_db:
        save_pbp(pbp_logs, db_path)

    # Create the game states from the play-by-play logs
    game_states = {}
    for game in games:
        game_id = game["game_id"]
        game_states[game_id] = create_game_states(
            pbp_logs[game_id],
            game["home_team"],
            game["away_team"],
            game_id,
            game["date_time_est"].split("T")[0],
        )

    # If save_to_db is True, save the game states to the database
    if save_to_db:
        save_game_states(game_states, db_path)

    # Create a dictionary with the game information
    game_info_list = []
    for game in games:
        game_info_dict = {
            "game_id": game["game_id"],
            "game_date": game["date_time_est"].split("T")[0],
            "game_time_est": game["date_time_est"].split("T")[1].rstrip("Z"),
            "home": game["home_team"],
            "away": game["away_team"],
            "game_status": game["status"],
            "season": game["season"],
            "season_type": game["season_type"],
            "pbp_logs": pbp_logs[game["game_id"]],
            "game_states": game_states[game["game_id"]],
        }
        game_info_list.append(game_info_dict)

    return game_info_list


def print_current_game_info(game_info):
    are_game_states_finalized = any(
        state["is_final_state"] for state in reversed(game_info["game_states"])
    )
    print()
    print("Game ID:", game_info["game_id"])
    print("Game Date:", game_info["game_date"])
    print("Game Time (EST):", game_info["game_time_est"])
    print("Home Team:", game_info["home"])
    print("Away Team:", game_info["away"])
    print("Game Status:", game_info["game_status"])
    print("Play-by-Play Log Count:", len(game_info["pbp_logs"]))
    print("Game States Count:", len(game_info["game_states"]))
    print("Are Game States Finalized:", are_game_states_finalized)
    if game_info["game_status"] != "Not Started":
        most_recent_state = game_info["game_states"][-1]
        print("Most Recent State:")
        print("  Remaining Time:", most_recent_state["remaining_time"])
        print("  Period:", most_recent_state["period"])
        print("  Home Score:", most_recent_state["home_score"])
        print("  Away Score:", most_recent_state["away_score"])
        print("  Total Score:", most_recent_state["total"])
        print("  Home Margin:", most_recent_state["home_margin"])
        print("  Players:", most_recent_state["players_data"])
    print()
