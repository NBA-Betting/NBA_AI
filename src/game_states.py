"""
game_states.py

This module parses play-by-play logs into game states and saves them to the database.
It consists of functions to:
- Create game states from play-by-play logs.
- Save game states to the SQLite database.

Functions:
- create_game_states(pbp_logs, home, away, game_id, game_date): Creates a list of game states from play-by-play logs.
- save_game_states(game_states, db_path): Saves the game states to the database.

Usage:
- Only used internally by other modules to process and save play-by-play data.
"""

import json
import logging
import sqlite3
from copy import deepcopy

from src.utils import validate_date_format, validate_game_ids

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

        if not pbp_logs:
            logging.warning(f"No play-by-play logs found for game ID {game_id}")
            return []

        def duration_to_seconds(duration_str):
            """
            Convert a duration string to seconds.

            Parameters:
            duration_str (str): Duration in 'PTXMYS' format.

            Returns:
            int: Duration in seconds.
            """
            minutes = int(duration_str.split("M")[0][2:])
            seconds = float(duration_str.split("M")[1][:-1])
            return minutes * 60 + seconds

        # Sort play-by-play logs by period, remaining time (clock), and play ID
        pbp_logs = sorted(
            pbp_logs,
            key=lambda x: (
                x["period"],
                -duration_to_seconds(x["clock"]),
                x["orderNumber"],
            ),
        )

        # Filter out logs where 'description' is not a key
        pbp_logs = [log for log in pbp_logs if "description" in log]

        game_states = []
        players = {"home": {}, "away": {}}

        for row in pbp_logs:
            if row.get("personId") is not None and row.get("playerNameI") is not None:
                team = "home" if row["teamTricode"] == home else "away"
                player_id = row["personId"]
                player_name = row["playerNameI"]

                if player_id not in players[team]:
                    players[team][player_id] = {"name": player_name, "points": 0}

                if row.get("pointsTotal") is not None:
                    points = int(row["pointsTotal"])
                    players[team][player_id]["points"] = points

            current_game_state = {
                "game_id": game_id,
                "play_id": int(row["orderNumber"]),
                "game_date": game_date,
                "home": home,
                "away": away,
                "clock": row["clock"],
                "period": int(row["period"]),
                "home_score": int(row["scoreHome"]),
                "away_score": int(row["scoreAway"]),
                "total": int(row["scoreHome"]) + int(row["scoreAway"]),
                "home_margin": int(row["scoreHome"]) - int(row["scoreAway"]),
                "is_final_state": row["description"] == "Game End",
                "players_data": deepcopy(players),
            }

            game_states.append(current_game_state)

        return game_states

    except Exception as e:
        logging.error(f"Game ID {game_id} - Failed to create game states: {e}")
        return []


def save_game_states(game_states, db_path):
    """
    Saves the game states to the database and updates Games.game_data_finalized to True if any state has is_final_state = True.
    Each game_id is processed in a separate transaction to ensure all-or-nothing behavior.

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
                        f"Game ID {game_id} - No game states to save. Skipping."
                    )
                    continue

                try:
                    conn.execute("BEGIN")
                    data_to_insert = [
                        (
                            game_id,
                            state["play_id"],
                            state["game_date"],
                            state["home"],
                            state["away"],
                            state["clock"],
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

                    conn.executemany(
                        """
                        INSERT OR REPLACE INTO GameStates (game_id, play_id, game_date, home, away, clock, period, home_score, away_score, total, home_margin, is_final_state, players_data)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        data_to_insert,
                    )

                    # Check if any state has is_final_state = True and update Games table accordingly
                    if any(state["is_final_state"] for state in states):
                        conn.execute(
                            """
                            UPDATE Games SET game_data_finalized = 1 WHERE game_id = ?
                            """,
                            (game_id,),
                        )

                    conn.commit()  # Commit the transaction if no errors occurred
                except Exception as e:
                    conn.rollback()  # Roll back the transaction if an error occurred
                    logging.error(f"Game ID {game_id} - Error saving game states: {e}")
                    overall_success = False  # Mark the overall operation as failed, but continue processing other game IDs

    except Exception as e:
        logging.error(f"Database connection error: {e}")
        return False  # Return False immediately if a database connection error occurred

    return overall_success  # Return True if the operation was successful for all game IDs, False otherwise
