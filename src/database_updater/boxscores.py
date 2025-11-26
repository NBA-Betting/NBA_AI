"""
boxscores.py

This module collects player and team boxscore statistics from NBA API.
Adapted from database_updater_CM to work with TEXT game_id schema.

Enhanced with CM features:
- Live endpoint support for in-progress games
- Better error handling and retry logic
- CSV error logging for tracking failed collections

Functions:
- get_boxscores(game_ids): Fetch boxscore data for a list of game IDs
- save_boxscores(boxscore_data, db_path): Save boxscore data to PlayerBox and TeamBox tables
- parse_live_boxscore(live_data, game_id): Parse live endpoint response
"""

import csv
import logging
import sqlite3
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

from nba_api.live.nba.endpoints import boxscore as LiveBoxScore
from nba_api.stats.endpoints import BoxScoreTraditionalV3
from tqdm import tqdm

from src.config import config
from src.utils import log_execution_time

DB_PATH = config["database"]["path"]
SLEEP_DURATION = 0.6  # Rate limiting


def convert_minutes_to_float(min_str):
    """
    Convert a time string in "MM:SS" format to a float representing total minutes.

    Args:
        min_str (str): Time string in "MM:SS" format.

    Returns:
        float: Total minutes as a float, or None if invalid.
    """
    if not min_str or min_str.strip() == "":
        return None
    parts = min_str.split(":")
    if len(parts) == 2:
        try:
            minutes = int(parts[0])
            seconds = int(parts[1])
            return minutes + seconds / 60.0
        except ValueError:
            return None
    else:
        try:
            return float(min_str)
        except ValueError:
            return None


def parse_boxscore_response(json_data, game_id: str) -> Tuple[List[dict], List[dict]]:
    """
    Parse BoxScoreTraditionalV3 response and extract player and team records.

    Args:
        json_data (dict): Response from BoxScoreTraditionalV3
        game_id (str): Game ID (TEXT format)

    Returns:
        tuple: (player_records, team_records)
    """
    player_records = []
    team_records = []

    if "boxScoreTraditional" not in json_data:
        logging.warning(f"No boxScoreTraditional in response for game {game_id}")
        return player_records, team_records

    for team_key in ["homeTeam", "awayTeam"]:
        team_data = json_data["boxScoreTraditional"][team_key]
        team_id = str(team_data["teamId"])  # Convert to TEXT
        team_tricode = team_data["teamTricode"]

        # Parse team stats
        stats = team_data.get("statistics", {})
        team_record = {
            "team_id": team_id,
            "game_id": game_id,
            "pts": stats.get("points"),
            "pts_allowed": None,  # Will need to calculate from opponent
            "reb": stats.get("reboundsTotal"),
            "ast": stats.get("assists"),
            "stl": stats.get("steals"),
            "blk": stats.get("blocks"),
            "tov": stats.get("turnovers"),
            "pf": stats.get("foulsPersonal"),
            "fga": stats.get("fieldGoalsAttempted"),
            "fgm": stats.get("fieldGoalsMade"),
            "fg_pct": stats.get("fieldGoalsPercentage"),
            "fg3a": stats.get("threePointersAttempted"),
            "fg3m": stats.get("threePointersMade"),
            "fg3_pct": stats.get("threePointersPercentage"),
            "fta": stats.get("freeThrowsAttempted"),
            "ftm": stats.get("freeThrowsMade"),
            "ft_pct": stats.get("freeThrowsPercentage"),
            "plus_minus": stats.get("plusMinusPoints"),
        }
        team_records.append(team_record)

        # Parse player stats
        for player in team_data.get("players", []):
            player_stats = player.get("statistics", {})

            player_record = {
                "player_id": player["personId"],
                "game_id": game_id,
                "team_id": team_id,
                "player_name": f"{player.get('firstName', '')} {player.get('familyName', '')}".strip(),
                "position": player.get("position"),
                "min": convert_minutes_to_float(player_stats.get("minutes")),
                "pts": player_stats.get("points"),
                "reb": player_stats.get("reboundsTotal"),
                "ast": player_stats.get("assists"),
                "stl": player_stats.get("steals"),
                "blk": player_stats.get("blocks"),
                "tov": player_stats.get("turnovers"),
                "pf": player_stats.get("foulsPersonal"),
                "oreb": player_stats.get("reboundsOffensive"),
                "dreb": player_stats.get("reboundsDefensive"),
                "fga": player_stats.get("fieldGoalsAttempted"),
                "fgm": player_stats.get("fieldGoalsMade"),
                "fg_pct": player_stats.get("fieldGoalsPercentage"),
                "fg3a": player_stats.get("threePointersAttempted"),
                "fg3m": player_stats.get("threePointersMade"),
                "fg3_pct": player_stats.get("threePointersPercentage"),
                "fta": player_stats.get("freeThrowsAttempted"),
                "ftm": player_stats.get("freeThrowsMade"),
                "ft_pct": player_stats.get("freeThrowsPercentage"),
                "plus_minus": player_stats.get("plusMinusPoints"),
            }
            player_records.append(player_record)

    # Calculate pts_allowed for each team
    if len(team_records) == 2:
        team_records[0]["pts_allowed"] = team_records[1]["pts"]
        team_records[1]["pts_allowed"] = team_records[0]["pts"]

    return player_records, team_records


def parse_live_boxscore(live_data, game_id: str) -> Tuple[List[dict], List[dict]]:
    """
    Parse live endpoint boxscore data for in-progress games.

    Args:
        live_data (dict): JSON data from nba_api.live.nba.endpoints.boxscore
        game_id (str): Game ID (TEXT format)

    Returns:
        tuple: (player_records, team_records)
    """
    player_records = []
    team_records = []

    if "game" not in live_data:
        logging.warning(f"No game data in live response for game {game_id}")
        return player_records, team_records

    game_data = live_data["game"]

    for team_key in ["homeTeam", "awayTeam"]:
        team_data = game_data[team_key]
        team_id = str(team_data["teamId"])

        # Parse team stats
        stats = team_data.get("statistics", {})
        team_record = {
            "team_id": team_id,
            "game_id": game_id,
            "pts": team_data.get("score", 0),
            "pts_allowed": None,  # Calculate after both teams
            "reb": stats.get("reboundsTotal"),
            "ast": stats.get("assists"),
            "stl": stats.get("steals"),
            "blk": stats.get("blocks"),
            "tov": stats.get("turnovers"),
            "pf": stats.get("foulsPersonal"),
            "fga": stats.get("fieldGoalsAttempted"),
            "fgm": stats.get("fieldGoalsMade"),
            "fg_pct": stats.get("fieldGoalsPercentage"),
            "fg3a": stats.get("threePointersAttempted"),
            "fg3m": stats.get("threePointersMade"),
            "fg3_pct": stats.get("threePointersPercentage"),
            "fta": stats.get("freeThrowsAttempted"),
            "ftm": stats.get("freeThrowsMade"),
            "ft_pct": stats.get("freeThrowsPercentage"),
            "plus_minus": stats.get("plusMinusPoints"),
        }
        team_records.append(team_record)

        # Parse player stats
        for player in team_data.get("players", []):
            player_stats = player.get("statistics", {})
            minutes_str = player_stats.get("minutes", "")
            min_val = convert_minutes_to_float(minutes_str)

            player_record = {
                "player_id": player["personId"],
                "game_id": game_id,
                "team_id": team_id,
                "player_name": player.get("name", ""),
                "position": player.get("position", ""),
                "min": min_val,
                "pts": player_stats.get("points"),
                "reb": player_stats.get("reboundsTotal"),
                "ast": player_stats.get("assists"),
                "stl": player_stats.get("steals"),
                "blk": player_stats.get("blocks"),
                "tov": player_stats.get("turnovers"),
                "pf": player_stats.get("foulsPersonal"),
                "oreb": player_stats.get("reboundsOffensive"),
                "dreb": player_stats.get("reboundsDefensive"),
                "fga": player_stats.get("fieldGoalsAttempted"),
                "fgm": player_stats.get("fieldGoalsMade"),
                "fg_pct": player_stats.get("fieldGoalsPercentage"),
                "fg3a": player_stats.get("threePointersAttempted"),
                "fg3m": player_stats.get("threePointersMade"),
                "fg3_pct": player_stats.get("threePointersPercentage"),
                "fta": player_stats.get("freeThrowsAttempted"),
                "ftm": player_stats.get("freeThrowsMade"),
                "ft_pct": player_stats.get("freeThrowsPercentage"),
                "plus_minus": player_stats.get("plusMinusPoints"),
            }
            player_records.append(player_record)

    # Calculate pts_allowed for each team
    if len(team_records) == 2:
        team_records[0]["pts_allowed"] = team_records[1]["pts"]
        team_records[1]["pts_allowed"] = team_records[0]["pts"]

    return player_records, team_records


def get_boxscore_with_fallback(
    game_id: str, use_live: bool = False
) -> Tuple[List[dict], List[dict]]:
    """
    Fetch boxscore for a single game with automatic endpoint selection and retry.

    Args:
        game_id (str): Game ID to fetch
        use_live (bool): Try live endpoint first (for in-progress games)

    Returns:
        tuple: (player_records, team_records)
    """
    try:
        if use_live:
            # Try live endpoint for in-progress games
            try:
                logging.debug(f"Fetching live boxscore for {game_id}")
                boxscore = LiveBoxScore.BoxScore(game_id=game_id).get_dict()
                return parse_live_boxscore(boxscore, game_id)
            except Exception as live_err:
                logging.debug(
                    f"Live endpoint failed for {game_id}, trying stats: {live_err}"
                )
                # Fall back to stats endpoint
                pass

        # Use stats endpoint (standard for completed games)
        logging.debug(f"Fetching stats boxscore for {game_id}")
        boxscore = BoxScoreTraditionalV3(game_id=game_id).get_dict()
        return parse_boxscore_response(boxscore, game_id)

    except Exception as e:
        # Retry once after delay
        logging.warning(f"Error fetching boxscore for {game_id}, retrying: {e}")
        time.sleep(2)
        try:
            boxscore = BoxScoreTraditionalV3(game_id=game_id).get_dict()
            return parse_boxscore_response(boxscore, game_id)
        except Exception as retry_err:
            logging.error(f"Retry failed for {game_id}: {retry_err}")
            raise


@log_execution_time(average_over="game_ids")
def get_boxscores(
    game_ids: List[str], check_game_status: bool = False, db_path=DB_PATH
) -> Dict[str, Tuple[List[dict], List[dict]]]:
    """
    Fetch boxscore data for multiple games from NBA API.

    Enhanced with:
    - Automatic endpoint selection (live vs stats) based on game status
    - Retry logic with exponential backoff
    - CSV error logging

    Args:
        game_ids (list): List of game IDs to fetch
        check_game_status (bool): If True, checks game status to determine endpoint
        db_path (str): Database path to check game status

    Returns:
        dict: {game_id: (player_records, team_records)}
    """
    logging.info(f"Fetching boxscores for {len(game_ids)} games...")

    boxscore_data = {}
    errors = []

    # Get game statuses if requested
    game_statuses = {}
    if check_game_status:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            for game_id in game_ids:
                cursor.execute("SELECT status FROM Games WHERE game_id=?", (game_id,))
                row = cursor.fetchone()
                if row:
                    game_statuses[game_id] = row[0]

    with tqdm(total=len(game_ids), desc="Fetching boxscores", unit="game") as pbar:
        for i, game_id in enumerate(game_ids):
            try:
                # Determine which endpoint to use
                use_live = game_statuses.get(game_id) == "In Progress"

                # Fetch with fallback logic
                player_records, team_records = get_boxscore_with_fallback(
                    game_id, use_live=use_live
                )
                boxscore_data[game_id] = (player_records, team_records)

                # Rate limiting
                if i < len(game_ids) - 1:
                    time.sleep(SLEEP_DURATION)

            except Exception as e:
                logging.warning(f"Error fetching boxscore for game {game_id}: {e}")
                errors.append((game_id, str(e)))
            finally:
                pbar.update(1)

    # Log errors to CSV if any occurred
    if errors:
        error_file = "boxscore_errors.csv"
        try:
            with open(error_file, mode="a", newline="") as csv_file:
                writer = csv.writer(csv_file)
                # Write header if file is new
                csv_file.seek(0, 2)  # Go to end
                if csv_file.tell() == 0:
                    writer.writerow(["timestamp", "game_id", "error"])
                # Write errors
                timestamp = datetime.now().isoformat()
                for game_id, error in errors:
                    writer.writerow([timestamp, game_id, error])
            logging.info(f"Logged {len(errors)} errors to {error_file}")
        except Exception as e:
            logging.error(f"Failed to write error CSV: {e}")

    logging.info(
        f"Successfully fetched {len(boxscore_data)} boxscores ({len(errors)} errors)"
    )
    return boxscore_data


@log_execution_time(average_over="boxscore_data")
def save_boxscores(
    boxscore_data: Dict[str, Tuple[List[dict], List[dict]]], db_path=DB_PATH
):
    """
    Save boxscore data to PlayerBox and TeamBox tables.

    Args:
        boxscore_data (dict): {game_id: (player_records, team_records)}
        db_path (str): Path to database
    """
    import sqlite3

    logging.info(f"Saving boxscores for {len(boxscore_data)} games...")

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()

        total_players = 0
        total_teams = 0

        try:
            for game_id, (player_records, team_records) in boxscore_data.items():
                # Save player records
                for player in player_records:
                    cursor.execute(
                        """
                        INSERT OR REPLACE INTO PlayerBox (
                        player_id, game_id, team_id, player_name, position,
                        min, pts, reb, ast, stl, blk, tov, pf,
                        oreb, dreb, fga, fgm, fg_pct,
                        fg3a, fg3m, fg3_pct,
                        fta, ftm, ft_pct, plus_minus
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                        (
                            player["player_id"],
                            player["game_id"],
                            player["team_id"],
                            player["player_name"],
                            player["position"],
                            player["min"],
                            player["pts"],
                            player["reb"],
                            player["ast"],
                            player["stl"],
                            player["blk"],
                            player["tov"],
                            player["pf"],
                            player["oreb"],
                            player["dreb"],
                            player["fga"],
                            player["fgm"],
                            player["fg_pct"],
                            player["fg3a"],
                            player["fg3m"],
                            player["fg3_pct"],
                            player["fta"],
                            player["ftm"],
                            player["ft_pct"],
                            player["plus_minus"],
                        ),
                    )
                    total_players += 1

                # Save team records
                for team in team_records:
                    cursor.execute(
                        """
                    INSERT OR REPLACE INTO TeamBox (
                        team_id, game_id, pts, pts_allowed, reb, ast, stl, blk, tov, pf,
                        fga, fgm, fg_pct, fg3a, fg3m, fg3_pct,
                        fta, ftm, ft_pct, plus_minus
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                        (
                            team["team_id"],
                            team["game_id"],
                            team["pts"],
                            team["pts_allowed"],
                            team["reb"],
                            team["ast"],
                            team["stl"],
                            team["blk"],
                            team["tov"],
                            team["pf"],
                            team["fga"],
                            team["fgm"],
                            team["fg_pct"],
                            team["fg3a"],
                            team["fg3m"],
                            team["fg3_pct"],
                            team["fta"],
                            team["ftm"],
                            team["ft_pct"],
                            team["plus_minus"],
                        ),
                    )
                    total_teams += 1

            conn.commit()
            logging.info(
                f"Saved {total_players} player records and {total_teams} team records"
            )

        except Exception as e:
            conn.rollback()
            logging.error(f"Error saving boxscores: {e}")
            raise
