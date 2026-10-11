"""
boxscores.py

This module collects player and team boxscore statistics from NBA API.
Adapted from database_updater_CM to work with TEXT game_id schema.

Enhanced with Sprint 15 standardization:
- ThreadPoolExecutor for concurrent API calls
- StageLogger integration for tracking
- Live endpoint support for in-progress games
- Better error handling and retry logic
- Standardized logging format

Functions:
- get_boxscores(game_ids): Fetch boxscore data for a list of game IDs
- save_boxscores(boxscore_data, db_path): Save boxscore data to PlayerBox and TeamBox tables
- parse_live_boxscore(live_data, game_id): Parse live endpoint response
"""

import logging
import os
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Suppress urllib3 connection pool warnings for cleaner output
logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from nba_api.live.nba.endpoints import boxscore as LiveBoxScore
from nba_api.stats.endpoints import BoxScoreTraditionalV3
from tqdm import tqdm

from src.config import config
from src.database import get_db
from src.utils import StageLogger, log_execution_time

DB_PATH = config["database"]["path"]


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
        error_str = str(e).lower()
        # Don't retry on permanent errors (404, 400-level, invalid game)
        if any(code in error_str for code in ("404", "400", "403", "not found")):
            logging.warning(f"Permanent error for {game_id}, skipping retry: {e}")
            raise

        # Retry once with backoff for transient errors (429, 500, timeout, connection)
        logging.warning(f"Transient error for {game_id}, retrying in 3s: {e}")
        time.sleep(3)
        try:
            boxscore = BoxScoreTraditionalV3(game_id=game_id).get_dict()
            return parse_boxscore_response(boxscore, game_id)
        except Exception as retry_err:
            logging.error(f"Retry failed for {game_id}: {retry_err}")
            raise


@log_execution_time(average_over="game_ids")
def get_boxscores(
    game_ids: List[str],
    check_game_status: bool = False,
    stage_logger: Optional[StageLogger] = None,
    db_path=DB_PATH,
) -> Dict[str, Tuple[List[dict], List[dict]]]:
    """
    Fetch boxscore data for multiple games from NBA API using concurrent requests.

    Args:
        game_ids (list): List of game IDs to fetch
        check_game_status (bool): If True, checks game status to determine endpoint
        stage_logger (StageLogger): Optional logger for tracking API calls
        db_path (str): Database path to check game status

    Returns:
        dict: {game_id: (player_records, team_records)}
    """
    logging.debug(f"Fetching boxscores for {len(game_ids)} games...")

    # Get game statuses if requested
    game_statuses = {}
    if check_game_status:
        with get_db(db_path) as conn:
            cursor = conn.cursor()
            for game_id in game_ids:
                cursor.execute("SELECT status FROM Games WHERE game_id=?", (game_id,))
                row = cursor.fetchone()
                if row:
                    game_statuses[game_id] = row[0]

    results = {}
    n = len(game_ids)

    # Daily updates (<=15 games): use threading for speed
    # Catch-up mode (>15 games): sequential with rate-limit pauses
    if n <= 15:
        thread_pool_size = min(3, os.cpu_count())
        with ThreadPoolExecutor(max_workers=thread_pool_size) as executor:
            futures = {
                executor.submit(
                    fetch_single_boxscore,
                    gid,
                    game_statuses.get(gid) == 2,
                ): gid
                for gid in game_ids
            }
            with tqdm(
                total=n, desc="Fetching boxscores", unit="game", leave=False
            ) as pbar:
                for future in as_completed(futures):
                    game_id, player_records, team_records = future.result()
                    results[game_id] = (player_records, team_records)
                    if stage_logger:
                        stage_logger.log_api_call()
                    pbar.update(1)
    else:
        # Sequential with adaptive rate limiting to avoid NBA API throttling.
        # The API consistently blocks after ~15 rapid requests, so we pause
        # every 10 requests to stay well under the threshold.
        burst_size = 10
        cooldown = 10  # seconds between bursts
        logging.info(
            f"Catch-up mode: {n} games, fetching {burst_size} at a time "
            f"with {cooldown}s cooldown"
        )
        failures = 0
        with tqdm(
            total=n, desc="Fetching boxscores", unit="game", leave=False
        ) as pbar:
            for i, gid in enumerate(game_ids):
                game_id, player_records, team_records = fetch_single_boxscore(
                    gid, game_statuses.get(gid) == 2
                )
                results[game_id] = (player_records, team_records)
                if stage_logger:
                    stage_logger.log_api_call()
                pbar.update(1)
                # Consecutive failures mean the API has started throttling, which
                # lasts 30-60+ minutes; stop instead of hanging on timeouts. Games
                # left out of the results are fetched on the next run.
                failures = 0 if (player_records or team_records) else failures + 1
                if failures == 3:
                    logging.warning(
                        "Boxscore catch-up stopped after 3 consecutive failures "
                        "(likely NBA API throttling)"
                    )
                    break
                # Cooldown between bursts to stay under rate limits
                if (i + 1) % burst_size == 0 and i + 1 < n:
                    time.sleep(cooldown)

    successful_count = sum(1 for data in results.values() if data[0] or data[1])
    failed_count = len(game_ids) - successful_count

    if failed_count > 0:
        logging.warning(
            f"Boxscore collection: {failed_count}/{len(game_ids)} games returned no data"
        )
    else:
        logging.debug(f"Fetched {successful_count} boxscores successfully")
    return results


def fetch_single_boxscore(
    game_id: str, use_live: bool = False
) -> Tuple[str, List[dict], List[dict]]:
    """
    Fetch boxscore data for a single game with fallback logic.

    Args:
        game_id (str): Game ID to fetch
        use_live (bool): Whether to try live endpoint first

    Returns:
        tuple: (game_id, player_records, team_records)
    """
    try:
        # Fetch with fallback logic
        player_records, team_records = get_boxscore_with_fallback(
            game_id, use_live=use_live
        )
        return game_id, player_records, team_records
    except Exception as e:
        logging.warning(f"Error fetching boxscore for game {game_id}: {e}")
        return game_id, [], []


@log_execution_time(average_over="boxscore_data")
def save_boxscores(
    boxscore_data: Dict[str, Tuple[List[dict], List[dict]]], db_path=DB_PATH
):
    """
    Save boxscore data to PlayerBox and TeamBox tables and update fetch timestamps.

    Args:
        boxscore_data (dict): {game_id: (player_records, team_records)}
        db_path (str): Path to database

    Returns:
        dict: Dictionary with counts {"added": X, "updated": Y}
    """
    import sqlite3

    logging.debug(f"Saving boxscores for {len(boxscore_data)} games...")

    with get_db(db_path) as conn:
        cursor = conn.cursor()

        # Auto-migration: Add boxscore_last_fetched_at column if it doesn't exist
        try:
            cursor.execute("ALTER TABLE Games ADD COLUMN boxscore_last_fetched_at TEXT")
            logging.debug("Added boxscore_last_fetched_at column to Games table")
        except sqlite3.OperationalError:
            # Column already exists
            pass

        total_players = 0
        total_teams = 0
        added_games = 0
        updated_games = 0

        try:
            for game_id, (player_records, team_records) in boxscore_data.items():
                # Check if this game already has boxscore data
                cursor.execute(
                    "SELECT COUNT(*) FROM PlayerBox WHERE game_id = ?", (game_id,)
                )
                existing_count = cursor.fetchone()[0]

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

                # Update boxscore_last_fetched_at timestamp
                cursor.execute(
                    "UPDATE Games SET boxscore_last_fetched_at = datetime('now') WHERE game_id = ?",
                    (game_id,),
                )

                # Track added vs updated
                if existing_count == 0:
                    added_games += 1
                else:
                    updated_games += 1

            conn.commit()
            logging.debug(
                f"Saved {total_players} player records and {total_teams} team records"
            )

            return {"added": added_games, "updated": updated_games}

        except Exception as e:
            conn.rollback()
            logging.error(f"Error saving boxscores: {e}")
            raise
