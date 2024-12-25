"""
boxscores.py

This module loads traditional boxscores into PlayerBox and TeamBox tables for any game.

Key Features:
- Load boxscore data for a specific game.
- Load boxscore data for a specific date.
- Initial load of boxscore data for all games within a date range.
- Update boxscore data for games that are in progress or recently completed.
- Saves errors encountered during the initial load to a CSV file.

Command Line Interface (CLI):
    python -m src.database_updater_CM.boxscores --task=single_game --game_id=GAME_ID --save_to_db
    python -m src.database_updater_CM.boxscores --task=single_date --date_str=YYYY-MM-DD --save_to_db
    python -m src.database_updater_CM.boxscores --task=initial_load --save_to_db
    python -m src.database_updater_CM.boxscores --task=update --save_to_db

Arguments:
- --task: The operation to perform (single_game, single_date, initial_load, update).
- --game_id: Required if task=single_game. The ID of the game to load boxscore data for.
- --date_str: Required if task=single_date. The date in YYYY-MM-DD format to load boxscore data for.
- --save_to_db: If passed, saves data to the database.

Example:
    python -m src.database_updater_CM.boxscores --task=single_game --game_id=0022100001 --save_to_db
"""

import argparse
import csv
import logging
import sqlite3
import time
from datetime import datetime, timedelta

from nba_api.live.nba.endpoints import boxscore as LiveBoxScore
from nba_api.stats.endpoints import BoxScoreTraditionalV3
from tqdm import tqdm

from src.config import config
from src.database_updater_CM.games import (
    date_to_season,
    get_current_season,
    get_initial_load_date_range,
)
from src.logging_config_CM import setup_logging

DB_PATH = config["database"]["path"]

SLEEP_DURATION = 0.5

setup_logging(
    log_level="INFO",
    log_file="custom_model.log",
    structured=True,
)
logger = logging.getLogger("boxscores")


# Utility Functions
def convert_minutes_to_float(min_str):
    """
    Convert a time string in "MM:SS" format to a float representing total minutes.

    Args:
        min_str (str): Time string in "MM:SS" format.

    Returns:
        float: Total minutes as a float.
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


def parse_stats_boxscore(json_data):
    """
    Parse the stats endpoint (BoxScoreTraditionalV3) response for a single game.
    Extracts player and team records according to PlayerBox and TeamBox schemas.

    Args:
        json_data (dict): Dictionary returned by BoxScoreTraditionalV3().get_dict()

    Returns:
        tuple: Two lists containing player records and team records.
    """

    # According to V3 structure:
    # json_data["boxScoreTraditional"]["homeTeam"] and json_data["boxScoreTraditional"]["awayTeam"] contain team and player stats.
    home = json_data["boxScoreTraditional"]["homeTeam"]
    away = json_data["boxScoreTraditional"]["awayTeam"]

    def parse_team(t):
        return (
            t["teamId"],
            json_data["boxScoreTraditional"]["gameId"],
            t["statistics"]["points"],
            None,  # pts_allowed computed after both teams processed
            t["statistics"]["reboundsTotal"],
            t["statistics"]["assists"],
            t["statistics"]["steals"],
            t["statistics"]["blocks"],
            t["statistics"]["turnovers"],
            t["statistics"]["foulsPersonal"],
            t["statistics"]["fieldGoalsAttempted"],
            t["statistics"]["fieldGoalsMade"],
            t["statistics"]["fieldGoalsPercentage"],
            t["statistics"]["threePointersAttempted"],
            t["statistics"]["threePointersMade"],
            t["statistics"]["threePointersPercentage"],
            t["statistics"]["freeThrowsAttempted"],
            t["statistics"]["freeThrowsMade"],
            t["statistics"]["freeThrowsPercentage"],
            t["statistics"].get("plusMinusPoints", None),
        )

    def parse_player(p, team_id):
        minutes_str = p["statistics"].get("minutes", "")
        min_val = convert_minutes_to_float(minutes_str)
        return (
            p["personId"],
            json_data["boxScoreTraditional"]["gameId"],
            team_id,
            f"{p['firstName']} {p['familyName']}",
            p.get("position", ""),  # Include position
            min_val,
            p["statistics"].get("points", None),
            p["statistics"].get("reboundsTotal", None),
            p["statistics"].get("assists", None),
            p["statistics"].get("steals", None),
            p["statistics"].get("blocks", None),
            p["statistics"].get("turnovers", None),
            p["statistics"].get("foulsPersonal", None),
            p["statistics"].get("reboundsOffensive", None),
            p["statistics"].get("reboundsDefensive", None),
            p["statistics"].get("fieldGoalsAttempted", None),
            p["statistics"].get("fieldGoalsMade", None),
            p["statistics"].get("fieldGoalsPercentage", None),
            p["statistics"].get("threePointersAttempted", None),
            p["statistics"].get("threePointersMade", None),
            p["statistics"].get("threePointersPercentage", None),
            p["statistics"].get("freeThrowsAttempted", None),
            p["statistics"].get("freeThrowsMade", None),
            p["statistics"].get("freeThrowsPercentage", None),
            p["statistics"].get("plusMinusPoints", None),
        )

    # Parse teams
    home_team_record = parse_team(home)
    away_team_record = parse_team(away)

    # Compute pts_allowed
    home_list = list(home_team_record)
    away_list = list(away_team_record)
    home_pts = home_list[2]
    away_pts = away_list[2]
    home_list[3] = away_pts  # pts_allowed for home
    away_list[3] = home_pts  # pts_allowed for away
    home_team_record = tuple(home_list)
    away_team_record = tuple(away_list)

    team_records = [home_team_record, away_team_record]

    # Parse players
    player_records = []
    for player in home["players"]:
        player_records.append(parse_player(player, home["teamId"]))
    for player in away["players"]:
        player_records.append(parse_player(player, away["teamId"]))

    return player_records, team_records


def parse_live_boxscore(live_data, game_id):
    """
    Parse the live endpoint's boxscore data and return team_records, player_records.
    Assumes structure based on nba_api live boxscore documentation.

    live_data example structure:
    live_data["game"]["homeTeam"] and live_data["game"]["awayTeam"] contain team and player stats.

    Adjust logic as needed if actual structure differs.

    Args:
        live_data (dict): JSON data from the live BoxScore endpoint.
        game_id (str): The game ID.

    Returns:
        tuple: Two lists containing player records and team records.
    """

    def parse_team(t):
        return (
            t["teamId"],
            game_id,
            t["score"],
            None,  # pts_allowed computed after both teams processed
            t["statistics"]["reboundsTotal"],
            t["statistics"]["assists"],
            t["statistics"]["steals"],
            t["statistics"]["blocks"],
            t["statistics"]["turnovers"],
            t["statistics"]["foulsPersonal"],
            t["statistics"]["fieldGoalsAttempted"],
            t["statistics"]["fieldGoalsMade"],
            t["statistics"]["fieldGoalsPercentage"],
            t["statistics"]["threePointersAttempted"],
            t["statistics"]["threePointersMade"],
            t["statistics"]["threePointersPercentage"],
            t["statistics"]["freeThrowsAttempted"],
            t["statistics"]["freeThrowsMade"],
            t["statistics"]["freeThrowsPercentage"],
            t["statistics"].get("plusMinusPoints", None),
        )

    def parse_player(p, team_id):
        minutes_str = p["statistics"].get("minutes", "")
        min_val = convert_minutes_to_float(minutes_str)
        return (
            p["personId"],
            game_id,
            team_id,
            p["name"],
            p.get("position", ""),
            min_val,
            p["statistics"].get("points", None),
            p["statistics"].get("reboundsTotal", None),
            p["statistics"].get("assists", None),
            p["statistics"].get("steals", None),
            p["statistics"].get("blocks", None),
            p["statistics"].get("turnovers", None),
            p["statistics"].get("foulsPersonal", None),
            p["statistics"].get("reboundsOffensive", None),
            p["statistics"].get("reboundsDefensive", None),
            p["statistics"].get("fieldGoalsAttempted", None),
            p["statistics"].get("fieldGoalsMade", None),
            p["statistics"].get("fieldGoalsPercentage", None),
            p["statistics"].get("threePointersAttempted", None),
            p["statistics"].get("threePointersMade", None),
            p["statistics"].get("threePointersPercentage", None),
            p["statistics"].get("freeThrowsAttempted", None),
            p["statistics"].get("freeThrowsMade", None),
            p["statistics"].get("freeThrowsPercentage", None),
            p["statistics"].get("plusMinusPoints", None),
        )

    home = live_data["game"]["homeTeam"]
    away = live_data["game"]["awayTeam"]

    home_team_record = parse_team(home)
    away_team_record = parse_team(away)

    # Compute pts_allowed
    home_list = list(home_team_record)
    away_list = list(away_team_record)
    home_pts = home_list[2]
    away_pts = away_list[2]
    home_list[3] = away_pts  # pts_allowed for home
    away_list[3] = home_pts  # pts_allowed for away
    home_team_record = tuple(home_list)
    away_team_record = tuple(away_list)

    team_records = [home_team_record, away_team_record]

    player_records = []
    for player in home["players"]:
        player_records.append(parse_player(player, home["teamId"]))
    for player in away["players"]:
        player_records.append(parse_player(player, away["teamId"]))

    return player_records, team_records


def insert_boxscores(conn, player_records, team_records, save_to_db=True):
    """
    Insert or update boxscore records in the database.

    Args:
        conn (sqlite3.Connection): The database connection.
        player_records (list): List of player records to insert/update.
        team_records (list): List of team records to insert/update.
        save_to_db (bool): Whether to commit the changes to the database.
    """
    if save_to_db:
        player_sql = """
        INSERT INTO PlayerBox (
            player_id, game_id, team_id, player_name, position, min, pts, reb, ast,
            stl, blk, tov, pf, oreb, dreb, fga, fgm, fg_pct, fg3a, fg3m, fg3_pct, fta, ftm, ft_pct, plus_minus
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(player_id, game_id) DO UPDATE SET
            team_id=excluded.team_id,
            player_name=excluded.player_name,
            position=excluded.position,
            min=excluded.min,
            pts=excluded.pts,
            reb=excluded.reb,
            ast=excluded.ast,
            stl=excluded.stl,
            blk=excluded.blk,
            tov=excluded.tov,
            pf=excluded.pf,
            oreb=excluded.oreb,
            dreb=excluded.dreb,
            fga=excluded.fga,
            fgm=excluded.fgm,
            fg_pct=excluded.fg_pct,
            fg3a=excluded.fg3a,
            fg3m=excluded.fg3m,
            fg3_pct=excluded.fg3_pct,
            fta=excluded.fta,
            ftm=excluded.ftm,
            ft_pct=excluded.ft_pct,
            plus_minus=excluded.plus_minus
        """

        team_sql = """
        INSERT INTO TeamBox (
            team_id, game_id, pts, pts_allowed, reb, ast, stl, blk, tov, pf,
            fga, fgm, fg_pct,
            fg3a, fg3m, fg3_pct,
            fta, ftm, ft_pct,
            plus_minus
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(team_id, game_id) DO UPDATE SET
            pts=excluded.pts,
            pts_allowed=excluded.pts_allowed,
            reb=excluded.reb,
            ast=excluded.ast,
            stl=excluded.stl,
            blk=excluded.blk,
            tov=excluded.tov,
            pf=excluded.pf,
            fga=excluded.fga,
            fgm=excluded.fgm,
            fg_pct=excluded.fg_pct,
            fg3a=excluded.fg3a,
            fg3m=excluded.fg3m,
            fg3_pct=excluded.fg3_pct,
            fta=excluded.fta,
            ftm=excluded.ftm,
            ft_pct=excluded.ft_pct,
            plus_minus=excluded.plus_minus
        """

        if team_records:
            conn.executemany(team_sql, team_records)
        if player_records:
            conn.executemany(player_sql, player_records)

        conn.commit()


# Load Functions
def load_boxscore_for_game(conn, game_id, save_to_db=True, use_live_endpoint=False):
    """
    Load boxscore data for a specific game and insert/update in the database.

    Args:
        conn (sqlite3.Connection): The database connection.
        game_id (str): The game ID.
        save_to_db (bool): Whether to commit the changes to the database.
        use_live_endpoint (bool): Whether to use the live endpoint for fetching data.
    """
    logger.debug(
        f"Loading boxscore for game_id: {game_id} using {'live' if use_live_endpoint else 'stats'} endpoint"
    )
    try:
        if use_live_endpoint:
            boxscore = LiveBoxScore.BoxScore(game_id=game_id).get_dict()
            player_records, team_records = parse_live_boxscore(boxscore, game_id)
        else:
            boxscore = BoxScoreTraditionalV3(game_id=game_id).get_dict()
            player_records, team_records = parse_stats_boxscore(boxscore)
        insert_boxscores(conn, player_records, team_records, save_to_db=save_to_db)
        logger.debug(f"Upserted boxscore for game_id {game_id}.")

        # Update boxscores_finalized flag if conditions are met
        cursor = conn.execute(
            "SELECT status, date_time_est FROM Games WHERE game_id=?", (game_id,)
        )
        row = cursor.fetchone()
        if row:
            status, date_time_est_str = row
            date_time_est = (
                datetime.fromisoformat(date_time_est_str.replace("Z", ""))
                if "Z" in date_time_est_str
                else datetime.fromisoformat(date_time_est_str)
            )
            now = datetime.now()
            if status == "Completed" and now - date_time_est > timedelta(hours=24):
                conn.execute(
                    "UPDATE Games SET boxscores_finalized=1 WHERE game_id=?", (game_id,)
                )
                conn.commit()
                logger.debug(f"Set boxscores_finalized=1 for game_id {game_id}")
    except Exception as e:
        logger.error(f"Error fetching boxscore for game_id {game_id}: {e}")
        time.sleep(2)
        try:
            if use_live_endpoint:
                boxscore = LiveBoxScore.BoxScore(game_id=game_id).get_dict()
                player_records, team_records = parse_live_boxscore(boxscore, game_id)
            else:
                boxscore = BoxScoreTraditionalV3(game_id=game_id).get_dict()
                player_records, team_records = parse_stats_boxscore(boxscore)
            insert_boxscores(conn, player_records, team_records, save_to_db=save_to_db)
            logger.info(f"Retry succeeded: Upserted boxscore for game_id {game_id}.")
        except Exception as e:
            logger.error(
                f"Retry failed for fetching boxscore for game_id {game_id}: {e}"
            )
            raise


def load_boxscores_for_date(conn, date_str, save_to_db=True, use_live_endpoint=False):
    """
    Load boxscore data for all games on a specific date and insert/update in the database.

    Args:
        conn (sqlite3.Connection): The database connection.
        date_str (str): The date in YYYY-MM-DD format.
        save_to_db (bool): Whether to commit the changes to the database.
        use_live_endpoint (bool): Whether to use the live endpoint for fetching data.

    Returns:
        tuple: Number of records inserted and dictionary of error game IDs by date.
    """
    logger.debug(f"Loading boxscores for date: {date_str}")
    cursor = conn.execute(
        "SELECT game_id FROM Games WHERE date(date_time_est)=date(?)", (date_str,)
    )
    games = [row[0] for row in cursor.fetchall()]

    inserted_count = 0
    error_dates = {}

    if not games:
        logger.debug(f"No games found for {date_str}.")
        return (0, error_dates)

    for game_id in games:
        logger.debug(f"Fetching boxscore for game_id: {game_id}")
        try:
            load_boxscore_for_game(
                conn,
                game_id,
                save_to_db=save_to_db,
                use_live_endpoint=use_live_endpoint,
            )
            inserted_count += 1
        except Exception:
            logger.error(f"Error fetching boxscore for game_id {game_id}")
            if date_str not in error_dates:
                error_dates[date_str] = []
            error_dates[date_str].append(game_id)
        time.sleep(SLEEP_DURATION)

    return (inserted_count, error_dates)


def load_initial_boxscores(conn, save_to_db=True):
    """
    Perform an initial load of boxscore data for all games within a date range.

    Args:
        conn (sqlite3.Connection): The database connection.
        save_to_db (bool): Whether to commit the changes to the database.
    """
    logger.info(
        "Starting initial boxscore load with seasonal and date-level monitoring..."
    )

    start_date, end_date = get_initial_load_date_range()
    end_date = min(
        end_date, datetime.now().date()
    )  # Ensure end_date is not in the future

    # Determine seasons
    seasons = {}
    current_date = start_date
    while current_date <= end_date:
        s = date_to_season(current_date.strftime("%Y-%m-%d"))
        if s not in seasons:
            seasons[s] = []
        seasons[s].append(current_date)
        current_date += timedelta(days=1)

    all_season_keys = sorted(seasons.keys(), reverse=True)  # Process newest to oldest
    total_seasons = len(all_season_keys)

    overall_games_inserted = 0
    overall_errors_count = 0
    all_error_dates = {}

    for idx, season_str in enumerate(all_season_keys, start=1):
        logger.info(f"Processing season {season_str} ({idx}/{total_seasons})")

        season_games_inserted = 0
        errors_count = 0
        error_dates = {}

        season_dates = seasons[season_str]

        with tqdm(
            total=len(season_dates), desc=f"Dates in {season_str}", unit="day"
        ) as pbar:
            for day_date in reversed(season_dates):  # Process newest to oldest
                game_date_str = day_date.strftime("%Y-%m-%d")
                pbar.set_postfix({"date": game_date_str})
                inserted, date_errors = load_boxscores_for_date(
                    conn, game_date_str, save_to_db=save_to_db
                )

                season_games_inserted += inserted
                if date_errors:
                    errors_count += len(date_errors)
                    error_dates.update(date_errors)

                pbar.update(1)
                time.sleep(SLEEP_DURATION)

        # End of the season, log a final summary
        logger.info(
            f"Completed loading season {season_str}. Inserted/Upserted {season_games_inserted} boxscores. "
            f"Errors: {errors_count}"
        )
        if error_dates:
            for date, game_ids in error_dates.items():
                logger.error(f"Errors on {date} for game IDs: {', '.join(game_ids)}")
            all_error_dates.update(error_dates)

        overall_games_inserted += season_games_inserted
        overall_errors_count += errors_count

    # Save errors to CSV
    if all_error_dates:
        with open("boxscore_load_errors.csv", mode="w", newline="") as csv_file:
            fieldnames = ["date", "game_id"]
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for date, game_ids in all_error_dates.items():
                for game_id in game_ids:
                    writer.writerow({"date": date, "game_id": game_id})

    logger.info(
        f"Initial load complete. Total boxscores inserted/updated: {overall_games_inserted}, Total errors: {overall_errors_count}"
    )


def update_boxscores(conn, save_to_db=True):
    """
    Update boxscore data for games that are in progress or recently completed.

    Args:
        conn (sqlite3.Connection): The database connection.
        save_to_db (bool): Whether to commit the changes to the database.
    """
    logger.info("Starting update of boxscores...")

    # Query games to update
    current_season = get_current_season()
    cursor = conn.execute(
        """
        SELECT game_id, status, date_time_est FROM Games
        WHERE (status='In Progress' OR status='Completed')
        AND boxscores_finalized=0
        AND season IN (?, ?)
    """,
        (current_season, str(int(current_season[:4]) - 1) + "-" + current_season[:4]),
    )
    rows = cursor.fetchall()

    if not rows:
        logger.info("No games require an update.")
        return

    updated_count = 0
    errors_count = 0
    error_dates = {}

    with tqdm(total=len(rows), desc="Updating boxscores", unit="game") as pbar:
        for game_id, status, date_time_est_str in rows:
            date_time_est = (
                datetime.fromisoformat(date_time_est_str.replace("Z", ""))
                if "Z" in date_time_est_str
                else datetime.fromisoformat(date_time_est_str)
            )
            try:
                if status == "In Progress":
                    # Use live endpoint
                    logger.debug(f"Fetching live boxscore for game_id: {game_id}")
                    load_boxscore_for_game(
                        conn, game_id, save_to_db=save_to_db, use_live_endpoint=True
                    )
                elif status == "Completed":
                    # Use stats endpoint
                    logger.debug(
                        f"Fetching stats boxscore for completed game_id: {game_id}"
                    )
                    load_boxscore_for_game(
                        conn, game_id, save_to_db=save_to_db, use_live_endpoint=False
                    )
                updated_count += 1
            except Exception as e:
                logger.error(f"Error updating boxscore for game_id {game_id}: {e}")
                errors_count += 1
                date_str = date_time_est.strftime("%Y-%m-%d")
                if date_str not in error_dates:
                    error_dates[date_str] = []
                error_dates[date_str].append(game_id)

            pbar.update(1)
            time.sleep(SLEEP_DURATION)

    if error_dates:
        for date, game_ids in error_dates.items():
            logger.error(f"Errors on {date} for game IDs: {', '.join(game_ids)}")

    logger.info(
        f"Update boxscores process completed. Updated: {updated_count}, Errors: {errors_count}"
    )


# Main Execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Boxscore data management")
    parser.add_argument(
        "--task",
        choices=["single_game", "single_date", "initial_load", "update"],
        required=True,
        help="The task to complete: 'single_game', 'single_date', 'initial_load', 'update'.",
    )
    parser.add_argument(
        "--game_id", type=str, help="Game ID required if task=single_game."
    )
    parser.add_argument(
        "--date_str",
        type=str,
        help="Date in YYYY-MM-DD format required if task=single_date.",
    )
    parser.add_argument(
        "--save_to_db",
        action="store_true",
        help="If set, commits data to the database.",
    )
    args = parser.parse_args()

    with sqlite3.connect(DB_PATH) as conn:
        if args.task == "single_game":
            if not args.game_id:
                raise ValueError("--game_id is required when task=single_game")
            load_boxscore_for_game(conn, args.game_id, save_to_db=args.save_to_db)
        elif args.task == "single_date":
            if not args.date_str:
                raise ValueError("--date_str is required when task=single_date")
            load_boxscores_for_date(conn, args.date_str, save_to_db=args.save_to_db)
        elif args.task == "initial_load":
            load_initial_boxscores(conn, save_to_db=args.save_to_db)
        elif args.task == "update":
            update_boxscores(conn, save_to_db=args.save_to_db)

    logger.info("Boxscore data management tasks completed.")
