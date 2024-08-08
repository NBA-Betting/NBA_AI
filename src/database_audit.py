"""
database_audit.py

This module conducts an audit of the database to identify issues with the data and save the results to a CSV file.
It consists of functions to:
- Verify the pre-game data finalized status for each game in the specified season.
- Verify the game data finalized status for each game in the specified season.
- Conduct an audit of the database to identify issues with the data and save the results to a CSV file.

Functions:
- get_pre_game_data(cursor, season): Verifies the pre-game data finalized status for each game in the specified season.
- get_game_data(cursor, season): Verifies the game data finalized status for each game in the specified season.
- get_log_data_issues(cursor, season): Verifies the PbP_Logs and GameStates record counts for each game_id.
- database_audit(season, output_file="audit_results.csv", db_path=DB_PATH): Conduct an audit of the database to identify issues with the data and save the results to a CSV file.
- print_summary_and_mismatches(season, final_df, pre_game_mismatches, game_data_mismatches, log_data_issues): Print a summary of the audit results and any mismatches found.

Usage:
- Script can be run directly from the command line (project root) to audit the database for a specific season:
    Set the season variable in the script (bottom of this file under __name__ == "__main__") and run:
    python -m src.database_audit
- Successful execution will print a summary of the audit results and any mismatches found, as well as save the results to a CSV file named 'audit_results.csv'.
"""

import sqlite3

import pandas as pd

from src.config import config
from src.database_updater.schedule import update_schedule

# Configuration
DB_PATH = config["database"]["path"]


def get_pre_game_data(cursor, season):
    """
    Verifies the pre-game data finalized status for each game in the specified season.
    "Is all required pre-game data finalized and available?"

    This function returns True or False for three different conditions:
    1. pre_game_flag: The current value of the pre_game_data_finalized flag in the Games table.
        The goal of the flag is to represent the state of the database.
    2. pre_game_db: Whether or not the database has finalized pre-game data,
        indicated by the presence of a feature set in the Features table and a final game state (is_final_state=1) in the GameStates table for each prior game involving one or both of the teams.
    3. pre_game_goal: Whether or not the database should have finalized pre-game data.
        This is indicated by all prior games involving one or both teams being marked as 'Completed' (assuming the schedule is up-to-date).

    Parameters:
        cursor (sqlite3.Cursor): Database cursor to execute the SQL query.
        season (str): The season to filter the games.

    Returns:
        list: A list of tuples containing the following information for each game:
            - game_id: The unique identifier of the game.
            - pre_game_flag: The current value of the pre_game_data_finalized flag in the Games table.
            - pre_game_db: Boolean indicating if the pre-game data requirements are met in the database.
            - pre_game_goal: Boolean indicating if all prior games involving one or both teams are 'Completed'.
    """
    cursor.execute(
        """
        SELECT g.game_id, g.pre_game_data_finalized AS pre_game_flag,
               EXISTS (
                   SELECT 1 FROM Features WHERE game_id = g.game_id
               ) AND EXISTS (
                   SELECT 1 FROM GameStates WHERE game_id = g.game_id AND is_final_state = 1
               ) AS pre_game_db,
               COUNT(pg.game_id) = 0 AS pre_game_goal
        FROM Games g
        LEFT JOIN Games pg ON (g.home_team = pg.home_team OR g.away_team = pg.away_team OR 
                               g.home_team = pg.away_team OR g.away_team = pg.home_team) 
                               AND pg.date_time_est < g.date_time_est
                               AND pg.season = g.season 
                               AND pg.season_type = g.season_type 
                               AND pg.status <> 'Completed'
        WHERE g.season = ? 
        AND g.season_type IN ('Regular Season', 'Post Season')
        GROUP BY g.game_id, g.pre_game_data_finalized
    """,
        (season,),
    )

    results = cursor.fetchall()
    return results


def get_game_data(cursor, season):
    """
    Verifies the game data finalized status for each game in the specified season.
    "Is all data (ex: PBP Logs and Game States) for a game final and available?"

    This function returns True or False for three different conditions:
    1. game_flag: The current value of the game_data_finalized flag in the Games table.
        The goal of the flag is to represent the state of the database.
    2. game_db: Whether or not the database has finalized game data
        indicated by the presence of a final game state (is_final_state=1) in the GameStates table.
    3. game_goal: Whether or not the database should have finalized game data.
        This is indicated by the game status being 'Completed' (assuming the schedule is up-to-date).

    Parameters:
        cursor (sqlite3.Cursor): Database cursor to execute the SQL query.
        season (str): The season to filter the games.

    Returns:
        list: A list of tuples containing the following information for each game:
            - game_id: The unique identifier of the game.
            - game_flag: The current value of the game_data_finalized flag in the Games table.
            - game_db: Boolean indicating if there is a final game state in the GameStates table.
            - game_goal: Boolean indicating if the game status is 'Completed'.
    """
    cursor.execute(
        """
        SELECT g.game_id, g.game_data_finalized AS game_flag, 
               EXISTS (
                   SELECT 1 FROM GameStates WHERE game_id = g.game_id AND is_final_state = 1
               ) AS game_db,
               g.status = 'Completed' AS game_goal
        FROM Games g
        WHERE g.season = ? 
        AND g.season_type IN ('Regular Season', 'Post Season')
    """,
        (season,),
    )

    results = cursor.fetchall()
    return results


def get_log_data_issues(cursor, season):
    """
    Verifies the PbP_Logs and GameStates record counts for each game_id.
    This function returns True if the counts don't match or either count is not between 300 and 700.

    Parameters:
        cursor (sqlite3.Cursor): Database cursor to execute the SQL query.
        season (str): The season to filter the games.

    Returns:
        list: A list of tuples containing the following information for each game:
            - game_id: The unique identifier of the game.
            - log_data_issue: Boolean indicating if there is a mismatch in the counts or if counts are out of the range [300, 700].
    """
    cursor.execute(
        """
        SELECT g.game_id,
               NOT (COALESCE(p.count, 0) BETWEEN 300 AND 800 AND COALESCE(s.count, 0) BETWEEN 300 AND 800 AND COALESCE(p.count, 0) = COALESCE(s.count, 0)) AS log_data_issue
        FROM Games g
        LEFT JOIN (
            SELECT game_id, COUNT(*) as count FROM PbP_Logs GROUP BY game_id
        ) p ON g.game_id = p.game_id
        LEFT JOIN (
            SELECT game_id, COUNT(*) as count FROM GameStates GROUP BY game_id
        ) s ON g.game_id = s.game_id
        WHERE g.season = ? 
        AND g.season_type IN ('Regular Season', 'Post Season')
    """,
        (season,),
    )

    results = cursor.fetchall()
    return results


def database_audit(season, output_file="audit_results.csv", db_path=DB_PATH):
    """
    Conduct an audit of the database to identify issues with the data and save the results to a CSV file.
    """
    update_schedule(season)

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT game_id, home_team, away_team, date_time_est, status, season, season_type
            FROM Games
            WHERE season = ? 
            AND season_type IN ('Regular Season', 'Post Season')
        """,
            (season,),
        )

        games = cursor.fetchall()
        games_df = pd.DataFrame(
            games,
            columns=[
                "game_id",
                "home_team",
                "away_team",
                "date_time_est",
                "status",
                "season",
                "season_type",
            ],
        )

        pre_game_results = get_pre_game_data(cursor, season)
        game_data_results = get_game_data(cursor, season)
        log_data_results = get_log_data_issues(cursor, season)

        pre_game_df = pd.DataFrame(
            pre_game_results,
            columns=["game_id", "pre_game_flag", "pre_game_db", "pre_game_goal"],
        )
        # Convert columns to boolean
        pre_game_df["pre_game_flag"] = pre_game_df["pre_game_flag"].astype(bool)
        pre_game_df["pre_game_db"] = pre_game_df["pre_game_db"].astype(bool)
        pre_game_df["pre_game_goal"] = pre_game_df["pre_game_goal"].astype(bool)

        game_data_df = pd.DataFrame(
            game_data_results, columns=["game_id", "game_flag", "game_db", "game_goal"]
        )
        # Convert columns to boolean
        game_data_df["game_flag"] = game_data_df["game_flag"].astype(bool)
        game_data_df["game_db"] = game_data_df["game_db"].astype(bool)
        game_data_df["game_goal"] = game_data_df["game_goal"].astype(bool)

        log_data_df = pd.DataFrame(
            log_data_results, columns=["game_id", "log_data_issue"]
        )
        # Convert column to boolean
        log_data_df["log_data_issue"] = log_data_df["log_data_issue"].astype(bool)

        # Debugging print statements
        print("Pre-game DataFrame head:")
        print(pre_game_df.head())
        print("Game Data DataFrame head:")
        print(game_data_df.head())
        print("Log Data Issues DataFrame head:")
        print(log_data_df.head())

        final_df = (
            games_df.merge(pre_game_df, on="game_id")
            .merge(game_data_df, on="game_id")
            .merge(log_data_df, on="game_id")
        )

        # Debugging print statement for final DataFrame
        print("Final DataFrame before sorting:")
        print(final_df.head())

        final_df["pre_game_mismatch"] = (
            final_df["pre_game_flag"] != final_df["pre_game_db"]
        ) | (final_df["pre_game_flag"] != final_df["pre_game_goal"])
        final_df["game_mismatch"] = (final_df["game_flag"] != final_df["game_db"]) | (
            final_df["game_flag"] != final_df["game_goal"]
        )

        final_df = final_df.sort_values(
            by=[
                "pre_game_mismatch",
                "game_mismatch",
                "log_data_issue",
                "date_time_est",
            ],
            ascending=[False, False, False, True],
        )

        # Debugging print statement for final DataFrame after sorting
        print("Final DataFrame after sorting:")
        print(final_df.head())

        if output_file:
            final_df.to_csv(output_file, index=False)

        pre_game_mismatches = final_df[final_df["pre_game_mismatch"]]
        game_data_mismatches = final_df[final_df["game_mismatch"]]
        log_data_issues = final_df[final_df["log_data_issue"]]

        print_summary_and_mismatches(
            season, final_df, pre_game_mismatches, game_data_mismatches, log_data_issues
        )

        conn.close()
        return final_df

    except sqlite3.Error as e:
        print(f"SQLite error: {e}")
        if conn:
            conn.close()
        raise


def print_summary_and_mismatches(
    season, final_df, pre_game_mismatches, game_data_mismatches, log_data_issues
):
    total_games = len(final_df)
    game_counts_by_status = final_df["status"].value_counts().to_dict()
    true_counts = {
        "pre_game_flag": final_df["pre_game_flag"].sum(),
        "pre_game_db": final_df["pre_game_db"].sum(),
        "pre_game_goal": final_df["pre_game_goal"].sum(),
        "game_flag": final_df["game_flag"].sum(),
        "game_db": final_df["game_db"].sum(),
        "game_goal": final_df["game_goal"].sum(),
    }
    total_pre_game_mismatches = pre_game_mismatches["game_id"].nunique()
    total_game_mismatches = game_data_mismatches["game_id"].nunique()
    total_log_data_issues = log_data_issues["game_id"].nunique()

    print(f"Season: {season}")
    print(f"Total games: {total_games}")
    print("Game counts by status:")
    for status, count in game_counts_by_status.items():
        print(f"  {status}: {count}")
    print("True counts for booleans:")
    for key, count in true_counts.items():
        print(f"  {key}: {count}")
    print(f"Total pre_game_mismatched games: {total_pre_game_mismatches}")
    print(f"Total game_mismatched games: {total_game_mismatches}")
    print(f"Total log_data_issue games: {total_log_data_issues}")

    if not pre_game_mismatches.empty:
        print("\nPre-game data mismatches found:")
        print(pre_game_mismatches)
        game_ids = ",".join(pre_game_mismatches["game_id"].astype(str))
        print(game_ids)
    else:
        print("No pre-game data mismatches found.")

    if not game_data_mismatches.empty:
        print("\nGame data mismatches found:")
        print(game_data_mismatches)
        game_ids = ",".join(game_data_mismatches["game_id"].astype(str))
        print(game_ids)
    else:
        print("No game data mismatches found.")

    if not log_data_issues.empty:
        print("\nLog data issues found:")
        print(log_data_issues)
        game_ids = ",".join(log_data_issues["game_id"].astype(str))
        print(game_ids)
    else:
        print("No log data issues found.")


if __name__ == "__main__":
    season = "2023-2024"
    final_df = database_audit(season, output_file="audit.csv")
