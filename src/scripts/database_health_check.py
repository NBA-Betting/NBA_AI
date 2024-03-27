import logging
import sqlite3

from src.utils import update_scheduled_games, validate_season_format


def evaluate_database_health(db_path, season):
    """
    Evaluates the health of the database by checking for inconsistencies in the data.

    Parameters:
    db_path (str): The path to the SQLite database.
    season (str): The season to evaluate.

    Returns:
    None
    """
    # Validate season format
    validate_season_format(season, abbreviated=False)

    logging.info(f"Evaluating database health for season {season}...")
    logging.info("--------------------------------------------------")

    # Update season schedule
    try:
        update_scheduled_games(season, db_path)
    except Exception as e:
        print(f"Failed to update season schedule: {e}")
        raise e

    # Connect to SQLite database
    with sqlite3.connect(db_path) as conn:

        # Create a cursor object
        cursor = conn.cursor()

        # Check for missing final states for completed games in the specified season and season_type
        logging.info("Checking for missing final states for completed games...")
        cursor.execute(
            """
            SELECT g.game_id
            FROM Games g
            LEFT JOIN GameStates gs ON g.game_id = gs.game_id AND gs.is_final_state = 1
            WHERE g.status = 'Completed'
            AND gs.game_id IS NULL
            AND g.season = ?
            AND g.season_type IN ('Regular Season', 'Post Season')
            """,
            (season,),
        )
        missing_final_states = cursor.fetchall()

        logging.info(
            f"Missing final states for completed games: {len(missing_final_states)}"
        )
        if missing_final_states:
            logging.info(f"Game IDs: {missing_final_states}")

        # Check for games that should have finalized prior states but do not
        logging.info(
            "Checking for games that should have finalized prior states but do not..."
        )
        cursor.execute(
            """
            SELECT g.game_id
            FROM Games g
            LEFT JOIN PriorStates ps ON g.game_id = ps.game_id
            WHERE g.season = ? AND g.season_type IN ('Regular Season', 'Post Season')
            AND NOT EXISTS (
                SELECT 1
                FROM Games gh
                WHERE gh.date_time_est < g.date_time_est
                AND gh.season = g.season
                AND gh.season_type IN ('Regular Season', 'Post Season')
                AND (gh.home_team = g.home_team OR gh.away_team = g.home_team)
                AND gh.status != 'Completed'
            )
            AND NOT EXISTS (
                SELECT 1
                FROM Games ga
                WHERE ga.date_time_est < g.date_time_est
                AND ga.season = g.season
                AND ga.season_type IN ('Regular Season', 'Post Season')
                AND (ga.home_team = g.away_team OR ga.away_team = g.away_team)
                AND ga.status != 'Completed'
            )
            AND (ps.game_id IS NULL OR ps.are_prior_states_finalized = 0)
            """,
            (season,),
        )
        games_that_should_have_finalized_prior_states_but_do_not = cursor.fetchall()

        logging.info(
            f"Games that should have finalized prior states but do not: {len(games_that_should_have_finalized_prior_states_but_do_not)}"
        )
        if games_that_should_have_finalized_prior_states_but_do_not:
            logging.info(
                f"Game IDs: {games_that_should_have_finalized_prior_states_but_do_not}"
            )

        # Check for games that have finalized prior states but should not
        logging.info(
            "Checking for games that have finalized prior states but should not..."
        )
        cursor.execute(
            """
                SELECT ps.game_id
                FROM PriorStates ps
                JOIN Games g ON ps.game_id = g.game_id
                WHERE g.season = ? AND g.season_type IN ('Regular Season', 'Post Season')
                AND ps.are_prior_states_finalized = 1
                AND (
                    EXISTS (
                        SELECT 1
                        FROM Games gh
                        WHERE gh.date_time_est < g.date_time_est
                        AND gh.season = g.season
                        AND gh.season_type IN ('Regular Season', 'Post Season')
                        AND (gh.home_team = g.home_team OR gh.away_team = g.home_team)
                        AND gh.status != 'Completed'
                    )
                    OR EXISTS (
                        SELECT 1
                        FROM Games ga
                        WHERE ga.date_time_est < g.date_time_est
                        AND ga.season = g.season
                        AND ga.season_type IN ('Regular Season', 'Post Season')
                        AND (ga.home_team = g.away_team OR ga.away_team = g.away_team)
                        AND ga.status != 'Completed'
                    )
                )
                """,
            (season,),
        )
        games_that_have_finalized_prior_states_but_should_not = cursor.fetchall()

        logging.info(
            f"Games that have finalized prior states but should not: {len(games_that_have_finalized_prior_states_but_should_not)}"
        )
        if games_that_have_finalized_prior_states_but_should_not:
            logging.info(
                f"Game IDs: {games_that_have_finalized_prior_states_but_should_not}"
            )

        # Check for games with invalid total or home margin values
        logging.info("Checking for games with invalid total or home margin values...")

        cursor.execute(
            """
            SELECT DISTINCT gs.game_id
            FROM GameStates gs
            JOIN Games g ON gs.game_id = g.game_id
            WHERE (gs.total != (gs.home_score + gs.away_score) OR gs.home_margin != (gs.home_score - gs.away_score))
            AND g.season = ? 
            AND g.season_type IN ('Regular Season', 'Post Season');
            """,
            (season,),
        )

        invalid_total_or_home_margin_games = cursor.fetchall()

        logging.info(
            f"Number of games with invalid total or home margin values: {len(invalid_total_or_home_margin_games)}"
        )
        if invalid_total_or_home_margin_games:
            logging.info(f"Game IDs: {invalid_total_or_home_margin_games}")

        # Check for games with invalid score order
        logging.info("Checking for games with invalid score order...")
        cursor.execute(
            """
            SELECT DISTINCT gs.game_id
            FROM (
                SELECT game_id,
                CASE WHEN home_score < LAG(home_score) OVER(PARTITION BY game_id ORDER BY play_id) THEN 1 ELSE 0 END AS invalid_home_score,
                CASE WHEN away_score < LAG(away_score) OVER(PARTITION BY game_id ORDER BY play_id) THEN 1 ELSE 0 END AS invalid_away_score,
                CASE WHEN total < LAG(total) OVER(PARTITION BY game_id ORDER BY play_id) THEN 1 ELSE 0 END AS invalid_total
                FROM GameStates
            ) gs
            JOIN Games g ON gs.game_id = g.game_id
            WHERE (gs.invalid_home_score = 1 OR gs.invalid_away_score = 1 OR gs.invalid_total = 1)
            AND g.season = ?
            AND g.season_type IN ('Regular Season', 'Post Season');
            """,
            (season,),
        )

        invalid_score_order_games = cursor.fetchall()

        logging.info(
            f"Number of games with invalid score order: {len(invalid_score_order_games)}"
        )
        if invalid_score_order_games:
            logging.info(f"Game IDs: {invalid_score_order_games}")
