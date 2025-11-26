"""
database_migration.py

Adds PlayerBox and TeamBox tables to the existing database with compatible schema.
This allows us to collect player and team boxscore statistics alongside existing PBP data.

Usage:
    python -m src.database_migration
"""

import logging
import sqlite3

from src.config import config

DB_PATH = config["database"]["path"]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def add_playerbox_teambox_tables():
    """
    Add PlayerBox and TeamBox tables to the database if they don't exist.
    Uses TEXT for game_id to match existing schema instead of INTEGER.
    """

    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()

        try:
            # Check existing tables
            existing_tables = [
                row[0]
                for row in cursor.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            ]

            logger.info(f"Existing tables: {existing_tables}")

            # Create ScheduleCache table for tracking schedule updates
            if "ScheduleCache" not in existing_tables:
                logger.info("Creating ScheduleCache table...")
                cursor.execute(
                    """
                    CREATE TABLE ScheduleCache (
                        season TEXT PRIMARY KEY,
                        last_update_datetime TEXT NOT NULL
                    )
                """
                )
                logger.info("✓ ScheduleCache table created")
            else:
                logger.info("ScheduleCache table already exists")

            # Create PlayerBox table (modified to use TEXT game_id)
            if "PlayerBox" not in existing_tables:
                logger.info("Creating PlayerBox table...")
                cursor.execute(
                    """
                    CREATE TABLE PlayerBox (
                        player_id INTEGER NOT NULL,
                        game_id TEXT NOT NULL,
                        team_id TEXT NOT NULL,
                        player_name TEXT,
                        position TEXT,
                        min REAL,
                        pts INTEGER,
                        reb INTEGER,
                        ast INTEGER,
                        stl INTEGER,
                        blk INTEGER,
                        tov INTEGER,
                        pf INTEGER,
                        oreb INTEGER,
                        dreb INTEGER,
                        fga INTEGER,
                        fgm INTEGER,
                        fg_pct REAL,
                        fg3a INTEGER,
                        fg3m INTEGER,
                        fg3_pct REAL,
                        fta INTEGER,
                        ftm INTEGER,
                        ft_pct REAL,
                        plus_minus INTEGER,
                        PRIMARY KEY (player_id, game_id),
                        FOREIGN KEY (game_id) REFERENCES Games(game_id)
                    )
                """
                )
                logger.info("✓ PlayerBox table created")
            else:
                logger.info("PlayerBox table already exists")

            # Create TeamBox table (modified to use TEXT game_id and team_id)
            if "TeamBox" not in existing_tables:
                logger.info("Creating TeamBox table...")
                cursor.execute(
                    """
                    CREATE TABLE TeamBox (
                        team_id TEXT NOT NULL,
                        game_id TEXT NOT NULL,
                        pts INTEGER,
                        pts_allowed INTEGER,
                        reb INTEGER,
                        ast INTEGER,
                        stl INTEGER,
                        blk INTEGER,
                        tov INTEGER,
                        pf INTEGER,
                        fga INTEGER,
                        fgm INTEGER,
                        fg_pct REAL,
                        fg3a INTEGER,
                        fg3m INTEGER,
                        fg3_pct REAL,
                        fta INTEGER,
                        ftm INTEGER,
                        ft_pct REAL,
                        plus_minus INTEGER,
                        PRIMARY KEY (team_id, game_id),
                        FOREIGN KEY (game_id) REFERENCES Games(game_id)
                    )
                """
                )
                logger.info("✓ TeamBox table created")
            else:
                logger.info("TeamBox table already exists")

            conn.commit()
            logger.info("✓ Migration complete")

            # Verify
            tables = [
                row[0]
                for row in cursor.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            ]
            logger.info(f"Final tables: {tables}")

        except Exception as e:
            conn.rollback()
            logger.error(f"Migration failed: {e}")
            raise


if __name__ == "__main__":
    logger.info(f"Migrating database: {DB_PATH}")
    add_playerbox_teambox_tables()
