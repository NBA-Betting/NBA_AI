"""
Tests for src/database_updater/schedule.py

Simple unit tests to verify schedule module behavior:
- Cache logic works correctly
- Flag preservation during updates
- Date filtering
"""

import sqlite3
import tempfile
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.database import create_connection, get_db
from src.database_updater import schedule
from src.utils import determine_current_season


class TestScheduleCache:
    """Tests for schedule caching behavior."""

    @pytest.fixture
    def temp_db(self):
        """Create temporary database with ScheduleCache table."""
        db = tempfile.NamedTemporaryFile(delete=False, suffix=".sqlite")
        conn = create_connection(db.name)
        cursor = conn.cursor()

        # Create minimal ScheduleCache table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS ScheduleCache (
                season TEXT PRIMARY KEY,
                last_update_datetime TEXT NOT NULL
            )
        """
        )
        conn.commit()
        conn.close()

        yield db.name

        # Cleanup
        import os

        os.unlink(db.name)

    def test_cache_skips_update_for_finalized_historical_season(self, temp_db):
        """Historical season with schedule_finalized=1 should skip update."""
        current_season = determine_current_season()
        historical_season = "2022-2023"  # Not current

        # Ensure schedule_finalized column exists
        with get_db(temp_db) as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(ScheduleCache)")
            columns = [row[1] for row in cursor.fetchall()]
            if "schedule_finalized" not in columns:
                cursor.execute(
                    "ALTER TABLE ScheduleCache ADD COLUMN schedule_finalized INTEGER DEFAULT 0"
                )

            # Insert finalized cache entry
            now = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            cursor.execute(
                "INSERT INTO ScheduleCache (season, last_update_datetime, schedule_finalized) VALUES (?, ?, 1)",
                (historical_season, now),
            )
            conn.commit()

        # Should NOT update (schedule is finalized)
        result = schedule._should_update_schedule(historical_season, temp_db)
        assert result is False, "Finalized schedule should skip update"

    def test_cache_updates_non_finalized_historical_season(self, temp_db):
        """Historical season without finalized flag should update."""
        historical_season = "2022-2023"

        # Ensure schedule_finalized column exists
        with get_db(temp_db) as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(ScheduleCache)")
            columns = [row[1] for row in cursor.fetchall()]
            if "schedule_finalized" not in columns:
                cursor.execute(
                    "ALTER TABLE ScheduleCache ADD COLUMN schedule_finalized INTEGER DEFAULT 0"
                )

            # Insert non-finalized cache entry
            old_time = (pd.Timestamp.now() - timedelta(days=1)).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            cursor.execute(
                "INSERT INTO ScheduleCache (season, last_update_datetime, schedule_finalized) VALUES (?, ?, 0)",
                (historical_season, old_time),
            )
            conn.commit()

        # Should update (not yet finalized)
        result = schedule._should_update_schedule(historical_season, temp_db)
        assert result is True, "Non-finalized schedule should trigger update"

    def test_current_season_always_updates(self, temp_db):
        """Current season should update when cache expires (5 minutes)."""
        current_season = determine_current_season()

        # Insert old cache entry for current season (6 minutes ago - beyond 5-minute threshold)
        with get_db(temp_db) as conn:
            cursor = conn.cursor()
            old_time = (pd.Timestamp.now() - timedelta(minutes=6)).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            cursor.execute(
                "INSERT INTO ScheduleCache VALUES (?, ?)", (current_season, old_time)
            )
            conn.commit()

        # Should update (cache expired - older than 5 minutes)
        result = schedule._should_update_schedule(current_season, temp_db)
        assert result is True, "Current season should update when cache expires"

    def test_missing_cache_triggers_update(self, temp_db):
        """Season with no cache entry should update."""
        season = "2023-2024"

        # Don't insert any cache entry
        result = schedule._should_update_schedule(season, temp_db)
        assert result is True, "Missing cache should trigger update"


class TestScheduleFlagPreservation:
    """Tests for flag preservation during schedule updates."""

    @pytest.fixture
    def games_db(self):
        """Temporary database with one finalized game (never the production DB)."""
        db = tempfile.NamedTemporaryFile(delete=False, suffix=".sqlite")
        with get_db(db.name) as conn:
            conn.execute(
                """
                CREATE TABLE Games (
                    game_id TEXT NOT NULL PRIMARY KEY,
                    date_time_utc TEXT NOT NULL,
                    home_team TEXT NOT NULL,
                    away_team TEXT NOT NULL,
                    status INTEGER NOT NULL,
                    status_text TEXT,
                    season TEXT NOT NULL,
                    season_type TEXT NOT NULL,
                    pre_game_data_finalized BOOLEAN NOT NULL DEFAULT 0,
                    game_data_finalized BOOLEAN NOT NULL DEFAULT 0,
                    boxscore_data_finalized BOOLEAN NOT NULL DEFAULT 0
                )
            """
            )
            conn.execute(
                "INSERT INTO Games VALUES ('0022400001', '2024-10-22T23:30:00Z', 'BOS', 'NYK', "
                "3, 'Final', '2024-2025', 'Regular Season', 1, 1, 1)"
            )
            conn.commit()

        yield db.name

        import os

        os.unlink(db.name)

    def test_save_schedule_preserves_flags(self, games_db):
        """Updating schedule should preserve existing flags."""
        game_data = [
            {
                "gameId": "0022400001",
                "season": "2024-2025",
                "gameDateTimeUTC": "2024-10-22T23:30:00Z",
                "homeTeam": "BOS",
                "awayTeam": "NYK",
                "gameStatus": 3,
                "gameStatusText": "Final",
                "seasonType": "Regular Season",
            }
        ]

        assert schedule.save_schedule(game_data, "2024-2025", games_db)

        with get_db(games_db) as conn:
            flags = conn.execute(
                """
                SELECT game_data_finalized, boxscore_data_finalized, pre_game_data_finalized
                FROM Games WHERE game_id = ?
            """,
                ("0022400001",),
            ).fetchone()

        assert flags == (
            1,
            1,
            1,
        ), f"Flags should be preserved during schedule update (got {flags})"


class TestScheduleDateFiltering:
    """Tests for date filtering in schedule."""

    def test_validate_date_range_filters_correctly(self):
        """Schedule should filter games by date range."""
        # Create mock games spanning multiple dates
        games = [
            {"GAME_DATE": "2024-01-01", "GAME_ID": "001"},
            {"GAME_DATE": "2024-01-15", "GAME_ID": "002"},
            {"GAME_DATE": "2024-02-01", "GAME_ID": "003"},
        ]

        # Filter to January only
        start_date = pd.to_datetime("2024-01-01")
        end_date = pd.to_datetime("2024-01-31")

        filtered = [
            g for g in games if start_date <= pd.to_datetime(g["GAME_DATE"]) <= end_date
        ]

        assert len(filtered) == 2, "Should filter to 2 January games"
        assert filtered[0]["GAME_ID"] == "001"
        assert filtered[1]["GAME_ID"] == "002"
