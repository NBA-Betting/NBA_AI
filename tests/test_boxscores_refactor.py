"""
Tests for Boxscores Stage 7 refactor: timestamp-based refetch logic, validator, StageLogger integration.

Tests cover:
- Timestamp-based caching logic in get_games_needing_boxscores()
- BoxscoresValidator comprehensive validation
- save_boxscores() return counts and database migration
- get_boxscores() ThreadPoolExecutor integration
- Both Live and Stats API endpoint handling
- Minutes-based finalization logic
"""

import sqlite3
import tempfile
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from src.database import create_connection, get_db
from src.database_updater.boxscores import (
    fetch_single_boxscore,
    get_boxscores,
    save_boxscores,
)
from src.database_updater.database_update_manager import (
    _mark_boxscore_games_finalized,
    get_games_needing_boxscores,
)
from src.database_updater.validators import BoxscoresValidator, Severity


class TestBoxscoreRefetchLogic:
    """Test timestamp-based refetch query logic."""

    @pytest.fixture
    def test_db(self):
        """Create temporary test database with schema."""
        db = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        conn = create_connection(db.name)
        cursor = conn.cursor()

        # Create schema with boxscore_last_fetched_at column
        cursor.execute(
            """
            CREATE TABLE Games (
                game_id TEXT PRIMARY KEY,
                date_time_utc TEXT,
                home_team TEXT,
                away_team TEXT,
                status INTEGER,
                season TEXT,
                season_type TEXT,
                boxscore_data_finalized BOOLEAN DEFAULT 0,
                boxscore_last_fetched_at TEXT
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE PlayerBox (
                player_id INTEGER,
                game_id TEXT,
                team_id INTEGER,
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
                PRIMARY KEY (player_id, game_id)
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE TeamBox (
                team_id INTEGER,
                game_id TEXT,
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
                PRIMARY KEY (team_id, game_id)
            )
        """
        )

        conn.commit()
        yield db.name
        conn.close()

    def test_in_progress_games_no_boxscores(self, test_db):
        """In-progress games with no boxscores should be fetched."""
        conn = create_connection(test_db)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO Games (game_id, season, season_type, status, date_time_utc, boxscore_data_finalized)
            VALUES ('0022300001', '2023-2024', 'Regular Season', 2, '2024-01-01T19:00:00', 0)
        """
        )
        conn.commit()

        games = get_games_needing_boxscores("2023-2024", test_db)
        assert "0022300001" in games
        conn.close()

    def test_in_progress_games_recent_fetch(self, test_db):
        """In-progress games fetched <5 min ago should be skipped."""
        conn = create_connection(test_db)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO Games (game_id, season, season_type, status, date_time_utc, 
                              boxscore_data_finalized, boxscore_last_fetched_at)
            VALUES ('0022300001', '2023-2024', 'Regular Season', 2, '2024-01-01T19:00:00', 0, 
                    datetime('now', '-2 minutes'))
        """
        )
        conn.commit()

        games = get_games_needing_boxscores("2023-2024", test_db)
        assert "0022300001" not in games
        conn.close()

    def test_in_progress_games_stale_fetch(self, test_db):
        """In-progress games fetched >5 min ago should be re-fetched."""
        conn = create_connection(test_db)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO Games (game_id, season, season_type, status, date_time_utc, 
                              boxscore_data_finalized, boxscore_last_fetched_at)
            VALUES ('0022300001', '2023-2024', 'Regular Season', 2, '2024-01-01T19:00:00', 0, 
                    datetime('now', '-10 minutes'))
        """
        )
        conn.commit()

        games = get_games_needing_boxscores("2023-2024", test_db)
        assert "0022300001" in games
        conn.close()

    def test_completed_not_finalized_needs_refetch(self, test_db):
        """Completed games not finalized with stale fetch should be re-fetched."""
        conn = create_connection(test_db)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO Games (game_id, season, season_type, status, date_time_utc, 
                              boxscore_data_finalized, boxscore_last_fetched_at)
            VALUES ('0022300001', '2023-2024', 'Regular Season', 3, '2024-01-01T19:00:00', 0, 
                    datetime('now', '-10 minutes'))
        """
        )
        conn.commit()

        games = get_games_needing_boxscores("2023-2024", test_db)
        assert "0022300001" in games
        conn.close()

    def test_completed_missing_boxscores_recent_game(self, test_db):
        """Recent completed games missing boxscores should be fetched (48h window)."""
        conn = create_connection(test_db)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO Games (game_id, season, season_type, status, date_time_utc, boxscore_data_finalized)
            VALUES ('0022300001', '2023-2024', 'Regular Season', 3, datetime('now', '-12 hours'), 0)
        """
        )
        conn.commit()

        games = get_games_needing_boxscores("2023-2024", test_db)
        assert "0022300001" in games
        conn.close()

    def test_completed_missing_boxscores_old_game(self, test_db):
        """Old completed games missing boxscores SHOULD be selected (ensures complete coverage)."""
        conn = create_connection(test_db)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO Games (game_id, season, season_type, status, date_time_utc, boxscore_data_finalized)
            VALUES ('0022300001', '2023-2024', 'Regular Season', 3, datetime('now', '-72 hours'), 0)
        """
        )
        conn.commit()

        games = get_games_needing_boxscores("2023-2024", test_db)
        assert "0022300001" in games  # Should be fetched to ensure data completeness
        conn.close()

    def test_finalized_games_skipped(self, test_db):
        """Games with boxscore_data_finalized=1 should be skipped."""
        conn = create_connection(test_db)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO Games (game_id, season, season_type, status, date_time_utc, boxscore_data_finalized)
            VALUES ('0022300001', '2023-2024', 'Regular Season', 3, '2024-01-01T19:00:00', 1)
        """
        )
        conn.commit()

        games = get_games_needing_boxscores("2023-2024", test_db)
        assert "0022300001" not in games
        conn.close()


class TestBoxscoresValidator:
    """Test BoxscoresValidator comprehensive validation."""

    @pytest.fixture
    def test_db(self):
        """Create test database with boxscore data."""
        db = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        conn = create_connection(db.name)
        cursor = conn.cursor()

        # Create schema
        cursor.execute(
            """
            CREATE TABLE Games (
                game_id TEXT PRIMARY KEY,
                status INTEGER
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE PlayerBox (
                player_id INTEGER,
                game_id TEXT,
                team_id INTEGER,
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
                PRIMARY KEY (player_id, game_id)
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE TeamBox (
                team_id INTEGER,
                game_id TEXT,
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
                PRIMARY KEY (team_id, game_id)
            )
        """
        )

        conn.commit()
        yield db.name
        conn.close()

    def test_missing_player_box_records(self, test_db):
        """Should detect completed games missing PlayerBox records."""
        conn = create_connection(test_db)
        cursor = conn.cursor()

        cursor.execute("INSERT INTO Games (game_id, status) VALUES ('0022300001', 3)")
        conn.commit()

        validator = BoxscoresValidator()
        result = validator.validate(["0022300001"], cursor)

        assert result.has_critical_issues
        missing_issue = next(
            (
                issue
                for issue in result.issues
                if issue.check_id == "MISSING_PLAYER_BOX"
            ),
            None,
        )
        assert missing_issue is not None
        assert missing_issue.severity == Severity.CRITICAL
        conn.close()

    def test_missing_team_box_records(self, test_db):
        """Should detect completed games missing TeamBox records."""
        conn = create_connection(test_db)
        cursor = conn.cursor()

        cursor.execute("INSERT INTO Games (game_id, status) VALUES ('0022300001', 3)")
        # Add some PlayerBox but no TeamBox
        cursor.execute(
            "INSERT INTO PlayerBox (game_id, team_id, player_id, min, pts) VALUES ('0022300001', 1, 1, 30, 20)"
        )
        conn.commit()

        validator = BoxscoresValidator()
        result = validator.validate(["0022300001"], cursor)

        assert result.has_critical_issues
        missing_issue = next(
            (issue for issue in result.issues if issue.check_id == "MISSING_TEAM_BOX"),
            None,
        )
        assert missing_issue is not None
        assert missing_issue.severity == Severity.CRITICAL
        conn.close()

    def test_invalid_player_count_per_team(self, test_db):
        """Should detect teams with unusual player counts."""
        conn = create_connection(test_db)
        cursor = conn.cursor()

        cursor.execute("INSERT INTO Games (game_id, status) VALUES ('0022300001', 3)")

        # Add only 3 players for team 1 (too few)
        for i in range(3):
            cursor.execute(
                "INSERT INTO PlayerBox (game_id, team_id, player_id, min, pts) VALUES ('0022300001', 1, ?, 30, 20)",
                (i,),
            )

        # Add normal count for team 2
        for i in range(8):
            cursor.execute(
                "INSERT INTO PlayerBox (game_id, team_id, player_id, min, pts) VALUES ('0022300001', 2, ?, 30, 20)",
                (i + 10,),
            )

        conn.commit()

        validator = BoxscoresValidator()
        result = validator.validate(["0022300001"], cursor)

        player_count_issue = next(
            (
                issue
                for issue in result.issues
                if issue.check_id == "INVALID_PLAYER_COUNT"
            ),
            None,
        )
        assert player_count_issue is not None
        assert player_count_issue.severity == Severity.WARNING
        conn.close()

    def test_invalid_team_count_per_game(self, test_db):
        """Should detect games without exactly 2 teams in TeamBox."""
        conn = create_connection(test_db)
        cursor = conn.cursor()

        cursor.execute("INSERT INTO Games (game_id, status) VALUES ('0022300001', 3)")

        # Add only 1 team (should be 2)
        cursor.execute(
            "INSERT INTO TeamBox (game_id, team_id, pts) VALUES ('0022300001', 1, 100)"
        )

        conn.commit()

        validator = BoxscoresValidator()
        result = validator.validate(["0022300001"], cursor)

        assert result.has_critical_issues
        team_count_issue = next(
            (
                issue
                for issue in result.issues
                if issue.check_id == "INVALID_TEAM_COUNT"
            ),
            None,
        )
        assert team_count_issue is not None
        assert team_count_issue.severity == Severity.CRITICAL
        conn.close()

    def test_low_minutes_validation(self, test_db):
        """Should detect teams with <240 minutes (incomplete games)."""
        conn = create_connection(test_db)
        cursor = conn.cursor()

        cursor.execute("INSERT INTO Games (game_id, status) VALUES ('0022300001', 3)")

        # Team 1: Only 200 minutes (too low)
        for i in range(5):
            cursor.execute(
                "INSERT INTO PlayerBox (game_id, team_id, player_id, min, pts) VALUES ('0022300001', 1, ?, 40, 20)",
                (i,),
            )

        # Team 2: Normal 240 minutes
        for i in range(5):
            cursor.execute(
                "INSERT INTO PlayerBox (game_id, team_id, player_id, min, pts) VALUES ('0022300001', 2, ?, 48, 20)",
                (i + 10,),
            )

        conn.commit()

        validator = BoxscoresValidator()
        result = validator.validate(["0022300001"], cursor)

        low_minutes_issue = next(
            (issue for issue in result.issues if issue.check_id == "LOW_MINUTES"),
            None,
        )
        assert low_minutes_issue is not None
        assert low_minutes_issue.severity == Severity.WARNING
        conn.close()

    def test_null_player_fields(self, test_db):
        """Should detect PlayerBox records with NULL critical fields."""
        conn = create_connection(test_db)
        cursor = conn.cursor()

        cursor.execute("INSERT INTO Games (game_id, status) VALUES ('0022300001', 3)")

        # Add player with NULL pts
        cursor.execute(
            "INSERT INTO PlayerBox (game_id, team_id, player_id, min, pts) VALUES ('0022300001', 1, 1, 30, NULL)"
        )

        conn.commit()

        validator = BoxscoresValidator()
        result = validator.validate(["0022300001"], cursor)

        null_issue = next(
            (
                issue
                for issue in result.issues
                if issue.check_id == "NULL_PLAYER_FIELDS"
            ),
            None,
        )
        assert null_issue is not None
        assert null_issue.severity == Severity.WARNING
        conn.close()


class TestSaveBoxscores:
    """Test save_boxscores return counts and database migration."""

    @pytest.fixture
    def test_db(self):
        """Create test database without boxscore_last_fetched_at column."""
        db = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        conn = create_connection(db.name)
        cursor = conn.cursor()

        # Create schema WITHOUT boxscore_last_fetched_at column to test migration
        cursor.execute(
            """
            CREATE TABLE Games (
                game_id TEXT PRIMARY KEY,
                date_time_utc TEXT,
                status INTEGER
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE PlayerBox (
                player_id INTEGER,
                game_id TEXT,
                team_id INTEGER,
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
                PRIMARY KEY (player_id, game_id)
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE TeamBox (
                team_id INTEGER,
                game_id TEXT,
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
                PRIMARY KEY (team_id, game_id)
            )
        """
        )

        cursor.execute("INSERT INTO Games (game_id, status) VALUES ('0022300001', 3)")
        conn.commit()
        yield db.name
        conn.close()

    def test_column_migration(self, test_db):
        """Should auto-migrate boxscore_last_fetched_at column."""
        # Verify column doesn't exist initially
        conn = create_connection(test_db)
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(Games)")
        columns = [row[1] for row in cursor.fetchall()]
        assert "boxscore_last_fetched_at" not in columns

        # Create sample boxscore data
        sample_data = {
            "0022300001": (
                [
                    {
                        "player_id": 1,
                        "game_id": "0022300001",
                        "team_id": 1,
                        "player_name": "Test Player",
                        "position": "PG",
                        "min": 30.5,
                        "pts": 20,
                        "reb": 5,
                        "ast": 8,
                        "stl": 2,
                        "blk": 0,
                        "tov": 3,
                        "pf": 2,
                        "oreb": 1,
                        "dreb": 4,
                        "fga": 15,
                        "fgm": 8,
                        "fg_pct": 0.533,
                        "fg3a": 6,
                        "fg3m": 3,
                        "fg3_pct": 0.500,
                        "fta": 4,
                        "ftm": 4,
                        "ft_pct": 1.000,
                        "plus_minus": 8,
                    }
                ],
                [
                    {
                        "team_id": 1,
                        "game_id": "0022300001",
                        "pts": 110,
                        "pts_allowed": 105,
                        "reb": 45,
                        "ast": 25,
                        "stl": 10,
                        "blk": 5,
                        "tov": 12,
                        "pf": 18,
                        "fga": 85,
                        "fgm": 42,
                        "fg_pct": 0.494,
                        "fg3a": 35,
                        "fg3m": 15,
                        "fg3_pct": 0.429,
                        "fta": 20,
                        "ftm": 17,
                        "ft_pct": 0.850,
                        "plus_minus": 5,
                    }
                ],
            )
        }

        # Save data (should trigger migration)
        result = save_boxscores(sample_data, test_db)

        # Verify column was added
        cursor.execute("PRAGMA table_info(Games)")
        columns = [row[1] for row in cursor.fetchall()]
        assert "boxscore_last_fetched_at" in columns

        # Verify timestamp was set
        cursor.execute(
            "SELECT boxscore_last_fetched_at FROM Games WHERE game_id = '0022300001'"
        )
        timestamp = cursor.fetchone()[0]
        assert timestamp is not None

        # Verify return counts for new game
        assert result["added"] == 1
        assert result["updated"] == 0

        conn.close()

    def test_update_existing_data(self, test_db):
        """Should return correct counts when updating existing data."""
        # Add initial data
        sample_data = {
            "0022300001": (
                [
                    {
                        "player_id": 1,
                        "game_id": "0022300001",
                        "team_id": 1,
                        "player_name": "Test Player",
                        "position": "PG",
                        "min": 30.5,
                        "pts": 20,
                        "reb": 5,
                        "ast": 8,
                        "stl": 2,
                        "blk": 0,
                        "tov": 3,
                        "pf": 2,
                        "oreb": 1,
                        "dreb": 4,
                        "fga": 15,
                        "fgm": 8,
                        "fg_pct": 0.533,
                        "fg3a": 6,
                        "fg3m": 3,
                        "fg3_pct": 0.500,
                        "fta": 4,
                        "ftm": 4,
                        "ft_pct": 1.000,
                        "plus_minus": 8,
                    }
                ],
                [
                    {
                        "team_id": 1,
                        "game_id": "0022300001",
                        "pts": 110,
                        "pts_allowed": 105,
                        "reb": 45,
                        "ast": 25,
                        "stl": 10,
                        "blk": 5,
                        "tov": 12,
                        "pf": 18,
                        "fga": 85,
                        "fgm": 42,
                        "fg_pct": 0.494,
                        "fg3a": 35,
                        "fg3m": 15,
                        "fg3_pct": 0.429,
                        "fta": 20,
                        "ftm": 17,
                        "ft_pct": 0.850,
                        "plus_minus": 5,
                    }
                ],
            )
        }

        # First save (added)
        result1 = save_boxscores(sample_data, test_db)
        assert result1["added"] == 1
        assert result1["updated"] == 0

        # Second save (updated)
        result2 = save_boxscores(sample_data, test_db)
        assert result2["added"] == 0
        assert result2["updated"] == 1


class TestMinutesBasedFinalization:
    """Test _mark_boxscore_games_finalized with minutes-based logic."""

    @pytest.fixture
    def test_db(self):
        """Create test database with complete boxscore data."""
        db = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        conn = create_connection(db.name)
        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE Games (
                game_id TEXT PRIMARY KEY,
                status INTEGER,
                date_time_utc TEXT,
                boxscore_data_finalized BOOLEAN DEFAULT 0
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE PlayerBox (
                player_id INTEGER,
                game_id TEXT,
                team_id INTEGER,
                min REAL,
                pts INTEGER
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE TeamBox (
                team_id INTEGER,
                game_id TEXT,
                pts INTEGER
            )
        """
        )

        conn.commit()
        yield db.name
        conn.close()

    def test_complete_game_gets_finalized(self, test_db):
        """Games with 240+ minutes per team should be finalized."""
        conn = create_connection(test_db)
        cursor = conn.cursor()

        cursor.execute(
            "INSERT INTO Games (game_id, status, date_time_utc, boxscore_data_finalized) VALUES ('0022300001', 3, '2023-10-24T23:30:00Z', 0)"
        )

        # Team 1: 240 minutes total across 8 players (30 min each)
        for i in range(8):
            cursor.execute(
                "INSERT INTO PlayerBox (game_id, team_id, player_id, min, pts) VALUES ('0022300001', 1, ?, 30, 20)",
                (i,),
            )

        # Team 2: 250 minutes total across 8 players (31.25 min each)
        for i in range(8):
            cursor.execute(
                "INSERT INTO PlayerBox (game_id, team_id, player_id, min, pts) VALUES ('0022300001', 2, ?, 31.25, 20)",
                (i + 10,),
            )

        # Add required TeamBox data (finalization logic checks for 2 teams)
        cursor.execute(
            "INSERT INTO TeamBox (game_id, team_id, pts) VALUES ('0022300001', 1, 110)"
        )
        cursor.execute(
            "INSERT INTO TeamBox (game_id, team_id, pts) VALUES ('0022300001', 2, 105)"
        )

        conn.commit()

        finalized_games = _mark_boxscore_games_finalized(["0022300001"], test_db)
        assert "0022300001" in finalized_games

        # Verify flag was set
        cursor.execute(
            "SELECT boxscore_data_finalized FROM Games WHERE game_id = '0022300001'"
        )
        assert cursor.fetchone()[0] == 1

        conn.close()

    def test_incomplete_game_not_finalized(self, test_db):
        """Games with <240 minutes per team should not be finalized."""
        conn = create_connection(test_db)
        cursor = conn.cursor()

        cursor.execute(
            "INSERT INTO Games (game_id, status, date_time_utc, boxscore_data_finalized) VALUES ('0022300001', 3, '2023-10-24T23:30:00Z', 0)"
        )

        # Team 1: Only 200 minutes (under threshold)
        for i in range(5):
            cursor.execute(
                "INSERT INTO PlayerBox (game_id, team_id, player_id, min, pts) VALUES ('0022300001', 1, ?, 40, 20)",
                (i,),
            )

        # Team 2: 240 minutes (at threshold)
        for i in range(5):
            cursor.execute(
                "INSERT INTO PlayerBox (game_id, team_id, player_id, min, pts) VALUES ('0022300001', 2, ?, 48, 20)",
                (i + 10,),
            )

        conn.commit()

        finalized_games = _mark_boxscore_games_finalized(["0022300001"], test_db)
        assert "0022300001" not in finalized_games

        # Verify flag was not set
        cursor.execute(
            "SELECT boxscore_data_finalized FROM Games WHERE game_id = '0022300001'"
        )
        assert cursor.fetchone()[0] == 0

        conn.close()

    def test_non_final_status_not_finalized(self, test_db):
        """Games with status != 3 should not be finalized regardless of minutes."""
        conn = create_connection(test_db)
        cursor = conn.cursor()

        cursor.execute(
            "INSERT INTO Games (game_id, status, date_time_utc, boxscore_data_finalized) VALUES ('0022300001', 2, '2023-10-24T23:30:00Z', 0)"
        )

        # Add plenty of minutes but status=2 (In Progress)
        for team_id in [1, 2]:
            for i in range(5):
                cursor.execute(
                    "INSERT INTO PlayerBox (game_id, team_id, player_id, min, pts) VALUES ('0022300001', ?, ?, 48, 20)",
                    (team_id, i + team_id * 10),
                )

        conn.commit()

        finalized_games = _mark_boxscore_games_finalized(["0022300001"], test_db)
        assert "0022300001" not in finalized_games

        conn.close()


class TestBoxscoreAPIMocking:
    """Test get_boxscores with mocked API endpoints."""

    @patch("src.database_updater.boxscores.fetch_single_boxscore")
    def test_concurrent_fetching(self, mock_fetch):
        """Should use ThreadPoolExecutor for concurrent API calls."""

        # Mock successful fetch for each game
        def mock_fetch_side_effect(game_id, use_live=False):
            return (game_id, [{"player_id": 1}], [{"team_id": 1}])

        mock_fetch.side_effect = mock_fetch_side_effect

        game_ids = ["0022300001", "0022300002", "0022300003"]
        result = get_boxscores(game_ids)

        # Should call fetch for each game
        assert mock_fetch.call_count == len(game_ids)
        assert len(result) == len(game_ids)

    @patch("src.database_updater.boxscores.get_boxscore_with_fallback")
    def test_single_game_fetch_error_handling(self, mock_fallback):
        """Should handle individual game fetch errors gracefully."""
        # Mock error for one game
        mock_fallback.side_effect = Exception("API Error")

        game_id, players, teams = fetch_single_boxscore("0022300001", use_live=False)

        assert game_id == "0022300001"
        assert players == []
        assert teams == []

    @patch("src.database_updater.boxscores.fetch_single_boxscore")
    def test_stage_logger_integration(self, mock_fetch):
        """Should track API calls in StageLogger."""
        from src.utils import StageLogger

        def mock_fetch_side_effect(game_id, use_live=False):
            return (game_id, [{"player_id": 1}], [{"team_id": 1}])

        mock_fetch.side_effect = mock_fetch_side_effect

        stage_logger = StageLogger("Boxscores")
        assert stage_logger.api_calls == 0

        game_ids = ["0022300001", "0022300002", "0022300003"]
        get_boxscores(game_ids, stage_logger=stage_logger)

        assert stage_logger.api_calls == 3
