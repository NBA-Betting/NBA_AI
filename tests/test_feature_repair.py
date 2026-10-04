"""Previously finalized empty opener features must be retried."""

import sqlite3

from src.database_updater.database_update_manager import get_games_with_incomplete_pre_game_data


def test_retry_finalized_empty_features_only_for_upcoming_supported_games(tmp_path):
    path = str(tmp_path / "features.sqlite")
    with sqlite3.connect(path) as conn:
        conn.executescript("""
            CREATE TABLE Games (
                game_id TEXT, season TEXT, season_type TEXT, status INTEGER,
                status_text TEXT, date_time_utc TEXT, home_team TEXT, away_team TEXT,
                pre_game_data_finalized INTEGER, game_data_finalized INTEGER,
                boxscore_data_finalized INTEGER);
            CREATE TABLE Features (game_id TEXT, feature_set TEXT);
        """)
        conn.executemany("INSERT INTO Games VALUES (?, '2026-2027', ?, ?, ?, ?, 'BOS', 'DET', 1, 1, 1)", [
            ("empty", "Regular Season", 1, "Scheduled", "2026-10-20T23:30:00Z"),
            ("valid", "Regular Season", 1, "Scheduled", "2026-10-20T23:30:00Z"),
            ("final", "Regular Season", 3, "Final", "2026-10-19T23:30:00Z"),
            ("preseason", "Pre Season", 1, "Scheduled", "2026-10-20T23:30:00Z"),
            ("postponed", "Regular Season", 1, "PPD", "2026-10-20T23:30:00Z"),
        ])
        conn.executemany("INSERT INTO Features VALUES (?, ?)", [
            ("empty", "{}"), ("valid", '{"Home_PPG": 110}'), ("final", "{}"),
            ("preseason", "{}"), ("postponed", "{}"),
        ])
    assert get_games_with_incomplete_pre_game_data("2026-2027", path) == ["empty"]
