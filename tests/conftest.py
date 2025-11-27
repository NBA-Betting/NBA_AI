"""
conftest.py

Pytest configuration and shared fixtures for NBA AI tests.
"""

import os
import sqlite3

# Ensure src is importable
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(scope="session")
def project_root():
    """Return the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def test_db_path(project_root):
    """Return the path to the test database (uses real DB for integration tests)."""
    from src.config import config

    return config["database"]["path"]


@pytest.fixture
def sample_game_ids():
    """Return sample game IDs for testing (2024-2025 season)."""
    return [
        "0022400001",  # First game of 2024-2025 season
        "0022400002",
        "0022400003",
    ]


@pytest.fixture
def sample_date():
    """Return a sample date with games."""
    return "2024-10-22"  # Season opener 2024-2025


@pytest.fixture
def invalid_game_ids():
    """Return invalid game IDs for error testing."""
    return [
        "invalid",
        "123",
        "00223",  # Too short
        "1022400001",  # Doesn't start with 00
    ]


@pytest.fixture
def mock_feature_set():
    """Return a mock feature set for predictor testing."""
    return {
        "home_avg_pts": 110.5,
        "home_avg_opp_pts": 108.2,
        "home_avg_fg_pct": 0.465,
        "home_avg_fg3_pct": 0.365,
        "home_avg_ft_pct": 0.785,
        "home_avg_reb": 44.2,
        "home_avg_ast": 25.8,
        "home_avg_stl": 7.5,
        "home_avg_blk": 5.2,
        "home_avg_tov": 13.5,
        "home_avg_pf": 19.8,
        "home_avg_plus_minus": 2.3,
        "home_avg_off_rating": 112.5,
        "home_avg_def_rating": 110.2,
        "home_avg_pace": 100.5,
        "home_avg_ts_pct": 0.575,
        "home_avg_efg_pct": 0.535,
        "home_games_played": 10,
        "home_games_won": 6,
        "home_games_lost": 4,
        "home_win_pct": 0.6,
        "away_avg_pts": 108.3,
        "away_avg_opp_pts": 109.1,
        "away_avg_fg_pct": 0.455,
        "away_avg_fg3_pct": 0.355,
        "away_avg_ft_pct": 0.775,
        "away_avg_reb": 43.5,
        "away_avg_ast": 24.2,
        "away_avg_stl": 7.2,
        "away_avg_blk": 4.8,
        "away_avg_tov": 14.2,
        "away_avg_pf": 20.5,
        "away_avg_plus_minus": -0.8,
        "away_avg_off_rating": 110.2,
        "away_avg_def_rating": 111.0,
        "away_avg_pace": 99.8,
        "away_avg_ts_pct": 0.565,
        "away_avg_efg_pct": 0.525,
        "away_games_played": 10,
        "away_games_won": 5,
        "away_games_lost": 5,
        "away_win_pct": 0.5,
        "is_home_b2b": 0,
        "is_away_b2b": 0,
    }


@pytest.fixture
def flask_test_client(project_root):
    """Create a Flask test client."""
    from src.web_app.app import create_app

    app = create_app(predictor="Baseline")
    app.config["TESTING"] = True

    with app.test_client() as client:
        yield client
