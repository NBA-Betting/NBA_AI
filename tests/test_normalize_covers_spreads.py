"""Tests for the historical Covers normalization command."""

import sqlite3

import pytest

from scripts.normalize_covers_spreads import classify_covers_row, normalize_database


@pytest.mark.parametrize(
    ("spread", "result", "home_score", "away_score", "expected"),
    [
        (-3.5, "W", 110, 100, ("keep", -3.5, "W")),
        (-3.5, "W", 100, 105, ("flip", 3.5, "L")),
        (3.5, "W", 100, 100, ("invalidate", None, None)),
        (0.0, "P", 100, 100, ("keep", 0.0, "P")),
        (-3.5, None, 110, 100, ("invalidate", None, None)),
        (-3.5, "W", None, None, ("invalidate", None, None)),
    ],
)
def test_classify_covers_row(
    spread, result, home_score, away_score, expected
):
    assert classify_covers_row(spread, result, home_score, away_score) == expected


def _create_test_database(path):
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE Betting (
            game_id TEXT PRIMARY KEY,
            covers_closing_spread REAL,
            spread_result TEXT,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE GameStates (
            game_id TEXT,
            home_score INTEGER,
            away_score INTEGER,
            is_final_state INTEGER
        );

        INSERT INTO Betting VALUES ('keep', -3.5, 'W', 'old');
        INSERT INTO Betting VALUES ('flip', -3.5, 'W', 'old');
        INSERT INTO Betting VALUES ('ambiguous', 3.5, 'W', 'old');
        INSERT INTO Betting VALUES ('unfinished', -2.5, NULL, 'old');

        INSERT INTO GameStates VALUES ('keep', 110, 100, 1);
        INSERT INTO GameStates VALUES ('flip', 100, 105, 1);
        INSERT INTO GameStates VALUES ('ambiguous', 100, 100, 1);
        INSERT INTO GameStates VALUES ('unfinished', 50, 48, 0);
        """
    )
    conn.commit()
    conn.close()


def test_dry_run_does_not_modify_database(tmp_path):
    database = tmp_path / "covers.sqlite"
    _create_test_database(database)

    counts = normalize_database(database)

    assert counts == {"keep": 1, "flip": 1, "invalidate": 2, "total": 4}
    conn = sqlite3.connect(database)
    rows = conn.execute(
        "SELECT game_id, covers_closing_spread, spread_result FROM Betting ORDER BY game_id"
    ).fetchall()
    conn.close()
    assert rows == [
        ("ambiguous", 3.5, "W"),
        ("flip", -3.5, "W"),
        ("keep", -3.5, "W"),
        ("unfinished", -2.5, None),
    ]


def test_apply_flips_safe_rows_and_invalidates_ambiguous_rows(tmp_path):
    database = tmp_path / "covers.sqlite"
    _create_test_database(database)

    counts = normalize_database(database, apply=True)

    assert counts == {"keep": 1, "flip": 1, "invalidate": 2, "total": 4}
    conn = sqlite3.connect(database)
    rows = conn.execute(
        "SELECT game_id, covers_closing_spread, spread_result FROM Betting ORDER BY game_id"
    ).fetchall()
    conn.close()
    assert rows == [
        ("ambiguous", None, None),
        ("flip", 3.5, "L"),
        ("keep", -3.5, "W"),
        ("unfinished", None, None),
    ]
