import sqlite3

import pytest

from src.pipeline.roster_assembler import NEW_PLAYER_MINUTES, RosterAssembler


@pytest.fixture
def roster_db(tmp_path):
    path = tmp_path / "rosters.sqlite"
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE Teams (team_id INTEGER, abbreviation TEXT);
        CREATE TABLE Games (
            game_id TEXT, home_team TEXT, away_team TEXT, date_time_utc TEXT,
            status INTEGER, season_type TEXT
        );
        CREATE TABLE Players (person_id INTEGER, team TEXT, roster_status INTEGER);
        CREATE TABLE PlayerBox (game_id TEXT, team_id INTEGER, player_id INTEGER, min REAL);
        CREATE TABLE InjuryReports (nba_player_id INTEGER, team TEXT, status TEXT,
                                    report_timestamp TEXT);
        INSERT INTO Teams VALUES (1, 'BOS'), (2, 'NYK');
        INSERT INTO Games VALUES
            ('opener', 'BOS', 'NYK', '2026-10-20T23:00:00Z', 1, 'Regular Season');
    """)
    yield conn, RosterAssembler(path)
    conn.close()


def add_game(conn, game_id, date, boxes=(), season_type="Regular Season", status=3):
    conn.execute(
        "INSERT INTO Games VALUES (?, 'BOS', 'NYK', ?, ?, ?)",
        (game_id, date, status, season_type),
    )
    conn.executemany(
        "INSERT INTO PlayerBox VALUES (?, ?, ?, ?)",
        [(game_id, team, player, minutes) for team, player, minutes in boxes],
    )


def test_current_rookies_project_without_past_games(roster_db):
    conn, assembler = roster_db
    conn.executemany(
        "INSERT INTO Players VALUES (?, ?, ?)",
        [(11, "BOS", 1), (12, "BOS", 1), (13, "BOS", 0), (21, "NYK", 1)],
    )
    roster = assembler._get_recent_roster(conn, "BOS", "2026-10-20T23:00:00Z")
    assert roster == [
        {"player_id": 11, "avg_minutes": NEW_PLAYER_MINUTES},
        {"player_id": 12, "avg_minutes": NEW_PLAYER_MINUTES},
    ]
    conn.commit()
    projected = assembler.get_projected_rosters(["opener"])["opener"]
    assert projected["home_players"] == [11, 12]
    assert projected["away_players"] == [21]


def test_transfers_use_last_three_games_and_rookies_remain(roster_db):
    conn, assembler = roster_db
    conn.executemany(
        "INSERT INTO Players VALUES (?, ?, 1)",
        [(11, "BOS"), (12, "BOS"), (13, "BOS"), (99, "NYK")],
    )
    for index, minutes in enumerate([48, 48, 6, 12, 18], start=1):
        add_game(
            conn, f"old{index}", f"2026-04-0{index}T23:00:00Z",
            [(1, 11, 30), (1, 99, 40), (2, 12, minutes)],
        )
    add_game(conn, "preseason", "2026-10-03T23:00:00Z", [(2, 12, 45)], "Pre Season")
    add_game(conn, "future", "2026-10-21T23:00:00Z", [(2, 12, 45)])
    add_game(conn, "unfinished", "2026-10-19T23:00:00Z", [(2, 12, 45)], status=2)
    roster = assembler._get_recent_roster(conn, "BOS", "2026-10-20T23:00:00Z")
    assert roster == [
        {"player_id": 11, "avg_minutes": 30.0},
        {"player_id": 12, "avg_minutes": 12.0},
        {"player_id": 13, "avg_minutes": NEW_PLAYER_MINUTES},
    ]


def test_recent_team_games_require_usable_regular_season_boxes(roster_db):
    conn, assembler = roster_db
    add_game(conn, "regular", "2026-04-01T23:00:00Z", [(1, 11, 25)])
    add_game(conn, "missing1", "2026-04-02T23:00:00Z")
    add_game(conn, "missing2", "2026-04-03T23:00:00Z", [(2, 21, 30)])
    add_game(conn, "dnp", "2026-04-04T23:00:00Z", [(1, 12, 0)])
    add_game(conn, "preseason", "2026-10-03T23:00:00Z", [(1, 13, 45)], "Pre Season")
    assert assembler._get_recent_roster(conn, "BOS", "2026-10-20T23:00:00Z") == [
        {"player_id": 11, "avg_minutes": 25.0},
    ]


def test_rookie_rosters_still_subtract_injuries_and_cap_players(roster_db):
    conn, assembler = roster_db
    conn.executemany("INSERT INTO Players VALUES (?, 'BOS', 1)", [(pid,) for pid in range(1, 18)])
    conn.executemany("INSERT INTO InjuryReports VALUES (?, 'BOS', ?, ?)", [
        (1, "Out", "2026-10-20T12:00:00Z"),
        (2, "Questionable", "2026-10-20T12:00:00Z"),
    ])
    conn.commit()
    projected = assembler.get_projected_rosters(["opener"])["opener"]
    assert projected["home_players"] == list(range(2, 17))
    assert projected["home_confidence"] == 0.95
    assert "BOS: 1 player(s) out" in projected["warnings"]
