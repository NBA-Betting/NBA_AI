"""
No-leakage tests: what a past game is predicted from must not depend on the
game itself or anything after its tip-off.

Features, rosters and L1 states are built for past games by the same code as
live predictions (backtests, Phase5 training cache), so a leak here would make
backtests look better than live results. Each test changes results at or
after a game's tip-off (L1 vectors; TeamBox and PlayerBox, not InjuryReports
or Coaches) and checks the inputs for that game don't move.
"""

import sqlite3
from pathlib import Path

import numpy as np
import pytest

from src.database import DB_PATH
from src.phase5.l1_store import L1Store
from src.pipeline.roster_assembler import RosterAssembler
from src.pipeline.team_features import TeamFeatureComputer

# ---------------------------------------------------------------------------
# L1 states (synthetic vector files)
# ---------------------------------------------------------------------------


def test_l1_state_is_from_before_tip_off(tmp_path):
    game_dates = {
        "g1": "2026-01-01T00:00:00Z",
        "g2": "2026-01-03T00:00:00Z",
        "g3": "2026-01-05T00:00:00Z",  # the game being predicted
        "g4": "2026-01-07T00:00:00Z",
    }
    np.savez(
        tmp_path / "7.npz",
        game_ids=np.array(list(game_dates), dtype=object),
        ability=np.arange(4, dtype=np.float32)[:, None].repeat(32, axis=1),
        uncertainty=np.ones((4, 32), dtype=np.float32),
        archetype_weights=np.ones(10, dtype=np.float32),
    )
    store = L1Store(game_dates, vectors_dir=tmp_path)

    ability, _, _ = store.state_before(7, game_dates["g3"])
    assert ability[0] == 1  # state after g2, not after g3 or g4
    assert store.state_before(7, game_dates["g1"]) is None  # no earlier game


# ---------------------------------------------------------------------------
# Team features and rosters (two real seasons, copied from the full DB)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def season_db(tmp_path_factory):
    """Two recent complete seasons copied out of the full DB, plus a
    mid-season game to predict."""
    source = Path(DB_PATH)
    if not source.exists() or source.stat().st_size < 5e9:
        pytest.skip("needs the full database (multiple complete seasons)")

    path = tmp_path_factory.mktemp("as_of") / "seasons.sqlite"
    conn = sqlite3.connect(path)
    conn.execute("ATTACH DATABASE ? AS src", (f"file:{source}?mode=ro",))
    (season,) = conn.execute("""SELECT season FROM src.Games
        WHERE season_type = 'Post Season' AND status = 3
        ORDER BY date_time_utc DESC LIMIT 1""").fetchone()
    start = int(season[:4])
    seasons = (f"{start - 1}-{start}", season)
    for table, where in [
        ("Teams", ""),
        ("Coaches", ""),
        ("Games", "WHERE season IN (?, ?)"),
        (
            "TeamBox",
            "WHERE game_id IN (SELECT game_id FROM src.Games WHERE season IN (?, ?))",
        ),
        (
            "PlayerBox",
            "WHERE game_id IN (SELECT game_id FROM src.Games WHERE season IN (?, ?))",
        ),
        ("InjuryReports", "WHERE season IN (?, ?)"),
    ]:
        conn.execute(
            f"CREATE TABLE {table} AS SELECT * FROM src.{table} {where}",
            seasons if where else (),
        )
    conn.commit()
    conn.execute("DETACH src")
    game_id, tip_off = conn.execute(
        """SELECT game_id, date_time_utc FROM Games
        WHERE season = ? AND season_type = 'Regular Season' AND status = 3
        AND date_time_utc LIKE ? ORDER BY date_time_utc LIMIT 1""",
        (season, f"{start + 1}-01-15%"),
    ).fetchone()
    conn.close()
    return path, game_id, tip_off


def corrupt_from(path, tip_off):
    """Change every result at or after tip_off (the game itself included):
    more points, and minutes reversed so rotation order flips."""
    with sqlite3.connect(path) as conn:
        later = "SELECT game_id FROM Games WHERE date_time_utc >= ?"
        conn.execute(
            f"UPDATE TeamBox SET pts = pts + 50 WHERE game_id IN ({later})", (tip_off,)
        )
        conn.execute(
            f"UPDATE PlayerBox SET min = 49 - min, pts = pts + 20 WHERE game_id IN ({later})",
            (tip_off,),
        )


def test_features_and_rosters_ignore_the_game_and_later(season_db, tmp_path):
    path, game_id, tip_off = season_db
    corrupted = tmp_path / "corrupted.sqlite"
    corrupted.write_bytes(Path(path).read_bytes())
    corrupt_from(corrupted, tip_off)

    def inputs(db):
        rosters = RosterAssembler(db, historical=True).get_projected_rosters([game_id])
        features = TeamFeatureComputer(db).compute_for_games([game_id], rosters)
        return rosters[game_id], features[game_id]

    roster, features = inputs(path)
    roster_after, features_after = inputs(corrupted)

    assert roster["home_players"] == roster_after["home_players"]
    assert roster["away_players"] == roster_after["away_players"]
    for key in ("home_team_features", "away_team_features", "game_context"):
        np.testing.assert_array_equal(features[key], features_after[key], err_msg=key)
    assert features["home_coach"] == features_after["home_coach"]
    assert features["away_coach"] == features_after["away_coach"]
