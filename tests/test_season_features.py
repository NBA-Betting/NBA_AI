"""Opening-season feature priors and per-game inference coverage."""

import json
import sqlite3
from pathlib import Path

import pytest

from src.config import config
from src.database_updater.prior_states import determine_prior_states_needed, load_prior_states
from src.predictions.features import FEATURE_NAMES, create_feature_sets, save_feature_sets
from src.predictions.prediction_manager import determine_predictor_class


@pytest.fixture
def season_db(tmp_path):
    path = str(tmp_path / "season.sqlite")
    with sqlite3.connect(path) as conn:
        conn.executescript("""
            CREATE TABLE Games (
                game_id TEXT PRIMARY KEY, home_team TEXT, away_team TEXT,
                date_time_utc TEXT, status INTEGER, season TEXT, season_type TEXT
            );
            CREATE TABLE GameStates (
                game_id TEXT PRIMARY KEY, game_date TEXT, home TEXT, away TEXT,
                home_score INTEGER, away_score INTEGER, is_final_state INTEGER,
                players_data TEXT
            );
            CREATE TABLE Features (game_id TEXT PRIMARY KEY, save_datetime TEXT, feature_set TEXT);
        """)
    return path


def add_game(path, gid, date, season, *, home="BOS", away="NYK", score=None,
             season_type="Regular Season", status=None):
    with sqlite3.connect(path) as conn:
        conn.execute("INSERT INTO Games VALUES (?, ?, ?, ?, ?, ?, ?)",
                     (gid, home, away, date + "T23:30:00Z",
                      status if status is not None else (3 if score else 1), season, season_type))
        if score:
            conn.execute("INSERT INTO GameStates VALUES (?, ?, ?, ?, ?, ?, 1, '{}')",
                         (gid, date, home, away, *score))


def opener_features(path):
    add_game(path, "0022500001", "2025-12-01", "2025-2026", score=(114, 106))
    add_game(path, "0022500002", "2026-04-01", "2025-2026", home="NYK", away="BOS", score=(110, 118))
    add_game(path, "0022600001", "2026-10-20", "2026-2027")
    states = load_prior_states(determine_prior_states_needed(["0022600001"], path), path)
    return states, create_feature_sets(states, path)


def test_opener_uses_previous_season_without_offseason_workload(season_db):
    states, features = opener_features(season_db)
    assert states["0022600001"]["home_prior_season"] is True
    assert states["0022600001"]["away_prior_season"] is True
    row = features["0022600001"]
    assert list(row) == FEATURE_NAMES
    assert row["Home_PPG"] == 116
    assert row["Away_PPG"] == 108
    assert row["Day_of_Season"] == 0
    assert row["Home_Rest_Days"] == row["Away_Rest_Days"] == 7
    assert row["Home_Game_Freq"] == row["Away_Game_Freq"] == 0


def test_prior_fallback_excludes_future_preseason_and_nonfinal(season_db):
    opener_features(season_db)
    add_game(season_db, "0022500003", "2026-12-01", "2025-2026", score=(900, 900))
    add_game(season_db, "0012500001", "2025-10-01", "2025-2026", score=(900, 900), season_type="Pre Season")
    add_game(season_db, "0022500004", "2026-04-02", "2025-2026", score=(900, 900), status=2)
    add_game(season_db, "0022600002", "2026-10-22", "2026-2027", score=(900, 900))
    states = load_prior_states(determine_prior_states_needed(["0022600001"], season_db), season_db)
    for side in ("home", "away"):
        assert [s["game_id"] for s in states["0022600001"][f"{side}_prior_states"]] == ["0022500001", "0022500002"]


def test_current_season_history_replaces_only_that_teams_prior(season_db):
    opener_features(season_db)
    add_game(season_db, "0022600000", "2026-10-18", "2026-2027", away="CHI", score=(121, 109))
    states = load_prior_states(determine_prior_states_needed(["0022600001"], season_db), season_db)
    row = create_feature_sets(states, season_db)["0022600001"]
    assert [s["game_id"] for s in states["0022600001"]["home_prior_states"]] == ["0022600000"]
    assert not states["0022600001"].get("home_prior_season")
    assert states["0022600001"]["away_prior_season"]
    assert row["Home_PPG"] == 121
    assert row["Away_PPG"] == 108
    assert row["Home_Rest_Days"] == 2
    assert row["Away_Rest_Days"] == 7
    assert row["Day_of_Season"] == 2


def test_fallback_retains_missing_current_states_for_retry(season_db):
    opener_features(season_db)
    add_game(season_db, "0022600000", "2026-10-18", "2026-2027", status=3)
    states = load_prior_states(determine_prior_states_needed(["0022600001"], season_db), season_db)
    assert states["0022600001"]["missing_prior_states"] == {"home": ["0022600000"], "away": ["0022600000"]}
    assert states["0022600001"]["home_prior_season"]


def test_no_history_abstains_instead_of_inventing_scores(season_db):
    add_game(season_db, "0022600001", "2026-10-20", "2026-2027")
    states = load_prior_states(determine_prior_states_needed(["0022600001"], season_db), season_db)
    assert create_feature_sets(states, season_db) == {"0022600001": {}}


def test_partial_current_history_keeps_missing_games_visible(season_db):
    opener_features(season_db)
    add_game(season_db, "0022600000", "2026-10-18", "2026-2027", score=(121, 109))
    add_game(season_db, "0022600010", "2026-10-19", "2026-2027", status=3)
    states = load_prior_states(determine_prior_states_needed(["0022600001"], season_db), season_db)
    assert states["0022600001"]["missing_prior_states"] == {"home": ["0022600010"], "away": ["0022600010"]}
    assert not states["0022600001"].get("home_prior_season")
    row = create_feature_sets(states, season_db)["0022600001"]
    assert row["Day_of_Season"] == 2
    assert row["Home_Rest_Days"] == row["Away_Rest_Days"] == 2


@pytest.mark.parametrize("name", ["Baseline", "Linear", "Tree", "MLP"])
def test_shipped_predictors_keep_valid_opener_in_mixed_invalid_batch(season_db, monkeypatch, name):
    _, features = opener_features(season_db)
    save_feature_sets(features, season_db)
    invalid = ["{}", "not json", "[]", "null"]
    missing = dict(features["0022600001"])
    del missing["Home_PPG"]
    invalid.append(json.dumps(missing))
    bad = dict(features["0022600001"], Home_PPG="bad")
    invalid.append(json.dumps(bad))
    bad = dict(features["0022600001"], Home_PPG=float("inf"))
    invalid.append(json.dumps(bad))
    ids = [f"00226000{i:02d}" for i in range(2, len(invalid) + 2)]
    with sqlite3.connect(season_db) as conn:
        conn.executemany("INSERT INTO Features VALUES (?, NULL, ?)", zip(ids, invalid))
        # Feature JSON ordering should not change scaler/model column ordering.
        conn.execute("INSERT INTO Features VALUES ('0022600099', NULL, ?)",
                     (json.dumps(dict(reversed(list(features["0022600001"].items())))),))
    from src.predictions import features as feature_module
    from src.predictions.prediction_engines import base_predictor
    monkeypatch.setattr(base_predictor, "load_feature_sets",
                        lambda game_ids: feature_module.load_feature_sets(game_ids, season_db))
    cls, _ = determine_predictor_class(name)
    model_paths = [str(Path(__file__).resolve().parents[1] / p)
                   for p in config["predictors"].get(name, {}).get("model_paths", [])]
    predictor = cls(model_paths=model_paths)
    expected = predictor.make_pre_game_predictions(["0022600001"])
    assert set(expected) == {"0022600001"}
    assert 60 <= expected["0022600001"]["pred_home_score"] <= 200
    predictions = predictor.make_pre_game_predictions(ids + ["0022600001", "0022600099", "0022600098"])
    assert set(predictions) == {"0022600001", "0022600099"}
    assert predictions["0022600001"] == expected["0022600001"]
    assert predictions["0022600099"] == expected["0022600001"]
    assert predictor.make_pre_game_predictions(ids) == {}


def test_canonical_features_match_shipped_training_scaler():
    root = Path(__file__).resolve().parents[1]
    assert FEATURE_NAMES == json.loads((root / "models/tree/scaler.json").read_text())["feature_names"]
