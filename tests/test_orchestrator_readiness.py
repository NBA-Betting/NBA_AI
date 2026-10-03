"""Regression tests for prediction failures and coverage reporting."""

import json
import sqlite3
from datetime import datetime, timezone

import pytest

from src.pipeline import orchestrator
from src.predictions import prediction_manager
from src.database_updater import database_update_manager


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    db_path = tmp_path / "pipeline.sqlite"
    with sqlite3.connect(db_path) as conn:
        conn.execute("""CREATE TABLE Predictions (
            game_id TEXT, predictor TEXT, prediction_datetime TEXT,
            prediction_set TEXT, PRIMARY KEY (game_id, predictor))""")
        conn.execute("""CREATE TABLE Games (
            game_id TEXT, status INTEGER, date_time_utc TEXT,
            season TEXT DEFAULT '2026-2027', season_type TEXT DEFAULT 'Regular Season',
            status_text TEXT DEFAULT 'Scheduled')""")
        conn.executemany(
            "INSERT INTO Games (game_id, status, date_time_utc) VALUES (?, 1, '2099-10-20T23:00:00Z')",
            [("0022600001",), ("0022600002",)],
        )
    # Baseline needs no files, so exercise failures without loading ML runtimes.
    monkeypatch.setattr(orchestrator, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(database_update_manager, "update_pre_game_data", lambda *args: None)
    return orchestrator.PipelineOrchestrator("2026-2027", str(db_path))


@pytest.mark.parametrize("raises", [True, False])
def test_zero_predictions_cannot_report_success(pipeline, monkeypatch, raises):
    def predict(*args, **kwargs):
        if raises:
            raise ValueError("missing features")
        return {}

    monkeypatch.setattr(prediction_manager, "make_pre_game_predictions", predict)
    monkeypatch.setattr(pipeline, "_stage_refresh_injuries", lambda _: {"status": "ok"})
    monkeypatch.setattr(pipeline, "_stage_refresh_betting", lambda _: {"status": "ok"})
    monkeypatch.setattr(
        pipeline, "_stage_find_todays_games",
        lambda: {"status": "ok", "game_ids": ["0022600001"]},
    )

    result = pipeline.run_pre_game()

    assert result["status"] != "success"
    assert result["predictions_generated"] == 0
    assert result["errors"]
    with sqlite3.connect(pipeline.db_path) as conn:
        status, errors = conn.execute("SELECT status, errors FROM PipelineRuns").fetchone()
    assert status != "success"
    assert json.loads(errors)


def test_partial_prediction_coverage_is_reported(pipeline, monkeypatch):
    monkeypatch.setattr(
        prediction_manager, "make_pre_game_predictions",
        lambda *args, **kwargs: {"0022600001": {}},
    )
    result = pipeline._stage_generate_predictions(["0022600001", "0022600002"], False)
    assert result["status"] == "partial"
    assert "0022600002" in result["error"]


@pytest.mark.parametrize("mode", ["pre-game", "full"])
def test_cli_returns_failure_for_pipeline_errors(pipeline, monkeypatch, mode, capsys):
    run = {"errors": ["Tree: missing features"], "warnings": [], "total_time": 0}
    monkeypatch.setattr(orchestrator, "PipelineOrchestrator", lambda **_: pipeline)
    monkeypatch.setattr(pipeline, "run_pre_game", lambda **_: run)
    monkeypatch.setattr(
        pipeline, "run_full",
        lambda **_: {"post_game": run, "pre_game": run, "total_time": 0},
    )
    monkeypatch.setattr(orchestrator.sys, "argv", ["orchestrator", "--mode", mode])
    assert orchestrator.main() == 1
    assert "ERROR: Tree: missing features" in capsys.readouterr().out


def test_existing_forecasts_refresh_and_started_games_are_preserved(pipeline, monkeypatch):
    old = {"pred_home_score": 100, "pred_away_score": 90}
    new = {"pred_home_score": 110, "pred_away_score": 105}
    with sqlite3.connect(pipeline.db_path) as conn:
        conn.execute("UPDATE Games SET status = 2 WHERE game_id = '0022600002'")
        conn.execute("INSERT INTO Games (game_id, status, date_time_utc) VALUES ('0022600003', 1, '2020-10-20T23:00:00Z')")
        conn.executemany(
            "INSERT INTO Predictions VALUES (?, 'Baseline', '2020-01-01', ?)",
            [(gid, json.dumps(old)) for gid in ["0022600001", "0022600002", "0022600003"]],
        )
    calls = []

    def predict(ids, name, save):
        calls.append((ids, name))
        return {gid: new for gid in ids}

    monkeypatch.setattr(prediction_manager, "make_pre_game_predictions", predict)
    result = pipeline._stage_generate_predictions(
        ["0022600001", "0022600002", "0022600003"], False,
    )

    assert result["status"] == "ok"
    assert calls == [(["0022600001"], "Baseline")]
    with sqlite3.connect(pipeline.db_path) as conn:
        rows = conn.execute("SELECT game_id, prediction_set FROM Predictions").fetchall()
    assert dict((gid, json.loads(pred)) for gid, pred in rows) == {
        "0022600001": new, "0022600002": old, "0022600003": old,
    }


def test_preseason_is_never_predicted(pipeline, monkeypatch):
    with sqlite3.connect(pipeline.db_path) as conn:
        conn.execute("UPDATE Games SET season_type = 'Pre Season'")
    def predict(*args, **kwargs):
        pytest.fail("Preseason predictions must not run")
    monkeypatch.setattr(prediction_manager, "make_pre_game_predictions", predict)
    result = pipeline._stage_generate_predictions(["0022600001"], False)
    assert result["status"] == "skipped"


@pytest.mark.parametrize("day, offset", [("2026-10-20", 4), ("2026-11-01", 4), ("2027-03-14", 5)])
def test_today_uses_eastern_dates_and_supported_season(pipeline, monkeypatch, day, offset):
    now = datetime.fromisoformat(f"{day}T{offset:02}:05:00+00:00")
    monkeypatch.setattr(orchestrator, "get_utc_now", lambda: now)
    monkeypatch.setattr(orchestrator, "get_current_eastern_datetime", lambda: now.astimezone(orchestrator.get_eastern_tz()))
    with sqlite3.connect(pipeline.db_path) as conn:
        conn.execute("DELETE FROM Games")
        conn.executemany("INSERT INTO Games VALUES (?, 1, ?, ?, ?, ?)", [
            ("regular", f"{day}T{offset:02}:30:00Z", "2026-2027", "Regular Season", "Scheduled"),
            ("preseason", f"{day}T23:00:00Z", "2026-2027", "Pre Season", "Scheduled"),
            ("other-season", f"{day}T23:00:00Z", "2025-2026", "Regular Season", "Scheduled"),
            ("postponed", f"{day}T23:00:00Z", "2026-2027", "Regular Season", "PPD"),
            ("started", f"{day}T{offset:02}:00:00Z", "2026-2027", "Regular Season", "Scheduled"),
        ])
    assert pipeline._stage_find_todays_games()["game_ids"] == ["regular"]


def test_missing_season_schedule_is_an_error(pipeline):
    with sqlite3.connect(pipeline.db_path) as conn:
        conn.execute("DELETE FROM Games")
    with pytest.raises(RuntimeError, match="No schedule loaded for 2026-2027"):
        pipeline._stage_find_todays_games()
