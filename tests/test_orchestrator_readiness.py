"""Regression tests for prediction failures and coverage reporting."""

import json
import sqlite3

import pytest

from src.pipeline import orchestrator
from src.predictions import prediction_manager


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    db_path = tmp_path / "pipeline.sqlite"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE Predictions (game_id TEXT, predictor TEXT)")
    # Baseline needs no files, so exercise failures without loading ML runtimes.
    monkeypatch.setattr(orchestrator, "PROJECT_ROOT", tmp_path)
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
