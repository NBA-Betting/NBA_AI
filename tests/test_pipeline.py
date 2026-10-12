"""
test_pipeline.py

Smoke tests for the production pipeline components.
Verifies that all pipeline modules can be imported and that
key integration points (predictor registry, web app) work.
"""

import subprocess
import sys


def _assert_imports(module, name):
    """Import a native ML module in a child process to avoid runtime conflicts."""
    result = subprocess.run(
        [sys.executable, "-c", f"from {module} import {name}"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


class TestPipelineImports:
    """Verify that all pipeline modules can be imported."""

    def test_phase5_predictor_imports(self):
        _assert_imports("src.pipeline.phase5_predictor", "Phase5Predictor")

    def test_phase3_predictor_imports(self):
        _assert_imports("src.pipeline.phase3_predictor", "Phase3Predictor")

    def test_ensemble_predictor_imports(self):
        _assert_imports("src.pipeline.ensemble_predictor", "EnsemblePredictor")

    def test_orchestrator_imports(self):
        _assert_imports("src.pipeline.orchestrator", "PipelineOrchestrator")


class TestPredictorRegistry:
    """Verify that all predictors are registered in the prediction manager."""

    def test_all_predictors_registered(self):
        from src.predictions.prediction_manager import VALID_PREDICTORS

        expected = {"Baseline", "Linear", "Tree", "MLP", "Phase5", "Phase3", "Ensemble"}
        assert expected == VALID_PREDICTORS


class TestWebAppIntegration:
    """Verify that the web app creates and serves basic routes."""

    def test_web_app_creates(self):
        from src.web_app.app import create_app

        app = create_app("Phase5")
        assert app is not None

    def test_dashboard_endpoint(self):
        from src.web_app.app import create_app

        app = create_app("Phase5")
        app.config["TESTING"] = True
        with app.test_client() as client:
            r = client.get("/dashboard?predictor=Phase5")
            assert r.status_code == 200


def test_l1_updater_waits_for_pbp():
    """A recent game without PBP stats holds back itself and every later
    game (the Kalman filter only moves forward); once it's older than
    PBP_WAIT_DAYS it goes ahead (with PBP masked; not checked here).
    (Child process: the updater loads torch.)"""
    code = """
from datetime import datetime, timedelta, timezone
from src.pipeline.l1_updater import PBP_WAIT_DAYS, L1IncrementalUpdater

now = datetime.now(timezone.utc)
day = lambda ago: (now - timedelta(days=ago)).strftime("%Y-%m-%dT%H:%M:%SZ")
seq = {
    "game_ids": ["a", "b", "c", "d"],
    "dates": [day(10), day(PBP_WAIT_DAYS + 1), day(1), day(0.5)],
    "has_pbp": [True, False, False, True],
}
# b is old enough to go ahead without PBP; c is recent, so c and d wait
assert L1IncrementalUpdater._ready_end(seq, 0) == 2
assert L1IncrementalUpdater._ready_end(seq, 2) == 2
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_l1_incremental_update_matches_full_extraction():
    """The daily updater continues the Kalman filter game by game
    (kalman_steps); training vectors come from filtering the whole career at
    once (extract_player). For the same games they must agree."""
    code = """
import sys
from pathlib import Path
from src.pipeline.l1_updater import L1_CHECKPOINT, L1IncrementalUpdater
if not L1_CHECKPOINT.exists():
    sys.exit(77)
import numpy as np
from src.database import get_db
from src.phase5.cache_builder import build_player_cache
from src.phase5.l1_extract import extract_player, kalman_steps

updater = L1IncrementalUpdater(device="cpu")
with get_db(str(updater.db_path)) as conn:
    seq = build_player_cache(conn, 1628983, updater.team_stats, updater.team_id_to_abbr,
                             min_games=1, require_profile=False)
seq = {k: v[:200] for k, v in seq.items()}  # one window: no overlap stitching
profile = updater._get_profile_tensor(1628983)
full = extract_player(seq, updater.model, updater.normalizer, profile, updater.device)
head = extract_player({k: v[:150] for k, v in seq.items()}, updater.model,
                      updater.normalizer, profile, updater.device)
ability, P = kalman_steps(seq, 150, 200, head["ability"][-1], head["uncertainty"][-1],
                          updater.model, updater.normalizer, updater.device)
np.testing.assert_allclose(ability, full["ability"][150:200], atol=1e-4)
np.testing.assert_allclose(P, full["uncertainty"][150:200], atol=1e-4)
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    if result.returncode == 77:
        import pytest

        pytest.skip("needs the Phase5 L1 model")
    assert result.returncode == 0, result.stderr[-2000:]
