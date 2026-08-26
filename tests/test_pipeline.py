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
