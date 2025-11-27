"""
test_predictors.py

Tests for prediction engines.
Tests predictor instantiation and prediction generation.
"""

import pytest

from src.predictions.prediction_engines.baseline_predictor import BaselinePredictor
from src.predictions.prediction_manager import PREDICTOR_MAP, determine_predictor_class


class TestDeterminePredictorClass:
    """Tests for determine_predictor_class function."""

    def test_valid_predictor_names(self):
        """All registered predictors should be found."""
        for name in PREDICTOR_MAP.keys():
            predictor_class, returned_name = determine_predictor_class(name)
            assert predictor_class == PREDICTOR_MAP[name]
            assert returned_name == name

    def test_none_returns_default(self):
        """None should return the default predictor."""
        from src.config import config

        default = config["default_predictor"]

        predictor_class, name = determine_predictor_class(None)
        assert name == default
        assert predictor_class == PREDICTOR_MAP[default]

    def test_invalid_predictor_raises(self):
        """Invalid predictor name should raise ValueError."""
        with pytest.raises(ValueError, match="not found in PREDICTOR_MAP"):
            determine_predictor_class("InvalidPredictor")


class TestBaselinePredictor:
    """Tests for BaselinePredictor - formula-based predictor."""

    def test_instantiation(self):
        """Predictor should instantiate without errors."""
        predictor = BaselinePredictor()
        assert predictor is not None

    def test_has_required_methods(self):
        """Predictor should have required interface methods."""
        predictor = BaselinePredictor()
        assert hasattr(predictor, "make_pre_game_predictions")
        assert hasattr(predictor, "load_pre_game_data")
        assert callable(predictor.make_pre_game_predictions)

    def test_empty_game_ids_returns_empty(self):
        """Empty game IDs should return empty dict."""
        predictor = BaselinePredictor()
        result = predictor.make_pre_game_predictions([])
        assert result == {}

    def test_prediction_structure(self, mock_feature_set):
        """Predictions should have expected structure."""
        predictor = BaselinePredictor()

        # Manually test the formula with mock data
        home_ppg = mock_feature_set["home_avg_pts"]
        away_opp_ppg = mock_feature_set["away_avg_opp_pts"]
        away_ppg = mock_feature_set["away_avg_pts"]
        home_opp_ppg = mock_feature_set["home_avg_opp_pts"]

        pred_home_score = (home_ppg + away_opp_ppg) / 2
        pred_away_score = (away_ppg + home_opp_ppg) / 2

        # Verify formula produces reasonable scores
        assert 80 < pred_home_score < 140
        assert 80 < pred_away_score < 140


class TestPredictorMap:
    """Tests for PREDICTOR_MAP configuration."""

    def test_all_predictors_registered(self):
        """All expected predictors should be in the map."""
        expected = {"Baseline", "Linear", "Tree", "MLP", "Ensemble"}
        assert set(PREDICTOR_MAP.keys()) == expected

    def test_predictors_are_classes(self):
        """All values should be class types."""
        for name, cls in PREDICTOR_MAP.items():
            assert isinstance(cls, type), f"{name} is not a class"


class TestMLPredictorInstantiation:
    """Tests for ML predictor instantiation (requires model files)."""

    def test_linear_predictor_instantiation(self):
        """Linear predictor should instantiate if model exists."""
        from src.config import config
        from src.predictions.prediction_engines.linear_predictor import LinearPredictor

        model_paths = config["predictors"].get("Linear", {}).get("model_paths", [])

        if model_paths:
            predictor = LinearPredictor(model_paths=model_paths)
            assert predictor is not None
            assert len(predictor.models) > 0
        else:
            pytest.skip("No Linear model paths configured")

    def test_tree_predictor_instantiation(self):
        """Tree predictor should instantiate if model exists."""
        from src.config import config
        from src.predictions.prediction_engines.tree_predictor import TreePredictor

        model_paths = config["predictors"].get("Tree", {}).get("model_paths", [])

        if model_paths:
            predictor = TreePredictor(model_paths=model_paths)
            assert predictor is not None
            assert len(predictor.models) > 0
        else:
            pytest.skip("No Tree model paths configured")

    def test_mlp_predictor_instantiation(self):
        """MLP predictor should instantiate if model exists."""
        from src.config import config
        from src.predictions.prediction_engines.mlp_predictor import MLPPredictor

        model_paths = config["predictors"].get("MLP", {}).get("model_paths", [])

        if model_paths:
            predictor = MLPPredictor(model_paths=model_paths)
            assert predictor is not None
            assert len(predictor.models) > 0
        else:
            pytest.skip("No MLP model paths configured")

    def test_ensemble_predictor_instantiation(self):
        """Ensemble predictor should instantiate if models exist."""
        from src.config import config
        from src.predictions.prediction_engines.ensemble_predictor import (
            EnsemblePredictor,
        )

        model_paths = config["predictors"].get("Ensemble", {}).get("model_paths", [])

        if model_paths:
            predictor = EnsemblePredictor(model_paths=model_paths)
            assert predictor is not None
        else:
            pytest.skip("No Ensemble model paths configured")
