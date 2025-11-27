"""
test_config.py

Tests for configuration loading and validation.
"""

import os
from pathlib import Path

import pytest


class TestConfigLoading:
    """Tests for config.py loading."""

    def test_config_loads(self):
        """Config should load without errors."""
        from src.config import config

        assert config is not None
        assert isinstance(config, dict)

    def test_required_keys_exist(self):
        """Required config keys should exist."""
        from src.config import config

        required_keys = ["database", "project", "predictors", "api", "web_app"]
        for key in required_keys:
            assert key in config, f"Missing required key: {key}"

    def test_database_path_exists(self):
        """Database path should point to existing file."""
        from src.config import config

        db_path = config["database"]["path"]
        assert os.path.exists(db_path), f"Database not found: {db_path}"

    def test_project_root_exists(self):
        """Project root should point to existing directory."""
        from src.config import config

        root = config["project"]["root"]
        assert os.path.isdir(root), f"Project root not found: {root}"

    def test_predictors_configured(self):
        """At least one predictor should be configured."""
        from src.config import config

        predictors = config["predictors"]
        assert len(predictors) > 0, "No predictors configured"

    def test_default_predictor_valid(self):
        """Default predictor should be in predictors list."""
        from src.config import config

        default = config["default_predictor"]
        predictors = config["predictors"]
        assert default in predictors, f"Default predictor '{default}' not in predictors"

    def test_api_settings(self):
        """API settings should have required values."""
        from src.config import config

        api = config["api"]
        assert "valid_seasons" in api
        assert "max_game_ids" in api
        assert isinstance(api["valid_seasons"], list)
        assert isinstance(api["max_game_ids"], int)
        assert api["max_game_ids"] > 0

    def test_web_app_secret_key(self):
        """Web app should have a secret key."""
        from src.config import config

        web_app = config["web_app"]
        assert "secret_key" in web_app
        assert len(web_app["secret_key"]) > 0


class TestConfigPaths:
    """Tests for path resolution in config."""

    def test_model_paths_resolve(self):
        """Model paths should resolve to valid paths (if configured)."""
        from src.config import config

        for predictor_name, predictor_config in config["predictors"].items():
            model_paths = predictor_config.get("model_paths", [])

            # Ensemble uses predictor names, not file paths
            if predictor_name == "Ensemble":
                continue

            for path in model_paths:
                # Path should be absolute or resolvable
                assert (
                    os.path.isabs(path) or Path(path).exists()
                ), f"Invalid model path for {predictor_name}: {path}"
