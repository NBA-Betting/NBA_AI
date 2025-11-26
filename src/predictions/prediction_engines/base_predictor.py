"""
base_predictor.py

This module provides abstract base classes for NBA game predictors.
It eliminates code duplication across prediction engines by providing common functionality.

Classes:
- BasePredictor: Abstract base class for all predictors (formula-based and ML-based).
- BaseMLPredictor: Abstract base class for ML predictors that require model loading.

Design:
- BasePredictor handles common operations: loading feature sets and current game data.
- BaseMLPredictor extends BasePredictor and adds model loading functionality.
- Concrete predictors only implement prediction logic specific to their approach.

Usage:
- Formula-based predictors (Baseline) inherit from BasePredictor.
- ML predictors (Linear, Tree, MLP) inherit from BaseMLPredictor.
- Subclasses must implement make_pre_game_predictions().
"""

from abc import ABC, abstractmethod

from src.predictions.features import load_feature_sets


class BasePredictor(ABC):
    """
    Abstract base class for all predictors.

    Provides common functionality for loading pre-game data (feature sets).
    Subclasses must implement make_pre_game_predictions().
    """

    def __init__(self, model_paths=None):
        """
        Initialize predictor.

        Args:
            model_paths (list, optional): List of paths to model files.
                                         Ignored for formula-based predictors.
        """
        self.model_paths = model_paths or []

    @abstractmethod
    def make_pre_game_predictions(self, game_ids):
        """
        Generate pre-game predictions for the specified game IDs.

        Args:
            game_ids (list): List of game IDs to generate predictions for.

        Returns:
            dict: Dictionary mapping game_id to prediction dict with keys:
                  - pred_home_score (float)
                  - pred_away_score (float)
                  - pred_home_win_pct (float)
                  - pred_players (dict)
        """
        pass

    def load_pre_game_data(self, game_ids):
        """
        Load pre-game feature sets for the specified game IDs.

        Args:
            game_ids (list): List of game IDs to load features for.

        Returns:
            dict: Dictionary mapping game_id to feature dictionary.
        """
        feature_sets = load_feature_sets(game_ids)
        return feature_sets


class BaseMLPredictor(BasePredictor):
    """
    Abstract base class for ML predictors.

    Extends BasePredictor to add model loading functionality.
    Subclasses must implement:
    - load_models(): Load ML model(s) from self.model_paths
    - make_pre_game_predictions(game_ids): Generate predictions using loaded models
    """

    def __init__(self, model_paths=None):
        """
        Initialize ML predictor and load models.

        Args:
            model_paths (list, optional): List of paths to model files.
        """
        super().__init__(model_paths)
        self.models = []
        self.load_models()

    @abstractmethod
    def load_models(self):
        """
        Load ML model(s) from self.model_paths.

        Implementations should:
        1. Iterate through self.model_paths
        2. Load each model using appropriate library (joblib, torch, etc.)
        3. Append loaded models to self.models list

        Raises:
            ValueError: If model files cannot be loaded.
        """
        pass
