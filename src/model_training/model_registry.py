"""
model_registry.py

JSON-based model registry for tracking trained models, versions, and performance metrics.

Features:
- Register new models after training
- Track model versions, metrics, and status (active/archived)
- Get best model for a given type
- Auto-deploy models by updating config.yaml

Usage:
    from src.model_training.model_registry import ModelRegistry

    registry = ModelRegistry()

    # Register a newly trained model
    registry.register_model(
        model_type="Linear",
        model_path="models/ridge_v1.0_mae10.5.joblib",
        metrics={"avg_score_mae": 10.5, "win_accuracy": 0.62},
        train_season="2023-2024",
        test_season="2024-2025"
    )

    # Get the best model for a type
    best = registry.get_best_model("Linear")

    # Deploy a model (update config.yaml)
    registry.deploy_model("Linear", "models/ridge_v1.0_mae10.5.joblib")
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml


class ModelRegistry:
    """
    JSON-based registry for tracking trained ML models.

    Stores model metadata including:
    - Model type (Linear, Tree, MLP, Ensemble)
    - Version and file path
    - Training/test seasons
    - Performance metrics
    - Status (active, testing, archived)
    """

    def __init__(self, registry_path: str = "models/registry.json"):
        """
        Initialize model registry.

        Args:
            registry_path: Path to registry JSON file
        """
        self.registry_path = Path(registry_path)
        self.models = self._load_registry()

    def _load_registry(self) -> list:
        """Load existing registry or create new one."""
        if self.registry_path.exists():
            with open(self.registry_path, "r") as f:
                data = json.load(f)
                return data.get("models", [])
        return []

    def _save_registry(self) -> None:
        """Save registry to JSON file."""
        self.registry_path.parent.mkdir(exist_ok=True)
        data = {"updated_at": datetime.now().isoformat(), "models": self.models}
        with open(self.registry_path, "w") as f:
            json.dump(data, f, indent=2)

    def register_model(
        self,
        model_type: str,
        model_path: str,
        metrics: dict,
        train_season: str,
        test_season: str,
        version: str = "1.0",
        hyperparameters: Optional[dict] = None,
        status: str = "testing",
    ) -> dict:
        """
        Register a newly trained model.

        Args:
            model_type: Type of model (Linear, Tree, MLP, Ensemble)
            model_path: Path to saved model file
            metrics: Performance metrics dict (must include 'avg_score_mae')
            train_season: Season used for training
            test_season: Season used for testing
            version: Model version string
            hyperparameters: Optional dict of hyperparameters used
            status: Initial status (testing, active, archived)

        Returns:
            The registered model entry dict
        """
        model_entry = {
            "id": len(self.models) + 1,
            "model_type": model_type,
            "version": version,
            "path": model_path,
            "train_season": train_season,
            "test_season": test_season,
            "metrics": metrics,
            "hyperparameters": hyperparameters or {},
            "status": status,
            "created_at": datetime.now().isoformat(),
        }

        self.models.append(model_entry)
        self._save_registry()

        logging.info(f"Registered model: {model_type} v{version} ({model_path})")
        return model_entry

    def get_models(
        self, model_type: Optional[str] = None, status: Optional[str] = None
    ) -> list:
        """
        Get models filtered by type and/or status.

        Args:
            model_type: Filter by model type (Linear, Tree, MLP, Ensemble)
            status: Filter by status (active, testing, archived)

        Returns:
            List of matching model entries
        """
        results = self.models

        if model_type:
            results = [m for m in results if m["model_type"] == model_type]

        if status:
            results = [m for m in results if m["status"] == status]

        return results

    def get_best_model(self, model_type: str, status: str = "active") -> Optional[dict]:
        """
        Get the best model for a given type based on avg_score_mae.

        Args:
            model_type: Model type to find best for
            status: Only consider models with this status

        Returns:
            Best model entry or None if no models found
        """
        models = self.get_models(model_type=model_type, status=status)

        if not models:
            return None

        # Sort by avg_score_mae (lower is better)
        models_with_mae = [m for m in models if "avg_score_mae" in m.get("metrics", {})]

        if not models_with_mae:
            return models[0]  # Return first if no MAE metrics

        return min(models_with_mae, key=lambda m: m["metrics"]["avg_score_mae"])

    def set_status(self, model_path: str, status: str) -> bool:
        """
        Update the status of a model.

        Args:
            model_path: Path to the model file
            status: New status (active, testing, archived)

        Returns:
            True if model found and updated, False otherwise
        """
        for model in self.models:
            if model["path"] == model_path:
                model["status"] = status
                model["updated_at"] = datetime.now().isoformat()
                self._save_registry()
                logging.info(f"Updated {model_path} status to {status}")
                return True

        logging.warning(f"Model not found in registry: {model_path}")
        return False

    def promote_to_active(self, model_path: str) -> bool:
        """
        Promote a model to active status, archiving any current active model of same type.

        Args:
            model_path: Path to model to promote

        Returns:
            True if successful
        """
        # Find the model to promote
        target_model = None
        for model in self.models:
            if model["path"] == model_path:
                target_model = model
                break

        if not target_model:
            logging.error(f"Model not found: {model_path}")
            return False

        # Archive current active model of same type
        for model in self.models:
            if (
                model["model_type"] == target_model["model_type"]
                and model["status"] == "active"
                and model["path"] != model_path
            ):
                model["status"] = "archived"
                model["updated_at"] = datetime.now().isoformat()
                logging.info(f"Archived previous active model: {model['path']}")

        # Promote target model
        target_model["status"] = "active"
        target_model["updated_at"] = datetime.now().isoformat()
        self._save_registry()

        logging.info(f"Promoted to active: {model_path}")
        return True

    def deploy_model(
        self, model_type: str, model_path: str, config_path: str = "config.yaml"
    ) -> bool:
        """
        Deploy a model by updating config.yaml with its path.

        Args:
            model_type: Type of model (Linear, Tree, MLP)
            model_path: Path to the model file
            config_path: Path to config.yaml

        Returns:
            True if config updated successfully
        """
        config_file = Path(config_path)

        if not config_file.exists():
            logging.error(f"Config file not found: {config_path}")
            return False

        # Load config
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)

        # Update model path
        if "predictors" not in config:
            config["predictors"] = {}

        if model_type not in config["predictors"]:
            config["predictors"][model_type] = {}

        config["predictors"][model_type]["model_paths"] = [model_path]

        # Also update Ensemble if it exists
        if "Ensemble" in config["predictors"]:
            ensemble_paths = config["predictors"]["Ensemble"].get("model_paths", {})
            if isinstance(ensemble_paths, dict):
                ensemble_paths[model_type] = [model_path]
                config["predictors"]["Ensemble"]["model_paths"] = ensemble_paths

        # Save config
        with open(config_file, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        logging.info(f"Deployed {model_type} model to config: {model_path}")

        # Promote in registry
        self.promote_to_active(model_path)

        return True

    def list_models(self) -> None:
        """Print a formatted list of all registered models."""
        if not self.models:
            print("No models registered.")
            return

        print(f"\n{'='*80}")
        print("  Model Registry")
        print(f"{'='*80}")
        print(
            f"  {'ID':<4}{'Type':<10}{'Version':<8}{'MAE':<8}{'Status':<10}{'Path':<40}"
        )
        print(f"  {'-'*76}")

        for model in self.models:
            mae = model.get("metrics", {}).get("avg_score_mae", "N/A")
            mae_str = f"{mae:.1f}" if isinstance(mae, (int, float)) else mae
            print(
                f"  {model['id']:<4}{model['model_type']:<10}{model['version']:<8}"
                f"{mae_str:<8}{model['status']:<10}{model['path']:<40}"
            )

        print(f"{'='*80}\n")


def register_from_metadata(metadata_path: str, status: str = "testing") -> dict:
    """
    Register a model from its metadata JSON file.

    Args:
        metadata_path: Path to metadata JSON file
        status: Initial status for the model

    Returns:
        Registered model entry
    """
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    registry = ModelRegistry()

    return registry.register_model(
        model_type=metadata["model_type"],
        model_path=metadata["model_file"],
        metrics=metadata["metrics"],
        train_season=metadata["train_season"],
        test_season=metadata["test_season"],
        version=metadata.get("version", "1.0"),
        hyperparameters=metadata.get("hyperparameters", {}),
        status=status,
    )


if __name__ == "__main__":
    # Example usage: list all registered models
    registry = ModelRegistry()
    registry.list_models()
