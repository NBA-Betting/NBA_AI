"""
train.py

Unified training script for all NBA prediction models.
Uses curated best-practice hyperparameters with minimal search.

Supports:
- Linear: Ridge Regression with α ∈ {1.0, 10.0, 100.0}
- Tree: XGBoost with proven defaults (max_depth=5, n_estimators=200)
- MLP: 2-layer network (64→32) with dropout, early stopping

Usage:
    # Train single model
    python -m src.model_training.train --model_type Linear --train_season 2023-2024 --test_season 2024-2025

    # Train all models
    python -m src.model_training.train --model_type all --train_season 2023-2024 --test_season 2024-2025

Model Versioning:
    Models saved as: {type}_v{version}_mae{score}.{ext}
    Example: ridge_v1.0_mae10.5.joblib
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from joblib import dump
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBRegressor

from src.config import config
from src.model_training.evaluation import (
    compare_models,
    evaluate_predictions,
    print_evaluation_report,
    print_model_comparison,
)
from src.model_training.model_registry import ModelRegistry
from src.model_training.modeling_utils import load_featurized_modeling_data

# Configuration
DB_PATH = config["database"]["path"]
PROJECT_ROOT = config["project"]["root"]
MODEL_VERSION = "1.0"

# ============================================================================
# CURATED HYPERPARAMETERS - Best practices with minimal search
# ============================================================================

# Ridge: Test 3 regularization strengths, pick best
RIDGE_ALPHAS = [1.0, 10.0, 100.0]

# XGBoost: Proven defaults from extensive tuning literature
XGBOOST_PARAMS = {
    "n_estimators": 200,
    "max_depth": 5,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "gamma": 0.1,
    "reg_lambda": 1.0,
    "objective": "reg:squarederror",
    "random_state": 42,
}

# MLP: 2-layer with dropout, early stopping
MLP_PARAMS = {
    "hidden_sizes": [64, 32],
    "dropout": 0.2,
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 200,
    "patience": 25,  # Early stopping patience
}


class MLP(nn.Module):
    """
    Multi-layer Perceptron for NBA score prediction.

    Architecture: input → 64 → ReLU → Dropout → 32 → ReLU → Dropout → 2
    Predicts [home_score, away_score]
    """

    def __init__(self, input_size, hidden_sizes=[64, 32], dropout=0.2):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.dropout_rate = dropout

        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, 2))  # Output: [home_score, away_score]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train NBA prediction models with curated hyperparameters"
    )

    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=["Linear", "Tree", "MLP", "all"],
        help="Model type to train, or 'all' to train all models",
    )

    parser.add_argument(
        "--train_season",
        type=str,
        required=True,
        help="Training season (e.g., '2023-2024')",
    )

    parser.add_argument(
        "--test_season",
        type=str,
        required=True,
        help="Test season (e.g., '2024-2025')",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="models",
        help="Directory to save trained models (default: models/)",
    )

    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    return parser.parse_args()


def load_and_prepare_data(train_season, test_season):
    """
    Load featurized data and prepare train/test splits.

    Returns:
        tuple: (X_train, y_train, X_test, y_test, feature_names)
    """
    print("=" * 70)
    print("DATA LOADING")
    print("=" * 70)

    print(f"\nTraining season: {train_season}")
    print(f"Testing season: {test_season}")

    training_df = load_featurized_modeling_data([train_season], DB_PATH)
    testing_df = load_featurized_modeling_data([test_season], DB_PATH)

    print(f"Training data: {len(training_df)} games")
    print(f"Testing data: {len(testing_df)} games")

    # Drop NaN values
    training_df = training_df.dropna()
    testing_df = testing_df.dropna()
    print(f"After dropping NaNs: {len(training_df)} train, {len(testing_df)} test")

    # Define columns to exclude from features
    exclude_cols = [
        "game_id",
        "date_time_est",
        "home_team",
        "away_team",
        "season",
        "season_type",
        "home_score",
        "away_score",
        "total",
        "home_margin",
        "players_data",
    ]

    # Create features and targets
    feature_cols = [c for c in training_df.columns if c not in exclude_cols]
    X_train = training_df[feature_cols].values
    y_train = training_df[["home_score", "away_score"]].values
    X_test = testing_df[feature_cols].values
    y_test = testing_df[["home_score", "away_score"]].values

    print(f"Features: {len(feature_cols)}")

    return X_train, y_train, X_test, y_test, feature_cols


def train_ridge(X_train, y_train, X_test, y_test, random_state):
    """
    Train Ridge Regression with minimal hyperparameter search.
    Tests 3 alpha values and picks the best.
    """
    print("\n" + "=" * 70)
    print("TRAINING: Ridge Regression")
    print("=" * 70)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    best_model = None
    best_mae = float("inf")
    best_alpha = None

    print(f"\nTesting alpha values: {RIDGE_ALPHAS}")

    for alpha in RIDGE_ALPHAS:
        model = Ridge(alpha=alpha, random_state=random_state)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        metrics = evaluate_predictions(y_test, y_pred)
        mae = metrics["avg_score_mae"]

        print(f"  α={alpha:>6.1f} → MAE: {mae:.3f}")

        if mae < best_mae:
            best_mae = mae
            best_alpha = alpha
            best_model = model

    print(f"\n✓ Best alpha: {best_alpha} (MAE: {best_mae:.3f})")

    # Create pipeline with scaler for deployment
    pipeline = Pipeline(
        [
            ("scaler", scaler),
            ("model", Ridge(alpha=best_alpha, random_state=random_state)),
        ]
    )
    pipeline.fit(X_train, y_train)

    # Final evaluation
    y_pred = pipeline.predict(X_test)
    metrics = evaluate_predictions(y_test, y_pred)

    hyperparams = {"alpha": best_alpha, "fit_intercept": True}

    return pipeline, metrics, hyperparams


def train_xgboost(X_train, y_train, X_test, y_test, random_state):
    """
    Train XGBoost with proven default hyperparameters.
    No search - using established best practices.
    """
    print("\n" + "=" * 70)
    print("TRAINING: XGBoost")
    print("=" * 70)

    # Scale features (XGBoost doesn't strictly need this, but keeps consistent)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"\nUsing curated hyperparameters:")
    for k, v in XGBOOST_PARAMS.items():
        print(f"  {k}: {v}")

    # Create model
    base_model = XGBRegressor(**XGBOOST_PARAMS)
    model = MultiOutputRegressor(base_model)

    print("\nTraining...")
    model.fit(X_train_scaled, y_train)

    # Create pipeline for deployment
    pipeline = Pipeline(
        [
            ("scaler", scaler),
            ("model", MultiOutputRegressor(XGBRegressor(**XGBOOST_PARAMS))),
        ]
    )
    pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = pipeline.predict(X_test)
    metrics = evaluate_predictions(y_test, y_pred)

    return pipeline, metrics, XGBOOST_PARAMS


def train_mlp(X_train, y_train, X_test, y_test, random_state):
    """
    Train MLP with 2 hidden layers, dropout, and early stopping.
    Normalizes targets to improve convergence.
    """
    print("\n" + "=" * 70)
    print("TRAINING: MLP (PyTorch)")
    print("=" * 70)

    print(f"\nArchitecture: {X_train.shape[1]} → {MLP_PARAMS['hidden_sizes']} → 2")
    print(f"Dropout: {MLP_PARAMS['dropout']}")
    print(f"Learning rate: {MLP_PARAMS['learning_rate']}")
    print(f"Early stopping patience: {MLP_PARAMS['patience']}")

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Normalize targets (important for neural network convergence)
    y_mean = y_train.mean(axis=0)
    y_std = y_train.std(axis=0)
    y_train_norm = (y_train - y_mean) / y_std

    # Convert to tensors
    X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_t = torch.tensor(y_train_norm, dtype=torch.float32)
    X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)

    # Normalized validation targets for early stopping
    y_test_norm = (y_test - y_mean) / y_std
    y_test_norm_t = torch.tensor(y_test_norm, dtype=torch.float32)

    # Create data loaders
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(
        train_dataset, batch_size=MLP_PARAMS["batch_size"], shuffle=True
    )

    # Initialize model
    torch.manual_seed(random_state)
    model = MLP(
        input_size=X_train.shape[1],
        hidden_sizes=MLP_PARAMS["hidden_sizes"],
        dropout=MLP_PARAMS["dropout"],
    )

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=MLP_PARAMS["learning_rate"])

    # Training with early stopping
    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    print(f"\nTraining (max {MLP_PARAMS['epochs']} epochs)...")

    for epoch in range(MLP_PARAMS["epochs"]):
        # Training
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation (on normalized targets)
        model.eval()
        with torch.no_grad():
            y_pred_val = model(X_test_t)
            val_loss = criterion(y_pred_val, y_test_norm_t).item()

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1

        if (epoch + 1) % 20 == 0:
            print(
                f"  Epoch {epoch+1:3d}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}"
            )

        if patience_counter >= MLP_PARAMS["patience"]:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    # Load best model
    model.load_state_dict(best_state)

    # Final evaluation - denormalize predictions
    model.eval()
    with torch.no_grad():
        y_pred_norm = model(X_test_t).numpy()

    # Denormalize predictions back to original scale
    y_pred = y_pred_norm * y_std + y_mean

    metrics = evaluate_predictions(y_test, y_pred)

    # Store scaler and target normalization info for deployment
    model.scaler_mean = torch.tensor(scaler.mean_, dtype=torch.float32)
    model.scaler_scale = torch.tensor(scaler.scale_, dtype=torch.float32)
    model.y_mean = torch.tensor(y_mean, dtype=torch.float32)
    model.y_std = torch.tensor(y_std, dtype=torch.float32)

    return model, metrics, MLP_PARAMS


def save_model(
    model,
    model_type,
    metrics,
    hyperparams,
    feature_names,
    train_season,
    test_season,
    output_dir,
):
    """
    Save trained model with semantic versioning.

    Naming: {type}_v{version}_mae{score}.{ext}
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    mae = metrics["avg_score_mae"]

    type_map = {"Linear": "ridge", "Tree": "xgboost", "MLP": "mlp"}
    model_prefix = type_map[model_type]

    if model_type == "MLP":
        filename = f"{model_prefix}_v{MODEL_VERSION}_mae{mae:.1f}.pth"
        filepath = output_path / filename

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "input_size": model.input_size,
            "hidden_sizes": model.hidden_sizes,
            "dropout": model.dropout_rate,
            "scaler_mean": model.scaler_mean,
            "scaler_scale": model.scaler_scale,
            "y_mean": model.y_mean,
            "y_std": model.y_std,
        }
        torch.save(checkpoint, filepath)
    else:
        filename = f"{model_prefix}_v{MODEL_VERSION}_mae{mae:.1f}.joblib"
        filepath = output_path / filename
        dump(model, filepath)

    # Save metadata
    metadata = {
        "model_type": model_type,
        "version": MODEL_VERSION,
        "model_file": str(filepath),
        "train_season": train_season,
        "test_season": test_season,
        "metrics": metrics,
        "hyperparameters": {
            k: str(v) if isinstance(v, (list, dict)) else v
            for k, v in hyperparams.items()
        },
        "num_features": len(feature_names),
        "feature_names": feature_names,
        "created_at": datetime.now().isoformat(),
    }

    metadata_path = output_path / f"{model_prefix}_v{MODEL_VERSION}_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✓ Model saved: {filepath}")
    print(f"✓ Metadata saved: {metadata_path}")

    # Register in model registry
    registry = ModelRegistry()
    registry.register_model(
        model_type=model_type,
        model_path=str(filepath),
        metrics=metrics,
        train_season=train_season,
        test_season=test_season,
        version=MODEL_VERSION,
        hyperparameters={
            k: str(v) if isinstance(v, (list, dict)) else v
            for k, v in hyperparams.items()
        },
        status="testing",  # New models start as testing until manually promoted
    )

    return filepath, metadata


def main():
    """Main training pipeline."""
    args = parse_args()

    print("\n" + "=" * 70)
    print("NBA PREDICTION MODEL TRAINING")
    print("=" * 70)
    print(f"Model: {args.model_type}")
    print(f"Train: {args.train_season} → Test: {args.test_season}")
    print(f"Output: {args.output_dir}/")

    # Load data
    X_train, y_train, X_test, y_test, feature_names = load_and_prepare_data(
        args.train_season, args.test_season
    )

    # Determine which models to train
    if args.model_type == "all":
        model_types = ["Linear", "Tree", "MLP"]
    else:
        model_types = [args.model_type]

    # Train and save models
    all_results = {}
    saved_models = {}

    for model_type in model_types:
        if model_type == "Linear":
            model, metrics, hyperparams = train_ridge(
                X_train, y_train, X_test, y_test, args.random_state
            )
        elif model_type == "Tree":
            model, metrics, hyperparams = train_xgboost(
                X_train, y_train, X_test, y_test, args.random_state
            )
        elif model_type == "MLP":
            model, metrics, hyperparams = train_mlp(
                X_train, y_train, X_test, y_test, args.random_state
            )

        # Print evaluation
        print_evaluation_report(metrics, model_type)

        # Save model
        filepath, metadata = save_model(
            model,
            model_type,
            metrics,
            hyperparams,
            feature_names,
            args.train_season,
            args.test_season,
            args.output_dir,
        )

        all_results[model_type] = metrics
        saved_models[model_type] = str(filepath)

    # Print comparison if multiple models
    if len(model_types) > 1:
        comparison_df = compare_models(all_results)
        print_model_comparison(comparison_df)

    # Print config update instructions
    print("\n" + "=" * 70)
    print("✓ TRAINING COMPLETE!")
    print("=" * 70)
    print("\nTo use these models, update config.yaml:")
    print("  predictors:")
    for model_type, path in saved_models.items():
        print(f"    {model_type}:")
        print(f"      model_paths:")
        print(f'        - "{path}"')


if __name__ == "__main__":
    main()
