"""
train.py

Unified training script for all NBA prediction models.
Eliminates code duplication across linear_model.py, xgb_model.py, and mlp_model.py.

Supports:
- Linear: Ridge Regression (scikit-learn)
- Tree: XGBoost Regression (xgboost)
- MLP: Multi-layer Perceptron (PyTorch)

Usage:
    python -m src.model_training.train --model_type Linear --train_seasons 2022-2023 --test_season 2023-2024
    python -m src.model_training.train --model_type Tree --train_seasons 2001-2022 --test_season 2023-2024
    python -m src.model_training.train --model_type MLP --train_seasons 2001-2022 --test_season 2023-2024 --epochs 100

Features:
- CLI arguments for flexible train/test splits
- Metadata tracking (hyperparams, metrics, training info)
- Consistent evaluation across all models
- Model saving with timestamps
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
from sklearn.metrics import log_loss
from sklearn.model_selection import RandomizedSearchCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBRegressor

from src.config import config
from src.model_training.evaluation import create_evaluations
from src.model_training.modeling_utils import load_featurized_modeling_data

# Configuration
DB_PATH = config["database"]["path"]
PROJECT_ROOT = config["project"]["root"]

# All available seasons for training (2001-2023)
ALL_SEASONS = [
    "2001-2002",
    "2002-2003",
    "2003-2004",
    "2004-2005",
    "2005-2006",
    "2006-2007",
    "2007-2008",
    "2008-2009",
    "2009-2010",
    "2010-2011",
    "2011-2012",
    "2012-2013",
    "2013-2014",
    "2014-2015",
    "2015-2016",
    "2016-2017",
    "2017-2018",
    "2018-2019",
    "2019-2020",
    "2020-2021",
    "2021-2022",
    "2022-2023",
]


class MLP(nn.Module):
    """Multi-layer Perceptron for score prediction."""

    def __init__(self, input_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train NBA prediction models with flexible configuration"
    )

    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=["Linear", "Tree", "MLP"],
        help="Type of model to train (Linear=Ridge, Tree=XGBoost, MLP=PyTorch)",
    )

    parser.add_argument(
        "--train_seasons",
        type=str,
        required=True,
        help="Training seasons (e.g., '2022-2023' or '2001-2022' for range)",
    )

    parser.add_argument(
        "--test_season", type=str, required=True, help="Test season (e.g., '2023-2024')"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="models",
        help="Directory to save trained models (default: models/)",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (MLP only, default: 100)",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training (MLP only, default: 32)",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Learning rate (MLP only, default: 0.001)",
    )

    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    return parser.parse_args()


def parse_season_range(season_str):
    """
    Parse season string into list of seasons.

    Examples:
        "2022-2023" → ["2022-2023"]
        "2001-2022" → ["2001-2002", "2002-2003", ..., "2021-2022"]
    """
    if "-" in season_str and len(season_str) == 9:  # Single season like "2022-2023"
        return [season_str]

    # Range like "2001-2022"
    start_year, end_year = map(int, season_str.split("-"))
    seasons = []
    for year in range(start_year, end_year):
        seasons.append(f"{year}-{year+1}")
    return seasons


def load_and_prepare_data(train_seasons, test_seasons):
    """
    Load featurized data and prepare train/test splits.

    Returns:
        tuple: (X_train, y_train, X_test, y_test, feature_names, train_df, test_df)
    """
    print("=" * 70)
    print("SECTION 1: Data Loading")
    print("=" * 70)

    print(f"\nTraining seasons: {train_seasons}")
    print(f"Testing seasons: {test_seasons}")

    training_df = load_featurized_modeling_data(train_seasons, DB_PATH)
    testing_df = load_featurized_modeling_data(test_seasons, DB_PATH)

    print(f"Training data shape: {training_df.shape}")
    print(f"Testing data shape: {testing_df.shape}")

    # Drop NaN values
    print("\nDropping rows with NaN values...")
    training_df = training_df.dropna()
    testing_df = testing_df.dropna()
    print(f"Training data shape after dropping NaNs: {training_df.shape}")
    print(f"Testing data shape after dropping NaNs: {testing_df.shape}")

    # Define columns to exclude from features
    game_info_columns = [
        "game_id",
        "date_time_est",
        "home_team",
        "away_team",
        "season",
        "season_type",
    ]
    game_results_columns = [
        "home_score",
        "away_score",
        "total",
        "home_margin",
        "players_data",
    ]

    # Create features and targets
    X_train = training_df.drop(columns=game_info_columns + game_results_columns)
    y_train = training_df[["home_score", "away_score"]]
    X_test = testing_df.drop(columns=game_info_columns + game_results_columns)
    y_test = testing_df[["home_score", "away_score"]]

    feature_names = X_train.columns.tolist()

    print(f"\nX_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")
    print(f"Number of features: {len(feature_names)}")

    return X_train, y_train, X_test, y_test, feature_names, training_df, testing_df


def preprocess_data(X_train, X_test):
    """
    Standardize features using StandardScaler.

    Returns:
        tuple: (X_train_scaled, X_test_scaled, scaler)
    """
    print("\n" + "=" * 70)
    print("SECTION 2: Data Preprocessing")
    print("=" * 70)

    print("\nStandardizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler


def train_linear_model(X_train_scaled, y_train, random_state):
    """Train Ridge Regression model with hyperparameter tuning."""
    print("\n" + "=" * 70)
    print("SECTION 3: Training Linear Model (Ridge Regression)")
    print("=" * 70)

    print("\nPerforming hyperparameter tuning...")
    param_distributions = {
        "alpha": np.logspace(-4, 4, 20),
        "fit_intercept": [True, False],
    }

    random_search = RandomizedSearchCV(
        Ridge(), param_distributions, n_iter=10, cv=5, random_state=random_state
    )
    random_search.fit(X_train_scaled, y_train)

    best_params = random_search.best_params_
    model = random_search.best_estimator_

    print(f"Best parameters: {best_params}")

    return model, best_params


def train_tree_model(X_train_scaled, y_train, random_state):
    """Train XGBoost model with hyperparameter tuning."""
    print("\n" + "=" * 70)
    print("SECTION 3: Training Tree Model (XGBoost)")
    print("=" * 70)

    print("\nPerforming hyperparameter tuning...")
    param_distributions = {
        "learning_rate": [0.01, 0.1, 0.2, 0.3],
        "max_depth": [3, 4, 5, 6, 7, 8, 9, 10],
        "n_estimators": [100, 200, 300, 400, 500],
        "subsample": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "gamma": [0, 0.25, 0.5, 1.0],
        "reg_lambda": [0.1, 1.0, 5.0, 10.0, 50.0, 100.0],
    }

    random_search = RandomizedSearchCV(
        MultiOutputRegressor(XGBRegressor(objective="reg:squarederror")),
        {f"estimator__{key}": value for key, value in param_distributions.items()},
        n_iter=10,
        cv=5,
        random_state=random_state,
    )
    random_search.fit(X_train_scaled, y_train)

    best_params = random_search.best_params_
    model = random_search.best_estimator_

    print(f"Best parameters: {best_params}")

    return model, best_params


def train_mlp_model(X_train_scaled, y_train, X_test_scaled, y_test, args):
    """Train PyTorch MLP model."""
    print("\n" + "=" * 70)
    print("SECTION 3: Training MLP Model (PyTorch)")
    print("=" * 70)

    # Normalize data
    mean = torch.tensor(X_train_scaled.mean(axis=0), dtype=torch.float32)
    std = torch.tensor(X_train_scaled.std(axis=0), dtype=torch.float32)

    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

    X_train_normalized = (X_train_tensor - mean) / std
    X_test_normalized = (X_test_tensor - mean) / std

    # Create DataLoaders
    train_dataset = TensorDataset(X_train_normalized, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # Initialize model
    input_size = X_train_scaled.shape[1]
    model = MLP(input_size)
    model.mean = mean
    model.std = std

    # Training setup
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Training loop
    print(f"\nTraining for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            avg_loss = epoch_loss / len(train_loader)
            print(f"Epoch [{epoch+1}/{args.epochs}], Loss: {avg_loss:.4f}")

    best_params = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "hidden_size": 64,
    }

    return model, best_params


def make_predictions(model, X_train_scaled, X_test_scaled, model_type):
    """Generate predictions on train and test sets."""
    print("\n" + "=" * 70)
    print("SECTION 4: Making Predictions")
    print("=" * 70)

    if model_type == "MLP":
        model.eval()
        with torch.no_grad():
            X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
            X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
            X_train_normalized = (X_train_tensor - model.mean) / model.std
            X_test_normalized = (X_test_tensor - model.mean) / model.std
            y_pred_train = model(X_train_normalized).numpy()
            y_pred = model(X_test_normalized).numpy()
    else:
        y_pred_train = model.predict(X_train_scaled)
        y_pred = model.predict(X_test_scaled)

    return y_pred_train, y_pred


def calculate_metrics(y_test, y_pred, y_train, y_pred_train):
    """Calculate core evaluation metrics."""
    print("\n" + "=" * 70)
    print("SECTION 5: Model Evaluation")
    print("=" * 70)

    # Extract predictions
    y_pred_home_score = y_pred[:, 0]
    y_pred_away_score = y_pred[:, 1]
    y_pred_home_margin = y_pred_home_score - y_pred_away_score

    # Win probability calculation
    def win_prob(score_diff):
        a = -0.2504
        b = 0.1949
        return 1 / (1 + np.exp(-(a + b * score_diff)))

    y_pred_home_win_prob = win_prob(y_pred_home_margin)

    # Core metrics
    home_score_mae = np.mean(np.abs(y_test["home_score"] - y_pred_home_score))
    away_score_mae = np.mean(np.abs(y_test["away_score"] - y_pred_away_score))
    home_margin_mae = np.mean(
        np.abs(y_test["home_score"] - y_test["away_score"] - y_pred_home_margin)
    )
    home_win_prob_log_loss = log_loss(
        (y_test["home_score"] > y_test["away_score"]).astype(int), y_pred_home_win_prob
    )

    metrics = {
        "home_score_mae": float(home_score_mae),
        "away_score_mae": float(away_score_mae),
        "home_margin_mae": float(home_margin_mae),
        "home_win_prob_log_loss": float(home_win_prob_log_loss),
    }

    print("\nCore Metrics:")
    print(f"Home Score MAE: {home_score_mae:.2f}")
    print(f"Away Score MAE: {away_score_mae:.2f}")
    print(f"Home Margin MAE: {home_margin_mae:.2f}")
    print(f"Home Win Probability Log Loss: {home_win_prob_log_loss:.4f}")

    return metrics, y_pred_home_win_prob, y_pred_home_margin


def save_model(
    model,
    model_type,
    output_dir,
    run_datetime,
    best_params,
    metrics,
    train_seasons,
    test_season,
    feature_names,
):
    """Save trained model and metadata."""
    print("\n" + "=" * 70)
    print("SECTION 6: Saving Model")
    print("=" * 70)

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Generate model filename
    model_name_map = {
        "Linear": "Ridge_Regression",
        "Tree": "XGBoost_Regression",
        "MLP": "MLP_Regression",
    }
    model_name = model_name_map[model_type]
    timestamp = run_datetime.replace(":", "-").replace(".", "-")

    if model_type == "MLP":
        model_file = output_path / f"{model_name}_{timestamp}.pth"
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "input_size": len(feature_names),
            "mean": model.mean,
            "std": model.std,
        }
        torch.save(checkpoint, model_file)
    else:
        model_file = output_path / f"{model_name}_{timestamp}.joblib"
        dump(model, model_file)

    # Save metadata
    metadata = {
        "model_type": model_type,
        "model_file": str(model_file),
        "timestamp": run_datetime,
        "train_seasons": train_seasons,
        "test_season": test_season,
        "hyperparameters": best_params,
        "metrics": metrics,
        "num_features": len(feature_names),
        "feature_names": feature_names,
    }

    metadata_file = output_path / f"{model_name}_{timestamp}_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✅ Model saved to: {model_file}")
    print(f"✅ Metadata saved to: {metadata_file}")

    return model_file, metadata_file


def main():
    """Main training pipeline."""
    args = parse_args()
    run_datetime = datetime.now().isoformat()

    print("\n" + "=" * 70)
    print(f"NBA Prediction Model Training - {args.model_type}")
    print("=" * 70)
    print(f"Run datetime: {run_datetime}")
    print(f"Random seed: {args.random_state}")

    # Parse season ranges
    train_seasons = parse_season_range(args.train_seasons)
    test_seasons = [args.test_season]

    # Load and prepare data
    X_train, y_train, X_test, y_test, feature_names, train_df, test_df = (
        load_and_prepare_data(train_seasons, test_seasons)
    )

    # Preprocess data
    X_train_scaled, X_test_scaled, scaler = preprocess_data(X_train, X_test)

    # Train model
    if args.model_type == "Linear":
        model, best_params = train_linear_model(
            X_train_scaled, y_train, args.random_state
        )
    elif args.model_type == "Tree":
        model, best_params = train_tree_model(
            X_train_scaled, y_train, args.random_state
        )
    elif args.model_type == "MLP":
        model, best_params = train_mlp_model(
            X_train_scaled, y_train, X_test_scaled, y_test, args
        )

    # Make predictions
    y_pred_train, y_pred = make_predictions(
        model, X_train_scaled, X_test_scaled, args.model_type
    )

    # Calculate metrics
    metrics, y_pred_home_win_prob, y_pred_home_margin = calculate_metrics(
        y_test, y_pred, y_train, y_pred_train
    )

    # Save model
    model_file, metadata_file = save_model(
        model,
        args.model_type,
        args.output_dir,
        run_datetime,
        best_params,
        metrics,
        train_seasons,
        test_seasons[0],
        feature_names,
    )

    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\nTo use this model, update config.yaml:")
    print(f"  predictors:")
    print(f"    {args.model_type}:")
    print(f"      model_paths:")
    print(f'        - "{model_file}"')


if __name__ == "__main__":
    main()
