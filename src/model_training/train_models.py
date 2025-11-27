"""
train_models.py

A unified model training script for NBA game prediction models.
This script implements best practices for model training with a 1:1 train-test split strategy
(one season for training, one season for testing).

Features:
- Consistent 1:1 train-test split strategy across all model types
- Minimal hyperparameter searching (focused on essential parameters only)
- Unified evaluation framework
- Support for Ridge Regression, XGBoost, and MLP models
- Optional logging to Weights & Biases

Usage:
    python -m src.model_training.train_models --model_type=Ridge --train_season=2023-2024 --test_season=2024-2025
    python -m src.model_training.train_models --model_type=XGBoost --train_season=2023-2024 --test_season=2024-2025
    python -m src.model_training.train_models --model_type=MLP --train_season=2023-2024 --test_season=2024-2025
    python -m src.model_training.train_models --model_type=all --train_season=2023-2024 --test_season=2024-2025

Arguments:
    --model_type: Type of model to train (Ridge, XGBoost, MLP, or all)
    --train_season: Season to use for training (e.g., 2023-2024)
    --test_season: Season to use for testing (e.g., 2024-2025)
    --log_level: Logging level (default: INFO)
    --log_to_wandb: Enable logging to Weights & Biases (default: False)
"""

import argparse
from datetime import datetime
import logging

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from joblib import dump
from sklearn.linear_model import Ridge
from sklearn.metrics import log_loss
from sklearn.model_selection import cross_val_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from src.config import config
from src.logging_config import setup_logging
from src.model_training.evaluation import create_evaluations
from src.model_training.mlp_model import MLP
from src.model_training.modeling_utils import load_featurized_modeling_data

# Configuration
DB_PATH = config["database"]["path"]
PROJECT_ROOT = config["project"]["root"]

# Define game info and result columns to exclude from features
GAME_INFO_COLUMNS = [
    "game_id",
    "date_time_est",
    "home_team",
    "away_team",
    "season",
    "season_type",
]
GAME_RESULTS_COLUMNS = [
    "home_score",
    "away_score",
    "total",
    "home_margin",
    "players_data",
]


def win_prob(score_diff):
    """
    Calculate win probabilities using the logistic (sigmoid) function.
    
    Parameters:
    score_diff: The difference between home and away scores.
    
    Returns:
    float: The win probability for the home team.
    """
    a = -0.2504  # Intercept term
    b = 0.1949  # Slope term
    return 1 / (1 + np.exp(-(a + b * score_diff)))


def prepare_data(train_season, test_season, db_path=DB_PATH):
    """
    Load and prepare data for model training using 1:1 train-test split.
    
    Parameters:
    train_season: Season to use for training (e.g., '2023-2024')
    test_season: Season to use for testing (e.g., '2024-2025')
    db_path: Path to the database
    
    Returns:
    tuple: (X_train, y_train, X_test, y_test, feature_names)
    """
    logging.info(f"Loading data for training season: {train_season}, testing season: {test_season}")
    
    # Load featurized modeling data
    training_df = load_featurized_modeling_data([train_season], db_path)
    testing_df = load_featurized_modeling_data([test_season], db_path)
    
    logging.info(f"Training data shape: {training_df.shape}")
    logging.info(f"Testing data shape: {testing_df.shape}")
    
    # Drop rows with NaN values
    training_df = training_df.dropna()
    testing_df = testing_df.dropna()
    
    logging.info(f"Training data shape after dropping NaNs: {training_df.shape}")
    logging.info(f"Testing data shape after dropping NaNs: {testing_df.shape}")
    
    # Prepare features and targets
    X_train = training_df.drop(columns=GAME_INFO_COLUMNS + GAME_RESULTS_COLUMNS)
    y_train = training_df[["home_score", "away_score"]]
    X_test = testing_df.drop(columns=GAME_INFO_COLUMNS + GAME_RESULTS_COLUMNS)
    y_test = testing_df[["home_score", "away_score"]]
    
    feature_names = X_train.columns.tolist()
    
    logging.info(f"Feature count: {len(feature_names)}")
    logging.info(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    logging.info(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    
    return X_train, y_train, X_test, y_test, feature_names


def evaluate_model(y_train, y_pred_train, y_test, y_pred_test):
    """
    Evaluate model performance and return metrics.
    
    Parameters:
    y_train: Actual training targets
    y_pred_train: Predicted training targets
    y_test: Actual test targets
    y_pred_test: Predicted test targets
    
    Returns:
    tuple: (core_metrics, train_evaluations, test_evaluations)
    """
    # Extract score components
    y_pred_home_score = y_pred_test[:, 0]
    y_pred_away_score = y_pred_test[:, 1]
    y_pred_home_margin = y_pred_home_score - y_pred_away_score
    y_pred_home_win_prob = win_prob(y_pred_home_margin)
    
    y_pred_train_home_score = y_pred_train[:, 0]
    y_pred_train_away_score = y_pred_train[:, 1]
    y_pred_train_home_margin = y_pred_train_home_score - y_pred_train_away_score
    y_pred_train_home_win_prob = win_prob(y_pred_train_home_margin)
    
    # Core metrics
    home_score_mae = np.mean(np.abs(y_test["home_score"].values - y_pred_home_score))
    away_score_mae = np.mean(np.abs(y_test["away_score"].values - y_pred_away_score))
    home_margin_mae = np.mean(
        np.abs(y_test["home_score"].values - y_test["away_score"].values - y_pred_home_margin)
    )
    home_win_prob_log_loss = log_loss(
        (y_test["home_score"].values > y_test["away_score"].values).astype(int),
        y_pred_home_win_prob
    )
    
    core_metrics = {
        "home_score_mae": home_score_mae,
        "away_score_mae": away_score_mae,
        "home_margin_mae": home_margin_mae,
        "home_win_prob_log_loss": home_win_prob_log_loss,
    }
    
    logging.info(f"\nCore Metrics:")
    logging.info(f"Home Score MAE: {home_score_mae:.2f}")
    logging.info(f"Away Score MAE: {away_score_mae:.2f}")
    logging.info(f"Home Margin MAE: {home_margin_mae:.2f}")
    logging.info(f"Home Win Probability Log Loss: {home_win_prob_log_loss:.4f}")
    
    # Full evaluation suite
    train_correct = {
        "home_score": y_train["home_score"],
        "away_score": y_train["away_score"],
        "home_margin_derived": y_train["home_score"] - y_train["away_score"],
        "total_points_derived": y_train["home_score"] + y_train["away_score"],
        "home_win_prob": (y_train["home_score"] > y_train["away_score"]).astype(int),
    }
    train_pred = {
        "home_score": y_pred_train_home_score,
        "away_score": y_pred_train_away_score,
        "home_margin_derived": y_pred_train_home_margin,
        "total_points_derived": y_pred_train_home_score + y_pred_train_away_score,
        "home_win_prob": y_pred_train_home_win_prob,
    }
    
    test_correct = {
        "home_score": y_test["home_score"],
        "away_score": y_test["away_score"],
        "home_margin_derived": y_test["home_score"] - y_test["away_score"],
        "total_points_derived": y_test["home_score"] + y_test["away_score"],
        "home_win_prob": (y_test["home_score"] > y_test["away_score"]).astype(int),
    }
    test_pred = {
        "home_score": y_pred_home_score,
        "away_score": y_pred_away_score,
        "home_margin_derived": y_pred_home_margin,
        "total_points_derived": y_pred_home_score + y_pred_away_score,
        "home_win_prob": y_pred_home_win_prob,
    }
    
    train_evaluations = pd.DataFrame(
        [{"train_" + k: v for k, v in create_evaluations(train_correct, train_pred).items()}]
    )
    test_evaluations = pd.DataFrame(
        [{"test_" + k: v for k, v in create_evaluations(test_correct, test_pred).items()}]
    )
    
    return core_metrics, train_evaluations, test_evaluations


def train_ridge_model(X_train, y_train, X_test, y_test, feature_names, log_to_wandb=False):
    """
    Train a Ridge Regression model with minimal hyperparameter tuning.
    
    Uses cross-validation to select the best alpha from a small set of values.
    """
    logging.info("Training Ridge Regression model...")
    run_datetime = datetime.now().isoformat()
    model_type = "Ridge_Regression"
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Minimal hyperparameter search: just alpha
    alphas = [0.1, 1.0, 10.0, 100.0]
    best_alpha = 1.0
    best_score = float('-inf')
    
    for alpha in alphas:
        model = Ridge(alpha=alpha)
        scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='neg_mean_absolute_error')
        mean_score = scores.mean()
        if mean_score > best_score:
            best_score = mean_score
            best_alpha = alpha
    
    logging.info(f"Best alpha: {best_alpha}")
    
    # Train final model with best alpha
    model = Ridge(alpha=best_alpha)
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)
    
    # Evaluate
    core_metrics, train_evals, test_evals = evaluate_model(y_train, y_pred_train, y_test, y_pred_test)
    
    # Retrain on all data
    X_all = np.concatenate((X_train, X_test))
    y_all = np.concatenate((y_train, y_test))
    
    final_scaler = StandardScaler()
    X_all_scaled = final_scaler.fit_transform(X_all)
    
    final_model = Ridge(alpha=best_alpha)
    final_model.fit(X_all_scaled, y_all)
    
    pipeline = Pipeline([("scaler", final_scaler), ("model", final_model)])
    
    # Save model
    model_filename = f"{PROJECT_ROOT}/models/{model_type}_{run_datetime}.joblib"
    dump(pipeline, model_filename)
    logging.info(f"Model saved to {model_filename}")
    
    # Log to wandb if enabled
    if log_to_wandb:
        _log_to_wandb(model_type, run_datetime, {"alpha": best_alpha}, 
                     core_metrics, train_evals, test_evals, feature_names,
                     X_train_scaled.shape, X_test_scaled.shape, model_filename)
    
    return model_filename, core_metrics


def train_xgboost_model(X_train, y_train, X_test, y_test, feature_names, log_to_wandb=False):
    """
    Train an XGBoost model with minimal hyperparameter tuning.
    
    Uses reasonable defaults with minimal search over key parameters.
    """
    logging.info("Training XGBoost model...")
    run_datetime = datetime.now().isoformat()
    model_type = "XGBoost_Regression"
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Minimal hyperparameters - reasonable defaults
    params = {
        "learning_rate": 0.1,
        "max_depth": 5,
        "n_estimators": 200,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "objective": "reg:squarederror",
    }
    
    logging.info(f"Using parameters: {params}")
    
    # Train model
    model = MultiOutputRegressor(XGBRegressor(**params))
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)
    
    # Evaluate
    core_metrics, train_evals, test_evals = evaluate_model(y_train, y_pred_train, y_test, y_pred_test)
    
    # Retrain on all data
    X_all = np.concatenate((X_train, X_test))
    y_all = np.concatenate((y_train, y_test))
    
    final_scaler = StandardScaler()
    X_all_scaled = final_scaler.fit_transform(X_all)
    
    final_model = MultiOutputRegressor(XGBRegressor(**params))
    final_model.fit(X_all_scaled, y_all)
    
    pipeline = Pipeline([("scaler", final_scaler), ("model", final_model)])
    
    # Save model
    model_filename = f"{PROJECT_ROOT}/models/{model_type}_{run_datetime}.joblib"
    dump(pipeline, model_filename)
    logging.info(f"Model saved to {model_filename}")
    
    # Log to wandb if enabled
    if log_to_wandb:
        _log_to_wandb(model_type, run_datetime, params,
                     core_metrics, train_evals, test_evals, feature_names,
                     X_train_scaled.shape, X_test_scaled.shape, model_filename)
    
    return model_filename, core_metrics


def train_mlp_model(X_train, y_train, X_test, y_test, feature_names, log_to_wandb=False):
    """
    Train an MLP model with reasonable defaults.
    
    Uses a simple architecture with minimal hyperparameter tuning.
    """
    logging.info("Training MLP model...")
    run_datetime = datetime.now().isoformat()
    model_type = "MLP_Regression"
    
    # Hyperparameters - reasonable defaults
    learning_rate = 0.001
    batch_size = 64
    epochs = 100
    
    # Convert to tensors
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)
    
    # Standardize
    mean = X_train_tensor.mean(dim=0)
    std = X_train_tensor.std(dim=0)
    std[std == 0] = 1  # Avoid division by zero
    X_train_scaled = (X_train_tensor - mean) / std
    X_test_scaled = (X_test_tensor - mean) / std
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_scaled, y_train_tensor)
    test_dataset = TensorDataset(X_test_scaled, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    input_size = X_train.shape[1]
    model = MLP(input_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        if (epoch + 1) % 20 == 0:
            logging.info(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")
    
    # Predictions
    model.eval()
    with torch.no_grad():
        y_pred_train = model(X_train_scaled).numpy()
        y_pred_test = model(X_test_scaled).numpy()
    
    # Evaluate
    core_metrics, train_evals, test_evals = evaluate_model(y_train, y_pred_train, y_test, y_pred_test)
    
    # Retrain on all data
    X_full = pd.concat([X_train, X_test], axis=0)
    y_full = pd.concat([y_train, y_test], axis=0)
    
    X_full_tensor = torch.tensor(X_full.values, dtype=torch.float32)
    y_full_tensor = torch.tensor(y_full.values, dtype=torch.float32)
    
    final_mean = X_full_tensor.mean(dim=0)
    final_std = X_full_tensor.std(dim=0)
    final_std[final_std == 0] = 1
    X_full_scaled = (X_full_tensor - final_mean) / final_std
    
    full_dataset = TensorDataset(X_full_scaled, y_full_tensor)
    full_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True)
    
    final_model = MLP(input_size)
    final_optimizer = torch.optim.Adam(final_model.parameters(), lr=learning_rate)
    
    for epoch in range(epochs):
        final_model.train()
        for inputs, targets in full_loader:
            final_optimizer.zero_grad()
            outputs = final_model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            final_optimizer.step()
    
    # Save model
    model_filename = f"{PROJECT_ROOT}/models/{model_type}_{run_datetime}.pth"
    torch.save(
        {
            "model_state_dict": final_model.state_dict(),
            "mean": final_mean,
            "std": final_std,
            "input_size": input_size,
        },
        model_filename,
    )
    logging.info(f"Model saved to {model_filename}")
    
    # Log to wandb if enabled
    if log_to_wandb:
        params = {"learning_rate": learning_rate, "batch_size": batch_size, "epochs": epochs}
        _log_to_wandb(model_type, run_datetime, params,
                     core_metrics, train_evals, test_evals, feature_names,
                     X_train.shape, X_test.shape, model_filename)
    
    return model_filename, core_metrics


def _log_to_wandb(model_type, run_datetime, params, core_metrics, 
                  train_evals, test_evals, feature_names, train_shape, test_shape, model_filename):
    """Helper function to log training results to Weights & Biases."""
    try:
        import wandb
        
        run = wandb.init(project="NBA AI", config=params)
        
        wandb.config.update({
            "model_type": model_type,
            "train_shape": train_shape,
            "test_shape": test_shape,
            "targets": ["home_score", "away_score"],
            "features": feature_names,
            "run_datetime": run_datetime,
        })
        
        wandb.summary.update(core_metrics)
        
        train_evaluations_table = wandb.Table(dataframe=train_evals)
        test_evaluations_table = wandb.Table(dataframe=test_evals)
        wandb.summary.update({"Train Evals": train_evaluations_table})
        wandb.summary.update({"Test Evals": test_evaluations_table})
        
        wandb.save(model_filename, base_path=PROJECT_ROOT)
        
        run.finish()
    except Exception as e:
        logging.warning(f"Failed to log to wandb: {e}")


def main():
    """Main function to handle command-line arguments and train models."""
    parser = argparse.ArgumentParser(
        description="Train NBA game prediction models with 1:1 train-test split."
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="all",
        choices=["Ridge", "XGBoost", "MLP", "all"],
        help="Type of model to train (Ridge, XGBoost, MLP, or all).",
    )
    parser.add_argument(
        "--train_season",
        type=str,
        default="2023-2024",
        help="Season to use for training (e.g., 2023-2024).",
    )
    parser.add_argument(
        "--test_season",
        type=str,
        default="2024-2025",
        help="Season to use for testing (e.g., 2024-2025).",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        help="Logging level (e.g., INFO, DEBUG).",
    )
    parser.add_argument(
        "--log_to_wandb",
        action="store_true",
        help="Enable logging to Weights & Biases.",
    )
    
    args = parser.parse_args()
    setup_logging(log_level=args.log_level.upper())
    
    logging.info(f"Starting model training...")
    logging.info(f"Training season: {args.train_season}")
    logging.info(f"Testing season: {args.test_season}")
    logging.info(f"Model type: {args.model_type}")
    
    # Prepare data
    X_train, y_train, X_test, y_test, feature_names = prepare_data(
        args.train_season, args.test_season
    )
    
    if len(X_train) == 0 or len(X_test) == 0:
        logging.error("No data available for training or testing. Please ensure the database is populated.")
        return
    
    # Train models
    results = {}
    
    if args.model_type in ["Ridge", "all"]:
        model_path, metrics = train_ridge_model(
            X_train, y_train, X_test, y_test, feature_names, args.log_to_wandb
        )
        results["Ridge"] = {"model_path": model_path, "metrics": metrics}
    
    if args.model_type in ["XGBoost", "all"]:
        model_path, metrics = train_xgboost_model(
            X_train, y_train, X_test, y_test, feature_names, args.log_to_wandb
        )
        results["XGBoost"] = {"model_path": model_path, "metrics": metrics}
    
    if args.model_type in ["MLP", "all"]:
        model_path, metrics = train_mlp_model(
            X_train, y_train, X_test, y_test, feature_names, args.log_to_wandb
        )
        results["MLP"] = {"model_path": model_path, "metrics": metrics}
    
    # Summary
    logging.info("\n" + "=" * 50)
    logging.info("Training Complete - Summary")
    logging.info("=" * 50)
    for model_name, result in results.items():
        logging.info(f"\n{model_name}:")
        logging.info(f"  Model saved to: {result['model_path']}")
        logging.info(f"  Home Margin MAE: {result['metrics']['home_margin_mae']:.2f}")
        logging.info(f"  Win Prob Log Loss: {result['metrics']['home_win_prob_log_loss']:.4f}")


if __name__ == "__main__":
    main()
