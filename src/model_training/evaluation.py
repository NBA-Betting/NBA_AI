"""
evaluation.py

Streamlined evaluation framework for NBA score prediction models.
Focuses on core metrics for production model comparison.

Core Metrics:
- Score prediction: MAE, RMSE for home/away/total scores
- Win probability: Accuracy, Brier score, Log loss
- Margin prediction: MAE for home_margin (spread equivalent)

Usage:
    from src.model_training.evaluation import evaluate_predictions, compare_models

    # Evaluate single model
    metrics = evaluate_predictions(y_true, y_pred)

    # Compare multiple models
    comparison_df = compare_models(model_results)
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
)


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Evaluate score predictions with core metrics.

    Args:
        y_true: Array of shape (n_samples, 2) with [home_score, away_score]
        y_pred: Array of shape (n_samples, 2) with [pred_home, pred_away]

    Returns:
        Dictionary with evaluation metrics
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Score predictions
    home_mae = mean_absolute_error(y_true[:, 0], y_pred[:, 0])
    away_mae = mean_absolute_error(y_true[:, 1], y_pred[:, 1])
    home_rmse = np.sqrt(mean_squared_error(y_true[:, 0], y_pred[:, 0]))
    away_rmse = np.sqrt(mean_squared_error(y_true[:, 1], y_pred[:, 1]))

    # Derived metrics
    true_margin = y_true[:, 0] - y_true[:, 1]
    pred_margin = y_pred[:, 0] - y_pred[:, 1]
    margin_mae = mean_absolute_error(true_margin, pred_margin)

    true_total = y_true[:, 0] + y_true[:, 1]
    pred_total = y_pred[:, 0] + y_pred[:, 1]
    total_mae = mean_absolute_error(true_total, pred_total)

    # Win probability (home win = margin > 0)
    true_home_win = (true_margin > 0).astype(int)
    pred_home_win = (pred_margin > 0).astype(int)
    win_accuracy = accuracy_score(true_home_win, pred_home_win)

    # Probability calibration (convert margin to probability)
    # Using logistic function: P(home_win) = 1 / (1 + exp(-k * margin))
    # k=0.15 approximates NBA historical data
    pred_win_prob = 1 / (1 + np.exp(-0.15 * pred_margin))
    pred_win_prob = np.clip(pred_win_prob, 0.001, 0.999)  # Avoid log(0)

    brier = brier_score_loss(true_home_win, pred_win_prob)
    logloss = log_loss(true_home_win, pred_win_prob)

    return {
        "home_mae": round(home_mae, 3),
        "away_mae": round(away_mae, 3),
        "avg_score_mae": round((home_mae + away_mae) / 2, 3),
        "home_rmse": round(home_rmse, 3),
        "away_rmse": round(away_rmse, 3),
        "margin_mae": round(margin_mae, 3),
        "total_mae": round(total_mae, 3),
        "win_accuracy": round(win_accuracy, 4),
        "brier_score": round(brier, 4),
        "log_loss": round(logloss, 4),
        "n_samples": len(y_true),
    }


def compare_models(model_results: dict) -> pd.DataFrame:
    """
    Compare multiple models and rank by performance.

    Args:
        model_results: Dict of {model_name: metrics_dict}

    Returns:
        DataFrame with models ranked by avg_score_mae
    """
    rows = []
    for model_name, metrics in model_results.items():
        row = {"model": model_name, **metrics}
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values("avg_score_mae", ascending=True)
    df["rank"] = range(1, len(df) + 1)

    # Reorder columns
    cols = [
        "rank",
        "model",
        "avg_score_mae",
        "margin_mae",
        "win_accuracy",
        "brier_score",
        "home_mae",
        "away_mae",
        "total_mae",
        "n_samples",
    ]
    cols = [c for c in cols if c in df.columns]

    return df[cols].reset_index(drop=True)


def print_evaluation_report(metrics: dict, model_name: str = "Model") -> None:
    """Print formatted evaluation report."""
    print(f"\n{'='*60}")
    print(f"  {model_name} Evaluation Report")
    print(f"{'='*60}")
    print(f"  Samples evaluated: {metrics['n_samples']}")
    print(f"\n  Score Prediction:")
    print(f"    Home Score MAE:  {metrics['home_mae']:.2f} pts")
    print(f"    Away Score MAE:  {metrics['away_mae']:.2f} pts")
    print(f"    Avg Score MAE:   {metrics['avg_score_mae']:.2f} pts")
    print(f"\n  Game Outcome:")
    print(f"    Margin MAE:      {metrics['margin_mae']:.2f} pts")
    print(f"    Total MAE:       {metrics['total_mae']:.2f} pts")
    print(f"    Win Accuracy:    {metrics['win_accuracy']*100:.1f}%")
    print(f"\n  Probability Calibration:")
    print(f"    Brier Score:     {metrics['brier_score']:.4f}")
    print(f"    Log Loss:        {metrics['log_loss']:.4f}")
    print(f"{'='*60}\n")


def print_model_comparison(comparison_df: pd.DataFrame) -> None:
    """Print formatted model comparison table."""
    print(f"\n{'='*80}")
    print("  Model Comparison (ranked by Avg Score MAE)")
    print(f"{'='*80}")
    print(
        f"  {'Rank':<6}{'Model':<12}{'Avg MAE':<10}{'Margin MAE':<12}{'Win Acc':<10}{'Brier':<10}"
    )
    print(f"  {'-'*60}")
    for _, row in comparison_df.iterrows():
        print(
            f"  {row['rank']:<6}{row['model']:<12}{row['avg_score_mae']:<10.2f}"
            f"{row['margin_mae']:<12.2f}{row['win_accuracy']*100:<10.1f}%{row['brier_score']:<10.4f}"
        )
    print(f"{'='*80}\n")
