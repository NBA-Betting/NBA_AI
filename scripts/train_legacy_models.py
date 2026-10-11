"""
train_legacy_models.py

Training script for legacy NBA prediction models (Ridge, XGBoost, MLP).
Trains on Features table (43 features) to predict home_score + away_score.

Improvements over v0.4 (src/model_training/train.py):
- Uses all available seasons for training (2000-2001 through 2024-2025)
- Cutoff date support for temporal splits
- Optuna hyperparameter tuning for XGBoost (50 trials)
- Larger MLP with BatchNorm + Huber loss + CosineAnnealingLR
- Missing values imputed with training mean (not dropped)
- Spread MAE and Win Accuracy in comparison table

Usage:
    python scripts/train_legacy_models.py                          # Train all three
    python scripts/train_legacy_models.py --models ridge xgboost   # Train specific
    python scripts/train_legacy_models.py --dry-run                # Just check data
    python scripts/train_legacy_models.py --cutoff-date 2026-03-31
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# Project root setup — must come before src imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import config
from src.database import get_db
from src.model_training.models import MLPv2

DB_PATH = config["database"]["path"]
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_VERSION = "0.5"

# 43 feature names in canonical order
FEATURE_NAMES = [
    "Home_Win_Pct",
    "Home_PPG",
    "Home_OPP_PPG",
    "Home_Net_PPG",
    "Away_Win_Pct",
    "Away_PPG",
    "Away_OPP_PPG",
    "Away_Net_PPG",
    "Win_Pct_Diff",
    "PPG_Diff",
    "OPP_PPG_Diff",
    "Net_PPG_Diff",
    "Home_Win_Pct_Home",
    "Home_PPG_Home",
    "Home_OPP_PPG_Home",
    "Home_Net_PPG_Home",
    "Away_Win_Pct_Away",
    "Away_PPG_Away",
    "Away_OPP_PPG_Away",
    "Away_Net_PPG_Away",
    "Win_Pct_Home_Away_Diff",
    "PPG_Home_Away_Diff",
    "OPP_PPG_Home_Away_Diff",
    "Net_PPG_Home_Away_Diff",
    "Time_Decay_Home_Win_Pct",
    "Time_Decay_Home_PPG",
    "Time_Decay_Home_OPP_PPG",
    "Time_Decay_Home_Net_PPG",
    "Time_Decay_Away_Win_Pct",
    "Time_Decay_Away_PPG",
    "Time_Decay_Away_OPP_PPG",
    "Time_Decay_Away_Net_PPG",
    "Time_Decay_Win_Pct_Diff",
    "Time_Decay_PPG_Diff",
    "Time_Decay_OPP_PPG_Diff",
    "Time_Decay_Net_PPG_Diff",
    "Day_of_Season",
    "Home_Rest_Days",
    "Home_Game_Freq",
    "Away_Rest_Days",
    "Away_Game_Freq",
    "Rest_Days_Diff",
    "Game_Freq_Diff",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_data(cutoff_date: str, val_season: str = "2025-2026"):
    """
    Load features + final scores from the database.

    Training: all seasons except val_season, games before cutoff_date.
    Validation: val_season games before cutoff_date.

    Returns (X_train, y_train, X_val, y_val, meta) where meta has counts.
    """
    log.info("Loading data from database...")
    t0 = time.time()

    with get_db(DB_PATH) as conn:
        query = """
            SELECT g.game_id, g.date_time_utc, g.season,
                   gs.home_score, gs.away_score,
                   f.feature_set
            FROM Features f
            INNER JOIN GameStates gs
                ON f.game_id = gs.game_id AND gs.is_final_state = 1
            INNER JOIN Games g
                ON f.game_id = g.game_id
            WHERE g.status = 3
              AND g.season_type IN ('Regular Season', 'Post Season')
              AND f.feature_set IS NOT NULL
              AND f.feature_set != '{}'
              AND f.feature_set != ''
              AND g.date_time_utc < ?
        """
        df = pd.read_sql_query(query, conn, params=[cutoff_date])

    log.info(f"  Loaded {len(df)} games in {time.time() - t0:.1f}s")

    # Parse JSON feature_set into columns
    log.info("  Parsing feature JSON...")
    features_parsed = df["feature_set"].apply(json.loads)
    features_df = pd.DataFrame(features_parsed.tolist())

    # Ensure canonical column order; add missing columns as NaN
    for col in FEATURE_NAMES:
        if col not in features_df.columns:
            features_df[col] = np.nan
    features_df = features_df[FEATURE_NAMES]

    # Targets
    y = df[["home_score", "away_score"]].values.astype(np.float64)

    # Split: training = all seasons except val_season, validation = val_season
    is_val = df["season"] == val_season
    X_all = features_df.values.astype(np.float64)

    X_train_raw, y_train = X_all[~is_val], y[~is_val]
    X_val_raw, y_val = X_all[is_val], y[is_val]

    # Impute missing with training mean, then standardize
    train_mean = np.nanmean(X_train_raw, axis=0)
    # Replace NaN in train_mean itself (feature entirely NaN) with 0
    train_mean = np.where(np.isnan(train_mean), 0.0, train_mean)

    X_train_imputed = np.where(np.isnan(X_train_raw), train_mean, X_train_raw)
    X_val_imputed = np.where(np.isnan(X_val_raw), train_mean, X_val_raw)

    train_std = X_train_imputed.std(axis=0)
    train_std = np.where(train_std == 0, 1.0, train_std)  # avoid division by zero

    X_train = (X_train_imputed - train_mean) / train_std
    X_val = (X_val_imputed - train_mean) / train_std

    # Collect season counts for metadata
    train_seasons = sorted(df.loc[~is_val, "season"].unique().tolist())
    val_seasons = sorted(df.loc[is_val, "season"].unique().tolist())

    meta = {
        "n_train": len(X_train),
        "n_val": len(X_val),
        "train_seasons": train_seasons,
        "val_seasons": val_seasons,
        "scaler_mean": train_mean,
        "scaler_std": train_std,
        "cutoff_date": cutoff_date,
    }

    log.info(f"  Train: {meta['n_train']} games ({len(train_seasons)} seasons)")
    log.info(f"  Val:   {meta['n_val']} games ({val_seasons})")

    return X_train, y_train, X_val, y_val, meta


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate(y_true, y_pred):
    """Compute evaluation metrics. Returns dict."""
    home_mae = np.mean(np.abs(y_true[:, 0] - y_pred[:, 0]))
    away_mae = np.mean(np.abs(y_true[:, 1] - y_pred[:, 1]))
    avg_mae = (home_mae + away_mae) / 2

    true_margin = y_true[:, 0] - y_true[:, 1]
    pred_margin = y_pred[:, 0] - y_pred[:, 1]
    spread_mae = np.mean(np.abs(true_margin - pred_margin))

    true_win = (true_margin > 0).astype(int)
    pred_win = (pred_margin > 0).astype(int)
    win_acc = np.mean(true_win == pred_win)

    return {
        "home_mae": round(float(home_mae), 3),
        "away_mae": round(float(away_mae), 3),
        "avg_score_mae": round(float(avg_mae), 3),
        "spread_mae": round(float(spread_mae), 3),
        "win_accuracy": round(float(win_acc), 4),
        "n_samples": len(y_true),
    }


# ---------------------------------------------------------------------------
# Ridge training
# ---------------------------------------------------------------------------


def train_ridge(X_train, y_train, X_val, y_val, meta):
    """Train Ridge regression with RidgeCV."""
    from sklearn.linear_model import RidgeCV

    log.info("=" * 60)
    log.info("TRAINING: Ridge Regression (RidgeCV)")
    log.info("=" * 60)

    alphas = [0.1, 1.0, 10.0, 50.0, 100.0, 300.0, 1000.0, 3000.0]
    log.info(f"  Alpha candidates: {alphas}")

    model = RidgeCV(alphas=alphas, store_cv_results=True)
    model.fit(X_train, y_train)

    best_alpha = model.alpha_
    log.info(f"  Best alpha: {best_alpha}")

    y_pred = model.predict(X_val)
    metrics = evaluate(y_val, y_pred)

    log.info(f"  Avg MAE: {metrics['avg_score_mae']:.3f}")
    log.info(f"  Spread MAE: {metrics['spread_mae']:.3f}")
    log.info(f"  Win Acc: {metrics['win_accuracy']:.4f}")

    hyperparams = {"alpha": float(best_alpha), "alphas_tested": alphas}
    return model, metrics, hyperparams


# ---------------------------------------------------------------------------
# XGBoost training with Optuna
# ---------------------------------------------------------------------------


def train_xgboost(X_train, y_train, X_val, y_val, meta, n_trials=50):
    """Train XGBoost with Optuna hyperparameter tuning."""
    import optuna
    from sklearn.multioutput import MultiOutputRegressor
    from xgboost import XGBRegressor

    log.info("=" * 60)
    log.info(f"TRAINING: XGBoost (Optuna, {n_trials} trials)")
    log.info("=" * 60)

    # Suppress Optuna info-level trial logs
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    best_score = float("inf")
    best_params = None
    best_model = None

    def objective(trial):
        nonlocal best_score, best_params, best_model

        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
            "subsample": trial.suggest_float("subsample", 0.6, 0.9),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.9),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 1.0),
            "objective": "reg:squarederror",
            "random_state": 42,
            "early_stopping_rounds": 20,
        }

        # Train separate models for home and away
        home_model = XGBRegressor(**params)
        away_model = XGBRegressor(**params)

        home_model.fit(
            X_train,
            y_train[:, 0],
            eval_set=[(X_val, y_val[:, 0])],
            verbose=False,
        )
        away_model.fit(
            X_train,
            y_train[:, 1],
            eval_set=[(X_val, y_val[:, 1])],
            verbose=False,
        )

        home_pred = home_model.predict(X_val)
        away_pred = away_model.predict(X_val)
        y_pred = np.column_stack([home_pred, away_pred])

        metrics = evaluate(y_val, y_pred)
        score = metrics["avg_score_mae"]

        if score < best_score:
            best_score = score
            best_params = params.copy()
            best_model = (home_model, away_model)

        return score

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    log.info(f"  Best trial: {study.best_trial.number}")
    log.info(f"  Best params: {study.best_params}")

    # Evaluate best model
    home_model, away_model = best_model
    home_pred = home_model.predict(X_val)
    away_pred = away_model.predict(X_val)
    y_pred = np.column_stack([home_pred, away_pred])
    metrics = evaluate(y_val, y_pred)

    log.info(f"  Avg MAE: {metrics['avg_score_mae']:.3f}")
    log.info(f"  Spread MAE: {metrics['spread_mae']:.3f}")
    log.info(f"  Win Acc: {metrics['win_accuracy']:.4f}")

    # Remove early_stopping_rounds from saved hyperparams (it is a training param)
    hyperparams = {k: v for k, v in best_params.items() if k != "early_stopping_rounds"}
    hyperparams["early_stopping_rounds"] = 20
    hyperparams["n_optuna_trials"] = n_trials

    return best_model, metrics, hyperparams


# ---------------------------------------------------------------------------
# MLP training
# ---------------------------------------------------------------------------


def train_mlp(X_train, y_train, X_val, y_val, meta):
    """Train enhanced MLP with Huber loss, CosineAnnealingLR, early stopping."""
    log.info("=" * 60)
    log.info("TRAINING: MLP (PyTorch)")
    log.info("=" * 60)

    hidden_sizes = [256, 128, 64]
    dropout = 0.2
    lr = 0.001
    epochs = 200
    patience = 25
    batch_size = 64

    log.info(f"  Architecture: {X_train.shape[1]} -> {hidden_sizes} -> 2")
    log.info(f"  Dropout: {dropout}, LR: {lr}, Batch: {batch_size}")
    log.info(f"  Loss: Huber, Scheduler: CosineAnnealingLR")
    log.info(f"  Epochs: {epochs}, Patience: {patience}")

    # Normalize targets
    y_mean = y_train.mean(axis=0)
    y_std = y_train.std(axis=0)
    y_train_norm = (y_train - y_mean) / y_std
    y_val_norm = (y_val - y_mean) / y_std

    # Tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train_norm, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val_norm, dtype=torch.float32)

    train_ds = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    torch.manual_seed(42)
    model = MLPv2(
        input_size=X_train.shape[1], hidden_sizes=hidden_sizes, dropout=dropout
    )

    criterion = nn.HuberLoss(delta=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_mae = float("inf")
    patience_counter = 0
    best_state = None
    best_epoch = 0

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        scheduler.step()

        # Validate (compute MAE in original scale)
        model.eval()
        with torch.no_grad():
            pred_norm = model(X_val_t).numpy()
            pred_orig = pred_norm * y_std + y_mean
            val_mae = np.mean(np.abs(y_val - pred_orig))

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1
        else:
            patience_counter += 1

        if (epoch + 1) % 20 == 0 or epoch == 0:
            log.info(
                f"  Epoch {epoch+1:3d}: train_loss={train_loss:.4f}, "
                f"val_mae={val_mae:.3f}, lr={scheduler.get_last_lr()[0]:.6f}"
            )

        if patience_counter >= patience:
            log.info(f"  Early stopping at epoch {epoch+1} (best epoch: {best_epoch})")
            break

    # Load best weights
    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        pred_norm = model(X_val_t).numpy()
        y_pred = pred_norm * y_std + y_mean

    metrics = evaluate(y_val, y_pred)

    log.info(f"  Best epoch: {best_epoch}")
    log.info(f"  Avg MAE: {metrics['avg_score_mae']:.3f}")
    log.info(f"  Spread MAE: {metrics['spread_mae']:.3f}")
    log.info(f"  Win Acc: {metrics['win_accuracy']:.4f}")

    # Store normalization params on model for saving
    model.scaler_mean = torch.tensor(meta["scaler_mean"], dtype=torch.float32)
    model.scaler_scale = torch.tensor(meta["scaler_std"], dtype=torch.float32)
    model.y_mean = torch.tensor(y_mean, dtype=torch.float32)
    model.y_std = torch.tensor(y_std, dtype=torch.float32)

    hyperparams = {
        "hidden_sizes": hidden_sizes,
        "dropout": dropout,
        "learning_rate": lr,
        "batch_size": batch_size,
        "epochs_trained": best_epoch,
        "max_epochs": epochs,
        "patience": patience,
        "loss": "HuberLoss(delta=1.0)",
        "scheduler": "CosineAnnealingLR",
    }

    return model, metrics, hyperparams


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------


# Each model is saved where its predictor loads it (config.yaml model_paths)
MODEL_DIRS = {"ridge": "linear", "xgboost": "tree", "mlp": "mlp"}


def save_model(model, model_name, metrics, hyperparams, meta):
    """Save model, scaler and metadata.json into models/<type>/.

    Overwrites the models the predictors load. Returns the model file paths.
    """
    model_dir = MODELS_DIR / MODEL_DIRS[model_name]
    model_dir.mkdir(parents=True, exist_ok=True)
    scaler_data = {
        "feature_names": FEATURE_NAMES,
        "mean": meta["scaler_mean"].tolist(),
        "std": meta["scaler_std"].tolist(),
    }

    if model_name == "ridge":
        from joblib import dump

        filepaths = [model_dir / "model.joblib"]
        dump(model, filepaths[0])
    elif model_name == "xgboost":
        # model is a (home_xgb, away_xgb) tuple, saved as native XGBoost JSON
        filepaths = [model_dir / "home_model.json", model_dir / "away_model.json"]
        for xgb_model, path in zip(model, filepaths):
            xgb_model.save_model(path)
    elif model_name == "mlp":
        filepaths = [model_dir / "model.pth"]
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
        torch.save(checkpoint, filepaths[0])
        scaler_data = None  # the MLP checkpoint carries its own scaler

    if scaler_data:
        with open(model_dir / "scaler.json", "w") as f:
            json.dump(scaler_data, f, indent=2)

    rel_paths = [str(p.relative_to(PROJECT_ROOT)) for p in filepaths]
    metadata = {
        "model_type": {"ridge": "Linear", "xgboost": "Tree", "mlp": "MLP"}[model_name],
        "version": MODEL_VERSION,
        **(
            {"model_files": rel_paths}
            if len(rel_paths) > 1
            else {"model_file": rel_paths[0]}
        ),
        "train_seasons": meta["train_seasons"],
        "val_season": meta["val_seasons"],
        "cutoff_date": meta["cutoff_date"],
        "metrics": metrics,
        "hyperparameters": {
            k: str(v) if isinstance(v, (list, dict)) else v
            for k, v in hyperparams.items()
        },
        "num_features": len(FEATURE_NAMES),
        "feature_names": FEATURE_NAMES,
        "created_at": datetime.now().isoformat(),
    }
    with open(model_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    log.info(f"  Saved: {', '.join(rel_paths)} (+ metadata.json)")
    return filepaths


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------


def print_comparison(results, meta):
    """Print comparison table of all trained models."""
    print("\n" + "=" * 90)
    print("  MODEL COMPARISON")
    print("=" * 90)
    header = (
        f"  {'Model':<12} {'Home MAE':>10} {'Away MAE':>10} {'Avg MAE':>10} "
        f"{'Spread MAE':>11} {'Win Acc':>9} {'N_train':>9} {'N_val':>7}"
    )
    print(header)
    print("  " + "-" * 84)
    for name, m in sorted(results.items(), key=lambda x: x[1]["avg_score_mae"]):
        print(
            f"  {name:<12} {m['home_mae']:>10.3f} {m['away_mae']:>10.3f} "
            f"{m['avg_score_mae']:>10.3f} {m['spread_mae']:>11.3f} "
            f"{m['win_accuracy']*100:>8.1f}% {meta['n_train']:>9d} {meta['n_val']:>7d}"
        )
    print("=" * 90)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train legacy NBA prediction models (Ridge, XGBoost, MLP)"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["ridge", "xgboost", "mlp"],
        choices=["ridge", "xgboost", "mlp"],
        help="Which models to train (default: all three)",
    )
    parser.add_argument(
        "--cutoff-date",
        default="2026-03-31",
        help="No games on or after this date in training/val (default: 2026-03-31)",
    )
    parser.add_argument(
        "--val-season",
        default="2025-2026",
        help="Season to hold out for validation (default: 2025-2026)",
    )
    parser.add_argument(
        "--xgb-trials",
        type=int,
        default=50,
        help="Number of Optuna trials for XGBoost (default: 50)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load data and report sizes without training",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("\n" + "=" * 60)
    print("  NBA LEGACY MODEL TRAINING (v0.5)")
    print("=" * 60)
    print(f"  Models:      {args.models}")
    print(f"  Cutoff date: {args.cutoff_date}")
    print(f"  Val season:  {args.val_season}")
    if "xgboost" in args.models:
        print(f"  XGB trials:  {args.xgb_trials}")
    print()

    # Load data
    X_train, y_train, X_val, y_val, meta = load_data(args.cutoff_date, args.val_season)

    if args.dry_run:
        print("\n[DRY RUN] Data loaded successfully. No training performed.")
        print(f"  Training:   {meta['n_train']} games")
        print(f"  Validation: {meta['n_val']} games")
        print(f"  Features:   {len(FEATURE_NAMES)}")
        print(f"  Train seasons: {meta['train_seasons']}")
        print(f"  Val seasons:   {meta['val_seasons']}")

        # Show feature stats
        print(f"\n  Feature stats (training set, standardized):")
        print(
            f"    Mean range: [{X_train.mean(axis=0).min():.4f}, {X_train.mean(axis=0).max():.4f}]"
        )
        print(
            f"    Std range:  [{X_train.std(axis=0).min():.4f}, {X_train.std(axis=0).max():.4f}]"
        )

        print(f"\n  Target stats (training set):")
        print(
            f"    Home score: mean={y_train[:, 0].mean():.1f}, std={y_train[:, 0].std():.1f}"
        )
        print(
            f"    Away score: mean={y_train[:, 1].mean():.1f}, std={y_train[:, 1].std():.1f}"
        )
        return

    # Train models
    results = {}

    for model_name in args.models:
        if model_name == "ridge":
            model, metrics, hyperparams = train_ridge(
                X_train, y_train, X_val, y_val, meta
            )
        elif model_name == "xgboost":
            model, metrics, hyperparams = train_xgboost(
                X_train,
                y_train,
                X_val,
                y_val,
                meta,
                n_trials=args.xgb_trials,
            )
        elif model_name == "mlp":
            model, metrics, hyperparams = train_mlp(
                X_train, y_train, X_val, y_val, meta
            )

        results[model_name] = metrics

        save_model(model, model_name, metrics, hyperparams, meta)

    # Comparison table
    if results:
        print_comparison(results, meta)

    print("\nTraining complete.")


if __name__ == "__main__":
    main()
