#!/usr/bin/env python3
"""
Phase B Training: L3 (Team Model) + L4 (Game Predictor) on frozen L1+L2 outputs.

Trains L3 and L4 jointly end-to-end against game-level spread/win targets.
L1 and L2 are frozen — their outputs are pre-computed in the Phase B cache.

Usage:
    python scripts/train_phase5_b.py                          # Full training
    python scripts/train_phase5_b.py --test --epochs 3        # Quick test
    python scripts/train_phase5_b.py --batch-size 64 --lr 3e-4  # Override hyperparams
"""

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.phase5.l3_config import L3Config
from src.phase5.l3_model import TeamModel, count_parameters as l3_count_params
from src.phase5.l4_config import L4Config
from src.phase5.l4_model import GamePredictor, count_parameters as l4_count_params
from src.phase5.l4_losses import game_prediction_loss, prediction_metrics
from src.phase5.game_dataset import (
    TRAIN_SEASONS,
    VAL_SEASON,
    load_phase_b_data,
    make_train_val_datasets,
)

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "phase5"
RESULTS_DIR = PROJECT_ROOT / "results" / "phase5"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train_epoch(
    l3_model: TeamModel,
    l4_model: GamePredictor,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None,
    l4_cfg: L4Config,
    device: torch.device,
    grad_clip: float = 1.0,
) -> dict[str, float]:
    """Train one epoch. Returns dict of mean loss components."""
    l3_model.train()
    l4_model.train()

    total_losses = {}
    n_batches = 0

    for batch in loader:
        # Move to device
        home_l2 = batch["home_l2"].to(device)
        away_l2 = batch["away_l2"].to(device)
        home_tf = batch["home_team_features"].to(device)
        away_tf = batch["away_team_features"].to(device)
        home_rs = batch["home_roster_summary"].to(device)
        away_rs = batch["away_roster_summary"].to(device)
        home_coach = batch["home_coach_idx"].to(device)
        away_coach = batch["away_coach_idx"].to(device)
        home_coach_g = batch["home_coach_games"].to(device)
        away_coach_g = batch["away_coach_games"].to(device)
        home_cont = batch["home_continuity"].to(device)
        away_cont = batch["away_continuity"].to(device)
        game_ctx = batch["game_context"].to(device)

        actual_margin = batch["actual_margin"].to(device)
        actual_win = batch["actual_home_win"].to(device)

        # L3 forward: home and away teams
        home_repr = l3_model(
            l2_team=home_l2,
            team_features=home_tf,
            roster_summary=home_rs,
            coach_idx=home_coach,
            roster_continuity=home_cont,
            coach_games=home_coach_g,
        )  # (B, 128)

        away_repr = l3_model(
            l2_team=away_l2,
            team_features=away_tf,
            roster_summary=away_rs,
            coach_idx=away_coach,
            roster_continuity=away_cont,
            coach_games=away_coach_g,
        )  # (B, 128)

        # L4 forward
        predictions = l4_model(
            team_home=home_repr,
            team_away=away_repr,
            l2_home=home_l2,
            l2_away=away_l2,
            context=game_ctx,
        )

        # Loss
        losses = game_prediction_loss(
            predictions=predictions,
            actual_margin=actual_margin,
            actual_home_win=actual_win,
            cfg=l4_cfg,
        )

        # Backward
        optimizer.zero_grad()
        losses["total_loss"].backward()
        torch.nn.utils.clip_grad_norm_(
            list(l3_model.parameters()) + list(l4_model.parameters()),
            grad_clip,
        )
        optimizer.step()

        # Accumulate
        for k, v in losses.items():
            if k not in total_losses:
                total_losses[k] = 0.0
            total_losses[k] += v.item()
        n_batches += 1

    if scheduler is not None:
        scheduler.step()

    # Average
    return {k: v / max(n_batches, 1) for k, v in total_losses.items()}


@torch.no_grad()
def evaluate(
    l3_model: TeamModel,
    l4_model: GamePredictor,
    loader: DataLoader,
    l4_cfg: L4Config,
    device: torch.device,
) -> tuple[dict[str, float], dict[str, float]]:
    """Evaluate on a dataset. Returns (loss_dict, metrics_dict)."""
    l3_model.eval()
    l4_model.eval()

    total_losses = {}
    n_batches = 0

    all_preds = {"spread_mu": [], "spread_sigma": [], "win_prob": []}
    all_targets = {"margin": [], "win": []}

    for batch in loader:
        home_l2 = batch["home_l2"].to(device)
        away_l2 = batch["away_l2"].to(device)
        home_tf = batch["home_team_features"].to(device)
        away_tf = batch["away_team_features"].to(device)
        home_rs = batch["home_roster_summary"].to(device)
        away_rs = batch["away_roster_summary"].to(device)
        home_coach = batch["home_coach_idx"].to(device)
        away_coach = batch["away_coach_idx"].to(device)
        home_coach_g = batch["home_coach_games"].to(device)
        away_coach_g = batch["away_coach_games"].to(device)
        home_cont = batch["home_continuity"].to(device)
        away_cont = batch["away_continuity"].to(device)
        game_ctx = batch["game_context"].to(device)

        actual_margin = batch["actual_margin"].to(device)
        actual_win = batch["actual_home_win"].to(device)

        # L3
        home_repr = l3_model(
            l2_team=home_l2,
            team_features=home_tf,
            roster_summary=home_rs,
            coach_idx=home_coach,
            roster_continuity=home_cont,
            coach_games=home_coach_g,
        )
        away_repr = l3_model(
            l2_team=away_l2,
            team_features=away_tf,
            roster_summary=away_rs,
            coach_idx=away_coach,
            roster_continuity=away_cont,
            coach_games=away_coach_g,
        )

        # L4
        predictions = l4_model(
            team_home=home_repr,
            team_away=away_repr,
            l2_home=home_l2,
            l2_away=away_l2,
            context=game_ctx,
        )

        # Loss
        losses = game_prediction_loss(
            predictions=predictions,
            actual_margin=actual_margin,
            actual_home_win=actual_win,
            cfg=l4_cfg,
        )

        for k, v in losses.items():
            if k not in total_losses:
                total_losses[k] = 0.0
            total_losses[k] += v.item()
        n_batches += 1

        # Accumulate predictions for metrics
        all_preds["spread_mu"].append(predictions["spread_mu"].cpu().numpy())
        all_preds["spread_sigma"].append(predictions["spread_sigma"].cpu().numpy())
        all_preds["win_prob"].append(predictions["win_prob"].cpu().numpy())

        all_targets["margin"].append(actual_margin.cpu().numpy())
        all_targets["win"].append(actual_win.cpu().numpy())

    avg_losses = {k: v / max(n_batches, 1) for k, v in total_losses.items()}
    metrics = prediction_metrics(all_preds, all_targets)

    return avg_losses, metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Phase B: Train L3+L4")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--l3-lr", type=float, default=5e-4)
    parser.add_argument("--l4-lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument(
        "--test",
        action="store_true",
        help="Quick test (3 epochs); checkpoints go to checkpoints/phase5/test",
    )
    parser.add_argument(
        "--train-seasons",
        type=str,
        default=",".join(TRAIN_SEASONS),
        help="Start,end seasons for training (inclusive)",
    )
    parser.add_argument("--val-season", type=str, default=VAL_SEASON)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # --- Load data ---
    data = load_phase_b_data()

    train_start, train_end = args.train_seasons.split(",")
    train_ds, val_ds = make_train_val_datasets(
        data,
        train_seasons=(train_start, train_end),
        val_season=args.val_season,
    )

    logger.info(f"Train dataset: {len(train_ds)} games")
    logger.info(f"Val dataset: {len(val_ds)} games")

    if len(train_ds) == 0 or len(val_ds) == 0:
        logger.error("Empty dataset! Check season ranges and Phase B cache.")
        sys.exit(1)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # --- Create models ---
    l3_cfg = L3Config()
    l4_cfg = L4Config()

    l3_model = TeamModel(l3_cfg).to(device)
    l4_model = GamePredictor(l4_cfg).to(device)

    l3_params = l3_count_params(l3_model)
    l4_params = l4_count_params(l4_model)
    logger.info(f"L3 TeamModel: {l3_params:,} params")
    logger.info(f"L4 GamePredictor: {l4_params:,} params")
    logger.info(f"Total trainable: {l3_params + l4_params:,} params")

    # --- Optimizer with param groups ---
    optimizer = torch.optim.AdamW(
        [
            {"params": l3_model.parameters(), "lr": args.l3_lr},
            {"params": l4_model.parameters(), "lr": args.l4_lr},
        ],
        weight_decay=args.weight_decay,
    )

    # Cosine annealing schedule
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6,
    )

    # --- Training loop ---
    out_dir = CHECKPOINT_DIR / "test" if args.test else CHECKPOINT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    best_val_mae = float("inf")
    best_epoch = 0
    patience_counter = 0
    history = []

    logger.info(f"\nStarting training for {args.epochs} epochs...")
    logger.info(
        f"  L3 lr={args.l3_lr}, L4 lr={args.l4_lr}, "
        f"wd={args.weight_decay}, batch={args.batch_size}, "
        f"patience={args.patience}"
    )

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Train
        train_losses = train_epoch(
            l3_model,
            l4_model,
            train_loader,
            optimizer,
            scheduler,
            l4_cfg,
            device,
            args.grad_clip,
        )

        # Validate
        val_losses, val_metrics = evaluate(
            l3_model,
            l4_model,
            val_loader,
            l4_cfg,
            device,
        )

        elapsed = time.time() - t0

        # Log
        lr_l3 = optimizer.param_groups[0]["lr"]
        lr_l4 = optimizer.param_groups[1]["lr"]
        logger.info(
            f"Epoch {epoch:3d}/{args.epochs} ({elapsed:.1f}s) | "
            f"Train loss={train_losses['total_loss']:.4f} "
            f"(spread={train_losses['spread']:.4f}, win={train_losses['win']:.4f}) | "
            f"Val MAE={val_metrics['spread_mae']:.2f}, "
            f"AUC={val_metrics['win_auc']:.3f}, "
            f"WinAcc={val_metrics['win_acc']:.3f}, "
            f"ECE={val_metrics['ece']:.4f}, "
            f"sigma={val_metrics['mean_sigma']:.2f} | "
            f"lr={lr_l3:.2e}/{lr_l4:.2e}"
        )

        # Track history
        epoch_data = {
            "epoch": epoch,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "val_metrics": val_metrics,
            "lr_l3": lr_l3,
            "lr_l4": lr_l4,
            "time_s": elapsed,
        }
        history.append(epoch_data)

        # Checkpoint (best val spread MAE)
        val_mae = val_metrics["spread_mae"]
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_val_loss = val_losses["total_loss"]
            best_epoch = epoch
            patience_counter = 0

            torch.save(
                {
                    "epoch": epoch,
                    "l3_state_dict": l3_model.state_dict(),
                    "l4_state_dict": l4_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "l3_config": l3_cfg.__dict__,
                    "l4_config": l4_cfg.__dict__,
                    "val_metrics": val_metrics,
                    "val_losses": val_losses,
                },
                str(out_dir / "phase_b_best.pt"),
            )
            logger.info(f"  ** New best: MAE={best_val_mae:.3f} at epoch {epoch}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info(
                    f"Early stopping at epoch {epoch} (patience={args.patience})"
                )
                break

        # Quick test mode: only run a few epochs
        if args.test and epoch >= 3:
            logger.info("Test mode: stopping after 3 epochs.")
            break

    # --- Final summary ---
    logger.info(f"\n{'='*70}")
    logger.info(f"Training complete!")
    logger.info(f"  Best epoch: {best_epoch}")
    logger.info(f"  Best val spread MAE: {best_val_mae:.3f}")
    logger.info(f"  Best val loss: {best_val_loss:.4f}")

    # Load best and evaluate one more time for full metrics
    best_ckpt = torch.load(
        str(out_dir / "phase_b_best.pt"),
        map_location=device,
        weights_only=False,
    )
    l3_model.load_state_dict(best_ckpt["l3_state_dict"])
    l4_model.load_state_dict(best_ckpt["l4_state_dict"])

    _, final_metrics = evaluate(l3_model, l4_model, val_loader, l4_cfg, device)
    logger.info(
        f"\nFinal validation metrics (best checkpoint, epoch {best_ckpt['epoch']}):"
    )
    for k, v in final_metrics.items():
        logger.info(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # Save training history
    # Convert any non-serializable values
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (np.floating, float)):
            return float(obj)
        elif isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        return obj

    results = {
        "args": vars(args),
        "l3_params": l3_params,
        "l4_params": l4_params,
        "best_epoch": best_epoch,
        "best_val_mae": best_val_mae,
        "final_metrics": final_metrics,
        "history": make_serializable(history),
    }
    results_path = RESULTS_DIR / "phase_b_results.json"
    with open(str(results_path), "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {results_path}")

    # Also save final checkpoint
    torch.save(
        {
            "epoch": epoch,
            "l3_state_dict": l3_model.state_dict(),
            "l4_state_dict": l4_model.state_dict(),
            "l3_config": l3_cfg.__dict__,
            "l4_config": l4_cfg.__dict__,
            "val_metrics": final_metrics,
        },
        str(out_dir / "phase_b_final.pt"),
    )
    logger.info(f"Final checkpoint saved to {out_dir / 'phase_b_final.pt'}")


if __name__ == "__main__":
    main()
