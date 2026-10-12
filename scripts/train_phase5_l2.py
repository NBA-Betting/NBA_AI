#!/usr/bin/env python3
"""
Phase 5 L2: Player Synergy Network Training Script.

Curriculum training:
  Epochs 1-10:  2-man WOWY loss only (w_5man=0.0)
  Epochs 11+:   introduce 5-man lineup loss (w_5man=0.3)

Usage:
    python scripts/train_phase5_l2.py                    # Full training
    python scripts/train_phase5_l2.py --epochs 5 --test  # Quick test (few batches)
"""

import argparse
import json
import logging
import math
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.phase5.l2_config import L2Config
from src.phase5.l2_model import PlayerSynergyNetwork, count_parameters
from src.phase5.l2_dataset import PairDataset, LineupDataset
from src.phase5.l2_losses import l2_total_loss, lineup_loss

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "phase5"
L2_CACHE_DIR = PROJECT_ROOT / "data" / "l2_cache"
L1_VECTORS_DIR = L2_CACHE_DIR / "l1_vectors"
WOWY_DB_PATH = L2_CACHE_DIR / "wowy.sqlite"

# Curriculum: epochs before introducing 5-man loss (default, overridden by cfg)
CURRICULUM_WARMUP = 10


def get_effective_w_5man(cfg: L2Config, epoch: int) -> float:
    """
    Compute effective 5-man loss weight for a given epoch.

    Before warmup: 0.0
    After warmup: linear ramp from 0 to cfg.w_5man over cfg.lineup_ramp_epochs.

    This avoids the abrupt curriculum transition that causes gradient conflict.
    """
    warmup = getattr(cfg, "curriculum_warmup", CURRICULUM_WARMUP)
    ramp_epochs = getattr(cfg, "lineup_ramp_epochs", 0)

    if epoch <= warmup:
        return 0.0

    if ramp_epochs <= 0:
        return cfg.w_5man

    elapsed = epoch - warmup
    ramp_frac = min(1.0, elapsed / ramp_epochs)
    return cfg.w_5man * ramp_frac


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def validate_data_available() -> None:
    """Check that required data files exist, with clear error messages."""
    errors = []

    if not L2_CACHE_DIR.exists():
        errors.append(
            f"L2 cache directory not found: {L2_CACHE_DIR}\n"
            f"  Build caches first (L1 vectors + WOWY extraction)."
        )
    else:
        if not L1_VECTORS_DIR.exists() or not any(L1_VECTORS_DIR.glob("*.npz")):
            errors.append(
                f"No L1 vector files found in: {L1_VECTORS_DIR}\n"
                f"  Run L1 inference to export player ability vectors."
            )

        if not WOWY_DB_PATH.exists():
            errors.append(
                f"WOWY database not found: {WOWY_DB_PATH}\n"
                f"  Run WOWY stint extraction first."
            )
        else:
            # Check that tables have data
            conn = sqlite3.connect(str(WOWY_DB_PATH))
            pair_count = conn.execute("SELECT COUNT(*) FROM PairwiseWOWY").fetchone()[0]
            lineup_count = conn.execute("SELECT COUNT(*) FROM LineupStints").fetchone()[
                0
            ]
            conn.close()

            if pair_count == 0:
                errors.append(
                    f"PairwiseWOWY table is empty in {WOWY_DB_PATH}\n"
                    f"  Run WOWY stint extraction to populate pairwise data."
                )
            if lineup_count == 0:
                errors.append(
                    f"LineupStints table is empty in {WOWY_DB_PATH}\n"
                    f"  Run WOWY stint extraction to populate lineup data.\n"
                    f"  (Training can proceed without lineup data — 5-man loss will be skipped.)"
                )

    if errors:
        for e in errors:
            logging.error(e)
        sys.exit(1)


def load_l1_vectors() -> dict[int, dict]:
    """
    Load all L1 ability vectors from npz files.

    Returns:
        {player_id: {ability: (N,32), uncertainty: (N,32),
                      game_ids: (N,), archetype_weights: (10,)}}
    """
    vectors = {}
    npz_files = sorted(L1_VECTORS_DIR.glob("*.npz"))
    logging.info(f"Loading L1 vectors from {len(npz_files)} files...")

    for path in npz_files:
        player_id = int(path.stem)
        data = np.load(str(path), allow_pickle=True)
        vectors[player_id] = {
            "ability": data["ability"],
            "uncertainty": data["uncertainty"],
            "game_ids": data["game_ids"],
            "archetype_weights": data["archetype_weights"],
        }

    logging.info(f"Loaded L1 vectors for {len(vectors)} players")
    return vectors


def build_player_to_idx(l1_vectors: dict[int, dict]) -> dict[int, int]:
    """
    Build player_id -> embedding index mapping.
    Index 0 is reserved for unknown/padding players.
    """
    player_ids = sorted(l1_vectors.keys())
    mapping = {pid: i + 1 for i, pid in enumerate(player_ids)}
    logging.info(
        f"Player-to-idx mapping: {len(mapping)} players (indices 1-{len(mapping)})"
    )
    return mapping


def season_to_year(season: str) -> int:
    """Convert '2007-2008' to 2007 (start year)."""
    return int(season.split("-")[0])


def load_pair_data(
    seasons: tuple[str, str] | None = None,
    val_season: str | None = None,
) -> tuple[list[dict], list[dict]]:
    """
    Load pairwise WOWY data from SQLite.

    Args:
        seasons: (start_season, end_season) inclusive range for training
        val_season: season to use for validation

    Returns:
        (train_pairs, val_pairs) — each a list of dicts
    """
    conn = sqlite3.connect(str(WOWY_DB_PATH))
    conn.row_factory = sqlite3.Row

    rows = conn.execute("""
        SELECT player_a, player_b, season, team_id,
               minutes_together, poss_together,
               net_rtg_together, net_rtg_a_only, net_rtg_b_only,
               stints_together, games_together
        FROM PairwiseWOWY
        WHERE poss_together > 0
        """).fetchall()
    conn.close()

    train_pairs = []
    val_pairs = []

    if seasons is not None:
        train_start = season_to_year(seasons[0])
        train_end = season_to_year(seasons[1])
    else:
        train_start, train_end = 2007, 2016

    val_start = season_to_year(val_season) if val_season else 2016

    for row in rows:
        season = row["season"]
        year = season_to_year(season)

        # WOWY residual: net_rtg(both) - net_rtg(A_only) - net_rtg(B_only)
        net_rtg_together = row["net_rtg_together"] or 0.0
        net_rtg_a_only = row["net_rtg_a_only"] or 0.0
        net_rtg_b_only = row["net_rtg_b_only"] or 0.0
        wowy_residual = net_rtg_together - net_rtg_a_only - net_rtg_b_only

        pair = {
            "player_a": row["player_a"],
            "player_b": row["player_b"],
            "season": season,
            "wowy_residual": wowy_residual,
            "shared_possessions": row["poss_together"],
            "minutes_together": row["minutes_together"],
            "years_together": 0,  # Will be enriched later if needed
        }

        if year == val_start:
            val_pairs.append(pair)
        elif train_start <= year <= train_end:
            train_pairs.append(pair)

    logging.info(
        f"PairwiseWOWY: {len(train_pairs)} train pairs, {len(val_pairs)} val pairs "
        f"(train {train_start}-{train_end}, val {val_start})"
    )
    return train_pairs, val_pairs


def load_lineup_data(
    seasons: tuple[str, str] | None = None,
    val_season: str | None = None,
    min_possessions: float = 50.0,
) -> tuple[list[dict], list[dict]]:
    """
    Load lineup stint data from SQLite.

    Returns:
        (train_lineups, val_lineups) — each a list of dicts
    """
    conn = sqlite3.connect(str(WOWY_DB_PATH))
    conn.row_factory = sqlite3.Row

    count = conn.execute("SELECT COUNT(*) FROM LineupStints").fetchone()[0]
    if count == 0:
        conn.close()
        logging.warning("LineupStints table is empty — 5-man loss will be disabled.")
        return [], []

    rows = conn.execute(
        """
        SELECT lineup_hash, season, team_id, player_ids,
               total_poss, total_pts_for, total_pts_against,
               off_rating, def_rating, n_stints, n_games
        FROM LineupStints
        WHERE total_poss >= ?
        """,
        (min_possessions,),
    ).fetchall()
    conn.close()

    train_lineups = []
    val_lineups = []

    if seasons is not None:
        train_start = season_to_year(seasons[0])
        train_end = season_to_year(seasons[1])
    else:
        train_start, train_end = 2007, 2016

    val_start = season_to_year(val_season) if val_season else 2016

    for row in rows:
        season = row["season"]
        year = season_to_year(season)

        player_ids_str = row["player_ids"]
        player_ids = [int(pid) for pid in player_ids_str.split(",")]

        # Net rating per 100 possessions
        total_poss = row["total_poss"]
        if total_poss > 0:
            net_rating = (
                (row["total_pts_for"] - row["total_pts_against"]) / total_poss
            ) * 100.0
        else:
            continue

        lineup = {
            "player_ids": player_ids,
            "team_id": row["team_id"],
            "season": season,
            "net_rating": net_rating,
            "possessions": total_poss,
        }

        if year == val_start:
            val_lineups.append(lineup)
        elif train_start <= year <= train_end:
            train_lineups.append(lineup)

    logging.info(
        f"LineupStints: {len(train_lineups)} train lineups, {len(val_lineups)} val lineups "
        f"(min_poss={min_possessions})"
    )
    return train_lineups, val_lineups


def build_pair_features_index(
    train_pairs: list[dict], val_pairs: list[dict]
) -> dict[tuple[int, int, str], dict]:
    """
    Build a lookup from (player_a, player_b, season) -> edge features
    for use by LineupDataset.
    """
    index = {}
    for pair in train_pairs + val_pairs:
        key = (
            min(pair["player_a"], pair["player_b"]),
            max(pair["player_a"], pair["player_b"]),
            pair["season"],
        )
        index[key] = {
            "shared_minutes": pair.get("minutes_together", 0.0),
            "wowy_net_rtg": pair.get("wowy_residual", 0.0),
            "years_together": pair.get("years_together", 0),
            "minutes_overlap_pct": 0.0,  # placeholder
        }
    return index


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_one_epoch_pairs(
    model: PlayerSynergyNetwork,
    pair_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    cfg: L2Config,
    device: str,
    test_mode: bool = False,
) -> dict[str, float]:
    """Train one epoch on pairwise WOWY data."""
    model.train()
    total_losses = {}
    n_batches = 0

    for batch in pair_loader:
        ability = batch["ability"].to(device)  # (B, 2, 32)
        archetypes = batch["archetypes"].to(device)  # (B, 2, 10)
        player_idx = batch["player_idx"].to(device)  # (B, 2)
        target_wowy = batch["target_wowy"].to(device)  # (B,)
        shared_poss = batch["shared_possessions"].to(device)  # (B,)

        # Predict pairwise synergy
        pred_synergy = model.predict_pairwise_synergy(
            ability, archetypes, player_idx
        )  # (B,)

        losses = l2_total_loss(
            pred_pair_synergy=pred_synergy,
            target_wowy=target_wowy,
            pair_possessions=shared_poss,
            model=model,
            cfg=cfg,
        )

        optimizer.zero_grad()
        losses["total"].backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip)
        optimizer.step()

        for k, v in losses.items():
            total_losses[k] = total_losses.get(k, 0) + v.item()
        n_batches += 1

        if test_mode and n_batches >= 5:
            break

    for k in total_losses:
        total_losses[k] /= max(n_batches, 1)
    return total_losses


def predict_lineup(
    model: PlayerSynergyNetwork, batch: dict, device: str, freeze_fm: bool = False
) -> torch.Tensor:
    """Lineup net rating prediction: individual contributions plus the sum of
    pairwise synergies. Shared by lineup training and validation.

    freeze_fm: detach the FM/archetype synergies and use the GATv2-pooled team
    vector norm as the individual term, so only GATv2/pooling layers learn.
    """
    ability = batch["ability"].to(device)
    mask = batch["mask"].to(device)
    outputs = model(
        ability=ability,
        uncertainty=batch["uncertainty"].to(device),
        archetypes=batch["archetypes"].to(device),
        mask=mask,
        player_idx=batch["player_idx"].to(device),
        edge_features=batch["edge_features"].to(device),
    )
    pairwise_total = outputs["pairwise_total"]  # (B, A, A)
    valid_pairs = (mask.unsqueeze(-1) & mask.unsqueeze(-2)) & ~torch.eye(
        pairwise_total.shape[1], dtype=torch.bool, device=device
    ).unsqueeze(0)
    if freeze_fm:
        sum_pairwise = (pairwise_total.detach() * valid_pairs.float()).sum(dim=(1, 2))
        sum_individual = outputs["team_player"].norm(dim=-1)
    else:
        sum_pairwise = (pairwise_total * valid_pairs.float()).sum(dim=(1, 2))
        sum_individual = (ability.norm(dim=-1) * mask.float()).sum(dim=1)
    return sum_individual + sum_pairwise


def train_one_epoch_lineups(
    model: PlayerSynergyNetwork,
    lineup_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    cfg: L2Config,
    device: str,
    test_mode: bool = False,
    freeze_fm: bool = False,
) -> dict[str, float]:
    """Train one epoch on 5-man lineup data.

    Args:
        freeze_fm: If True, freeze FM synergy vectors (MLP + residual + gate)
                   during lineup training to prevent gradient conflict with
                   the pairwise objective. Only GATv2, pooling, and synergy
                   aggregation layers are updated.
    """
    model.train()
    total_losses = {}
    n_batches = 0

    for batch in lineup_loader:
        pred_lineup = predict_lineup(model, batch, device, freeze_fm)
        L_5man = lineup_loss(
            pred_lineup,
            batch["target_net_rtg"].to(device),
            batch["possessions"].to(device),
        )
        loss = cfg.w_5man * L_5man

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip)
        optimizer.step()

        total_losses["lineup"] = total_losses.get("lineup", 0) + L_5man.item()
        total_losses["total"] = total_losses.get("total", 0) + loss.item()
        n_batches += 1

        if test_mode and n_batches >= 3:
            break

    for k in total_losses:
        total_losses[k] /= max(n_batches, 1)
    return total_losses


@torch.no_grad()
def validate(
    model: PlayerSynergyNetwork,
    pair_loader: DataLoader,
    lineup_loader: DataLoader | None,
    cfg: L2Config,
    device: str,
    freeze_fm: bool = False,
    test_mode: bool = False,
) -> dict[str, float]:
    """
    Validate on held-out season.

    Reports:
    - Pairwise: MAE, Huber loss, sign accuracy
    - Lineup: training loss and MAE (when lineup data is available)
    - Total: pair loss + cfg.w_5man * lineup loss
    """
    model.eval()
    metrics = {
        "pair_loss": 0.0,
        "pair_mae": 0.0,
        "pair_sign_acc": 0.0,
    }
    n_pair_batches = 0
    n_pair_samples = 0

    # --- Pairwise validation ---
    for batch in pair_loader:
        ability = batch["ability"].to(device)
        archetypes = batch["archetypes"].to(device)
        player_idx = batch["player_idx"].to(device)
        target_wowy = batch["target_wowy"].to(device)
        shared_poss = batch["shared_possessions"].to(device)

        pred_synergy = model.predict_pairwise_synergy(ability, archetypes, player_idx)

        losses = l2_total_loss(
            pred_pair_synergy=pred_synergy,
            target_wowy=target_wowy,
            pair_possessions=shared_poss,
            model=model,
            cfg=cfg,
        )

        B = target_wowy.shape[0]
        metrics["pair_loss"] += losses["pairwise"].item() * B
        metrics["pair_mae"] += (pred_synergy - target_wowy).abs().sum().item()

        # Sign accuracy: does predicted direction match target direction?
        # (exclude near-zero targets to avoid noise)
        significant = target_wowy.abs() > 0.5
        if significant.any():
            correct_sign = (
                (pred_synergy[significant].sign() == target_wowy[significant].sign())
                .float()
                .sum()
                .item()
            )
            metrics["pair_sign_acc"] += correct_sign
            n_pair_samples += significant.sum().item()

        n_pair_batches += B

        if test_mode and n_pair_batches >= 256:
            break

    if n_pair_batches > 0:
        metrics["pair_loss"] /= n_pair_batches
        metrics["pair_mae"] /= n_pair_batches
    if n_pair_samples > 0:
        metrics["pair_sign_acc"] /= n_pair_samples

    # --- Lineup validation: the training lineup loss (possession-weighted
    # Huber over lineups with enough possessions), every epoch, so the model
    # selection metric keeps one scale through the curriculum ---
    metrics["lineup_loss"] = 0.0
    metrics["lineup_mae"] = 0.0
    n_lineup_samples = 0
    if lineup_loader is not None:
        for batch in lineup_loader:
            pred_lineup = predict_lineup(model, batch, device, freeze_fm)
            target = batch["target_net_rtg"].to(device)
            B = target.shape[0]
            metrics["lineup_loss"] += (
                lineup_loss(pred_lineup, target, batch["possessions"].to(device)).item() * B
            )
            metrics["lineup_mae"] += (pred_lineup - target).abs().sum().item()
            n_lineup_samples += B
            if test_mode and n_lineup_samples >= 64:
                break
        if n_lineup_samples > 0:
            metrics["lineup_loss"] /= n_lineup_samples
            metrics["lineup_mae"] /= n_lineup_samples

    # Selection metric: fixed weights (cfg.w_5man, not the warm-up ramp)
    metrics["total"] = metrics["pair_loss"] + cfg.w_5man * metrics["lineup_loss"]

    return metrics


# ---------------------------------------------------------------------------
# Checkpoint utilities
# ---------------------------------------------------------------------------


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_metrics: dict,
    path: Path,
    player_to_idx: dict[int, int],
):
    """Save model checkpoint, with the player -> embedding row mapping (the
    game cache must index players the same way)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_metrics": val_metrics,
            "player_to_idx": {str(k): v for k, v in player_to_idx.items()},
        },
        path,
    )
    logging.info(f"  Checkpoint saved: {path}")


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


def train(
    cfg: L2Config,
    epochs: int = 30,
    device: str = "cuda",
    test_mode: bool = False,
):
    """Full L2 training with curriculum."""
    logging.info("=== Phase 5 L2: Player Synergy Network Training ===")

    # --- Load data ---
    l1_vectors = load_l1_vectors()
    player_to_idx = build_player_to_idx(l1_vectors)

    train_pairs, val_pairs = load_pair_data(
        seasons=cfg.pretrain_seasons,
        val_season=cfg.val_season,
    )
    train_lineups, val_lineups = load_lineup_data(
        seasons=cfg.pretrain_seasons,
        val_season=cfg.val_season,
        min_possessions=getattr(cfg, "lineup_min_possessions", 50),
    )

    has_lineups = len(train_lineups) > 0

    # --- Build pair features index for lineup edge features ---
    pair_features = build_pair_features_index(train_pairs, val_pairs)

    # --- Adjust n_players in config to match actual data ---
    n_players_needed = len(player_to_idx) + 1  # +1 for index 0 (padding)
    if n_players_needed > cfg.n_players:
        logging.warning(
            f"More players ({n_players_needed}) than cfg.n_players ({cfg.n_players}). "
            f"Increasing n_players."
        )
        cfg.n_players = n_players_needed

    # --- Build datasets ---
    train_pair_ds = PairDataset(train_pairs, l1_vectors, player_to_idx, cfg)
    val_pair_ds = PairDataset(val_pairs, l1_vectors, player_to_idx, cfg)

    train_lineup_ds = None
    val_lineup_ds = None
    if has_lineups:
        train_lineup_ds = LineupDataset(
            train_lineups, l1_vectors, player_to_idx, pair_features, cfg
        )
        val_lineup_ds = LineupDataset(
            val_lineups, l1_vectors, player_to_idx, pair_features, cfg
        )

    # --- DataLoaders ---
    train_pair_loader = DataLoader(
        train_pair_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    val_pair_loader = DataLoader(
        val_pair_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    train_lineup_loader = None
    val_lineup_loader = None
    if has_lineups and train_lineup_ds is not None:
        train_lineup_loader = DataLoader(
            train_lineup_ds,
            batch_size=cfg.lineup_batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        )
        val_lineup_loader = DataLoader(
            val_lineup_ds,
            batch_size=cfg.lineup_batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )

    logging.info(
        f"Datasets: {len(train_pair_ds)} train pairs, {len(val_pair_ds)} val pairs"
    )
    if has_lineups:
        logging.info(
            f"          {len(train_lineup_ds)} train lineups, {len(val_lineup_ds)} val lineups"
        )
    else:
        logging.info("          No lineup data — 5-man loss disabled for all epochs.")

    # --- Model ---
    model = PlayerSynergyNetwork(cfg).to(device)
    logging.info(f"Model parameters: {count_parameters(model):,}")

    # --- Optimizer / Scheduler ---
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    patience_counter = 0

    warmup = getattr(cfg, "curriculum_warmup", CURRICULUM_WARMUP)
    ramp_epochs = getattr(cfg, "lineup_ramp_epochs", 0)
    freeze_fm = getattr(cfg, "freeze_fm_for_lineup", False)
    min_poss = getattr(cfg, "lineup_min_possessions", 50)

    logging.info(
        f"Training: {epochs} epochs, curriculum warmup={warmup} epochs "
        f"(2-man only), then w_5man ramps to {cfg.w_5man} over {ramp_epochs} epochs"
    )
    logging.info(
        f"Patience: {cfg.patience}, gradient clip: {cfg.gradient_clip}, "
        f"freeze_fm_for_lineup: {freeze_fm}, lineup_min_poss: {min_poss}"
    )
    logging.info("-" * 80)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        w_5man_eff = get_effective_w_5man(cfg, epoch) if has_lineups else 0.0
        use_5man = w_5man_eff > 0.0

        # --- Curriculum: temporarily set w_5man for loss computation ---
        effective_cfg = L2Config(
            **{k: getattr(cfg, k) for k in cfg.__dataclass_fields__ if k != "w_5man"},
            w_5man=w_5man_eff,
        )

        # --- Train on pairs (primary) ---
        pair_losses = train_one_epoch_pairs(
            model, train_pair_loader, optimizer, effective_cfg, device, test_mode
        )

        # --- Train on lineups (secondary, after warmup) ---
        lineup_losses = {}
        if use_5man and train_lineup_loader is not None:
            lineup_losses = train_one_epoch_lineups(
                model,
                train_lineup_loader,
                optimizer,
                effective_cfg,
                device,
                test_mode,
                freeze_fm=freeze_fm,
            )

        scheduler.step()

        # --- Validate ---
        val_metrics = validate(
            model,
            val_pair_loader,
            val_lineup_loader,
            cfg,
            device,
            freeze_fm=freeze_fm,
            test_mode=test_mode,
        )

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        # --- Logging ---
        phase_str = f"2man+5man(w={w_5man_eff:.3f})" if use_5man else "2man-only"
        lineup_str = ""
        if lineup_losses:
            lineup_str = f" lineup={lineup_losses.get('lineup', 0):.4f}"

        val_lineup_str = ""
        if val_metrics.get("lineup_mae", 0) > 0:
            val_lineup_str = f" lineup_mae={val_metrics['lineup_mae']:.3f}"

        logging.info(
            f"Epoch {epoch:3d}/{epochs} [{phase_str}] | LR {lr:.2e} | {elapsed:.1f}s | "
            f"Train: pair={pair_losses.get('pairwise', 0):.4f} "
            f"fm_reg={pair_losses.get('fm_reg', 0):.4f}{lineup_str} | "
            f"Val: pair_mae={val_metrics['pair_mae']:.3f} "
            f"sign_acc={val_metrics['pair_sign_acc']:.3f}{val_lineup_str} "
            f"total={val_metrics['total']:.4f}"
        )

        # --- Checkpointing ---
        if val_metrics["total"] < best_val_loss:
            best_val_loss = val_metrics["total"]
            patience_counter = 0
            save_checkpoint(
                model,
                optimizer,
                epoch,
                val_metrics,
                CHECKPOINT_DIR / "l2_best.pt",
                player_to_idx,
            )
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience:
                logging.info(
                    f"Early stopping at epoch {epoch} (patience={cfg.patience})"
                )
                break

    # --- Save final checkpoint ---
    save_checkpoint(
        model,
        optimizer,
        epoch,
        val_metrics,
        CHECKPOINT_DIR / "l2_final.pt",
        player_to_idx,
    )

    logging.info("-" * 80)
    logging.info(f"Training complete. Best val loss: {best_val_loss:.4f}")
    logging.info(f"Checkpoints: {CHECKPOINT_DIR / 'l2_best.pt'}")

    return model


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def get_experiment_configs(epochs: int = 30) -> dict[str, L2Config]:
    """
    Return named experiment configs for the sign accuracy regression diagnosis.

    Variants:
      A (baseline): Original settings — w_5man=0.3, abrupt curriculum, 50 poss
      B (low_w): Lower w_5man=0.05, so 5-man gradient is 0.2x of 2-man (not 1.2x)
      C (ramp): Gradual ramp over 10 epochs + higher possession threshold (200)
      D (freeze_fm): Freeze FM vectors during lineup phase — only GATv2/pool update
      E (ramp+freeze): Combined: gradual ramp + FM frozen for lineups
      F (pair_only): No 5-man loss at all — pure pairwise baseline
    """
    configs = {}

    # A: Original (reproduces the sign accuracy regression)
    configs["A_baseline"] = L2Config(
        w_5man=0.3,
        curriculum_warmup=10,
        lineup_ramp_epochs=0,  # abrupt
        lineup_min_possessions=50,
        freeze_fm_for_lineup=False,
    )

    # B: Lower weight — reduce gradient domination from 1.2x to 0.2x
    configs["B_low_w"] = L2Config(
        w_5man=0.05,
        curriculum_warmup=10,
        lineup_ramp_epochs=0,
        lineup_min_possessions=50,
        freeze_fm_for_lineup=False,
    )

    # C: Gradual ramp + higher possession threshold for cleaner targets
    configs["C_ramp_200poss"] = L2Config(
        w_5man=0.1,
        curriculum_warmup=10,
        lineup_ramp_epochs=10,  # ramp over 10 epochs
        lineup_min_possessions=200,  # cleaner targets (std 13 vs 23)
        freeze_fm_for_lineup=False,
    )

    # D: Freeze FM vectors during lineup phase
    configs["D_freeze_fm"] = L2Config(
        w_5man=0.3,
        curriculum_warmup=10,
        lineup_ramp_epochs=0,
        lineup_min_possessions=50,
        freeze_fm_for_lineup=True,
    )

    # E: Combined: ramp + freeze
    configs["E_ramp_freeze"] = L2Config(
        w_5man=0.1,
        curriculum_warmup=10,
        lineup_ramp_epochs=10,
        lineup_min_possessions=200,
        freeze_fm_for_lineup=True,
    )

    # F: Pure pairwise (no 5-man loss) — upper bound for sign accuracy
    configs["F_pair_only"] = L2Config(
        w_5man=0.0,
        curriculum_warmup=999,  # never introduce 5-man
        lineup_ramp_epochs=0,
        lineup_min_possessions=50,
        freeze_fm_for_lineup=False,
    )

    return configs


def main():
    parser = argparse.ArgumentParser(
        description="Train Phase 5 L2 Player Synergy Network"
    )
    parser.add_argument(
        "--epochs", type=int, default=None, help="Override number of epochs"
    )
    parser.add_argument(
        "--test", action="store_true", help="Quick test mode (few batches per epoch)"
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        choices=[
            "A_baseline",
            "B_low_w",
            "C_ramp_200poss",
            "D_freeze_fm",
            "E_ramp_freeze",
            "F_pair_only",
            "compare_all",
        ],
        help="Run a specific experiment variant or 'compare_all' for full comparison",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        logging.warning("CUDA not available, falling back to CPU")
        device = "cpu"

    # Validate data prerequisites
    validate_data_available()

    if args.experiment == "compare_all":
        # Run all variants and print comparison
        all_configs = get_experiment_configs()
        epochs = args.epochs if args.epochs is not None else 30
        if args.test:
            epochs = min(epochs, 3)

        results = {}
        for name, cfg in all_configs.items():
            logging.info(f"\n{'='*80}")
            logging.info(f"  EXPERIMENT: {name}")
            logging.info(f"{'='*80}")
            model = train(cfg, epochs=epochs, device=device, test_mode=args.test)

            # Load best checkpoint and report metrics
            ckpt_path = CHECKPOINT_DIR / "l2_best.pt"
            if ckpt_path.exists():
                ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
                results[name] = ckpt.get("val_metrics", {})
                # Save per-experiment checkpoint
                exp_ckpt = CHECKPOINT_DIR / f"l2_{name}.pt"
                torch.save(ckpt, exp_ckpt)
                logging.info(f"  Saved experiment checkpoint: {exp_ckpt}")

        # Print comparison table
        logging.info(f"\n{'='*80}")
        logging.info("  COMPARISON TABLE")
        logging.info(f"{'='*80}")
        logging.info(
            f"{'Experiment':<20s} {'sign_acc':>10s} {'pair_mae':>10s} {'lineup_mae':>10s} {'total':>10s}"
        )
        logging.info("-" * 62)
        for name, metrics in results.items():
            logging.info(
                f"{name:<20s} "
                f"{metrics.get('pair_sign_acc', 0):.4f}     "
                f"{metrics.get('pair_mae', 0):.4f}     "
                f"{metrics.get('lineup_mae', 0):.4f}     "
                f"{metrics.get('total', 0):.4f}"
            )
    elif args.experiment:
        all_configs = get_experiment_configs()
        cfg = all_configs[args.experiment]
        epochs = args.epochs if args.epochs is not None else cfg.epochs
        if args.test:
            epochs = min(epochs, 3)
        train(cfg, epochs=epochs, device=device, test_mode=args.test)
    else:
        cfg = L2Config()
        epochs = args.epochs if args.epochs is not None else cfg.epochs
        if args.test:
            logging.info("TEST MODE: running with reduced batches per epoch")
            epochs = min(epochs, 3)
        train(cfg, epochs=epochs, device=device, test_mode=args.test)


if __name__ == "__main__":
    main()
