#!/usr/bin/env python3
"""
Phase C Training: L2+L3+L4 joint fine-tuning with discriminative learning rates.

Unfreezes L2, L3, and L4 jointly. L1 remains frozen — we reuse pre-computed
L1 per-player ability vectors from the L2 cache. The key difference from Phase B:
L2 forward passes are live (unfrozen), so gradients flow back through L2.

Gradual unfreezing schedule:
  - Epochs 1-3:  L4 only
  - Epochs 4-6:  L3 + L4
  - Epochs 7+:   L2 + L3 + L4

Discriminative learning rates:
  - L2: 3e-5  (lowest — preserve learned synergy representations)
  - L3: 1e-4
  - L4: 3e-4  (highest — prediction head adapts fastest)

Usage:
    python scripts/train_phase5_c.py                      # Full training
    python scripts/train_phase5_c.py --test --epochs 5    # Quick test
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.database import DB_PATH, get_db
from src.phase5.game_dataset import (
    TRAIN_SEASONS,
    VAL_SEASON,
    load_phase_b_data,
    split_and_normalize,
)
from src.phase5.l1_store import L1Store, load_game_dates
from src.phase5.l2_config import L2Config
from src.phase5.l2_model import (
    PlayerSynergyNetwork,
    count_parameters as l2_count_params,
)
from src.phase5.l3_config import L3Config
from src.phase5.l3_model import TeamModel, count_parameters as l3_count_params
from src.phase5.l4_config import L4Config
from src.phase5.l4_model import GamePredictor, count_parameters as l4_count_params
from src.phase5.l4_losses import game_prediction_loss, prediction_metrics

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "phase5"
RESULTS_DIR = PROJECT_ROOT / "results" / "phase5"

# Built by scripts/build_phase5_cache.py
GAMES_CACHE = PROJECT_ROOT / "data" / "phase5_games"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Phase C Dataset: per-player L1 vectors + L3/L4 features per game
# ---------------------------------------------------------------------------


class PhaseCDataset(Dataset):
    """
    Each sample provides per-player L1 vectors (for live L2 forward) plus
    team features, game context, and targets (for L3/L4 forward).

    L1 vectors are frozen: loaded from pre-computed cache, no gradient.
    L2 runs live on these vectors each forward pass.
    """

    def __init__(
        self,
        indices: np.ndarray,
        # Per-game player data (N, 2, max_roster, ...)
        player_abilities: np.ndarray,  # (N, 2, A, 32)
        player_uncertainties: np.ndarray,  # (N, 2, A, 32)
        player_archetypes: np.ndarray,  # (N, 2, A, 10)
        player_masks: np.ndarray,  # (N, 2, A) bool
        player_indices: np.ndarray,  # (N, 2, A) int64
        # Phase B cache data (shared arrays, indexed by full game position)
        team_features: np.ndarray,  # (N, 2, 34)
        game_context: np.ndarray,  # (N, 14)
        roster_summaries: np.ndarray,  # (N, 2, 12)
        targets_margin: np.ndarray,  # (N,)
        targets_win: np.ndarray,  # (N,)
        # Coach data
        coach_indices: np.ndarray | None = None,  # (N, 2) int64
        coach_games: np.ndarray | None = None,  # (N, 2) float32
        # Normalization (from training set)
        l3_mean: np.ndarray | None = None,
        l3_std: np.ndarray | None = None,
        l4_mean: np.ndarray | None = None,
        l4_std: np.ndarray | None = None,
        rs_mean: np.ndarray | None = None,
        rs_std: np.ndarray | None = None,
    ):
        self.indices = indices
        self.player_abilities = player_abilities
        self.player_uncertainties = player_uncertainties
        self.player_archetypes = player_archetypes
        self.player_masks = player_masks
        self.player_indices = player_indices
        self.team_features = team_features
        self.game_context = game_context
        self.roster_summaries = roster_summaries
        self.targets_margin = targets_margin
        self.targets_win = targets_win
        self.coach_indices = coach_indices
        self.coach_games = coach_games
        # Normalization
        self.l3_mean = l3_mean
        self.l3_std = l3_std
        self.l4_mean = l4_mean
        self.l4_std = l4_std
        self.rs_mean = rs_mean
        self.rs_std = rs_std

    def __len__(self) -> int:
        return len(self.indices)

    def _normalize(
        self, x: np.ndarray, mean: np.ndarray | None, std: np.ndarray | None
    ) -> np.ndarray:
        if mean is not None and std is not None:
            return (x - mean) / std
        return x

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        i = self.indices[idx]

        # Per-player L1 vectors (frozen, for live L2 forward)
        home_abilities = self.player_abilities[i, 0]  # (A, 32)
        away_abilities = self.player_abilities[i, 1]
        home_uncertainties = self.player_uncertainties[i, 0]  # (A, 32)
        away_uncertainties = self.player_uncertainties[i, 1]
        home_archetypes = self.player_archetypes[i, 0]  # (A, 10)
        away_archetypes = self.player_archetypes[i, 1]
        home_mask = self.player_masks[i, 0]  # (A,)
        away_mask = self.player_masks[i, 1]
        home_pidx = self.player_indices[i, 0]  # (A,)
        away_pidx = self.player_indices[i, 1]

        # Team features (normalized)
        home_tf = self._normalize(self.team_features[i, 0], self.l3_mean, self.l3_std)
        away_tf = self._normalize(self.team_features[i, 1], self.l3_mean, self.l3_std)

        # Roster summaries (normalized)
        home_rs = self._normalize(
            self.roster_summaries[i, 0], self.rs_mean, self.rs_std
        )
        away_rs = self._normalize(
            self.roster_summaries[i, 1], self.rs_mean, self.rs_std
        )

        # Game context (normalized)
        gc = self._normalize(self.game_context[i], self.l4_mean, self.l4_std)

        # Coach indices and experience
        if self.coach_indices is not None:
            home_coach_idx = int(self.coach_indices[i, 0])
            away_coach_idx = int(self.coach_indices[i, 1])
        else:
            home_coach_idx = 0
            away_coach_idx = 0

        if self.coach_games is not None:
            home_coach_games = float(self.coach_games[i, 0])
            away_coach_games = float(self.coach_games[i, 1])
        else:
            home_coach_games = 0.0
            away_coach_games = 0.0

        # Roster continuity (raw)
        home_continuity = self.team_features[i, 0, 32]
        away_continuity = self.team_features[i, 1, 32]

        return {
            # L2 inputs (per-player L1 vectors)
            "home_abilities": torch.tensor(home_abilities, dtype=torch.float32),
            "away_abilities": torch.tensor(away_abilities, dtype=torch.float32),
            "home_uncertainties": torch.tensor(home_uncertainties, dtype=torch.float32),
            "away_uncertainties": torch.tensor(away_uncertainties, dtype=torch.float32),
            "home_archetypes": torch.tensor(home_archetypes, dtype=torch.float32),
            "away_archetypes": torch.tensor(away_archetypes, dtype=torch.float32),
            "home_player_mask": torch.tensor(home_mask, dtype=torch.bool),
            "away_player_mask": torch.tensor(away_mask, dtype=torch.bool),
            "home_player_idx": torch.tensor(home_pidx, dtype=torch.long),
            "away_player_idx": torch.tensor(away_pidx, dtype=torch.long),
            # L3 inputs
            "home_team_features": torch.tensor(home_tf, dtype=torch.float32),
            "away_team_features": torch.tensor(away_tf, dtype=torch.float32),
            "home_roster_summary": torch.tensor(home_rs, dtype=torch.float32),
            "away_roster_summary": torch.tensor(away_rs, dtype=torch.float32),
            "home_coach_idx": torch.tensor(home_coach_idx, dtype=torch.long),
            "away_coach_idx": torch.tensor(away_coach_idx, dtype=torch.long),
            "home_coach_games": torch.tensor(home_coach_games, dtype=torch.float32),
            "away_coach_games": torch.tensor(away_coach_games, dtype=torch.float32),
            "home_continuity": torch.tensor([home_continuity], dtype=torch.float32),
            "away_continuity": torch.tensor([away_continuity], dtype=torch.float32),
            # L4 inputs
            "game_context": torch.tensor(gc, dtype=torch.float32),
            # Targets
            "actual_margin": torch.tensor(self.targets_margin[i], dtype=torch.float32),
            "actual_home_win": torch.tensor(self.targets_win[i], dtype=torch.float32),
        }


# ---------------------------------------------------------------------------
# Data loading: the Phase5 game cache plus per-player L1 states
# ---------------------------------------------------------------------------


def build_phase_c_data(cache_dir: Path = GAMES_CACHE) -> dict:
    """Phase C inputs: the game cache plus each roster player's L1 state before
    the game (L2 runs live on these during training, as in predictions)."""
    data = load_phase_b_data(cache_dir)
    rosters = np.load(cache_dir / "rosters.npy")
    dates = np.load(cache_dir / "dates.npy", allow_pickle=True)
    player_to_idx = {
        int(k): v
        for k, v in json.loads((cache_dir / "player_to_idx.json").read_text()).items()
    }
    with get_db(DB_PATH) as conn:
        store = L1Store(load_game_dates(conn))
    n, _, max_roster = rosters.shape
    arrays = store.roster_arrays(
        list(rosters.reshape(n * 2, max_roster)),
        [d for d in dates for _ in (0, 1)],
        player_to_idx,
        max_roster,
    )
    for key, name in (
        ("abilities", "player_abilities"),
        ("uncertainties", "player_uncertainties"),
        ("archetypes", "player_archetypes"),
        ("mask", "player_masks"),
        ("player_idx", "player_indices"),
    ):
        data[name] = arrays[key].reshape(n, 2, *arrays[key].shape[1:])
    data["player_to_idx"] = player_to_idx
    data["coach_to_idx"] = json.loads((cache_dir / "coach_mapping.json").read_text())[
        "coach_to_idx"
    ]
    data["n_players_for_embed"] = len(player_to_idx) + 1
    logger.info(f"Phase C data: {n} games, rosters up to {max_roster} players")
    return data


def make_train_val_datasets(
    data: dict,
    train_seasons: tuple[str, str] = TRAIN_SEASONS,
    val_season: str = VAL_SEASON,
) -> tuple[PhaseCDataset, PhaseCDataset, dict]:
    """Phase C train/val datasets on Phase B's split and normalization
    (split_and_normalize). Returns the datasets and the stats as saved in the
    checkpoint."""
    train_indices, val_indices, stats = split_and_normalize(
        data, train_seasons, val_season
    )
    common_kwargs = dict(
        player_abilities=data["player_abilities"],
        player_uncertainties=data["player_uncertainties"],
        player_archetypes=data["player_archetypes"],
        player_masks=data["player_masks"],
        player_indices=data["player_indices"],
        team_features=data["team_features"],
        game_context=data["game_context"],
        roster_summaries=data["roster_summaries"],
        targets_margin=data["targets_margin"],
        targets_win=data["targets_win"],
        coach_indices=data.get("coach_indices"),
        coach_games=data.get("coach_games"),
        **stats,
    )

    train_ds = PhaseCDataset(indices=train_indices, **common_kwargs)
    val_ds = PhaseCDataset(indices=val_indices, **common_kwargs)
    norm = {
        "l3_mean": stats["l3_mean"].tolist(),
        "l3_std": stats["l3_std"].tolist(),
        "l4_mean": stats["l4_mean"].tolist(),
        "l4_std": stats["l4_std"].tolist(),
        "roster_summary_mean": stats["rs_mean"].tolist(),
        "roster_summary_std": stats["rs_std"].tolist(),
    }
    return train_ds, val_ds, norm


# ---------------------------------------------------------------------------
# Gradual unfreezing
# ---------------------------------------------------------------------------


# L2 layers that build the team vector. L2 pretraining fits pairwise synergies
# and never trains these, so they are tuned with L3 (same schedule and LR).
L2_TEAM_LAYERS = ("gat", "player_pool", "synergy_agg")


def l2_param_groups(l2_model: PlayerSynergyNetwork) -> tuple[list, list]:
    """(pretrained L2 params, team-vector layer params)."""
    pretrained, team = [], []
    for name, p in l2_model.named_parameters():
        (team if name.split(".")[0] in L2_TEAM_LAYERS else pretrained).append(p)
    return pretrained, team


def apply_freeze_schedule(
    epoch: int,
    l2_model: PlayerSynergyNetwork,
    l3_model: TeamModel,
    l4_model: GamePredictor,
    l4_only_epochs: int = 3,
    l3l4_epochs: int = 6,
) -> str:
    """Gradual unfreezing: L4, then L3 with L2's team-vector layers, then all
    of L2. Returns a description of what is trainable."""
    l2_pretrained, l2_team = l2_param_groups(l2_model)
    stage = 1 if epoch <= l4_only_epochs else 2 if epoch <= l3l4_epochs else 3
    for params, trainable in (
        (l4_model.parameters(), True),
        (l3_model.parameters(), stage >= 2),
        (l2_team, stage >= 2),
        (l2_pretrained, stage >= 3),
    ):
        for p in params:
            p.requires_grad = trainable
    return ("L4 only", "L3+L4 (+L2 team layers)", "L2+L3+L4")[stage - 1]


# ---------------------------------------------------------------------------
# L2 drift monitoring
# ---------------------------------------------------------------------------


class DriftMonitor:
    """Tracks how much L2 weights have drifted from their initial values.

    Computes relative L2 norm drift: ||theta - theta_0|| / ||theta_0||
    for key parameter groups. Large drift suggests catastrophic forgetting.
    """

    def __init__(self, model: nn.Module, name: str = "L2"):
        self.name = name
        self.initial_params = {}
        for pname, param in model.named_parameters():
            self.initial_params[pname] = param.data.clone().detach()

    def compute_drift(self, model: nn.Module) -> dict[str, float]:
        """Compute relative drift for each parameter group."""
        drifts = {}
        total_drift_num = 0.0
        total_drift_den = 0.0

        for pname, param in model.named_parameters():
            if pname in self.initial_params:
                init = self.initial_params[pname]
                diff = (param.data - init).norm().item()
                init_norm = init.norm().item()
                if init_norm > 1e-8:
                    total_drift_num += diff**2
                    total_drift_den += init_norm**2

        overall = (total_drift_num / max(total_drift_den, 1e-8)) ** 0.5
        drifts["overall"] = overall
        return drifts


# ---------------------------------------------------------------------------
# Forward pass: L2 -> L3 -> L4
# ---------------------------------------------------------------------------


def forward_l2_l3_l4(
    batch: dict[str, torch.Tensor],
    l2_model: PlayerSynergyNetwork,
    l3_model: TeamModel,
    l4_model: GamePredictor,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Run L2 (live) -> L3 -> L4 forward pass.

    L1 vectors are frozen inputs. L2 forward is live (gradients flow through).
    """
    # Move to device
    home_ab = batch["home_abilities"].to(device)
    away_ab = batch["away_abilities"].to(device)
    home_unc = batch["home_uncertainties"].to(device)
    away_unc = batch["away_uncertainties"].to(device)
    home_arch = batch["home_archetypes"].to(device)
    away_arch = batch["away_archetypes"].to(device)
    home_mask = batch["home_player_mask"].to(device)
    away_mask = batch["away_player_mask"].to(device)
    home_pidx = batch["home_player_idx"].to(device)
    away_pidx = batch["away_player_idx"].to(device)

    home_tf = batch["home_team_features"].to(device)
    away_tf = batch["away_team_features"].to(device)
    home_rs = batch["home_roster_summary"].to(device)
    away_rs = batch["away_roster_summary"].to(device)
    home_coach = batch["home_coach_idx"].to(device)
    away_coach = batch["away_coach_idx"].to(device)
    home_coach_games = batch["home_coach_games"].to(device)
    away_coach_games = batch["away_coach_games"].to(device)
    home_cont = batch["home_continuity"].to(device)
    away_cont = batch["away_continuity"].to(device)
    game_ctx = batch["game_context"].to(device)

    # --- L2 forward (live, unfrozen when scheduled) ---
    home_l2_out = l2_model(
        ability=home_ab,
        uncertainty=home_unc,
        archetypes=home_arch,
        mask=home_mask,
        player_idx=home_pidx,
    )
    away_l2_out = l2_model(
        ability=away_ab,
        uncertainty=away_unc,
        archetypes=away_arch,
        mask=away_mask,
        player_idx=away_pidx,
    )

    home_l2 = home_l2_out["team_vector"]  # (B, 134)
    away_l2 = away_l2_out["team_vector"]  # (B, 134)

    # --- L3 forward ---
    home_repr = l3_model(
        l2_team=home_l2,
        team_features=home_tf,
        roster_summary=home_rs,
        coach_idx=home_coach,
        roster_continuity=home_cont,
        coach_games=home_coach_games,
    )  # (B, 128)

    away_repr = l3_model(
        l2_team=away_l2,
        team_features=away_tf,
        roster_summary=away_rs,
        coach_idx=away_coach,
        roster_continuity=away_cont,
        coach_games=away_coach_games,
    )  # (B, 128)

    # --- L4 forward ---
    predictions = l4_model(
        team_home=home_repr,
        team_away=away_repr,
        l2_home=home_l2,
        l2_away=away_l2,
        context=game_ctx,
    )

    return predictions


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train_epoch(
    l2_model: PlayerSynergyNetwork,
    l3_model: TeamModel,
    l4_model: GamePredictor,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    l4_cfg: L4Config,
    device: torch.device,
    grad_clip: float = 1.0,
) -> dict[str, float]:
    """Train one epoch. Returns dict of mean loss components."""
    l2_model.train()
    l3_model.train()
    l4_model.train()

    total_losses = {}
    n_batches = 0

    for batch in loader:
        actual_margin = batch["actual_margin"].to(device)
        actual_win = batch["actual_home_win"].to(device)

        # Forward
        predictions = forward_l2_l3_l4(batch, l2_model, l3_model, l4_model, device)

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

        # Clip gradients for all trainable params
        all_params = (
            list(l2_model.parameters())
            + list(l3_model.parameters())
            + list(l4_model.parameters())
        )
        trainable_params = [p for p in all_params if p.requires_grad]
        if trainable_params:
            torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)

        optimizer.step()

        for k, v in losses.items():
            if k not in total_losses:
                total_losses[k] = 0.0
            total_losses[k] += v.item()
        n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in total_losses.items()}


@torch.no_grad()
def evaluate(
    l2_model: PlayerSynergyNetwork,
    l3_model: TeamModel,
    l4_model: GamePredictor,
    loader: DataLoader,
    l4_cfg: L4Config,
    device: torch.device,
) -> tuple[dict[str, float], dict[str, float]]:
    """Evaluate on a dataset. Returns (loss_dict, metrics_dict)."""
    l2_model.eval()
    l3_model.eval()
    l4_model.eval()

    total_losses = {}
    n_batches = 0

    all_preds = {"spread_mu": [], "spread_sigma": [], "win_prob": []}
    all_targets = {"margin": [], "win": []}

    for batch in loader:
        actual_margin = batch["actual_margin"].to(device)
        actual_win = batch["actual_home_win"].to(device)

        predictions = forward_l2_l3_l4(batch, l2_model, l3_model, l4_model, device)

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
    parser = argparse.ArgumentParser(description="Phase C: Fine-tune L2+L3+L4")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--l2-lr", type=float, default=3e-5)
    parser.add_argument("--l3-lr", type=float, default=1e-4)
    parser.add_argument("--l4-lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument(
        "--l4-only-epochs",
        type=int,
        default=3,
        help="Epochs with only L4 trainable",
    )
    parser.add_argument(
        "--l3l4-epochs",
        type=int,
        default=6,
        help="Epochs with L3+L4 trainable (after this, L2+L3+L4)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Quick test (5 epochs); checkpoints go to checkpoints/phase5/test",
    )
    parser.add_argument(
        "--train-seasons",
        type=str,
        default=",".join(TRAIN_SEASONS),
        help="Start,end seasons for training (inclusive)",
    )
    parser.add_argument("--val-season", type=str, default=VAL_SEASON)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--drift-threshold",
        type=float,
        default=0.5,
        help="L2 drift threshold for warning (relative norm change)",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # --- Load and build data ---
    data = build_phase_c_data()

    train_start, train_end = args.train_seasons.split(",")
    train_ds, val_ds, norm = make_train_val_datasets(
        data,
        train_seasons=(train_start, train_end),
        val_season=args.val_season,
    )

    logger.info(f"Train dataset: {len(train_ds)} games")
    logger.info(f"Val dataset: {len(val_ds)} games")

    # Everything the predictor needs besides the weights, saved with them
    inference = {
        "normalization": norm,
        "coach_to_idx": data["coach_to_idx"],
        "player_to_idx": data["player_to_idx"],
        "max_roster": int(data["player_masks"].shape[2]),
        "train_seasons": [train_start, train_end],
        "val_season": args.val_season,
    }

    if len(train_ds) == 0 or len(val_ds) == 0:
        logger.error("Empty dataset! Check season ranges.")
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
    l2_cfg = L2Config()
    l3_cfg = L3Config()
    l4_cfg = L4Config()

    # Ensure L2 embedding table is big enough
    l2_cfg.n_players = max(l2_cfg.n_players, data["n_players_for_embed"])

    l2_model = PlayerSynergyNetwork(l2_cfg).to(device)
    l3_model = TeamModel(l3_cfg).to(device)
    l4_model = GamePredictor(l4_cfg).to(device)

    # --- Load pretrained weights ---
    # L2: the checkpoint that made the cache's L2 vectors (Phase B's inputs)
    l2_path = Path(data["metadata"].get("l2_checkpoint", CHECKPOINT_DIR / "l2_best.pt"))
    l2_ckpt = torch.load(str(l2_path), map_location=device, weights_only=False)
    l2_model.load_state_dict(l2_ckpt["model_state_dict"])
    logger.info(f"Loaded L2 weights from {l2_path} (epoch {l2_ckpt.get('epoch', '?')})")

    # L3 + L4: from Phase B joint training
    b_ckpt = torch.load(
        str(CHECKPOINT_DIR / "phase_b_best.pt"),
        map_location=device,
        weights_only=False,
    )
    l3_model.load_state_dict(b_ckpt["l3_state_dict"])
    l4_model.load_state_dict(b_ckpt["l4_state_dict"])
    logger.info(
        f"Loaded L3+L4 weights from phase_b_best.pt "
        f"(epoch {b_ckpt.get('epoch', '?')}, "
        f"MAE={b_ckpt['val_metrics']['spread_mae']:.3f}, "
        f"AUC={b_ckpt['val_metrics']['win_auc']:.3f})"
    )

    l2_params = l2_count_params(l2_model)
    l3_params = l3_count_params(l3_model)
    l4_params = l4_count_params(l4_model)
    logger.info(f"L2 PlayerSynergyNetwork: {l2_params:,} params")
    logger.info(f"L3 TeamModel: {l3_params:,} params")
    logger.info(f"L4 GamePredictor: {l4_params:,} params")
    logger.info(f"Total: {l2_params + l3_params + l4_params:,} params")

    # --- Drift monitor ---
    drift_monitor = DriftMonitor(l2_model, name="L2")

    # --- Optimizer with discriminative learning rates ---
    l2_pretrained, l2_team = l2_param_groups(l2_model)
    optimizer = torch.optim.AdamW(
        [
            {"params": l2_pretrained, "lr": args.l2_lr},
            {"params": l2_team, "lr": args.l3_lr},
            {"params": l3_model.parameters(), "lr": args.l3_lr},
            {"params": l4_model.parameters(), "lr": args.l4_lr},
        ],
        weight_decay=args.weight_decay,
    )

    # Cosine annealing (applied per param group automatically)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6,
    )

    # --- Evaluate Phase B baseline (L2 frozen, L3+L4 from Phase B) ---
    logger.info("\nEvaluating Phase B baseline (before fine-tuning)...")
    baseline_losses, baseline_metrics = evaluate(
        l2_model, l3_model, l4_model, val_loader, l4_cfg, device
    )
    logger.info(
        f"Phase B baseline: MAE={baseline_metrics['spread_mae']:.3f}, "
        f"AUC={baseline_metrics['win_auc']:.3f}, "
        f"WinAcc={baseline_metrics['win_acc']:.3f}, "
        f"ECE={baseline_metrics['ece']:.4f}"
    )

    # --- Training loop ---
    out_dir = CHECKPOINT_DIR / "test" if args.test else CHECKPOINT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    def save_best(epoch, val_metrics, val_losses, l2_drift):
        torch.save(
            {
                "epoch": epoch,
                "l2_state_dict": l2_model.state_dict(),
                "l3_state_dict": l3_model.state_dict(),
                "l4_state_dict": l4_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "l2_config": l2_cfg.__dict__,
                "l3_config": l3_cfg.__dict__,
                "l4_config": l4_cfg.__dict__,
                "val_metrics": val_metrics,
                "val_losses": val_losses,
                "l2_drift": l2_drift,
                "baseline_metrics": baseline_metrics,
                "inference": inference,
            },
            str(out_dir / "phase_c_best.pt"),
        )

    # The Phase B starting point is epoch 0: fine-tuning must beat it on val,
    # or the Phase B weights are what gets promoted
    best_val_mae = baseline_metrics["spread_mae"]
    best_val_loss = baseline_losses["total_loss"]
    best_epoch = 0
    patience_counter = 0
    save_best(0, baseline_metrics, baseline_losses, 0.0)
    history = []

    logger.info(f"\nStarting Phase C training for {args.epochs} epochs...")
    logger.info(
        f"  L2 lr={args.l2_lr}, L3 lr={args.l3_lr}, L4 lr={args.l4_lr}, "
        f"wd={args.weight_decay}, batch={args.batch_size}, "
        f"patience={args.patience}"
    )
    logger.info(
        f"  Gradual unfreezing: "
        f"L4 only (ep 1-{args.l4_only_epochs}), "
        f"L3+L4 (ep {args.l4_only_epochs+1}-{args.l3l4_epochs}), "
        f"L2+L3+L4 (ep {args.l3l4_epochs+1}+)"
    )

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Apply freeze schedule
        schedule_name = apply_freeze_schedule(
            epoch,
            l2_model,
            l3_model,
            l4_model,
            l4_only_epochs=args.l4_only_epochs,
            l3l4_epochs=args.l3l4_epochs,
        )

        # Count currently trainable params
        n_trainable = sum(
            p.numel()
            for m in [l2_model, l3_model, l4_model]
            for p in m.parameters()
            if p.requires_grad
        )

        # Train
        train_losses = train_epoch(
            l2_model,
            l3_model,
            l4_model,
            train_loader,
            optimizer,
            l4_cfg,
            device,
            args.grad_clip,
        )

        # Step scheduler
        scheduler.step()

        # Validate
        val_losses, val_metrics = evaluate(
            l2_model,
            l3_model,
            l4_model,
            val_loader,
            l4_cfg,
            device,
        )

        # L2 drift monitoring
        drift = drift_monitor.compute_drift(l2_model)
        drift_str = f"drift={drift['overall']:.4f}"
        if drift["overall"] > args.drift_threshold:
            drift_str += " [WARNING: HIGH DRIFT]"

        elapsed = time.time() - t0

        lr_l2 = optimizer.param_groups[0]["lr"]
        lr_l3 = optimizer.param_groups[2]["lr"]
        lr_l4 = optimizer.param_groups[3]["lr"]

        logger.info(
            f"Epoch {epoch:3d}/{args.epochs} ({elapsed:.1f}s) [{schedule_name}] "
            f"({n_trainable:,} params) | "
            f"Train loss={train_losses['total_loss']:.4f} "
            f"(spr={train_losses['spread']:.4f}, win={train_losses['win']:.4f}) | "
            f"Val MAE={val_metrics['spread_mae']:.2f}, "
            f"AUC={val_metrics['win_auc']:.3f}, "
            f"WinAcc={val_metrics['win_acc']:.3f}, "
            f"ECE={val_metrics['ece']:.4f}, "
            f"sigma={val_metrics['mean_sigma']:.2f} | "
            f"lr={lr_l2:.1e}/{lr_l3:.1e}/{lr_l4:.1e} | "
            f"L2 {drift_str}"
        )

        # Track history
        epoch_data = {
            "epoch": epoch,
            "schedule": schedule_name,
            "n_trainable": n_trainable,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "val_metrics": val_metrics,
            "lr_l2": lr_l2,
            "lr_l3": lr_l3,
            "lr_l4": lr_l4,
            "l2_drift": drift["overall"],
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

            save_best(epoch, val_metrics, val_losses, drift["overall"])
            logger.info(f"  ** New best: MAE={best_val_mae:.3f} at epoch {epoch}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info(
                    f"Early stopping at epoch {epoch} (patience={args.patience})"
                )
                break

        if args.test and epoch >= 5:
            logger.info("Test mode: stopping after 5 epochs.")
            break

    # --- Final summary ---
    logger.info(f"\n{'='*70}")
    logger.info("Phase C Training complete!")
    logger.info(f"  Best epoch: {best_epoch}")
    logger.info(f"  Best val spread MAE: {best_val_mae:.3f}")
    logger.info(f"  Best val loss: {best_val_loss:.4f}")

    # Load best checkpoint and evaluate
    best_ckpt = torch.load(
        str(out_dir / "phase_c_best.pt"),
        map_location=device,
        weights_only=False,
    )
    l2_model.load_state_dict(best_ckpt["l2_state_dict"])
    l3_model.load_state_dict(best_ckpt["l3_state_dict"])
    l4_model.load_state_dict(best_ckpt["l4_state_dict"])

    _, final_metrics = evaluate(
        l2_model, l3_model, l4_model, val_loader, l4_cfg, device
    )

    logger.info(
        f"\nFinal validation metrics (best checkpoint, epoch {best_ckpt['epoch']}):"
    )
    for k, v in final_metrics.items():
        logger.info(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # Compare to Phase B baseline
    logger.info("\nComparison to Phase B baseline:")
    for key in ["spread_mae", "win_auc", "win_acc", "ece"]:
        b_val = baseline_metrics[key]
        c_val = final_metrics[key]
        delta = c_val - b_val
        # Lower is better for MAE and ECE; unchanged when Phase B was kept
        gain = -delta if key in ("spread_mae", "ece") else delta
        direction = "better" if gain > 0 else "worse" if gain < 0 else "unchanged"
        logger.info(f"  {key}: {b_val:.4f} -> {c_val:.4f} ({delta:+.4f}, {direction})")

    logger.info(f"  L2 drift at best: {best_ckpt.get('l2_drift', 'N/A')}")

    # Save results
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
        "l2_params": l2_params,
        "l3_params": l3_params,
        "l4_params": l4_params,
        "best_epoch": best_epoch,
        "best_val_mae": best_val_mae,
        "baseline_metrics": baseline_metrics,
        "final_metrics": final_metrics,
        "history": make_serializable(history),
    }
    results_path = RESULTS_DIR / "phase_c_results.json"
    with open(str(results_path), "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {results_path}")

    # Save final checkpoint
    torch.save(
        {
            "epoch": epoch,
            "l2_state_dict": l2_model.state_dict(),
            "l3_state_dict": l3_model.state_dict(),
            "l4_state_dict": l4_model.state_dict(),
            "l2_config": l2_cfg.__dict__,
            "l3_config": l3_cfg.__dict__,
            "l4_config": l4_cfg.__dict__,
            "val_metrics": final_metrics,
            "inference": inference,
        },
        str(out_dir / "phase_c_final.pt"),
    )
    logger.info(f"Final checkpoint saved to {out_dir / 'phase_c_final.pt'}")


if __name__ == "__main__":
    main()
