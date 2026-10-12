"""
Phase B Game Dataset: loads pre-computed L2 vectors + team features + targets.

Each sample is one game with all inputs needed for L3 + L4 forward passes.
All heavy computation (L1 → L2 forward pass, feature extraction) was done
during cache building. This dataset just indexes into numpy arrays.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# Built by scripts/build_phase5_cache.py
PHASE_B_CACHE_DIR = (
    Path(__file__).resolve().parent.parent.parent / "data" / "phase5_games"
)


class PhaseBDataset(Dataset):
    """Each sample is one game with pre-computed L2 vectors + team features + targets.

    All arrays are indexed by game position (same ordering across all files).
    Normalization is applied on-the-fly using pre-computed stats from the
    training set (passed at construction time).
    """

    def __init__(
        self,
        indices: np.ndarray,
        l2_vectors: np.ndarray,  # (N, 2, 134)
        team_features: np.ndarray,  # (N, 2, 34)
        game_context: np.ndarray,  # (N, 14)
        roster_summaries: np.ndarray,  # (N, 2, 12)
        targets_margin: np.ndarray,  # (N,)
        targets_win: np.ndarray,  # (N,)
        # Coach data
        coach_indices: np.ndarray | None = None,  # (N, 2) int64
        coach_games: np.ndarray | None = None,  # (N, 2) float32
        # Normalization stats (from training set)
        l3_mean: np.ndarray | None = None,
        l3_std: np.ndarray | None = None,
        l4_mean: np.ndarray | None = None,
        l4_std: np.ndarray | None = None,
        rs_mean: np.ndarray | None = None,
        rs_std: np.ndarray | None = None,
    ):
        """
        Args:
            indices: array of integer indices into the full cache arrays
                     (allows train/val splitting without copying data)
            l2_vectors: full cache array (N_total, 2, 134)
            team_features: full cache array (N_total, 2, 34)
            game_context: full cache array (N_total, 14)
            roster_summaries: full cache array (N_total, 2, 12)
            targets_*: full cache arrays (N_total,)
            coach_indices: (N_total, 2) int64 — embedding indices (0=unknown)
            coach_games: (N_total, 2) float32 — cumulative games coached
            l3/l4/rs_mean/std: normalization parameters (from training set)
        """
        self.indices = indices
        self.l2_vectors = l2_vectors
        self.team_features = team_features
        self.game_context = game_context
        self.roster_summaries = roster_summaries
        self.targets_margin = targets_margin
        self.targets_win = targets_win
        self.coach_indices = coach_indices
        self.coach_games = coach_games

        # Normalization (None means no normalization)
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

        # L2 outputs (frozen, pre-computed) -- NO normalization
        # L2 vectors are already in a learned representation space
        home_l2 = self.l2_vectors[i, 0]  # (134,)
        away_l2 = self.l2_vectors[i, 1]  # (134,)

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

        # Roster continuity (from team_features index 32)
        # This is already in the team features but L3 needs it separately
        # for the ContinuityGate. Use raw (un-normalized) value since the
        # gate applies its own learned sigmoid transform.
        home_continuity = self.team_features[i, 0, 32]  # roster_continuity
        away_continuity = self.team_features[i, 1, 32]

        return {
            # L2 outputs (frozen)
            "home_l2": torch.tensor(home_l2, dtype=torch.float32),
            "away_l2": torch.tensor(away_l2, dtype=torch.float32),
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


def load_phase_b_data(
    cache_dir: Path | None = None,
) -> dict:
    """Load the Phase5 game cache.

    Returns dict with all numpy arrays + metadata needed for PhaseBDataset.
    Normalization stats come from the training split (make_train_val_datasets).
    """
    if cache_dir is None:
        cache_dir = PHASE_B_CACHE_DIR

    logger.info(f"Loading Phase B cache from {cache_dir}")

    l2_data = np.load(str(cache_dir / "l2_vectors.npz"))
    rs_data = np.load(str(cache_dir / "roster_summaries.npz"))
    tf_data = np.load(str(cache_dir / "team_features.npz"))
    gc_data = np.load(str(cache_dir / "game_context.npz"))
    tgt_data = np.load(str(cache_dir / "targets.npz"))
    game_ids = np.load(str(cache_dir / "game_ids.npy"), allow_pickle=True)
    seasons = np.load(str(cache_dir / "seasons.npy"), allow_pickle=True)

    # Coach data (optional — may not exist in older caches)
    coach_indices_path = cache_dir / "coach_indices.npy"
    coach_games_path = cache_dir / "coach_games.npy"
    if coach_indices_path.exists() and coach_games_path.exists():
        coach_indices = np.load(str(coach_indices_path))
        coach_games = np.load(str(coach_games_path))
        logger.info(
            f"  Loaded coach data: {coach_indices.shape}, "
            f"coverage={(coach_indices > 0).any(axis=1).mean():.1%}"
        )
    else:
        coach_indices = None
        coach_games = None
        logger.info("  No coach data found (using placeholder 0s)")

    with open(str(cache_dir / "metadata.json")) as f:
        metadata = json.load(f)

    data = {
        "l2_vectors": l2_data["vectors"],
        "roster_summaries": rs_data["summaries"],
        "team_features": tf_data["features"],
        "game_context": gc_data["features"],
        "targets_margin": tgt_data["margin"],
        "targets_win": tgt_data["home_win"],
        "game_ids": game_ids,
        "seasons": seasons,
        "metadata": metadata,
        "coach_indices": coach_indices,
        "coach_games": coach_games,
    }

    logger.info(
        f"  Loaded {len(game_ids)} games, "
        f"L2 vectors {data['l2_vectors'].shape}, "
        f"team_features {data['team_features'].shape}"
    )
    return data


def season_start_year(season: str) -> int:
    """Convert '2017-2018' to 2017."""
    return int(season.split("-")[0])


# Default split (2025-26 is the held-out test season, scripts/backtest.py)
TRAIN_SEASONS = ("2008-2009", "2023-2024")
VAL_SEASON = "2024-2025"


def split_and_normalize(
    data: dict,
    train_seasons: tuple[str, str] = TRAIN_SEASONS,
    val_season: str = VAL_SEASON,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Train/val game indices by season (train_seasons inclusive) and the
    normalization stats, from the training split only. Phase B and Phase C
    share it, so Phase C's epoch 0 (the Phase B weights) sees the same inputs.

    Returns (train_indices, val_indices, stats) with stats keys l3_mean,
    l3_std, l4_mean, l4_std, rs_mean, rs_std.
    """
    seasons = data["seasons"]
    train_start = season_start_year(train_seasons[0])
    train_end = season_start_year(train_seasons[1])
    val_year = season_start_year(val_season)

    train_indices = []
    val_indices = []

    for i, s in enumerate(seasons):
        if not s:  # empty season string
            continue
        try:
            year = season_start_year(s)
        except (ValueError, IndexError):
            continue

        if train_start <= year <= train_end:
            train_indices.append(i)
        elif year == val_year:
            val_indices.append(i)

    train_indices = np.array(train_indices, dtype=np.int64)
    val_indices = np.array(val_indices, dtype=np.int64)

    logger.info(
        f"Train: {len(train_indices)} games ({train_seasons[0]} - {train_seasons[1]})"
    )
    logger.info(f"Val: {len(val_indices)} games ({val_season})")

    def stats(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean = x.mean(axis=0).astype(np.float32)
        std = x.std(axis=0).astype(np.float32)
        std[std < 1e-8] = 1e-8
        return mean, std

    # Team features and roster summaries: home and away rows pooled
    l3_mean, l3_std = stats(data["team_features"][train_indices].reshape(-1, 34))
    l4_mean, l4_std = stats(data["game_context"][train_indices])
    rs_mean, rs_std = stats(data["roster_summaries"][train_indices].reshape(-1, 12))
    return train_indices, val_indices, {
        "l3_mean": l3_mean,
        "l3_std": l3_std,
        "l4_mean": l4_mean,
        "l4_std": l4_std,
        "rs_mean": rs_mean,
        "rs_std": rs_std,
    }


def make_train_val_datasets(
    data: dict,
    train_seasons: tuple[str, str] = TRAIN_SEASONS,
    val_season: str = VAL_SEASON,
) -> tuple[PhaseBDataset, PhaseBDataset]:
    """Phase B train/val datasets (see split_and_normalize)."""
    train_indices, val_indices, norm = split_and_normalize(
        data, train_seasons, val_season
    )
    common_kwargs = dict(
        l2_vectors=data["l2_vectors"],
        team_features=data["team_features"],
        game_context=data["game_context"],
        roster_summaries=data["roster_summaries"],
        targets_margin=data["targets_margin"],
        targets_win=data["targets_win"],
        coach_indices=data.get("coach_indices"),
        coach_games=data.get("coach_games"),
        **norm,
    )

    train_ds = PhaseBDataset(indices=train_indices, **common_kwargs)
    val_ds = PhaseBDataset(indices=val_indices, **common_kwargs)

    return train_ds, val_ds
