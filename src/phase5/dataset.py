"""
PyTorch datasets for NKE-H training.

Two dataset modes:
1. SingleGameDataset — Phase 1 (hierarchy pre-training): random player-game samples
2. CareerSequenceDataset — Phase 2 (sequential Kalman training): player career sequences
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

# Age input to the Kalman transition (aging drift), in years from a typical prime
AGE_CENTER, AGE_SCALE = 27.0, 5.0


def scaled_age(context: torch.Tensor) -> torch.Tensor:
    """(T, 1) aging input from raw context column 0 (age at the game, years)."""
    return (context[:, 0:1] - AGE_CENTER) / AGE_SCALE

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "phase5_cache"


def load_metadata() -> dict:
    """Load cache metadata including normalization stats and feature names."""
    path = CACHE_DIR / "metadata.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Cache metadata not found at {path}. Run cache_builder first."
        )
    with open(path) as f:
        return json.load(f)


def load_profiles() -> dict[str, np.ndarray]:
    """Load all player profiles."""
    path = CACHE_DIR / "profiles.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"Profiles not found at {path}. Run cache_builder first."
        )
    return dict(np.load(path))


def load_archetypes() -> np.ndarray:
    """Load archetype centroids."""
    path = CACHE_DIR / "archetypes.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"Archetypes not found at {path}. Run cache_builder first."
        )
    return np.load(path)["centroids"]


def _get_split_bounds(splits: dict, split_name: str) -> tuple[int, int]:
    """Get (min_year, max_year) for a data split.

    Split values are dicts with 'min_year' and 'max_year' keys.
    Years refer to the start year of the season (matching game_id encoding,
    e.g., 2017 = the 2017-18 season, game_id code '17').
    """
    s = splits.get(split_name, {})
    if isinstance(s, dict):
        return s.get("min_year", 2001), s.get("max_year", 2030)
    # Legacy string format fallback
    return 2001, 2030


class Normalizer:
    """Applies z-score normalization using pre-computed stats from metadata."""

    def __init__(self, metadata: dict):
        self.box_mean = torch.tensor(metadata["box_mean"], dtype=torch.float32)
        self.box_std = torch.tensor(metadata["box_std"], dtype=torch.float32)
        self.ctx_mean = torch.tensor(metadata["context_mean"], dtype=torch.float32)
        self.ctx_std = torch.tensor(metadata["context_std"], dtype=torch.float32)

        if "pbp_mean" in metadata:
            self.pbp_mean = torch.tensor(metadata["pbp_mean"], dtype=torch.float32)
            self.pbp_std = torch.tensor(metadata["pbp_std"], dtype=torch.float32)
        else:
            self.pbp_mean = None
            self.pbp_std = None

        # Profile normalization: centering + scaling so all features are roughly in [-1, 1]
        # Format: (value - offset) / scale
        self.profile_offsets = torch.tensor(
            [78.0, 215.0, 30.0, 0.0, 2010.0, 80.0, 0.0, 0.0, 0.0],
            dtype=torch.float32,
        )
        self.profile_scales = torch.tensor(
            [
                6.0,  # height_inches: ~72-84, centered at 78, range ±6
                40.0,  # weight: ~175-300, centered at 215, range ±40
                30.0,  # draft_pick: 1-60, centered at 30, range ±30
                1.0,  # undrafted: already 0/1
                12.0,  # birth_year: actually rookie year ~1998-2025, centered at 2010
                6.0,  # wingspan_inches: ~74-90, centered at 80, range ±6
                1.0,  # pos_g: already 0/1
                1.0,  # pos_f: already 0/1
                1.0,  # pos_c: already 0/1
            ],
            dtype=torch.float32,
        )

    def normalize_box(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        return (x - self.box_mean.to(device)) / self.box_std.to(device)

    def normalize_pbp(self, x: torch.Tensor) -> torch.Tensor:
        if self.pbp_mean is None:
            return x
        device = x.device
        return (x - self.pbp_mean.to(device)) / self.pbp_std.to(device)

    def normalize_context(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        return (x - self.ctx_mean.to(device)) / self.ctx_std.to(device)

    def normalize_profile(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        return (x - self.profile_offsets.to(device)) / self.profile_scales.to(device)


# ---------------------------------------------------------------------------
# Phase 1: Single-game dataset
# ---------------------------------------------------------------------------


class SingleGameDataset(Dataset):
    """
    Random access dataset of individual player-game observations.
    Used for Phase 1 hierarchy pre-training.

    Each sample is one player's data for one game, including:
    - box_stats, pbp_stats, context (inputs)
    - profile (static player features)
    - stat_target (current game stats for reconstruction)
    - next_game_target (next game stats for prediction)
    - dpm_target (DARKO DPM for impact prediction)
    - has_dpm (whether DPM target is available)
    - has_next (whether next-game target is available)
    """

    def __init__(
        self,
        player_ids: list[int],
        metadata: dict,
        split: str = "pretrain",
    ):
        """
        Args:
            player_ids: list of player IDs to include
            metadata: cache metadata dict
            split: data split for season filtering
        """
        self.metadata = metadata
        self.normalizer = Normalizer(metadata)
        self.profiles_data = load_profiles()

        # Build profile lookup: person_id → index
        pids = self.profiles_data["person_ids"]
        self.profile_idx = {int(pid): i for i, pid in enumerate(pids)}

        # Determine season bounds for this split
        splits = metadata["data_splits"]
        split_map = {
            "pretrain": "l1_pretrain",
            "pretrain_val": "l1_pretrain_val",
            "train": "main_train",
            "val": "val",
            "test": "test",
        }
        split_key = split_map.get(split, "l1_pretrain")
        self.min_season_year, self.max_season_year = _get_split_bounds(
            splits, split_key
        )

        # Build index: list of (player_id, game_index) tuples
        self.samples = []
        self.player_data = {}

        for pid in player_ids:
            cache_path = CACHE_DIR / "players" / f"{pid}.npz"
            if not cache_path.exists():
                continue
            if pid not in self.profile_idx:
                continue

            data = np.load(cache_path, allow_pickle=True)
            game_ids = data["game_ids"]

            # Filter games by season
            valid_indices = []
            for i, gid in enumerate(game_ids):
                # Extract season year from game_id prefix
                # game_id format: 00XXYYYYYY where XX encodes season
                try:
                    season_code = int(str(gid)[3:5])
                    season_year = 2000 + season_code
                except (ValueError, IndexError):
                    continue

                if self.min_season_year <= season_year <= self.max_season_year:
                    valid_indices.append(i)

            if len(valid_indices) < 2:
                continue

            self.player_data[pid] = {
                "box_stats": data["box_stats"],
                "pbp_stats": data["pbp_stats"],
                "context": data["context"],
                "dpm_targets": data["dpm_targets"],
                "rapm_targets": data["rapm_targets"],
                "has_dpm": data["has_dpm"],
                "has_pbp": data.get("has_pbp", np.ones(len(game_ids), dtype=bool)),
            }

            valid_set = set(valid_indices)
            for idx in valid_indices:
                # Next game must also be in valid indices (no cross-split leakage)
                has_next = (idx + 1) in valid_set
                self.samples.append((pid, idx, has_next))

        logger.info(
            f"SingleGameDataset ({split}): {len(self.samples)} samples "
            f"from {len(self.player_data)} players"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def _get_profile(self, pid: int) -> torch.Tensor:
        """Get normalized profile vector for a player."""
        idx = self.profile_idx[pid]
        profile_cols = self.metadata["profile_columns"]
        vals = [float(self.profiles_data[col][idx]) for col in profile_cols]
        return torch.tensor(vals, dtype=torch.float32)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        pid, game_idx, has_next = self.samples[index]
        data = self.player_data[pid]

        box = torch.tensor(data["box_stats"][game_idx], dtype=torch.float32)
        pbp = torch.tensor(data["pbp_stats"][game_idx], dtype=torch.float32)
        ctx = torch.tensor(data["context"][game_idx], dtype=torch.float32)
        dpm = torch.tensor(data["dpm_targets"][game_idx], dtype=torch.float32)
        has_dpm = torch.tensor(data["has_dpm"][game_idx], dtype=torch.bool)
        rapm = torch.tensor(data["rapm_targets"][game_idx], dtype=torch.float32)
        has_rapm = torch.tensor(bool(rapm.any()), dtype=torch.bool)
        profile = self._get_profile(pid)

        # Normalize inputs
        box_norm = self.normalizer.normalize_box(box)

        # Stat targets: normalized (same z-score scale as inputs — prevents
        # high-variance stats like points/minutes from dominating the loss)
        stat_target = box_norm.clone()

        # Next-game target (also normalized)
        if has_next:
            next_raw = torch.tensor(
                data["box_stats"][game_idx + 1], dtype=torch.float32
            )
            next_target = self.normalizer.normalize_box(next_raw)
            has_next_t = torch.tensor(True)
        else:
            next_target = torch.zeros_like(box)
            has_next_t = torch.tensor(False)

        # PBP availability
        has_pbp = bool(data["has_pbp"][game_idx])

        pbp_norm = (
            self.normalizer.normalize_pbp(pbp) if has_pbp else torch.zeros_like(pbp)
        )
        ctx_norm = self.normalizer.normalize_context(ctx)

        return {
            "box_stats": box_norm,
            "pbp_stats": pbp_norm,
            "context": ctx_norm,
            "profile": self.normalizer.normalize_profile(profile),
            "stat_target": stat_target,
            "next_game_target": next_target,
            "dpm_target": dpm,
            "has_dpm": has_dpm,
            "rapm_target": rapm,
            "has_rapm": has_rapm,
            "has_next": has_next_t,
        }


# ---------------------------------------------------------------------------
# Phase 2: Career sequence dataset
# ---------------------------------------------------------------------------


class CareerSequenceDataset(Dataset):
    """
    Dataset of player career sequences for Phase 2 Kalman training.

    Each sample is a player's chronological game sequence (up to max_len),
    padded with a mask indicating valid timesteps.
    """

    def __init__(
        self,
        player_ids: list[int],
        metadata: dict,
        max_len: int = 300,
        split: str = "pretrain",
    ):
        self.metadata = metadata
        self.normalizer = Normalizer(metadata)
        self.max_len = max_len
        self.profiles_data = load_profiles()

        pids = self.profiles_data["person_ids"]
        self.profile_idx = {int(pid): i for i, pid in enumerate(pids)}

        # Determine season bounds
        splits = metadata["data_splits"]
        split_map = {
            "pretrain": "l1_pretrain",
            "pretrain_val": "l1_pretrain_val",
            "train": "main_train",
            "val": "val",
            "test": "test",
        }
        split_key = split_map.get(split, "l1_pretrain")
        self.min_season_year, self.max_season_year = _get_split_bounds(
            splits, split_key
        )

        # Load player data and filter by season
        self.players = []
        self.player_cache = {}

        for pid in player_ids:
            cache_path = CACHE_DIR / "players" / f"{pid}.npz"
            if not cache_path.exists():
                continue
            if pid not in self.profile_idx:
                continue

            data = np.load(cache_path, allow_pickle=True)
            game_ids = data["game_ids"]

            # Filter games by season
            valid_mask = np.zeros(len(game_ids), dtype=bool)
            for i, gid in enumerate(game_ids):
                try:
                    season_code = int(str(gid)[3:5])
                    season_year = 2000 + season_code
                    if self.min_season_year <= season_year <= self.max_season_year:
                        valid_mask[i] = True
                except (ValueError, IndexError):
                    continue

            n_valid = valid_mask.sum()
            if n_valid < 10:  # need at least 10 games for meaningful sequence
                continue

            # Store filtered data
            has_pbp_arr = data.get("has_pbp", np.ones(len(game_ids), dtype=bool))
            team_ids_arr = data.get("team_ids")
            filtered_ctx = data["context"][valid_mask]
            self.player_cache[pid] = {
                "box_stats": data["box_stats"][valid_mask],
                "pbp_stats": data["pbp_stats"][valid_mask],
                "context": filtered_ctx,
                # Career-start context for initialization (game 0, not truncation start)
                "init_context": filtered_ctx[0].copy(),
                "dpm_targets": data["dpm_targets"][valid_mask],
                "rapm_targets": data["rapm_targets"][valid_mask],
                "has_dpm": data["has_dpm"][valid_mask],
                "has_pbp": has_pbp_arr[valid_mask],
                "team_ids": (
                    team_ids_arr[valid_mask]
                    if team_ids_arr is not None
                    else np.zeros(n_valid, dtype=np.int64)
                ),
            }
            self.players.append(pid)

        logger.info(
            f"CareerSequenceDataset ({split}): {len(self.players)} players, "
            f"max_len={max_len}"
        )

    def __len__(self) -> int:
        return len(self.players)

    def _get_profile(self, pid: int) -> torch.Tensor:
        idx = self.profile_idx[pid]
        profile_cols = self.metadata["profile_columns"]
        vals = [float(self.profiles_data[col][idx]) for col in profile_cols]
        return torch.tensor(vals, dtype=torch.float32)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        pid = self.players[index]
        data = self.player_cache[pid]
        T_full = len(data["box_stats"])

        # Truncate to max_len (take most recent games)
        team_ids_arr = data.get("team_ids")
        if T_full > self.max_len:
            start = T_full - self.max_len
            box = data["box_stats"][start:]
            pbp = data["pbp_stats"][start:]
            ctx = data["context"][start:]
            dpm = data["dpm_targets"][start:]
            rapm = data["rapm_targets"][start:]
            has_dpm = data["has_dpm"][start:]
            if team_ids_arr is not None:
                team_ids_arr = team_ids_arr[start:]
            T = self.max_len
        else:
            box = data["box_stats"]
            pbp = data["pbp_stats"]
            ctx = data["context"]
            dpm = data["dpm_targets"]
            rapm = data["rapm_targets"]
            has_dpm = data["has_dpm"]
            T = T_full

        # Pad to max_len
        pad_len = self.max_len - T
        if pad_len > 0:
            box = np.pad(box, ((0, pad_len), (0, 0)), mode="constant")
            pbp = np.pad(pbp, ((0, pad_len), (0, 0)), mode="constant")
            ctx = np.pad(ctx, ((0, pad_len), (0, 0)), mode="constant")
            dpm = np.pad(dpm, ((0, pad_len), (0, 0)), mode="constant")
            rapm = np.pad(rapm, ((0, pad_len), (0, 0)), mode="constant")
            has_dpm = np.pad(has_dpm, (0, pad_len), mode="constant")

        # Mask: True for valid timesteps
        mask = np.zeros(self.max_len, dtype=bool)
        mask[:T] = True

        # Convert to tensors
        box_t = torch.tensor(box, dtype=torch.float32)
        pbp_t = torch.tensor(pbp, dtype=torch.float32)
        ctx_t = torch.tensor(ctx, dtype=torch.float32)
        dpm_t = torch.tensor(dpm, dtype=torch.float32)
        rapm_t = torch.tensor(rapm, dtype=torch.float32)
        has_dpm_t = torch.tensor(has_dpm, dtype=torch.bool)
        # has_rapm: True when either off_rapm or def_rapm is non-zero
        has_rapm_t = rapm_t.any(dim=-1).bool()  # (max_len,)
        mask_t = torch.tensor(mask, dtype=torch.bool)
        profile = self._get_profile(pid)

        # Age sequence for Kalman prediction step
        age_t = scaled_age(ctx_t)  # (max_len, 1)

        # Days-since-last-game for time-scaled Kalman drift + process noise
        # Context column 8 is raw days_since_last_game (extract BEFORE normalization)
        days_gap_t = ctx_t[:, 8:9].clone()  # (max_len, 1) — raw calendar days

        # Normalize box stats first (targets use same normalization)
        box_norm = self.normalizer.normalize_box(box_t)

        # Stat targets: normalized (same z-score scale as inputs)
        stat_target = box_norm.clone()
        next_target = torch.zeros_like(box_t)
        has_next = torch.zeros(self.max_len, dtype=torch.bool)
        if T > 1:
            next_target[: T - 1] = box_norm[1:T]
            has_next[: T - 1] = True

        # Zero PBP for games without PBP data
        pbp_norm = self.normalizer.normalize_pbp(pbp_t)
        # Zero out PBP for games without PBP data (has_pbp=False)
        has_pbp_data = data.get("has_pbp")
        if has_pbp_data is not None:
            has_pbp_padded = np.zeros(self.max_len, dtype=bool)
            has_pbp_padded[:T] = has_pbp_data[:T]
            no_pbp_mask = ~torch.tensor(has_pbp_padded, dtype=torch.bool)
            pbp_norm[no_pbp_mask] = 0.0
        ctx_norm = self.normalizer.normalize_context(ctx_t)

        # Compute trade mask (True = first game on new team)
        trade_mask = np.zeros(self.max_len, dtype=bool)
        if team_ids_arr is not None and len(team_ids_arr) > 1:
            for t in range(1, T):
                if team_ids_arr[t] != team_ids_arr[t - 1] and team_ids_arr[t] != 0:
                    trade_mask[t] = True
        trade_mask_t = torch.tensor(trade_mask, dtype=torch.bool)

        # Career-start context for PopulationPrior initialization
        # (avoids using truncation-start context which has wrong career_games_played)
        init_ctx = torch.tensor(data["init_context"], dtype=torch.float32)
        init_ctx_norm = self.normalizer.normalize_context(init_ctx)

        return {
            "box_stats": box_norm,  # (max_len, n_box)
            "pbp_stats": pbp_norm,  # (max_len, n_pbp)
            "context": ctx_norm,  # (max_len, n_ctx)
            "init_context": init_ctx_norm,  # (n_ctx,) — career start context
            "profile": self.normalizer.normalize_profile(profile),  # (n_profile,)
            "age": age_t,  # (max_len, 1)
            "days_gap": days_gap_t,  # (max_len, 1) — raw days since last game
            "mask": mask_t,  # (max_len,)
            "stat_target": stat_target,  # (max_len, n_box)
            "next_game_target": next_target,  # (max_len, n_box)
            "dpm_target": dpm_t,  # (max_len, 3)
            "has_dpm": has_dpm_t,  # (max_len,)
            "rapm_target": rapm_t,  # (max_len, 2)
            "has_rapm": has_rapm_t,  # (max_len,)
            "has_next": has_next,  # (max_len,)
            "trade_mask": trade_mask_t,  # (max_len,)
            "seq_len": torch.tensor(T, dtype=torch.long),
        }
