#!/usr/bin/env python3
"""
Extract per-player-per-game L1 ability vectors from the trained NKE-H model.

Runs the Kalman filter over each player's full career to produce:
  - ability[t]: (32,) ability vector at each game t
  - uncertainty[t]: (32,) P diagonal at each game t
  - archetype_weights: (10,) soft archetype assignment

Output: data/l2_cache/l1_vectors/{person_id}.npz

Resume-capable: skips players whose output file already exists.

Usage:
    python scripts/extract_l1_vectors.py [--force]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.phase5.config import NKEHConfig
from src.phase5.dataset import CACHE_DIR, Normalizer, load_metadata, load_profiles
from src.phase5.l1_extract import MAX_LEN, extract_player
from src.phase5.model import NKEH

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CKPT_PATH = PROJECT_ROOT / "checkpoints" / "phase5" / "phase2_best.pt"
OUTPUT_DIR = PROJECT_ROOT / "data" / "l2_cache" / "l1_vectors"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def load_model(metadata: dict) -> NKEH:
    """Load NKE-H model from phase2 checkpoint."""
    cfg = NKEHConfig(
        n_box_stats=len(metadata["box_stat_columns"]),
        n_pbp_stats=len(metadata["pbp_stat_columns"]),
        n_context=len(metadata["context_columns"]),
        n_profile=len(metadata["profile_columns"]),
    )
    model = NKEH(cfg)

    if not CKPT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}")

    logger.info(f"Loading checkpoint: {CKPT_PATH}")
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.to(DEVICE)
    model.eval()

    epoch = ckpt.get("epoch", "?")
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model loaded (epoch {epoch}, {n_params:,} params) on {DEVICE}")
    return model


def main():
    parser = argparse.ArgumentParser(description="Extract L1 ability vectors for L2")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Override L1 checkpoint path (default: checkpoints/phase5/phase2_best.pt)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory (default: data/l2_cache/l1_vectors/)",
    )
    args = parser.parse_args()

    # Allow overriding globals
    global CKPT_PATH, OUTPUT_DIR
    if args.checkpoint:
        CKPT_PATH = Path(args.checkpoint)
    if args.output_dir:
        OUTPUT_DIR = Path(args.output_dir)

    # Setup
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metadata = load_metadata()
    normalizer = Normalizer(metadata)
    profiles_data = load_profiles()

    # Build profile lookup
    pids_array = profiles_data["person_ids"]
    profile_idx = {int(pid): i for i, pid in enumerate(pids_array)}
    profile_columns = metadata["profile_columns"]

    # Load model
    model = load_model(metadata)

    # Discover all player cache files
    player_dir = CACHE_DIR / "players"
    player_files = sorted(player_dir.glob("*.npz"))
    logger.info(f"Found {len(player_files)} player cache files")

    # Filter to players with profiles
    valid_players = []
    for pf in player_files:
        pid = int(pf.stem)
        if pid in profile_idx:
            valid_players.append((pid, pf))
        else:
            pass  # No profile, skip silently

    logger.info(f"{len(valid_players)} players have profiles")

    # Process players
    processed = 0
    skipped = 0
    errors = 0
    total_games = 0
    t0 = time.time()

    for i, (pid, cache_path) in enumerate(valid_players):
        out_path = OUTPUT_DIR / f"{pid}.npz"

        # Resume: skip if already processed
        if out_path.exists() and not args.force:
            skipped += 1
            continue

        try:
            data = np.load(cache_path, allow_pickle=True)
            player_data = {
                "game_ids": list(data["game_ids"]),
                "box_stats": data["box_stats"],
                "pbp_stats": data["pbp_stats"],
                "context": data["context"],
                "has_pbp": data.get(
                    "has_pbp", np.ones(len(data["game_ids"]), dtype=bool)
                ),
            }
            game_ids = data["game_ids"]
            N = len(game_ids)

            if N < 2:
                continue

            # Get profile vector
            idx = profile_idx[pid]
            vals = [float(profiles_data[col][idx]) for col in profile_columns]
            profile_vec = torch.tensor(vals, dtype=torch.float32)

            # Extract vectors
            result = extract_player(player_data, model, normalizer, profile_vec, DEVICE)

            # Save
            np.savez_compressed(
                out_path,
                game_ids=game_ids,
                ability=result["ability"],
                uncertainty=result["uncertainty"],
                archetype_weights=result["archetype_weights"],
            )

            processed += 1
            total_games += N

        except Exception as e:
            logger.error(f"Error processing player {pid}: {e}")
            errors += 1

        # Progress logging
        done = processed + skipped + errors
        if done % 100 == 0 and done > 0:
            elapsed = time.time() - t0
            rate = processed / elapsed if elapsed > 0 else 0
            logger.info(
                f"Progress: {done}/{len(valid_players)} "
                f"({processed} processed, {skipped} skipped, {errors} errors) "
                f"[{rate:.1f} players/sec, {total_games:,} total games]"
            )

    elapsed = time.time() - t0
    logger.info(
        f"Done: {processed} processed, {skipped} skipped, {errors} errors "
        f"in {elapsed:.1f}s ({total_games:,} total games)"
    )

    # Save summary metadata
    summary = {
        "n_players": processed + skipped,
        "n_processed": processed,
        "n_skipped": skipped,
        "n_errors": errors,
        "total_games": total_games,
        "d_ability": 32,
        "n_archetypes": 10,
        "max_len": MAX_LEN,
        "checkpoint": str(CKPT_PATH),
        "device": str(DEVICE),
    }
    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved to {OUTPUT_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
