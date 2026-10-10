#!/usr/bin/env python3
"""
Build the Phase5 game cache that L3/L4 train on (Phase B and Phase C).

Every finished regular-season and postseason game since 2001-02 is described
as of its tip-off by the same code that live predictions use:
  - TeamFeatureComputer: L3 team features, L4 game context, coach inputs
  - RosterAssembler(historical=True): pre-game rosters. Not the players who
    logged minutes, whose count tracks the final margin (blowouts empty the
    bench) and leaked the result into training.
  - L1Store: each roster player's L1 state before the game
plus targets and, for Phase B, frozen L2 team vectors and roster summaries.

Output (data/phase5_games/):
  game_ids.npy, seasons.npy, dates.npy   (N,)
  targets.npz            margin, home_win, total
  team_features.npz      features (N, 2, 34), raw
  game_context.npz       features (N, 14), raw
  rosters.npy            (N, 2, R) player ids by projected minutes, 0 = none
  coach_indices.npy      (N, 2) into coach_mapping.json, 0 = unknown
  coach_games.npy        (N, 2) career games coached before the game
  coach_mapping.json     {"coach_to_idx": {coach_id: idx}}
  l2_vectors.npz         vectors (N, 2, d_l2), frozen L2 output (Phase B)
  roster_summaries.npz   summaries (N, 2, 12)
  player_to_idx.json     L2 embedding index for each player with L1 vectors
  metadata.json

Usage:
    python scripts/build_phase5_cache.py
    python scripts/build_phase5_cache.py --test 300   # most recent 300 games
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.database import DB_PATH, get_db  # noqa: E402
from src.phase5.l1_store import L1Store, load_game_dates  # noqa: E402
from src.phase5.l2_config import L2Config  # noqa: E402
from src.phase5.l2_model import PlayerSynergyNetwork  # noqa: E402
from src.phase5.roster_summary import compute_roster_summary  # noqa: E402
from src.pipeline.roster_assembler import RosterAssembler  # noqa: E402
from src.pipeline.team_features import TeamFeatureComputer  # noqa: E402

OUTPUT_DIR = PROJECT_ROOT / "data" / "phase5_games"
L2_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "phase5" / "l2_best.pt"
FIRST_SEASON = "2001-2002"
FEATURE_CHUNK = 2000  # games per TeamFeatureComputer call

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def load_games(conn, limit: int | None) -> list[tuple]:
    """(game_id, date_time_utc, season, home_pts, away_pts) for finished games."""
    rows = conn.execute(
        """
        SELECT g.game_id, g.date_time_utc, g.season, th.pts, ta.pts
        FROM Games g
        JOIN Teams ht ON ht.abbreviation = g.home_team
        JOIN Teams at ON at.abbreviation = g.away_team
        JOIN TeamBox th ON th.game_id = g.game_id AND th.team_id = ht.team_id
        JOIN TeamBox ta ON ta.game_id = g.game_id AND ta.team_id = at.team_id
        WHERE g.status = 3 AND g.season >= ?
        AND g.season_type IN ('Regular Season', 'Post Season')
        ORDER BY g.date_time_utc
        """,
        (FIRST_SEASON,),
    ).fetchall()
    return rows[-limit:] if limit else rows


def l2_team_vectors(arrays: dict, checkpoint: Path, device) -> np.ndarray:
    """Frozen L2 team vectors for each team-game's roster arrays."""
    cfg = L2Config()
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"]
    cfg.n_players = max(cfg.n_players, state["fm_synergy.residual.weight"].shape[0])
    model = PlayerSynergyNetwork(cfg)
    model.load_state_dict(state)
    model.to(device).eval()

    out = []
    with torch.no_grad():
        for s in range(0, len(arrays["mask"]), 512):
            batch = {
                k: torch.as_tensor(arrays[k][s : s + 512], device=device)
                for k in (
                    "abilities",
                    "uncertainties",
                    "archetypes",
                    "mask",
                    "player_idx",
                )
            }
            vec = model(
                ability=batch["abilities"],
                uncertainty=batch["uncertainties"],
                archetypes=batch["archetypes"],
                mask=batch["mask"],
                player_idx=batch["player_idx"],
            )["team_vector"]
            out.append(vec.cpu().numpy())
    return np.concatenate(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--l2-checkpoint", type=Path, default=L2_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--test", type=int, help="Only the most recent N games")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_roster = L2Config().max_roster
    t0 = time.time()

    with get_db(DB_PATH) as conn:
        games = load_games(conn, args.test)
        game_dates = load_game_dates(conn)
    game_ids = [g[0] for g in games]
    logger.info(f"{len(game_ids)} games from {games[0][1]} to {games[-1][1]}")

    rosters = RosterAssembler(historical=True).get_projected_rosters(game_ids)
    logger.info(f"Rosters: {time.time() - t0:.0f}s")

    computer = TeamFeatureComputer()
    features = {}
    for s in range(0, len(game_ids), FEATURE_CHUNK):
        chunk = game_ids[s : s + FEATURE_CHUNK]
        features.update(computer.compute_for_games(chunk, rosters))
        logger.info(f"Features: {s + len(chunk)}/{len(game_ids)}")

    store = L1Store(game_dates)
    player_to_idx = {pid: i + 1 for i, pid in enumerate(store.player_ids())}
    team_rosters = [
        rosters.get(gid, {}).get(f"{side}_players", [])
        for gid in game_ids
        for side in ("home", "away")
    ]
    arrays = store.roster_arrays(
        team_rosters,
        [game_dates[gid] for gid in game_ids for _ in (0, 1)],
        player_to_idx,
        max_roster,
    )
    # A game needs at least one player with an L1 state on each side
    has_players = arrays["mask"].any(axis=1).reshape(-1, 2).all(axis=1)
    keep = np.array([gid in features for gid in game_ids]) & has_players
    logger.info(f"Usable games: {keep.sum()}/{len(game_ids)}")

    def per_game(a):
        return a.reshape(len(game_ids), 2, *a.shape[1:])[keep]

    l2 = per_game(l2_team_vectors(arrays, args.l2_checkpoint, device))
    summaries = per_game(
        np.stack(
            [
                compute_roster_summary(a, u, m)
                for a, u, m in zip(
                    arrays["abilities"], arrays["uncertainties"], arrays["mask"]
                )
            ]
        )
    )
    kept = [g for g, k in zip(games, keep) if k]
    kept_ids = [g[0] for g in kept]

    roster_ids = np.zeros((len(kept_ids), 2, max_roster), dtype=np.int64)
    for i, gid in enumerate(kept_ids):
        for side, key in enumerate(("home_players", "away_players")):
            pids = rosters[gid][key][:max_roster]
            roster_ids[i, side, : len(pids)] = pids

    coaches = [
        [features[gid][f"{side}_coach"] for side in ("home", "away")]
        for gid in kept_ids
    ]
    coach_to_idx = {
        c: i + 1 for i, c in enumerate(sorted({c for g in coaches for c, _ in g if c}))
    }

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    margin = np.array([g[3] - g[4] for g in kept], dtype=np.float32)
    np.save(out / "game_ids.npy", np.array(kept_ids, dtype=object))
    np.save(out / "seasons.npy", np.array([g[2] for g in kept], dtype=object))
    np.save(out / "dates.npy", np.array([g[1] for g in kept], dtype=object))
    np.savez(
        out / "targets.npz",
        margin=margin,
        home_win=(margin > 0).astype(np.float32),
        total=np.array([g[3] + g[4] for g in kept], dtype=np.float32),
    )
    np.savez(
        out / "team_features.npz",
        features=np.array(
            [
                [features[g]["home_team_features"], features[g]["away_team_features"]]
                for g in kept_ids
            ],
            dtype=np.float32,
        ),
    )
    np.savez(
        out / "game_context.npz",
        features=np.array(
            [features[g]["game_context"] for g in kept_ids], dtype=np.float32
        ),
    )
    np.save(out / "rosters.npy", roster_ids)
    np.save(
        out / "coach_indices.npy",
        np.array(
            [[coach_to_idx.get(c, 0) for c, _ in g] for g in coaches], dtype=np.int64
        ),
    )
    np.save(
        out / "coach_games.npy",
        np.array([[n for _, n in g] for g in coaches], dtype=np.float32),
    )
    (out / "coach_mapping.json").write_text(json.dumps({"coach_to_idx": coach_to_idx}))
    np.savez(out / "l2_vectors.npz", vectors=l2.astype(np.float32))
    np.savez(out / "roster_summaries.npz", summaries=summaries.astype(np.float32))
    (out / "player_to_idx.json").write_text(json.dumps(player_to_idx))
    (out / "metadata.json").write_text(
        json.dumps(
            {
                "n_games": len(kept_ids),
                "date_range": [kept[0][1], kept[-1][1]],
                "max_roster": max_roster,
                "l2_checkpoint": str(args.l2_checkpoint),
                "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            indent=2,
        )
    )
    logger.info(f"Saved {len(kept_ids)} games to {out} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
