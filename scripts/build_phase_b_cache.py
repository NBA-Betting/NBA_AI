#!/usr/bin/env python3
"""
Phase B cache builder: pre-compute frozen L2 team vectors for every game.

Since L1 and L2 are frozen during Phase B (L3+L4 training), we pre-compute
their outputs once to avoid redundant forward passes during training.

Outputs saved to data/phase_b_cache/:
  - l2_vectors.npz: shape (N_games, 2, 134)  -- L2 team vectors [home, away]
  - roster_summaries.npz: shape (N_games, 2, 12) -- roster composition summaries
  - targets.npz: margin, home_win, total for each game
  - game_ids.npy: game_id strings for index alignment
  - travel_context.npz: updated game_context with travel features filled in
  - metadata.json: normalization stats, player mapping, etc.

Usage:
    python scripts/build_phase_b_cache.py             # Full build (~30 min)
    python scripts/build_phase_b_cache.py --test 100  # Quick test (100 games)
"""

import argparse
import json
import logging
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.phase5.l2_config import L2Config
from src.phase5.roster_summary import compute_roster_summary
from src.phase5.l2_model import PlayerSynergyNetwork
from src.phase5.travel import compute_travel_features

DB_PATH = PROJECT_ROOT / "data" / "NBA_AI_full.sqlite"
L1_VECTORS_DIR = PROJECT_ROOT / "data" / "l2_cache" / "l1_vectors"
L2_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "phase5" / "l2_best.pt"
L3L4_CACHE_DIR = PROJECT_ROOT / "data" / "l3l4_cache"
OUTPUT_DIR = PROJECT_ROOT / "data" / "phase_b_cache"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# L1 vector loading
# ---------------------------------------------------------------------------


def load_l1_vectors() -> dict[int, dict]:
    """Load all L1 ability vectors from npz files.

    Returns:
        {player_id: {ability: (N,32), uncertainty: (N,32),
                     game_ids: (N,), archetype_weights: (10,)}}
    """
    vectors = {}
    npz_files = sorted(L1_VECTORS_DIR.glob("*.npz"))
    logger.info(f"Loading L1 vectors from {len(npz_files)} files...")

    for path in npz_files:
        player_id = int(path.stem)
        data = np.load(str(path), allow_pickle=True)
        vectors[player_id] = {
            "ability": data["ability"],  # (N, 32)
            "uncertainty": data["uncertainty"],  # (N, 32)
            "game_ids": list(data["game_ids"]),  # list of str
            "archetype_weights": data["archetype_weights"],  # (10,)
        }

    logger.info(f"Loaded L1 vectors for {len(vectors)} players")
    return vectors


def build_player_to_idx(l1_vectors: dict[int, dict]) -> dict[int, int]:
    """Build player_id -> embedding index mapping. Index 0 = padding."""
    player_ids = sorted(l1_vectors.keys())
    return {pid: i + 1 for i, pid in enumerate(player_ids)}


def get_l1_at_game(
    l1_data: dict, game_id: str, use_latest: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get L1 vectors for a player at (or just before) a specific game.

    Returns: (ability (32,), uncertainty (32,), archetype_weights (10,))
    If game_id is not found, uses the most recent observation before it.
    If no prior observation exists, returns None.
    """
    game_ids = l1_data["game_ids"]
    archetype = l1_data["archetype_weights"]

    if game_id in game_ids:
        idx = game_ids.index(game_id)
        # Use the state BEFORE this game (idx-1) for no-leakage.
        # If this is the first game, use that state (it's the prior).
        if idx > 0:
            idx = idx - 1
        return l1_data["ability"][idx], l1_data["uncertainty"][idx], archetype

    # Game not in this player's history — find the most recent prior game.
    # Game IDs are chronologically ordered in the L1 cache.
    # We need to find the latest game_id that is lexicographically < game_id.
    # Since NBA game IDs are zero-padded strings, lex order = chronological order.
    best_idx = -1
    for i, gid in enumerate(game_ids):
        if gid < game_id:
            best_idx = i
        else:
            break

    if best_idx >= 0:
        return l1_data["ability"][best_idx], l1_data["uncertainty"][best_idx], archetype

    if use_latest and len(game_ids) > 0:
        return l1_data["ability"][0], l1_data["uncertainty"][0], archetype

    return None, None, None


# ---------------------------------------------------------------------------
# Roster summary computation (12 features from L1 vectors)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Travel features
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Main cache building
# ---------------------------------------------------------------------------


def build_cache(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # --- Load L1 vectors ---
    l1_vectors = load_l1_vectors()
    player_to_idx = build_player_to_idx(l1_vectors)

    # --- Load L2 model ---
    cfg = L2Config()
    # Ensure the embedding table is big enough
    cfg.n_players = max(cfg.n_players, len(player_to_idx) + 1)
    model = PlayerSynergyNetwork(cfg)
    ckpt = torch.load(str(L2_CHECKPOINT), map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    model.to(device)
    logger.info(
        f"Loaded L2 model from {L2_CHECKPOINT} (epoch {ckpt.get('epoch', '?')})"
    )

    # --- Load L3/L4 cache (for game_ids, team_features, game_context) ---
    tf_data = np.load(str(L3L4_CACHE_DIR / "team_features.npz"), allow_pickle=True)
    gc_data = np.load(str(L3L4_CACHE_DIR / "game_context.npz"), allow_pickle=True)
    with open(str(L3L4_CACHE_DIR / "metadata.json")) as f:
        l3l4_meta = json.load(f)

    cache_game_ids = list(tf_data["game_ids"])
    team_features = tf_data["features"]  # (N, 2, 34)
    game_context = gc_data["features"].copy()  # (N, 14), will update travel

    n_games = len(cache_game_ids)
    logger.info(f"L3/L4 cache: {n_games} games")

    # Build game_id -> index lookup
    game_id_to_idx = {gid: i for i, gid in enumerate(cache_game_ids)}

    # --- Load game data from database ---
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")

    # Get game info: teams, date, scores
    logger.info("Loading game info from database...")
    games_info = {}
    rows = conn.execute("""
        SELECT g.game_id, g.home_team, g.away_team, g.date_time_utc, g.season,
               th.pts AS home_pts, ta.pts AS away_pts
        FROM Games g
        JOIN Teams ht ON g.home_team = ht.abbreviation
        JOIN TeamBox th ON th.game_id = g.game_id AND th.team_id = ht.team_id
        JOIN Teams at2 ON g.away_team = at2.abbreviation
        JOIN TeamBox ta ON ta.game_id = g.game_id AND ta.team_id = at2.team_id
        WHERE g.status = 3
        """).fetchall()
    for row in rows:
        gid, home, away, dt, season, h_pts, a_pts = row
        games_info[gid] = {
            "home_team": home,
            "away_team": away,
            "date_time_utc": dt,
            "season": season,
            "home_pts": h_pts,
            "away_pts": a_pts,
        }
    logger.info(f"Loaded info for {len(games_info)} games")

    # Get rosters (player_id, team_id) for each game from PlayerBox
    logger.info("Loading rosters from PlayerBox...")
    roster_query = conn.execute("""
        SELECT pb.game_id, pb.player_id, pb.team_id, pb.min
        FROM PlayerBox pb
        WHERE pb.min > 0
        ORDER BY pb.game_id, pb.team_id, pb.min DESC
        """).fetchall()

    # Build per-game rosters grouped by team_id
    game_rosters: dict[str, dict[str, list[int]]] = {}
    for gid, pid, tid, minutes in roster_query:
        if gid not in game_rosters:
            game_rosters[gid] = {}
        if tid not in game_rosters[gid]:
            game_rosters[gid][tid] = []
        game_rosters[gid][tid].append(pid)
    logger.info(f"Loaded rosters for {len(game_rosters)} games")

    # Build team abbreviation -> team_id mapping
    team_abbr_to_id = {}
    rows = conn.execute("SELECT team_id, abbreviation FROM Teams").fetchall()
    for tid, abbr in rows:
        team_abbr_to_id[abbr] = tid

    # --- Determine which games to process ---
    if args.test:
        game_ids_to_process = cache_game_ids[: args.test]
    else:
        game_ids_to_process = cache_game_ids

    n_process = len(game_ids_to_process)
    logger.info(f"Processing {n_process} games...")

    # --- Pre-allocate output arrays ---
    max_roster = cfg.max_roster  # 15
    l2_vectors = np.zeros(
        (n_process, 2, cfg.d_l2_output), dtype=np.float32
    )  # (N, 2, 134)
    roster_summaries = np.zeros((n_process, 2, 12), dtype=np.float32)
    targets_margin = np.zeros(n_process, dtype=np.float32)
    targets_win = np.zeros(n_process, dtype=np.float32)
    targets_total = np.zeros(n_process, dtype=np.float32)
    seasons = np.array([""] * n_process, dtype=object)
    valid_mask = np.ones(n_process, dtype=bool)

    # Track travel feature updates
    travel_updated = 0

    # --- Process games in batches ---
    batch_size = 64
    batch_abilities = []
    batch_uncertainties = []
    batch_archetypes = []
    batch_masks = []
    batch_player_idxs = []
    batch_game_indices = []  # index into output arrays
    batch_side = []  # 0=home, 1=away

    # For roster summaries
    batch_roster_abilities = []
    batch_roster_uncertainties = []
    batch_roster_masks = []

    t0 = time.time()
    skipped = 0

    def flush_batch():
        """Run L2 forward on accumulated batch."""
        nonlocal batch_abilities, batch_uncertainties, batch_archetypes
        nonlocal batch_masks, batch_player_idxs, batch_game_indices, batch_side
        nonlocal batch_roster_abilities, batch_roster_uncertainties, batch_roster_masks

        if not batch_abilities:
            return

        # Stack and move to device
        ability_t = torch.tensor(
            np.stack(batch_abilities), dtype=torch.float32, device=device
        )
        unc_t = torch.tensor(
            np.stack(batch_uncertainties), dtype=torch.float32, device=device
        )
        arch_t = torch.tensor(
            np.stack(batch_archetypes), dtype=torch.float32, device=device
        )
        mask_t = torch.tensor(np.stack(batch_masks), dtype=torch.bool, device=device)
        pidx_t = torch.tensor(
            np.stack(batch_player_idxs), dtype=torch.long, device=device
        )

        with torch.no_grad():
            out = model(
                ability=ability_t,
                uncertainty=unc_t,
                archetypes=arch_t,
                mask=mask_t,
                player_idx=pidx_t,
            )
            team_vec = out["team_vector"].cpu().numpy()  # (B, 134)

        for i in range(len(batch_game_indices)):
            game_idx = batch_game_indices[i]
            side = batch_side[i]
            l2_vectors[game_idx, side] = team_vec[i]

            # Roster summary
            rs = compute_roster_summary(
                batch_roster_abilities[i],
                batch_roster_uncertainties[i],
                batch_roster_masks[i],
            )
            roster_summaries[game_idx, side] = rs

        # Clear batch
        batch_abilities = []
        batch_uncertainties = []
        batch_archetypes = []
        batch_masks = []
        batch_player_idxs = []
        batch_game_indices = []
        batch_side = []
        batch_roster_abilities = []
        batch_roster_uncertainties = []
        batch_roster_masks = []

    for proc_idx, game_id in enumerate(game_ids_to_process):
        if proc_idx % 5000 == 0 and proc_idx > 0:
            elapsed = time.time() - t0
            rate = proc_idx / elapsed
            eta = (n_process - proc_idx) / max(rate, 1e-6)
            logger.info(
                f"  [{proc_idx}/{n_process}] {rate:.1f} games/s, "
                f"ETA {eta/60:.1f} min, skipped {skipped}"
            )

        # Get game info
        if game_id not in games_info:
            valid_mask[proc_idx] = False
            skipped += 1
            continue

        info = games_info[game_id]
        home_team = info["home_team"]
        away_team = info["away_team"]

        # Targets
        h_pts = info["home_pts"]
        a_pts = info["away_pts"]
        targets_margin[proc_idx] = h_pts - a_pts
        targets_win[proc_idx] = 1.0 if h_pts > a_pts else 0.0
        targets_total[proc_idx] = h_pts + a_pts
        seasons[proc_idx] = info["season"]

        # Travel features (update game_context indices 8-11)
        cache_idx = game_id_to_idx.get(game_id)
        if cache_idx is not None:
            travel = compute_travel_features(
                conn, game_id, home_team, away_team, info["date_time_utc"]
            )
            game_context[cache_idx, 8] = travel["travel_dist_home"]
            game_context[cache_idx, 9] = travel["travel_dist_away"]
            game_context[cache_idx, 10] = travel["tz_crossings_home"]
            game_context[cache_idx, 11] = travel["tz_crossings_away"]
            travel_updated += 1

        # Get rosters
        if game_id not in game_rosters:
            valid_mask[proc_idx] = False
            skipped += 1
            continue

        game_roster = game_rosters[game_id]

        # Resolve team_ids for home and away
        home_tid = team_abbr_to_id.get(home_team)
        away_tid = team_abbr_to_id.get(away_team)

        if home_tid is None or away_tid is None:
            valid_mask[proc_idx] = False
            skipped += 1
            continue

        home_pids = game_roster.get(home_tid, [])
        away_pids = game_roster.get(away_tid, [])

        if len(home_pids) == 0 or len(away_pids) == 0:
            valid_mask[proc_idx] = False
            skipped += 1
            continue

        # Build L2 inputs for each side
        for side, pids in enumerate([home_pids, away_pids]):
            pids = pids[:max_roster]  # truncate to max
            n = len(pids)

            abilities = np.zeros((max_roster, 32), dtype=np.float32)
            uncertainties = np.zeros((max_roster, 32), dtype=np.float32)
            archetypes = np.zeros((max_roster, cfg.n_archetypes), dtype=np.float32)
            mask = np.zeros(max_roster, dtype=np.float32)
            player_idxs = np.zeros(max_roster, dtype=np.int64)

            filled = 0
            for pid in pids:
                if pid not in l1_vectors:
                    continue
                ab, unc, arch = get_l1_at_game(l1_vectors[pid], game_id)
                if ab is None:
                    # Try with use_latest for players whose first game is this one
                    ab, unc, arch = get_l1_at_game(
                        l1_vectors[pid], game_id, use_latest=True
                    )
                    if ab is None:
                        continue

                abilities[filled] = ab
                uncertainties[filled] = unc
                archetypes[filled] = arch
                mask[filled] = 1.0
                player_idxs[filled] = player_to_idx.get(pid, 0)
                filled += 1

            if filled == 0:
                # No valid players — skip this game
                valid_mask[proc_idx] = False
                skipped += 1
                break

            batch_abilities.append(abilities)
            batch_uncertainties.append(uncertainties)
            batch_archetypes.append(archetypes)
            batch_masks.append(mask.astype(bool))
            batch_player_idxs.append(player_idxs)
            batch_game_indices.append(proc_idx)
            batch_side.append(side)

            batch_roster_abilities.append(abilities.copy())
            batch_roster_uncertainties.append(uncertainties.copy())
            batch_roster_masks.append(mask.copy())

            if len(batch_abilities) >= batch_size:
                flush_batch()

    # Flush remaining
    flush_batch()
    conn.close()

    elapsed = time.time() - t0
    logger.info(
        f"Processed {n_process} games in {elapsed:.1f}s "
        f"({n_process/elapsed:.1f} games/s), skipped {skipped}, "
        f"travel updated for {travel_updated} games"
    )

    # --- Filter to valid games only ---
    valid_indices = np.where(valid_mask)[0]
    logger.info(f"Valid games: {len(valid_indices)} / {n_process}")

    l2_vectors = l2_vectors[valid_indices]
    roster_summaries = roster_summaries[valid_indices]
    targets_margin = targets_margin[valid_indices]
    targets_win = targets_win[valid_indices]
    targets_total = targets_total[valid_indices]
    valid_game_ids = [game_ids_to_process[i] for i in valid_indices]
    valid_seasons = seasons[valid_indices]

    # Map valid game_ids to L3/L4 cache indices for team_features and game_context
    cache_indices = np.array(
        [game_id_to_idx[gid] for gid in valid_game_ids if gid in game_id_to_idx],
        dtype=np.int64,
    )
    valid_team_features = team_features[cache_indices]  # (N_valid, 2, 34)
    valid_game_context = game_context[cache_indices]  # (N_valid, 14)

    # --- Compute normalization stats for new features ---
    # Travel features (indices 8-11) now have real values; recompute l4 stats
    l4_mean = valid_game_context.mean(axis=0)
    l4_std = valid_game_context.std(axis=0)
    l4_std[l4_std < 1e-8] = 1e-8  # prevent div-by-zero

    # Roster summary normalization
    # Flatten both home and away for stats
    rs_flat = roster_summaries.reshape(-1, 12)
    rs_mean = rs_flat.mean(axis=0)
    rs_std = rs_flat.std(axis=0)
    rs_std[rs_std < 1e-8] = 1e-8

    # L2 vector normalization
    l2_flat = l2_vectors.reshape(-1, 134)
    l2_mean = l2_flat.mean(axis=0)
    l2_std = l2_flat.std(axis=0)
    l2_std[l2_std < 1e-8] = 1e-8

    # --- Save ---
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        str(OUTPUT_DIR / "l2_vectors.npz"),
        vectors=l2_vectors,
    )
    np.savez_compressed(
        str(OUTPUT_DIR / "roster_summaries.npz"),
        summaries=roster_summaries,
    )
    np.savez_compressed(
        str(OUTPUT_DIR / "team_features.npz"),
        features=valid_team_features,
    )
    np.savez_compressed(
        str(OUTPUT_DIR / "game_context.npz"),
        features=valid_game_context,
    )
    np.savez_compressed(
        str(OUTPUT_DIR / "targets.npz"),
        margin=targets_margin,
        home_win=targets_win,
        total=targets_total,
    )
    np.save(str(OUTPUT_DIR / "game_ids.npy"), np.array(valid_game_ids, dtype=object))
    np.save(str(OUTPUT_DIR / "seasons.npy"), np.array(valid_seasons, dtype=object))

    # Metadata
    metadata = {
        "n_games": len(valid_game_ids),
        "l2_dim": int(cfg.d_l2_output),
        "roster_summary_dim": 12,
        "team_features_dim": 34,
        "game_context_dim": 14,
        "normalization": {
            "l2_mean": l2_mean.tolist(),
            "l2_std": l2_std.tolist(),
            "roster_summary_mean": rs_mean.tolist(),
            "roster_summary_std": rs_std.tolist(),
            "l3_mean": l3l4_meta["normalization"]["l3_mean"],
            "l3_std": l3l4_meta["normalization"]["l3_std"],
            "l4_mean": l4_mean.tolist(),
            "l4_std": l4_std.tolist(),
        },
        "l2_checkpoint": str(L2_CHECKPOINT),
        "l2_epoch": ckpt.get("epoch", -1),
        "skipped_games": skipped,
        "travel_features_filled": travel_updated,
        "player_to_idx_size": len(player_to_idx),
    }
    with open(str(OUTPUT_DIR / "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    # The predictor needs the same player -> embedding index mapping as training
    with open(str(OUTPUT_DIR / "player_to_idx.json"), "w") as f:
        json.dump({str(pid): idx for pid, idx in player_to_idx.items()}, f)

    logger.info(f"Cache saved to {OUTPUT_DIR}")
    logger.info(f"  L2 vectors: {l2_vectors.shape}")
    logger.info(f"  Roster summaries: {roster_summaries.shape}")
    logger.info(f"  Team features: {valid_team_features.shape}")
    logger.info(f"  Game context: {valid_game_context.shape}")
    logger.info(f"  Targets: margin/win/total each {targets_margin.shape}")

    # Quick sanity check
    logger.info(f"\nSanity check:")
    logger.info(
        f"  Margin range: [{targets_margin.min():.0f}, {targets_margin.max():.0f}], mean={targets_margin.mean():.1f}"
    )
    logger.info(f"  Win rate: {targets_win.mean():.3f}")
    logger.info(
        f"  Total range: [{targets_total.min():.0f}, {targets_total.max():.0f}], mean={targets_total.mean():.1f}"
    )
    logger.info(
        f"  L2 vector norm (mean): {np.linalg.norm(l2_vectors, axis=-1).mean():.3f}"
    )


def main():
    parser = argparse.ArgumentParser(description="Build Phase B cache")
    parser.add_argument(
        "--test",
        type=int,
        default=None,
        help="Only process first N games (for testing)",
    )
    args = parser.parse_args()
    build_cache(args)


if __name__ == "__main__":
    main()
