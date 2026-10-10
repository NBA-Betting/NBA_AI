"""
L1 Updater: keep player ability vectors current for live predictions.

For each player with games newer than the last one in their vector file, the
player's L1 input sequence is built by src.phase5.cache_builder (the same code
that builds the L1 training cache), and the Kalman filter is continued over
the new games (src.phase5.l1_extract). Players without a file get a full
extraction. The filter only moves forward: a game backfilled before a player's
last stored game is not added (re-extracting the career would also re-derive
history the deployed L1 model was trained on differently).

Usage:
    python -m src.pipeline.l1_updater                    # Update all players with new games
    python -m src.pipeline.l1_updater --player-ids 201142,203999
    python -m src.pipeline.l1_updater --dry-run          # Report what would be updated
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from bisect import bisect_right
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.database import DB_PATH, get_db
from src.phase5.cache_builder import (
    L1_SEASON_TYPES,
    PROFILE_COLUMNS,
    _load_team_id_to_abbr,
    _load_team_rolling_stats,
    build_player_cache,
    player_profile,
)
from src.phase5.config import NKEHConfig
from src.phase5.dataset import Normalizer, load_metadata, load_profiles
from src.phase5.l1_extract import extract_player, kalman_steps
from src.phase5.model import NKEH

logger = logging.getLogger(__name__)

L1_VECTORS_DIR = PROJECT_ROOT / "data" / "l2_cache" / "l1_vectors"
L1_CHECKPOINT = PROJECT_ROOT / "models" / "phase5" / "l1.pt"

# A game without PBP stats is held back this long for enrichment to catch up,
# then processed with PBP masked (as training does for games without PBP)
PBP_WAIT_DAYS = 2


class L1IncrementalUpdater:
    """Keeps L1 ability vectors current using the Kalman filter."""

    def __init__(
        self,
        checkpoint_path: str | Path = L1_CHECKPOINT,
        db_path: str | Path = DB_PATH,
        vectors_dir: str | Path = L1_VECTORS_DIR,
        device: str | None = None,
    ):
        self.db_path = Path(db_path)
        self.vectors_dir = Path(vectors_dir)
        self.vectors_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.metadata = load_metadata()
        self.normalizer = Normalizer(self.metadata)
        self._profiles_data = load_profiles()
        self._profile_idx = {
            int(pid): i for i, pid in enumerate(self._profiles_data["person_ids"])
        }
        self.model = self._load_model(checkpoint_path)

        with get_db(str(self.db_path)) as conn:
            self.team_stats = _load_team_rolling_stats(conn)
            self.team_id_to_abbr = _load_team_id_to_abbr(conn)

        logger.info(
            f"L1 updater ready: {len(self._profile_idx)} player profiles, "
            f"{len(self.team_stats)} team-season stats, device={self.device}"
        )

    def _load_model(self, checkpoint_path: str | Path) -> NKEH:
        """Load the trained NKEH model."""
        cfg = NKEHConfig(
            n_box_stats=len(self.metadata["box_stat_columns"]),
            n_pbp_stats=len(self.metadata["pbp_stat_columns"]),
            n_context=len(self.metadata["context_columns"]),
            n_profile=len(self.metadata["profile_columns"]),
        )
        model = NKEH(cfg)
        ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        model.to(self.device)
        model.eval()
        logger.info(
            f"Loaded NKEH model (epoch {ckpt.get('epoch', '?')}) on {self.device}"
        )
        return model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_all_new(self, dry_run: bool = False) -> dict[int, int]:
        """
        Find all players with new games and update their L1 vectors.

        Returns:
            dict mapping player_id → number of new games processed
        """
        results = {}
        players_with_new = self._find_players_with_new_games()

        if not players_with_new:
            logger.info("No players have new games to process")
            return results

        logger.info(f"Found {len(players_with_new)} players with new games")

        if dry_run:
            for pid, n_new in players_with_new.items():
                logger.info(f"  [dry-run] Player {pid}: {n_new} new game(s)")
            return {pid: 0 for pid in players_with_new}

        t0 = time.time()
        n_errors = 0
        for pid in players_with_new:
            try:
                n_processed = self.update_player(pid)
                if n_processed > 0:
                    results[pid] = n_processed
            except Exception as e:
                logger.warning(f"  Error updating player {pid}: {e}")
                n_errors += 1

        logger.info(
            f"L1 update complete: {len(results)} players, {sum(results.values())} "
            f"games in {time.time() - t0:.1f}s ({n_errors} errors)"
        )
        return results

    def update_player(self, player_id: int) -> int:
        """
        Update a single player's L1 vectors with any new games.

        Returns the number of games processed.
        """
        with get_db(str(self.db_path)) as conn:
            seq = build_player_cache(
                conn,
                player_id,
                self.team_stats,
                self.team_id_to_abbr,
                min_games=1,
                require_profile=False,
            )
        if seq is None:
            return 0

        npz_path = self.vectors_dir / f"{player_id}.npz"
        if npz_path.exists():
            return self._continue(player_id, seq, npz_path)
        return self._full_extraction(player_id, seq, npz_path)

    # ------------------------------------------------------------------
    # Updates
    # ------------------------------------------------------------------

    def _continue(self, player_id: int, seq: dict, npz_path: Path) -> int:
        """Kalman steps over the games dated after the last stored one."""
        existing = np.load(str(npz_path), allow_pickle=True)
        last_id = str(existing["game_ids"][-1])
        if last_id not in seq["game_ids"]:
            logger.warning(f"  Player {player_id}: last stored game {last_id} not in DB")
            return 0
        last_date = seq["dates"][seq["game_ids"].index(last_id)]
        start = bisect_right(seq["dates"], last_date)
        end = self._ready_end(seq, start)
        if end == start:
            return 0

        abilities, Ps = kalman_steps(
            seq,
            start,
            end,
            existing["ability"][-1],
            existing["uncertainty"][-1],
            self.model,
            self.normalizer,
            self.device,
        )
        np.savez_compressed(
            str(npz_path),
            game_ids=np.array(
                list(existing["game_ids"]) + seq["game_ids"][start:end], dtype=object
            ),
            ability=np.concatenate([existing["ability"], abilities]),
            uncertainty=np.concatenate([existing["uncertainty"], Ps]),
            archetype_weights=existing["archetype_weights"],
        )
        logger.debug(f"  Player {player_id}: +{end - start} games")
        return end - start

    def _full_extraction(self, player_id: int, seq: dict, npz_path: Path) -> int:
        """Filter the player's whole career (new player or stale file)."""
        end = self._ready_end(seq, 0)
        if end == 0:
            return 0
        seq = {k: v[:end] for k, v in seq.items()}
        out = extract_player(
            seq,
            self.model,
            self.normalizer,
            self._get_profile_tensor(player_id),
            self.device,
        )
        np.savez_compressed(
            str(npz_path),
            game_ids=np.array(seq["game_ids"], dtype=object),
            ability=out["ability"],
            uncertainty=out["uncertainty"],
            archetype_weights=out["archetype_weights"],
        )
        logger.info(f"  Player {player_id}: extracted {end} games (full sequence)")
        return end

    @staticmethod
    def _ready_end(seq: dict, start: int) -> int:
        """End of the run of games from `start` that can be processed now: stop
        at a recent game whose PBP stats aren't in yet (later games wait too)."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=PBP_WAIT_DAYS)
        end = start
        while end < len(seq["game_ids"]):
            played = datetime.fromisoformat(seq["dates"][end].replace("Z", "+00:00"))
            if not seq["has_pbp"][end] and played > cutoff:
                break
            end += 1
        return end

    def _get_profile_tensor(self, player_id: int) -> torch.Tensor:
        """Profile vector: the training cache's, else from the database."""
        if player_id in self._profile_idx:
            idx = self._profile_idx[player_id]
            profile_cols = self.metadata["profile_columns"]
            vals = [float(self._profiles_data[col][idx]) for col in profile_cols]
            return torch.tensor(vals, dtype=torch.float32)

        with get_db(str(self.db_path)) as conn:
            profile_dict = player_profile(conn, player_id)
        return torch.tensor(
            [profile_dict.get(col, 0) for col in PROFILE_COLUMNS], dtype=torch.float32
        )

    # ------------------------------------------------------------------
    # Discovery: which players have new games?
    # ------------------------------------------------------------------

    def _find_players_with_new_games(self) -> dict[int, int]:
        """
        Find players with games newer than the last game in their L1 vectors.

        Compares dates rather than counts: a backfilled game older than a
        player's last processed game doesn't make him due for an update.
        Returns {player_id: n_new_games}.
        """
        placeholders = ",".join("?" * len(L1_SEASON_TYPES))
        with get_db(str(self.db_path)) as conn:
            rows = conn.execute(
                f"""
                SELECT pb.player_id, COUNT(*), MAX(g.date_time_utc)
                FROM PlayerBox pb
                JOIN Games g ON pb.game_id = g.game_id
                WHERE g.status = 3
                AND pb.min > 0
                AND g.season_type IN ({placeholders})
                GROUP BY pb.player_id
                """,
                L1_SEASON_TYPES,
            ).fetchall()
            game_dates = dict(
                conn.execute(
                    f"SELECT game_id, date_time_utc FROM Games "
                    f"WHERE status = 3 AND season_type IN ({placeholders})",
                    L1_SEASON_TYPES,
                ).fetchall()
            )

        results = {}
        for pid, n_db, latest in rows:
            npz_path = self.vectors_dir / f"{pid}.npz"
            if npz_path.exists():
                game_ids = np.load(str(npz_path), allow_pickle=True)["game_ids"]
                if latest <= game_dates.get(str(game_ids[-1]), ""):
                    continue
                n_new = max(n_db - len(game_ids), 1)
            else:
                n_new = n_db
            results[int(pid)] = n_new

        return results


def main():
    parser = argparse.ArgumentParser(description="L1 Incremental Vector Updater")
    parser.add_argument(
        "--player-ids",
        type=str,
        default=None,
        help="Comma-separated player IDs to update (default: all with new games)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be updated without making changes",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (cuda/cpu). Default: auto-detect",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    updater = L1IncrementalUpdater(device=args.device)

    if args.player_ids:
        pids = [int(p) for p in args.player_ids.split(",")]
        if args.dry_run:
            logger.info(f"[dry-run] Would update {len(pids)} players")
        else:
            for pid in pids:
                n = updater.update_player(pid)
                logger.info(f"Player {pid}: {n} new games")
    else:
        results = updater.update_all_new(dry_run=args.dry_run)
        if results:
            total = sum(results.values())
            logger.info(f"Updated {len(results)} players, {total} total new games")


if __name__ == "__main__":
    main()
