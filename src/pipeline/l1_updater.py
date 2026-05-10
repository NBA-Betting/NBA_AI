"""
L1 Incremental Updater: Kalman-step updates for player ability vectors.

Instead of replaying a player's full career (300+ games), this module:
  1. Loads the player's existing L1 vectors (.npz with game_ids, ability, uncertainty)
  2. Finds new games in the DB since the last processed game
  3. Runs one Kalman predict + update step per new game
  4. Appends the new vectors and saves the updated .npz

For brand-new players with no existing file, falls back to full-sequence extraction.

Usage:
    python -m src.pipeline.l1_updater                    # Update all players with new games
    python -m src.pipeline.l1_updater --player-ids 201142,203999
    python -m src.pipeline.l1_updater --dry-run          # Report what would be updated
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.database import DB_PATH, get_db
from src.phase5.cache_builder import (
    ALTITUDE_MAP,
    BOX_STAT_COLUMNS,
    CONTEXT_COLUMNS,
    HISTORICAL_TO_MODERN,
    PBP_STAT_COLUMNS,
    _load_pbp_stats,
    _load_player_games,
    _load_player_profile,
    _load_team_rolling_stats,
)
from src.phase5.config import NKEHConfig
from src.phase5.dataset import CACHE_DIR as L1_CACHE_DIR
from src.phase5.dataset import Normalizer, load_metadata, load_profiles
from src.phase5.model import NKEH

logger = logging.getLogger(__name__)

L1_VECTORS_DIR = PROJECT_ROOT / "data" / "l2_cache" / "l1_vectors"
L1_CHECKPOINT = PROJECT_ROOT / "models" / "phase5" / "l1.pt"

# Maximum sequence length for full-sequence fallback (new players)
MAX_LEN = 300


class L1IncrementalUpdater:
    """Incrementally updates L1 ability vectors using the Kalman filter."""

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

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Load model and normalization infrastructure
        self.metadata = load_metadata()
        self.normalizer = Normalizer(self.metadata)
        self._profiles_data = load_profiles()
        self._profile_idx = {
            int(pid): i for i, pid in enumerate(self._profiles_data["person_ids"])
        }
        self.model = self._load_model(checkpoint_path)

        # Preload team stats and team_id→abbr mapping for context features
        with get_db(str(self.db_path)) as conn:
            self.team_stats = _load_team_rolling_stats(conn)
            self.team_id_to_abbr = self._load_team_id_to_abbr(conn)

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

    @staticmethod
    def _load_team_id_to_abbr(conn) -> dict[int, str]:
        """Build team_id → abbreviation mapping."""
        rows = conn.execute("SELECT team_id, abbreviation FROM Teams").fetchall()
        return {int(row[0]): row[1] for row in rows}

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
        n_updated = 0
        n_errors = 0

        for pid, n_new in players_with_new.items():
            try:
                n_processed = self.update_player(pid)
                if n_processed > 0:
                    results[pid] = n_processed
                    n_updated += 1
            except Exception as e:
                logger.warning(f"  Error updating player {pid}: {e}")
                n_errors += 1

        elapsed = time.time() - t0
        total_games = sum(results.values())
        logger.info(
            f"L1 update complete: {n_updated} players, {total_games} games "
            f"in {elapsed:.1f}s ({n_errors} errors)"
        )
        return results

    def update_player(self, player_id: int) -> int:
        """
        Update a single player's L1 vectors with any new games.

        Returns the number of new games processed.
        """
        npz_path = self.vectors_dir / f"{player_id}.npz"

        if npz_path.exists():
            return self._incremental_update(player_id, npz_path)
        else:
            return self._full_extraction(player_id, npz_path)

    # ------------------------------------------------------------------
    # Incremental update (existing player)
    # ------------------------------------------------------------------

    def _incremental_update(self, player_id: int, npz_path: Path) -> int:
        """Run incremental Kalman steps for new games."""
        # Load existing vectors
        existing = np.load(str(npz_path), allow_pickle=True)
        existing_game_ids = set(existing["game_ids"].tolist())
        last_ability = existing["ability"][-1]  # (32,)
        last_P = existing["uncertainty"][-1]  # (32,)
        arch_weights = existing["archetype_weights"]  # (10,)

        # Find new games from DB
        with get_db(str(self.db_path)) as conn:
            all_games = _load_player_games(conn, player_id)
            pbp_data = _load_pbp_stats(conn, player_id)

        new_games = [g for g in all_games if g["game_id"] not in existing_game_ids]
        if not new_games:
            return 0

        # Load profile
        profile_vec = self._get_profile_tensor(player_id)
        if profile_vec is None:
            logger.warning(f"  No profile for player {player_id}, skipping")
            return 0

        # We need the full game list to compute context features correctly
        # (career_games_played, rest_days need the full sequence)
        n_existing = len(existing["game_ids"])

        # Find the last existing game's date for rest_days computation
        last_existing_game = None
        for g in all_games:
            if g["game_id"] in existing_game_ids:
                last_existing_game = g

        # Run Kalman steps for each new game
        new_abilities = []
        new_Ps = []
        new_game_ids = []

        mu = torch.tensor(
            last_ability, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        P = torch.tensor(last_P, dtype=torch.float32, device=self.device).unsqueeze(0)
        prev_game = last_existing_game
        career_count = n_existing

        for game in new_games:
            career_count += 1

            # Build context for this game
            context = self._build_game_context(
                game, prev_game, career_count, player_id, pbp_data
            )
            box = self._build_box_stats(game)
            pbp = self._build_pbp_stats(game["game_id"], pbp_data)

            # Normalize
            box_t = self.normalizer.normalize_box(
                torch.tensor(box, dtype=torch.float32).unsqueeze(0)
            ).to(self.device)
            pbp_t = self.normalizer.normalize_pbp(
                torch.tensor(pbp, dtype=torch.float32).unsqueeze(0)
            ).to(self.device)
            ctx_t = self.normalizer.normalize_context(
                torch.tensor(context, dtype=torch.float32).unsqueeze(0)
            ).to(self.device)

            # Extract age and days_gap BEFORE normalization (as done in extract_l1_vectors.py)
            age = torch.tensor(
                [[(context[0] - 8.0) / 8.0]], dtype=torch.float32, device=self.device
            )
            days_gap = torch.tensor(
                [[context[8]]], dtype=torch.float32, device=self.device
            )

            # Kalman predict
            with torch.no_grad():
                mu_pred, P_pred = self.model.kalman_predict(mu, P, age, days_gap)

                # Game encoder observation
                mu_obs, log_sigma_obs = self.model.game_encoder(
                    box_t, pbp_t, ctx_t, mu_pred
                )
                sigma_obs = F.softplus(log_sigma_obs)

                # Kalman update
                mu, P = self.model.kalman_update(mu_pred, P_pred, mu_obs, sigma_obs)

            new_abilities.append(mu[0].cpu().numpy())
            new_Ps.append(P[0].cpu().numpy())
            new_game_ids.append(game["game_id"])
            prev_game = game

        # Append to existing data and save
        updated_game_ids = np.concatenate(
            [
                existing["game_ids"],
                np.array(new_game_ids, dtype=object),
            ]
        )
        updated_ability = np.concatenate(
            [
                existing["ability"],
                np.array(new_abilities, dtype=np.float32),
            ]
        )
        updated_P = np.concatenate(
            [
                existing["uncertainty"],
                np.array(new_Ps, dtype=np.float32),
            ]
        )

        np.savez_compressed(
            str(npz_path),
            game_ids=updated_game_ids,
            ability=updated_ability,
            uncertainty=updated_P,
            archetype_weights=arch_weights,
        )

        n_new = len(new_game_ids)
        logger.debug(
            f"  Player {player_id}: +{n_new} games "
            f"({n_existing}→{n_existing + n_new} total)"
        )
        return n_new

    # ------------------------------------------------------------------
    # Full extraction fallback (new player, no existing .npz)
    # ------------------------------------------------------------------

    def _full_extraction(self, player_id: int, npz_path: Path) -> int:
        """Full-sequence extraction for a new player."""
        with get_db(str(self.db_path)) as conn:
            all_games = _load_player_games(conn, player_id)
            pbp_data = _load_pbp_stats(conn, player_id)

        if not all_games:
            return 0

        profile_vec = self._get_profile_tensor(player_id)
        if profile_vec is None:
            logger.warning(f"  No profile for new player {player_id}, skipping")
            return 0

        # Build full sequences
        n = len(all_games)
        box_seq = np.zeros((n, len(BOX_STAT_COLUMNS)), dtype=np.float32)
        pbp_seq = np.zeros((n, len(PBP_STAT_COLUMNS)), dtype=np.float32)
        ctx_seq = np.zeros((n, len(CONTEXT_COLUMNS)), dtype=np.float32)
        has_pbp = np.zeros(n, dtype=bool)
        game_ids = []

        prev_game = None
        prev_season = None
        season_game_count = 0
        profile_data = self._get_profile_dict(player_id)
        birth_year = profile_data.get("birth_year", 1990)

        for i, game in enumerate(all_games):
            game_ids.append(game["game_id"])
            box_seq[i] = self._build_box_stats(game)

            pbp = pbp_data.get(game["game_id"])
            if pbp:
                has_pbp[i] = True
                for j, col in enumerate(PBP_STAT_COLUMNS):
                    pbp_seq[i, j] = float(pbp.get(col, 0))

            season = game.get("season", "")
            if season != prev_season:
                season_game_count = 0
                prev_season = season
            season_game_count += 1

            ctx_seq[i] = self._build_game_context(
                game,
                prev_game,
                i + 1,
                player_id,
                pbp_data,
                birth_year=birth_year,
                season_game_count=season_game_count,
            )
            prev_game = game

        # Run through model
        init_context_raw = ctx_seq[0].copy()

        # Handle sequence length > MAX_LEN (use last MAX_LEN games)
        if n > MAX_LEN:
            # Use last MAX_LEN games, accepting reduced context for earliest games
            start = n - MAX_LEN
            box_in = box_seq[start:]
            pbp_in = pbp_seq[start:]
            ctx_in = ctx_seq[start:]
            has_pbp_in = has_pbp[start:]
            T = MAX_LEN
        else:
            box_in = box_seq
            pbp_in = pbp_seq
            ctx_in = ctx_seq
            has_pbp_in = has_pbp
            T = n

        # Normalize
        box_t = self.normalizer.normalize_box(torch.tensor(box_in, dtype=torch.float32))
        pbp_t = self.normalizer.normalize_pbp(torch.tensor(pbp_in, dtype=torch.float32))
        no_pbp_mask = ~torch.tensor(has_pbp_in, dtype=torch.bool)
        pbp_t[no_pbp_mask] = 0.0

        ctx_t = self.normalizer.normalize_context(
            torch.tensor(ctx_in, dtype=torch.float32)
        )
        profile_norm = self.normalizer.normalize_profile(profile_vec)
        init_ctx_norm = self.normalizer.normalize_context(
            torch.tensor(init_context_raw, dtype=torch.float32)
        )

        # Age and days_gap (extracted before normalization)
        age = (torch.tensor(ctx_in[:, 0:1], dtype=torch.float32) - 8.0) / 8.0
        days_gap = torch.tensor(ctx_in[:, 8:9], dtype=torch.float32)

        # Pad to MAX_LEN
        pad_len = MAX_LEN - T
        mask = torch.zeros(MAX_LEN, dtype=torch.bool)
        mask[:T] = True

        if pad_len > 0:
            box_t = torch.cat([box_t, torch.zeros(pad_len, box_t.shape[1])], dim=0)
            pbp_t = torch.cat([pbp_t, torch.zeros(pad_len, pbp_t.shape[1])], dim=0)
            ctx_t = torch.cat([ctx_t, torch.zeros(pad_len, ctx_t.shape[1])], dim=0)
            age = torch.cat([age, torch.zeros(pad_len, 1)], dim=0)
            days_gap = torch.cat([days_gap, torch.zeros(pad_len, 1)], dim=0)

        # Forward pass
        with torch.no_grad():
            out = self.model.forward_sequence(
                box_stats_seq=box_t.unsqueeze(0).to(self.device),
                pbp_stats_seq=pbp_t.unsqueeze(0).to(self.device),
                context_seq=ctx_t.unsqueeze(0).to(self.device),
                profile=profile_norm.unsqueeze(0).to(self.device),
                age_seq=age.unsqueeze(0).to(self.device),
                seq_mask=mask.unsqueeze(0).to(self.device),
                days_gap_seq=days_gap.unsqueeze(0).to(self.device),
                init_context=init_ctx_norm.unsqueeze(0).to(self.device),
            )

        ability = out["ability"][0, :T].cpu().numpy()
        P = out["P"][0, :T].cpu().numpy()
        arch_weights = out["archetype_weights"][0].cpu().numpy()

        # For long careers where we truncated, pad with zeros for early games
        if n > MAX_LEN:
            start = n - MAX_LEN
            full_ability = np.zeros((n, ability.shape[1]), dtype=np.float32)
            full_P = np.ones(
                (n, P.shape[1]), dtype=np.float32
            )  # high uncertainty for early games
            full_ability[start:] = ability
            full_P[start:] = P
            ability = full_ability
            P = full_P

        np.savez_compressed(
            str(npz_path),
            game_ids=np.array(game_ids, dtype=object),
            ability=ability,
            uncertainty=P,
            archetype_weights=arch_weights,
        )

        logger.info(f"  New player {player_id}: extracted {n} games (full sequence)")
        return n

    # ------------------------------------------------------------------
    # Context and feature construction helpers
    # ------------------------------------------------------------------

    def _build_box_stats(self, game: dict) -> np.ndarray:
        """Extract 16-d box stats from a game dict."""
        stats = np.zeros(len(BOX_STAT_COLUMNS), dtype=np.float32)
        for j, col in enumerate(BOX_STAT_COLUMNS):
            val = game.get(col)
            stats[j] = float(val) if val is not None else 0.0
        return stats

    def _build_pbp_stats(self, game_id: str, pbp_data: dict[str, dict]) -> np.ndarray:
        """Extract 56-d PBP stats for a game, zero-fill if missing."""
        stats = np.zeros(len(PBP_STAT_COLUMNS), dtype=np.float32)
        pbp = pbp_data.get(game_id)
        if pbp:
            for j, col in enumerate(PBP_STAT_COLUMNS):
                stats[j] = float(pbp.get(col, 0))
        return stats

    def _build_game_context(
        self,
        game: dict,
        prev_game: dict | None,
        career_count: int,
        player_id: int,
        pbp_data: dict,
        birth_year: float | None = None,
        season_game_count: int | None = None,
    ) -> np.ndarray:
        """Build 12-d context vector for a single game."""
        ctx = np.zeros(len(CONTEXT_COLUMNS), dtype=np.float32)

        date_str = self._extract_game_date(game.get("date_time_utc", ""))
        season = game.get("season", "")

        # If birth_year not passed, look it up from profiles
        if birth_year is None:
            if player_id in self._profile_idx:
                idx = self._profile_idx[player_id]
                birth_year = float(self._profiles_data["birth_year"][idx])
            else:
                birth_year = 1990.0

        # Age
        if date_str and birth_year:
            try:
                game_year = int(date_str[:4])
                game_month = int(date_str[5:7])
                ctx[0] = game_year - birth_year + (game_month - 6) / 12.0
            except (ValueError, IndexError):
                ctx[0] = 25.0

        # Rest days and days_since_last_game
        if prev_game:
            prev_date_str = self._extract_game_date(prev_game.get("date_time_utc", ""))
            if prev_date_str and date_str:
                try:
                    d1 = datetime.strptime(prev_date_str, "%Y-%m-%d")
                    d2 = datetime.strptime(date_str, "%Y-%m-%d")
                    rest = (d2 - d1).days - 1
                    ctx[1] = max(0, min(rest, 30))
                    ctx[8] = (d2 - d1).days
                except (ValueError, TypeError):
                    ctx[1] = 1.0
                    ctx[8] = 2.0
            else:
                ctx[1] = 1.0
                ctx[8] = 2.0
        else:
            ctx[1] = 3.0
            ctx[8] = 7.0

        # Home flag
        home_team = game.get("home_team", "")
        away_team = game.get("away_team", "")
        player_team_id = game.get("team_id")
        home_modern = HISTORICAL_TO_MODERN.get(home_team, home_team)
        away_modern = HISTORICAL_TO_MODERN.get(away_team, away_team)

        player_abbr = ""
        if player_team_id and self.team_id_to_abbr:
            player_abbr = self.team_id_to_abbr.get(int(player_team_id), "")
        is_home = player_abbr in (home_team, home_modern) if player_abbr else False
        ctx[2] = 1.0 if is_home else 0.0

        # Team and opponent stats
        opponent = away_modern if is_home else home_modern
        player_team = home_modern if is_home else away_modern

        opp_stats = self.team_stats.get((opponent, season), {})
        tm_stats = self.team_stats.get((player_team, season), {})

        ctx[3] = opp_stats.get("drtg", 110.0)
        ctx[4] = tm_stats.get("pace", 200.0)
        ctx[9] = opp_stats.get("pace", 200.0)
        ctx[10] = tm_stats.get("ortg", 110.0)

        # Minutes share
        player_min = float(game.get("min", 0) or 0)
        ctx[5] = player_min / 240.0

        # Season progress
        if season_game_count is not None:
            ctx[6] = min(season_game_count / 82.0, 1.0)
        else:
            ctx[6] = 0.5  # default mid-season

        # Career games played
        ctx[7] = float(career_count)

        # Altitude
        ctx[11] = float(ALTITUDE_MAP.get(home_team, 0))

        return ctx

    @staticmethod
    def _extract_game_date(date_time_utc: str) -> str:
        """Extract YYYY-MM-DD from a datetime string."""
        if not date_time_utc:
            return ""
        return date_time_utc[:10]

    @staticmethod
    def _default_profile_dict() -> dict:
        """
        League-average / sentinel profile used for in-season call-ups not yet
        in the PlayerAttributes table.

        Values mirror the field-level defaults in
        src.phase5.cache_builder._load_player_profile so behaviour is
        consistent with players who have a partial PlayerAttributes row.
        """
        return {
            "height_inches": 78,  # ~6'6"
            "weight": 215,
            "draft_pick": 60,  # last-pick equivalent
            "undrafted": 1.0,  # treat unknown call-ups as undrafted
            "birth_year": 2000,  # roughly current rookie age
            "wingspan_inches": 78,
            "pos_g": 0.0,
            "pos_f": 1.0,  # default to forward (matches cache_builder)
            "pos_c": 0.0,
        }

    def _get_profile_tensor(self, player_id: int) -> torch.Tensor | None:
        """Get profile tensor for a player from the profiles cache."""
        if player_id in self._profile_idx:
            idx = self._profile_idx[player_id]
            profile_cols = self.metadata["profile_columns"]
            vals = [float(self._profiles_data[col][idx]) for col in profile_cols]
            return torch.tensor(vals, dtype=torch.float32)

        # Fallback: load from DB for players not in cache
        with get_db(str(self.db_path)) as conn:
            profile_dict = _load_player_profile(conn, player_id)

        if not profile_dict:
            # No PlayerAttributes row (typical for late-season call-ups added to
            # the Players table after the last collect_player_attributes run).
            # Use a default profile so we still build L1 vectors instead of
            # silently skipping the player.
            logger.info(
                f"  Player {player_id}: no PlayerAttributes row, "
                f"using default profile (call-up fallback)"
            )
            profile_dict = self._default_profile_dict()

        from src.phase5.cache_builder import PROFILE_COLUMNS

        vec = np.array(
            [profile_dict.get(col, 0) for col in PROFILE_COLUMNS],
            dtype=np.float32,
        )
        return torch.tensor(vec, dtype=torch.float32)

    def _get_profile_dict(self, player_id: int) -> dict:
        """Get raw profile dict for a player."""
        with get_db(str(self.db_path)) as conn:
            profile_dict = _load_player_profile(conn, player_id)
        if not profile_dict:
            return self._default_profile_dict()
        return profile_dict

    # ------------------------------------------------------------------
    # Discovery: which players have new games?
    # ------------------------------------------------------------------

    def _find_players_with_new_games(self) -> dict[int, int]:
        """
        Find players whose DB has games not yet in their L1 vectors.

        Uses a single DB query for efficiency, then compares against cached counts.
        Returns {player_id: n_new_games}.
        """
        # Single query: count finalized games per player
        with get_db(str(self.db_path)) as conn:
            rows = conn.execute("""
                SELECT pb.player_id, COUNT(*) as n_games
                FROM PlayerBox pb
                JOIN Games g ON pb.game_id = g.game_id
                WHERE g.status = 3
                AND pb.min > 0
                AND g.season_type IN ('Regular Season', 'Post Season')
                GROUP BY pb.player_id
                """).fetchall()

        db_counts = {int(row[0]): row[1] for row in rows}

        # Compare against existing .npz files
        results = {}
        for pid, n_db in db_counts.items():
            npz_path = self.vectors_dir / f"{pid}.npz"
            if npz_path.exists():
                existing = np.load(str(npz_path), allow_pickle=True)
                n_existing = len(existing["game_ids"])
            else:
                n_existing = 0

            n_new = n_db - n_existing
            if n_new > 0:
                results[pid] = n_new

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
