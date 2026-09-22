"""
Team Feature Computer: compute L3 team features (34-d) and L4 game context (14-d).

For games already in the L3/L4 cache: look up directly.
For new/upcoming games: compute on-the-fly using rolling stats from TeamBox.

The features match exactly what build_l3l4_cache.py produces, using the same
formulas and normalization stats.

Usage:
    from src.pipeline.team_features import TeamFeatureComputer
    tfc = TeamFeatureComputer()
    features = tfc.compute_for_games(["0022500900"])
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

from src.database import DB_PATH, get_db
from src.phase5.travel import compute_travel_features

logger = logging.getLogger(__name__)

L3L4_CACHE_DIR = PROJECT_ROOT / "data" / "l3l4_cache"

# Altitude map (must match build_l3l4_cache.py exactly)
ALTITUDE_MAP = {
    "DEN": 5280,
    "UTA": 4226,
    "SLC": 4226,
    "PHX": 1086,
    "OKC": 1201,
    "ATL": 1050,
    "SAS": 650,
    "MEM": 337,
    "DAL": 430,
    "HOU": 80,
    "MIN": 830,
    "MIL": 617,
    "IND": 715,
    "CLE": 653,
    "DET": 600,
    "CHI": 594,
    "CHA": 751,
    "NOP": 7,
    "POR": 50,
    "SAC": 30,
    "LAL": 233,
    "LAC": 233,
    "GSW": 0,
    "BOS": 141,
    "NYK": 33,
    "BKN": 33,
    "PHI": 39,
    "TOR": 249,
    "WAS": 0,
    "ORL": 82,
    "MIA": 6,
}

# Rolling feature defaults (league average, matching build_l3l4_cache.py)
ROLLING_DEFAULTS = {
    "eFG_off": 0.50,
    "TOV_off": 0.13,
    "ORB_off": 0.25,
    "FTR_off": 0.27,
    "eFG_def": 0.50,
    "TOV_def": 0.13,
    "ORB_def": 0.25,
    "FTR_def": 0.27,
    "ORtg": 110.0,
    "DRtg": 110.0,
    "NetRtg": 0.0,
    "Pace": 95.0,
}
DEFENSE_DEFAULTS = {
    "opp_3PA_rate": 0.35,
    "opp_paint_pts_rate": 0.40,
    "steal_rate": 0.08,
    "block_rate": 0.05,
}


class TeamFeatureComputer:
    """Compute L3 team features and L4 game context for Phase 5 prediction."""

    def __init__(
        self, db_path: str | Path = DB_PATH, cache_dir: str | Path = L3L4_CACHE_DIR
    ):
        self.db_path = str(db_path)
        self.cache_dir = Path(cache_dir)

        # Load normalization stats from cache
        meta_path = self.cache_dir / "metadata.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            norm = meta["normalization"]
            self.l3_mean = np.array(norm["l3_mean"], dtype=np.float32)
            self.l3_std = np.array(norm["l3_std"], dtype=np.float32)
            self.l4_mean = np.array(norm["l4_mean"], dtype=np.float32)
            self.l4_std = np.array(norm["l4_std"], dtype=np.float32)
        else:
            logger.warning(
                "L3/L4 cache metadata not found, using identity normalization"
            )
            self.l3_mean = np.zeros(34, dtype=np.float32)
            self.l3_std = np.ones(34, dtype=np.float32)
            self.l4_mean = np.zeros(14, dtype=np.float32)
            self.l4_std = np.ones(14, dtype=np.float32)

        # Load cached features if available
        self._cache_game_ids = None
        self._cache_l3 = None
        self._cache_l4 = None
        self._cache_index = None
        self._load_cache()

    def _load_cache(self):
        """Load the pre-built L3/L4 cache for fast historical lookups."""
        l3_path = self.cache_dir / "team_features.npz"
        l4_path = self.cache_dir / "game_context.npz"

        if l3_path.exists() and l4_path.exists():
            l3_data = np.load(str(l3_path), allow_pickle=True)
            l4_data = np.load(str(l4_path), allow_pickle=True)
            self._cache_game_ids = l3_data["game_ids"]
            self._cache_l3 = l3_data["features"]
            self._cache_l4 = l4_data["features"]
            # Build index for O(1) lookup
            self._cache_index = {gid: i for i, gid in enumerate(self._cache_game_ids)}
            logger.debug(f"Loaded L3/L4 cache: {len(self._cache_index)} games")

    def compute_for_games(self, game_ids: list[str]) -> dict[str, dict]:
        """
        Compute team features and game context for the given games.

        Returns {game_id: {
            "home_team_features": np.array(34,),  # raw, unnormalized
            "away_team_features": np.array(34,),
            "game_context": np.array(14,),         # raw, unnormalized
            "l3_mean": np.array(34,),
            "l3_std": np.array(34,),
            "l4_mean": np.array(14,),
            "l4_std": np.array(14,),
        }}
        """
        results = {}

        # Split into cached vs needs-computation
        cached_ids = []
        compute_ids = []
        for gid in game_ids:
            if self._cache_index and gid in self._cache_index:
                cached_ids.append(gid)
            else:
                compute_ids.append(gid)

        # Look up cached games
        for gid in cached_ids:
            idx = self._cache_index[gid]
            l3 = self._cache_l3[idx]  # (2, 34)
            l4 = self._cache_l4[idx]  # (14,)
            results[gid] = {
                "home_team_features": l3[0],
                "away_team_features": l3[1],
                "game_context": l4,
                "l3_mean": self.l3_mean,
                "l3_std": self.l3_std,
                "l4_mean": self.l4_mean,
                "l4_std": self.l4_std,
            }

        # Compute features for new games
        if compute_ids:
            computed = self._compute_live(compute_ids)
            results.update(computed)

        # Travel dims [8-11] are filled by the Phase B builder at training time,
        # not by the L3/L4 cache; fill them the same way here.
        with get_db(self.db_path) as conn:
            for gid, feats in results.items():
                row = conn.execute(
                    "SELECT home_team, away_team, date_time_utc FROM Games WHERE game_id = ?",
                    (gid,),
                ).fetchone()
                if row is None:
                    continue
                travel = compute_travel_features(conn, gid, row[0], row[1], row[2])
                ctx = np.array(feats["game_context"], dtype=np.float32, copy=True)
                ctx[8] = travel["travel_dist_home"]
                ctx[9] = travel["travel_dist_away"]
                ctx[10] = travel["tz_crossings_home"]
                ctx[11] = travel["tz_crossings_away"]
                feats["game_context"] = ctx

        return results

    def _compute_live(self, game_ids: list[str]) -> dict[str, dict]:
        """Compute L3/L4 features on-the-fly for games not in cache."""
        results = {}

        with get_db(self.db_path) as conn:
            # Load game info
            placeholders = ",".join("?" * len(game_ids))
            games = conn.execute(
                f"""
                SELECT game_id, home_team, away_team, date_time_utc, season, season_type
                FROM Games WHERE game_id IN ({placeholders})
                """,
                game_ids,
            ).fetchall()

            team_id_map = {
                str(r[0]): r[1]
                for r in conn.execute(
                    "SELECT team_id, abbreviation FROM Teams"
                ).fetchall()
            }

            for game in games:
                game_id, home, away, date_utc, season, season_type = game
                game_date = date_utc[:10] if date_utc else ""

                # L3: Rolling team features for home and away
                home_feats = self._compute_team_rolling(
                    conn, home, game_date, season, team_id_map
                )
                away_feats = self._compute_team_rolling(
                    conn, away, game_date, season, team_id_map
                )

                # L4: Game context
                context = self._compute_game_context(
                    conn, game_id, home, away, game_date, season, season_type
                )

                results[game_id] = {
                    "home_team_features": home_feats,
                    "away_team_features": away_feats,
                    "game_context": context,
                    "l3_mean": self.l3_mean,
                    "l3_std": self.l3_std,
                    "l4_mean": self.l4_mean,
                    "l4_std": self.l4_std,
                }

        return results

    def _compute_team_rolling(
        self,
        conn,
        team_abbr: str,
        before_date: str,
        season: str,
        team_id_map: dict[str, str],
    ) -> np.ndarray:
        """
        Compute 34-d rolling team features for a team before a given date.

        Uses the same formulas as build_l3l4_cache.py.

        Optimized: splits the old TeamBox self-join into two simpler queries:
        1. Find the team's last 15 game_ids from Games (uses indexed columns).
        2. Load TeamBox rows for those game_ids for the team and opponents separately.
        """
        features = np.zeros(34, dtype=np.float32)

        # Get team_id
        team_id = None
        for tid, abbr in team_id_map.items():
            if abbr == team_abbr:
                team_id = tid
                break

        if team_id is None:
            logger.warning(f"Team {team_abbr} not found in Teams table")
            return features

        # Step 1: Find last 15 game_ids for this team (indexed on home_team/away_team)
        game_rows = conn.execute(
            """
            SELECT game_id, date_time_utc FROM Games
            WHERE (home_team = ? OR away_team = ?)
            AND status = 3
            AND date_time_utc < ?
            AND season_type IN ('Regular Season', 'Post Season')
            ORDER BY date_time_utc DESC
            LIMIT 15
            """,
            (team_abbr, team_abbr, before_date),
        ).fetchall()

        if not game_rows:
            # Fill with defaults
            for i, feat in enumerate(ROLLING_DEFAULTS.values()):
                features[i] = feat
                features[i + 12] = feat
            return features

        game_ids = [r[0] for r in game_rows]
        game_dates = {r[0]: r[1] for r in game_rows}
        placeholders = ",".join("?" * len(game_ids))
        int_team_id = int(team_id)

        # Step 2a: Load TeamBox stats for this team in those games
        team_rows = conn.execute(
            f"""
            SELECT game_id, pts, fga, fgm, fg3a, fg3m, fta, ftm, tov, stl, blk
            FROM TeamBox
            WHERE team_id = ? AND game_id IN ({placeholders})
            """,
            [int_team_id] + game_ids,
        ).fetchall()
        team_stats = {r[0]: r[1:] for r in team_rows}

        # Step 2b: Load opponent TeamBox stats for those games
        opp_rows = conn.execute(
            f"""
            SELECT game_id, pts, fga, fgm, fg3a, fg3m, fta, ftm, tov
            FROM TeamBox
            WHERE team_id != ? AND game_id IN ({placeholders})
            """,
            [int_team_id] + game_ids,
        ).fetchall()
        opp_stats = {r[0]: r[1:] for r in opp_rows}

        # Compute per-game metrics (most recent first, matching game_rows order)
        metrics = []
        for gid in game_ids:
            if gid not in team_stats or gid not in opp_stats:
                continue

            pts, fga, fgm, fg3a, fg3m, fta, ftm, tov, stl, blk = team_stats[gid]
            opp_pts, opp_fga, opp_fgm, opp_fg3a, opp_fg3m, opp_fta, opp_ftm, opp_tov = (
                opp_stats[gid]
            )

            # Possessions (simplified — no OREB data in this query)
            poss = max(fga + tov + 0.44 * fta, 1)
            opp_poss = max(opp_fga + opp_tov + 0.44 * opp_fta, 1)

            m = {
                "eFG_off": (fgm + 0.5 * fg3m) / max(fga, 1),
                "TOV_off": tov / max(fga + 0.44 * fta + tov, 1),
                "ORB_off": 0.25,  # default — would need PlayerBox OREB for accurate value
                "FTR_off": fta / max(fga, 1),
                "eFG_def": (opp_fgm + 0.5 * opp_fg3m) / max(opp_fga, 1),
                "TOV_def": opp_tov / max(opp_fga + 0.44 * opp_fta + opp_tov, 1),
                "ORB_def": 0.25,  # default
                "FTR_def": opp_fta / max(opp_fga, 1),
                "ORtg": 100.0 * pts / poss,
                "DRtg": 100.0 * opp_pts / opp_poss,
                "NetRtg": 100.0 * pts / poss - 100.0 * opp_pts / opp_poss,
                "Pace": poss,
                "opp_3PA_rate": opp_fg3a / max(opp_fga, 1),
                "opp_paint_pts_rate": 2 * max(opp_fgm - opp_fg3m, 0) / max(opp_pts, 1),
                "steal_rate": stl / opp_poss,
                "block_rate": blk / max(opp_fga, 1),
            }
            metrics.append(m)

        if not metrics:
            for i, feat in enumerate(ROLLING_DEFAULTS.values()):
                features[i] = feat
                features[i + 12] = feat
            return features

        # Compute rolling averages (5-game and 15-game)
        rolling_cols = list(ROLLING_DEFAULTS.keys())
        for i, col in enumerate(rolling_cols):
            vals = [m[col] for m in metrics]
            # 5-game (use up to 5 most recent)
            features[i] = np.mean(vals[:5]) if vals else ROLLING_DEFAULTS[col]
            # 15-game (use all available up to 15)
            features[i + 12] = np.mean(vals) if vals else ROLLING_DEFAULTS[col]

        # Coaching features (indices 24-27) — placeholder zeros
        # These would require coach lookup; using 0 matches training default
        features[24:28] = 0.0

        # Defensive scheme proxies (indices 28-31)
        defense_cols = [
            "opp_3PA_rate",
            "opp_paint_pts_rate",
            "steal_rate",
            "block_rate",
        ]
        for j, col in enumerate(defense_cols):
            vals = [m[col] for m in metrics]
            features[28 + j] = np.mean(vals) if vals else DEFENSE_DEFAULTS[col]

        # Roster continuity (index 32) — placeholder 0.5
        features[32] = 0.5

        # Multi-year trend (index 33) — placeholder 0.5
        features[33] = 0.5

        return features

    def _compute_game_context(
        self,
        conn,
        game_id: str,
        home: str,
        away: str,
        game_date: str,
        season: str,
        season_type: str,
    ) -> np.ndarray:
        """Compute 14-d game context features."""
        ctx = np.zeros(14, dtype=np.float32)

        # [0] home_flag = 1.0 (always from home perspective)
        ctx[0] = 1.0

        # [1] arena_altitude
        ctx[1] = float(ALTITUDE_MAP.get(home, 0))

        # [2-7] Rest, B2B, games-in-7d for home and away
        for i, team in enumerate([home, away]):
            offset = 2 + i  # rest: 2,3; b2b: 4,5; g7d: 6,7

            # Find last game date for this team
            row = conn.execute(
                """
                SELECT date_time_utc FROM Games
                WHERE (home_team = ? OR away_team = ?) AND status = 3
                AND date_time_utc < ?
                ORDER BY date_time_utc DESC LIMIT 1
                """,
                (team, team, game_date),
            ).fetchone()

            if row:
                last_date_str = row[0][:10]
                try:
                    last_date = datetime.strptime(last_date_str, "%Y-%m-%d")
                    curr_date = datetime.strptime(game_date, "%Y-%m-%d")
                    rest = min((curr_date - last_date).days, 7)
                    ctx[offset] = float(rest)  # rest_days
                    ctx[offset + 2] = 1.0 if rest <= 1 else 0.0  # is_b2b
                except (ValueError, TypeError):
                    ctx[offset] = 3.0
                    ctx[offset + 2] = 0.0
            else:
                ctx[offset] = 3.0
                ctx[offset + 2] = 0.0

            # Games in last 7 days
            try:
                curr_date = datetime.strptime(game_date, "%Y-%m-%d")
                week_ago = (curr_date - timedelta(days=7)).strftime("%Y-%m-%d")
                row7 = conn.execute(
                    """
                    SELECT COUNT(*) FROM Games
                    WHERE (home_team = ? OR away_team = ?) AND status = 3
                    AND date_time_utc >= ? AND date_time_utc < ?
                    """,
                    (team, team, week_ago, game_date),
                ).fetchone()
                ctx[offset + 4] = float(row7[0]) if row7 else 1.0
            except (ValueError, TypeError):
                ctx[offset + 4] = 1.0

        # [8-9] travel_dist (placeholder 0)
        # [10-11] tz_crossings (placeholder 0)

        # [12] season_progress
        row = conn.execute(
            """
            SELECT COUNT(*) FROM Games
            WHERE (home_team = ? OR away_team = ?) AND status = 3
            AND season = ? AND date_time_utc <= ?
            """,
            (home, home, season, game_date),
        ).fetchone()
        n_games = row[0] if row else 41
        ctx[12] = min(n_games / 82.0, 1.0)

        # [13] is_playoffs
        ctx[13] = 1.0 if season_type and "Post" in season_type else 0.0

        return ctx
