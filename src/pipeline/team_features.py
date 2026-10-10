"""
Team Feature Computer: Phase5 L3 team features (34-d), L4 game context (14-d)
and coach inputs for any games, computed as of each game's tip-off.

This is the single implementation: the training cache builder
(scripts/build_phase5_cache.py) and live predictions both call it, so
training and inference features cannot drift apart.

Usage:
    from src.pipeline.team_features import TeamFeatureComputer
    tfc = TeamFeatureComputer()
    features = tfc.compute_for_games(["0022500900"])
"""

from __future__ import annotations

import sqlite3
import logging
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from src.database import DB_PATH, get_db
from src.phase5.arena_data import HISTORICAL_TO_MODERN
from src.phase5.travel import compute_travel_features

logger = logging.getLogger(__name__)


# Arena altitude (feet) by home team
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

# Rolling feature defaults (league average) when a team has no games yet
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
    """Compute L3 team features, L4 game context and coach inputs for Phase5."""

    def __init__(self, db_path: str | Path = DB_PATH):
        self.db_path = str(db_path)

    def compute_for_games(
        self, game_ids: list[str], rosters: dict | None = None
    ) -> dict[str, dict]:
        """
        Compute features for the given games from data before each tip-off.

        rosters: projected rosters ({game_id: {"home_players": [...],
        "away_players": [...]}}, as from RosterAssembler), used to estimate
        roster continuity before a team's first game of the season.

        Returns {game_id: {
            "home_team_features": np.array(34,),  # raw, unnormalized
            "away_team_features": np.array(34,),
            "game_context": np.array(14,),         # raw, unnormalized
            "home_coach": (coach_id or None, career games coached before),
            "away_coach": (coach_id or None, career games coached before),
        }}
        """
        results = {}
        if not game_ids:
            return results

        with get_db(self.db_path) as conn:
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
            # One replay over all games covers every request in the batch
            coaching = _compute_coaching_features(
                conn,
                [(team, g[4], g[3] or "") for g in games for team in g[1:3]],
            )

            for game_id, home, away, date_utc, season, season_type in games:
                # Full tip time, so a team's earlier game on the same UTC date
                # counts as prior
                date_utc = date_utc or ""
                context = self._compute_game_context(
                    conn, game_id, home, away, date_utc, season, season_type
                )
                travel = compute_travel_features(conn, game_id, home, away, date_utc)
                context[8:12] = [
                    travel["travel_dist_home"],
                    travel["travel_dist_away"],
                    travel["tz_crossings_home"],
                    travel["tz_crossings_away"],
                ]
                entry = {"game_context": context}
                for side, team in (("home", home), ("away", away)):
                    roster = (rosters or {}).get(game_id, {}).get(f"{side}_players")
                    feats = self._compute_team_rolling(
                        conn, team, date_utc, season, team_id_map, roster
                    )
                    coach_feats, coach_id, coach_games = coaching[
                        (team, season, date_utc)
                    ]
                    feats[24:28] = coach_feats
                    entry[f"{side}_team_features"] = feats
                    entry[f"{side}_coach"] = (coach_id, coach_games)
                results[game_id] = entry

        return results

    def _compute_team_rolling(
        self,
        conn,
        team_abbr: str,
        before_date: str,
        season: str,
        team_id_map: dict[str, str],
        roster: list[int] | None = None,
    ) -> np.ndarray:
        """
        Compute 34-d rolling team features for a team before a given date.

        Coaching features (indices 24-27) are left at zero and filled in
        compute_for_games.
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

        # Step 1: Find the team's recent game_ids (extra rows in case some lack
        # box scores; the builder only rolls over games with both TeamBox rows)
        game_rows = conn.execute(
            """
            SELECT game_id, home_team, away_team FROM Games
            WHERE (home_team = ? OR away_team = ?)
            AND status = 3
            AND date_time_utc < ?
            AND season_type IN ('Regular Season', 'Post Season')
            ORDER BY date_time_utc DESC
            LIMIT 20
            """,
            (team_abbr, team_abbr, before_date),
        ).fetchall()
        game_ids = [r[0] for r in game_rows]
        abbr_to_tid = {abbr: tid for tid, abbr in team_id_map.items()}
        team_ids = {team_id} | {
            abbr_to_tid[a]
            for r in game_rows
            for a in r[1:]
            if a in abbr_to_tid
        }

        metrics = []
        if game_ids:
            placeholders = ",".join("?" * len(game_ids))

            # Step 2: TeamBox stats for both teams in those games (filter on
            # team_id too so the (team_id, game_id) index is used)
            tid_placeholders = ",".join("?" * len(team_ids))
            box = defaultdict(dict)  # game_id -> {"team": row, "opp": row}
            for row in conn.execute(
                f"""
                SELECT game_id, team_id, pts, fga, fgm, fg3a, fg3m, fta, tov, stl, blk
                FROM TeamBox
                WHERE team_id IN ({tid_placeholders}) AND game_id IN ({placeholders})
                """,
                [*team_ids, *game_ids],
            ):
                box[row[0]]["team" if str(row[1]) == team_id else "opp"] = row[2:]

            # Step 3: OREB/DREB aggregated from PlayerBox (TeamBox lacks them)
            reb = {}
            for gid, tid, oreb, dreb in conn.execute(
                f"""
                SELECT game_id, team_id, SUM(oreb), SUM(dreb) FROM PlayerBox
                WHERE game_id IN ({placeholders}) AND min > 0
                GROUP BY game_id, team_id
                """,
                game_ids,
            ):
                reb[(gid, "team" if str(tid) == team_id else "opp")] = (
                    oreb or 0,
                    dreb or 0,
                )

            # Per-game metrics (most recent first)
            for gid in game_ids:
                if "team" not in box[gid] or "opp" not in box[gid]:
                    continue
                pts, fga, fgm, fg3a, fg3m, fta, tov, stl, blk = box[gid]["team"]
                opp_pts, opp_fga, opp_fgm, opp_fg3a, opp_fg3m, opp_fta, opp_tov, _, _ = (
                    box[gid]["opp"]
                )
                oreb, dreb = reb.get((gid, "team"), (0, 0))
                opp_oreb, opp_dreb = reb.get((gid, "opp"), (0, 0))

                poss = max(fga - oreb + tov + 0.44 * fta, 1)
                opp_poss = max(opp_fga - opp_oreb + opp_tov + 0.44 * opp_fta, 1)
                ortg = 100.0 * pts / poss
                drtg = 100.0 * opp_pts / opp_poss

                metrics.append(
                    {
                        "eFG_off": (fgm + 0.5 * fg3m) / max(fga, 1),
                        "TOV_off": tov / max(fga + 0.44 * fta + tov, 1),
                        "ORB_off": oreb / max(oreb + opp_dreb, 1),
                        "FTR_off": fta / max(fga, 1),
                        "eFG_def": (opp_fgm + 0.5 * opp_fg3m) / max(opp_fga, 1),
                        "TOV_def": opp_tov
                        / max(opp_fga + 0.44 * opp_fta + opp_tov, 1),
                        "ORB_def": opp_oreb / max(opp_oreb + dreb, 1),
                        "FTR_def": opp_fta / max(opp_fga, 1),
                        "ORtg": ortg,
                        "DRtg": drtg,
                        "NetRtg": ortg - drtg,
                        "Pace": poss,
                        "opp_3PA_rate": opp_fg3a / max(opp_fga, 1),
                        "opp_paint_pts_rate": 2
                        * max(opp_fgm - opp_fg3m, 0)
                        / max(opp_pts, 1),
                        "steal_rate": stl / opp_poss,
                        "block_rate": blk / max(opp_fga, 1),
                    }
                )
                if len(metrics) == 15:
                    break

        # Rolling averages (5-game and 15-game); league-average defaults when
        # the team has no history
        for i, (col, default) in enumerate(ROLLING_DEFAULTS.items()):
            vals = [m[col] for m in metrics]
            features[i] = np.mean(vals[:5]) if vals else default
            features[i + 12] = np.mean(vals) if vals else default

        # Defensive scheme proxies (indices 28-31), 15-game only
        for j, (col, default) in enumerate(DEFENSE_DEFAULTS.items()):
            vals = [m[col] for m in metrics]
            features[28 + j] = np.mean(vals) if vals else default

        # Roster continuity (index 32) and multi-year trend (index 33)
        features[32] = _roster_continuity(
            conn, team_abbr, team_id, season, before_date, roster
        )
        features[33] = _multi_year_trend(conn, team_abbr, team_id, season)

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
                AND season_type IN ('Regular Season', 'Post Season')
                ORDER BY date_time_utc DESC LIMIT 1
                """,
                (team, team, game_date),
            ).fetchone()

            if row:
                last_date_str = row[0][:10]
                try:
                    last_date = datetime.strptime(last_date_str, "%Y-%m-%d")
                    curr_date = datetime.strptime(game_date[:10], "%Y-%m-%d")
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
                curr_date = datetime.strptime(game_date[:10], "%Y-%m-%d")
                week_ago = (curr_date - timedelta(days=7)).strftime("%Y-%m-%d")
                row7 = conn.execute(
                    """
                    SELECT COUNT(*) FROM Games
                    WHERE (home_team = ? OR away_team = ?) AND status = 3
                    AND date_time_utc >= ? AND date_time_utc < ?
                    AND season_type IN ('Regular Season', 'Post Season')
                    """,
                    (team, team, week_ago, game_date),
                ).fetchone()
                ctx[offset + 4] = float(row7[0]) if row7 else 1.0
            except (ValueError, TypeError):
                ctx[offset + 4] = 1.0

        # [8-11] travel distance / timezone crossings: set in compute_for_games

        # [12] season_progress: the home team's Nth game of the season,
        # counting this one
        row = conn.execute(
            """
            SELECT COUNT(*) FROM Games
            WHERE (home_team = ? OR away_team = ?) AND status = 3
            AND season = ? AND date_time_utc < ?
            AND season_type IN ('Regular Season', 'Post Season')
            """,
            (home, home, season, game_date),
        ).fetchone()
        n_games = row[0] + 1 if row else 41
        ctx[12] = min(n_games / 82.0, 1.0)

        # [13] is_playoffs
        ctx[13] = 1.0 if season_type and "Post" in season_type else 0.0

        return ctx


def _roster_continuity(
    conn,
    team_abbr: str,
    team_id: str,
    season: str,
    before_date: str,
    roster: list[int] | None = None,
) -> float:
    """Fraction of this season's minutes so far played by players who were on
    the team last season. Before the team has played, estimated from the
    projected roster (0.5 without one)."""
    start = int(season[:4])
    prev_season = f"{start - 1}-{start}"
    rows = conn.execute(
        """
        SELECT pb.player_id, SUM(pb.min) FROM Games g
        JOIN PlayerBox pb ON pb.game_id = g.game_id
        WHERE (g.home_team = ? OR g.away_team = ?) AND g.status = 3
        AND g.season_type = 'Regular Season' AND g.season = ?
        AND g.date_time_utc < ? AND pb.team_id = ? AND pb.min > 0
        GROUP BY pb.player_id
        """,
        (team_abbr, team_abbr, season, before_date, team_id),
    ).fetchall()
    total_min = sum(m for _, m in rows)
    prev_players = {
        r[0]
        for r in conn.execute(
            """
            SELECT DISTINCT pb.player_id FROM Games g
            JOIN PlayerBox pb ON pb.game_id = g.game_id
            WHERE (g.home_team = ? OR g.away_team = ?) AND g.status = 3
            AND g.season_type = 'Regular Season'
            AND g.season = ? AND pb.team_id = ? AND pb.min > 0
            """,
            (team_abbr, team_abbr, prev_season, team_id),
        )
    }
    if total_min == 0:
        if not roster:
            return 0.5
        return _projected_continuity(conn, roster, prev_season, prev_players)
    return sum(m for pid, m in rows if pid in prev_players) / total_min


# Before a team's first game: weight the projected roster by last season's minutes,
# with newcomers floored and the roster trimmed to a rotation. On 2024-25/2025-26
# openers this tracked the full-season value with MAE 0.07 vs 0.19 for 0.5.
CONTINUITY_NEWCOMER_MINUTES = 1000.0
CONTINUITY_ROSTER_SIZE = 13


def _projected_continuity(
    conn, roster: list[int], prev_season: str, prev_players: set
) -> float:
    placeholders = ",".join("?" * len(roster))
    prev_min = dict(
        conn.execute(
            f"""
            SELECT pb.player_id, SUM(pb.min) FROM Games g
            JOIN PlayerBox pb ON pb.game_id = g.game_id
            WHERE g.season = ? AND g.season_type = 'Regular Season' AND g.status = 3
            AND pb.min > 0 AND pb.player_id IN ({placeholders})
            GROUP BY pb.player_id
            """,
            (prev_season, *roster),
        ).fetchall()
    )
    weights = sorted(
        (
            (max(prev_min.get(pid, 0.0), CONTINUITY_NEWCOMER_MINUTES), pid)
            for pid in roster
        ),
        reverse=True,
    )[:CONTINUITY_ROSTER_SIZE]
    total = sum(w for w, _ in weights)
    return sum(w for w, pid in weights if pid in prev_players) / total


def _multi_year_trend(conn, team_abbr: str, team_id: str, season: str) -> float:
    """Mean regular-season win% over the team's previous (up to) 3 seasons.
    0.5 with no history."""
    rows = conn.execute(
        """
        SELECT g.season, AVG(CASE WHEN t.pts > t.pts_allowed THEN 1.0 ELSE 0.0 END)
        FROM TeamBox t
        JOIN Games g ON g.game_id = t.game_id
        WHERE t.team_id = ? AND g.status = 3
        AND g.season_type = 'Regular Season' AND g.season < ?
        GROUP BY g.season ORDER BY g.season DESC LIMIT 3
        """,
        (team_id, season),
    ).fetchall()
    return float(np.mean([r[1] for r in rows])) if rows else 0.5


# Coaching feature normalizers
COACH_TENURE_NORM = 500.0
COACH_GAMES_SINCE_NORM = 200.0


def _compute_coaching_features(
    conn, requests: list[tuple[str, str, str]]
) -> dict[tuple[str, str, str], tuple[np.ndarray, str | None, int]]:
    """
    Coaching state for (team_abbr, season, date) requests: the L3 features
    (indices 24-27), the head coach id, and that coach's career games before
    the date (the L3 coach-embedding inputs).

    Replays every completed game since 2001-02 and reads each team's state
    just before the requested date. When the Coaches table
    has no head coach for the team's season yet (e.g. a new season before the
    table is refreshed), the team's most recent head coach is carried forward.
    """
    team_abbr = {
        str(r[0]): r[1] for r in conn.execute("SELECT team_id, abbreviation FROM Teams")
    }
    abbr_to_tid = {abbr: tid for tid, abbr in team_abbr.items()}
    for hist, modern in HISTORICAL_TO_MODERN.items():
        if modern in abbr_to_tid:
            abbr_to_tid.setdefault(hist, abbr_to_tid[modern])

    # (abbr, "2001-2002") -> coach_id; first head coach listed per team-season,
    # also keyed by the team's historical abbreviations
    coach_map: dict[tuple[str, str], str] = {}
    try:
        coach_rows = conn.execute(
            """
            SELECT team_id, season, coach_id FROM Coaches
            WHERE is_head_coach = 1 ORDER BY team_id, season, ROWID
            """
        ).fetchall()
    except sqlite3.OperationalError:
        # No Coaches table (e.g. starter DB): training's no-coach default
        logger.warning("Coaches table missing; coaching features set to zero")
        coach_rows = []
    for team_id, c_season, coach_id in coach_rows:
        abbr = team_abbr.get(str(team_id), "???")
        year = int(c_season[:4])
        season = f"{year}-{year + 1}"
        if (abbr, season) in coach_map:
            continue
        coach_map[(abbr, season)] = coach_id
        for hist, modern in HISTORICAL_TO_MODERN.items():
            if modern == abbr:
                coach_map[(hist, season)] = coach_id

    # Seasons after the newest Coaches season (table not yet refreshed) carry
    # the team's latest head coach forward so state keeps advancing; gaps in
    # covered seasons stay coachless, as they were in training.
    latest_coach_season = max((s for _, s in coach_map), default="")

    def coach_for(abbr: str, team: str, season: str) -> str | None:
        coach = coach_map.get((abbr, season)) or coach_map.get((team, season))
        if coach is None and season > latest_coach_season:
            earlier = [s for (t, s) in coach_map if t == team]
            if earlier:
                coach = coach_map[(team, max(earlier))]
        return coach

    team_tenure = defaultdict(lambda: defaultdict(int))  # coach -> team -> games
    career = defaultdict(lambda: [0, 0])  # coach -> [wins, games]
    season_coach: dict[str, str] = {}  # team -> coach of last processed season
    last_season: dict[str, str] = {}
    was_new: dict[str, bool] = {}
    games_since = defaultdict(int)

    def features_for(team: str, season: str):
        coach = coach_for(team, team, season)
        if coach is None:
            return np.zeros(4, dtype=np.float32), None, 0
        if last_season.get(team) != season:
            # First game of a new season: same transition as the replay
            prev = season_coach.get(team)
            is_new = prev is not None and prev != coach
            since = 0 if is_new else games_since[team]
        else:
            is_new = was_new.get(team, False)
            since = games_since[team]
        wins, n = career[coach]
        feats = np.array(
            [
                team_tenure[coach][team] / COACH_TENURE_NORM,
                wins / n if n > 0 else 0.5,
                1.0 if is_new else 0.0,
                since / COACH_GAMES_SINCE_NORM,
            ],
            dtype=np.float32,
        )
        return feats, coach, n

    scores = {
        (gid, str(tid)): pts
        for gid, tid, pts in conn.execute(
            """
            SELECT tb.game_id, tb.team_id, tb.pts FROM TeamBox tb
            JOIN Games g ON tb.game_id = g.game_id
            WHERE g.season >= '2001-2002' AND g.status = 3
            AND g.season_type IN ('Regular Season', 'Post Season')
            """
        )
    }
    games = conn.execute(
        """
        SELECT game_id, date_time_utc, home_team, away_team, season FROM Games
        WHERE season >= '2001-2002' AND status = 3
        AND season_type IN ('Regular Season', 'Post Season')
        ORDER BY date_time_utc, game_id
        """
    ).fetchall()

    pending = sorted(set(requests), key=lambda r: r[2])
    results = {}
    i = 0
    for game_id, date_utc, home, away, season in games:
        while i < len(pending) and pending[i][2] <= date_utc:
            team, req_season, _ = pending[i]
            results[pending[i]] = features_for(team, req_season)
            i += 1
        home_pts = scores.get((game_id, abbr_to_tid.get(home)))
        away_pts = scores.get((game_id, abbr_to_tid.get(away)))
        for abbr, pts, opp_pts in ((home, home_pts, away_pts), (away, away_pts, home_pts)):
            team = HISTORICAL_TO_MODERN.get(abbr, abbr)
            coach = coach_for(abbr, team, season)
            if coach is None:
                continue
            if last_season.get(team) != season:
                prev = season_coach.get(team)
                was_new[team] = prev is not None and prev != coach
                season_coach[team] = coach
                last_season[team] = season
                if was_new[team]:
                    games_since[team] = 0
            team_tenure[coach][team] += 1
            career[coach][1] += 1
            if pts is not None and opp_pts is not None and pts > opp_pts:
                career[coach][0] += 1
            games_since[team] += 1
    for req in pending[i:]:
        results[req] = features_for(req[0], req[1])
    return results
