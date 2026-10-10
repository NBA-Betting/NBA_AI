"""
Phase 5 NKE-H cache builder.

Assembles per-player career sequences from PlayerBox, PBPPlayerGameStatsV2,
DarkoDPM, PlayerAttributes, and DataballrWingspan into numpy arrays for
efficient training.

Output structure:
    data/phase5_cache/
        players/{person_id}.npz   — per-player career data
        profiles.npz              — static player profiles for prior network
        archetypes.npz            — K=10 archetype centroids
        metadata.json             — feature names, normalization stats, splits
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.database import get_db

logger = logging.getLogger(__name__)

CACHE_DIR = PROJECT_ROOT / "data" / "phase5_cache"
PLAYERS_DIR = CACHE_DIR / "players"

# ---------------------------------------------------------------------------
# Feature definitions
# ---------------------------------------------------------------------------

# 16 standard box stats from PlayerBox (same order as Phase 3)
BOX_STAT_COLUMNS = [
    "min",
    "pts",
    "oreb",
    "dreb",
    "ast",
    "stl",
    "blk",
    "tov",
    "pf",
    "fga",
    "fgm",
    "fg3a",
    "fg3m",
    "fta",
    "ftm",
    "plus_minus",
]

# Selected PBP stats (56 features for player ability modeling, incl. expanded defense)
PBP_STAT_COLUMNS = [
    # Seconds and possessions (4)
    "seconds_played_off",
    "seconds_played_def",
    "off_poss",
    "def_poss",
    # Shot zones FGM/FGA (10)
    "at_rim_fgm",
    "at_rim_fga",
    "short_mid_fgm",
    "short_mid_fga",
    "long_mid_fgm",
    "long_mid_fga",
    "corner3_fgm",
    "corner3_fga",
    "arc3_fgm",
    "arc3_fga",
    # Assisted/unassisted (4 — aggregate 2pt and 3pt)
    "assisted_at_rim",
    "assisted_short_mid",
    "unassisted_at_rim",
    "unassisted_short_mid",
    # Assists given by zone (5)
    "ast_at_rim",
    "ast_short_mid",
    "ast_long_mid",
    "ast_corner3",
    "ast_arc3",
    # Blocks by zone (2 — aggregate)
    "block_at_rim",
    "block_short_mid",
    # Rebounds (6)
    "oreb_at_rim",
    "oreb_short_mid",
    "dreb_at_rim",
    "dreb_short_mid",
    "dreb_long_mid",
    "dreb_arc3",
    # Other (7)
    "self_oreb",
    "on_floor_oreb",
    "putbacks",
    "bad_pass_turnovers",
    "lost_ball_turnovers",
    "fts_made",
    "fts_missed",
    # Steals by type (2 — steals forced by the player)
    "bad_pass_steals",
    "lost_ball_steals",
    # Shots blocked (5 — player's shots that were blocked by defender)
    "blocked_at_rim",
    "blocked_short_mid",
    "blocked_long_mid",
    "blocked_corner3",
    "blocked_arc3",
    # Additional blocks made by zone (3 — complements block_at_rim/short_mid)
    "block_long_mid",
    "block_corner3",
    "block_arc3",
    # Additional defensive rebounds (1 — corner3 zone)
    "dreb_corner3",
    # Defensive rebound opportunities (1 — context for rebound rate)
    "dreb_opportunities",
    # Fouls (6 — expanded from 4, adding loose_ball and offensive fouls)
    "personal_fouls",
    "shooting_fouls",
    "loose_ball_fouls",
    "offensive_fouls",
    "shooting_fouls_drawn",
    "charge_fouls_drawn",
]

# Context features computed per game (12)
CONTEXT_COLUMNS = [
    "age_at_game",  # years (float)
    "rest_days",  # days since last game for this player
    "home_flag",  # 1.0 if home, 0.0 if away
    "opponent_drtg",  # opponent defensive rating (rolling season avg)
    "team_pace",  # team pace (rolling season avg)
    "minutes_share",  # player minutes / 240 (total team minutes)
    "season_progress",  # 0.0 = first game, 1.0 = last game of season
    "career_games_played",  # running count
    "days_since_last_game",  # calendar days (captures offseason gaps)
    "opponent_pace",  # opponent pace (rolling season avg)
    "team_ortg",  # team offensive rating (rolling season avg)
    "altitude_ft",  # venue altitude in feet (Denver effect)
]

# DPM training targets (3)
DPM_TARGET_COLUMNS = ["o_dpm", "d_dpm", "dpm"]

# Player profile features for prior network
PROFILE_COLUMNS = [
    "height_inches",
    "weight",
    "draft_pick",
    "undrafted",
    "birth_year",
    "wingspan_inches",
    "pos_g",
    "pos_f",
    "pos_c",  # one-hot position
]

# Season-level RAPM target (auxiliary, slower-moving)
RAPM_TARGET_COLUMNS = ["off_rapm", "def_rapm"]

# Altitude lookup (feet above sea level for NBA arenas)
# Historical team abbreviation → modern abbreviation
HISTORICAL_TO_MODERN = {
    "NJN": "BKN",
    "SEA": "OKC",
    "NOH": "NOP",
    "NOK": "NOP",
    "CHH": "CHA",
    "VAN": "MEM",
    "WSB": "WAS",
    "SDC": "LAC",
    "KCK": "SAC",
    "BUF": "LAC",
    "NOJ": "NOP",
    "CLT": "CHA",
}

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
    "CLT": 751,
    "NOP": 7,
    "NOH": 7,
    "POR": 50,
    "SAC": 30,
    "LAL": 233,
    "LAC": 233,
    "GSW": 0,
    "BOS": 141,
    "NYK": 33,
    "BKN": 33,
    "NJN": 33,
    "PHI": 39,
    "TOR": 249,
    "WAS": 0,
    "ORL": 82,
    "MIA": 6,
    "SEA": 0,
    "VAN": 0,
    "NOK": 7,
    "CHH": 751,
}


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


# Game types in L1 sequences, for training and live updates alike. The current
# l1.pt and the Phase B/C data it fed were built from regular-season games only.
L1_SEASON_TYPES = ("Regular Season",)

# Profile for players without a PlayerAttributes row (e.g. call-ups)
DEFAULT_PROFILE = {
    "height_inches": 78,
    "weight": 215,
    "draft_pick": 60,
    "undrafted": 1.0,
    "birth_year": 2000,  # debut season; see _load_player_profile
    "wingspan_inches": 78,
    "pos_g": 0.0,
    "pos_f": 1.0,
    "pos_c": 0.0,
}


def _load_player_games(conn, person_id: int) -> list[dict]:
    """Load a player's L1_SEASON_TYPES games, chronologically."""
    placeholders = ",".join("?" * len(L1_SEASON_TYPES))
    cursor = conn.execute(
        f"""
        SELECT
            pb.game_id,
            g.date_time_utc,
            g.home_team,
            g.away_team,
            g.season,
            pb.team_id,
            pb.min, pb.pts, pb.oreb, pb.dreb, pb.ast, pb.stl, pb.blk,
            pb.tov, pb.pf, pb.fga, pb.fgm, pb.fg3a, pb.fg3m,
            pb.fta, pb.ftm, pb.plus_minus
        FROM PlayerBox pb
        JOIN Games g ON pb.game_id = g.game_id
        WHERE pb.player_id = ?
        AND g.status = 3
        AND pb.min > 0
        AND g.season_type IN ({placeholders})
        ORDER BY g.date_time_utc
        """,
        (person_id, *L1_SEASON_TYPES),
    )
    columns = [
        "game_id",
        "date_time_utc",
        "home_team",
        "away_team",
        "season",
        "team_id",
    ] + BOX_STAT_COLUMNS
    return [dict(zip(columns, row)) for row in cursor]


def _load_pbp_stats(conn, person_id: int) -> dict[str, dict]:
    """Load PBP enriched stats for a player, keyed by game_id."""
    cols = ", ".join(PBP_STAT_COLUMNS)
    cursor = conn.execute(
        f"SELECT game_id, {cols} FROM PBPPlayerGameStatsV2 WHERE player_id = ?",
        (person_id,),
    )
    result = {}
    for row in cursor:
        game_id = row[0]
        result[game_id] = dict(zip(PBP_STAT_COLUMNS, row[1:]))
    return result


def _load_dpm_targets(conn, person_id: int) -> dict[str, dict]:
    """Load DARKO DPM targets for a player, keyed by game_date."""
    cursor = conn.execute(
        "SELECT game_date, o_dpm, d_dpm, dpm FROM DarkoDPM WHERE person_id = ?",
        (person_id,),
    )
    result = {}
    for row in cursor:
        result[row[0]] = {"o_dpm": row[1], "d_dpm": row[2], "dpm": row[3]}
    return result


def _debut_year(conn, person_id: int) -> int | None:
    """Start year of the player's first season with minutes."""
    row = conn.execute(
        """
        SELECT MIN(CAST(substr(g.season, 1, 4) AS INTEGER)) FROM PlayerBox pb
        JOIN Games g ON g.game_id = pb.game_id
        WHERE pb.player_id = ? AND pb.min > 0 AND g.status = 3
        """,
        (person_id,),
    ).fetchone()
    return row[0] if row else None


def _load_player_profile(conn, person_id: int) -> dict:
    """Load static player profile for prior network.

    "birth_year" is the player's debut season (PlayerAttributes.from_year), so
    context column 0 ("age") is years since debut. That is what the trained L1
    model saw: PlayerAttributes.birth_year held from_year when its cache was
    built (real birth years were filled in later).
    """
    cursor = conn.execute(
        """
        SELECT
            pa.height_inches, pa.weight, pa.draft_number, pa.undrafted,
            pa.from_year, pa.position,
            dw.wingspan_in
        FROM PlayerAttributes pa
        LEFT JOIN DataballrWingspan dw ON pa.person_id = dw.nba_id
        WHERE pa.person_id = ?
        """,
        (person_id,),
    )
    row = cursor.fetchone()
    if not row:
        return {}

    height, weight, draft_pick, undrafted, birth_year, position, wingspan = row

    # Position one-hot (G, F, C)
    pos_g, pos_f, pos_c = 0.0, 0.0, 0.0
    if position:
        pos_upper = position.upper()
        if "G" in pos_upper:
            pos_g = 1.0
        if "F" in pos_upper:
            pos_f = 1.0
        if "C" in pos_upper:
            pos_c = 1.0
        if pos_g == 0.0 and pos_f == 0.0 and pos_c == 0.0:
            pos_f = 1.0  # default to forward

    return {
        "height_inches": height or 78,  # default 6'6"
        "weight": weight or 215,
        "draft_pick": draft_pick or 60,  # undrafted → 60 (last pick equivalent)
        "undrafted": float(undrafted or 0),
        "birth_year": birth_year or _debut_year(conn, person_id) or 2000,
        "wingspan_inches": wingspan
        or (height or 78),  # default to height if no wingspan
        "pos_g": pos_g,
        "pos_f": pos_f,
        "pos_c": pos_c,
    }


def _load_season_rapm(conn, person_id: int) -> dict[str, dict]:
    """Load season-level RAPM from databallr, keyed by year."""
    cursor = conn.execute(
        """
        SELECT year, off_rapm, def_rapm
        FROM DataballrMultiYearRAPM
        WHERE nba_id = ?
        AND playoffs = 0
        """,
        (person_id,),
    )
    result = {}
    for row in cursor:
        year = str(row[0])
        result[year] = {"off_rapm": row[1] or 0.0, "def_rapm": row[2] or 0.0}
    return result


def _load_team_id_to_abbr(conn) -> dict[int, str]:
    """Load team_id → abbreviation mapping from Teams table."""
    cursor = conn.execute("SELECT team_id, abbreviation FROM Teams")
    return {int(row[0]): row[1] for row in cursor}


def _load_team_rolling_stats(conn) -> dict[tuple[str, str], dict]:
    """
    Pre-compute team stats (points, opponent points) per team per season.
    Covers both home and away games. Uses points as proxy for ORTG/DRTG.
    Returns {(team_abbr, season): {ortg, drtg, pace}}.
    """
    # Get team points and opponent points per game, both home and away
    cursor = conn.execute(
        """
        SELECT team_abbr, season, AVG(team_pts) as avg_pts, AVG(opp_pts) as avg_opp_pts
        FROM (
            -- Home games
            SELECT g.home_team as team_abbr, g.season,
                   tb.pts as team_pts, tb_opp.pts as opp_pts
            FROM Games g
            JOIN TeamBox tb ON g.game_id = tb.game_id
            JOIN TeamBox tb_opp ON g.game_id = tb_opp.game_id AND tb_opp.team_id != tb.team_id
            JOIN Teams t ON tb.team_id = t.team_id AND t.abbreviation = g.home_team
            WHERE g.status = 3

            UNION ALL

            -- Away games
            SELECT g.away_team as team_abbr, g.season,
                   tb.pts as team_pts, tb_opp.pts as opp_pts
            FROM Games g
            JOIN TeamBox tb ON g.game_id = tb.game_id
            JOIN TeamBox tb_opp ON g.game_id = tb_opp.game_id AND tb_opp.team_id != tb.team_id
            JOIN Teams t ON tb.team_id = t.team_id AND t.abbreviation = g.away_team
            WHERE g.status = 3
        )
        GROUP BY team_abbr, season
        """,
    )
    result = {}
    for row in cursor:
        team, season, avg_pts, avg_opp_pts = row
        # Approximate ratings: scale points to per-100-possessions
        # NBA average ~100 possessions/game, so raw points ≈ ORTG
        result[(team, season)] = {
            "pace": (avg_pts or 100)
            + (avg_opp_pts or 100),  # total points ≈ pace proxy
            "ortg": avg_pts or 110.0,
            "drtg": avg_opp_pts or 110.0,
        }
    return result


# ---------------------------------------------------------------------------
# Per-player cache building
# ---------------------------------------------------------------------------


def _extract_game_date(date_time_utc: str) -> str:
    """Extract YYYY-MM-DD from datetime string."""
    if not date_time_utc:
        return ""
    return date_time_utc[:10]


def build_player_cache(
    conn,
    person_id: int,
    team_stats: dict,
    team_id_to_abbr: dict[int, str] | None = None,
    min_games: int = 5,
    require_profile: bool = True,
) -> dict | None:
    """
    Build cache data for one player: the L1 input sequence used both for
    training and for live vector updates.

    Players with fewer than min_games games return None, as do players without
    a PlayerAttributes row unless require_profile=False (DEFAULT_PROFILE).

    Returns dict with:
        game_ids: list of game_id strings
        box_stats: (N, 16) float array
        pbp_stats: (N, len(PBP_STAT_COLUMNS)) float array
        context: (N, 12) float array
        dpm_targets: (N, 3) float array (o_dpm, d_dpm, dpm)
        rapm_targets: (N, 2) float array (off_rapm, def_rapm)
        has_dpm: (N,) bool array — whether DPM target is available
        has_pbp: (N,) bool array — whether PBP stats are available
    """
    games = _load_player_games(conn, person_id)
    if not games or len(games) < min_games:
        return None

    profile = _load_player_profile(conn, person_id)
    if not profile:
        if require_profile:
            return None
        profile = {**DEFAULT_PROFILE, "birth_year": int(games[0]["season"][:4])}

    pbp_data = _load_pbp_stats(conn, person_id)
    dpm_data = _load_dpm_targets(conn, person_id)
    rapm_data = _load_season_rapm(conn, person_id)

    n_games = len(games)
    box_stats = np.zeros((n_games, len(BOX_STAT_COLUMNS)), dtype=np.float32)
    pbp_stats = np.zeros((n_games, len(PBP_STAT_COLUMNS)), dtype=np.float32)
    context = np.zeros((n_games, len(CONTEXT_COLUMNS)), dtype=np.float32)
    dpm_targets = np.zeros((n_games, 3), dtype=np.float32)
    rapm_targets = np.zeros((n_games, 2), dtype=np.float32)
    has_dpm = np.zeros(n_games, dtype=bool)
    has_pbp = np.zeros(n_games, dtype=bool)
    game_ids = []
    team_ids = []

    birth_year = profile.get("birth_year", 1990)
    prev_date = None
    prev_season = None
    season_game_count = 0

    for i, game in enumerate(games):
        game_id = game["game_id"]
        game_ids.append(game_id)
        team_ids.append(int(game.get("team_id", 0)))
        date_str = _extract_game_date(game.get("date_time_utc", ""))

        # --- Box stats ---
        for j, col in enumerate(BOX_STAT_COLUMNS):
            val = game.get(col)
            box_stats[i, j] = float(val) if val is not None else 0.0

        # --- PBP stats ---
        pbp = pbp_data.get(game_id)
        if pbp:
            has_pbp[i] = True
            for j, col in enumerate(PBP_STAT_COLUMNS):
                pbp_stats[i, j] = float(pbp.get(col, 0))

        # --- DPM targets ---
        dpm = dpm_data.get(date_str)
        if dpm:
            has_dpm[i] = True
            dpm_targets[i, 0] = dpm.get("o_dpm", 0.0) or 0.0
            dpm_targets[i, 1] = dpm.get("d_dpm", 0.0) or 0.0
            dpm_targets[i, 2] = dpm.get("dpm", 0.0) or 0.0

        # --- Season RAPM targets ---
        season = game.get("season", "")
        if season:
            # Extract year from season (e.g., "2024-2025" → "2025")
            year = season.split("-")[-1] if "-" in season else season
            rapm = rapm_data.get(year, {})
            rapm_targets[i, 0] = rapm.get("off_rapm", 0.0)
            rapm_targets[i, 1] = rapm.get("def_rapm", 0.0)

        # --- Context features ---
        # Age
        if date_str and birth_year:
            try:
                game_year = int(date_str[:4])
                game_month = int(date_str[5:7])
                age = game_year - birth_year + (game_month - 6) / 12.0  # approx
                context[i, 0] = age
            except (ValueError, IndexError):
                context[i, 0] = 25.0  # default

        # Rest days
        if prev_date and date_str:
            try:
                from datetime import datetime

                d1 = datetime.strptime(prev_date, "%Y-%m-%d")
                d2 = datetime.strptime(date_str, "%Y-%m-%d")
                rest = (d2 - d1).days - 1  # subtract 1 (game day itself)
                context[i, 1] = max(0, min(rest, 30))  # clamp
                context[i, 8] = (d2 - d1).days  # days_since_last_game (unclamped)
            except (ValueError, TypeError):
                context[i, 1] = 1.0
                context[i, 8] = 2.0
        else:
            context[i, 1] = 3.0  # first game of career: assume some rest
            context[i, 8] = 7.0

        # Home flag
        home_team = game.get("home_team", "")
        away_team = game.get("away_team", "")
        player_team_id = game.get("team_id")

        # Normalize historical team abbreviations
        home_modern = HISTORICAL_TO_MODERN.get(home_team, home_team)
        away_modern = HISTORICAL_TO_MODERN.get(away_team, away_team)

        # Determine if player is home via team_id → abbreviation lookup
        player_abbr = ""
        if team_id_to_abbr and player_team_id:
            player_abbr = team_id_to_abbr.get(int(player_team_id), "")
        is_home = player_abbr in (home_team, home_modern) if player_abbr else False
        context[i, 2] = 1.0 if is_home else 0.0

        # Opponent and team stats (use modern abbreviations for lookup)
        opponent = away_modern if is_home else home_modern
        player_team_abbr = home_modern if is_home else away_modern

        opp_stats = team_stats.get((opponent, season), {})
        tm_stats = team_stats.get((player_team_abbr, season), {})

        context[i, 3] = opp_stats.get("drtg", 110.0)  # opponent_drtg
        context[i, 4] = tm_stats.get("pace", 200.0)  # team_pace (pts+opp_pts scale)
        context[i, 9] = opp_stats.get("pace", 200.0)  # opponent_pace
        context[i, 10] = tm_stats.get("ortg", 110.0)  # team_ortg

        # Minutes share
        player_min = float(game.get("min", 0) or 0)
        context[i, 5] = player_min / 240.0  # 5 players * 48 min

        # Season progress (game number within season / 82)
        if season != prev_season:
            season_game_count = 0
            prev_season = season
        season_game_count += 1
        context[i, 6] = min(season_game_count / 82.0, 1.0)

        # Career games played
        context[i, 7] = float(i + 1)

        # Altitude
        venue_team = home_team  # game is at home team's arena
        context[i, 11] = float(ALTITUDE_MAP.get(venue_team, 0))

        prev_date = date_str

    return {
        "game_ids": game_ids,
        "box_stats": box_stats,
        "pbp_stats": pbp_stats,
        "context": context,
        "dpm_targets": dpm_targets,
        "rapm_targets": rapm_targets,
        "has_dpm": has_dpm,
        "has_pbp": has_pbp,
        "team_ids": np.array(team_ids, dtype=np.int64),
        "dates": [g.get("date_time_utc", "") for g in games],
    }


# ---------------------------------------------------------------------------
# Archetype computation
# ---------------------------------------------------------------------------


def compute_archetypes(conn, k: int = 10) -> np.ndarray:
    """
    Compute K archetype centroids from 2K ratings data.
    Falls back to k-means on career box stats if 2K data unavailable.

    Returns: (K, D) array of archetype centroids.
    """
    from sklearn.cluster import KMeans

    # Try 2K ratings first
    cursor = conn.execute(
        "SELECT person_id, attributes FROM TwoKRatings WHERE attributes IS NOT NULL"
    )
    rows = list(cursor)

    if len(rows) >= 100:
        logger.info(f"Computing archetypes from {len(rows)} 2K ratings")
        # Parse attribute JSON and build feature matrix
        features = []
        for _, attrs_json in rows:
            try:
                attrs = json.loads(attrs_json)
                # Extract numeric values, sorted by key for consistency
                vals = [
                    float(v)
                    for k, v in sorted(attrs.items())
                    if isinstance(v, (int, float))
                ]
                if len(vals) >= 20:
                    features.append(vals)
            except (json.JSONDecodeError, TypeError):
                continue

        if len(features) >= 100:
            X = np.array(features, dtype=np.float32)
            # Normalize
            X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            km.fit(X)
            return km.cluster_centers_.astype(np.float32)

    # Fallback: k-means on career box stat averages
    logger.info("2K data insufficient, computing archetypes from career box stats")
    cursor = conn.execute("""
        SELECT player_id,
            AVG(pts), AVG(oreb), AVG(dreb), AVG(ast), AVG(stl), AVG(blk),
            AVG(tov), AVG(fga), AVG(fgm), AVG(fg3a), AVG(fg3m), AVG(fta), AVG(ftm)
        FROM PlayerBox
        GROUP BY player_id
        HAVING COUNT(*) >= 50
        """)
    features = []
    for row in cursor:
        features.append(list(row[1:]))

    X = np.array(features, dtype=np.float32)
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X)
    return km.cluster_centers_.astype(np.float32)


# ---------------------------------------------------------------------------
# Main cache building
# ---------------------------------------------------------------------------


def build_cache(min_games: int = 5):
    """Build the full Phase 5 cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    PLAYERS_DIR.mkdir(parents=True, exist_ok=True)

    with get_db() as conn:
        # Get all players with sufficient games
        cursor = conn.execute(
            """
            SELECT DISTINCT player_id
            FROM PlayerBox pb
            JOIN Games g ON pb.game_id = g.game_id
            WHERE g.status = 3
            GROUP BY player_id
            HAVING COUNT(*) >= ?
            """,
            (min_games,),
        )
        player_ids = [row[0] for row in cursor]
        logger.info(f"Found {len(player_ids)} players with >= {min_games} games")

        # Pre-compute team rolling stats and team_id mapping
        logger.info("Computing team rolling stats...")
        team_stats = _load_team_rolling_stats(conn)
        team_id_to_abbr = _load_team_id_to_abbr(conn)
        logger.info(
            f"Team stats: {len(team_stats)} team-season entries, "
            f"{len(team_id_to_abbr)} team ID mappings"
        )

        # Build per-player caches
        profiles = {}
        player_list = []
        start = time.time()

        for i, pid in enumerate(player_ids):
            try:
                cache = build_player_cache(conn, pid, team_stats, team_id_to_abbr)
                if cache is None:
                    continue

                # Save player cache
                np.savez_compressed(
                    PLAYERS_DIR / f"{pid}.npz",
                    game_ids=np.array(cache["game_ids"], dtype=object),
                    box_stats=cache["box_stats"],
                    pbp_stats=cache["pbp_stats"],
                    context=cache["context"],
                    dpm_targets=cache["dpm_targets"],
                    rapm_targets=cache["rapm_targets"],
                    has_dpm=cache["has_dpm"],
                    has_pbp=cache["has_pbp"],
                    team_ids=cache["team_ids"],
                )

                # Load profile
                profile = _load_player_profile(conn, pid)
                if profile:
                    profiles[pid] = profile
                    player_list.append(pid)

            except Exception as e:
                logger.warning(f"Error building cache for player {pid}: {e}")
                continue

            if (i + 1) % 200 == 0:
                elapsed = time.time() - start
                rate = (i + 1) / elapsed
                eta = (len(player_ids) - i - 1) / rate
                logger.info(
                    f"  [{i+1}/{len(player_ids)}] {rate:.1f} players/sec, ETA {eta:.0f}s"
                )

        elapsed = time.time() - start
        logger.info(f"Built caches for {len(player_list)} players in {elapsed:.1f}s")

        # Save profiles
        profile_arrays = {}
        for col in PROFILE_COLUMNS:
            profile_arrays[col] = np.array(
                [profiles[pid].get(col, 0.0) for pid in player_list],
                dtype=np.float32,
            )
        profile_arrays["person_ids"] = np.array(player_list, dtype=np.int64)
        np.savez_compressed(CACHE_DIR / "profiles.npz", **profile_arrays)
        logger.info(f"Saved {len(player_list)} player profiles")

        # Compute archetypes
        logger.info("Computing archetypes...")
        archetypes = compute_archetypes(conn)
        np.savez_compressed(CACHE_DIR / "archetypes.npz", centroids=archetypes)
        logger.info(f"Saved {archetypes.shape[0]} archetypes ({archetypes.shape[1]}d)")

        # Save metadata
        # Compute normalization stats from random sample of players
        logger.info("Computing normalization stats from random player sample...")
        rng = np.random.RandomState(42)
        sample_pids = rng.choice(
            player_list, size=min(500, len(player_list)), replace=False
        )
        all_box = []
        all_pbp = []
        all_ctx = []
        for pid in sample_pids:
            path = PLAYERS_DIR / f"{pid}.npz"
            if path.exists():
                data = np.load(path, allow_pickle=True)
                all_box.append(data["box_stats"])
                if data["pbp_stats"].sum() > 0:
                    all_pbp.append(data["pbp_stats"])
                all_ctx.append(data["context"])

        box_all = np.concatenate(all_box, axis=0)
        ctx_all = np.concatenate(all_ctx, axis=0)

        metadata = {
            "n_players": len(player_list),
            "box_stat_columns": BOX_STAT_COLUMNS,
            "pbp_stat_columns": PBP_STAT_COLUMNS,
            "context_columns": CONTEXT_COLUMNS,
            "dpm_target_columns": DPM_TARGET_COLUMNS,
            "rapm_target_columns": RAPM_TARGET_COLUMNS,
            "profile_columns": PROFILE_COLUMNS,
            "box_mean": box_all.mean(axis=0).tolist(),
            "box_std": (box_all.std(axis=0) + 1e-8).tolist(),
            "context_mean": ctx_all.mean(axis=0).tolist(),
            "context_std": (ctx_all.std(axis=0) + 1e-8).tolist(),
            "data_splits": {
                "l1_pretrain": {"min_year": 2001, "max_year": 2016},
                "l1_pretrain_val": {"min_year": 2017, "max_year": 2017},
                "main_train": {"min_year": 2018, "max_year": 2022},
                "val": {"min_year": 2023, "max_year": 2023},
                "test": {"min_year": 2024, "max_year": 2030},
            },
        }

        if all_pbp:
            pbp_all = np.concatenate(all_pbp, axis=0)
            metadata["pbp_mean"] = pbp_all.mean(axis=0).tolist()
            metadata["pbp_std"] = (pbp_all.std(axis=0) + 1e-8).tolist()

        with open(CACHE_DIR / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info("Saved metadata.json")

    logger.info("Cache build complete!")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build Phase 5 NKE-H cache")
    parser.add_argument(
        "--min-games",
        type=int,
        default=5,
        help="Minimum games for a player to be included",
    )
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    build_cache(min_games=args.min_games)
