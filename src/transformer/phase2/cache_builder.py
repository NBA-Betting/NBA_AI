"""
Cache Builder for Phase 2.

Pre-extracts per-game features from the database to disk.
Run once before training to avoid per-sample DB queries.

Usage:
    python -m src.transformer.phase2.cache_builder --seasons 2018-2019 2019-2020 ...
"""

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from src.database import get_db
from src.transformer.gamestates.tokenizer import GameStateTokenizer

logger = logging.getLogger(__name__)

# 30 NBA team tricodes -> integer indices
TEAM_TO_IDX = {
    "ATL": 0,
    "BOS": 1,
    "BKN": 2,
    "CHA": 3,
    "CHI": 4,
    "CLE": 5,
    "DAL": 6,
    "DEN": 7,
    "DET": 8,
    "GSW": 9,
    "HOU": 10,
    "IND": 11,
    "LAC": 12,
    "LAL": 13,
    "MEM": 14,
    "MIA": 15,
    "MIL": 16,
    "MIN": 17,
    "NOP": 18,
    "NYK": 19,
    "OKC": 20,
    "ORL": 21,
    "PHI": 22,
    "PHX": 23,
    "POR": 24,
    "SAC": 25,
    "SAS": 26,
    "TOR": 27,
    "UTA": 28,
    "WAS": 29,
}

# Historical team codes → current franchise codes
HISTORICAL_TO_CURRENT = {
    "NJN": "BKN",  # New Jersey Nets → Brooklyn Nets
    "SEA": "OKC",  # Seattle SuperSonics → Oklahoma City Thunder
    "NOH": "NOP",  # New Orleans Hornets → New Orleans Pelicans
    "NOK": "NOP",  # New Orleans/OKC Hornets → New Orleans Pelicans
    "VAN": "MEM",  # Vancouver Grizzlies → Memphis Grizzlies
    "CHH": "CHA",  # Charlotte Hornets (old) → Charlotte Hornets (current)
}

# 16 PlayerBox stats in order
STAT_COLUMNS = [
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

# Position mapping: DB positions -> 4 categories
POSITION_MAP = {
    "PG": 0,
    "SG": 0,
    "G": 0,  # Guard
    "SF": 1,
    "PF": 1,
    "F": 1,  # Forward
    "C": 2,  # Center
    "GF": 3,
    "NA": 3,
    "": 3,
    None: 3,  # Unknown
}


@dataclass
class PerGameFeatures:
    """Pre-extracted features for a single game."""

    game_id: str
    game_date: str
    season: str
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    margin: int
    total: int
    is_overtime: bool
    reg_home_score: int
    reg_away_score: int
    home_player_points: list[tuple[int, int]] = field(default_factory=list)
    away_player_points: list[tuple[int, int]] = field(default_factory=list)
    is_playoff: bool = False
    home_player_stats: list = field(
        default_factory=list
    )  # [(pid, [16 floats], pos_idx, pm_available), ...]
    away_player_stats: list = field(default_factory=list)
    home_team_efficiency: list = field(
        default_factory=list
    )  # [8 floats: eFG, TS, TOV%, FTR, 3PAR, AST_R, Pace, Net]
    away_team_efficiency: list = field(default_factory=list)


def _parse_players_data(
    players_data_str: Optional[str],
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Parse players_data JSON into per-side [(player_id, points), ...] lists."""
    if not players_data_str:
        return [], []

    try:
        data = json.loads(players_data_str)
    except (json.JSONDecodeError, TypeError):
        return [], []

    home_players = []
    away_players = []

    for side_key, player_list in [("home", home_players), ("away", away_players)]:
        side_data = data.get(side_key, {})
        for pid_str, pdata in side_data.items():
            try:
                pid = int(pid_str)
                if isinstance(pdata, dict):
                    points = int(pdata.get("points", 0))
                else:
                    points = 0
                player_list.append((pid, points))
            except (ValueError, TypeError):
                continue

    return home_players, away_players


def _get_regulation_scores(
    game_id: str, home_score: int, away_score: int, db_path: Optional[str] = None
) -> tuple[int, int, bool]:
    """Get regulation-end scores. Returns (reg_home, reg_away, is_overtime)."""
    with get_db(db_path) as conn:
        row = conn.execute(
            """
            SELECT home_score, away_score
            FROM GameStates
            WHERE game_id = ? AND period = 4
            ORDER BY play_id DESC
            LIMIT 1
            """,
            (game_id,),
        ).fetchone()

    if row is None:
        return home_score, away_score, False

    reg_home, reg_away = row
    is_overtime = (home_score != reg_home) or (away_score != reg_away)
    return reg_home, reg_away, is_overtime


def _batch_get_regulation_scores(
    game_ids: list[str], db_path: Optional[str] = None
) -> dict:
    """Batch-fetch regulation-end scores for all games. Returns {game_id: (reg_home, reg_away)}."""
    if not game_ids:
        return {}

    result = {}
    # Query in chunks to avoid SQLite variable limit
    chunk_size = 500
    with get_db(db_path) as conn:
        for i in range(0, len(game_ids), chunk_size):
            chunk = game_ids[i : i + chunk_size]
            placeholders = ",".join(["?"] * len(chunk))
            rows = conn.execute(
                f"""
                SELECT game_id, home_score, away_score
                FROM GameStates
                WHERE game_id IN ({placeholders}) AND period = 4
                ORDER BY game_id, play_id DESC
                """,
                chunk,
            ).fetchall()

            # Group by game_id, take first row (highest play_id due to DESC)
            seen = set()
            for game_id, home_score, away_score in rows:
                if game_id not in seen:
                    seen.add(game_id)
                    result[game_id] = (home_score, away_score)

    return result


def _batch_query_playerbox(
    game_ids: list[str],
    db_path: Optional[str] = None,
    max_players: int = 15,
) -> dict:
    """Batch-fetch PlayerBox data for all games. Returns {game_id: {team_abbrev: [(pid, stats, pos_idx, pm_avail), ...]}}."""
    if not game_ids:
        return {}

    result = {}
    chunk_size = 500

    with get_db(db_path) as conn:
        for i in range(0, len(game_ids), chunk_size):
            chunk = game_ids[i : i + chunk_size]
            placeholders = ",".join(["?"] * len(chunk))

            rows = conn.execute(
                f"""
                SELECT pb.game_id, pb.player_id, t.abbreviation, pb.position,
                       pb.min, pb.pts, pb.oreb, pb.dreb, pb.ast, pb.stl, pb.blk,
                       pb.tov, pb.pf, pb.fga, pb.fgm, pb.fg3a, pb.fg3m, pb.fta,
                       pb.ftm, pb.plus_minus
                FROM PlayerBox pb
                JOIN Teams t ON pb.team_id = t.team_id
                WHERE pb.game_id IN ({placeholders})
                  AND pb.min > 0
                ORDER BY pb.game_id, t.abbreviation, pb.min DESC
                """,
                chunk,
            ).fetchall()

            # Group by game_id and team
            for row in rows:
                game_id = row[0]
                player_id = row[1]
                team_abbrev = row[2]
                position = row[3]

                # Map historical team codes
                team_abbrev = HISTORICAL_TO_CURRENT.get(team_abbrev, team_abbrev)

                # Extract 16 stats, coalesce NULL to 0.0
                stats = [
                    float(row[4 + j]) if row[4 + j] is not None else 0.0
                    for j in range(16)
                ]

                # Position index
                pos_idx = POSITION_MAP.get(position, 3)

                # plus_minus availability (last stat, index 15)
                pm_avail = row[4 + 15] is not None

                if game_id not in result:
                    result[game_id] = {}
                if team_abbrev not in result[game_id]:
                    result[game_id][team_abbrev] = []

                result[game_id][team_abbrev].append(
                    (player_id, stats, pos_idx, pm_avail)
                )

    # Trim to max_players per team (already sorted by minutes DESC from SQL)
    for game_id in result:
        for team in result[game_id]:
            result[game_id][team] = result[game_id][team][:max_players]

    return result


def _batch_query_teambox(
    game_ids: list[str],
    db_path: Optional[str] = None,
) -> dict:
    """Batch-fetch TeamBox data and compute 8 efficiency features per team per game.

    Returns {game_id: {team_abbrev: [8 floats]}}.
    Features: eFG%, TS%, TOV%, FT_Rate, 3PA_Rate, AST_Ratio, Pace, Net_Points.
    """
    if not game_ids:
        return {}

    result = {}
    chunk_size = 500

    with get_db(db_path) as conn:
        for i in range(0, len(game_ids), chunk_size):
            chunk = game_ids[i : i + chunk_size]
            placeholders = ",".join(["?"] * len(chunk))

            rows = conn.execute(
                f"""
                SELECT tb.game_id, t.abbreviation,
                       tb.pts, tb.pts_allowed, tb.reb, tb.ast,
                       tb.tov, tb.fga, tb.fgm, tb.fg3a, tb.fg3m,
                       tb.fta, tb.ftm
                FROM TeamBox tb
                JOIN Teams t ON tb.team_id = t.team_id
                WHERE tb.game_id IN ({placeholders})
                """,
                chunk,
            ).fetchall()

            for row in rows:
                game_id, team_abbrev = row[0], row[1]
                team_abbrev = HISTORICAL_TO_CURRENT.get(team_abbrev, team_abbrev)
                pts, pts_allowed, reb, ast = row[2], row[3], row[4], row[5]
                tov, fga, fgm, fg3a, fg3m = row[6], row[7], row[8], row[9], row[10]
                fta, ftm = row[11], row[12]

                # Compute efficiency features (guard against division by zero)
                efg = (fgm + 0.5 * fg3m) / fga if fga > 0 else 0.0
                tsa = 2 * (fga + 0.44 * fta)
                ts = pts / tsa if tsa > 0 else 0.0
                tov_denom = fga + 0.44 * fta + tov
                tov_pct = tov / tov_denom if tov_denom > 0 else 0.0
                ft_rate = fta / fga if fga > 0 else 0.0
                three_pa_rate = fg3a / fga if fga > 0 else 0.0
                ast_ratio = ast / fgm if fgm > 0 else 0.0
                # Pace: approximate possessions (oreb not in TeamBox, added separately)
                pace = fga + 0.44 * fta + tov  # oreb subtracted later if available
                net_pts = pts - pts_allowed

                efficiency = [
                    efg,
                    ts,
                    tov_pct,
                    ft_rate,
                    three_pa_rate,
                    ast_ratio,
                    pace,
                    net_pts,
                ]

                if game_id not in result:
                    result[game_id] = {}
                result[game_id][team_abbrev] = efficiency

    return result


def _compute_oreb_from_playerbox(
    playerbox_data: dict,
) -> dict:
    """Sum oreb per team per game from PlayerBox data (already fetched).

    Returns {game_id: {team_abbrev: total_oreb}}.
    PlayerBox stats order: min, pts, oreb(2), dreb, ast, stl, blk, tov, pf, fga, fgm, fg3a, fg3m, fta, ftm, pm
    """
    result = {}
    for game_id, teams in playerbox_data.items():
        result[game_id] = {}
        for team, players in teams.items():
            total_oreb = sum(p[1][2] for p in players)  # stats[2] = oreb
            result[game_id][team] = total_oreb
    return result


def _build_player_experience(db_path: Optional[str] = None) -> dict:
    """Build {player_id: from_year} mapping from Players table."""
    with get_db(db_path) as conn:
        rows = conn.execute(
            "SELECT person_id, from_year FROM Players WHERE from_year IS NOT NULL"
        ).fetchall()
    return {int(pid): int(from_year) for pid, from_year in rows}


def build_cache(
    seasons: list[str],
    cache_dir: str = "data/phase2_cache",
    db_path: Optional[str] = None,
    skip_gamestates: bool = False,
    player_mapping_path: Optional[str] = None,
) -> dict:
    """
    Build the Phase 2 feature cache from the database.

    Outputs:
        game_features.pt - Dict[game_id -> PerGameFeatures dict]
        season_index.pt  - Dict[(team, season) -> list[game_id]] chronologically ordered
        team_mapping.json - TEAM_TO_IDX dict
        gamestates_cache.pt - Dict[game_id -> dict of numpy arrays]
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Building Phase 2 cache for seasons: {seasons}")

    # Query all final-state game data in batch
    season_placeholders = ",".join(["?"] * len(seasons))
    query = f"""
        SELECT g.game_id, g.home_team, g.away_team, g.date_time_utc, g.season,
               gs.home_score, gs.away_score, gs.players_data, g.season_type
        FROM Games g
        JOIN GameStates gs ON gs.game_id = g.game_id AND gs.is_final_state = 1
        WHERE g.season IN ({season_placeholders})
          AND g.status = 3
          AND g.game_data_finalized = 1
          AND g.season_type IN ('Regular Season', 'Post Season')
        ORDER BY g.date_time_utc
    """

    with get_db(db_path) as conn:
        rows = conn.execute(query, seasons).fetchall()

    logger.info(f"Fetched {len(rows)} games from database")

    # Batch-fetch regulation scores (avoid N+1 pattern)
    game_id_list = [row[0] for row in rows]
    reg_scores = _batch_get_regulation_scores(game_id_list, db_path)

    # Batch-fetch PlayerBox data
    playerbox_data = _batch_query_playerbox(game_id_list, db_path)
    n_with_playerbox = sum(1 for gid in game_id_list if gid in playerbox_data)
    logger.info(f"PlayerBox data: {n_with_playerbox}/{len(game_id_list)} games")

    # Batch-fetch TeamBox efficiency features
    teambox_data = _batch_query_teambox(game_id_list, db_path)
    n_with_teambox = sum(1 for gid in game_id_list if gid in teambox_data)
    logger.info(f"TeamBox data: {n_with_teambox}/{len(game_id_list)} games")

    # Compute oreb from PlayerBox for Pace correction
    oreb_data = _compute_oreb_from_playerbox(playerbox_data)

    # Adjust Pace in teambox_data by subtracting oreb
    for gid, teams in teambox_data.items():
        oreb_game = oreb_data.get(gid, {})
        for team_abbrev, eff in teams.items():
            team_oreb = oreb_game.get(team_abbrev, 0)
            eff[6] -= team_oreb  # Pace = fga + 0.44*fta + tov - oreb

    # Build player experience mapping
    player_experience = _build_player_experience(db_path)
    logger.info(
        f"Player experience: {len(player_experience)} players with from_year data"
    )

    game_features = {}
    season_index = defaultdict(list)

    for (
        game_id,
        home_team,
        away_team,
        date_utc,
        season,
        home_score,
        away_score,
        players_data,
        season_type,
    ) in rows:
        # Map historical team codes to current franchise codes
        home_team = HISTORICAL_TO_CURRENT.get(home_team, home_team)
        away_team = HISTORICAL_TO_CURRENT.get(away_team, away_team)

        home_players, away_players = _parse_players_data(players_data)

        game_date = str(date_utc)[:10] if date_utc else ""

        margin = home_score - away_score
        total = home_score + away_score

        # Get regulation scores from batch query
        reg_data = reg_scores.get(game_id)
        if reg_data is not None:
            reg_home, reg_away = reg_data
            is_overtime = (home_score != reg_home) or (away_score != reg_away)
        else:
            reg_home, reg_away, is_overtime = home_score, away_score, False

        # Get PlayerBox data for this game
        pb_game = playerbox_data.get(game_id, {})
        home_player_stats = pb_game.get(home_team, [])
        away_player_stats = pb_game.get(away_team, [])

        # If we have PlayerBox data, also derive player_points from it
        if home_player_stats:
            home_players = [
                (pid, int(stats[1])) for pid, stats, _, _ in home_player_stats
            ]
        if away_player_stats:
            away_players = [
                (pid, int(stats[1])) for pid, stats, _, _ in away_player_stats
            ]

        # Get TeamBox efficiency
        tb_game = teambox_data.get(game_id, {})
        home_eff = tb_game.get(home_team, [])
        away_eff = tb_game.get(away_team, [])

        is_playoff = season_type == "Post Season"

        features = PerGameFeatures(
            game_id=game_id,
            game_date=game_date,
            season=season,
            home_team=home_team,
            away_team=away_team,
            home_score=home_score,
            away_score=away_score,
            margin=margin,
            total=total,
            is_overtime=is_overtime,
            is_playoff=is_playoff,
            reg_home_score=reg_home,
            reg_away_score=reg_away,
            home_player_points=home_players,
            away_player_points=away_players,
            home_player_stats=home_player_stats,
            away_player_stats=away_player_stats,
            home_team_efficiency=home_eff,
            away_team_efficiency=away_eff,
        )

        game_features[game_id] = features

        # Build season index: both teams reference this game
        season_index[(home_team, season)].append(game_id)
        season_index[(away_team, season)].append(game_id)

    # Convert season_index defaultdict to regular dict for serialization
    season_index = dict(season_index)

    logger.info(
        f"Processed {len(game_features)} games, {len(season_index)} team-season entries"
    )

    # Log per-season counts
    season_counts = defaultdict(int)
    for gf in game_features.values():
        season_counts[gf.season] += 1
    for s, c in sorted(season_counts.items()):
        logger.info(f"  {s}: {c} games")

    # Save game features as plain dicts (avoids pickle class-path issues)
    game_features_dicts = {gid: asdict(gf) for gid, gf in game_features.items()}
    game_features_path = cache_path / "game_features.pt"
    torch.save(game_features_dicts, game_features_path)
    logger.info(f"Saved game features to {game_features_path}")

    # Save season index
    season_index_path = cache_path / "season_index.pt"
    torch.save(season_index, season_index_path)
    logger.info(f"Saved season index to {season_index_path}")

    # Save team mapping
    mapping_path = cache_path / "team_mapping.json"
    with open(mapping_path, "w") as f:
        json.dump(TEAM_TO_IDX, f, indent=2)

    if player_mapping_path:
        # Load external player mapping for consistency with pre-training cache
        with open(player_mapping_path) as f:
            player_id_map = {int(k): v for k, v in json.load(f).items()}
        logger.info(
            f"Loaded external player mapping: {len(player_id_map)} players from {player_mapping_path}"
        )
        # Append players the mapping hasn't seen (existing indices unchanged)
        seen = set()
        for gf in game_features.values():
            for entry in (gf.home_player_stats or gf.home_player_points) + (
                gf.away_player_stats or gf.away_player_points
            ):
                seen.add(entry[0])
        next_idx = max(player_id_map.values(), default=0) + 1
        for pid in sorted(seen - player_id_map.keys()):
            player_id_map[pid] = next_idx
            next_idx += 1
    else:
        # Build player ID mapping: raw NBA API IDs -> sequential indices (0=padding)
        all_player_ids = set()
        for gf in game_features.values():
            if gf.home_player_stats:
                for entry in gf.home_player_stats:
                    all_player_ids.add(entry[0])
                for entry in gf.away_player_stats:
                    all_player_ids.add(entry[0])
            else:
                for entry in gf.home_player_points:
                    all_player_ids.add(entry[0])
                for entry in gf.away_player_points:
                    all_player_ids.add(entry[0])

        # Sort for reproducibility, 0 is reserved for padding
        player_id_map = {pid: idx + 1 for idx, pid in enumerate(sorted(all_player_ids))}

    # Always save the mapping (whether loaded or generated)
    player_mapping_path_out = cache_path / "player_mapping.json"
    with open(player_mapping_path_out, "w") as f:
        json.dump({str(k): v for k, v in player_id_map.items()}, f)
    logger.info(
        f"Saved player mapping: {len(player_id_map)} unique players "
        f"(indices 1-{max(player_id_map.values()) if player_id_map else 0}, 0=padding)"
    )

    # Save player experience mapping
    exp_path = cache_path / "player_experience.json"
    with open(exp_path, "w") as f:
        json.dump({str(k): v for k, v in player_experience.items()}, f)
    logger.info(
        f"Saved player experience: {len(player_experience)} players to {exp_path}"
    )

    # Build GameStates cache
    if not skip_gamestates:
        _build_gamestates_cache(list(game_features.keys()), cache_path, db_path)
    else:
        logger.info("Skipping GameStates cache (skip_gamestates=True)")

    return {
        "game_features": game_features,
        "season_index": season_index,
        "player_id_map": player_id_map,
        "n_games": len(game_features),
    }


def update_cache(
    new_game_ids: list[str],
    cache_dir: str = "data/phase3_cache",
    db_path: Optional[str] = None,
) -> dict:
    """
    Incrementally add new games to an existing cache.

    Loads existing cache files, queries/processes only the new games,
    merges them in, and saves. Much faster than a full rebuild when
    only a few games are new (~30s vs ~20min for 33K games).

    Falls back to full rebuild if cache files are missing.

    Returns:
        Dict with n_games (total) and n_new (added this run).
    """
    cache_path = Path(cache_dir)
    required_files = [
        "game_features.pt",
        "gamestates_cache.pt",
        "season_index.pt",
        "player_mapping.json",
    ]

    if not all((cache_path / f).exists() for f in required_files):
        logger.warning(
            "Existing cache incomplete at %s — cannot do incremental update", cache_dir
        )
        return None

    if not new_game_ids:
        logger.info("No new games to add")
        return {"n_games": 0, "n_new": 0}

    logger.info(
        "Incremental cache update: %d new games at %s", len(new_game_ids), cache_dir
    )

    # --- Load existing cache files ---
    existing_features = torch.load(cache_path / "game_features.pt", weights_only=False)
    existing_season_index = torch.load(
        cache_path / "season_index.pt", weights_only=False
    )
    existing_gs_cache = torch.load(
        cache_path / "gamestates_cache.pt", weights_only=False
    )
    with open(cache_path / "player_mapping.json") as f:
        player_id_map = {int(k): v for k, v in json.load(f).items()}

    logger.info(
        "Loaded existing cache: %d games, %d player mappings",
        len(existing_features),
        len(player_id_map),
    )

    # --- Query new game data from DB ---
    placeholders = ",".join(["?"] * len(new_game_ids))
    query = f"""
        SELECT g.game_id, g.home_team, g.away_team, g.date_time_utc, g.season,
               gs.home_score, gs.away_score, gs.players_data, g.season_type
        FROM Games g
        JOIN GameStates gs ON gs.game_id = g.game_id AND gs.is_final_state = 1
        WHERE g.game_id IN ({placeholders})
          AND g.status = 3
          AND g.game_data_finalized = 1
        ORDER BY g.date_time_utc
    """
    with get_db(db_path) as conn:
        rows = conn.execute(query, new_game_ids).fetchall()

    if not rows:
        logger.info("No finalized game data found for new game IDs")
        return {"n_games": len(existing_features), "n_new": 0}

    fetched_ids = [row[0] for row in rows]
    logger.info("Fetched %d new games from database", len(fetched_ids))

    # --- Batch-fetch supporting data for new games only ---
    reg_scores = _batch_get_regulation_scores(fetched_ids, db_path)
    playerbox_data = _batch_query_playerbox(fetched_ids, db_path)
    teambox_data = _batch_query_teambox(fetched_ids, db_path)
    oreb_data = _compute_oreb_from_playerbox(playerbox_data)

    # Adjust pace
    for gid, teams in teambox_data.items():
        oreb_game = oreb_data.get(gid, {})
        for team_abbrev, eff in teams.items():
            eff[6] -= oreb_game.get(team_abbrev, 0)

    # --- Process new games into PerGameFeatures ---
    new_player_ids = set()

    for (
        game_id,
        home_team,
        away_team,
        date_utc,
        season,
        home_score,
        away_score,
        players_data,
        season_type,
    ) in rows:
        home_team = HISTORICAL_TO_CURRENT.get(home_team, home_team)
        away_team = HISTORICAL_TO_CURRENT.get(away_team, away_team)

        home_players, away_players = _parse_players_data(players_data)
        game_date = str(date_utc)[:10] if date_utc else ""
        margin = home_score - away_score
        total = home_score + away_score

        reg_data = reg_scores.get(game_id)
        if reg_data is not None:
            reg_home, reg_away = reg_data
            is_overtime = (home_score != reg_home) or (away_score != reg_away)
        else:
            reg_home, reg_away, is_overtime = home_score, away_score, False

        pb_game = playerbox_data.get(game_id, {})
        home_player_stats = pb_game.get(home_team, [])
        away_player_stats = pb_game.get(away_team, [])

        if home_player_stats:
            home_players = [
                (pid, int(stats[1])) for pid, stats, _, _ in home_player_stats
            ]
        if away_player_stats:
            away_players = [
                (pid, int(stats[1])) for pid, stats, _, _ in away_player_stats
            ]

        # Collect new player IDs
        for pid, _ in home_players + away_players:
            new_player_ids.add(pid)
        for entry in home_player_stats + away_player_stats:
            new_player_ids.add(entry[0])

        tb_game = teambox_data.get(game_id, {})
        is_playoff = season_type == "Post Season"

        features = PerGameFeatures(
            game_id=game_id,
            game_date=game_date,
            season=season,
            home_team=home_team,
            away_team=away_team,
            home_score=home_score,
            away_score=away_score,
            margin=margin,
            total=total,
            is_overtime=is_overtime,
            is_playoff=is_playoff,
            reg_home_score=reg_home,
            reg_away_score=reg_away,
            home_player_points=home_players,
            away_player_points=away_players,
            home_player_stats=home_player_stats,
            away_player_stats=away_player_stats,
            home_team_efficiency=tb_game.get(home_team, []),
            away_team_efficiency=tb_game.get(away_team, []),
        )

        existing_features[game_id] = asdict(features)

        # Update season index (append to end — games are ordered by date)
        for team in (home_team, away_team):
            key = (team, season)
            if key not in existing_season_index:
                existing_season_index[key] = []
            if game_id not in existing_season_index[key]:
                existing_season_index[key].append(game_id)

    # --- Extend player mapping (append-only to preserve existing indices) ---
    truly_new = sorted(new_player_ids - set(player_id_map.keys()))
    if truly_new:
        next_idx = max(player_id_map.values()) + 1 if player_id_map else 1
        for pid in truly_new:
            player_id_map[pid] = next_idx
            next_idx += 1
        logger.info(
            "Added %d new players to mapping (now %d total)",
            len(truly_new),
            len(player_id_map),
        )

    # --- Tokenize only new games' GameStates ---
    tokenizer = GameStateTokenizer(db_path)
    n_tokenized = 0
    for game_id in fetched_ids:
        result = tokenizer.tokenize_game(game_id)
        if result is not None:
            existing_gs_cache[game_id] = result
            n_tokenized += 1

    logger.info("Tokenized %d/%d new games", n_tokenized, len(fetched_ids))

    # --- Save all cache files ---
    torch.save(existing_features, cache_path / "game_features.pt")
    logger.info("Saved game features: %d total games", len(existing_features))

    torch.save(existing_season_index, cache_path / "season_index.pt")

    with open(cache_path / "player_mapping.json", "w") as f:
        json.dump({str(k): v for k, v in player_id_map.items()}, f)

    torch.save(existing_gs_cache, cache_path / "gamestates_cache.pt")
    logger.info("Saved GameStates cache: %d total games", len(existing_gs_cache))

    # Re-save player experience (fast — just a DB query)
    player_experience = _build_player_experience(db_path)
    with open(cache_path / "player_experience.json", "w") as f:
        json.dump({str(k): v for k, v in player_experience.items()}, f)

    # Save team mapping (static, but keep it consistent)
    with open(cache_path / "team_mapping.json", "w") as f:
        json.dump(TEAM_TO_IDX, f, indent=2)

    n_total = len(existing_features)
    n_new = len(fetched_ids)
    logger.info("Incremental update complete: %d total games (%d new)", n_total, n_new)

    return {"n_games": n_total, "n_new": n_new}


def _build_gamestates_cache(
    game_ids: list[str],
    cache_path: Path,
    db_path: Optional[str] = None,
):
    """Pre-tokenize GameStates for all games using GameStateTokenizer."""
    logger.info(f"Building GameStates cache for {len(game_ids)} games...")

    tokenizer = GameStateTokenizer(db_path)
    gs_cache = {}
    n_success = 0
    n_fail = 0

    for i, game_id in enumerate(game_ids):
        result = tokenizer.tokenize_game(game_id)
        if result is not None:
            gs_cache[game_id] = result
            n_success += 1
        else:
            n_fail += 1

        if (i + 1) % 1000 == 0:
            logger.info(
                f"  Tokenized {i + 1}/{len(game_ids)} games ({n_success} success, {n_fail} fail)"
            )

    gs_path = cache_path / "gamestates_cache.pt"
    torch.save(gs_cache, gs_path)
    logger.info(f"Saved GameStates cache: {n_success} games to {gs_path}")


def _dict_to_per_game_features(d: dict) -> PerGameFeatures:
    """Reconstruct PerGameFeatures from a plain dict."""
    # Convert player points lists of lists back to lists of tuples
    d = dict(d)
    d["home_player_points"] = [tuple(p) for p in d.get("home_player_points", [])]
    d["away_player_points"] = [tuple(p) for p in d.get("away_player_points", [])]
    # Convert player stats nested lists back to list of tuples: (pid, [stats], pos_idx, pm_avail)
    for key in ("home_player_stats", "away_player_stats"):
        raw = d.get(key, [])
        d[key] = (
            [(entry[0], entry[1], entry[2], entry[3]) for entry in raw] if raw else []
        )
    return PerGameFeatures(**d)


def load_cache(cache_dir: str = "data/phase2_cache") -> dict:
    """Load the pre-built cache from disk."""
    cache_path = Path(cache_dir)

    raw_features = torch.load(cache_path / "game_features.pt", weights_only=False)
    game_features = {
        gid: _dict_to_per_game_features(d) for gid, d in raw_features.items()
    }
    season_index = torch.load(cache_path / "season_index.pt", weights_only=False)
    gs_cache_path = cache_path / "gamestates_cache.pt"
    gs_cache = (
        torch.load(gs_cache_path, weights_only=False) if gs_cache_path.exists() else {}
    )

    with open(cache_path / "team_mapping.json") as f:
        team_mapping = json.load(f)

    player_mapping_path = cache_path / "player_mapping.json"
    if player_mapping_path.exists():
        with open(player_mapping_path) as f:
            player_id_map = {int(k): v for k, v in json.load(f).items()}
    else:
        player_id_map = {}

    player_exp_path = cache_path / "player_experience.json"
    if player_exp_path.exists():
        with open(player_exp_path) as f:
            player_experience = {int(k): v for k, v in json.load(f).items()}
    else:
        player_experience = {}

    return {
        "game_features": game_features,
        "season_index": season_index,
        "gs_cache": gs_cache,
        "team_mapping": team_mapping,
        "player_id_map": player_id_map,
        "player_experience": player_experience,
    }


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Build Phase 2 cache")
    parser.add_argument(
        "--seasons",
        nargs="+",
        default=[
            "2018-2019",
            "2019-2020",
            "2020-2021",
            "2021-2022",
            "2022-2023",
            "2023-2024",
            "2024-2025",
            "2025-2026",
        ],
        help="Seasons to cache",
    )
    parser.add_argument("--cache-dir", default="data/phase2_cache")
    parser.add_argument(
        "--skip-gamestates",
        action="store_true",
        help="Skip building GameStates cache (for pre-training)",
    )
    parser.add_argument(
        "--player-mapping",
        type=str,
        default=None,
        help="Path to external player_mapping.json",
    )

    args = parser.parse_args()
    result = build_cache(
        args.seasons,
        args.cache_dir,
        skip_gamestates=args.skip_gamestates,
        player_mapping_path=args.player_mapping,
    )
    print(f"\nCache built: {result['n_games']} games")
