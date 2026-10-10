#!/usr/bin/env python3
"""
Export a compressed starter database for GitHub release.

Creates a database of the current and previous seasons with everything needed
to run the web app and daily pipeline, then compresses it with gzip. Two
seasons, so a new install always has a full past season (dashboard record,
prior-season features for opening games) and its first pipeline run only
fetches games played since the export.

Included:
- Both seasons' games, box scores, PBP, GameStates, betting, features
- All predictions (every predictor)
- Injury reports and cache
- Reference data (players, teams, attributes, coaches, schedule)
- Pipeline run history

Excluded:
- Historical seasons (only needed for model training)
- Training-only tables (RAPM, Darkball, lineups, etc.)

Output: data/NBA_AI_starter.sqlite.gz (~100-200 MB compressed)

Usage:
    python scripts/export_starter_db.py                  # current + previous season
    python scripts/export_starter_db.py --season 2025-2026 --no-compress
"""

import argparse
import gzip
import logging
import shutil
import sqlite3
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.database import DB_PATH
from src.utils import determine_current_season

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _season_start_date(season: str) -> str:
    """Derive approximate season start date (Oct 1) from season string."""
    start_year = season.split("-")[0]
    return f"{start_year}-10-01"


def export_starter_db(
    source_path: str, output_path: str, seasons: list[str], compress: bool = True
):
    """Export a starter database with the given seasons of data."""
    t0 = time.time()
    season_start = _season_start_date(min(seasons))
    season_list = ", ".join(f"'{s}'" for s in seasons)

    logger.info(f"Source: {source_path}")
    logger.info(f"Seasons: {', '.join(seasons)}")

    src = sqlite3.connect(source_path)
    src.execute("PRAGMA journal_mode=WAL")

    # Verify source has data for this season
    for season in seasons:
        n_games = src.execute(
            "SELECT COUNT(*) FROM Games WHERE season=?", (season,)
        ).fetchone()[0]
        if n_games == 0:
            logger.error(f"No games found for season {season} in source database")
            src.close()
            sys.exit(1)
        logger.info(f"Found {n_games:,} games for {season}")

    # Create fresh output
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    dst = sqlite3.connect(output_path)

    # --- Copy schema for ALL tables (keeps structure even for empty ones) ---
    tables = src.execute(
        "SELECT name, sql FROM sqlite_master WHERE type='table' AND name != 'sqlite_sequence'"
        " AND name NOT LIKE '%backup%'"  # internal backup copies
    ).fetchall()
    for name, sql in tables:
        if sql:
            dst.execute(sql)
    dst.commit()

    # --- Define what to export ---
    game_id_filter = (
        f"game_id IN (SELECT game_id FROM Games WHERE season IN ({season_list}))"
    )

    season_filtered = {
        "Games": f"SELECT * FROM Games WHERE season IN ({season_list})",
        "PlayerBox": f"SELECT * FROM PlayerBox WHERE {game_id_filter}",
        "TeamBox": f"SELECT * FROM TeamBox WHERE {game_id_filter}",
        "GameStates": f"SELECT * FROM GameStates WHERE {game_id_filter}",
        "PbP_Logs": f"SELECT * FROM PbP_Logs WHERE {game_id_filter}",
        "Betting": f"SELECT * FROM Betting WHERE {game_id_filter}",
        "Features": f"SELECT * FROM Features WHERE {game_id_filter}",
        "Predictions": f"SELECT * FROM Predictions WHERE {game_id_filter}",
        "InjuryReports": f"SELECT * FROM InjuryReports WHERE season IN ({season_list})",
        "InjuryCache": f"SELECT * FROM InjuryCache WHERE report_date >= '{season_start}'",
    }

    reference_tables = [
        "Players",
        "Teams",
        "PlayerAttributes",
        "PlayersCache",
        "ScheduleCache",
        "CoversAttempts",
        "ESPNGameMapping",
        "PipelineRuns",
        "Coaches",  # the daily pipeline's coaches step would refetch it
    ]

    skip_tables = [
        "DarkoDPM",
        "DataballrMultiYearRAPM",
        "DataballrPlayerMetrics",
        "DataballrSQRAPM",
        "DataballrWingspan",
        "RAPMArchive",
        "LineupStats",
        "TwoKRatings",
        "PBPPlayerGameStatsV2",
        "GameRotationRoster",
    ]

    # --- Export season-filtered tables ---
    total_rows = 0
    logger.info("\nSeason-filtered tables:")
    for table, query in season_filtered.items():
        try:
            rows = src.execute(query).fetchall()
            if rows:
                cols = [
                    desc[0]
                    for desc in src.execute(
                        f"SELECT * FROM {table} LIMIT 1"
                    ).description
                ]
                placeholders = ",".join(["?"] * len(cols))
                dst.executemany(f"INSERT INTO {table} VALUES ({placeholders})", rows)
                dst.commit()
            total_rows += len(rows)
            logger.info(f"  {table}: {len(rows):,} rows")
        except Exception as e:
            logger.warning(f"  {table}: FAILED ({e})")

    # --- Export reference tables (full copy) ---
    logger.info("\nReference tables (full copy):")
    for table in reference_tables:
        try:
            rows = src.execute(f"SELECT * FROM {table}").fetchall()
            if rows:
                cols = [
                    desc[0]
                    for desc in src.execute(
                        f"SELECT * FROM {table} LIMIT 1"
                    ).description
                ]
                placeholders = ",".join(["?"] * len(cols))
                dst.executemany(f"INSERT INTO {table} VALUES ({placeholders})", rows)
                dst.commit()
            total_rows += len(rows)
            logger.info(f"  {table}: {len(rows):,} rows")
        except Exception as e:
            logger.warning(f"  {table}: FAILED ({e})")

    logger.info(f"\nSkipped {len(skip_tables)} training-only tables")

    # --- Verify all source tables are accounted for ---
    all_src_tables = {
        r[0]
        for r in src.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name != 'sqlite_sequence'"
            " AND name NOT LIKE '%backup%'"
        ).fetchall()
    }
    accounted = set(season_filtered) | set(reference_tables) | set(skip_tables)
    unaccounted = all_src_tables - accounted
    if unaccounted:
        logger.warning(f"Unaccounted tables in source DB: {unaccounted}")

    src.close()

    # --- Create indexes ---
    logger.info("\nCreating indexes...")
    indexes = [
        "CREATE INDEX IF NOT EXISTS idx_playerbox_game ON PlayerBox(game_id)",
        "CREATE INDEX IF NOT EXISTS idx_playerbox_player ON PlayerBox(player_id)",
        "CREATE INDEX IF NOT EXISTS idx_gamestates_game ON GameStates(game_id)",
        "CREATE INDEX IF NOT EXISTS idx_gamestates_final ON GameStates(game_id, is_final_state)",
        "CREATE INDEX IF NOT EXISTS idx_games_home_date ON Games(home_team, status, date_time_utc)",
        "CREATE INDEX IF NOT EXISTS idx_games_away_date ON Games(away_team, status, date_time_utc)",
        "CREATE INDEX IF NOT EXISTS idx_betting_game ON Betting(game_id)",
        "CREATE INDEX IF NOT EXISTS idx_betting_lines ON Betting(lines_finalized)",
        "CREATE INDEX IF NOT EXISTS idx_teambox_team ON TeamBox(team_id, game_id)",
        "CREATE INDEX IF NOT EXISTS idx_features_game ON Features(game_id)",
        "CREATE INDEX IF NOT EXISTS idx_predictions_predictor ON Predictions(predictor, game_id)",
        "CREATE INDEX IF NOT EXISTS idx_pbp_game ON PbP_Logs(game_id)",
    ]
    for idx_sql in indexes:
        dst.execute(idx_sql)
    dst.commit()

    # --- Compact ---
    dst.execute("PRAGMA journal_mode=WAL")
    logger.info("Compacting...")
    dst.execute("VACUUM")
    dst.close()

    raw_size = out_path.stat().st_size
    raw_mb = raw_size / (1024 * 1024)
    logger.info(f"Uncompressed: {raw_mb:.1f} MB")

    # --- Verify ---
    logger.info("\nVerifying export...")
    verify_starter_db(output_path)

    # --- Compress ---
    final_path = output_path
    if compress:
        gz_path = output_path + ".gz"
        logger.info(f"\nCompressing to {gz_path}...")
        with open(output_path, "rb") as f_in:
            with gzip.open(gz_path, "wb", compresslevel=6) as f_out:
                shutil.copyfileobj(f_in, f_out)
        gz_size = Path(gz_path).stat().st_size
        gz_mb = gz_size / (1024 * 1024)
        ratio = gz_size / raw_size * 100
        logger.info(f"Compressed: {gz_mb:.1f} MB ({ratio:.0f}% of original)")
        out_path.unlink()
        final_path = gz_path

    elapsed = time.time() - t0
    logger.info(f"\nDone in {elapsed:.0f}s: {final_path}")
    logger.info(f"Total rows exported: {total_rows:,}")


def verify_starter_db(db_path: str):
    """Verify the exported database has the essential data."""
    conn = sqlite3.connect(db_path)
    errors = []

    checks = [
        (
            "Games",
            "SELECT COUNT(*) FROM Games WHERE status = 3",
            100,
            "completed games",
        ),
        (
            "Games (upcoming)",
            "SELECT COUNT(*) FROM Games WHERE status = 1",
            0,
            "upcoming games",
        ),
        ("PlayerBox", "SELECT COUNT(*) FROM PlayerBox", 100, "player box scores"),
        ("TeamBox", "SELECT COUNT(*) FROM TeamBox", 100, "team box scores"),
        ("GameStates", "SELECT COUNT(*) FROM GameStates", 100, "game states"),
        ("Features", "SELECT COUNT(*) FROM Features", 100, "feature sets"),
        ("Predictions", "SELECT COUNT(*) FROM Predictions", 1, "predictions"),
        ("Betting", "SELECT COUNT(*) FROM Betting", 1, "betting lines"),
        ("Players", "SELECT COUNT(*) FROM Players", 100, "players"),
        ("Teams", "SELECT COUNT(*) FROM Teams", 30, "teams"),
    ]

    for name, query, min_count, desc in checks:
        try:
            count = conn.execute(query).fetchone()[0]
            status = "OK" if count >= min_count else "LOW"
            if count < min_count:
                errors.append(f"{name}: only {count} {desc} (expected >= {min_count})")
            logger.info(f"  {status}: {name} = {count:,} {desc}")
        except Exception as e:
            errors.append(f"{name}: query failed ({e})")
            logger.info(f"  FAIL: {name} ({e})")

    # Check predictions have multiple predictors
    predictors = conn.execute(
        "SELECT predictor, COUNT(*) FROM Predictions GROUP BY predictor"
    ).fetchall()
    logger.info(f"  Predictors: {', '.join(f'{p}({c})' for p, c in predictors)}")

    conn.close()

    if errors:
        logger.warning(f"\n{len(errors)} verification warning(s):")
        for e in errors:
            logger.warning(f"  - {e}")
    else:
        logger.info("  All checks passed")


def main():
    parser = argparse.ArgumentParser(
        description="Export a compressed starter database for GitHub release"
    )
    parser.add_argument(
        "--season",
        nargs="+",
        help="Seasons to export, e.g. 2025-2026 (default: current and previous)",
    )
    parser.add_argument(
        "--output",
        default="data/NBA_AI_starter.sqlite",
        help="Output path (before compression)",
    )
    parser.add_argument(
        "--no-compress",
        action="store_true",
        help="Skip gzip compression",
    )
    args = parser.parse_args()

    seasons = args.season
    if not seasons:
        current = determine_current_season()
        start = int(current[:4])
        seasons = [f"{start - 1}-{start}", current]
    export_starter_db(str(DB_PATH), args.output, seasons, compress=not args.no_compress)


if __name__ == "__main__":
    main()
