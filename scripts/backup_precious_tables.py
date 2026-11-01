#!/usr/bin/env python3
"""
Weekly backup of irreplaceable tables to data/backups/daily/ (cron, Sundays).

Most of the database (PBP, boxscores, game states) can be refetched from the
NBA API, but some tables cannot be reconstructed if lost:
- Predictions: pre-game predictions are point-in-time; once a game is played
  they can never be honestly regenerated.
- Betting: closing lines are scraped live and not available historically.
- InjuryReports: pre-game injury snapshots, also point-in-time.
- PipelineRuns: run history/audit trail.

Retention: the newest 8 backups (two months of weekly runs). Longer-term
history lives in the single full-DB backup taken at a season boundary.

Usage:
    python scripts/backup_precious_tables.py [--db PATH] [--outdir PATH]
"""

import argparse
import gzip
import logging
import shutil
import sqlite3
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.database import DB_PATH

PRECIOUS_TABLES = ["Predictions", "Betting", "InjuryReports", "PipelineRuns"]
KEEP = 8

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def backup(db_path: str, outdir: Path) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    stamp = date.today().strftime("%Y%m%d")
    raw_path = outdir / f"precious_{stamp}.sqlite"
    gz_path = outdir / f"precious_{stamp}.sqlite.gz"

    if gz_path.exists():
        logger.info("Backup for %s already exists, skipping", stamp)
        return gz_path

    # A run that died midway leaves its uncompressed file; start clean
    raw_path.unlink(missing_ok=True)
    tmp_path = gz_path.with_name(gz_path.name + ".tmp")
    conn = sqlite3.connect(raw_path)
    try:
        conn.execute("ATTACH DATABASE ? AS src", (f"file:{db_path}?mode=ro",))
        for table in PRECIOUS_TABLES:
            conn.execute(f"CREATE TABLE {table} AS SELECT * FROM src.{table}")
            n = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            logger.info("  %s: %s rows", table, f"{n:,}")
        conn.commit()
        conn.execute("DETACH src")
    finally:
        conn.close()

    # Compress to a temporary name: a partial .gz would count as the day's backup
    with open(raw_path, "rb") as fin, gzip.open(tmp_path, "wb", compresslevel=6) as fout:
        shutil.copyfileobj(fin, fout)
    tmp_path.replace(gz_path)
    raw_path.unlink()
    logger.info("Backup written: %s (%.1f MB)", gz_path, gz_path.stat().st_size / 1e6)
    return gz_path


def rotate(outdir: Path):
    """Keep the newest KEEP backups."""
    backups = sorted(outdir.glob("precious_*.sqlite.gz"))
    for old in backups[:-KEEP]:
        old.unlink()
        logger.info("Rotated out old backup: %s", old.name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=str(DB_PATH))
    parser.add_argument(
        "--outdir", default=str(PROJECT_ROOT / "data" / "backups" / "daily")
    )
    args = parser.parse_args()

    # Uses a read-only ATTACH via URI so the backup can never write to the
    # main database.
    outdir = Path(args.outdir)
    backup(args.db, outdir)
    rotate(outdir)


if __name__ == "__main__":
    main()
