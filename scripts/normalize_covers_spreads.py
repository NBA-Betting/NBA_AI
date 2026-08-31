#!/usr/bin/env python3
"""Normalize historical Covers lines to the home team's perspective.

The old matchup parser discarded which team a Covers spread described. For a
completed game, the stored ATS result lets us determine whether a row is
already home-oriented or needs to be flipped. Rows that cannot be determined
unambiguously are invalidated instead of guessed.

The command is read-only by default. Pass ``--apply`` to commit changes:

    python scripts/normalize_covers_spreads.py --database /path/to/nba.sqlite
    python scripts/normalize_covers_spreads.py --database /path/to/nba.sqlite --apply
"""

import argparse
import sqlite3
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import quote

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.database_updater.covers import _invert_spread_result  # noqa: E402


def _home_ats_result(home_margin: float, home_spread: float) -> str:
    """Calculate the home team's ATS result using the dashboard push threshold."""
    margin_vs_line = home_margin + home_spread
    if abs(margin_vs_line) < 0.25:
        return "P"
    return "W" if margin_vs_line > 0 else "L"


def classify_covers_row(
    spread: Optional[float],
    spread_result: Optional[str],
    home_score: Optional[int],
    away_score: Optional[int],
) -> tuple[str, Optional[float], Optional[str]]:
    """Classify and normalize one legacy Covers row.

    Returns ``(action, spread, result)`` where action is ``keep``, ``flip``,
    or ``invalidate``.
    """
    if (
        spread is None
        or spread_result not in {"W", "L", "P"}
        or home_score is None
        or away_score is None
    ):
        return "invalidate", None, None

    spread = float(spread)
    home_margin = home_score - away_score
    inverted_result = _invert_spread_result(spread_result)

    matches_home = spread_result == _home_ats_result(home_margin, spread)
    matches_away = inverted_result == _home_ats_result(home_margin, -spread)

    if matches_home and not matches_away:
        return "keep", spread, spread_result
    if matches_away and not matches_home:
        normalized_spread = -spread
        if normalized_spread == 0:
            normalized_spread = 0.0
        return "flip", normalized_spread, inverted_result

    # A pick'em push is identical from either perspective and is safe to keep.
    if matches_home and matches_away and spread == 0 and spread_result == "P":
        return "keep", 0.0, "P"

    return "invalidate", None, None


def _connect(database_path: Path, apply: bool) -> sqlite3.Connection:
    if apply:
        return sqlite3.connect(database_path)

    read_only_uri = f"file:{quote(str(database_path), safe='/')}?mode=ro"
    return sqlite3.connect(read_only_uri, uri=True)


def normalize_database(database_path: Path, apply: bool = False) -> Counter:
    """Inspect every legacy Covers spread and optionally apply safe corrections."""
    if not database_path.is_file():
        raise FileNotFoundError(f"Database not found: {database_path}")

    conn = _connect(database_path, apply)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT
                b.game_id,
                b.covers_closing_spread,
                b.spread_result,
                gs.home_score,
                gs.away_score
            FROM Betting b
            LEFT JOIN GameStates gs
              ON gs.rowid = (
                  SELECT final_state.rowid
                  FROM GameStates final_state
                  WHERE final_state.game_id = b.game_id
                    AND final_state.is_final_state = 1
                  ORDER BY final_state.rowid DESC
                  LIMIT 1
              )
            WHERE b.covers_closing_spread IS NOT NULL
            ORDER BY b.game_id
            """
        ).fetchall()

        counts = Counter()
        updates = []
        invalidations = []
        updated_at = datetime.now(timezone.utc).isoformat()

        for row in rows:
            action, spread, result = classify_covers_row(
                row["covers_closing_spread"],
                row["spread_result"],
                row["home_score"],
                row["away_score"],
            )
            counts[action] += 1
            if action == "flip":
                updates.append((spread, result, updated_at, row["game_id"]))
            elif action == "invalidate":
                invalidations.append((updated_at, row["game_id"]))

        counts["total"] = len(rows)

        if apply:
            conn.executemany(
                """
                UPDATE Betting
                SET covers_closing_spread = ?, spread_result = ?, updated_at = ?
                WHERE game_id = ?
                """,
                updates,
            )
            conn.executemany(
                """
                UPDATE Betting
                SET covers_closing_spread = NULL, spread_result = NULL, updated_at = ?
                WHERE game_id = ?
                """,
                invalidations,
            )
            conn.commit()

        return counts
    except Exception:
        if apply:
            conn.rollback()
        raise
    finally:
        conn.close()


def _default_database_path() -> Path:
    from src.config import config

    return Path(config["database"]["path"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database",
        type=Path,
        help="SQLite database path (defaults to the configured DATABASE_PATH)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Commit safe flips and invalidate ambiguous rows (default: dry run)",
    )
    args = parser.parse_args()

    database_path = (args.database or _default_database_path()).expanduser().resolve()
    counts = normalize_database(database_path, apply=args.apply)
    mode = "APPLIED" if args.apply else "DRY RUN"

    print(f"Covers normalization: {mode}")
    print(f"Database: {database_path}")
    print(f"Rows inspected: {counts['total']}")
    print(f"Already home perspective: {counts['keep']}")
    print(f"Away perspective to flip: {counts['flip']}")
    print(f"Ambiguous rows to invalidate: {counts['invalidate']}")
    if not args.apply:
        print("No changes written. Re-run with --apply to update this database.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
