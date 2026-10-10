#!/usr/bin/env python3
"""
parse_pbp_stats_v2.py

Parse PbP_Logs into enriched per-game player stats (~95 columns) using
lineup tracking, possession counting, and pbpstats-equivalent stat computation.

Usage:
    python scripts/parse_pbp_stats_v2.py --game 0022400001   # single game test
    python scripts/parse_pbp_stats_v2.py --season 2024-25    # one season
    python scripts/parse_pbp_stats_v2.py --all                # all seasons
    python scripts/parse_pbp_stats_v2.py --all --resume       # skip processed games
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.database import get_db, DB_PATH
from src.logging_config import setup_logging
from src.pbp.constants import STAT_COLUMNS
from src.pbp.stat_computer import process_game

# ---------------------------------------------------------------------------
# DB schema
# ---------------------------------------------------------------------------

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS PBPPlayerGameStatsV2 (
    player_id INTEGER NOT NULL,
    game_id TEXT NOT NULL,
    team_id INTEGER NOT NULL,
    {stat_cols},
    lineup_starters_found INTEGER DEFAULT 1,
    PRIMARY KEY (player_id, game_id)
)
""".format(
    stat_cols=",\n    ".join(
        f"{col} {'REAL' if col.startswith(('seconds_', 'total_')) else 'INTEGER'} DEFAULT 0"
        for col in STAT_COLUMNS
    )
)

INSERT_COLS = (
    ["player_id", "game_id", "team_id"] + STAT_COLUMNS + ["lineup_starters_found"]
)
INSERT_SQL = (
    f"INSERT OR REPLACE INTO PBPPlayerGameStatsV2 ({', '.join(INSERT_COLS)}) "
    f"VALUES ({', '.join('?' * len(INSERT_COLS))})"
)


# ---------------------------------------------------------------------------
# Game loading
# ---------------------------------------------------------------------------


def load_game_events(conn, game_id: str) -> list[dict]:
    """Load all PBP events for a game from PbP_Logs, sorted by play_id."""
    cursor = conn.execute(
        "SELECT log_data FROM PbP_Logs WHERE game_id = ? ORDER BY play_id",
        (game_id,),
    )
    events = []
    for (log_data_str,) in cursor:
        try:
            events.append(json.loads(log_data_str))
        except (json.JSONDecodeError, TypeError):
            continue
    return events


def box_player_names(conn, game_id: str) -> dict[int, tuple[int, list[str]]]:
    """{player_id: (team_id, name forms)} from the game's box score, so subs
    of players with no other actions resolve ("SUB: Waters III FOR ...")."""
    names = {}
    for pid, tid, full in conn.execute(
        "SELECT player_id, team_id, player_name FROM PlayerBox WHERE game_id = ?",
        (game_id,),
    ):
        parts = (full or "").split()
        if len(parts) < 2:
            continue
        forms = {full, " ".join(parts[1:]), parts[-1]}
        if parts[-1].rstrip(".").upper() in ("JR", "SR", "II", "III", "IV", "V"):
            forms.add(parts[-2])
        names[int(pid)] = (int(tid), sorted(forms))
    return names


def get_game_teams(conn, game_id: str) -> tuple[int, int]:
    """Get home and away team IDs for a game from the Games table."""
    # Try to get team IDs from TeamBox (more reliable)
    cursor = conn.execute(
        "SELECT DISTINCT team_id FROM TeamBox WHERE game_id = ?", (game_id,)
    )
    team_ids = [int(row[0]) for row in cursor]
    if len(team_ids) == 2:
        # Determine home/away from Games table
        cursor = conn.execute(
            "SELECT home_team, away_team FROM Games WHERE game_id = ?", (game_id,)
        )
        row = cursor.fetchone()
        if row:
            home_abbr, away_abbr = row
            # Map abbreviation to team_id via Teams table
            cursor = conn.execute(
                "SELECT team_id, abbreviation FROM Teams WHERE team_id IN (?, ?)",
                tuple(team_ids),
            )
            abbr_map = {r[1]: int(r[0]) for r in cursor}
            home_id = abbr_map.get(home_abbr, team_ids[0])
            away_id = abbr_map.get(away_abbr, team_ids[1])
            return int(home_id), int(away_id)
        return int(team_ids[0]), int(team_ids[1])

    # Fallback: extract from PBP events
    return 0, 0


def get_game_teams_from_pbp(events: list[dict]) -> tuple[int, int]:
    """Extract team IDs from PBP events when Games/TeamBox unavailable."""
    team_ids = set()
    for evt in events[:100]:
        tid = evt.get("teamId", 0)
        if tid and tid != 0:
            team_ids.add(tid)
        if len(team_ids) >= 2:
            break
    team_list = sorted(team_ids)
    if len(team_list) >= 2:
        return team_list[0], team_list[1]
    return 0, 0


# ---------------------------------------------------------------------------
# Game ID queries
# ---------------------------------------------------------------------------


def get_game_ids_for_season(conn, season: str) -> list[str]:
    """Get all game_ids with PBP data for a season.

    Accepts season in '2024-25' or '2024-2025' format.
    """
    # Normalize: accept '2024-25' → '2024-2025'
    if len(season) == 7 and season[4] == "-":
        start_year = season[:4]
        end_short = season[5:]
        season = f"{start_year}-{int(start_year[:2] + end_short):04d}"

    cursor = conn.execute(
        """
        SELECT DISTINCT g.game_id
        FROM Games g
        JOIN PbP_Logs p ON g.game_id = p.game_id
        WHERE g.season = ?
        AND g.status = 3
        ORDER BY g.game_id
        """,
        (season,),
    )
    return [row[0] for row in cursor]


def get_all_game_ids(conn) -> list[str]:
    """Games whose PBP and box score are final. Stats are built once per game
    (resume skips processed games), so a game fetched mid-play must wait;
    the box score supplies names for resolving substitutions."""
    cursor = conn.execute(
        """
        SELECT DISTINCT p.game_id FROM PbP_Logs p
        JOIN Games g ON g.game_id = p.game_id
        WHERE g.game_data_finalized = 1 AND g.boxscore_data_finalized = 1
        ORDER BY p.game_id
        """
    )
    return [row[0] for row in cursor]


def get_processed_game_ids(conn) -> set[str]:
    """Get game_ids already in PBPPlayerGameStatsV2."""
    try:
        cursor = conn.execute("SELECT DISTINCT game_id FROM PBPPlayerGameStatsV2")
        return {row[0] for row in cursor}
    except Exception:
        return set()


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------


def process_games(
    game_ids: list[str],
    conn,
    batch_size: int = 100,
):
    """Process a list of games and insert results into the DB.

    Returns the number of games processed (failures are logged and skipped).
    """
    total = len(game_ids)
    if total == 0:
        print("No games to process.")
        return 0

    print(f"Processing {total} games...")
    start_time = time.time()
    games_processed = 0
    rows_inserted = 0
    batch_rows = []
    starters_failures = 0

    for i, game_id in enumerate(game_ids):
        try:
            events_raw = load_game_events(conn, game_id)
            if not events_raw:
                continue

            home_id, away_id = get_game_teams(conn, game_id)
            if not home_id or not away_id:
                home_id, away_id = get_game_teams_from_pbp(events_raw)
            if not home_id or not away_id:
                logging.warning(f"Could not determine teams for {game_id}, skipping")
                continue

            result = process_game(
                events_raw, home_id, away_id, game_id, box_player_names(conn, game_id)
            )

            for pid, pdata in result.items():
                row = [pid, game_id, pdata["team_id"]]
                stats = pdata["stats"]
                for col in STAT_COLUMNS:
                    row.append(stats.get(col, 0))
                row.append(1 if pdata.get("lineup_starters_found", True) else 0)
                batch_rows.append(tuple(row))

            if result:
                first_player = next(iter(result.values()))
                if not first_player.get("lineup_starters_found", True):
                    starters_failures += 1

            games_processed += 1

        except Exception as e:
            logging.warning(f"Error processing {game_id}: {e}")
            continue

        # Batch commit
        if len(batch_rows) >= batch_size * 20:  # ~20 players per game
            conn.executemany(INSERT_SQL, batch_rows)
            conn.commit()
            rows_inserted += len(batch_rows)
            batch_rows = []

        # Progress reporting
        if (i + 1) % 100 == 0 or (i + 1) == total:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (total - i - 1) / rate if rate > 0 else 0
            print(
                f"  [{i+1}/{total}] {rate:.1f} games/sec, "
                f"ETA {eta:.0f}s, rows so far: {rows_inserted + len(batch_rows)}"
            )

    # Final batch
    if batch_rows:
        conn.executemany(INSERT_SQL, batch_rows)
        conn.commit()
        rows_inserted += len(batch_rows)

    elapsed = time.time() - start_time
    print(
        f"\nDone. Processed {games_processed} games, "
        f"{rows_inserted} player-game rows in {elapsed:.1f}s"
    )
    if starters_failures:
        print(f"  Starter detection failures: {starters_failures} games")
    return games_processed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Parse PBP data into enriched per-game player stats (~95 columns)."
    )
    parser.add_argument("--game", type=str, help="Process a single game_id")
    parser.add_argument(
        "--season", type=str, help="Process all games in a season (e.g., 2024-25)"
    )
    parser.add_argument(
        "--all", action="store_true", help="Process all games with PBP data"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Skip games already processed (default: True)",
    )
    parser.add_argument(
        "--no-resume", dest="resume", action="store_false", help="Reprocess all games"
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="WARNING",
        help="Logging level (default: WARNING)",
    )
    parser.add_argument("--db_path", type=str, default=None)
    args = parser.parse_args()

    setup_logging(log_level=args.log_level.upper())

    with get_db(args.db_path) as conn:
        # Create table
        conn.execute(CREATE_TABLE_SQL)
        conn.commit()

        # Get game list
        if args.game:
            game_ids = [args.game]
        elif args.season:
            game_ids = get_game_ids_for_season(conn, args.season)
            print(f"Found {len(game_ids)} games for season {args.season}")
        elif args.all:
            game_ids = get_all_game_ids(conn)
            print(f"Found {len(game_ids)} total games with PBP data")
        else:
            parser.print_help()
            return

        # Resume: filter out already processed
        if args.resume and not args.game:
            already = get_processed_game_ids(conn)
            before = len(game_ids)
            game_ids = [gid for gid in game_ids if gid not in already]
            skipped = before - len(game_ids)
            if skipped:
                print(
                    f"Resume: {skipped} games already processed, {len(game_ids)} remaining"
                )

        process_games(game_ids, conn)

        # Final stats
        cursor = conn.execute(
            "SELECT COUNT(*), COUNT(DISTINCT game_id) FROM PBPPlayerGameStatsV2"
        )
        total_rows, total_games = cursor.fetchone()
        print(f"\nPBPPlayerGameStatsV2: {total_rows:,} rows, {total_games:,} games")


if __name__ == "__main__":
    main()
