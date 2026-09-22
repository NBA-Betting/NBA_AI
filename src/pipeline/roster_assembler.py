"""
Roster Assembler: project who will play in upcoming games.

Strategy: "injury subtraction"
  1. For each team, get their most recent game's active roster (PlayerBox, min > 0)
  2. Subtract players listed as "Out" in the latest InjuryReports
  3. "Questionable" players are included (they play ~70% of the time)
  4. Sort by average recent minutes to identify rotation order

Usage:
    from src.pipeline.roster_assembler import RosterAssembler
    ra = RosterAssembler()
    rosters = ra.get_projected_rosters(["0022500900", "0022500901"])
"""

from __future__ import annotations

import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

from src.database import DB_PATH, get_db

logger = logging.getLogger(__name__)


class RosterAssembler:
    """Projects rosters for upcoming games using injury subtraction."""

    def __init__(self, db_path: str | Path = DB_PATH):
        self.db_path = str(db_path)

    def get_projected_rosters(self, game_ids: list[str]) -> dict[str, dict]:
        """
        Project rosters for the given game IDs.

        Returns:
            {game_id: {
                "home_players": [player_id, ...],  # sorted by avg minutes desc
                "away_players": [player_id, ...],
                "home_team": str,
                "away_team": str,
                "home_confidence": float,  # 1.0 = no ambiguity
                "away_confidence": float,
                "warnings": [str, ...],
            }}
        """
        results = {}

        with get_db(self.db_path) as conn:
            # Load game metadata
            games = self._load_game_info(conn, game_ids)

            # Load latest injury reports (today's)
            injuries_by_team = self._load_current_injuries(conn)

            for game_id in game_ids:
                game = games.get(game_id)
                if not game:
                    logger.warning(f"Game {game_id} not found in DB")
                    continue

                home_team = game["home_team"]
                away_team = game["away_team"]
                warnings = []

                # Get recent roster for each team
                home_roster = self._get_recent_roster(
                    conn, home_team, game["date_time_utc"]
                )
                away_roster = self._get_recent_roster(
                    conn, away_team, game["date_time_utc"]
                )

                # Apply injury subtraction
                home_out = injuries_by_team.get(home_team, {}).get("out", set())
                away_out = injuries_by_team.get(away_team, {}).get("out", set())
                home_questionable = injuries_by_team.get(home_team, {}).get(
                    "questionable", set()
                )
                away_questionable = injuries_by_team.get(away_team, {}).get(
                    "questionable", set()
                )

                home_players = [
                    p for p in home_roster if p["player_id"] not in home_out
                ]
                away_players = [
                    p for p in away_roster if p["player_id"] not in away_out
                ]

                # Confidence: reduce for each questionable player on roster
                home_n_questionable = sum(
                    1 for p in home_players if p["player_id"] in home_questionable
                )
                away_n_questionable = sum(
                    1 for p in away_players if p["player_id"] in away_questionable
                )
                home_confidence = max(0.5, 1.0 - 0.05 * home_n_questionable)
                away_confidence = max(0.5, 1.0 - 0.05 * away_n_questionable)

                # Warnings
                if len(home_players) < 10:
                    warnings.append(
                        f"{home_team}: only {len(home_players)} players projected"
                    )
                if len(away_players) < 10:
                    warnings.append(
                        f"{away_team}: only {len(away_players)} players projected"
                    )
                if home_out:
                    warnings.append(f"{home_team}: {len(home_out)} player(s) out")
                if away_out:
                    warnings.append(f"{away_team}: {len(away_out)} player(s) out")

                # Cap at 15 players (model max), sorted by avg minutes
                results[game_id] = {
                    "home_players": [p["player_id"] for p in home_players[:15]],
                    "away_players": [p["player_id"] for p in away_players[:15]],
                    "home_team": home_team,
                    "away_team": away_team,
                    "home_confidence": home_confidence,
                    "away_confidence": away_confidence,
                    "warnings": warnings,
                }

        return results

    def _load_game_info(self, conn, game_ids: list[str]) -> dict[str, dict]:
        """Load basic game info for the given IDs."""
        placeholders = ",".join("?" * len(game_ids))
        rows = conn.execute(
            f"""
            SELECT game_id, home_team, away_team, date_time_utc, status
            FROM Games
            WHERE game_id IN ({placeholders})
            """,
            game_ids,
        ).fetchall()
        return {
            row[0]: {
                "home_team": row[1],
                "away_team": row[2],
                "date_time_utc": row[3],
                "status": row[4],
            }
            for row in rows
        }

    def _get_recent_roster(
        self,
        conn,
        team_abbr: str,
        before_date: str,
    ) -> list[dict]:
        """
        Get the most recent active roster for a team, sorted by avg minutes.

        Returns list of {player_id, avg_minutes} dicts.
        """
        # Step 1: Find the team's last 3 game IDs
        game_rows = conn.execute(
            """
            SELECT game_id
            FROM Games
            WHERE (home_team = ? OR away_team = ?)
            AND status = 3
            AND date_time_utc < ?
            ORDER BY date_time_utc DESC
            LIMIT 3
            """,
            (team_abbr, team_abbr, before_date),
        ).fetchall()

        if not game_rows:
            return []

        game_ids = [row[0] for row in game_rows]

        # Step 2: Get team_id for this abbreviation
        team_row = conn.execute(
            "SELECT team_id FROM Teams WHERE abbreviation = ?",
            (team_abbr,),
        ).fetchone()
        if not team_row:
            return []
        team_id = team_row[0]

        # Step 3: Get players from those games for this team
        placeholders = ",".join("?" * len(game_ids))
        rows = conn.execute(
            f"""
            SELECT player_id, AVG(min) as avg_min
            FROM PlayerBox
            WHERE game_id IN ({placeholders})
            AND team_id = ?
            AND min > 0
            GROUP BY player_id
            ORDER BY avg_min DESC
            """,
            (*game_ids, team_id),
        ).fetchall()

        roster = {int(row[0]): float(row[1]) for row in rows}

        # Reconcile with the NBA's current player list (Players.team, refreshed
        # daily): drop players who have since left the team, and add players who
        # have joined but have not yet played for it (offseason moves, trades).
        current = {
            int(row[0])
            for row in conn.execute(
                "SELECT person_id FROM Players WHERE team = ? AND roster_status = 1",
                (team_abbr,),
            )
        }
        if current:
            roster = {pid: m for pid, m in roster.items() if pid in current}
            for pid in current - roster.keys():
                # Order new arrivals by their own recent minutes, wherever played
                row = conn.execute(
                    """
                    SELECT AVG(pb.min) FROM PlayerBox pb
                    JOIN Games g ON g.game_id = pb.game_id
                    WHERE pb.player_id = ? AND pb.min > 0 AND g.date_time_utc < ?
                    ORDER BY g.date_time_utc DESC LIMIT 3
                    """,
                    (pid, before_date),
                ).fetchone()
                if row and row[0] is not None:
                    roster[pid] = float(row[0])

        return [
            {"player_id": pid, "avg_minutes": m}
            for pid, m in sorted(roster.items(), key=lambda x: -x[1])
        ]

    def _load_current_injuries(self, conn) -> dict[str, dict[str, set[int]]]:
        """
        Load the most recent injury reports, grouped by team.

        Returns {team_abbr: {"out": {pid, ...}, "questionable": {pid, ...}}}.
        """
        # Get the most recent report date
        row = conn.execute("SELECT MAX(report_timestamp) FROM InjuryReports").fetchone()
        if not row or not row[0]:
            return {}

        latest_date = row[0][:10]  # YYYY-MM-DD

        # Load all injuries from that date
        rows = conn.execute(
            """
            SELECT nba_player_id, team, status
            FROM InjuryReports
            WHERE report_timestamp LIKE ?
            AND nba_player_id IS NOT NULL
            """,
            (f"{latest_date}%",),
        ).fetchall()

        result: dict[str, dict[str, set[int]]] = {}
        for pid, team, status in rows:
            if team not in result:
                result[team] = {"out": set(), "questionable": set()}

            status_lower = (status or "").lower()
            if status_lower == "out":
                result[team]["out"].add(int(pid))
            elif status_lower in ("questionable", "doubtful"):
                result[team]["questionable"].add(int(pid))

        return result
