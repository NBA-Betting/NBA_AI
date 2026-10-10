"""
Roster Assembler: project who will play in upcoming games.

Strategy: "injury subtraction"
  1. For each team, take the players who logged minutes in its last 3 games
  2. Reconcile with who is on the team now: drop departures, add arrivals
  3. Subtract players listed as "Out" or "Doubtful" in the game-day InjuryReports
  4. "Questionable" players are included (they play ~70% of the time)
  5. Sort by average recent minutes to identify rotation order

Team membership (step 2) comes from the current Players table for upcoming
games. For past games (historical=True: training data and backtests) it is
inferred from appearances, using only roster facts public before tip-off:
a player whose next appearance is for another team has left; one whose next
appearance (within ARRIVAL_WINDOW_DAYS) is for this team after playing
elsewhere, or nowhere, has arrived. Both modes share every other step.

Usage:
    from src.pipeline.roster_assembler import RosterAssembler
    ra = RosterAssembler()
    rosters = ra.get_projected_rosters(["0022500900", "0022500901"])
"""

from __future__ import annotations

import logging
from bisect import bisect_left
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

from src.database import DB_PATH, get_db
from src.phase5.arena_data import franchise_abbrs

logger = logging.getLogger(__name__)
# Keep players without NBA history in the roster, below established rotation players.
NEW_PLAYER_MINUTES = 5.0
# Historical mode: how far ahead an arrival's first game for the team may be
ARRIVAL_WINDOW_DAYS = 30


class RosterAssembler:
    """Projects rosters for upcoming games using injury subtraction."""

    def __init__(self, db_path: str | Path = DB_PATH, historical: bool = False):
        self.db_path = str(db_path)
        self.historical = historical
        self._appearances = None  # historical mode, loaded on first use
        self._team_appearances = None

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

            injuries_by_date = {}

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

                # Apply injury subtraction. Matched by player ID: a player is on
                # only one roster, so the report's team label isn't needed.
                game_date = (
                    datetime.fromisoformat(game["date_time_utc"].replace("Z", "+00:00"))
                    .astimezone(ZoneInfo("America/New_York"))
                    .date()
                )
                if game_date not in injuries_by_date:
                    injuries_by_date[game_date] = self._load_injuries(conn, game_date)
                injuries = injuries_by_date[game_date]
                home_out = {p["player_id"] for p in home_roster} & injuries["out"]
                away_out = {p["player_id"] for p in away_roster} & injuries["out"]
                home_questionable = injuries["questionable"]
                away_questionable = injuries["questionable"]

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
        # Games keeps the abbreviation of the day (SEA), Teams today's (OKC)
        abbrs = franchise_abbrs(team_abbr)
        team_row = conn.execute(
            "SELECT team_id FROM Teams WHERE abbreviation = ?",
            (abbrs[0],),
        ).fetchone()
        if not team_row:
            return []
        team_id = team_row[0]

        # Step 1: Find the team's last 3 games with usable player boxes
        q = ",".join("?" * len(abbrs))
        game_rows = conn.execute(
            f"""
            SELECT g.game_id
            FROM Games g
            WHERE (home_team IN ({q}) OR away_team IN ({q}))
            AND status = 3
            AND date_time_utc < ?
            AND season_type IN ('Regular Season', 'Post Season')
            AND EXISTS (
                SELECT 1 FROM PlayerBox pb
                WHERE pb.game_id = g.game_id AND pb.team_id = ? AND pb.min > 0
            )
            ORDER BY date_time_utc DESC
            LIMIT 3
            """,
            (*abbrs, *abbrs, before_date, team_id),
        ).fetchall()

        game_ids = [row[0] for row in game_rows]

        # Step 2: Get players from those games for this team
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

        # Reconcile with who is on the team now: drop players who have left,
        # add players who joined but haven't played for it (moves, trades)
        if self.historical:
            current = self._historical_members(conn, int(team_id), before_date, roster)
        else:
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
                roster[pid] = self._recent_minutes(conn, pid, before_date)

        return [
            {"player_id": pid, "avg_minutes": m}
            for pid, m in sorted(roster.items(), key=lambda x: (-x[1], x[0]))
        ]

    def _recent_minutes(self, conn, pid: int, before_date: str) -> float:
        """Average minutes over a player's last 3 games before the date."""
        if self.historical:
            dates, _, minutes = self._load_appearances(conn).get(pid, ([], [], []))
            recent = minutes[: bisect_left(dates, before_date)][-3:]
            return sum(recent) / len(recent) if recent else NEW_PLAYER_MINUTES
        row = conn.execute(
            """
            SELECT AVG(min) FROM (
                SELECT pb.min FROM PlayerBox pb
                JOIN Games g ON g.game_id = pb.game_id
                WHERE pb.player_id = ? AND pb.min > 0
                AND g.date_time_utc < ? AND g.status = 3
                AND g.season_type IN ('Regular Season', 'Post Season')
                ORDER BY g.date_time_utc DESC LIMIT 3
            )
            """,
            (pid, before_date),
        ).fetchone()
        return float(row[0]) if row and row[0] is not None else NEW_PLAYER_MINUTES

    def _load_appearances(self, conn) -> dict[int, tuple[list, list, list]]:
        """{player_id: (dates, team_ids, minutes)} of every regular/postseason
        appearance, chronologically (historical mode)."""
        if self._appearances is None:
            apps: dict[int, tuple[list, list, list]] = {}
            by_team: dict[int, list[tuple[str, int]]] = {}
            for pid, team_id, minutes, date_utc in conn.execute(
                """
                SELECT pb.player_id, pb.team_id, pb.min, g.date_time_utc
                FROM PlayerBox pb JOIN Games g ON g.game_id = pb.game_id
                WHERE pb.min > 0 AND g.status = 3
                AND g.season_type IN ('Regular Season', 'Post Season')
                ORDER BY g.date_time_utc
                """
            ):
                dates, teams, mins = apps.setdefault(int(pid), ([], [], []))
                dates.append(date_utc)
                teams.append(int(team_id))
                mins.append(float(minutes))
                by_team.setdefault(int(team_id), []).append((date_utc, int(pid)))
            self._appearances = apps
            self._team_appearances = by_team
        return self._appearances

    def _historical_members(
        self, conn, team_id: int, before_date: str, recent: dict
    ) -> set[int]:
        """Team members as of a past tip-off, from appearances (see module doc)."""
        apps = self._load_appearances(conn)
        window_end = (
            datetime.fromisoformat(before_date.replace("Z", "+00:00"))
            + timedelta(days=ARRIVAL_WINDOW_DAYS)
        ).strftime("%Y-%m-%dT%H:%M:%SZ")
        members = set()
        # Recent players stay unless their next appearance is for another team
        for pid in recent:
            dates, teams, _ = apps.get(pid, ([], [], []))
            i = bisect_left(dates, before_date)
            if i == len(teams) or teams[i] == team_id:
                members.add(pid)
        # Arrivals: next appearance is for this team, previous one was not
        team_apps = self._team_appearances.get(team_id, [])
        start = bisect_left(team_apps, (before_date, 0))
        end = bisect_left(team_apps, (window_end, 0))
        for _, pid in team_apps[start:end]:
            dates, teams, _ = apps[pid]
            i = bisect_left(dates, before_date)
            if teams[i] == team_id and (i == 0 or teams[i - 1] != team_id):
                members.add(pid)
        return members

    def _load_injuries(self, conn, game_date) -> dict[str, set[int]]:
        """
        Injury statuses for a game date (Eastern): the latest report covering
        each team's game that day (that day's, or the previous evening's for
        teams that haven't submitted yet).

        Returns {"out": {pid, ...}, "questionable": {pid, ...}}. Doubtful
        players are treated as out.
        """
        result: dict[str, set[int]] = {"out": set(), "questionable": set()}
        rows = conn.execute(
            """
            SELECT nba_player_id, status FROM InjuryReports
            WHERE game_date = ? AND source = 'NBA_Official'
            AND nba_player_id IS NOT NULL
            """,
            (game_date.isoformat(),),
        ).fetchall()

        for pid, status in rows:
            status_lower = (status or "").lower()
            if status_lower in ("out", "doubtful"):
                result["out"].add(int(pid))
            elif status_lower == "questionable":
                result["questionable"].add(int(pid))

        return result
