"""
health_check.py

CLI tool for validating NBA AI data pipeline health.

This tool performs comprehensive validation of data completeness and quality
across all pipeline stages. It operates in two phases:

Phase 1: Run the full data pipeline (optional, can be skipped with --skip-pipeline)
Phase 2: Season-level database validation with checks for:
    - Completeness: Expected record counts per table
    - Structure/Values: Data format and range validation
    - Flag Consistency: Flags match underlying data state
    - Referential Integrity: No orphaned records
    - Temporal: Status matches game dates

Usage:
    python -m src.health_check --season=2025-2026
    python -m src.health_check --season=2025-2026 --skip-pipeline --json
    python -m src.health_check --season=2025-2026 --format=table

Exit Codes:
    0: All checks passed
    1: Warnings detected (non-critical issues)
    2: Critical failures detected
"""

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from src.config import config
from src.database import get_db
from src.database_updater.betting import CLOSING_SPREAD_SQL
from src.logging_config import setup_logging
from src.utils import get_current_eastern_date, to_eastern_date, validate_season_format

# Configuration
DB_PATH = config["database"]["path"]


# =============================================================================
# Data Classes
# =============================================================================


class CheckStatus(Enum):
    """Status of a health check."""

    PASS = "pass"
    WARN = "warn"
    CRITICAL = "critical"
    SKIP = "skip"


@dataclass
class CheckResult:
    """Result of a single health check."""

    stage: str  # Pipeline stage (e.g., "Games", "PbP", "GameStates")
    category: str  # Check category (e.g., "completeness", "flag_consistency")
    check_name: str  # Specific check name
    status: CheckStatus
    message: str
    expected: Optional[Any] = None
    actual: Optional[Any] = None
    details: Optional[dict] = None
    query_time_ms: Optional[float] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON output."""
        return {
            "stage": self.stage,
            "category": self.category,
            "check_name": self.check_name,
            "status": self.status.value,
            "message": self.message,
            "expected": self.expected,
            "actual": self.actual,
            "details": self.details,
            "query_time_ms": self.query_time_ms,
        }


@dataclass
class HealthReport:
    """Aggregated health check report."""

    season: str
    results: list[CheckResult] = field(default_factory=list)
    pipeline_ran: bool = False
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    def add(self, result: CheckResult):
        """Add a check result."""
        self.results.append(result)

    @property
    def passed(self) -> int:
        """Count of passed checks."""
        return sum(1 for r in self.results if r.status == CheckStatus.PASS)

    @property
    def warnings(self) -> int:
        """Count of warning checks."""
        return sum(1 for r in self.results if r.status == CheckStatus.WARN)

    @property
    def critical(self) -> int:
        """Count of critical checks."""
        return sum(1 for r in self.results if r.status == CheckStatus.CRITICAL)

    @property
    def skipped(self) -> int:
        """Count of skipped checks."""
        return sum(1 for r in self.results if r.status == CheckStatus.SKIP)

    @property
    def exit_code(self) -> int:
        """
        Determine exit code based on results.
        0 = all pass, 1 = warnings, 2 = critical
        """
        if self.critical > 0:
            return 2
        elif self.warnings > 0:
            return 1
        return 0

    def summary_table(self) -> str:
        """Generate summary table output."""
        lines = []
        lines.append("")
        lines.append("=" * 80)
        lines.append(f"HEALTH CHECK REPORT: {self.season}")
        lines.append("=" * 80)

        # Group by stage
        stages = {}
        for r in self.results:
            if r.stage not in stages:
                stages[r.stage] = []
            stages[r.stage].append(r)

        for stage, checks in stages.items():
            stage_status = CheckStatus.PASS
            for c in checks:
                if c.status == CheckStatus.CRITICAL:
                    stage_status = CheckStatus.CRITICAL
                    break
                elif (
                    c.status == CheckStatus.WARN
                    and stage_status != CheckStatus.CRITICAL
                ):
                    stage_status = CheckStatus.WARN

            status_icon = {"pass": "✓", "warn": "⚠", "critical": "✗", "skip": "○"}[
                stage_status.value
            ]
            lines.append(f"\n{status_icon} {stage}")
            lines.append("-" * 40)

            for c in checks:
                icon = {"pass": "✓", "warn": "⚠", "critical": "✗", "skip": "○"}[
                    c.status.value
                ]
                lines.append(f"  {icon} [{c.category}] {c.check_name}: {c.message}")
                if c.expected is not None and c.actual is not None:
                    lines.append(f"      Expected: {c.expected}, Actual: {c.actual}")

        lines.append("")
        lines.append("=" * 80)
        lines.append(
            f"SUMMARY: {self.passed} passed, {self.warnings} warnings, "
            f"{self.critical} critical, {self.skipped} skipped"
        )
        if self.start_time and self.end_time:
            duration = (self.end_time - self.start_time).total_seconds()
            lines.append(f"Duration: {duration:.1f}s")
        lines.append("=" * 80)

        return "\n".join(lines)

    def to_json(self) -> str:
        """Generate JSON output."""
        return json.dumps(
            {
                "season": self.season,
                "pipeline_ran": self.pipeline_ran,
                "start_time": self.start_time.isoformat() if self.start_time else None,
                "end_time": self.end_time.isoformat() if self.end_time else None,
                "summary": {
                    "passed": self.passed,
                    "warnings": self.warnings,
                    "critical": self.critical,
                    "skipped": self.skipped,
                    "exit_code": self.exit_code,
                },
                "results": [r.to_dict() for r in self.results],
            },
            indent=2,
        )


# =============================================================================
# Season Health Checker
# =============================================================================


class SeasonHealthChecker:
    """
    Performs comprehensive health checks for a single season.

    Checks are organized by pipeline stage and check category.
    """

    # Expected games per season (approximations)
    # Regular seasons: 1230 games (30 teams × 82 games ÷ 2)
    # Shortened seasons: 2011-12 (990), 2019-20 (~1059), 2020-21 (1080)
    SHORTENED_SEASONS = {
        "2011-2012": 990,
        "2019-2020": 1059,
        "2020-2021": 1080,
    }
    DEFAULT_REGULAR_SEASON_GAMES = 1230

    # Per-game expected record counts
    PBP_MIN_PLAYS = 200
    PBP_MAX_PLAYS = 800
    GAMESTATES_MIN = 200
    GAMESTATES_MAX = 800
    PLAYERBOX_MIN = 22
    PLAYERBOX_MAX = 36
    TEAMBOX_EXPECTED = 2

    # Data availability by season
    PLAYERBOX_START_SEASON = "2023-2024"
    BETTING_START_SEASON = "2007-2008"
    INJURY_START_SEASON = (
        "2023-2024"  # NBA Official injury PDFs collected from this season
    )

    def __init__(self, season: str, db_path: str = DB_PATH):
        """
        Initialize the checker.

        Args:
            season: Season string (e.g., "2024-2025")
            db_path: Path to the SQLite database
        """
        self.season = season
        self.db_path = db_path
        self.report = HealthReport(season=season)

    def run_all(self) -> HealthReport:
        """Run all health checks and return the report."""
        self.report.start_time = datetime.now()

        logging.info(f"Running health checks for {self.season}...")

        # Run checks by stage
        self._check_games()
        self._check_pbp()
        self._check_game_states()
        self._check_boxscores()
        self._check_features()
        self._check_predictions()
        self._check_betting()
        self._check_injuries()
        self._check_players()
        self._check_flag_consistency()

        self.report.end_time = datetime.now()
        return self.report

    def _timed_query(self, cursor, query: str, params: tuple = ()) -> tuple:
        """Execute a query and return (result, time_ms)."""
        start = time.time()
        cursor.execute(query, params)
        result = cursor.fetchall()
        elapsed_ms = (time.time() - start) * 1000
        return result, elapsed_ms

    def _add_result(
        self,
        stage: str,
        category: str,
        check_name: str,
        status: CheckStatus,
        message: str,
        expected=None,
        actual=None,
        details=None,
        query_time_ms=None,
    ):
        """Add a check result to the report."""
        self.report.add(
            CheckResult(
                stage=stage,
                category=category,
                check_name=check_name,
                status=status,
                message=message,
                expected=expected,
                actual=actual,
                details=details,
                query_time_ms=query_time_ms,
            )
        )

    # -------------------------------------------------------------------------
    # Games Checks
    # -------------------------------------------------------------------------
    def _check_games(self):
        """Check Games table completeness and structure."""
        stage = "Games"

        with get_db(self.db_path) as conn:
            cursor = conn.cursor()

            # 1. Check game count
            result, qtime = self._timed_query(
                cursor,
                """
                SELECT COUNT(*) FROM Games
                WHERE season = ?
                AND season_type IN ('Regular Season', 'Post Season')
                """,
                (self.season,),
            )
            game_count = result[0][0]

            expected = self.SHORTENED_SEASONS.get(
                self.season, self.DEFAULT_REGULAR_SEASON_GAMES
            )
            # Allow for post-season variation (up to ~100 playoff games)
            min_expected = expected * 0.95
            max_expected = expected + 120  # Include playoffs

            if game_count >= min_expected:
                self._add_result(
                    stage,
                    "completeness",
                    "game_count",
                    CheckStatus.PASS,
                    f"Found {game_count} games",
                    expected=f"{int(min_expected)}-{int(max_expected)}",
                    actual=game_count,
                    query_time_ms=qtime,
                )
            elif game_count > 0:
                self._add_result(
                    stage,
                    "completeness",
                    "game_count",
                    CheckStatus.WARN,
                    f"Low game count: {game_count}",
                    expected=f"{int(min_expected)}-{int(max_expected)}",
                    actual=game_count,
                    query_time_ms=qtime,
                )
            else:
                self._add_result(
                    stage,
                    "completeness",
                    "game_count",
                    CheckStatus.CRITICAL,
                    "No games found",
                    expected=f">{int(min_expected)}",
                    actual=0,
                    query_time_ms=qtime,
                )

            # 2. Check for duplicate game_ids
            result, qtime = self._timed_query(
                cursor,
                """
                SELECT game_id, COUNT(*) as cnt
                FROM Games WHERE season = ?
                GROUP BY game_id HAVING cnt > 1
                """,
                (self.season,),
            )
            if len(result) == 0:
                self._add_result(
                    stage,
                    "integrity",
                    "no_duplicate_game_ids",
                    CheckStatus.PASS,
                    "No duplicate game_ids",
                    query_time_ms=qtime,
                )
            else:
                self._add_result(
                    stage,
                    "integrity",
                    "no_duplicate_game_ids",
                    CheckStatus.CRITICAL,
                    f"Found {len(result)} duplicate game_ids",
                    details={"duplicates": [r[0] for r in result[:10]]},
                    query_time_ms=qtime,
                )

            # 3. Check status distribution for completed games
            result, qtime = self._timed_query(
                cursor,
                """
                SELECT status, COUNT(*) as cnt
                FROM Games WHERE season = ?
                AND season_type IN ('Regular Season', 'Post Season')
                GROUP BY status
                """,
                (self.season,),
            )
            status_dist = {r[0]: r[1] for r in result}

            # Status 3 = Final
            final_count = status_dist.get(3, 0)
            today = get_current_eastern_date()

            # Check if season has started
            result2, _ = self._timed_query(
                cursor,
                """
                SELECT MIN(date_time_utc) FROM Games
                WHERE season = ? AND season_type IN ('Regular Season', 'Post Season')
                """,
                (self.season,),
            )
            first_game_date = result2[0][0] and to_eastern_date(result2[0][0])

            if first_game_date and today.isoformat() > first_game_date:
                # Season has started, expect some final games
                if final_count > 0:
                    self._add_result(
                        stage,
                        "temporal",
                        "games_finalized",
                        CheckStatus.PASS,
                        f"{final_count} games with status=Final",
                        details=status_dist,
                        query_time_ms=qtime,
                    )
                else:
                    self._add_result(
                        stage,
                        "temporal",
                        "games_finalized",
                        CheckStatus.WARN,
                        "No games with status=Final despite season start",
                        details=status_dist,
                        query_time_ms=qtime,
                    )
            else:
                self._add_result(
                    stage,
                    "temporal",
                    "games_finalized",
                    CheckStatus.SKIP,
                    "Season not yet started",
                    query_time_ms=qtime,
                )

    # -------------------------------------------------------------------------
    # PbP Checks
    # -------------------------------------------------------------------------
    def _check_pbp(self):
        """Check PbP_Logs table completeness."""
        stage = "PbP"

        with get_db(self.db_path) as conn:
            cursor = conn.cursor()

            # 1. Get completed games count
            result, _ = self._timed_query(
                cursor,
                """
                SELECT COUNT(*) FROM Games
                WHERE season = ? AND status = 3
                AND season_type IN ('Regular Season', 'Post Season')
                """,
                (self.season,),
            )
            completed_games = result[0][0]

            if completed_games == 0:
                self._add_result(
                    stage,
                    "completeness",
                    "pbp_coverage",
                    CheckStatus.SKIP,
                    "No completed games to check",
                )
                return

            # 2. Check PbP coverage for completed games
            result, qtime = self._timed_query(
                cursor,
                """
                SELECT COUNT(DISTINCT g.game_id)
                FROM Games g
                JOIN PbP_Logs p ON g.game_id = p.game_id
                WHERE g.season = ? AND g.status = 3
                AND g.season_type IN ('Regular Season', 'Post Season')
                """,
                (self.season,),
            )
            pbp_games = result[0][0]
            coverage_pct = (pbp_games / completed_games * 100) if completed_games else 0

            if coverage_pct >= 99:
                self._add_result(
                    stage,
                    "completeness",
                    "pbp_coverage",
                    CheckStatus.PASS,
                    f"{pbp_games}/{completed_games} games have PbP ({coverage_pct:.1f}%)",
                    expected=completed_games,
                    actual=pbp_games,
                    query_time_ms=qtime,
                )
            elif coverage_pct >= 90:
                self._add_result(
                    stage,
                    "completeness",
                    "pbp_coverage",
                    CheckStatus.WARN,
                    f"PbP coverage at {coverage_pct:.1f}%",
                    expected=completed_games,
                    actual=pbp_games,
                    query_time_ms=qtime,
                )
            else:
                self._add_result(
                    stage,
                    "completeness",
                    "pbp_coverage",
                    CheckStatus.CRITICAL,
                    f"Low PbP coverage: {coverage_pct:.1f}%",
                    expected=completed_games,
                    actual=pbp_games,
                    query_time_ms=qtime,
                )

            # 3. Check play count distribution
            result, qtime = self._timed_query(
                cursor,
                """
                SELECT p.game_id, COUNT(*) as play_count
                FROM PbP_Logs p
                JOIN Games g ON p.game_id = g.game_id
                WHERE g.season = ? AND g.status = 3
                AND g.season_type IN ('Regular Season', 'Post Season')
                GROUP BY p.game_id
                HAVING play_count < ? OR play_count > ?
                """,
                (self.season, self.PBP_MIN_PLAYS, self.PBP_MAX_PLAYS),
            )
            outliers = len(result)

            if outliers == 0:
                self._add_result(
                    stage,
                    "structure",
                    "play_count_range",
                    CheckStatus.PASS,
                    f"All games have {self.PBP_MIN_PLAYS}-{self.PBP_MAX_PLAYS} plays",
                    query_time_ms=qtime,
                )
            else:
                # Few outliers are OK (overtime games, etc.)
                outlier_pct = (outliers / pbp_games * 100) if pbp_games else 0
                if outlier_pct <= 5:
                    self._add_result(
                        stage,
                        "structure",
                        "play_count_range",
                        CheckStatus.PASS,
                        f"{outliers} games with unusual play counts ({outlier_pct:.1f}%)",
                        details={"outlier_games": [r[0] for r in result[:5]]},
                        query_time_ms=qtime,
                    )
                else:
                    self._add_result(
                        stage,
                        "structure",
                        "play_count_range",
                        CheckStatus.WARN,
                        f"{outliers} games with unusual play counts ({outlier_pct:.1f}%)",
                        details={"outlier_games": [r[0] for r in result[:10]]},
                        query_time_ms=qtime,
                    )

    # -------------------------------------------------------------------------
    # GameStates Checks
    # -------------------------------------------------------------------------
    def _check_game_states(self):
        """Check GameStates table completeness and structure."""
        stage = "GameStates"

        with get_db(self.db_path) as conn:
            cursor = conn.cursor()

            # 1. Get games with game_data_finalized=1
            result, _ = self._timed_query(
                cursor,
                """
                SELECT COUNT(*) FROM Games
                WHERE season = ? AND game_data_finalized = 1
                AND season_type IN ('Regular Season', 'Post Season')
                """,
                (self.season,),
            )
            finalized_games = result[0][0]

            if finalized_games == 0:
                self._add_result(
                    stage,
                    "completeness",
                    "gamestates_coverage",
                    CheckStatus.SKIP,
                    "No finalized games to check",
                )
                return

            # 2. Check GameStates coverage
            result, qtime = self._timed_query(
                cursor,
                """
                SELECT COUNT(DISTINCT g.game_id)
                FROM Games g
                JOIN GameStates gs ON g.game_id = gs.game_id
                WHERE g.season = ? AND g.game_data_finalized = 1
                AND g.season_type IN ('Regular Season', 'Post Season')
                """,
                (self.season,),
            )
            gs_games = result[0][0]
            coverage_pct = (gs_games / finalized_games * 100) if finalized_games else 0

            if coverage_pct >= 99:
                self._add_result(
                    stage,
                    "completeness",
                    "gamestates_coverage",
                    CheckStatus.PASS,
                    f"{gs_games}/{finalized_games} finalized games have GameStates",
                    expected=finalized_games,
                    actual=gs_games,
                    query_time_ms=qtime,
                )
            else:
                self._add_result(
                    stage,
                    "completeness",
                    "gamestates_coverage",
                    CheckStatus.CRITICAL,
                    f"GameStates missing for finalized games: {coverage_pct:.1f}%",
                    expected=finalized_games,
                    actual=gs_games,
                    query_time_ms=qtime,
                )

            # 3. Check is_final_state uniqueness
            result, qtime = self._timed_query(
                cursor,
                """
                SELECT g.game_id, COUNT(*) as final_count
                FROM Games g
                JOIN GameStates gs ON g.game_id = gs.game_id
                WHERE g.season = ? AND g.game_data_finalized = 1
                AND gs.is_final_state = 1
                AND g.season_type IN ('Regular Season', 'Post Season')
                GROUP BY g.game_id
                HAVING final_count != 1
                """,
                (self.season,),
            )
            violations = len(result)

            if violations == 0:
                self._add_result(
                    stage,
                    "structure",
                    "is_final_state_unique",
                    CheckStatus.PASS,
                    "Each finalized game has exactly 1 final state",
                    query_time_ms=qtime,
                )
            else:
                self._add_result(
                    stage,
                    "structure",
                    "is_final_state_unique",
                    CheckStatus.CRITICAL,
                    f"{violations} games have multiple or missing final states",
                    details={"games": [r[0] for r in result[:10]]},
                    query_time_ms=qtime,
                )

            # 4. Check state count distribution
            result, qtime = self._timed_query(
                cursor,
                """
                SELECT gs.game_id, COUNT(*) as state_count
                FROM GameStates gs
                JOIN Games g ON gs.game_id = g.game_id
                WHERE g.season = ? AND g.game_data_finalized = 1
                AND g.season_type IN ('Regular Season', 'Post Season')
                GROUP BY gs.game_id
                HAVING state_count < ? OR state_count > ?
                """,
                (self.season, self.GAMESTATES_MIN, self.GAMESTATES_MAX),
            )
            outliers = len(result)

            if outliers == 0:
                self._add_result(
                    stage,
                    "structure",
                    "state_count_range",
                    CheckStatus.PASS,
                    f"All games have {self.GAMESTATES_MIN}-{self.GAMESTATES_MAX} states",
                    query_time_ms=qtime,
                )
            else:
                outlier_pct = (outliers / gs_games * 100) if gs_games else 0
                if outlier_pct <= 5:
                    self._add_result(
                        stage,
                        "structure",
                        "state_count_range",
                        CheckStatus.PASS,
                        f"{outliers} games with unusual state counts",
                        query_time_ms=qtime,
                    )
                else:
                    self._add_result(
                        stage,
                        "structure",
                        "state_count_range",
                        CheckStatus.WARN,
                        f"{outliers} games with unusual state counts ({outlier_pct:.1f}%)",
                        query_time_ms=qtime,
                    )

    # -------------------------------------------------------------------------
    # Boxscores Checks
    # -------------------------------------------------------------------------
    def _check_boxscores(self):
        """Check PlayerBox and TeamBox tables."""
        stage = "Boxscores"

        with get_db(self.db_path) as conn:
            cursor = conn.cursor()

            # Check if this season should have boxscore data
            if self.season < self.PLAYERBOX_START_SEASON:
                self._add_result(
                    stage,
                    "completeness",
                    "boxscore_coverage",
                    CheckStatus.SKIP,
                    f"BoxScore data only available from {self.PLAYERBOX_START_SEASON}",
                )
                return

            # 1. Get games with boxscore_data_finalized=1
            result, _ = self._timed_query(
                cursor,
                """
                SELECT COUNT(*) FROM Games
                WHERE season = ? AND boxscore_data_finalized = 1
                AND season_type IN ('Regular Season', 'Post Season')
                """,
                (self.season,),
            )
            finalized_games = result[0][0]

            if finalized_games == 0:
                self._add_result(
                    stage,
                    "completeness",
                    "boxscore_coverage",
                    CheckStatus.SKIP,
                    "No games with boxscore_data_finalized=1",
                )
                return

            # 2. Check PlayerBox coverage
            result, qtime = self._timed_query(
                cursor,
                """
                SELECT COUNT(DISTINCT g.game_id)
                FROM Games g
                JOIN PlayerBox pb ON g.game_id = pb.game_id
                WHERE g.season = ? AND g.boxscore_data_finalized = 1
                AND g.season_type IN ('Regular Season', 'Post Season')
                """,
                (self.season,),
            )
            pb_games = result[0][0]
            coverage_pct = (pb_games / finalized_games * 100) if finalized_games else 0

            if coverage_pct >= 99:
                self._add_result(
                    stage,
                    "completeness",
                    "playerbox_coverage",
                    CheckStatus.PASS,
                    f"{pb_games}/{finalized_games} games have PlayerBox data",
                    expected=finalized_games,
                    actual=pb_games,
                    query_time_ms=qtime,
                )
            else:
                self._add_result(
                    stage,
                    "completeness",
                    "playerbox_coverage",
                    CheckStatus.CRITICAL,
                    f"PlayerBox missing: {coverage_pct:.1f}%",
                    expected=finalized_games,
                    actual=pb_games,
                    query_time_ms=qtime,
                )

            # 3. Check TeamBox - exactly 2 per game
            result, qtime = self._timed_query(
                cursor,
                """
                SELECT g.game_id, COUNT(*) as team_count
                FROM Games g
                JOIN TeamBox tb ON g.game_id = tb.game_id
                WHERE g.season = ? AND g.boxscore_data_finalized = 1
                AND g.season_type IN ('Regular Season', 'Post Season')
                GROUP BY g.game_id
                HAVING team_count != 2
                """,
                (self.season,),
            )
            violations = len(result)

            if violations == 0:
                self._add_result(
                    stage,
                    "structure",
                    "teambox_count",
                    CheckStatus.PASS,
                    "All games have exactly 2 TeamBox records",
                    query_time_ms=qtime,
                )
            else:
                self._add_result(
                    stage,
                    "structure",
                    "teambox_count",
                    CheckStatus.CRITICAL,
                    f"{violations} games without exactly 2 TeamBox records",
                    details={"games": [r[0] for r in result[:10]]},
                    query_time_ms=qtime,
                )

            # 4. Check PlayerBox count per game
            result, qtime = self._timed_query(
                cursor,
                """
                SELECT pb.game_id, COUNT(*) as player_count
                FROM PlayerBox pb
                JOIN Games g ON pb.game_id = g.game_id
                WHERE g.season = ? AND g.boxscore_data_finalized = 1
                AND g.season_type IN ('Regular Season', 'Post Season')
                GROUP BY pb.game_id
                HAVING player_count < ? OR player_count > ?
                """,
                (self.season, self.PLAYERBOX_MIN, self.PLAYERBOX_MAX),
            )
            outliers = len(result)

            if outliers == 0:
                self._add_result(
                    stage,
                    "structure",
                    "playerbox_count_range",
                    CheckStatus.PASS,
                    f"All games have {self.PLAYERBOX_MIN}-{self.PLAYERBOX_MAX} players",
                    query_time_ms=qtime,
                )
            else:
                outlier_pct = (outliers / pb_games * 100) if pb_games else 0
                if outlier_pct <= 5:
                    self._add_result(
                        stage,
                        "structure",
                        "playerbox_count_range",
                        CheckStatus.PASS,
                        f"{outliers} games with unusual player counts",
                        query_time_ms=qtime,
                    )
                else:
                    self._add_result(
                        stage,
                        "structure",
                        "playerbox_count_range",
                        CheckStatus.WARN,
                        f"{outliers} games with unusual player counts ({outlier_pct:.1f}%)",
                        query_time_ms=qtime,
                    )

            # 4. Final PBP score agrees with the box score (both feed models:
            # GameStates the legacy features, TeamBox the Phase5 features)
            result, qtime = self._timed_query(
                cursor,
                """
                SELECT g.game_id FROM Games g
                JOIN GameStates gs ON gs.game_id = g.game_id AND gs.is_final_state = 1
                JOIN Teams th ON th.abbreviation = g.home_team
                JOIN Teams ta ON ta.abbreviation = g.away_team
                JOIN TeamBox bh ON bh.game_id = g.game_id AND bh.team_id = th.team_id
                JOIN TeamBox ba ON ba.game_id = g.game_id AND ba.team_id = ta.team_id
                WHERE g.season = ? AND g.boxscore_data_finalized = 1
                AND g.season_type IN ('Regular Season', 'Post Season')
                AND (gs.home_score != bh.pts OR gs.away_score != ba.pts)
                """,
                (self.season,),
            )
            mismatched = [r[0] for r in result]
            self._add_result(
                stage,
                "consistency",
                "final_score_matches_box",
                CheckStatus.WARN if mismatched else CheckStatus.PASS,
                (
                    f"{len(mismatched)} games' final PBP score differs from the box score: "
                    f"{', '.join(mismatched[:5])}"
                    if mismatched
                    else "Final PBP scores match the box scores"
                ),
                expected=0,
                actual=len(mismatched),
                query_time_ms=qtime,
            )

    # -------------------------------------------------------------------------
    # Features Checks
    # -------------------------------------------------------------------------
    def _check_features(self):
        """Check Features table completeness."""
        stage = "Features"

        with get_db(self.db_path) as conn:
            cursor = conn.cursor()

            # 1. Get games with pre_game_data_finalized=1
            result, _ = self._timed_query(
                cursor,
                """
                SELECT COUNT(*) FROM Games
                WHERE season = ? AND pre_game_data_finalized = 1
                AND season_type IN ('Regular Season', 'Post Season')
                """,
                (self.season,),
            )
            finalized_games = result[0][0]

            if finalized_games == 0:
                self._add_result(
                    stage,
                    "completeness",
                    "features_coverage",
                    CheckStatus.SKIP,
                    "No games with pre_game_data_finalized=1",
                )
                return

            # 2. Check Features coverage
            result, qtime = self._timed_query(
                cursor,
                """
                SELECT COUNT(DISTINCT g.game_id)
                FROM Games g
                JOIN Features f ON g.game_id = f.game_id
                WHERE g.season = ? AND g.pre_game_data_finalized = 1
                AND g.season_type IN ('Regular Season', 'Post Season')
                """,
                (self.season,),
            )
            f_games = result[0][0]
            coverage_pct = (f_games / finalized_games * 100) if finalized_games else 0

            if coverage_pct >= 99:
                self._add_result(
                    stage,
                    "completeness",
                    "features_coverage",
                    CheckStatus.PASS,
                    f"{f_games}/{finalized_games} games have Features",
                    expected=finalized_games,
                    actual=f_games,
                    query_time_ms=qtime,
                )
            else:
                self._add_result(
                    stage,
                    "completeness",
                    "features_coverage",
                    CheckStatus.CRITICAL,
                    f"Features missing for finalized games: {coverage_pct:.1f}%",
                    expected=finalized_games,
                    actual=f_games,
                    query_time_ms=qtime,
                )

            # 3. Check for NULL or empty feature_set values
            result, qtime = self._timed_query(
                cursor,
                """
                SELECT COUNT(*) FROM Features f
                JOIN Games g ON f.game_id = g.game_id
                WHERE g.season = ? AND g.pre_game_data_finalized = 1
                AND (f.feature_set IS NULL OR f.feature_set = '' OR f.feature_set = '{}')
                AND g.season_type IN ('Regular Season', 'Post Season')
                """,
                (self.season,),
            )
            null_features = result[0][0]

            if null_features == 0:
                self._add_result(
                    stage,
                    "structure",
                    "no_null_features",
                    CheckStatus.PASS,
                    "No NULL/empty feature_set values",
                    query_time_ms=qtime,
                )
            else:
                self._add_result(
                    stage,
                    "structure",
                    "no_null_features",
                    CheckStatus.WARN,
                    f"{null_features} games have NULL/empty feature_set",
                    query_time_ms=qtime,
                )

    # -------------------------------------------------------------------------
    # Predictions Checks
    # -------------------------------------------------------------------------
    def _check_predictions(self):
        """Check Predictions table completeness."""
        stage = "Predictions"

        # Get predictors from config
        predictors = list(config.get("predictors", {}).keys())
        if not predictors:
            self._add_result(
                stage,
                "completeness",
                "predictions_coverage",
                CheckStatus.SKIP,
                "No predictors configured",
            )
            return

        with get_db(self.db_path) as conn:
            cursor = conn.cursor()

            # Games eligible for predictions: pre-game data finalized and
            # already tipped off. Upcoming games are flagged finalized days
            # ahead, which would make the count drift with the schedule.
            now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            result, _ = self._timed_query(
                cursor,
                """
                SELECT COUNT(*) FROM Games
                WHERE season = ? AND pre_game_data_finalized = 1
                AND season_type IN ('Regular Season', 'Post Season')
                AND date_time_utc <= ?
                """,
                (self.season, now_utc),
            )
            eligible_games = result[0][0]

            if eligible_games == 0:
                self._add_result(
                    stage,
                    "completeness",
                    "predictions_coverage",
                    CheckStatus.SKIP,
                    "No games eligible for predictions",
                )
                return

            # Check each predictor
            for predictor in predictors:
                result, qtime = self._timed_query(
                    cursor,
                    """
                    SELECT COUNT(DISTINCT g.game_id)
                    FROM Games g
                    JOIN Predictions p ON g.game_id = p.game_id
                    WHERE g.season = ? AND g.pre_game_data_finalized = 1
                    AND p.predictor = ?
                    AND g.season_type IN ('Regular Season', 'Post Season')
                    AND g.date_time_utc <= ?
                    """,
                    (self.season, predictor, now_utc),
                )
                pred_games = result[0][0]
                coverage_pct = (
                    (pred_games / eligible_games * 100) if eligible_games else 0
                )

                if coverage_pct >= 95:
                    self._add_result(
                        stage,
                        "completeness",
                        f"predictions_{predictor}",
                        CheckStatus.PASS,
                        f"{predictor}: {pred_games}/{eligible_games} games ({coverage_pct:.1f}%)",
                        expected=eligible_games,
                        actual=pred_games,
                        query_time_ms=qtime,
                    )
                elif coverage_pct >= 50:
                    self._add_result(
                        stage,
                        "completeness",
                        f"predictions_{predictor}",
                        CheckStatus.WARN,
                        f"{predictor}: {pred_games}/{eligible_games} games ({coverage_pct:.1f}%)",
                        expected=eligible_games,
                        actual=pred_games,
                        query_time_ms=qtime,
                    )
                else:
                    self._add_result(
                        stage,
                        "completeness",
                        f"predictions_{predictor}",
                        CheckStatus.WARN,
                        f"{predictor}: Low coverage {coverage_pct:.1f}%",
                        expected=eligible_games,
                        actual=pred_games,
                        query_time_ms=qtime,
                    )

    # -------------------------------------------------------------------------
    # Betting Checks
    # -------------------------------------------------------------------------
    def _check_betting(self):
        """Check Betting table completeness."""
        stage = "Betting"

        if self.season < self.BETTING_START_SEASON:
            self._add_result(
                stage,
                "completeness",
                "betting_coverage",
                CheckStatus.SKIP,
                f"Betting data only available from {self.BETTING_START_SEASON}",
            )
            return

        with get_db(self.db_path) as conn:
            cursor = conn.cursor()

            # Get completed games
            result, _ = self._timed_query(
                cursor,
                """
                SELECT COUNT(*) FROM Games
                WHERE season = ? AND status = 3
                AND season_type IN ('Regular Season', 'Post Season')
                """,
                (self.season,),
            )
            completed_games = result[0][0]

            if completed_games == 0:
                self._add_result(
                    stage,
                    "completeness",
                    "betting_coverage",
                    CheckStatus.SKIP,
                    "No completed games to check",
                )
                return

            # Check closing-line coverage (a Betting row can be a placeholder
            # or hold lines the dashboard doesn't read)
            result, qtime = self._timed_query(
                cursor,
                f"""
                SELECT COUNT(DISTINCT g.game_id)
                FROM Games g
                JOIN Betting b ON g.game_id = b.game_id
                WHERE g.season = ? AND g.status = 3
                AND g.season_type IN ('Regular Season', 'Post Season')
                AND {CLOSING_SPREAD_SQL} IS NOT NULL
                """,
                (self.season,),
            )
            betting_games = result[0][0]
            coverage_pct = (
                (betting_games / completed_games * 100) if completed_games else 0
            )

            if coverage_pct >= 95:
                self._add_result(
                    stage,
                    "completeness",
                    "betting_coverage",
                    CheckStatus.PASS,
                    f"{betting_games}/{completed_games} games have a closing line ({coverage_pct:.1f}%)",
                    expected=completed_games,
                    actual=betting_games,
                    query_time_ms=qtime,
                )
            elif coverage_pct >= 50:
                self._add_result(
                    stage,
                    "completeness",
                    "betting_coverage",
                    CheckStatus.WARN,
                    f"Betting coverage: {coverage_pct:.1f}%",
                    expected=completed_games,
                    actual=betting_games,
                    query_time_ms=qtime,
                )
            else:
                self._add_result(
                    stage,
                    "completeness",
                    "betting_coverage",
                    CheckStatus.WARN,
                    f"Low betting coverage: {coverage_pct:.1f}%",
                    expected=completed_games,
                    actual=betting_games,
                    query_time_ms=qtime,
                )

    # -------------------------------------------------------------------------
    # Injuries Checks
    # -------------------------------------------------------------------------
    def _check_injuries(self):
        """Check InjuryReports table completeness."""
        stage = "Injuries"

        if self.season < self.INJURY_START_SEASON:
            self._add_result(
                stage,
                "completeness",
                "injury_coverage",
                CheckStatus.SKIP,
                f"Injury data only collected from {self.INJURY_START_SEASON}",
            )
            return

        with get_db(self.db_path) as conn:
            cursor = conn.cursor()

            # Eastern game days (the schedule's dates; evening tip-offs are
            # already the next day in UTC)
            result, _ = self._timed_query(
                cursor,
                """
                SELECT DISTINCT date_time_utc FROM Games
                WHERE season = ? AND status = 3
                AND season_type IN ('Regular Season', 'Post Season')
                """,
                (self.season,),
            )
            game_days = sorted({to_eastern_date(r[0]) for r in result})

            if not game_days:
                self._add_result(
                    stage,
                    "completeness",
                    "injury_coverage",
                    CheckStatus.SKIP,
                    "No completed games to check",
                )
                return

            result, qtime = self._timed_query(
                cursor,
                """
                SELECT DISTINCT DATE(report_timestamp), game_date FROM InjuryReports
                WHERE source = 'NBA_Official' AND season = ?
                """,
                (self.season,),
            )
            report_days = {r[0] for r in result}
            # A game day is covered when some report lists its games
            # (rows carry the game date they're for)
            covered_days = {r[1] for r in result if r[1]}
            uncovered = [d for d in game_days if d not in covered_days]
            covered = len(game_days) - len(uncovered)
            coverage_pct = covered / len(game_days) * 100
            message = (
                f"{covered}/{len(game_days)} game days have an injury report "
                f"({coverage_pct:.1f}%)"
            )
            if uncovered:
                message += f"; missing: {', '.join(uncovered[-5:])}"

            if coverage_pct >= 95:
                status = CheckStatus.PASS
            elif coverage_pct >= 50:
                status = CheckStatus.WARN
            else:
                status = CheckStatus.CRITICAL
            self._add_result(
                stage,
                "completeness",
                "injury_coverage",
                status,
                message,
                expected=len(game_days),
                actual=covered,
                query_time_ms=qtime,
            )

            # Game days marked fetched in InjuryCache but with no report that
            # day point to failed fetches (off-days legitimately have none, and
            # "empty" reports listed no injuries)
            result, qtime = self._timed_query(
                cursor,
                "SELECT report_date FROM InjuryCache WHERE report_date BETWEEN ? AND ?"
                " AND COALESCE(status, '') != 'empty'",
                (game_days[0], game_days[-1]),
            )
            empty = sorted(
                r[0]
                for r in result
                if r[0] in set(game_days) and r[0] not in report_days
            )
            self._add_result(
                stage,
                "integrity",
                "injury_cache_integrity",
                CheckStatus.WARN if len(empty) > 3 else CheckStatus.PASS,
                (
                    f"{len(empty)} game days fetched with no report: {', '.join(empty[-5:])}"
                    if empty
                    else "Every fetched game day has injury data"
                ),
                expected=0,
                actual=len(empty),
                query_time_ms=qtime,
            )

    # -------------------------------------------------------------------------
    # Players Checks
    # -------------------------------------------------------------------------
    def _check_players(self):
        """Check Players table (not season-scoped)."""
        stage = "Players"

        with get_db(self.db_path) as conn:
            cursor = conn.cursor()

            # 1. Check total player count
            result, qtime = self._timed_query(
                cursor, "SELECT COUNT(*) FROM Players", ()
            )
            player_count = result[0][0]

            # Expect at least 500 players (active roster ~450 + historical)
            if player_count >= 500:
                self._add_result(
                    stage,
                    "completeness",
                    "player_count",
                    CheckStatus.PASS,
                    f"Found {player_count} players",
                    query_time_ms=qtime,
                )
            elif player_count > 0:
                self._add_result(
                    stage,
                    "completeness",
                    "player_count",
                    CheckStatus.WARN,
                    f"Low player count: {player_count}",
                    expected=">=500",
                    actual=player_count,
                    query_time_ms=qtime,
                )
            else:
                self._add_result(
                    stage,
                    "completeness",
                    "player_count",
                    CheckStatus.CRITICAL,
                    "No players found",
                    query_time_ms=qtime,
                )

            # 2. Check for duplicate person_ids
            result, qtime = self._timed_query(
                cursor,
                """
                SELECT person_id, COUNT(*) as cnt
                FROM Players GROUP BY person_id HAVING cnt > 1
                """,
                (),
            )
            if len(result) == 0:
                self._add_result(
                    stage,
                    "integrity",
                    "no_duplicate_person_ids",
                    CheckStatus.PASS,
                    "No duplicate person_ids",
                    query_time_ms=qtime,
                )
            else:
                self._add_result(
                    stage,
                    "integrity",
                    "no_duplicate_person_ids",
                    CheckStatus.CRITICAL,
                    f"Found {len(result)} duplicate person_ids",
                    query_time_ms=qtime,
                )

    # -------------------------------------------------------------------------
    # Flag Consistency Checks
    # -------------------------------------------------------------------------
    def _check_flag_consistency(self):
        """Check that flags match underlying data state."""
        stage = "Flags"

        with get_db(self.db_path) as conn:
            cursor = conn.cursor()

            # 1. game_data_finalized=1 but no PbP
            result, qtime = self._timed_query(
                cursor,
                """
                SELECT COUNT(*) FROM Games g
                WHERE g.season = ? AND g.game_data_finalized = 1
                AND g.season_type IN ('Regular Season', 'Post Season')
                AND NOT EXISTS (
                    SELECT 1 FROM PbP_Logs p WHERE p.game_id = g.game_id
                )
                """,
                (self.season,),
            )
            violations = result[0][0]

            if violations == 0:
                self._add_result(
                    stage,
                    "flag_consistency",
                    "game_data_has_pbp",
                    CheckStatus.PASS,
                    "All game_data_finalized=1 games have PbP",
                    query_time_ms=qtime,
                )
            else:
                self._add_result(
                    stage,
                    "flag_consistency",
                    "game_data_has_pbp",
                    CheckStatus.CRITICAL,
                    f"{violations} games with game_data_finalized=1 but no PbP",
                    query_time_ms=qtime,
                )

            # 2. game_data_finalized=1 but no GameStates with is_final_state=1
            result, qtime = self._timed_query(
                cursor,
                """
                SELECT COUNT(*) FROM Games g
                WHERE g.season = ? AND g.game_data_finalized = 1
                AND g.season_type IN ('Regular Season', 'Post Season')
                AND NOT EXISTS (
                    SELECT 1 FROM GameStates gs
                    WHERE gs.game_id = g.game_id AND gs.is_final_state = 1
                )
                """,
                (self.season,),
            )
            violations = result[0][0]

            if violations == 0:
                self._add_result(
                    stage,
                    "flag_consistency",
                    "game_data_has_final_state",
                    CheckStatus.PASS,
                    "All game_data_finalized=1 games have is_final_state=1",
                    query_time_ms=qtime,
                )
            else:
                self._add_result(
                    stage,
                    "flag_consistency",
                    "game_data_has_final_state",
                    CheckStatus.CRITICAL,
                    f"{violations} games with game_data_finalized=1 but no final state",
                    query_time_ms=qtime,
                )

            # 3. boxscore_data_finalized=1 but no PlayerBox
            # Only check for seasons with boxscore data
            if self.season >= self.PLAYERBOX_START_SEASON:
                result, qtime = self._timed_query(
                    cursor,
                    """
                    SELECT COUNT(*) FROM Games g
                    WHERE g.season = ? AND g.boxscore_data_finalized = 1
                    AND g.season_type IN ('Regular Season', 'Post Season')
                    AND NOT EXISTS (
                        SELECT 1 FROM PlayerBox pb WHERE pb.game_id = g.game_id
                    )
                    """,
                    (self.season,),
                )
                violations = result[0][0]

                if violations == 0:
                    self._add_result(
                        stage,
                        "flag_consistency",
                        "boxscore_has_playerbox",
                        CheckStatus.PASS,
                        "All boxscore_data_finalized=1 games have PlayerBox",
                        query_time_ms=qtime,
                    )
                else:
                    self._add_result(
                        stage,
                        "flag_consistency",
                        "boxscore_has_playerbox",
                        CheckStatus.CRITICAL,
                        f"{violations} games with boxscore_data_finalized=1 but no PlayerBox",
                        query_time_ms=qtime,
                    )

            # 4. pre_game_data_finalized=1 but no Features
            result, qtime = self._timed_query(
                cursor,
                """
                SELECT COUNT(*) FROM Games g
                WHERE g.season = ? AND g.pre_game_data_finalized = 1
                AND g.season_type IN ('Regular Season', 'Post Season')
                AND NOT EXISTS (
                    SELECT 1 FROM Features f WHERE f.game_id = g.game_id
                )
                """,
                (self.season,),
            )
            violations = result[0][0]

            if violations == 0:
                self._add_result(
                    stage,
                    "flag_consistency",
                    "pregame_has_features",
                    CheckStatus.PASS,
                    "All pre_game_data_finalized=1 games have Features",
                    query_time_ms=qtime,
                )
            else:
                self._add_result(
                    stage,
                    "flag_consistency",
                    "pregame_has_features",
                    CheckStatus.CRITICAL,
                    f"{violations} games with pre_game_data_finalized=1 but no Features",
                    query_time_ms=qtime,
                )

            # 5. Flag dependency: pre_game=1 requires either game_data=1 OR status != 3
            # (Upcoming games can have features from prior games without PbP)
            result, qtime = self._timed_query(
                cursor,
                """
                SELECT COUNT(*) FROM Games g
                WHERE g.season = ? AND g.pre_game_data_finalized = 1
                AND g.game_data_finalized = 0 AND g.status = 3
                AND g.season_type IN ('Regular Season', 'Post Season')
                """,
                (self.season,),
            )
            violations = result[0][0]

            if violations == 0:
                self._add_result(
                    stage,
                    "flag_consistency",
                    "flag_dependency_pregame",
                    CheckStatus.PASS,
                    "No completed games with pre_game=1 but game_data=0",
                    query_time_ms=qtime,
                )
            else:
                self._add_result(
                    stage,
                    "flag_consistency",
                    "flag_dependency_pregame",
                    CheckStatus.WARN,
                    f"{violations} completed games with pre_game=1 but game_data=0",
                    query_time_ms=qtime,
                )


# =============================================================================
# CLI
# =============================================================================


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="NBA AI Health Check - Validate data pipeline health",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m src.health_check --season=2025-2026
    python -m src.health_check --season=2025-2026 --skip-pipeline --json
    python -m src.health_check --season=2025-2026 --format=table --log_level=DEBUG
        """,
    )

    parser.add_argument(
        "--season",
        required=True,
        help="Season to check (e.g., 2024-2025 or 'Current')",
    )
    parser.add_argument(
        "--skip-pipeline",
        action="store_true",
        help="Skip running the data pipeline (may report stale data)",
    )
    parser.add_argument(
        "--format",
        choices=["table", "compact"],
        default="table",
        help="Output format (default: table)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )
    parser.add_argument(
        "--log_level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: WARNING)",
    )
    parser.add_argument(
        "--predictor",
        default=None,
        help="Predictor to use when running pipeline (default: from config)",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)

    # Resolve season
    from src.utils import determine_current_season

    season = args.season
    if season.lower() == "current":
        season = determine_current_season()
        logging.info(f"Resolved 'Current' to season: {season}")

    # Validate season format
    try:
        validate_season_format(season)
    except ValueError as e:
        print(f"Error: Invalid season format: {e}", file=sys.stderr)
        sys.exit(2)

    # Phase 1: Run pipeline (unless skipped)
    if not args.skip_pipeline:
        from src.database_updater.database_update_manager import update_database

        print(f"Phase 1: Running data pipeline for {season}...")
        predictor = args.predictor or config.get("default_predictor")
        try:
            update_database(season=season, predictor=predictor)
            print("Phase 1: Pipeline complete ✓")
        except Exception as e:
            print(f"Phase 1: Pipeline failed - {e}", file=sys.stderr)
            logging.exception("Pipeline error")
            # Continue with health checks even if pipeline fails
    else:
        print("Phase 1: Skipped (--skip-pipeline)")
        logging.warning("Pipeline skipped - health checks may report stale data issues")

    # Phase 2: Run health checks
    print(f"\nPhase 2: Running health checks for {season}...")
    checker = SeasonHealthChecker(season)
    report = checker.run_all()
    report.pipeline_ran = not args.skip_pipeline

    # Output results
    if args.json:
        print(report.to_json())
    else:
        print(report.summary_table())

    # Exit with appropriate code
    sys.exit(report.exit_code)


if __name__ == "__main__":
    main()
