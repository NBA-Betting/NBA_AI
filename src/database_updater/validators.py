"""
validators.py

Unified validation framework for NBA AI data pipeline.

Provides standardized validation classes, result types, and severity levels
for inline validation during pipeline execution.

Key Features:
- Standardized ValidationResult with severity levels (CRITICAL/WARNING/INFO)
- Per-game and per-batch validation support
- Blocking logic for critical failures
- Consistent logging format integration
- Auto-fix capabilities for safe issues (flags, retries)
- Integrated into pipeline stages for immediate issue detection

Usage:
    validator = ScheduleValidator()
    result = validator.validate(game_ids, cursor)

    if result.has_critical_issues:
        logger.error(f"Validation failed: {result.summary()}")
        # Skip games or retry

    logger.info(f"[Schedule] +3 ~47 -0 {result.log_suffix()} | 1.2s")
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


class Severity(Enum):
    """Issue severity levels."""

    CRITICAL = "CRITICAL"  # Data corruption, missing required data - blocks pipeline
    WARNING = "WARNING"  # Data quality issues, unexpected values - log but continue
    INFO = "INFO"  # Informational notices, normal edge cases


@dataclass
class ValidationIssue:
    """Represents a single validation issue."""

    check_id: str  # e.g., "NULL_TEAMS", "MISSING_FINAL_STATE"
    severity: Severity
    message: str  # Human-readable description
    count: int = 1  # Number of occurrences
    sample_data: List[str] = field(default_factory=list)  # Sample game_ids or details
    fixable: bool = False  # Can be auto-fixed

    def __str__(self):
        severity_symbol = {
            Severity.CRITICAL: "🔴",
            Severity.WARNING: "🟡",
            Severity.INFO: "🔵",
        }
        symbol = severity_symbol.get(self.severity, "⚪")
        fixable = " [AUTO-FIX]" if self.fixable else ""
        sample = f" (e.g., {self.sample_data[0]})" if self.sample_data else ""
        return f"{symbol} {self.check_id}: {self.message} (n={self.count}){sample}{fixable}"


@dataclass
class ValidationResult:
    """Results from validation checks on a stage."""

    stage_name: str  # e.g., "Schedule", "PBP", "GameStates"
    total_checked: int  # Total games/records validated
    issues: List[ValidationIssue] = field(default_factory=list)

    @property
    def has_critical_issues(self) -> bool:
        """Check if any critical issues exist (should block pipeline)."""
        return any(issue.severity == Severity.CRITICAL for issue in self.issues)

    @property
    def has_warnings(self) -> bool:
        """Check if any warnings exist."""
        return any(issue.severity == Severity.WARNING for issue in self.issues)

    @property
    def critical_count(self) -> int:
        """Count of critical issues."""
        return sum(
            issue.count for issue in self.issues if issue.severity == Severity.CRITICAL
        )

    @property
    def warning_count(self) -> int:
        """Count of warning issues."""
        return sum(
            issue.count for issue in self.issues if issue.severity == Severity.WARNING
        )

    @property
    def info_count(self) -> int:
        """Count of info issues."""
        return sum(
            issue.count for issue in self.issues if issue.severity == Severity.INFO
        )

    def log_suffix(self) -> str:
        """
        Generate compact suffix for INFO log line.

        Examples:
            "" (no issues)
            "WARN: 18 TBD"
            "CRIT: 3 missing final states"
            "WARN: 18 TBD, CRIT: 2 null dates"
        """
        parts = []

        if self.has_critical_issues:
            # Show first critical issue message
            first_critical = next(
                i for i in self.issues if i.severity == Severity.CRITICAL
            )
            parts.append(
                f"CRIT: {first_critical.count} {first_critical.check_id.lower().replace('_', ' ')}"
            )

        if self.has_warnings:
            # Show first warning message
            first_warning = next(
                i for i in self.issues if i.severity == Severity.WARNING
            )
            parts.append(
                f"WARN: {first_warning.count} {first_warning.check_id.lower().replace('_', ' ')}"
            )

        return ", ".join(parts) if parts else ""

    def summary(self) -> str:
        """Generate detailed multi-line summary for logging."""
        if not self.issues:
            return f"{self.stage_name} validation: PASS ({self.total_checked} checked)"

        lines = [f"{self.stage_name} validation: {len(self.issues)} issue types found"]
        for issue in self.issues:
            lines.append(f"  {issue}")

        return "\n".join(lines)

    def get_fixable_issues(self) -> List[ValidationIssue]:
        """Get list of auto-fixable issues."""
        return [issue for issue in self.issues if issue.fixable]

    def get_failed_game_ids(self) -> List[str]:
        """Get unique list of game_ids that failed validation."""
        game_ids = []
        for issue in self.issues:
            if issue.severity == Severity.CRITICAL:
                game_ids.extend(issue.sample_data)
        return list(set(game_ids))


class BaseValidator:
    """Base class for stage-specific validators."""

    def __init__(self, stage_name: str):
        self.stage_name = stage_name
        self.logger = logging.getLogger(f"{__name__}.{stage_name}")

    def validate(self, game_ids: List[str], cursor) -> ValidationResult:
        """
        Run validation checks for this stage.

        Args:
            game_ids: List of game IDs to validate
            cursor: Database cursor for queries

        Returns:
            ValidationResult with any issues found
        """
        raise NotImplementedError("Subclasses must implement validate()")

    def _check_null_fields(
        self,
        game_ids: List[str],
        cursor,
        table: str,
        fields: List[str],
        check_id: str = "NULL_FIELDS",
    ) -> Optional[ValidationIssue]:
        """
        Helper: Check for NULL values in critical fields.

        Args:
            game_ids: Game IDs to check
            cursor: Database cursor
            table: Table name (e.g., "Games")
            fields: List of field names that must not be NULL
            check_id: Identifier for this check

        Returns:
            ValidationIssue if NULLs found, else None
        """
        placeholders = ",".join("?" * len(game_ids))
        null_conditions = " OR ".join(f"{field} IS NULL" for field in fields)

        query = f"""
            SELECT game_id FROM {table}
            WHERE game_id IN ({placeholders})
            AND ({null_conditions})
        """

        cursor.execute(query, game_ids)
        results = cursor.fetchall()

        if results:
            failed_ids = [row[0] for row in results]
            return ValidationIssue(
                check_id=check_id,
                severity=Severity.CRITICAL,
                message=f"NULL values in critical fields ({', '.join(fields)})",
                count=len(failed_ids),
                sample_data=failed_ids[:5],
                fixable=False,
            )

        return None

    def _check_count_threshold(
        self,
        actual_count: int,
        expected_min: int,
        expected_max: Optional[int] = None,
        item_name: str = "items",
        check_id: str = "COUNT_THRESHOLD",
    ) -> Optional[ValidationIssue]:
        """
        Helper: Check if count is within expected range.

        Args:
            actual_count: Actual count observed
            expected_min: Minimum expected count
            expected_max: Maximum expected count (optional)
            item_name: Name of items being counted (for message)
            check_id: Identifier for this check

        Returns:
            ValidationIssue if out of range, else None
        """
        if actual_count < expected_min:
            return ValidationIssue(
                check_id=check_id,
                severity=Severity.WARNING,
                message=f"Low {item_name} count: {actual_count} (expected ≥{expected_min})",
                count=actual_count,
                fixable=False,
            )

        if expected_max and actual_count > expected_max:
            return ValidationIssue(
                check_id=check_id,
                severity=Severity.WARNING,
                message=f"High {item_name} count: {actual_count} (expected ≤{expected_max})",
                count=actual_count,
                fixable=False,
            )

        return None


class ScheduleValidator(BaseValidator):
    """Validator for Schedule stage (Games table)."""

    def __init__(self):
        super().__init__("Schedule")

    def validate(self, game_ids: List[str], cursor) -> ValidationResult:
        """
        Validate schedule data for given game IDs.

        Checks:
        - NULL critical fields (home_team, away_team, date_time_utc)
        - TBD teams (warning)
        - Game count thresholds for completed seasons
        - Status field validity
        """
        result = ValidationResult(
            stage_name=self.stage_name, total_checked=len(game_ids)
        )

        if not game_ids:
            return result

        # Check 1: NULL critical fields
        null_issue = self._check_null_fields(
            game_ids,
            cursor,
            "Games",
            ["home_team", "away_team", "date_time_utc"],
            check_id="NULL_TEAMS_OR_DATE",
        )
        if null_issue:
            result.issues.append(null_issue)

        # Check 2: TBD teams (warning, not critical)
        placeholders = ",".join("?" * len(game_ids))
        cursor.execute(
            f"""
            SELECT game_id FROM Games
            WHERE game_id IN ({placeholders})
            AND (home_team = 'TBD' OR away_team = 'TBD')
        """,
            game_ids,
        )
        tbd_results = cursor.fetchall()

        if tbd_results:
            tbd_ids = [row[0] for row in tbd_results]
            result.issues.append(
                ValidationIssue(
                    check_id="TBD_TEAMS",
                    severity=Severity.WARNING,
                    message="Teams not yet determined (TBD)",
                    count=len(tbd_ids),
                    sample_data=tbd_ids[:5],
                    fixable=False,
                )
            )

        # Check 3: Invalid status values (aligned with NBA API status codes)
        valid_statuses = [1, 2, 3]  # 1=Not Started, 2=In Progress, 3=Final
        cursor.execute(
            f"""
            SELECT game_id, status FROM Games
            WHERE game_id IN ({placeholders})
            AND status NOT IN ({','.join('?' * len(valid_statuses))})
        """,
            game_ids + valid_statuses,
        )
        invalid_status_results = cursor.fetchall()

        if invalid_status_results:
            invalid_ids = [row[0] for row in invalid_status_results]
            result.issues.append(
                ValidationIssue(
                    check_id="INVALID_STATUS",
                    severity=Severity.WARNING,
                    message=f"Invalid status values (expected: 1=Not Started, 2=In Progress, 3=Final)",
                    count=len(invalid_ids),
                    sample_data=invalid_ids[:5],
                    fixable=False,
                )
            )

        return result


class PlayerValidator(BaseValidator):
    """Validator for Players stage (Players table)."""

    def __init__(self):
        super().__init__("Players")

    def validate(self, player_ids: List[int], cursor) -> ValidationResult:
        """
        Validate player data for given person IDs.

        Checks:
        - NULL critical fields (first_name, last_name, full_name)
        - Duplicate person_ids
        - Player count thresholds
        """
        result = ValidationResult(
            stage_name=self.stage_name, total_checked=len(player_ids)
        )

        if not player_ids:
            return result

        placeholders = ",".join("?" * len(player_ids))

        # Check 1: NULL critical fields
        cursor.execute(
            f"""
            SELECT person_id FROM Players
            WHERE person_id IN ({placeholders})
            AND (first_name IS NULL OR last_name IS NULL OR full_name IS NULL)
        """,
            player_ids,
        )
        null_results = cursor.fetchall()

        if null_results:
            null_ids = [row[0] for row in null_results]
            result.issues.append(
                ValidationIssue(
                    check_id="NULL_NAMES",
                    severity=Severity.CRITICAL,
                    message="NULL values in critical fields (first_name, last_name, full_name)",
                    count=len(null_ids),
                    sample_data=[str(id) for id in null_ids[:5]],
                    fixable=False,
                )
            )

        # Note: DUPLICATE_PLAYERS check removed - Players table has UNIQUE constraint on person_id
        # Note: validate_total_count() moved to health check CLI - not needed for inline validation

        return result


class InjuryValidator(BaseValidator):
    """Validator for Injuries stage (InjuryReports table)."""

    def __init__(self):
        super().__init__("Injuries")

    def validate(self, date_range: tuple, cursor) -> ValidationResult:
        """
        Validate injury data for given date range.

        Args:
            date_range: Tuple of (start_date, end_date) as strings
            cursor: Database cursor

        Returns:
            ValidationResult with issues found
        """
        start_date, end_date = date_range
        result = ValidationResult(stage_name=self.stage_name, total_checked=1)

        # Check 1: NULL critical fields
        cursor.execute(
            """
            SELECT id FROM InjuryReports
            WHERE source = 'NBA_Official'
            AND report_timestamp BETWEEN ? AND ?
            AND (player_name IS NULL OR status IS NULL OR report_timestamp IS NULL)
        """,
            (start_date, end_date),
        )
        null_results = cursor.fetchall()

        if null_results:
            null_ids = [row[0] for row in null_results]
            result.issues.append(
                ValidationIssue(
                    check_id="NULL_INJURY_FIELDS",
                    severity=Severity.CRITICAL,
                    message="NULL values in critical fields (player_name, status, report_timestamp)",
                    count=len(null_ids),
                    sample_data=[str(id) for id in null_ids[:5]],
                    fixable=False,
                )
            )

        # Check 2: Invalid status values
        valid_statuses = ["Out", "Available", "Questionable", "Doubtful", "Probable"]
        cursor.execute(
            f"""
            SELECT id, status FROM InjuryReports
            WHERE source = 'NBA_Official'
            AND report_timestamp BETWEEN ? AND ?
            AND status NOT IN ({','.join('?' * len(valid_statuses))})
        """,
            (start_date, end_date) + tuple(valid_statuses),
        )
        invalid_status_results = cursor.fetchall()

        if invalid_status_results:
            invalid_ids = [row[0] for row in invalid_status_results]
            result.issues.append(
                ValidationIssue(
                    check_id="INVALID_STATUS",
                    severity=Severity.WARNING,
                    message=f"Invalid status values (expected: {', '.join(valid_statuses)})",
                    count=len(invalid_ids),
                    sample_data=[f"{r[0]}:{r[1]}" for r in invalid_status_results[:5]],
                    fixable=False,
                )
            )

        # Check 3: Duplicate records (semantic duplicates by player/game date/team)
        # Note: Uses player_name + game_date + team as the semantic key (the
        # table's unique key) because nba_player_id can be NULL for unmatched
        # players; one report lists a team's back-to-back games separately
        cursor.execute(
            """
            SELECT player_name, game_date, team, COUNT(*) as cnt
            FROM InjuryReports
            WHERE source = 'NBA_Official'
            AND report_timestamp BETWEEN ? AND ?
            GROUP BY player_name, game_date, COALESCE(team, '')
            HAVING cnt > 1
        """,
            (start_date, end_date),
        )
        dup_results = cursor.fetchall()

        if dup_results:
            result.issues.append(
                ValidationIssue(
                    check_id="DUPLICATE_INJURIES",
                    severity=Severity.CRITICAL,
                    message="Duplicate injury records found (same player/date/team)",
                    count=len(dup_results),
                    sample_data=[f"{r[0]}:{r[1]}" for r in dup_results[:5]],
                    fixable=True,
                )
            )

        # Note: validate_date_coverage() moved to health check CLI - not needed for inline validation

        return result


class BettingValidator(BaseValidator):
    """Validator for Betting table with opening/current/closing separation."""

    def __init__(self):
        super().__init__("Betting")

    def validate(self, game_ids: Optional[List[str]], cursor) -> ValidationResult:
        """
        Validate betting data for given games.

        Checks:
        - NULL closing lines for completed finalized games
        - Unrealistic spread values (>50 points)
        - Unrealistic total values (<150 or >300 points)
        - Missing finalization flag for completed games with closing lines
        - Orphaned current lines (current without opening)
        """
        result = ValidationResult(stage_name=self.stage_name, total_checked=0)

        if game_ids:
            # Validate specific games
            placeholders = ",".join(["?"] * len(game_ids))
            query = f"""
                SELECT b.game_id, g.status,
                       b.espn_opening_spread, b.espn_opening_total,
                       b.espn_current_spread, b.espn_current_total,
                       b.espn_closing_spread, b.espn_closing_total,
                       b.covers_closing_spread, b.covers_closing_total,
                       b.lines_finalized
                FROM Betting b
                JOIN Games g ON b.game_id = g.game_id
                WHERE b.game_id IN ({placeholders})
            """
            cursor.execute(query, game_ids)
        else:
            # Validate all betting data
            cursor.execute(
                """
                SELECT b.game_id, g.status,
                       b.espn_opening_spread, b.espn_opening_total,
                       b.espn_current_spread, b.espn_current_total,
                       b.espn_closing_spread, b.espn_closing_total,
                       b.covers_closing_spread, b.covers_closing_total,
                       b.lines_finalized
                FROM Betting b
                JOIN Games g ON b.game_id = g.game_id
            """
            )

        rows = cursor.fetchall()
        result.total_checked = len(rows)

        # Track issues
        missing_closing_finalized = []  # Finalized but no closing lines
        unrealistic_spreads = []
        unrealistic_totals = []
        completed_no_closing = []  # Completed game but no closing lines
        orphaned_current = []  # Current lines without opening

        for row in rows:
            game_id = row[0]
            status = row[1]
            espn_opening_spread, espn_opening_total = row[2], row[3]
            espn_current_spread, espn_current_total = row[4], row[5]
            espn_closing_spread, espn_closing_total = row[6], row[7]
            covers_closing_spread, covers_closing_total = row[8], row[9]
            lines_finalized = row[10]

            # Check: Finalized games should have closing lines
            if lines_finalized == 1:
                has_closing = (
                    espn_closing_spread is not None or covers_closing_spread is not None
                )
                if not has_closing:
                    missing_closing_finalized.append(game_id)

            # Check: Completed games should eventually have closing lines
            if status == 3:  # Final
                has_any_closing = (
                    espn_closing_spread is not None
                    or espn_closing_total is not None
                    or covers_closing_spread is not None
                    or covers_closing_total is not None
                )
                if not has_any_closing:
                    completed_no_closing.append(game_id)

            # Check: Current lines should have corresponding opening lines
            if espn_current_spread is not None and espn_opening_spread is None:
                orphaned_current.append(game_id)

            # Check: Unrealistic spread values (check all spread types)
            for spread_val, spread_type in [
                (espn_opening_spread, "opening"),
                (espn_current_spread, "current"),
                (espn_closing_spread, "espn_closing"),
                (covers_closing_spread, "covers_closing"),
            ]:
                if spread_val is not None and abs(spread_val) > 50:
                    unrealistic_spreads.append(f"{game_id}:{spread_type}={spread_val}")

            # Check: Unrealistic total values (check all total types)
            for total_val, total_type in [
                (espn_opening_total, "opening"),
                (espn_current_total, "current"),
                (espn_closing_total, "espn_closing"),
                (covers_closing_total, "covers_closing"),
            ]:
                if total_val is not None and (total_val < 150 or total_val > 300):
                    unrealistic_totals.append(f"{game_id}:{total_type}={total_val}")

        if missing_closing_finalized:
            result.issues.append(
                ValidationIssue(
                    check_id="FINALIZED_NO_CLOSING",
                    severity=Severity.CRITICAL,
                    message="Games marked finalized but missing closing lines",
                    count=len(missing_closing_finalized),
                    sample_data=missing_closing_finalized[:5],
                    fixable=False,
                )
            )

        if completed_no_closing:
            result.issues.append(
                ValidationIssue(
                    check_id="COMPLETED_NO_CLOSING",
                    severity=Severity.WARNING,
                    message="Completed games without closing lines (may be outside ESPN window)",
                    count=len(completed_no_closing),
                    sample_data=completed_no_closing[:5],
                    fixable=False,
                )
            )

        if orphaned_current:
            result.issues.append(
                ValidationIssue(
                    check_id="ORPHANED_CURRENT",
                    severity=Severity.INFO,
                    message="Current lines without corresponding opening lines",
                    count=len(orphaned_current),
                    sample_data=orphaned_current[:5],
                    fixable=False,
                )
            )

        if unrealistic_spreads:
            result.issues.append(
                ValidationIssue(
                    check_id="UNREALISTIC_SPREAD",
                    severity=Severity.WARNING,
                    message="Spreads >50 points detected",
                    count=len(unrealistic_spreads),
                    sample_data=unrealistic_spreads[:5],
                    fixable=False,
                )
            )

        if unrealistic_totals:
            result.issues.append(
                ValidationIssue(
                    check_id="UNREALISTIC_TOTAL",
                    severity=Severity.WARNING,
                    message="Totals outside 150-300 range detected",
                    count=len(unrealistic_totals),
                    sample_data=unrealistic_totals[:5],
                    fixable=False,
                )
            )

        return result


class PbPValidator(BaseValidator):
    """
    Validator for Play-by-Play data quality and completeness.

    Checks:
    - MISSING_PBP: Games with status='Completed' but no PBP data
    - LOW_PLAY_COUNT: Games with suspiciously few plays (<200)
    - STALE_INPROGRESS_PBP: In-progress games with PBP >20 min old
    - NO_FINAL_STATE: Games with PBP but no final GameState
    - DUPLICATE_PLAYS: Games with duplicate play_ids
    """

    def __init__(self):
        super().__init__("PBP")

    def validate(self, game_ids: List[str], cursor) -> ValidationResult:
        """
        Validate PBP data for specified games.

        Parameters:
            game_ids (List[str]): Game IDs to validate.
            cursor: Database cursor.

        Returns:
            ValidationResult: Validation results with any issues found.
        """
        result = ValidationResult(
            stage_name=self.stage_name, total_checked=len(game_ids)
        )

        if not game_ids:
            return result

        placeholders = ",".join("?" * len(game_ids))

        # Check 1: Completed games missing PBP
        cursor.execute(
            f"""
            SELECT g.game_id
            FROM Games g
            WHERE g.game_id IN ({placeholders})
            AND g.status = 3  -- Final
            AND NOT EXISTS (SELECT 1 FROM PbP_Logs WHERE game_id = g.game_id)
            """,
            game_ids,
        )
        missing_pbp = [row[0] for row in cursor.fetchall()]

        # Check 2: Completed games with suspiciously low play counts
        # Note: In-progress games naturally have few plays, so only check completed games
        cursor.execute(
            f"""
            SELECT p.game_id, COUNT(*) as play_count
            FROM PbP_Logs p
            JOIN Games g ON p.game_id = g.game_id
            WHERE p.game_id IN ({placeholders})
            AND g.status = 3  -- Only check completed games
            GROUP BY p.game_id
            HAVING play_count < 200
            """,
            game_ids,
        )
        low_play_count = [f"{row[0]} ({row[1]} plays)" for row in cursor.fetchall()]

        # Check 3: In-progress games with stale PBP (>20 min old)
        cursor.execute(
            f"""
            SELECT g.game_id, g.pbp_last_fetched_at,
                   CAST((julianday('now') - julianday(g.pbp_last_fetched_at)) * 24 * 60 AS INTEGER) as minutes_ago
            FROM Games g
            WHERE g.game_id IN ({placeholders})
            AND g.status = 2  -- In Progress
            AND g.pbp_last_fetched_at IS NOT NULL
            AND g.pbp_last_fetched_at < datetime('now', '-20 minutes')
            """,
            game_ids,
        )
        stale_inprogress = [f"{row[0]} ({row[2]} min ago)" for row in cursor.fetchall()]

        # Check 4: Games with PBP but no final GameState
        cursor.execute(
            f"""
            SELECT p.game_id
            FROM (SELECT DISTINCT game_id FROM PbP_Logs WHERE game_id IN ({placeholders})) p
            WHERE NOT EXISTS (
                SELECT 1 FROM GameStates gs
                WHERE gs.game_id = p.game_id
                AND gs.is_final_state = 1
            )
            """,
            game_ids,
        )
        no_final_state = [row[0] for row in cursor.fetchall()]

        # Check 5: Duplicate play_ids within a game
        cursor.execute(
            f"""
            SELECT game_id, play_id, COUNT(*) as dup_count
            FROM PbP_Logs
            WHERE game_id IN ({placeholders})
            GROUP BY game_id, play_id
            HAVING dup_count > 1
            """,
            game_ids,
        )
        duplicates = [
            f"{row[0]} (play_id {row[1]} x{row[2]})" for row in cursor.fetchall()
        ]

        # Add issues to result
        if missing_pbp:
            result.issues.append(
                ValidationIssue(
                    check_id="MISSING_PBP",
                    severity=Severity.CRITICAL,
                    message="Completed games without PBP data",
                    count=len(missing_pbp),
                    sample_data=missing_pbp[:5],
                    fixable=True,  # Can refetch
                )
            )

        if low_play_count:
            result.issues.append(
                ValidationIssue(
                    check_id="LOW_PLAY_COUNT",
                    severity=Severity.WARNING,
                    message="Games with <200 plays (possible incomplete data)",
                    count=len(low_play_count),
                    sample_data=low_play_count[:5],
                    fixable=True,  # Can refetch
                )
            )

        if stale_inprogress:
            result.issues.append(
                ValidationIssue(
                    check_id="STALE_INPROGRESS_PBP",
                    severity=Severity.WARNING,
                    message="In-progress games with PBP >20 min old",
                    count=len(stale_inprogress),
                    sample_data=stale_inprogress[:5],
                    fixable=True,  # Can refetch
                )
            )

        if no_final_state:
            result.issues.append(
                ValidationIssue(
                    check_id="NO_FINAL_STATE",
                    severity=Severity.WARNING,
                    message="Games with PBP but no final GameState",
                    count=len(no_final_state),
                    sample_data=no_final_state[:5],
                    fixable=True,  # GameStates stage can fix
                )
            )

        if duplicates:
            result.issues.append(
                ValidationIssue(
                    check_id="DUPLICATE_PLAYS",
                    severity=Severity.CRITICAL,
                    message="Duplicate play_ids detected",
                    count=len(duplicates),
                    sample_data=duplicates[:5],
                    fixable=True,  # Can refetch
                )
            )

        return result


class GameStatesValidator(BaseValidator):
    """
    Validator for GameStates data quality and completeness.

    Checks:
    - MISSING_GAME_STATES: Completed games with PBP but no GameStates
    - NO_FINAL_STATE: Completed games without is_final_state=1
    - LOW_STATE_COUNT: Games with suspiciously few states (<100)
    - SCORE_MISMATCH: Final GameState score doesn't match Games table
    - INVALID_SCORES: Games with negative or impossible scores
    """

    def __init__(self):
        super().__init__("GameStates")

    def validate(self, game_ids: List[str], cursor) -> ValidationResult:
        """
        Validate GameStates data for specified games.

        Parameters:
            game_ids (List[str]): Game IDs to validate.
            cursor: Database cursor.

        Returns:
            ValidationResult: Validation results with any issues found.
        """
        result = ValidationResult(
            stage_name=self.stage_name, total_checked=len(game_ids)
        )

        if not game_ids:
            return result

        placeholders = ",".join("?" * len(game_ids))

        # Check 1: Completed games with PBP but no GameStates
        cursor.execute(
            f"""
            SELECT g.game_id
            FROM Games g
            WHERE g.game_id IN ({placeholders})
            AND g.status = 3  -- Final
            AND EXISTS (SELECT 1 FROM PbP_Logs WHERE game_id = g.game_id)
            AND NOT EXISTS (SELECT 1 FROM GameStates WHERE game_id = g.game_id)
            """,
            game_ids,
        )
        missing_states = [row[0] for row in cursor.fetchall()]

        # Check 2: Completed games without final state marker
        cursor.execute(
            f"""
            SELECT g.game_id
            FROM Games g
            WHERE g.game_id IN ({placeholders})
            AND g.status = 3  -- Final
            AND EXISTS (SELECT 1 FROM GameStates WHERE game_id = g.game_id)
            AND NOT EXISTS (
                SELECT 1 FROM GameStates gs 
                WHERE gs.game_id = g.game_id AND gs.is_final_state = 1
            )
            """,
            game_ids,
        )
        no_final_marker = [row[0] for row in cursor.fetchall()]

        # Check 3: Games with suspiciously low state counts
        cursor.execute(
            f"""
            SELECT gs.game_id, COUNT(*) as state_count
            FROM GameStates gs
            JOIN Games g ON gs.game_id = g.game_id
            WHERE gs.game_id IN ({placeholders})
            AND g.status = 3  -- Only check completed games
            GROUP BY gs.game_id
            HAVING state_count < 100
            """,
            game_ids,
        )
        low_state_count = [f"{row[0]} ({row[1]} states)" for row in cursor.fetchall()]

        # Check 4: Invalid scores (negative or impossibly high)
        cursor.execute(
            f"""
            SELECT game_id, home_score, away_score
            FROM GameStates
            WHERE game_id IN ({placeholders})
            AND (home_score < 0 OR away_score < 0 OR home_score > 200 OR away_score > 200)
            """,
            game_ids,
        )
        invalid_scores = [f"{row[0]} ({row[1]}-{row[2]})" for row in cursor.fetchall()]

        # Check 5: Duplicate play_ids within GameStates
        cursor.execute(
            f"""
            SELECT game_id, play_id, COUNT(*) as dup_count
            FROM GameStates
            WHERE game_id IN ({placeholders})
            GROUP BY game_id, play_id
            HAVING dup_count > 1
            """,
            game_ids,
        )
        duplicates = [
            f"{row[0]} (play_id {row[1]} x{row[2]})" for row in cursor.fetchall()
        ]

        # Add issues to result
        if missing_states:
            result.issues.append(
                ValidationIssue(
                    check_id="MISSING_GAME_STATES",
                    severity=Severity.CRITICAL,
                    message="Completed games with PBP but no GameStates",
                    count=len(missing_states),
                    sample_data=missing_states[:5],
                    fixable=True,  # Can regenerate from PBP
                )
            )

        if no_final_marker:
            result.issues.append(
                ValidationIssue(
                    check_id="NO_FINAL_STATE",
                    severity=Severity.WARNING,
                    message="Completed games without is_final_state=1 marker",
                    count=len(no_final_marker),
                    sample_data=no_final_marker[:5],
                    fixable=True,  # Can regenerate from PBP
                )
            )

        if low_state_count:
            result.issues.append(
                ValidationIssue(
                    check_id="LOW_STATE_COUNT",
                    severity=Severity.WARNING,
                    message="Completed games with <100 states (possible incomplete parsing)",
                    count=len(low_state_count),
                    sample_data=low_state_count[:5],
                    fixable=True,  # Can regenerate from PBP
                )
            )

        if invalid_scores:
            result.issues.append(
                ValidationIssue(
                    check_id="INVALID_SCORES",
                    severity=Severity.CRITICAL,
                    message="States with invalid scores (negative or >200)",
                    count=len(invalid_scores),
                    sample_data=invalid_scores[:5],
                    fixable=True,  # Can regenerate from PBP
                )
            )

        if duplicates:
            result.issues.append(
                ValidationIssue(
                    check_id="DUPLICATE_STATES",
                    severity=Severity.WARNING,
                    message="Duplicate play_ids in GameStates",
                    count=len(duplicates),
                    sample_data=duplicates[:5],
                    fixable=True,  # Can regenerate from PBP
                )
            )

        return result


class BoxscoresValidator(BaseValidator):
    """Validator for Boxscores stage (PlayerBox and TeamBox tables)."""

    def __init__(self):
        super().__init__("Boxscores")

    def validate(self, game_ids: List[str], cursor) -> ValidationResult:
        """
        Validate boxscore data for given game IDs.

        Checks:
        - Missing PlayerBox/TeamBox records for games
        - Player count per team (5-15 players typical)
        - Team count per game (exactly 2 teams)
        - Total minutes validation (240+ minutes per team for complete games)
        - NULL critical fields (pts, min, team_id)
        """
        result = ValidationResult(
            stage_name=self.stage_name, total_checked=len(game_ids)
        )

        if not game_ids:
            return result

        placeholders = ",".join("?" * len(game_ids))

        # Check 1: Missing PlayerBox records
        cursor.execute(
            f"""
            SELECT g.game_id FROM Games g
            LEFT JOIN PlayerBox pb ON g.game_id = pb.game_id
            WHERE g.game_id IN ({placeholders})
            AND g.status = 3  -- Only check completed games
            AND pb.game_id IS NULL
        """,
            game_ids,
        )
        missing_player_records = [row[0] for row in cursor.fetchall()]

        if missing_player_records:
            result.issues.append(
                ValidationIssue(
                    check_id="MISSING_PLAYER_BOX",
                    severity=Severity.CRITICAL,
                    message="Games missing PlayerBox records",
                    count=len(missing_player_records),
                    sample_data=missing_player_records[:5],
                    fixable=True,
                )
            )

        # Check 2: Missing TeamBox records
        cursor.execute(
            f"""
            SELECT g.game_id FROM Games g
            LEFT JOIN TeamBox tb ON g.game_id = tb.game_id
            WHERE g.game_id IN ({placeholders})
            AND g.status = 3  -- Only check completed games
            AND tb.game_id IS NULL
        """,
            game_ids,
        )
        missing_team_records = [row[0] for row in cursor.fetchall()]

        if missing_team_records:
            result.issues.append(
                ValidationIssue(
                    check_id="MISSING_TEAM_BOX",
                    severity=Severity.CRITICAL,
                    message="Games missing TeamBox records",
                    count=len(missing_team_records),
                    sample_data=missing_team_records[:5],
                    fixable=True,
                )
            )

        # Check 3: Player count per team (should be 5-18 players)
        # Note: Teams can have >15 players due to two-way contracts, 10-day contracts, etc.
        cursor.execute(
            f"""
            SELECT pb.game_id, pb.team_id, COUNT(*) as player_count
            FROM PlayerBox pb
            JOIN Games g ON pb.game_id = g.game_id
            WHERE pb.game_id IN ({placeholders})
            AND g.status = 3  -- Only check completed games
            GROUP BY pb.game_id, pb.team_id
            HAVING player_count < 5 OR player_count > 18
        """,
            game_ids,
        )
        invalid_player_counts = [
            f"{row[0]}:{row[1]}({row[2]})" for row in cursor.fetchall()
        ]

        if invalid_player_counts:
            result.issues.append(
                ValidationIssue(
                    check_id="INVALID_PLAYER_COUNT",
                    severity=Severity.WARNING,
                    message="Teams with unusual player count (expected 5-18)",
                    count=len(invalid_player_counts),
                    sample_data=invalid_player_counts[:5],
                    fixable=True,
                )
            )

        # Check 4: Team count per game (should be exactly 2)
        cursor.execute(
            f"""
            SELECT tb.game_id, COUNT(*) as team_count
            FROM TeamBox tb
            JOIN Games g ON tb.game_id = g.game_id
            WHERE tb.game_id IN ({placeholders})
            AND g.status = 3  -- Only check completed games
            GROUP BY tb.game_id
            HAVING team_count != 2
        """,
            game_ids,
        )
        invalid_team_counts = [f"{row[0]}({row[1]})" for row in cursor.fetchall()]

        if invalid_team_counts:
            result.issues.append(
                ValidationIssue(
                    check_id="INVALID_TEAM_COUNT",
                    severity=Severity.CRITICAL,
                    message="Games with incorrect team count (expected 2)",
                    count=len(invalid_team_counts),
                    sample_data=invalid_team_counts[:5],
                    fixable=True,
                )
            )

        # Check 5: Total minutes validation (239+ minutes per team for complete games)
        # Note: Using 239 instead of 240 to account for floating-point precision issues
        cursor.execute(
            f"""
            SELECT pb.game_id, pb.team_id, SUM(pb.min) as total_minutes
            FROM PlayerBox pb
            JOIN Games g ON pb.game_id = g.game_id
            WHERE pb.game_id IN ({placeholders})
            AND g.status = 3  -- Only check completed games
            AND pb.min IS NOT NULL
            GROUP BY pb.game_id, pb.team_id
            HAVING total_minutes < 239
        """,
            game_ids,
        )
        low_minutes = [f"{row[0]}:{row[1]}({row[2]}min)" for row in cursor.fetchall()]

        if low_minutes:
            result.issues.append(
                ValidationIssue(
                    check_id="LOW_MINUTES",
                    severity=Severity.WARNING,
                    message="Teams with <239 minutes (incomplete game or missing data)",
                    count=len(low_minutes),
                    sample_data=low_minutes[:5],
                    fixable=True,
                )
            )

        # Check 6: NULL critical fields in PlayerBox
        # Note: min=NULL is expected for DNP (Did Not Play) players, so only check pts and team_id
        cursor.execute(
            f"""
            SELECT COUNT(*) FROM PlayerBox pb
            JOIN Games g ON pb.game_id = g.game_id
            WHERE pb.game_id IN ({placeholders})
            AND g.status = 3  -- Only check completed games
            AND (pb.pts IS NULL OR pb.team_id IS NULL)
        """,
            game_ids,
        )
        null_player_fields = cursor.fetchone()[0]

        if null_player_fields > 0:
            result.issues.append(
                ValidationIssue(
                    check_id="NULL_PLAYER_FIELDS",
                    severity=Severity.WARNING,
                    message="PlayerBox records with NULL critical fields (pts, team_id)",
                    count=null_player_fields,
                    sample_data=[],
                    fixable=True,
                )
            )

        # Check 7: NULL critical fields in TeamBox
        cursor.execute(
            f"""
            SELECT COUNT(*) FROM TeamBox tb
            JOIN Games g ON tb.game_id = g.game_id
            WHERE tb.game_id IN ({placeholders})
            AND g.status = 3  -- Only check completed games
            AND (tb.pts IS NULL OR tb.team_id IS NULL)
        """,
            game_ids,
        )
        null_team_fields = cursor.fetchone()[0]

        if null_team_fields > 0:
            result.issues.append(
                ValidationIssue(
                    check_id="NULL_TEAM_FIELDS",
                    severity=Severity.WARNING,
                    message="TeamBox records with NULL critical fields (pts, team_id)",
                    count=null_team_fields,
                    sample_data=[],
                    fixable=True,
                )
            )

        return result


class FeaturesValidator(BaseValidator):
    """Validator for Features stage (feature sets for ML predictions)."""

    def __init__(self):
        super().__init__("Features")

    def validate(self, game_ids: List[str], cursor) -> ValidationResult:
        """
        Validate feature data for given game IDs.

        Checks:
        - Missing Features records for games that should have them
        - Empty feature sets (games with prior states but no features)
        - Feature count consistency (should be 43 features per game)
        - NULL critical fields (game_id, feature_set)
        """
        result = ValidationResult(
            stage_name=self.stage_name, total_checked=len(game_ids)
        )

        if not game_ids:
            return result

        placeholders = ",".join("?" * len(game_ids))

        # Check 1: Missing Features records for games with game_data_finalized
        cursor.execute(
            f"""
            SELECT g.game_id FROM Games g
            LEFT JOIN Features f ON g.game_id = f.game_id
            WHERE g.game_id IN ({placeholders})
            AND g.status = 3  -- Only check completed games
            AND g.game_data_finalized = 1
            AND g.pre_game_data_finalized = 1
            AND f.game_id IS NULL
        """,
            game_ids,
        )
        missing_features = [row[0] for row in cursor.fetchall()]

        if missing_features:
            result.issues.append(
                ValidationIssue(
                    check_id="MISSING_FEATURES",
                    severity=Severity.WARNING,
                    message="Games missing Features records despite being finalized",
                    count=len(missing_features),
                    sample_data=missing_features[:5],
                    fixable=True,
                )
            )

        # Check 2: Empty feature sets (should have features but don't)
        cursor.execute(
            f"""
            SELECT f.game_id FROM Features f
            JOIN Games g ON f.game_id = g.game_id
            WHERE f.game_id IN ({placeholders})
            AND g.pre_game_data_finalized = 1
            AND (f.feature_set IS NULL OR f.feature_set = '{{}}' OR LENGTH(f.feature_set) < 10)
        """,
            game_ids,
        )
        empty_features = [row[0] for row in cursor.fetchall()]

        if empty_features:
            result.issues.append(
                ValidationIssue(
                    check_id="EMPTY_FEATURES",
                    severity=Severity.INFO,
                    message="Games with empty feature sets (likely early season)",
                    count=len(empty_features),
                    sample_data=empty_features[:5],
                    fixable=False,
                )
            )

        # Check 3: Feature count consistency (should be 43 features)
        cursor.execute(
            f"""
            SELECT f.game_id, LENGTH(f.feature_set) - LENGTH(REPLACE(f.feature_set, ':', '')) as key_count
            FROM Features f
            WHERE f.game_id IN ({placeholders})
            AND f.feature_set IS NOT NULL
            AND LENGTH(f.feature_set) > 10
        """,
            game_ids,
        )
        feature_counts = cursor.fetchall()
        inconsistent_counts = [(gid, cnt) for gid, cnt in feature_counts if cnt != 43]

        if inconsistent_counts:
            result.issues.append(
                ValidationIssue(
                    check_id="INCONSISTENT_FEATURE_COUNT",
                    severity=Severity.WARNING,
                    message="Games with unexpected feature count (expected 43)",
                    count=len(inconsistent_counts),
                    sample_data=[
                        f"{gid}: {cnt} features" for gid, cnt in inconsistent_counts[:5]
                    ],
                    fixable=True,
                )
            )

        return result


class PredictionsValidator(BaseValidator):
    """Validator for Predictions stage."""

    def __init__(self):
        super().__init__("Predictions")

    def validate(
        self, game_ids: List[str], cursor, predictor_name: str = None
    ) -> ValidationResult:
        """
        Validate prediction data for given game IDs.

        Checks:
        - Missing predictions for games that should have them
        - Prediction values within reasonable ranges

        Note: Predictions can be generated for completed games (for historical analysis),
        so we don't validate prediction timing.
        """
        result = ValidationResult(
            stage_name=self.stage_name, total_checked=len(game_ids)
        )

        if not game_ids:
            return result

        placeholders = ",".join("?" * len(game_ids))

        # Check 1: Missing predictions for games with features
        if predictor_name:
            cursor.execute(
                f"""
                SELECT g.game_id FROM Games g
                JOIN Features f ON g.game_id = f.game_id
                LEFT JOIN Predictions p ON g.game_id = p.game_id AND p.predictor = ?
                WHERE g.game_id IN ({placeholders})
                AND g.pre_game_data_finalized = 1
                AND LENGTH(f.feature_set) > 10
                AND p.game_id IS NULL
            """,
                [predictor_name] + game_ids,
            )
            missing_predictions = [row[0] for row in cursor.fetchall()]

            if missing_predictions:
                result.issues.append(
                    ValidationIssue(
                        check_id="MISSING_PREDICTIONS",
                        severity=Severity.WARNING,
                        message=f"Games missing predictions for {predictor_name}",
                        count=len(missing_predictions),
                        sample_data=missing_predictions[:5],
                        fixable=True,
                    )
                )

        # Check 2: Prediction values in reasonable range (50-180 points per team)
        cursor.execute(
            f"""
            SELECT p.game_id, p.prediction_set FROM Predictions p
            WHERE p.game_id IN ({placeholders})
        """,
            game_ids,
        )
        import json

        unreasonable_predictions = []

        for game_id, pred_set in cursor.fetchall():
            try:
                pred = json.loads(pred_set)
                # Check both key formats (pred_home_score and home_score)
                home_score = pred.get("pred_home_score") or pred.get("home_score", 0)
                away_score = pred.get("pred_away_score") or pred.get("away_score", 0)
                if home_score and away_score:
                    if not (50 <= home_score <= 180) or not (50 <= away_score <= 180):
                        unreasonable_predictions.append(game_id)
            except (json.JSONDecodeError, TypeError):
                pass  # Skip unparseable JSON silently

        if unreasonable_predictions:
            result.issues.append(
                ValidationIssue(
                    check_id="UNREASONABLE_SCORES",
                    severity=Severity.WARNING,
                    message="Predictions with scores outside reasonable range (50-180)",
                    count=len(unreasonable_predictions),
                    sample_data=unreasonable_predictions[:5],
                    fixable=True,
                )
            )

        return result


# Export main classes
__all__ = [
    "Severity",
    "ValidationIssue",
    "ValidationResult",
    "BaseValidator",
    "ScheduleValidator",
    "PlayerValidator",
    "InjuryValidator",
    "BettingValidator",
    "PbPValidator",
    "GameStatesValidator",
    "BoxscoresValidator",
    "FeaturesValidator",
    "PredictionsValidator",
]
