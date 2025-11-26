"""Stub validators - to be implemented."""

from typing import List

from .base_validator import BaseValidator, ValidationIssue


class VolumeValidator(BaseValidator):
    """Validates game data volume and counts."""

    def validate(self) -> List[ValidationIssue]:
        """Run volume validation checks."""
        self.issues = []
        self._check_pbp_play_counts()
        self._check_gamestate_counts()
        return self.issues

    def _check_pbp_play_counts(self):
        """VOL-001: PBP play counts outside normal range."""
        query = """
        SELECT game_id, COUNT(*) as play_count
        FROM PbP_Logs
        GROUP BY game_id
        HAVING COUNT(*) < 300 OR COUNT(*) > 800
        """
        results = self.run_query_dict(query)
        if results:
            self.add_issue(
                "VOL-001",
                "warning",
                "PBP play counts outside normal range (300-800)",
                len(results),
                results[:10],
            )

    def _check_gamestate_counts(self):
        """VOL-002: GameState counts don't match PBP counts."""
        query = """
        SELECT 
            p.game_id,
            COUNT(*) as pbp_count,
            (SELECT COUNT(*) FROM GameStates gs WHERE gs.game_id = p.game_id) as gs_count
        FROM PbP_Logs p
        GROUP BY p.game_id
        HAVING COUNT(*) != (SELECT COUNT(*) FROM GameStates gs WHERE gs.game_id = p.game_id)
        LIMIT 100
        """
        results = self.run_query_dict(query)
        if results:
            self.add_issue(
                "VOL-002",
                "warning",
                "GameState counts don't match PBP counts",
                len(results),
                results[:10],
            )

    def fix(self, check_id: str = None) -> int:
        return 0


class TemporalValidator(BaseValidator):
    """Validates temporal constraints and chronological ordering."""

    def validate(self) -> List[ValidationIssue]:
        """Run temporal validation checks."""
        self.issues = []
        self._check_future_games()
        self._check_chronological_order()
        return self.issues

    def _check_future_games(self):
        """TEMP-001: Completed games in the future."""
        query = """
        SELECT game_id, date_time_est, status
        FROM Games
        WHERE status = 'Completed'
        AND datetime(date_time_est) > datetime('now')
        """
        results = self.run_query_dict(query)
        if results:
            self.add_issue(
                "TEMP-001",
                "warning",
                "Completed games scheduled in future",
                len(results),
                results[:10],
            )

    def _check_chronological_order(self):
        """TEMP-002: Play IDs not chronological."""
        query = """
        SELECT DISTINCT game_id
        FROM GameStates gs1
        WHERE EXISTS (
            SELECT 1 FROM GameStates gs2
            WHERE gs2.game_id = gs1.game_id
            AND gs2.play_id > gs1.play_id
            AND gs2.rowid < gs1.rowid
        )
        LIMIT 100
        """
        results = self.run_query_dict(query)
        if results:
            self.add_issue(
                "TEMP-002",
                "warning",
                "Play IDs not in chronological order (expected for live-collected games)",
                len(results),
                results[:10],
            )

    def fix(self, check_id: str = None) -> int:
        return 0


class AlignmentValidator(BaseValidator):
    """Validates PBP to GameState alignment."""

    def validate(self) -> List[ValidationIssue]:
        """Run alignment validation checks."""
        self.issues = []
        return self.issues

    def fix(self, check_id: str = None) -> int:
        return 0


class PriorStateValidator(BaseValidator):
    """Validates prior state selection and usage."""

    def validate(self) -> List[ValidationIssue]:
        """Run prior state validation checks."""
        self.issues = []
        return self.issues

    def fix(self, check_id: str = None) -> int:
        return 0


class FeatureValidator(BaseValidator):
    """Validates feature calculation correctness."""

    def validate(self) -> List[ValidationIssue]:
        """Run feature validation checks."""
        self.issues = []
        return self.issues

    def fix(self, check_id: str = None) -> int:
        return 0
