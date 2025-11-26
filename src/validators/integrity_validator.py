"""Validator for data integrity checks."""

from typing import List

from .base_validator import BaseValidator, ValidationIssue


class IntegrityValidator(BaseValidator):
    """Validates referential integrity and NULL values."""

    def validate(self) -> List[ValidationIssue]:
        """Run all integrity validation checks."""
        self.issues = []

        self._check_orphaned_pbp_logs()
        self._check_orphaned_game_states()
        self._check_orphaned_features()
        self._check_orphaned_predictions()
        self._check_null_critical_fields()
        self._check_duplicate_game_states()

        return self.issues

    def _check_orphaned_pbp_logs(self):
        """INTEGRITY-001: PBP_Logs without matching Games record."""
        query = """
        SELECT p.game_id, COUNT(*) as log_count
        FROM PbP_Logs p
        WHERE NOT EXISTS (SELECT 1 FROM Games g WHERE g.game_id = p.game_id)
        GROUP BY p.game_id
        """
        results = self.run_query_dict(query)

        if results:
            self.add_issue(
                check_id="INTEGRITY-001",
                severity="critical",
                message=f"PBP_Logs without matching Games record",
                count=len(results),
                sample_data=results[:10],
                fixable=True,
            )

    def _check_orphaned_game_states(self):
        """INTEGRITY-002: GameStates without matching Games record."""
        query = """
        SELECT gs.game_id, COUNT(*) as state_count
        FROM GameStates gs
        WHERE NOT EXISTS (SELECT 1 FROM Games g WHERE g.game_id = gs.game_id)
        GROUP BY gs.game_id
        """
        results = self.run_query_dict(query)

        if results:
            self.add_issue(
                check_id="INTEGRITY-002",
                severity="critical",
                message=f"GameStates without matching Games record",
                count=len(results),
                sample_data=results[:10],
                fixable=True,
            )

    def _check_orphaned_features(self):
        """INTEGRITY-003: Features without matching Games record."""
        query = """
        SELECT f.game_id
        FROM Features f
        WHERE NOT EXISTS (SELECT 1 FROM Games g WHERE g.game_id = f.game_id)
        """
        results = self.run_query_dict(query)

        if results:
            self.add_issue(
                check_id="INTEGRITY-003",
                severity="critical",
                message=f"Features without matching Games record",
                count=len(results),
                sample_data=results[:10],
                fixable=True,
            )

    def _check_orphaned_predictions(self):
        """INTEGRITY-004: Predictions without matching Games record."""
        query = """
        SELECT p.game_id, p.predictor
        FROM Predictions p
        WHERE NOT EXISTS (SELECT 1 FROM Games g WHERE g.game_id = p.game_id)
        """
        results = self.run_query_dict(query)

        if results:
            self.add_issue(
                check_id="INTEGRITY-004",
                severity="warning",
                message=f"Predictions without matching Games record",
                count=len(results),
                sample_data=results[:10],
                fixable=True,
            )

    def _check_null_critical_fields(self):
        """INTEGRITY-005: NULL values in critical fields."""
        checks = [
            ("Games", ["game_id", "home_team", "away_team", "date_time_est", "season"]),
            (
                "GameStates",
                ["game_id", "play_id", "home", "away", "home_score", "away_score"],
            ),
            ("Features", ["game_id"]),
        ]

        null_issues = []
        for table, fields in checks:
            for field in fields:
                query = f"SELECT COUNT(*) FROM {table} WHERE {field} IS NULL"
                result = self.run_query(query)
                if result and result[0][0] > 0:
                    null_issues.append(
                        {"table": table, "field": field, "count": result[0][0]}
                    )

        if null_issues:
            self.add_issue(
                check_id="INTEGRITY-005",
                severity="critical",
                message=f"NULL values in critical fields",
                count=sum(x["count"] for x in null_issues),
                sample_data=null_issues,
                fixable=False,
            )

    def _check_duplicate_game_states(self):
        """INTEGRITY-006: Duplicate GameStates (same game_id + play_id)."""
        query = """
        SELECT game_id, play_id, COUNT(*) as dup_count
        FROM GameStates
        GROUP BY game_id, play_id
        HAVING COUNT(*) > 1
        LIMIT 100
        """
        results = self.run_query_dict(query)

        if results:
            self.add_issue(
                check_id="INTEGRITY-006",
                severity="critical",
                message=f"Duplicate GameStates detected (same game_id + play_id)",
                count=len(results),
                sample_data=results[:10],
                fixable=True,
            )

    def fix(self, check_id: str = None) -> int:
        """Fix integrity issues."""
        fixed = 0

        if check_id in [None, "INTEGRITY-001"]:
            # Delete orphaned PBP_Logs
            query = """
            DELETE FROM PbP_Logs
            WHERE NOT EXISTS (SELECT 1 FROM Games g WHERE g.game_id = PbP_Logs.game_id)
            """
            fixed += self.execute_update(query)

        if check_id in [None, "INTEGRITY-002"]:
            # Delete orphaned GameStates
            query = """
            DELETE FROM GameStates
            WHERE NOT EXISTS (SELECT 1 FROM Games g WHERE g.game_id = GameStates.game_id)
            """
            fixed += self.execute_update(query)

        if check_id in [None, "INTEGRITY-003"]:
            # Delete orphaned Features
            query = """
            DELETE FROM Features
            WHERE NOT EXISTS (SELECT 1 FROM Games g WHERE g.game_id = Features.game_id)
            """
            fixed += self.execute_update(query)

        if check_id in [None, "INTEGRITY-004"]:
            # Delete orphaned Predictions
            query = """
            DELETE FROM Predictions
            WHERE NOT EXISTS (SELECT 1 FROM Games g WHERE g.game_id = Predictions.game_id)
            """
            fixed += self.execute_update(query)

        if check_id in [None, "INTEGRITY-006"]:
            # Keep only first occurrence of duplicates
            query = """
            DELETE FROM GameStates
            WHERE rowid NOT IN (
                SELECT MIN(rowid)
                FROM GameStates
                GROUP BY game_id, play_id
            )
            """
            fixed += self.execute_update(query)

        return fixed
