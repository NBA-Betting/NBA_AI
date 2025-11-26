"""Validator for score consistency checks."""

from typing import List

from .base_validator import BaseValidator, ValidationIssue


class ScoreValidator(BaseValidator):
    """Validates score consistency across tables."""

    def validate(self) -> List[ValidationIssue]:
        """Run all score validation checks."""
        self.issues = []

        self._check_score_monotonicity()
        self._check_negative_scores()
        self._check_final_score_consistency()
        self._check_score_gaps()

        return self.issues

    def _check_score_monotonicity(self):
        """SCORE-001: Scores decrease within a game."""
        query = """
        SELECT DISTINCT game_id, play_id, period, home_score, away_score
        FROM GameStates gs1
        WHERE EXISTS (
            SELECT 1 FROM GameStates gs2
            WHERE gs2.game_id = gs1.game_id
            AND gs2.period = gs1.period
            AND gs2.play_id < gs1.play_id
            AND (gs2.home_score > gs1.home_score OR gs2.away_score > gs1.away_score)
        )
        LIMIT 100
        """
        results = self.run_query_dict(query)

        if results:
            self.add_issue(
                check_id="SCORE-001",
                severity="critical",
                message=f"Scores decreased within same period (non-monotonic)",
                count=len(results),
                sample_data=results[:5],
                fixable=False,
            )

    def _check_negative_scores(self):
        """SCORE-002: Negative scores found."""
        query = """
        SELECT game_id, play_id, period, home_score, away_score
        FROM GameStates
        WHERE home_score < 0 OR away_score < 0
        LIMIT 100
        """
        results = self.run_query_dict(query)

        if results:
            self.add_issue(
                check_id="SCORE-002",
                severity="critical",
                message=f"Negative scores detected",
                count=len(results),
                sample_data=results[:10],
                fixable=False,
            )

    def _check_final_score_consistency(self):
        """SCORE-003: Multiple different final scores for same game."""
        query = """
        SELECT game_id, COUNT(DISTINCT home_score || '-' || away_score) as unique_finals,
               GROUP_CONCAT(DISTINCT home_score || '-' || away_score) as scores
        FROM GameStates
        WHERE is_final_state = 1
        GROUP BY game_id
        HAVING COUNT(DISTINCT home_score || '-' || away_score) > 1
        """
        results = self.run_query_dict(query)

        if results:
            self.add_issue(
                check_id="SCORE-003",
                severity="critical",
                message=f"Games with multiple different final scores",
                count=len(results),
                sample_data=results[:10],
                fixable=False,
            )

    def _check_score_gaps(self):
        """SCORE-004: Unrealistic score increases (>10 points in single play)."""
        query = """
        WITH ScoreChanges AS (
            SELECT 
                game_id,
                play_id,
                home_score - LAG(home_score) OVER (PARTITION BY game_id ORDER BY play_id) as home_diff,
                away_score - LAG(away_score) OVER (PARTITION BY game_id ORDER BY play_id) as away_diff
            FROM GameStates
        )
        SELECT game_id, play_id, home_diff, away_diff
        FROM ScoreChanges
        WHERE home_diff > 10 OR away_diff > 10
        LIMIT 100
        """
        results = self.run_query_dict(query)

        if results:
            self.add_issue(
                check_id="SCORE-004",
                severity="warning",
                message=f"Unrealistic score jumps detected (>10 points in one play)",
                count=len(results),
                sample_data=results[:10],
                fixable=False,
            )

    def fix(self, check_id: str = None) -> int:
        """Score issues cannot be auto-fixed."""
        self.logger.warning("Score issues require data re-collection - cannot auto-fix")
        return 0
