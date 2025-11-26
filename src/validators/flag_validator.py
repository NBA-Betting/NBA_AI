"""Validator for finalization flags logic."""

from typing import List

from .base_validator import BaseValidator, ValidationIssue


class FlagValidator(BaseValidator):
    """Validates game_data_finalized and pre_game_data_finalized flags."""

    def validate(self) -> List[ValidationIssue]:
        """Run all flag validation checks."""
        self.issues = []

        self._check_game_data_flag_missing_final_state()
        self._check_game_data_flag_without_pbp()
        self._check_pre_game_flag_missing_features()
        self._check_pre_game_flag_missing_prior_states()
        self._check_flag_consistency()
        self._check_premature_finalization()

        return self.issues

    def _check_game_data_flag_missing_final_state(self):
        """FLAG-001: Games with game_data_finalized=1 but no final GameState."""
        query = """
        SELECT g.game_id, g.status, g.date_time_est
        FROM Games g
        WHERE g.game_data_finalized = 1
        AND NOT EXISTS (
            SELECT 1 FROM GameStates gs
            WHERE gs.game_id = g.game_id AND gs.is_final_state = 1
        )
        """
        results = self.run_query_dict(query)

        if results:
            self.add_issue(
                check_id="FLAG-001",
                severity="critical",
                message=f"Games marked finalized without final GameState",
                count=len(results),
                sample_data=results[:5],
                fixable=True,
            )

    def _check_game_data_flag_without_pbp(self):
        """FLAG-002: Games with game_data_finalized=1 but no PBP data."""
        query = """
        SELECT g.game_id, g.status, g.date_time_est
        FROM Games g
        WHERE g.game_data_finalized = 1
        AND NOT EXISTS (
            SELECT 1 FROM PbP_Logs p WHERE p.game_id = g.game_id
        )
        """
        results = self.run_query_dict(query)

        if results:
            self.add_issue(
                check_id="FLAG-002",
                severity="critical",
                message=f"Games marked finalized without PBP data",
                count=len(results),
                sample_data=results[:5],
                fixable=True,
            )

    def _check_pre_game_flag_missing_features(self):
        """FLAG-003: Games with pre_game_data_finalized=1 but no Features."""
        query = """
        SELECT g.game_id, g.home_team, g.away_team, g.date_time_est
        FROM Games g
        WHERE g.pre_game_data_finalized = 1
        AND NOT EXISTS (
            SELECT 1 FROM Features f WHERE f.game_id = g.game_id
        )
        """
        results = self.run_query_dict(query)

        if results:
            self.add_issue(
                check_id="FLAG-003",
                severity="critical",
                message=f"Games marked pre-game finalized without Features",
                count=len(results),
                sample_data=results[:5],
                fixable=False,
            )

    def _check_pre_game_flag_missing_prior_states(self):
        """FLAG-004: Games with pre_game_data_finalized=1 but teams missing prior games."""
        query = """
        WITH PriorGames AS (
            SELECT 
                g1.game_id,
                g1.home_team,
                g1.away_team,
                g1.date_time_est,
                (SELECT COUNT(*) FROM Games g2 
                 WHERE (g2.home_team = g1.home_team OR g2.away_team = g1.home_team)
                 AND g2.date_time_est < g1.date_time_est
                 AND g2.season = g1.season
                 AND g2.season_type IN ('Regular Season', 'Post Season')
                 AND g2.game_data_finalized = 1) as home_prior_count,
                (SELECT COUNT(*) FROM Games g2 
                 WHERE (g2.home_team = g1.away_team OR g2.away_team = g1.away_team)
                 AND g2.date_time_est < g1.date_time_est
                 AND g2.season = g1.season
                 AND g2.season_type IN ('Regular Season', 'Post Season')
                 AND g2.game_data_finalized = 1) as away_prior_count
            FROM Games g1
            WHERE g1.pre_game_data_finalized = 1
        )
        SELECT game_id, home_team, away_team, date_time_est, 
               home_prior_count, away_prior_count
        FROM PriorGames
        WHERE home_prior_count = 0 OR away_prior_count = 0
        """
        results = self.run_query_dict(query)

        if results:
            self.add_issue(
                check_id="FLAG-004",
                severity="warning",
                message=f"Pre-game finalized but teams have no prior finalized games",
                count=len(results),
                sample_data=results[:5],
                fixable=False,
            )

    def _check_flag_consistency(self):
        """FLAG-005: Pre-game finalized but game data not finalized."""
        query = """
        SELECT game_id, home_team, away_team, status
        FROM Games
        WHERE pre_game_data_finalized = 1
        AND game_data_finalized = 0
        """
        results = self.run_query_dict(query)

        if results:
            self.add_issue(
                check_id="FLAG-005",
                severity="critical",
                message=f"Pre-game finalized without game data finalized (logic error)",
                count=len(results),
                sample_data=results[:5],
                fixable=True,
            )

    def _check_premature_finalization(self):
        """FLAG-006: Completed games not marked finalized."""
        query = """
        SELECT g.game_id, g.status, g.date_time_est
        FROM Games g
        WHERE g.status = 'Completed'
        AND g.game_data_finalized = 0
        AND EXISTS (
            SELECT 1 FROM GameStates gs
            WHERE gs.game_id = g.game_id AND gs.is_final_state = 1
        )
        """
        results = self.run_query_dict(query)

        if results:
            self.add_issue(
                check_id="FLAG-006",
                severity="warning",
                message=f"Completed games with final state but flag not set",
                count=len(results),
                sample_data=results[:5],
                fixable=True,
            )

    def fix(self, check_id: str = None) -> int:
        """Fix flag inconsistencies."""
        fixed = 0

        if check_id in [None, "FLAG-001"]:
            # Unset game_data_finalized if no final state exists
            query = """
            UPDATE Games SET game_data_finalized = 0
            WHERE game_data_finalized = 1
            AND NOT EXISTS (
                SELECT 1 FROM GameStates gs
                WHERE gs.game_id = Games.game_id AND gs.is_final_state = 1
            )
            """
            fixed += self.execute_update(query)

        if check_id in [None, "FLAG-002"]:
            # Unset game_data_finalized if no PBP exists
            query = """
            UPDATE Games SET game_data_finalized = 0
            WHERE game_data_finalized = 1
            AND NOT EXISTS (
                SELECT 1 FROM PbP_Logs p WHERE p.game_id = Games.game_id
            )
            """
            fixed += self.execute_update(query)

        if check_id in [None, "FLAG-005"]:
            # Unset pre_game_data_finalized if game_data not finalized
            query = """
            UPDATE Games SET pre_game_data_finalized = 0
            WHERE pre_game_data_finalized = 1
            AND game_data_finalized = 0
            """
            fixed += self.execute_update(query)

        if check_id in [None, "FLAG-006"]:
            # Set game_data_finalized for completed games with final state
            query = """
            UPDATE Games SET game_data_finalized = 1
            WHERE status = 'Completed'
            AND game_data_finalized = 0
            AND EXISTS (
                SELECT 1 FROM GameStates gs
                WHERE gs.game_id = Games.game_id AND gs.is_final_state = 1
            )
            """
            fixed += self.execute_update(query)

        return fixed
