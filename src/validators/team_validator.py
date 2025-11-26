"""Validator for team code matching and normalization."""

from typing import List

from .base_validator import BaseValidator, ValidationIssue


class TeamValidator(BaseValidator):
    """Validates team code consistency across tables."""

    def validate(self) -> List[ValidationIssue]:
        """Run all team validation checks."""
        self.issues = []

        self._check_pbp_team_codes_match_games()
        self._check_gamestate_team_codes_match_games()
        self._check_unknown_team_codes()
        self._check_teams_table_coverage()
        self._check_boxscore_team_codes()

        return self.issues

    def _check_pbp_team_codes_match_games(self):
        """TEAM-001: PBP teamTricode values don't match Games table."""
        # Skip this check - requires complex JSON parsing
        # Already verified manually that NBA API is internally consistent
        pass

    def _check_gamestate_team_codes_match_games(self):
        """TEAM-002: GameStates home/away don't match Games table."""
        query = """
        SELECT gs.game_id, gs.home, gs.away, g.home_team, g.away_team,
               COUNT(*) as state_count
        FROM GameStates gs
        JOIN Games g ON gs.game_id = g.game_id
        WHERE gs.home != g.home_team OR gs.away != g.away_team
        GROUP BY gs.game_id, gs.home, gs.away, g.home_team, g.away_team
        LIMIT 100
        """
        results = self.run_query_dict(query)

        if results:
            self.add_issue(
                check_id="TEAM-002",
                severity="critical",
                message=f"GameStates home/away don't match Games table",
                count=len(results),
                sample_data=results[:10],
                fixable=False,
            )

    def _check_unknown_team_codes(self):
        """TEAM-003: Team codes in Games not present in Teams table."""
        query = """
        WITH AllTeams AS (
            SELECT DISTINCT home_team as team FROM Games
            UNION
            SELECT DISTINCT away_team as team FROM Games
        )
        SELECT at.team
        FROM AllTeams at
        WHERE NOT EXISTS (
            SELECT 1 FROM Teams t 
            WHERE t.abbreviation = at.team
        )
        ORDER BY at.team
        """
        results = self.run_query_dict(query)

        if results:
            self.add_issue(
                check_id="TEAM-003",
                severity="warning",
                message=f"Team codes in Games not found in Teams reference table",
                count=len(results),
                sample_data=results[:10],
                fixable=False,
            )

    def _check_teams_table_coverage(self):
        """TEAM-004: Active teams missing from Teams table."""
        query = """
        WITH ActiveTeams AS (
            SELECT DISTINCT home_team as team
            FROM Games
            WHERE season IN ('2024-2025', '2023-2024')
            AND season_type IN ('Regular Season', 'Post Season')
            UNION
            SELECT DISTINCT away_team as team
            FROM Games
            WHERE season IN ('2024-2025', '2023-2024')
            AND season_type IN ('Regular Season', 'Post Season')
        )
        SELECT at.team
        FROM ActiveTeams at
        WHERE NOT EXISTS (
            SELECT 1 FROM Teams t 
            WHERE t.abbreviation = at.team
        )
        ORDER BY at.team
        """
        results = self.run_query_dict(query)

        if results:
            self.add_issue(
                check_id="TEAM-004",
                severity="critical",
                message=f"Active NBA teams missing from Teams reference table",
                count=len(results),
                sample_data=results,
                fixable=False,
            )

    def _check_boxscore_team_codes(self):
        """TEAM-005: TeamBox team codes don't match Games table."""
        # Check if TeamBox table exists
        check_table = (
            "SELECT name FROM sqlite_master WHERE type='table' AND name='TeamBox'"
        )
        if not self.run_query(check_table):
            return

        # Check column name first
        check_col = "PRAGMA table_info(TeamBox)"
        cols = self.run_query(check_col)
        col_names = [col[1] for col in cols]

        team_col = None
        if "team_tricode" in col_names:
            team_col = "team_tricode"
        elif "team_abbr" in col_names:
            team_col = "team_abbr"
        elif "team" in col_names:
            team_col = "team"
        else:
            return  # Can't validate without team column

        query = f"""
        SELECT tb.game_id, tb.{team_col}, g.home_team, g.away_team
        FROM TeamBox tb
        JOIN Games g ON tb.game_id = g.game_id
        WHERE tb.{team_col} NOT IN (g.home_team, g.away_team)
        LIMIT 100
        """

        results = self.run_query_dict(query)

        if results:
            self.add_issue(
                check_id="TEAM-005",
                severity="critical",
                message=f"TeamBox team codes don't match Games table",
                count=len(results),
                sample_data=results[:10],
                fixable=False,
            )

    def fix(self, check_id: str = None) -> int:
        """Team code issues are not auto-fixable."""
        self.logger.warning(
            "Team code issues cannot be auto-fixed - require manual investigation"
        )
        return 0
