"""
Phase 2 Sequence Builder.

Constructs full-season context from the pre-built cache.
For a target game, returns all prior same-season games for both teams.
"""

import bisect
import logging
from dataclasses import dataclass, field
from typing import Optional

from src.transformer.phase2.cache_builder import PerGameFeatures, TEAM_TO_IDX

logger = logging.getLogger(__name__)


@dataclass
class Phase2Sample:
    """Complete sample for a target game."""

    game_id: str
    season: str

    # Per-team context (lists of PerGameFeatures, chronologically ordered)
    home_context: list[PerGameFeatures] = field(default_factory=list)
    away_context: list[PerGameFeatures] = field(default_factory=list)

    # Indices into context lists marking recent games
    home_recent_indices: list[int] = field(default_factory=list)
    away_recent_indices: list[int] = field(default_factory=list)

    # Days before target for each context game
    home_days_before: list[int] = field(default_factory=list)
    away_days_before: list[int] = field(default_factory=list)

    # Target game info
    home_team: str = ""
    away_team: str = ""
    home_roster: list[int] = field(default_factory=list)
    away_roster: list[int] = field(default_factory=list)
    home_rest_days: int = 0
    away_rest_days: int = 0

    # Labels (regulation scores)
    target_home_score: float = 0.0
    target_away_score: float = 0.0


class Phase2SequenceBuilder:
    """Builds full-season context sequences from cached features."""

    def __init__(
        self,
        game_features: dict[str, PerGameFeatures],
        season_index: dict[tuple, list[str]],
        n_recent_games: int = 5,
        max_context_games: int = 82,
        projected_rosters: Optional[dict[str, tuple[list[int], list[int]]]] = None,
    ):
        self.game_features = game_features
        self.season_index = season_index
        # {game_id: (home, away)} pre-game rosters, as live predictions see them
        self.projected_rosters = projected_rosters or {}
        self.n_recent_games = n_recent_games
        self.max_context_games = max_context_games

        # Pre-compute game dates for efficient binary search
        self._game_dates = {
            gid: gf.game_date for gid, gf in game_features.items()
        }

    def _get_prior_games(self, team: str, season: str, target_date: str) -> list[PerGameFeatures]:
        """Get all games for team in season before target_date, chronologically."""
        key = (team, season)
        game_ids = self.season_index.get(key, [])
        if not game_ids:
            return []

        # Filter to games before target date
        prior = []
        for gid in game_ids:
            gf = self.game_features[gid]
            if gf.game_date < target_date:
                prior.append(gf)

        # Sort by date (should already be sorted, but ensure)
        prior.sort(key=lambda gf: gf.game_date)

        # Limit to max context games
        if len(prior) > self.max_context_games:
            prior = prior[-self.max_context_games:]

        return prior

    def _compute_days_before(self, context_date: str, target_date: str) -> int:
        """Compute days between two date strings (YYYY-MM-DD)."""
        from datetime import datetime
        try:
            ctx = datetime.strptime(context_date[:10], "%Y-%m-%d")
            tgt = datetime.strptime(target_date[:10], "%Y-%m-%d")
            return max(0, (tgt - ctx).days)
        except (ValueError, TypeError):
            return 0

    def _extract_roster(self, game_features: PerGameFeatures, is_home: bool) -> list[int]:
        """Extract player IDs from a game's player data (keys only, no points)."""
        stats = game_features.home_player_stats if is_home else game_features.away_player_stats
        if stats:
            return [entry[0] for entry in stats]
        players = game_features.home_player_points if is_home else game_features.away_player_points
        return [entry[0] for entry in players]

    def _compute_rest_days(self, context: list[PerGameFeatures], target_date: str) -> int:
        """Compute rest days: days since team's last game before target."""
        if not context:
            return 7  # Default if no prior games
        last_game = context[-1]
        return self._compute_days_before(last_game.game_date, target_date)

    def build_sample(self, game_id: str) -> Optional[Phase2Sample]:
        """Build a complete Phase2Sample for a target game."""
        target = self.game_features.get(game_id)
        if target is None:
            return None

        home_team = target.home_team
        away_team = target.away_team
        season = target.season
        target_date = target.game_date

        # Get full-season context for both teams
        home_context = self._get_prior_games(home_team, season, target_date)
        away_context = self._get_prior_games(away_team, season, target_date)

        # Compute days before target for each context game
        home_days_before = [
            self._compute_days_before(gf.game_date, target_date)
            for gf in home_context
        ]
        away_days_before = [
            self._compute_days_before(gf.game_date, target_date)
            for gf in away_context
        ]

        # Mark recent game indices
        n_home = len(home_context)
        n_away = len(away_context)
        home_recent_indices = list(range(
            max(0, n_home - self.n_recent_games), n_home
        ))
        away_recent_indices = list(range(
            max(0, n_away - self.n_recent_games), n_away
        ))

        # Target roster: the pre-game projection when given (training must see
        # what live predictions see), else the players who played
        if game_id in self.projected_rosters:
            home_roster, away_roster = self.projected_rosters[game_id]
        else:
            home_roster = self._extract_roster(target, is_home=True)
            away_roster = self._extract_roster(target, is_home=False)

        # Compute rest days
        home_rest_days = self._compute_rest_days(home_context, target_date)
        away_rest_days = self._compute_rest_days(away_context, target_date)

        # Labels: use regulation scores
        target_home_score = float(target.reg_home_score)
        target_away_score = float(target.reg_away_score)

        return Phase2Sample(
            game_id=game_id,
            season=season,
            home_context=home_context,
            away_context=away_context,
            home_recent_indices=home_recent_indices,
            away_recent_indices=away_recent_indices,
            home_days_before=home_days_before,
            away_days_before=away_days_before,
            home_team=home_team,
            away_team=away_team,
            home_roster=home_roster,
            away_roster=away_roster,
            home_rest_days=home_rest_days,
            away_rest_days=away_rest_days,
            target_home_score=target_home_score,
            target_away_score=target_away_score,
        )

    def get_training_game_ids(
        self, seasons: list[str], min_history: int = 3
    ) -> list[str]:
        """
        Return game_ids where both teams have min_history+ prior same-season games.
        """
        valid_ids = []

        for game_id, gf in self.game_features.items():
            if gf.season not in seasons:
                continue

            home_prior = self._get_prior_games(gf.home_team, gf.season, gf.game_date)
            away_prior = self._get_prior_games(gf.away_team, gf.season, gf.game_date)

            if len(home_prior) >= min_history and len(away_prior) >= min_history:
                valid_ids.append(game_id)

        # Sort by date for reproducibility
        valid_ids.sort(key=lambda gid: self.game_features[gid].game_date)

        logger.info(f"Found {len(valid_ids)} valid games for seasons {seasons} "
                     f"(min_history={min_history})")
        return valid_ids
