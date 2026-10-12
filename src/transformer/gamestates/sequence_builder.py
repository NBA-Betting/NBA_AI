"""
Sequence Builder for GameStates Model.

Constructs input data for each training sample using GameStates score trajectories
instead of play-by-play data. Reuses the same temporal ordering and prior game
lookup logic as the PBP sequence builder.

Usage:
    builder = GameStateSequenceBuilder(tokenizer, n_games=5)
    arrays = builder.build_sequence(target_game_id)
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.database import get_db
from src.transformer.gamestates.tokenizer import GameStateTokenizer
from src.utils import to_eastern_date

# Cap for GameStates rows per game. Typical game has ~400-500 rows.
MAX_GAMESTATES_ROWS = 750


@dataclass
class GSTeamHistory:
    """Historical GameStates data for a single team."""
    team: str
    game_data: list[dict]  # List of tokenized game dicts from GameStateTokenizer
    game_ids: list[str]


@dataclass
class GSMatchupSequence:
    """Complete sequence data for one prediction sample."""
    target_game_id: str
    target_date: str
    home_team: str
    away_team: str
    home_history: GSTeamHistory
    away_history: GSTeamHistory
    home_score: Optional[int]
    away_score: Optional[int]
    final_home_score: Optional[int] = None
    final_away_score: Optional[int] = None
    is_overtime: bool = False


class GameStateSequenceBuilder:
    """
    Builds historical game sequences using GameStates score trajectories.

    Same temporal ordering as the PBP SequenceBuilder — fetches the last N
    completed games for both teams before the target game date.
    """

    def __init__(
        self,
        tokenizer: GameStateTokenizer,
        n_games: int = 5,
        db_path: Optional[str] = None,
    ):
        self.tokenizer = tokenizer
        self.n_games = n_games

        if db_path is None:
            from src.config import config
            db_path = config["database"]["path"]
        self.db_path = db_path

        self._game_cache: dict[str, dict] = {}

    def _get_prior_games(self, team: str, before_date: str, n_games: int) -> list[str]:
        """Get game IDs for the last N games before a given date."""
        with get_db(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT game_id
                FROM Games
                WHERE (home_team = ? OR away_team = ?)
                AND date(date_time_utc, '-5 hours') < date(?)
                AND game_data_finalized = 1
                AND status = 3
                AND season_type IN ('Regular Season', 'Post Season')
                ORDER BY date_time_utc DESC
                LIMIT ?
                """,
                (team, team, before_date, n_games),
            )
            return [row[0] for row in cursor.fetchall()]

    def _get_game_with_cache(self, game_id: str) -> Optional[dict]:
        """Get tokenized GameStates, using cache if available."""
        if game_id not in self._game_cache:
            tokenized = self.tokenizer.tokenize_game(game_id)
            if tokenized:
                self._game_cache[game_id] = tokenized
            else:
                return None
        return self._game_cache[game_id]

    def _get_team_history(self, team: str, before_date: str) -> GSTeamHistory:
        """Get historical GameStates sequence for a team."""
        game_ids = self._get_prior_games(team, before_date, self.n_games)

        game_data = []
        valid_game_ids = []
        for game_id in game_ids:
            tokenized = self._get_game_with_cache(game_id)
            if tokenized:
                game_data.append(tokenized)
                valid_game_ids.append(game_id)

        return GSTeamHistory(team=team, game_data=game_data, game_ids=valid_game_ids)

    def build_sequence(self, target_game_id: str) -> Optional[GSMatchupSequence]:
        """Build complete sequence data for a target game."""
        with get_db(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT home_team, away_team, date_time_utc, status,
                       (SELECT home_score FROM GameStates
                        WHERE game_id = g.game_id AND is_final_state = 1) as final_home_score,
                       (SELECT away_score FROM GameStates
                        WHERE game_id = g.game_id AND is_final_state = 1) as final_away_score,
                       (SELECT home_score FROM GameStates
                        WHERE game_id = g.game_id AND period <= 4
                        ORDER BY play_id DESC LIMIT 1) as reg_home_score,
                       (SELECT away_score FROM GameStates
                        WHERE game_id = g.game_id AND period <= 4
                        ORDER BY play_id DESC LIMIT 1) as reg_away_score,
                       EXISTS(SELECT 1 FROM GameStates
                              WHERE game_id = g.game_id AND period > 4) as is_overtime
                FROM Games g
                WHERE game_id = ?
                """,
                (target_game_id,),
            )
            row = cursor.fetchone()

        if not row:
            logging.warning(f"Target game {target_game_id} not found")
            return None

        (home_team, away_team, date_time_utc, status,
         final_home_score, final_away_score,
         reg_home_score, reg_away_score, is_overtime) = row

        target_date = to_eastern_date(date_time_utc)

        if is_overtime:
            train_home = reg_home_score
            train_away = reg_away_score
        else:
            train_home = final_home_score
            train_away = final_away_score

        home_history = self._get_team_history(home_team, target_date)
        away_history = self._get_team_history(away_team, target_date)

        return GSMatchupSequence(
            target_game_id=target_game_id,
            target_date=target_date,
            home_team=home_team,
            away_team=away_team,
            home_history=home_history,
            away_history=away_history,
            home_score=train_home if status == 3 else None,
            away_score=train_away if status == 3 else None,
            final_home_score=final_home_score if status == 3 else None,
            final_away_score=final_away_score if status == 3 else None,
            is_overtime=bool(is_overtime),
        )

    def get_training_game_ids(
        self,
        seasons: list[str],
        min_history_games: int = 3,
    ) -> list[str]:
        """Get all valid game IDs for training from specified seasons."""
        with get_db(self.db_path) as conn:
            placeholders = ",".join(["?"] * len(seasons))
            cursor = conn.execute(
                f"""
                SELECT game_id, home_team, away_team, date_time_utc
                FROM Games
                WHERE season IN ({placeholders})
                AND status = 3
                AND game_data_finalized = 1
                AND season_type IN ('Regular Season', 'Post Season')
                ORDER BY date_time_utc
                """,
                seasons,
            )
            all_games = cursor.fetchall()

        valid_game_ids = []
        team_game_counts: dict[str, int] = {}

        for game_id, home_team, away_team, date_time_utc in all_games:
            home_count = team_game_counts.get(home_team, 0)
            away_count = team_game_counts.get(away_team, 0)

            if home_count >= min_history_games and away_count >= min_history_games:
                valid_game_ids.append(game_id)

            team_game_counts[home_team] = home_count + 1
            team_game_counts[away_team] = away_count + 1

        return valid_game_ids

    def clear_cache(self):
        """Clear the game cache."""
        self._game_cache.clear()

    @property
    def cache_size(self) -> int:
        return len(self._game_cache)


def sequence_to_arrays(sequence: GSMatchupSequence) -> dict:
    """
    Convert a GSMatchupSequence into numpy arrays for PyTorch.

    Each team's history games are padded to the same number of rows.
    """
    def games_to_array(game_data_list: list[dict]) -> Optional[dict]:
        if not game_data_list:
            return None

        feature_keys = ["periods", "clock_buckets", "home_score_buckets",
                        "away_score_buckets", "margin_buckets"]

        game_lengths = [g["length"] for g in game_data_list]
        max_len = min(max(game_lengths), MAX_GAMESTATES_ROWS)

        result = {}
        for key in feature_keys:
            padded = []
            for game, length in zip(game_data_list, game_lengths):
                arr = game[key][:max_len]  # Truncate if needed
                actual_len = len(arr)
                if actual_len < max_len:
                    padding = np.zeros(max_len - actual_len, dtype=arr.dtype)
                    arr = np.concatenate([arr, padding])
                padded.append(arr)
            result[key] = np.stack(padded)

        result["game_lengths"] = np.array(
            [min(l, MAX_GAMESTATES_ROWS) for l in game_lengths], dtype=np.int64
        )
        return result

    return {
        "target_game_id": sequence.target_game_id,
        "target_date": sequence.target_date,
        "home_team": sequence.home_team,
        "away_team": sequence.away_team,
        "home_history": games_to_array(sequence.home_history.game_data),
        "away_history": games_to_array(sequence.away_history.game_data),
        "home_score": sequence.home_score,
        "away_score": sequence.away_score,
        "final_home_score": sequence.final_home_score,
        "final_away_score": sequence.final_away_score,
        "is_overtime": sequence.is_overtime,
    }
