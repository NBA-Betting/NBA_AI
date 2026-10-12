"""L1 player vectors as of a point in time, and roster arrays for L2.

One implementation for everything that consumes L1 vectors: the Phase5 game
cache builder, Phase C training, live predictions and backtests.

Files: data/l2_cache/l1_vectors/{player_id}.npz with game_ids, ability (N, 32),
uncertainty (N, 32) and archetype_weights (10,); row i is the state after the
player's i-th game, in date order.
"""

from __future__ import annotations

from bisect import bisect_left
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
L1_VECTORS_DIR = PROJECT_ROOT / "data" / "l2_cache" / "l1_vectors"

D_ABILITY = 32
N_ARCHETYPES = 10


class L1Store:
    """Lazily loaded L1 vectors with lookups as of a tip-off time."""

    def __init__(self, game_dates: dict[str, str], vectors_dir: Path = L1_VECTORS_DIR):
        """game_dates: {game_id: date_time_utc} covering the vectors' games."""
        self.vectors_dir = Path(vectors_dir)
        self.game_dates = game_dates
        self._players: dict[int, tuple | None] = {}

    def player_ids(self) -> list[int]:
        return sorted(int(p.stem) for p in self.vectors_dir.glob("*.npz"))

    def _load(self, pid: int):
        if pid not in self._players:
            path = self.vectors_dir / f"{pid}.npz"
            if path.exists():
                data = np.load(path, allow_pickle=True)
                dates = [self.game_dates.get(str(g), "") for g in data["game_ids"]]
                self._players[pid] = (
                    dates,
                    data["ability"],
                    data["uncertainty"],
                    data["archetype_weights"],
                )
            else:
                self._players[pid] = None
        return self._players[pid]

    def state_before(self, pid: int, before_date: str):
        """(ability, uncertainty, archetype_weights) after the player's last
        game before before_date, or None if he has no earlier game."""
        player = self._load(pid)
        if player is None:
            return None
        dates, ability, uncertainty, archetypes = player
        i = bisect_left(dates, before_date)
        if i == 0:
            return None
        return ability[i - 1], uncertainty[i - 1], archetypes

    def roster_arrays(
        self,
        rosters: list[list[int]],
        before_dates: list[str],
        player_to_idx: dict[int, int],
        max_players: int,
    ) -> dict[str, np.ndarray]:
        """L2 inputs for rosters (one per team-game, ordered by projected
        minutes): the first max_players players with an earlier L1 state.

        Returns arrays of shape (n, max_players, ...): abilities, uncertainties,
        archetypes, mask (bool), player_idx, player_ids.
        """
        n = len(rosters)
        out = {
            "abilities": np.zeros((n, max_players, D_ABILITY), dtype=np.float32),
            "uncertainties": np.zeros((n, max_players, D_ABILITY), dtype=np.float32),
            "archetypes": np.zeros((n, max_players, N_ARCHETYPES), dtype=np.float32),
            "mask": np.zeros((n, max_players), dtype=bool),
            "player_idx": np.zeros((n, max_players), dtype=np.int64),
            "player_ids": np.zeros((n, max_players), dtype=np.int64),
        }
        for k, (roster, before_date) in enumerate(zip(rosters, before_dates)):
            j = 0
            for pid in roster:
                if j == max_players:
                    break
                state = self.state_before(int(pid), before_date)
                if state is None:
                    continue
                (
                    out["abilities"][k, j],
                    out["uncertainties"][k, j],
                    out["archetypes"][k, j],
                ) = state
                out["mask"][k, j] = True
                out["player_idx"][k, j] = player_to_idx.get(int(pid), 0)
                out["player_ids"][k, j] = pid
                j += 1
        return out


def load_game_dates(conn) -> dict[str, str]:
    """{game_id: date_time_utc} for all games."""
    return dict(conn.execute("SELECT game_id, date_time_utc FROM Games").fetchall())
