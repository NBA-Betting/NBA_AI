"""
Phase 2 Dataset and Collate Function.

Constructs training samples from cached features with full-season context,
per-player contributions, GameStates dynamics, and home/away augmentation.

Changes from baseline:
- Team-relative scores: scores[0] is always the encoded team's score
- Team-relative GameStates: swap home/away score buckets and invert margin when team was away
- Player form tensors: per-roster-player (points, days) appearance history
"""

import logging
import random
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from src.transformer.phase2.cache_builder import (
    TEAM_TO_IDX,
    PerGameFeatures,
    HISTORICAL_TO_CURRENT,
)
from src.transformer.phase2.sequence_builder import Phase2SequenceBuilder

logger = logging.getLogger(__name__)

# Fixed NBA normalization constants
SCORE_MEAN = 108.0
SCORE_STD = 13.0
MARGIN_STD = 13.0
TOTAL_MEAN = 216.0
TOTAL_STD = 20.0
POINTS_NORM = 30.0

# Fixed domain divisors for 16 PlayerBox stats (same order as STAT_COLUMNS)
STAT_NORMS_16 = np.array(
    [
        48.0,
        30.0,
        5.0,
        12.0,
        10.0,
        3.0,
        3.0,
        5.0,
        6.0,
        25.0,
        15.0,
        12.0,
        8.0,
        12.0,
        10.0,
        30.0,
    ],
    dtype=np.float32,
)
STAT_NORMS_17 = np.array(
    [
        48.0,
        30.0,
        5.0,
        12.0,
        10.0,
        3.0,
        3.0,
        5.0,
        6.0,
        25.0,
        15.0,
        12.0,
        8.0,
        12.0,
        10.0,
        30.0,
        20.0,  # years_in_league
    ],
    dtype=np.float32,
)
# Default: 16 stats (set dynamically based on n_player_stats)
STAT_NORMS = STAT_NORMS_16

# Team efficiency normalization constants (mean, std from historical data)
EFFICIENCY_MEANS = np.array(
    [0.539, 0.574, 0.121, 0.262, 0.381, 0.621, 111.3, 0.0], dtype=np.float32
)
EFFICIENCY_STDS = np.array(
    [0.068, 0.063, 0.034, 0.092, 0.099, 0.095, 7.534, 13.0], dtype=np.float32
)


def _normalize_scores(team_score: float, opp_score: float) -> list[float]:
    """Normalize game scores to [team, opponent, margin, total] (team-relative)."""
    return [
        (team_score - SCORE_MEAN) / SCORE_STD,
        (opp_score - SCORE_MEAN) / SCORE_STD,
        (team_score - opp_score) / MARGIN_STD,
        (team_score + opp_score - TOTAL_MEAN) / TOTAL_STD,
    ]


def _compute_gs_summary(gs_data: dict, team_is_away: bool) -> np.ndarray:
    """Compute 6 summary features from GameStates margin_buckets."""
    margin_buckets = gs_data["margin_buckets"]
    if team_is_away:
        margin_buckets = 120 - margin_buckets  # Invert for away team perspective

    margins = margin_buckets.astype(np.float32) - 60.0  # Convert to actual margins

    max_lead = np.max(margins) / 30.0
    max_deficit = np.min(margins) / 30.0
    # Count sign changes (lead changes)
    signs = np.sign(margins)
    signs = signs[signs != 0]  # Remove ties
    lead_changes = np.sum(signs[1:] != signs[:-1]) / 20.0 if len(signs) > 1 else 0.0
    score_volatility = np.std(margins) / 15.0 if len(margins) > 1 else 0.0
    final_margin = margins[-1] if len(margins) > 0 else 0.0
    close_game = 1.0 if abs(final_margin) <= 5 else 0.0
    blowout = 1.0 if abs(final_margin) >= 20 else 0.0

    return np.array(
        [max_lead, max_deficit, lead_changes, score_volatility, close_game, blowout],
        dtype=np.float32,
    )


def _encode_team_context(
    context: list[PerGameFeatures],
    team: str,
    days_before: list[int],
    recent_indices: list[int],
    gs_cache: dict,
    player_id_map: dict[int, int],
    max_players: int = 15,
    max_roster: int = 15,
    roster_ids: Optional[np.ndarray] = None,
    max_appearances: int = 40,
    enable_player_form: bool = False,
    n_player_stats: int = 0,
    n_efficiency_features: int = 0,
    player_experience: Optional[dict] = None,
    season_start_year: Optional[int] = None,
) -> dict:
    """Encode a team's context games into tensor arrays.

    Scores and GameStates are now team-relative: scores[0] is always the
    encoded team's score, margin is from the team's perspective.
    """
    n_games = len(context)

    scores = np.zeros((n_games, 4), dtype=np.float32)
    opponent_ids = np.zeros(n_games, dtype=np.int64)
    location = np.zeros(n_games, dtype=np.int64)
    player_ids = np.zeros((n_games, max_players), dtype=np.int64)
    player_points = np.zeros((n_games, max_players), dtype=np.float32)
    player_mask = np.ones((n_games, max_players), dtype=bool)  # True=padding
    days_before_arr = np.zeros(n_games, dtype=np.int64)
    is_recent = np.zeros(n_games, dtype=bool)

    # Track per-game team-is-away flag for GameStates swapping
    team_is_away = np.zeros(n_games, dtype=bool)

    # Player stats arrays (only when n_player_stats > 0)
    use_stats = n_player_stats > 0
    stat_norms = (
        STAT_NORMS_17[:n_player_stats]
        if n_player_stats == 17
        else STAT_NORMS_16[:n_player_stats]
    )
    if use_stats:
        player_stats_arr = np.zeros(
            (n_games, max_players, n_player_stats), dtype=np.float32
        )
        player_positions = np.full(
            (n_games, max_players), 3, dtype=np.int64
        )  # 3 = UNK default
        player_pm_available = np.zeros((n_games, max_players), dtype=np.float32)

    # Efficiency features (only when enabled)
    use_efficiency = n_efficiency_features > 0
    if use_efficiency:
        efficiency_features = np.zeros(
            (n_games, n_efficiency_features), dtype=np.float32
        )
        gs_summary_features = np.zeros((n_games, 6), dtype=np.float32)
        context_flags = np.zeros((n_games, 2), dtype=np.float32)

    for i, gf in enumerate(context):
        # Opponent and location relative to THIS team
        if gf.home_team == team:
            opponent_ids[i] = TEAM_TO_IDX.get(gf.away_team, 0)
            location[i] = 1  # home
            players = gf.home_player_points
            players_stats = gf.home_player_stats
            # Team-relative scores: team is home, so team_score=home, opp=away
            scores[i] = _normalize_scores(gf.home_score, gf.away_score)
        else:
            opponent_ids[i] = TEAM_TO_IDX.get(gf.home_team, 0)
            location[i] = 0  # away
            players = gf.away_player_points
            players_stats = gf.away_player_stats
            team_is_away[i] = True
            # Team-relative scores: team is away, so team_score=away, opp=home
            scores[i] = _normalize_scores(gf.away_score, gf.home_score)

        # Player contributions
        if use_stats and players_stats:
            for j, (pid, stats, pos_idx, pm_avail) in enumerate(
                players_stats[:max_players]
            ):
                player_ids[i, j] = player_id_map.get(pid, 0)
                raw_stats = np.array(stats, dtype=np.float32)
                if (
                    n_player_stats == 17
                    and player_experience is not None
                    and season_start_year is not None
                ):
                    # Append years_in_league as 17th stat
                    from_year = player_experience.get(pid, season_start_year)
                    years = min(max(season_start_year - from_year, 0), 20)
                    raw_stats = np.append(raw_stats, float(years))
                player_stats_arr[i, j] = raw_stats / stat_norms
                player_points[i, j] = stats[1] / POINTS_NORM  # pts is index 1
                player_positions[i, j] = pos_idx
                player_pm_available[i, j] = float(pm_avail)
                player_mask[i, j] = False
        else:
            # Legacy path: only points
            for j, (pid, pts) in enumerate(players[:max_players]):
                player_ids[i, j] = player_id_map.get(pid, 0)
                player_points[i, j] = pts / POINTS_NORM
                player_mask[i, j] = False

        # Efficiency features (team-relative)
        if use_efficiency:
            if gf.home_team == team:
                eff = gf.home_team_efficiency
            else:
                eff = gf.away_team_efficiency
            if eff:
                eff_arr = np.array(eff, dtype=np.float32)
                efficiency_features[i] = (eff_arr - EFFICIENCY_MEANS) / EFFICIENCY_STDS

            # Context flags
            context_flags[i, 0] = float(gf.is_overtime)
            context_flags[i, 1] = float(gf.is_playoff)

            # GS summary (computed from cached margin_buckets)
            gs_data = gs_cache.get(gf.game_id)
            if gs_data is not None:
                gs_summary_features[i] = _compute_gs_summary(gs_data, team_is_away[i])

        # Days before target
        days_before_arr[i] = min(days_before[i], 179)

    # Mark recent games
    for idx in recent_indices:
        if idx < n_games:
            is_recent[idx] = True

    # Collect GameStates for recent games (with team-relative swapping)
    gs_periods_list = []
    gs_clock_list = []
    gs_home_score_list = []
    gs_away_score_list = []
    gs_margin_list = []
    gs_lengths_list = []

    for idx in recent_indices:
        if idx >= n_games:
            continue
        gf = context[idx]
        gs_data = gs_cache.get(gf.game_id)
        if gs_data is not None:
            gs_periods_list.append(gs_data["periods"])
            gs_clock_list.append(gs_data["clock_buckets"])

            if team_is_away[idx]:
                # Team was away: swap home/away score buckets, invert margin
                gs_home_score_list.append(gs_data["away_score_buckets"])
                gs_away_score_list.append(gs_data["home_score_buckets"])
                # margin_bucket = home_margin + 60, invert: 120 - bucket
                gs_margin_list.append(120 - gs_data["margin_buckets"])
            else:
                gs_home_score_list.append(gs_data["home_score_buckets"])
                gs_away_score_list.append(gs_data["away_score_buckets"])
                gs_margin_list.append(gs_data["margin_buckets"])

            gs_lengths_list.append(gs_data["length"])
        else:
            # Empty placeholder
            gs_periods_list.append(np.zeros(1, dtype=np.int64))
            gs_clock_list.append(np.zeros(1, dtype=np.int64))
            gs_home_score_list.append(np.zeros(1, dtype=np.int64))
            gs_away_score_list.append(np.zeros(1, dtype=np.int64))
            gs_margin_list.append(np.zeros(1, dtype=np.int64))
            gs_lengths_list.append(1)

    # Pad GS sequences to same length
    n_recent = len(gs_periods_list)
    if n_recent > 0:
        max_gs_rows = max(len(p) for p in gs_periods_list)
        gs_periods = np.zeros((n_recent, max_gs_rows), dtype=np.int64)
        gs_clock = np.zeros((n_recent, max_gs_rows), dtype=np.int64)
        gs_home_score = np.zeros((n_recent, max_gs_rows), dtype=np.int64)
        gs_away_score = np.zeros((n_recent, max_gs_rows), dtype=np.int64)
        gs_margin = np.zeros((n_recent, max_gs_rows), dtype=np.int64)

        for j in range(n_recent):
            length = len(gs_periods_list[j])
            gs_periods[j, :length] = gs_periods_list[j]
            gs_clock[j, :length] = gs_clock_list[j]
            gs_home_score[j, :length] = gs_home_score_list[j]
            gs_away_score[j, :length] = gs_away_score_list[j]
            gs_margin[j, :length] = gs_margin_list[j]

        gs_lengths = np.array(gs_lengths_list, dtype=np.int64)
    else:
        gs_periods = np.zeros((1, 1), dtype=np.int64)
        gs_clock = np.zeros((1, 1), dtype=np.int64)
        gs_home_score = np.zeros((1, 1), dtype=np.int64)
        gs_away_score = np.zeros((1, 1), dtype=np.int64)
        gs_margin = np.zeros((1, 1), dtype=np.int64)
        gs_lengths = np.array([1], dtype=np.int64)

    result = {
        "scores": scores,
        "opponent_ids": opponent_ids,
        "location": location,
        "player_ids": player_ids,
        "player_points": player_points,
        "player_mask": player_mask,
        "days_before": days_before_arr,
        "is_recent": is_recent,
        "gs_periods": gs_periods,
        "gs_clock_buckets": gs_clock,
        "gs_home_score_buckets": gs_home_score,
        "gs_away_score_buckets": gs_away_score,
        "gs_margin_buckets": gs_margin,
        "gs_lengths": gs_lengths,
    }

    if use_stats:
        result["player_stats"] = player_stats_arr
        result["player_positions"] = player_positions
        result["player_pm_available"] = player_pm_available

    if use_efficiency:
        result["efficiency_features"] = efficiency_features
        result["gs_summary_features"] = gs_summary_features
        result["context_flags"] = context_flags

    # Player form tensors: per-roster-player appearance history
    if enable_player_form and roster_ids is not None:
        R = len(roster_ids)
        A = max_appearances

        form_points = np.zeros((R, A), dtype=np.float32)
        form_days = np.zeros((R, A), dtype=np.int64)
        form_mask = np.ones((R, A), dtype=bool)  # True=padding

        if use_stats:
            form_stats = np.zeros((R, A, n_player_stats), dtype=np.float32)
            form_pm_available = np.zeros((R, A), dtype=np.float32)

        for r, rid in enumerate(roster_ids):
            if rid == 0:
                continue  # Padding player
            # The player's most recent A appearances, kept in chronological order
            appearances = [
                (g, p)
                for g in range(n_games)
                for p in range(max_players)
                if player_ids[g, p] == rid
            ][-A:]
            for app_idx, (g, p) in enumerate(appearances):
                form_points[r, app_idx] = player_points[g, p]
                form_days[r, app_idx] = days_before_arr[g]
                form_mask[r, app_idx] = False
                if use_stats:
                    form_stats[r, app_idx] = player_stats_arr[g, p]
                    form_pm_available[r, app_idx] = player_pm_available[g, p]

        result["roster_form_points"] = form_points
        result["roster_form_days"] = form_days
        result["roster_form_mask"] = form_mask
        if use_stats:
            result["roster_form_stats"] = form_stats
            result["roster_form_pm_available"] = form_pm_available

    return result


def _build_h2h_records(game_features: dict[str, PerGameFeatures]) -> list[tuple]:
    """Build sorted list of (date_str, home_idx, away_idx, margin) for H2H lookups."""
    records = []
    for gf in game_features.values():
        home_idx = TEAM_TO_IDX.get(gf.home_team, -1)
        away_idx = TEAM_TO_IDX.get(gf.away_team, -1)
        if home_idx < 0 or away_idx < 0:
            continue
        records.append(
            (gf.game_date, home_idx, away_idx, gf.home_score - gf.away_score)
        )
    records.sort(key=lambda x: x[0])
    return records


def _compute_h2h_features(h2h_records: list[tuple], target_date: str) -> np.ndarray:
    """
    Compute (30, 30, 3) H2H feature matrix from games strictly before target_date.

    Edge features per directed pair (i, j):
      [0] win_rate: fraction of times i beat j, centered at 0.5 -> [-0.5, 0.5]
      [1] avg_margin: mean margin when i was home vs j (normalized by 13)
      [2] n_meetings: min(total_meetings, 20) / 20
    """
    n_teams = 30
    h2h = np.zeros((n_teams, n_teams, 3), dtype=np.float32)

    # Accumulate stats
    wins = np.zeros((n_teams, n_teams), dtype=np.float32)
    games = np.zeros((n_teams, n_teams), dtype=np.float32)
    margin_sum = np.zeros((n_teams, n_teams), dtype=np.float32)

    # Binary search for cutoff
    import bisect

    cutoff = bisect.bisect_left([r[0] for r in h2h_records], target_date)

    for i in range(cutoff):
        _, home_idx, away_idx, margin = h2h_records[i]

        # Home perspective: home vs away
        games[home_idx, away_idx] += 1
        margin_sum[home_idx, away_idx] += margin
        if margin > 0:
            wins[home_idx, away_idx] += 1

        # Away perspective: away vs home
        games[away_idx, home_idx] += 1
        margin_sum[away_idx, home_idx] += -margin
        if margin < 0:
            wins[away_idx, home_idx] += 1

    # Compute features
    mask = games > 0
    h2h[mask, 0] = wins[mask] / games[mask] - 0.5  # win rate centered at 0
    h2h[mask, 1] = margin_sum[mask] / games[mask] / MARGIN_STD  # normalized avg margin
    h2h[:, :, 2] = np.minimum(games, 20) / 20.0  # capped meeting count

    return h2h


class Phase2Dataset(Dataset):
    """
    Phase 2 dataset with full-season context and per-player contributions.

    Returns tensor dicts with home_ and away_ prefixed keys.
    """

    def __init__(
        self,
        game_features: dict[str, PerGameFeatures],
        season_index: dict,
        gs_cache: dict,
        seasons: list[str],
        player_id_map: dict[int, int] | None = None,
        n_recent_games: int = 5,
        max_context_games: int = 82,
        min_history_games: int = 3,
        max_players_per_game: int = 15,
        max_roster_size: int = 15,
        enable_augmentation: bool = False,
        enable_player_form: bool = False,
        max_player_appearances: int = 40,
        n_player_stats: int = 0,
        enable_team_gat: bool = False,
        n_efficiency_features: int = 0,
        player_experience: Optional[dict] = None,
        aux_features: Optional[dict[str, list[float]]] = None,
        projected_rosters: Optional[dict[str, tuple[list[int], list[int]]]] = None,
    ):
        self.game_features = game_features
        self.gs_cache = gs_cache
        self.player_id_map = player_id_map or {}
        self.max_players = max_players_per_game
        self.max_roster = max_roster_size
        self.enable_augmentation = enable_augmentation
        self.enable_player_form = enable_player_form
        self.max_player_appearances = max_player_appearances
        self.n_player_stats = n_player_stats
        self.enable_team_gat = enable_team_gat
        self.n_efficiency_features = n_efficiency_features
        self.player_experience = player_experience or {}
        self.aux_features = aux_features  # game_id -> [N floats]

        # Build H2H records for GAT if enabled
        self.h2h_records = None
        if enable_team_gat:
            self.h2h_records = _build_h2h_records(game_features)
            logger.info(
                f"Built H2H records: {len(self.h2h_records)} game records for GAT"
            )

        self.builder = Phase2SequenceBuilder(
            game_features=game_features,
            season_index=season_index,
            n_recent_games=n_recent_games,
            max_context_games=max_context_games,
            projected_rosters=projected_rosters,
        )

        self.game_ids = self.builder.get_training_game_ids(
            seasons=seasons,
            min_history=min_history_games,
        )
        if projected_rosters is not None:
            # Games the live path couldn't project a roster for aren't trained on
            self.game_ids = [g for g in self.game_ids if g in projected_rosters]

        logger.info(f"Phase2Dataset: {len(self.game_ids)} games for seasons {seasons}")

    def __len__(self) -> int:
        return len(self.game_ids)

    def __getitem__(self, idx: int) -> Optional[dict]:
        game_id = self.game_ids[idx]
        sample = self.builder.build_sample(game_id)

        if sample is None:
            return self.__getitem__((idx + 1) % len(self.game_ids))

        # Build roster arrays first (needed for form tensors)
        home_roster = np.zeros(self.max_roster, dtype=np.int64)
        away_roster = np.zeros(self.max_roster, dtype=np.int64)
        for i, pid in enumerate(sample.home_roster[: self.max_roster]):
            home_roster[i] = self.player_id_map.get(pid, 0)
        for i, pid in enumerate(sample.away_roster[: self.max_roster]):
            away_roster[i] = self.player_id_map.get(pid, 0)

        # Determine season start year for player experience
        target_gf = self.game_features[game_id]
        season_start_year = (
            int(target_gf.season.split("-")[0]) if target_gf.season else None
        )

        # Encode both teams
        common_kwargs = dict(
            gs_cache=self.gs_cache,
            player_id_map=self.player_id_map,
            max_players=self.max_players,
            max_roster=self.max_roster,
            max_appearances=self.max_player_appearances,
            enable_player_form=self.enable_player_form,
            n_player_stats=self.n_player_stats,
            n_efficiency_features=self.n_efficiency_features,
            player_experience=self.player_experience,
            season_start_year=season_start_year,
        )
        home_data = _encode_team_context(
            sample.home_context,
            sample.home_team,
            sample.home_days_before,
            sample.home_recent_indices,
            roster_ids=home_roster,
            **common_kwargs,
        )
        away_data = _encode_team_context(
            sample.away_context,
            sample.away_team,
            sample.away_days_before,
            sample.away_recent_indices,
            roster_ids=away_roster,
            **common_kwargs,
        )

        # Build result dict
        result = {}

        # Add prefixed team data
        for prefix, data in [("home_", home_data), ("away_", away_data)]:
            for key, value in data.items():
                result[prefix + key] = torch.from_numpy(value)

        # Target game features
        result["home_roster"] = torch.from_numpy(home_roster)
        result["away_roster"] = torch.from_numpy(away_roster)
        result["home_rest_days"] = torch.tensor(
            min(sample.home_rest_days, 29), dtype=torch.int64
        )
        result["away_rest_days"] = torch.tensor(
            min(sample.away_rest_days, 29), dtype=torch.int64
        )

        # Labels
        result["target_home_score"] = torch.tensor(
            sample.target_home_score, dtype=torch.float32
        )
        result["target_away_score"] = torch.tensor(
            sample.target_away_score, dtype=torch.float32
        )

        # H2H features for Team GAT
        if self.enable_team_gat and self.h2h_records is not None:
            target_date = self.game_features[game_id].game_date
            h2h = _compute_h2h_features(self.h2h_records, target_date)
            result["h2h_features"] = torch.from_numpy(h2h)  # (30, 30, 3)

        # Auxiliary matchup features (Phase 6 Exp 1+)
        if self.aux_features is not None:
            feats = self.aux_features.get(game_id)
            if feats is not None:
                result["aux_features"] = torch.tensor(feats, dtype=torch.float32)
            else:
                n = len(next(iter(self.aux_features.values())))
                result["aux_features"] = torch.zeros(n, dtype=torch.float32)

        # Home/away augmentation: 50% chance of swapping
        if self.enable_augmentation and random.random() < 0.5:
            result = self._swap_home_away(result)

        return result

    @staticmethod
    def _swap_home_away(result: dict) -> dict:
        """Swap home and away teams (labels negated)."""
        swapped = {}

        for key, value in result.items():
            if key.startswith("home_"):
                new_key = "away_" + key[5:]
                swapped[new_key] = value
            elif key.startswith("away_"):
                new_key = "home_" + key[5:]
                swapped[new_key] = value
            elif key == "target_home_score":
                swapped["target_away_score"] = value
            elif key == "target_away_score":
                swapped["target_home_score"] = value
            else:
                swapped[key] = value

        # Swap aux features: home/away pairs and negate directional features
        if "aux_features" in swapped:
            af = swapped["aux_features"].clone()
            # Swap home/away pairs: (0,1), (2,3), (4,5)
            af[0], af[1] = swapped["aux_features"][1], swapped["aux_features"][0]
            af[2], af[3] = swapped["aux_features"][3], swapped["aux_features"][2]
            af[4], af[5] = swapped["aux_features"][5], swapped["aux_features"][4]
            # Negate directional: rest_diff[7], vegas_spread[8], vegas_ml_prob[10]
            af[7] = -af[7]
            af[8] = -af[8]
            af[10] = -af[10]
            swapped["aux_features"] = af

        return swapped


def collate_phase2(batch: list[Optional[dict]]) -> Optional[dict]:
    """
    Collate Phase 2 samples, padding variable-length dimensions.

    Pads: n_games (max across batch), max_players (max across batch),
          max_gs_rows (max across batch).
    Adds game_mask (bool, True=padding) per team.
    """
    # Filter None samples
    batch = [s for s in batch if s is not None]
    if not batch:
        return None

    B = len(batch)

    # Check if optional feature tensors are present
    has_form = "home_roster_form_points" in batch[0]
    has_stats = "home_player_stats" in batch[0]
    has_efficiency = "home_efficiency_features" in batch[0]

    # Determine max dimensions
    max_home_games = max(s["home_scores"].shape[0] for s in batch)
    max_away_games = max(s["away_scores"].shape[0] for s in batch)
    max_home_players = max(s["home_player_ids"].shape[1] for s in batch)
    max_away_players = max(s["away_player_ids"].shape[1] for s in batch)
    max_home_gs_recent = max(s["home_gs_periods"].shape[0] for s in batch)
    max_away_gs_recent = max(s["away_gs_periods"].shape[0] for s in batch)
    max_home_gs_rows = max(s["home_gs_periods"].shape[1] for s in batch)
    max_away_gs_rows = max(s["away_gs_periods"].shape[1] for s in batch)

    result = {}

    # Pad per-team sequences
    for prefix, max_games, max_players, max_gs_recent, max_gs_rows in [
        (
            "home_",
            max_home_games,
            max_home_players,
            max_home_gs_recent,
            max_home_gs_rows,
        ),
        (
            "away_",
            max_away_games,
            max_away_players,
            max_away_gs_recent,
            max_away_gs_rows,
        ),
    ]:
        # Game-level features
        scores = torch.zeros(B, max_games, 4)
        opponent_ids = torch.zeros(B, max_games, dtype=torch.int64)
        location = torch.zeros(B, max_games, dtype=torch.int64)
        days_before = torch.zeros(B, max_games, dtype=torch.int64)
        is_recent = torch.zeros(B, max_games, dtype=torch.bool)
        game_mask = torch.ones(B, max_games, dtype=torch.bool)  # True=padding

        # Player-level features
        player_ids = torch.zeros(B, max_games, max_players, dtype=torch.int64)
        player_points = torch.zeros(B, max_games, max_players)
        player_mask = torch.ones(B, max_games, max_players, dtype=torch.bool)

        # Player stats features (only when present)
        if has_stats:
            n_stats = batch[0][prefix + "player_stats"].shape[-1]
            player_stats_t = torch.zeros(B, max_games, max_players, n_stats)
            player_positions_t = torch.full(
                (B, max_games, max_players), 3, dtype=torch.int64
            )
            player_pm_available_t = torch.zeros(B, max_games, max_players)

        # Efficiency features (only when present)
        if has_efficiency:
            n_eff = batch[0][prefix + "efficiency_features"].shape[-1]
            efficiency_t = torch.zeros(B, max_games, n_eff)
            gs_summary_t = torch.zeros(B, max_games, 6)
            context_flags_t = torch.zeros(B, max_games, 2)

        # GameStates features
        gs_periods = torch.zeros(B, max_gs_recent, max_gs_rows, dtype=torch.int64)
        gs_clock = torch.zeros(B, max_gs_recent, max_gs_rows, dtype=torch.int64)
        gs_home_score = torch.zeros(B, max_gs_recent, max_gs_rows, dtype=torch.int64)
        gs_away_score = torch.zeros(B, max_gs_recent, max_gs_rows, dtype=torch.int64)
        gs_margin = torch.zeros(B, max_gs_recent, max_gs_rows, dtype=torch.int64)
        gs_lengths = torch.zeros(B, max_gs_recent, dtype=torch.int64)

        for i, s in enumerate(batch):
            ng = s[prefix + "scores"].shape[0]
            np_ = s[prefix + "player_ids"].shape[1]
            nr = s[prefix + "gs_periods"].shape[0]
            ngr = s[prefix + "gs_periods"].shape[1]

            scores[i, :ng] = s[prefix + "scores"]
            opponent_ids[i, :ng] = s[prefix + "opponent_ids"]
            location[i, :ng] = s[prefix + "location"]
            days_before[i, :ng] = s[prefix + "days_before"]
            is_recent[i, :ng] = s[prefix + "is_recent"]
            game_mask[i, :ng] = False

            player_ids[i, :ng, :np_] = s[prefix + "player_ids"]
            player_points[i, :ng, :np_] = s[prefix + "player_points"]
            player_mask[i, :ng, :np_] = s[prefix + "player_mask"]

            if has_stats:
                player_stats_t[i, :ng, :np_] = s[prefix + "player_stats"]
                player_positions_t[i, :ng, :np_] = s[prefix + "player_positions"]
                player_pm_available_t[i, :ng, :np_] = s[prefix + "player_pm_available"]

            if has_efficiency:
                efficiency_t[i, :ng] = s[prefix + "efficiency_features"]
                gs_summary_t[i, :ng] = s[prefix + "gs_summary_features"]
                context_flags_t[i, :ng] = s[prefix + "context_flags"]

            gs_periods[i, :nr, :ngr] = s[prefix + "gs_periods"]
            gs_clock[i, :nr, :ngr] = s[prefix + "gs_clock_buckets"]
            gs_home_score[i, :nr, :ngr] = s[prefix + "gs_home_score_buckets"]
            gs_away_score[i, :nr, :ngr] = s[prefix + "gs_away_score_buckets"]
            gs_margin[i, :nr, :ngr] = s[prefix + "gs_margin_buckets"]
            gs_lengths[i, :nr] = s[prefix + "gs_lengths"]

        result[prefix + "scores"] = scores
        result[prefix + "opponent_ids"] = opponent_ids
        result[prefix + "location"] = location
        result[prefix + "days_before"] = days_before
        result[prefix + "is_recent"] = is_recent
        result[prefix + "game_mask"] = game_mask
        result[prefix + "player_ids"] = player_ids
        result[prefix + "player_points"] = player_points
        result[prefix + "player_mask"] = player_mask
        result[prefix + "gs_periods"] = gs_periods
        result[prefix + "gs_clock_buckets"] = gs_clock
        result[prefix + "gs_home_score_buckets"] = gs_home_score
        result[prefix + "gs_away_score_buckets"] = gs_away_score
        result[prefix + "gs_margin_buckets"] = gs_margin
        result[prefix + "gs_lengths"] = gs_lengths

        if has_stats:
            result[prefix + "player_stats"] = player_stats_t
            result[prefix + "player_positions"] = player_positions_t
            result[prefix + "player_pm_available"] = player_pm_available_t

        if has_efficiency:
            result[prefix + "efficiency_features"] = efficiency_t
            result[prefix + "gs_summary_features"] = gs_summary_t
            result[prefix + "context_flags"] = context_flags_t

    # Player form tensors (fixed shape R×A, just stack)
    if has_form:
        has_form_stats = "home_roster_form_stats" in batch[0]
        for prefix in ["home_", "away_"]:
            result[prefix + "roster_form_points"] = torch.stack(
                [s[prefix + "roster_form_points"] for s in batch]
            )
            result[prefix + "roster_form_days"] = torch.stack(
                [s[prefix + "roster_form_days"] for s in batch]
            )
            result[prefix + "roster_form_mask"] = torch.stack(
                [s[prefix + "roster_form_mask"] for s in batch]
            )
            if has_form_stats:
                result[prefix + "roster_form_stats"] = torch.stack(
                    [s[prefix + "roster_form_stats"] for s in batch]
                )
                result[prefix + "roster_form_pm_available"] = torch.stack(
                    [s[prefix + "roster_form_pm_available"] for s in batch]
                )

    # Target game features (no padding needed)
    result["home_roster"] = torch.stack([s["home_roster"] for s in batch])
    result["away_roster"] = torch.stack([s["away_roster"] for s in batch])
    result["home_rest_days"] = torch.stack([s["home_rest_days"] for s in batch])
    result["away_rest_days"] = torch.stack([s["away_rest_days"] for s in batch])
    result["target_home_scores"] = torch.stack([s["target_home_score"] for s in batch])
    result["target_away_scores"] = torch.stack([s["target_away_score"] for s in batch])

    # H2H features for Team GAT (fixed 30×30×3, just stack)
    if "h2h_features" in batch[0]:
        result["h2h_features"] = torch.stack([s["h2h_features"] for s in batch])

    # Auxiliary matchup features (fixed size, just stack)
    if "aux_features" in batch[0]:
        result["aux_features"] = torch.stack([s["aux_features"] for s in batch])

    return result
