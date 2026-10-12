"""
Sequence Builder for NBA Transformer Model.

This module constructs the INPUT data for each training sample. The core idea:
    - For a game we want to predict (the "target game"), we look backwards in time
      and fetch the last N completed games for BOTH the home and away teams.
    - This creates a "historical context window" -- the transformer model learns to
      predict game outcomes based on how each team has recently played.

CRITICAL: Strict temporal ordering is enforced throughout. We ONLY look at games
that occurred BEFORE the target game's date. This prevents "data leakage" -- a
common machine learning pitfall where future information accidentally leaks into
the training data, making the model appear better than it really is.

Usage:
    builder = SequenceBuilder(tokenizer, n_games=5)
    sequence_data = builder.build_sequence(target_game_id)
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from src.database import get_db
from src.transformer.tokenizer import PBPTokenizer, TokenizedGame
from src.utils import to_eastern_date

# Safety cap for play truncation. The longest regulation game in our dataset
# is 701 plays (MEM @ GSW, Nov 2024). 750 provides margin for future data.
MAX_REGULATION_PLAYS = 750


# ---------------------------------------------------------------------------
# Data classes: These are simple containers that hold the structured data
# flowing through the pipeline. Think of them as labeled boxes.
# ---------------------------------------------------------------------------

@dataclass
class TeamHistory:
    """
    Historical game data for a single team.

    This holds the tokenized play-by-play data from a team's recent games.
    "Tokenized" means the raw play descriptions (e.g., "LeBron James makes
    3-pointer") have been converted into numeric IDs that a neural network
    can process -- similar to how language models convert words to numbers.
    """

    team: str  # Team tricode, e.g. "BOS" for Boston Celtics
    games: list[TokenizedGame]  # List of past games, ordered most recent first
    game_ids: list[str]  # Corresponding game IDs (parallel list with 'games')
    game_dates: list[str] = field(default_factory=list)  # ISO dates for schedule features


@dataclass
class MatchupSequence:
    """
    Complete sequence data for one prediction sample (one matchup).

    This is the full package needed for a single training example:
      - Both teams' recent game histories (the MODEL INPUT)
      - The actual final scores (the LABELS/TARGETS the model tries to predict)

    During training, the model sees the histories and tries to predict the scores.
    During inference (predicting future games), scores will be None.
    """

    target_game_id: str  # The game we want to predict
    target_date: str  # Date of the target game (ISO format, e.g. "2024-01-15")
    home_team: str  # Home team tricode
    away_team: str  # Away team tricode
    home_history: TeamHistory  # Home team's last N games (the input context)
    away_history: TeamHistory  # Away team's last N games (the input context)
    home_score: Optional[int]  # Training target (regulation-end when filtering OT, else final)
    away_score: Optional[int]  # Training target (regulation-end when filtering OT, else final)
    final_home_score: Optional[int] = None  # Always the true final score (including OT)
    final_away_score: Optional[int] = None  # Always the true final score (including OT)
    is_overtime: bool = False               # Whether this game went to overtime
    home_season_games: int = 0              # Total games home team played before target (Phase 1c)
    away_season_games: int = 0              # Total games away team played before target (Phase 1c)
    season: str = ""                        # Season string (e.g., "2024-2025") for game count query
    target_home_roster: list[int] = field(default_factory=list)  # Player IDs from target game (Phase 1b)
    target_away_roster: list[int] = field(default_factory=list)  # Player IDs from target game (Phase 1b)


# ---------------------------------------------------------------------------
# Main builder class
# ---------------------------------------------------------------------------

class SequenceBuilder:
    """
    Builds historical game sequences for transformer input.

    For each target game, fetches the last N completed games for both teams,
    ensuring strict temporal ordering (no future data leakage).

    Example: If we want to predict BOS vs LAL on Jan 15, and n_games=5,
    the builder will fetch BOS's 5 most recent games before Jan 15 and
    LAL's 5 most recent games before Jan 15.
    """

    def __init__(
        self,
        tokenizer: PBPTokenizer,
        n_games: int = 5,
        db_path: Optional[str] = None,
    ):
        """
        Initialize sequence builder.

        Args:
            tokenizer: Initialized PBPTokenizer
            n_games: Number of historical games to fetch per team
            db_path: Path to database. If None, uses config default.
        """
        self.tokenizer = tokenizer
        self.n_games = n_games

        if db_path is None:
            from src.config import config

            db_path = config["database"]["path"]

        self.db_path = db_path

        # Cache for tokenized games to avoid re-tokenizing the same game.
        # The same game often appears in multiple teams' histories (e.g., a BOS
        # vs LAL game appears in BOTH teams' histories). This LRU-style dict
        # cache stores the tokenized result so we only do the expensive
        # tokenization once per game, regardless of how many times it's needed.
        self._game_cache: dict[str, TokenizedGame] = {}

    def _get_prior_games(
        self, team: str, before_date: str, n_games: int
    ) -> list[tuple[str, str]]:
        """
        Get game IDs and dates for the last N games for a team before a given date.

        This is where TEMPORAL ORDERING is enforced: the query uses
        "date(date_time_utc, '-5 hours') < date(?)" (the Eastern date: every NBA
        tip-off, EST or EDT, lands on its Eastern day after -5 hours) to ensure
        we ONLY retrieve games that happened BEFORE the target game. This
        prevents data leakage.

        Args:
            team: Team tricode (e.g., "BOS")
            before_date: ISO date string (e.g., "2024-01-15")
            n_games: Number of games to fetch

        Returns:
            List of (game_id, date_string) tuples, most recent first
        """
        with get_db(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT game_id, date_time_utc
                FROM Games
                WHERE (home_team = ? OR away_team = ?)
                AND date(date_time_utc, '-5 hours') < date(?)    -- STRICT: only games BEFORE the target date
                AND game_data_finalized = 1           -- Only fully recorded games
                AND status = 3                        -- status=3 means the game is completed
                AND season_type IN ('Regular Season', 'Post Season')
                ORDER BY date_time_utc DESC           -- Most recent games first
                LIMIT ?                               -- Only take the last N games
                """,
                (team, team, before_date, n_games),
            )
            rows = cursor.fetchall()

        # Return (game_id, date) tuples — dates used for schedule features (Phase 1c)
        return [(row[0], to_eastern_date(row[1])) for row in rows]

    def _get_game_with_cache(self, game_id: str) -> Optional[TokenizedGame]:
        """
        Get tokenized game, using cache if available.

        Tokenizing a game (fetching play-by-play from DB, converting each play
        to numeric tokens) is expensive. Since the same game can appear in
        multiple teams' history windows, caching avoids redundant work.
        For example, game "BOS_vs_LAL_20240115" would appear in both BOS's
        and LAL's histories -- we only tokenize it once.
        """
        if game_id not in self._game_cache:
            # Cache miss: tokenize from the database and store
            tokenized = self.tokenizer.tokenize_game(game_id
            )
            if tokenized:
                self._game_cache[game_id] = tokenized
            else:
                return None

        # Cache hit (or just-cached): return the stored result
        return self._game_cache[game_id]

    def _get_team_history(
        self, team: str, before_date: str
    ) -> TeamHistory:
        """
        Get historical game sequence for a team.

        Combines the two steps: (1) find the right game IDs, (2) tokenize each one.
        Some games may fail to tokenize (e.g., missing play-by-play data), so we
        track which ones succeeded in valid_game_ids.

        Args:
            team: Team tricode
            before_date: ISO date string

        Returns:
            TeamHistory with tokenized games, ordered most recent first
            (games[0] is the most recent, games[-1] is the oldest)
        """
        # Step 1: Find the last N game IDs for this team before the target date
        # IMPORTANT: _get_prior_games() returns games in DESC order (most recent first).
        # This ordering is relied upon by sequence_to_arrays() for roster extraction.
        prior_results = self._get_prior_games(team, before_date, self.n_games)

        # Step 2: Tokenize each game (convert play-by-play text to numeric tokens)
        games = []
        valid_game_ids = []
        valid_game_dates = []

        for game_id, game_date in prior_results:
            tokenized = self._get_game_with_cache(game_id)
            if tokenized:
                games.append(tokenized)
                valid_game_ids.append(game_id)
                valid_game_dates.append(game_date)
            # If tokenization fails, we silently skip that game

        return TeamHistory(team=team, games=games, game_ids=valid_game_ids,
                           game_dates=valid_game_dates)

    def build_sequence(self, target_game_id: str) -> Optional[MatchupSequence]:
        """
        Build complete sequence data for a target game prediction.

        This is the main entry point. Given a game we want to predict, it:
          1. Looks up the target game's metadata (teams, date, final score)
          2. Fetches historical games for both teams (looking BACKWARDS in time)
          3. Packages everything into a MatchupSequence

        Args:
            target_game_id: Game ID to predict

        Returns:
            MatchupSequence with all historical data, or None if target not found
        """
        # Step 1: Fetch the target game's metadata and both score types.
        # - Final scores: from is_final_state=1 (includes OT if played)
        # - Regulation scores: from the last play in period <= 4
        # - OT detection: whether any play exists in period > 4
        with get_db(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT home_team, away_team, date_time_utc, status, season,
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

        (home_team, away_team, date_time_utc, status, season,
         final_home_score, final_away_score,
         reg_home_score, reg_away_score, is_overtime) = row

        # Eastern calendar date of the tip-off (evening games fall on the next
        # UTC day), for temporal filtering
        target_date = to_eastern_date(date_time_utc)

        # Training targets: use regulation scores for OT games (cleaner signal),
        # final scores for regulation games (they're identical anyway).
        if is_overtime:
            train_home = reg_home_score
            train_away = reg_away_score
        else:
            train_home = final_home_score
            train_away = final_away_score

        # Step 2: Get historical sequences for both teams.
        # Each call looks backwards from target_date to find the team's last N games.
        home_history = self._get_team_history(home_team, target_date)
        away_history = self._get_team_history(away_team, target_date)

        # Step 2b: Count total games each team has played this season before
        # the target date. Used for season_game_number schedule features (Phase 1c).
        home_season_games = 0
        away_season_games = 0
        if season:
            with get_db(self.db_path) as conn:
                for team, attr in [(home_team, "home"), (away_team, "away")]:
                    cursor = conn.execute(
                        """
                        SELECT COUNT(*) FROM Games
                        WHERE (home_team = ? OR away_team = ?)
                        AND date(date_time_utc, '-5 hours') < date(?)
                        AND season = ?
                        AND status = 3
                        AND season_type IN ('Regular Season', 'Post Season')
                        """,
                        (team, team, target_date, season),
                    )
                    count = cursor.fetchone()[0]
                    if attr == "home":
                        home_season_games = count
                    else:
                        away_season_games = count

        # Step 2c: Extract target game roster from GameStates.players_data (Phase 1b).
        # Uses the actual players who participated in the target game (ground truth).
        # For inference on future games, this would need to be approximated from
        # injury reports, projected lineups, etc.
        target_home_roster = []
        target_away_roster = []
        if status == 3:  # Only for completed games
            with get_db(self.db_path) as conn:
                cursor = conn.execute(
                    """
                    SELECT players_data FROM GameStates
                    WHERE game_id = ? AND is_final_state = 1
                    """,
                    (target_game_id,),
                )
                row = cursor.fetchone()
                if row and row[0]:
                    try:
                        players_data = json.loads(row[0])
                        if "home" in players_data:
                            target_home_roster = [int(pid) for pid in players_data["home"].keys()]
                        if "away" in players_data:
                            target_away_roster = [int(pid) for pid in players_data["away"].keys()]
                    except (json.JSONDecodeError, ValueError, TypeError):
                        logging.debug(f"Failed to parse players_data for {target_game_id}")

        # Step 3: Package everything into a MatchupSequence.
        # Scores are only included if the game is completed (status == 3).
        # For future/unplayed games, scores will be None.
        return MatchupSequence(
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
            home_season_games=home_season_games,
            away_season_games=away_season_games,
            season=season or "",
            target_home_roster=target_home_roster,
            target_away_roster=target_away_roster,
        )

    def build_sequences_batch(
        self, game_ids: list[str], show_progress: bool = True
    ) -> list[MatchupSequence]:
        """
        Build sequences for multiple games (convenience wrapper).

        Iterates over a list of game IDs, calling build_sequence() on each one.
        Failed builds (returning None) are silently skipped.

        Args:
            game_ids: List of game IDs to process
            show_progress: Whether to show a tqdm progress bar in the terminal

        Returns:
            List of successfully built MatchupSequence objects
        """
        from tqdm import tqdm

        sequences = []
        iterator = tqdm(game_ids, desc="Building sequences", leave=False) if show_progress else game_ids

        for game_id in iterator:
            seq = self.build_sequence(game_id)
            if seq:
                sequences.append(seq)

        return sequences

    def get_training_game_ids(
        self,
        seasons: list[str],
        min_history_games: int = 3,
    ) -> list[str]:
        """
        Get all valid game IDs for training from specified seasons.

        Not every game is usable for training. Early-season games are excluded
        because teams haven't played enough prior games to form a meaningful
        historical context. For example, if min_history_games=3, then a team's
        first 3 games of the season are skipped because we can't build a full
        history window for them.

        A game is valid for training if:
        1. It's completed (status = 3)
        2. game_data_finalized = 1
        3. BOTH teams have at least min_history_games prior games in the data

        Args:
            seasons: List of seasons (e.g., ["2023-2024"])
            min_history_games: Minimum number of prior games required

        Returns:
            List of game IDs suitable for training, in chronological order
        """
        with get_db(self.db_path) as conn:
            # Build SQL placeholders for the IN clause: "?,?,?" for N seasons
            placeholders = ",".join(["?"] * len(seasons))

            # Fetch ALL completed games in chronological order.
            # We process them sequentially below to build up game counts.
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

        # Walk through games chronologically, keeping a running tally of how
        # many games each team has played. This dict acts as a counter:
        # {"BOS": 15, "LAL": 12, ...} meaning BOS has played 15 games so far.
        valid_game_ids = []
        team_game_counts: dict[str, int] = {}

        for game_id, home_team, away_team, date_time_utc in all_games:
            home_count = team_game_counts.get(home_team, 0)
            away_count = team_game_counts.get(away_team, 0)

            # Only include this game if BOTH teams have played enough prior games.
            # This ensures we can build a full history window for both sides.
            # e.g., if min_history_games=3 and BOS has played 5 but LAL only 2,
            # we skip this game because LAL doesn't have enough history yet.
            if home_count >= min_history_games and away_count >= min_history_games:
                valid_game_ids.append(game_id)

            # IMPORTANT: We always increment counts regardless of whether the game
            # was valid. The count tracks total games played, not just valid ones.
            # This game still contributes to the team's history for future games.
            team_game_counts[home_team] = home_count + 1
            team_game_counts[away_team] = away_count + 1

        return valid_game_ids

    def clear_cache(self):
        """Clear the game cache to free memory."""
        self._game_cache.clear()

    @property
    def cache_size(self) -> int:
        """Number of games currently cached."""
        return len(self._game_cache)


# ---------------------------------------------------------------------------
# Conversion to numpy arrays (bridge between Python objects and PyTorch)
# ---------------------------------------------------------------------------

def _compute_schedule_arrays(
    target_date: str,
    game_dates: list[str],
    season_game_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute schedule/temporal arrays for a team's history games.

    Args:
        target_date: ISO date string for the target game (e.g., "2024-01-15")
        game_dates: Dates of history games, most recent first
        season_game_count: Total games this team played before the target date

    Returns:
        (days_before_target, season_game_number) — both shape (n_games,), int64
    """
    from datetime import date as Date

    target = Date.fromisoformat(target_date)

    # Per-game: days between each history game and the target game
    days_before = []
    for gd in game_dates:
        gap = (target - Date.fromisoformat(gd)).days
        days_before.append(gap)
    days_before = np.clip(days_before, 0, 179).astype(np.int64)

    # Per-game: season_game_count - position_index
    # Most recent history game (i=0) was the team's Nth game,
    # the one before (i=1) was game N-1, etc.
    season_nums = [max(season_game_count - i, 0) for i in range(len(game_dates))]
    season_nums = np.clip(season_nums, 0, 109).astype(np.int64)

    return days_before, season_nums


def sequence_to_arrays(
    sequence: MatchupSequence,
    tokenizer: PBPTokenizer,
) -> dict:
    """
    Convert a MatchupSequence (Python objects) into numpy arrays for PyTorch.

    Neural networks operate on fixed-size numeric tensors, but games have different
    numbers of plays. This function handles that mismatch by PADDING shorter games
    with zeros so all games in a team's history have the same length.

    For example, if a team's 5 recent games had [320, 290, 410, 350, 300] plays,
    all would be padded to 410 (the maximum), with zeros filling the extra slots.
    The model learns to ignore the zero-padded positions.

    Args:
        sequence: MatchupSequence from builder
        tokenizer: PBPTokenizer instance (needed to convert plays to numeric arrays)

    Returns:
        Dict with numpy arrays for home/away history, rosters, and metadata
    """
    def games_to_array(games: list[TokenizedGame]) -> dict:
        """
        Convert a list of TokenizedGame objects into stacked numpy arrays.

        Each game's plays become a 1D array of token IDs. Since games have
        different numbers of plays, shorter games are PADDED with zeros to
        match the longest game. The result is a 2D array: (n_games, max_plays).
        """
        if not games:
            return None

        all_tensors = []
        game_lengths = []  # Track actual (unpadded) length of each game

        for game in games:
            # Truncate very long games to keep memory bounded.
            # NOTE: This is a silent truncation. Longest observed regulation game
            # is 701 plays (MEM @ GSW, Nov 2024); cap is 750 for safety margin.
            # Games exceeding this are likely data errors or extreme outliers.
            plays = game.plays[:MAX_REGULATION_PLAYS]
            if len(game.plays) > MAX_REGULATION_PLAYS:
                logging.warning(
                    f"Game {game.game_id} has {len(game.plays)} plays, "
                    f"truncating to {MAX_REGULATION_PLAYS}"
                )
            # Convert play objects to numpy arrays (one array per feature type,
            # e.g., action_type_ids, player_ids, etc.)
            tensors = tokenizer.plays_to_tensors(plays)
            all_tensors.append(tensors)
            game_lengths.append(len(plays))

        # Stack all tensor types (e.g., action_type_ids, player_ids, etc.)
        # into 2D arrays of shape (n_games, max_plays_across_games)
        result = {}
        for key in all_tensors[0].keys():
            # Find the longest game in this team's history
            max_len = max(game_lengths)
            padded = []
            for tensors, length in zip(all_tensors, game_lengths):
                arr = tensors[key]
                if length < max_len:
                    # PAD shorter games with zeros to match the longest game.
                    # Zeros are ignored by the model (via attention masking).
                    padding = np.zeros(max_len - length, dtype=arr.dtype)
                    arr = np.concatenate([arr, padding])
                padded.append(arr)
            # Stack into shape: (n_games, max_plays)
            # e.g., for 5 history games with max 410 plays: (5, 410)
            result[key] = np.stack(padded)

        # Store the actual lengths so the model knows where real data ends
        # and padding begins. Essential for attention masking.
        result["game_lengths"] = np.array(game_lengths, dtype=np.int64)
        return result

    # Compute schedule/temporal arrays for Phase 1c (days_before_target, season_game_number)
    home_history_arrays = games_to_array(sequence.home_history.games)
    away_history_arrays = games_to_array(sequence.away_history.games)

    if home_history_arrays is not None and sequence.home_history.game_dates:
        days_before, season_nums = _compute_schedule_arrays(
            sequence.target_date, sequence.home_history.game_dates,
            sequence.home_season_games,
        )
        home_history_arrays["days_before_target"] = days_before
        home_history_arrays["season_game_number"] = season_nums

    if away_history_arrays is not None and sequence.away_history.game_dates:
        days_before, season_nums = _compute_schedule_arrays(
            sequence.target_date, sequence.away_history.game_dates,
            sequence.away_season_games,
        )
        away_history_arrays["days_before_target"] = days_before
        away_history_arrays["season_game_number"] = season_nums

    # Compute rest_days for each team (days since last game before target).
    # rest_days = days_before_target[0] - 1 because days_before counts the gap
    # from the history game's date to the target date (B2B = gap of 1 = 0 rest days).
    # Clipped to [0, 29] for embedding vocab size of 30.
    home_rest_days = int(np.clip(
        home_history_arrays["days_before_target"][0] - 1, 0, 29
    )) if (
        home_history_arrays is not None
        and "days_before_target" in home_history_arrays
        and len(home_history_arrays["days_before_target"]) > 0
    ) else 1

    away_rest_days = int(np.clip(
        away_history_arrays["days_before_target"][0] - 1, 0, 29
    )) if (
        away_history_arrays is not None
        and "days_before_target" in away_history_arrays
        and len(away_history_arrays["days_before_target"]) > 0
    ) else 1

    # Build the final output dict containing everything for one training sample
    return {
        "target_game_id": sequence.target_game_id,
        "target_date": sequence.target_date,
        "home_team": sequence.home_team,
        "away_team": sequence.away_team,
        # Convert each team's history games into padded numpy arrays
        "home_history": home_history_arrays,
        "away_history": away_history_arrays,
        # Rest days for the target game (Combined Phase 1)
        "home_rest_days": home_rest_days,
        "away_rest_days": away_rest_days,
        # Roster arrays: convert player NBA IDs to numeric vocab indices.
        # Phase 1b: Uses the TARGET GAME's actual roster from GameStates.players_data
        # (ground truth). Falls back to most recent prior game if target roster unavailable.
        # For inference on future games, this would be approximated from injury reports.
        "home_roster": np.array(
            [tokenizer.player_vocab.get(p, 0) for p in sequence.target_home_roster]
            if sequence.target_home_roster
            else (
                [tokenizer.player_vocab.get(p, 0) for p in sequence.home_history.games[0].player_ids_home]
                if sequence.home_history.games else []
            ),
            dtype=np.int64
        ),
        "away_roster": np.array(
            [tokenizer.player_vocab.get(p, 0) for p in sequence.target_away_roster]
            if sequence.target_away_roster
            else (
                [tokenizer.player_vocab.get(p, 0) for p in sequence.away_history.games[0].player_ids_away]
                if sequence.away_history.games else []
            ),
            dtype=np.int64
        ),
        # The labels (targets) for training -- what the model tries to predict.
        # home_score/away_score are regulation scores (training targets).
        # final scores are carried through for evaluation of OT games.
        "home_score": sequence.home_score,
        "away_score": sequence.away_score,
        "final_home_score": sequence.final_home_score,
        "final_away_score": sequence.final_away_score,
        "is_overtime": sequence.is_overtime,
    }


def test_sequence_builder():
    """Test sequence builder functionality."""
    logging.basicConfig(level=logging.INFO)

    # Load or build tokenizer
    tokenizer_path = "data/tokenized/test_tokenizer.json"
    tokenizer = PBPTokenizer()

    try:
        tokenizer.load(tokenizer_path)
        print("Loaded existing tokenizer")
    except FileNotFoundError:
        print("Building tokenizer...")
        tokenizer.build_vocab_from_db(seasons=["2023-2024"], min_count=5)
        tokenizer.save(tokenizer_path)

    # Create sequence builder
    builder = SequenceBuilder(tokenizer, n_games=5)

    # Get valid training games
    print("\nGetting valid training games for 2023-2024...")
    valid_games = builder.get_training_game_ids(
        seasons=["2023-2024"], min_history_games=3
    )
    print(f"Found {len(valid_games)} valid games for training")

    if valid_games:
        # Test on first valid game
        test_game_id = valid_games[0]
        print(f"\nBuilding sequence for game {test_game_id}...")

        sequence = builder.build_sequence(test_game_id)

        if sequence:
            print(f"  Target: {sequence.home_team} vs {sequence.away_team}")
            print(f"  Date: {sequence.target_date}")
            print(f"  Home history: {len(sequence.home_history.games)} games")
            print(f"  Away history: {len(sequence.away_history.games)} games")
            print(f"  Final score: {sequence.home_score} - {sequence.away_score}")

            # Show historical game IDs
            print(f"\n  Home team ({sequence.home_team}) prior games:")
            for i, gid in enumerate(sequence.home_history.game_ids):
                game = sequence.home_history.games[i]
                print(f"    {i+1}. {gid}: {game.home_team} vs {game.away_team}, {game.home_score}-{game.away_score}")

            print(f"\n  Away team ({sequence.away_team}) prior games:")
            for i, gid in enumerate(sequence.away_history.game_ids):
                game = sequence.away_history.games[i]
                print(f"    {i+1}. {gid}: {game.home_team} vs {game.away_team}, {game.home_score}-{game.away_score}")

            # Convert to arrays
            print("\n  Converting to arrays...")
            arrays = sequence_to_arrays(sequence, tokenizer)
            print(f"  Home history shape: {arrays['home_history']['action_type_ids'].shape if arrays['home_history'] else 'None'}")
            print(f"  Away history shape: {arrays['away_history']['action_type_ids'].shape if arrays['away_history'] else 'None'}")
            print(f"  Home roster: {len(arrays['home_roster'])} players")
            print(f"  Away roster: {len(arrays['away_roster'])} players")

        print(f"\n  Cache size: {builder.cache_size} games")

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(test_sequence_builder())
