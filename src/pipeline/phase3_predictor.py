"""
Phase 3 Transformer Predictor: roster-temporal graph model inference pipeline.

Implements the BasePredictor interface for the Phase 3 Exp 5 transformer model
(~42M params). For each game: loads the cache, builds full-season context
sequences for both teams, runs the model forward pass, and returns spread/win/total
predictions.

The model is lazy-loaded on first prediction call to avoid startup cost.

Checkpoint priority:
    1. Exp 5b slim (sigma-capped) if available
    2. Exp 5 (original roster-temporal) as fallback

Usage:
    from src.pipeline.phase3_predictor import Phase3Predictor
    predictor = Phase3Predictor()
    preds = predictor.make_pre_game_predictions(["0022500900"])
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from src.predictions.prediction_engines.base_predictor import BasePredictor

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Checkpoint search order (first existing file wins)
DEFAULT_CHECKPOINT_PATHS = [
    PROJECT_ROOT / "models" / "phase3" / "model.pt",
    PROJECT_ROOT / "models" / "phase3" / "model_fallback.pt",
]

# Config search order (must match selected checkpoint)
DEFAULT_CONFIG_PATHS = [
    PROJECT_ROOT / "models" / "phase3" / "config.yaml",
    PROJECT_ROOT / "models" / "phase3" / "config_fallback.yaml",
]

DEFAULT_CACHE_DIR = PROJECT_ROOT / "data" / "phase3_cache"

# Projected players per team for upcoming games. Training rosters were the
# players who logged minutes in the game itself (~10.9/team), and that count
# tracks the final margin, so the model reads roster size as a margin signal.
# Uncapped projected rosters (~13-15) score no better than predicting the mean
# margin; 11 gave the best-calibrated spreads on 2024-26 games (MAE 11.7 vs
# 12.9). Retraining on pre-game rosters is the real fix.
ROSTER_CAP = 11


class Phase3Predictor(BasePredictor):
    """Phase 3 transformer model predictor (Exp 5 / 5b roster-temporal)."""

    def __init__(self, model_paths: Optional[dict] = None, historical: bool = False):
        """
        Initialize Phase 3 predictor.

        Args:
            model_paths: Optional dict with keys:
                checkpoint: path to model .pt file
                config: path to .yaml config file
                cache_dir: path to phase3 cache directory
            historical: predict past games as of their tip-off (backtests):
                pre-game rosters from appearances, and no cache update
        """
        super().__init__()
        model_paths = model_paths or {}
        self.historical = historical
        self._roster_assembler = None  # created on first use, then reused

        self._checkpoint_path = model_paths.get("checkpoint")
        self._config_path = model_paths.get("config")
        self._cache_dir = Path(model_paths.get("cache_dir", str(DEFAULT_CACHE_DIR)))

        self._model = None
        self._config = None
        self._cache = None
        self._device = None
        self._models_loaded = False

    def _resolve_checkpoint(self) -> tuple[Path, Path]:
        """
        Find the best available checkpoint and matching config.

        Returns:
            (checkpoint_path, config_path)

        Raises:
            FileNotFoundError if no checkpoint can be found.
        """
        # Explicit paths take priority
        if self._checkpoint_path and self._config_path:
            ckpt = Path(self._checkpoint_path)
            cfg = Path(self._config_path)
            if ckpt.exists() and cfg.exists():
                return ckpt, cfg
            if not ckpt.exists():
                raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
            if not cfg.exists():
                raise FileNotFoundError(f"Config not found: {cfg}")

        # Auto-detect: try each pair in priority order
        for ckpt_path, cfg_path in zip(DEFAULT_CHECKPOINT_PATHS, DEFAULT_CONFIG_PATHS):
            if ckpt_path.exists() and cfg_path.exists():
                logger.info("Auto-detected checkpoint: %s", ckpt_path.name)
                return ckpt_path, cfg_path

        # Final fallback: try any checkpoint with any config
        for ckpt_path in DEFAULT_CHECKPOINT_PATHS:
            if ckpt_path.exists():
                for cfg_path in DEFAULT_CONFIG_PATHS:
                    if cfg_path.exists():
                        logger.warning(
                            "Using mismatched checkpoint/config: %s / %s",
                            ckpt_path.name,
                            cfg_path.name,
                        )
                        return ckpt_path, cfg_path

        raise FileNotFoundError(
            "No Phase 3 checkpoint found. Searched: "
            + ", ".join(str(p) for p in DEFAULT_CHECKPOINT_PATHS)
        )

    def _ensure_models_loaded(self):
        """Lazy-load model, config, and cache on first prediction call."""
        if self._models_loaded:
            return

        # Resolve paths
        ckpt_path, cfg_path = self._resolve_checkpoint()

        # Load config
        from src.transformer.phase2.config import Phase2ExperimentConfig

        self._config = Phase2ExperimentConfig.from_yaml(str(cfg_path))
        logger.info("Loaded config from %s", cfg_path)

        # Load cache
        if not self._cache_dir.exists():
            raise FileNotFoundError(
                f"Phase 3 cache not found at {self._cache_dir}. "
                "Run Phase3CacheUpdater.append_new_games() first."
            )

        from src.transformer.phase2.cache_builder import load_cache

        self._cache = load_cache(str(self._cache_dir))
        logger.info(
            "Loaded Phase 3 cache: %d games, %d player mappings",
            len(self._cache["game_features"]),
            len(self._cache["player_id_map"]),
        )

        # Determine device
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Build model
        from src.transformer.phase2.models.phase2_model import Phase2Model

        model_cfg = self._config.model
        # Ensure n_players is large enough for the player mapping
        if self._cache["player_id_map"]:
            max_player_idx = max(self._cache["player_id_map"].values())
            if model_cfg.n_players <= max_player_idx:
                model_cfg.n_players = max_player_idx + 1
                logger.info(
                    "Expanded n_players to %d to fit player mapping",
                    model_cfg.n_players,
                )

        self._model = Phase2Model(model_cfg).to(self._device)

        # Load checkpoint (strict=False for backward compat with renamed keys)
        ckpt = torch.load(str(ckpt_path), map_location=self._device, weights_only=False)
        state_dict = ckpt.get("model_state_dict") or ckpt.get("ema_state_dict") or ckpt
        missing, unexpected = self._model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning("Missing keys in checkpoint: %s", missing[:5])
        if unexpected:
            logger.warning("Unexpected keys in checkpoint: %s", unexpected[:5])
        # Training validated and selected the checkpoint on its EMA weights, so
        # use those. Overlay by parameter so tied embeddings get the EMA values.
        ema = ckpt.get("ema_state_dict") if isinstance(ckpt, dict) else None
        if ema and "shadow" in ema:
            with torch.no_grad():
                for name, param in self._model.named_parameters():
                    if name in ema["shadow"]:
                        param.copy_(ema["shadow"][name])

        self._model.eval()

        n_params = sum(p.numel() for p in self._model.parameters())
        logger.info(
            "Phase 3 model loaded on %s: %s params from %s",
            self._device,
            f"{n_params:,}",
            ckpt_path.name,
        )
        self._models_loaded = True

    def _ensure_cache_fresh(self):
        """Update Phase 3 cache if new games exist in DB but not in cache."""
        try:
            from src.pipeline.phase3_cache_updater import Phase3CacheUpdater

            updater = Phase3CacheUpdater(cache_dir=str(self._cache_dir))
            if not updater.is_fresh():
                n_new = len(updater.find_new_games())
                logger.info(f"Phase 3 cache has {n_new} new games — triggering rebuild")
                updater.append_new_games()
                # Reload cache after rebuild
                self._cache = None
                self._models_loaded = False
        except Exception as e:
            logger.warning(f"Phase 3 cache freshness check failed: {e}")

    def make_pre_game_predictions(self, game_ids: list[str]) -> dict:
        """
        Generate pre-game predictions for the given games.

        Automatically updates the Phase 3 cache if new games are available,
        then builds full-season context sequences for both teams and runs
        the transformer forward pass.

        Returns:
            {game_id: {
                pred_home_score, pred_away_score, pred_home_win_pct,
                pred_spread, pred_spread_sigma, pred_total, pred_total_sigma,
                roster_confidence
            }}
        """
        if not self.historical:
            self._ensure_cache_fresh()
        self._ensure_models_loaded()

        predictions = {}
        for game_id in game_ids:
            try:
                pred = self._predict_single_game(game_id)
                if pred is not None:
                    predictions[game_id] = pred
            except Exception as e:
                logger.warning("Error predicting game %s: %s", game_id, e)

        logger.info(
            "Phase 3 predictions: %d/%d games successful",
            len(predictions),
            len(game_ids),
        )
        return predictions

    def _predict_single_game(self, game_id: str) -> Optional[dict]:
        """
        Build input batch for a single game and run model inference.

        Every game, finished or not, is built as of its tip-off: context from
        prior cached games and the projected roster from RosterAssembler. (The
        cache's own sample for a finished game uses the players who logged
        minutes, which reveals the result.)

        Returns prediction dict, or None if the game cannot be processed.
        """
        game_features = self._cache["game_features"]
        season_index = self._cache["season_index"]
        gs_cache = self._cache["gs_cache"]
        player_id_map = self._cache["player_id_map"]
        player_experience = self._cache.get("player_experience", {})

        from src.transformer.phase2.sequence_builder import (
            Phase2SequenceBuilder,
            Phase2Sample,
        )

        data_cfg = self._config.data
        builder = Phase2SequenceBuilder(
            game_features=game_features,
            season_index=season_index,
            n_recent_games=data_cfg.n_recent_games,
            max_context_games=data_cfg.max_context_games,
        )

        sample, roster_confidence = self._build_pre_game_sample(game_id, builder)

        if sample is None:
            return None

        # Check minimum history
        if (
            len(sample.home_context) < data_cfg.min_history_games
            or len(sample.away_context) < data_cfg.min_history_games
        ):
            logger.warning(
                "Insufficient history for game %s (home=%d, away=%d, min=%d)",
                game_id,
                len(sample.home_context),
                len(sample.away_context),
                data_cfg.min_history_games,
            )
            self.skipped[game_id] = (
                f"fewer than {data_cfg.min_history_games} games this season"
            )
            return None

        # Encode the sample into tensor batch
        batch = self._encode_sample(sample, player_id_map, gs_cache, player_experience)

        # Move batch to device
        batch = {
            k: v.to(self._device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        # Run inference
        with torch.no_grad():
            output = self._model(batch)

        # Extract predictions
        spread_mu = output.spread_mean.item()
        spread_sigma = output.spread_std.item()
        home_score_mu = output.home_score_mean.item()
        away_score_mu = output.away_score_mean.item()
        win_prob = output.home_win_prob.item()

        total_mu = home_score_mu + away_score_mu
        total_sigma = float(
            np.sqrt(
                output.home_score_std.item() ** 2 + output.away_score_std.item() ** 2
            )
        )

        return {
            "pred_home_score": round(home_score_mu, 1),
            "pred_away_score": round(away_score_mu, 1),
            "pred_home_win_pct": round(win_prob, 4),
            "pred_spread": round(spread_mu, 2),
            "pred_spread_sigma": round(spread_sigma, 2),
            "pred_total": round(total_mu, 1),
            "pred_total_sigma": round(total_sigma, 2),
            "roster_confidence": round(roster_confidence, 3),
        }

    def _build_pre_game_sample(
        self, game_id: str, builder
    ) -> tuple[Optional["Phase2Sample"], float]:
        """
        Build a Phase2Sample for a game as of its tip-off.

        Looks up game info from the DB, gets projected rosters via
        RosterAssembler, and constructs context sequences from the team's
        prior completed games already in the cache.

        Returns:
            (sample, roster_confidence) or (None, 0.0) on failure.
        """
        from src.transformer.phase2.sequence_builder import Phase2Sample
        from src.transformer.phase2.cache_builder import HISTORICAL_TO_CURRENT
        from src.database import DB_PATH, get_db

        db_path = DB_PATH

        # Step 1: Look up game metadata from the DB
        with get_db(str(db_path)) as conn:
            row = conn.execute(
                "SELECT home_team, away_team, date_time_utc, season "
                "FROM Games WHERE game_id = ?",
                (game_id,),
            ).fetchone()

        if row is None:
            logger.warning("Game %s not found in database", game_id)
            return None, 0.0

        home_team, away_team, date_time_utc, season = row

        # Normalize historical team abbreviations
        home_team = HISTORICAL_TO_CURRENT.get(home_team, home_team)
        away_team = HISTORICAL_TO_CURRENT.get(away_team, away_team)

        # Extract date string (YYYY-MM-DD) for comparison with cache dates
        target_date = date_time_utc[:10] if date_time_utc else None
        if not target_date:
            logger.warning("Game %s has no date_time_utc", game_id)
            return None, 0.0

        if not season:
            logger.warning("Game %s has no season", game_id)
            return None, 0.0

        logger.info(
            "Building pre-game sample for %s: %s @ %s on %s (%s)",
            game_id,
            away_team,
            home_team,
            target_date,
            season,
        )

        # Step 2: Get context sequences from the cache (prior completed games)
        home_context = builder._get_prior_games(home_team, season, target_date)
        away_context = builder._get_prior_games(away_team, season, target_date)

        # Compute days before target for each context game
        home_days_before = [
            builder._compute_days_before(gf.game_date, target_date)
            for gf in home_context
        ]
        away_days_before = [
            builder._compute_days_before(gf.game_date, target_date)
            for gf in away_context
        ]

        # Mark recent game indices
        n_home = len(home_context)
        n_away = len(away_context)
        home_recent_indices = list(
            range(max(0, n_home - builder.n_recent_games), n_home)
        )
        away_recent_indices = list(
            range(max(0, n_away - builder.n_recent_games), n_away)
        )

        # Compute rest days from context
        home_rest_days = builder._compute_rest_days(home_context, target_date)
        away_rest_days = builder._compute_rest_days(away_context, target_date)

        # Step 3: Get projected rosters via RosterAssembler
        roster_confidence = 1.0
        try:
            if self._roster_assembler is None:
                from src.pipeline.roster_assembler import RosterAssembler

                self._roster_assembler = RosterAssembler(
                    db_path=str(db_path), historical=self.historical
                )
            rosters = self._roster_assembler.get_projected_rosters([game_id])
            roster_info = rosters.get(game_id)

            if roster_info:
                home_roster = roster_info["home_players"]
                away_roster = roster_info["away_players"]
                roster_confidence = min(
                    roster_info["home_confidence"],
                    roster_info["away_confidence"],
                )
                if roster_info.get("warnings"):
                    for w in roster_info["warnings"]:
                        logger.info("Roster warning for %s: %s", game_id, w)
            else:
                logger.warning(
                    "RosterAssembler returned no roster for %s, "
                    "falling back to last-game roster",
                    game_id,
                )
                home_roster = self._fallback_roster_from_context(
                    home_context, home_team
                )
                away_roster = self._fallback_roster_from_context(
                    away_context, away_team
                )
                roster_confidence = 0.5
        except Exception as e:
            logger.warning(
                "RosterAssembler failed for %s (%s), using context fallback",
                game_id,
                e,
            )
            home_roster = self._fallback_roster_from_context(home_context, home_team)
            away_roster = self._fallback_roster_from_context(away_context, away_team)
            roster_confidence = 0.5

        sample = Phase2Sample(
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
            home_roster=home_roster[:ROSTER_CAP],
            away_roster=away_roster[:ROSTER_CAP],
            home_rest_days=home_rest_days,
            away_rest_days=away_rest_days,
            target_home_score=0.0,  # Unknown for upcoming games
            target_away_score=0.0,
        )

        return sample, roster_confidence

    @staticmethod
    def _fallback_roster_from_context(context: list, team: str) -> list[int]:
        """
        Extract a roster from the most recent context game as a fallback
        when the RosterAssembler is unavailable.
        """
        if not context:
            return []
        last_game = context[-1]
        if last_game.home_team == team:
            stats = last_game.home_player_stats
            players = last_game.home_player_points
        else:
            stats = last_game.away_player_stats
            players = last_game.away_player_points
        if stats:
            return [entry[0] for entry in stats]
        return [entry[0] for entry in players]

    def _encode_sample(
        self,
        sample,
        player_id_map: dict,
        gs_cache: dict,
        player_experience: dict,
    ) -> dict:
        """
        Encode a Phase2Sample into a batched tensor dict (batch_size=1).

        This replicates the logic in Phase2Dataset.__getitem__ and
        collate_phase2, but for a single inference sample.
        """
        from src.transformer.phase2.dataset import _encode_team_context

        data_cfg = self._config.data
        model_cfg = self._config.model
        max_players = data_cfg.max_players_per_game
        max_roster = data_cfg.max_roster_size

        # Build roster arrays (mapped to model indices)
        home_roster = np.zeros(max_roster, dtype=np.int64)
        away_roster = np.zeros(max_roster, dtype=np.int64)
        for i, pid in enumerate(sample.home_roster[:max_roster]):
            home_roster[i] = player_id_map.get(pid, 0)
        for i, pid in enumerate(sample.away_roster[:max_roster]):
            away_roster[i] = player_id_map.get(pid, 0)

        # Determine season start year for player experience
        # Use sample.season directly (works for both cached and upcoming games)
        season_start_year = None
        if sample.season:
            try:
                season_start_year = int(sample.season.split("-")[0])
            except (ValueError, IndexError):
                pass

        # Common kwargs for _encode_team_context
        common_kwargs = dict(
            gs_cache=gs_cache,
            player_id_map=player_id_map,
            max_players=max_players,
            max_roster=max_roster,
            max_appearances=getattr(model_cfg, "max_player_appearances", 40),
            enable_player_form=model_cfg.enable_player_form,
            n_player_stats=data_cfg.n_player_stats,
            n_efficiency_features=model_cfg.n_efficiency_features,
            player_experience=player_experience,
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

        # Build result dict with batch dimension (unsqueeze numpy, then to tensor)
        result = {}
        for prefix, data in [("home_", home_data), ("away_", away_data)]:
            for key, value in data.items():
                result[prefix + key] = torch.from_numpy(value).unsqueeze(0)

        # Roster and rest days (with batch dim)
        result["home_roster"] = torch.from_numpy(home_roster).unsqueeze(0)
        result["away_roster"] = torch.from_numpy(away_roster).unsqueeze(0)
        result["home_rest_days"] = torch.tensor(
            min(sample.home_rest_days, 29), dtype=torch.int64
        ).unsqueeze(0)
        result["away_rest_days"] = torch.tensor(
            min(sample.away_rest_days, 29), dtype=torch.int64
        ).unsqueeze(0)

        # Game mask: True = padding (for batch=1, no padding needed, all False)
        for prefix in ["home_", "away_"]:
            n_games = result[prefix + "scores"].shape[1]
            result[prefix + "game_mask"] = torch.zeros(1, n_games, dtype=torch.bool)

        return result
