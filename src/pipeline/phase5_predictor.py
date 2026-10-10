"""
Phase 5 Hierarchical Predictor: L1→L2→L3→L4 inference pipeline.

Implements the BasePredictor interface for integration with the existing
prediction system. For each game: assembles roster, loads L1 vectors,
runs L2→L3→L4 forward pass, returns spread/win/total predictions.

Usage:
    from src.pipeline.phase5_predictor import Phase5Predictor
    predictor = Phase5Predictor()
    preds = predictor.make_pre_game_predictions(["0022500900"])
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

from src.database import DB_PATH, get_db
from src.phase5.l1_store import L1Store, load_game_dates
from src.phase5.l2_config import L2Config
from src.phase5.roster_summary import compute_roster_summary
from src.phase5.l2_model import PlayerSynergyNetwork
from src.phase5.l3_config import L3Config
from src.phase5.l3_model import TeamModel
from src.phase5.l4_config import L4Config
from src.phase5.l4_model import GamePredictor
from src.pipeline.roster_assembler import RosterAssembler
from src.pipeline.team_features import TeamFeatureComputer
from src.predictions.prediction_engines.base_predictor import BasePredictor

logger = logging.getLogger(__name__)

MODELS_DIR = PROJECT_ROOT / "models" / "phase5"
# Inference inputs for checkpoints that predate saving them in the checkpoint
LEGACY_PHASE_B_CACHE = PROJECT_ROOT / "data" / "phase_b_cache"

# Legacy checkpoints were trained on the players who logged minutes in each game
# (~10.7/team), a count that tracks the final margin, so they read roster size
# as a margin signal. Feed them 11 players: on 2025-26 validation games with
# pre-game rosters, spread MAE was 11.90 / 11.92 / 12.57 / 13.03 at 10 / 11 /
# 12 / 15 players (Vegas 10.91). Checkpoints trained on pre-game rosters
# (scripts/build_phase5_cache.py) take their full max_roster instead.
LEGACY_ROSTER_CAP = 11
# A team with fewer players that have L1 vectors looks weaker to the model
# (roster-size signal), so its prediction would be misleading: skip the game
MIN_L1_PLAYERS = 8
# Spread uncertainty for win probability: residual RMSE of the 2025-26 backtest
SPREAD_SIGMA = 15.0


class Phase5Predictor(BasePredictor):
    """Phase 5 Hierarchical model predictor."""

    def __init__(self, model_paths=None, historical: bool = False):
        """historical: predict past games as of their tip-off (backtests):
        pre-game rosters from appearances, and no L1 vector update."""
        super().__init__(model_paths)
        self.historical = historical
        self._models_loaded = False
        self._device = None
        self._l2_model = None
        self._l3_model = None
        self._l4_model = None
        self._l2_cfg = None
        self._roster_assembler = RosterAssembler(historical=historical)
        self._team_features = TeamFeatureComputer()
        self._l1_updater = None  # Lazy-loaded on first freshness check

        # Normalization stats and player index mapping
        self._l3_mean = None
        self._l3_std = None
        self._l4_mean = None
        self._l4_std = None
        self._rs_mean = None
        self._rs_std = None
        self._player_to_idx = None
        self._max_players = None
        self._l1_store = None

    def _ensure_models_loaded(self):
        """Lazy-load models on first prediction call."""
        if self._models_loaded:
            return

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load configs
        self._l2_cfg = L2Config()
        l3_cfg = L3Config()
        l4_cfg = L4Config()

        c_ckpt = torch.load(
            str(MODELS_DIR / "l3l4.pt"), map_location=self._device, weights_only=False
        )
        inference = c_ckpt.get("inference") or self._legacy_inference()
        norm = inference["normalization"]
        self._l3_mean = np.array(norm["l3_mean"], dtype=np.float32)
        self._l3_std = np.array(norm["l3_std"], dtype=np.float32)
        self._l4_mean = np.array(norm["l4_mean"], dtype=np.float32)
        self._l4_std = np.array(norm["l4_std"], dtype=np.float32)
        self._rs_mean = np.array(norm["roster_summary_mean"], dtype=np.float32)
        self._rs_std = np.array(norm["roster_summary_std"], dtype=np.float32)
        self._player_to_idx = {int(k): v for k, v in inference["player_to_idx"].items()}
        # Coaches unseen in training map to index 0 (no learned embedding)
        self._coach_to_idx = inference["coach_to_idx"]
        self._max_players = inference["max_roster"]
        with get_db(str(DB_PATH)) as conn:
            self._l1_store = L1Store(load_game_dates(conn))
        self._l2_cfg.n_players = max(
            self._l2_cfg.n_players, max(self._player_to_idx.values()) + 1
        )

        self._l2_model = PlayerSynergyNetwork(self._l2_cfg).to(self._device)
        self._l3_model = TeamModel(l3_cfg).to(self._device)
        self._l4_model = GamePredictor(l4_cfg).to(self._device)
        self._l3_model.load_state_dict(c_ckpt["l3_state_dict"], strict=False)
        self._l4_model.load_state_dict(c_ckpt["l4_state_dict"], strict=False)
        # L2 is fine-tuned jointly with L3/L4; use the weights L3/L4 were
        # trained against (standalone l2.pt only for checkpoints without them)
        if "l2_state_dict" in c_ckpt:
            l2_state = c_ckpt["l2_state_dict"]
        else:
            l2_state = torch.load(
                str(MODELS_DIR / "l2.pt"), map_location=self._device, weights_only=False
            )["model_state_dict"]
        self._l2_model.load_state_dict(l2_state, strict=False)

        self._l2_model.eval()
        self._l3_model.eval()
        self._l4_model.eval()

        logger.info(
            f"Phase 5 models loaded on {self._device}: "
            f"L2={sum(p.numel() for p in self._l2_model.parameters()):,}, "
            f"L3={sum(p.numel() for p in self._l3_model.parameters()):,}, "
            f"L4={sum(p.numel() for p in self._l4_model.parameters()):,}"
        )
        self._models_loaded = True

    @staticmethod
    def _legacy_inference() -> dict:
        """Inference inputs for checkpoints that don't carry them: the
        training-split normalization in models/phase5/norm_stats.json (Phase C
        trained on it; val spread MAE 10.37 vs 10.47 with the Phase B metadata
        stats) and the mappings in the old Phase B cache."""
        coach_path = LEGACY_PHASE_B_CACHE / "coach_mapping.json"
        return {
            "normalization": json.loads((MODELS_DIR / "norm_stats.json").read_text()),
            "player_to_idx": json.loads(
                (LEGACY_PHASE_B_CACHE / "player_to_idx.json").read_text()
            ),
            "coach_to_idx": (
                json.loads(coach_path.read_text())["coach_to_idx"]
                if coach_path.exists()
                else {}
            ),
            "max_roster": LEGACY_ROSTER_CAP,
        }

    def _ensure_l1_fresh(self):
        """Update L1 vectors if any players have new games since last update."""
        try:
            if self._l1_updater is None:
                from src.pipeline.l1_updater import L1IncrementalUpdater

                self._l1_updater = L1IncrementalUpdater(
                    db_path=str(DB_PATH), device="cpu"
                )
            results = self._l1_updater.update_all_new()
            if results:
                total = sum(results.values())
                logger.info(
                    f"L1 vectors updated: {len(results)} players, {total} new games"
                )
        except Exception as e:
            logger.warning(f"L1 freshness check failed (using cached vectors): {e}")

    def make_pre_game_predictions(self, game_ids: list[str]) -> dict:
        """
        Generate pre-game predictions for the given games.

        Automatically updates L1 vectors if new games have been played since
        the last update, ensuring fresh player ability estimates.

        Returns {game_id: {
            pred_home_win_pct, pred_spread, pred_spread_sigma,
            prediction_type, roster_confidence
        }}
        """
        if not self.historical:
            self._ensure_l1_fresh()
        self._ensure_models_loaded()

        # Step 1: Assemble rosters
        rosters = self._roster_assembler.get_projected_rosters(game_ids)

        # Step 2: Compute team features and game context
        team_feats = self._team_features.compute_for_games(game_ids, rosters)

        # Step 3: For each game, run L1→L2→L3→L4
        predictions = {}
        for game_id in game_ids:
            if game_id not in rosters or game_id not in team_feats:
                logger.warning(f"Missing data for game {game_id}, skipping")
                continue

            try:
                pred = self._predict_single_game(
                    game_id, rosters[game_id], team_feats[game_id]
                )
                if pred is not None:
                    predictions[game_id] = pred
            except Exception as e:
                logger.warning(f"Error predicting game {game_id}: {e}")

        return predictions

    def _predict_single_game(
        self, game_id: str, roster: dict, features: dict
    ) -> dict | None:
        """Run the full L2→L3→L4 forward pass for a single game (None if
        the game is skipped, with the reason in self.skipped)."""
        device = self._device

        # L1 states before tip-off for the roster players
        tip_off = self._l1_store.game_dates[game_id]
        home_data = self._load_player_data(roster["home_players"], tip_off)
        away_data = self._load_player_data(roster["away_players"], tip_off)
        n_home, n_away = int(home_data["mask"].sum()), int(away_data["mask"].sum())
        if min(n_home, n_away) < MIN_L1_PLAYERS:
            self.skipped[game_id] = (
                f"too few players with L1 vectors (home {n_home}, away {n_away})"
            )
            logger.warning(f"Skipping game {game_id}: {self.skipped[game_id]}")
            return None

        # Normalize team features and game context
        home_tf = self._normalize(
            features["home_team_features"], self._l3_mean, self._l3_std
        )
        away_tf = self._normalize(
            features["away_team_features"], self._l3_mean, self._l3_std
        )
        game_ctx = self._normalize(
            features["game_context"], self._l4_mean, self._l4_std
        )

        # Compute roster summaries (12-d: mean ability stats)
        home_rs = self._compute_roster_summary(home_data)
        away_rs = self._compute_roster_summary(away_data)
        home_rs_norm = self._normalize(home_rs, self._rs_mean, self._rs_std)
        away_rs_norm = self._normalize(away_rs, self._rs_mean, self._rs_std)

        # Continuity from team features (index 32, raw)
        home_cont = features["home_team_features"][32]
        away_cont = features["away_team_features"][32]

        # Coach embedding inputs: training index and career games coached
        home_coach_idx, home_coach_games = self._coach_inputs(features, "home")
        away_coach_idx, away_coach_games = self._coach_inputs(features, "away")

        # Convert to tensors
        with torch.no_grad():
            # L2 forward
            home_l2_out = self._l2_model(
                ability=home_data["abilities"].unsqueeze(0).to(device),
                uncertainty=home_data["uncertainties"].unsqueeze(0).to(device),
                archetypes=home_data["archetypes"].unsqueeze(0).to(device),
                mask=home_data["mask"].unsqueeze(0).to(device),
                player_idx=home_data["player_idx"].unsqueeze(0).to(device),
            )
            away_l2_out = self._l2_model(
                ability=away_data["abilities"].unsqueeze(0).to(device),
                uncertainty=away_data["uncertainties"].unsqueeze(0).to(device),
                archetypes=away_data["archetypes"].unsqueeze(0).to(device),
                mask=away_data["mask"].unsqueeze(0).to(device),
                player_idx=away_data["player_idx"].unsqueeze(0).to(device),
            )

            home_l2 = home_l2_out["team_vector"]  # (1, 134)
            away_l2 = away_l2_out["team_vector"]

            # L3 forward
            home_repr = self._l3_model(
                l2_team=home_l2,
                team_features=torch.tensor(home_tf, dtype=torch.float32)
                .unsqueeze(0)
                .to(device),
                roster_summary=torch.tensor(home_rs_norm, dtype=torch.float32)
                .unsqueeze(0)
                .to(device),
                coach_idx=torch.tensor([home_coach_idx], dtype=torch.long).to(device),
                roster_continuity=torch.tensor([[home_cont]], dtype=torch.float32).to(
                    device
                ),
                coach_games=torch.tensor([home_coach_games], dtype=torch.float32).to(
                    device
                ),
            )
            away_repr = self._l3_model(
                l2_team=away_l2,
                team_features=torch.tensor(away_tf, dtype=torch.float32)
                .unsqueeze(0)
                .to(device),
                roster_summary=torch.tensor(away_rs_norm, dtype=torch.float32)
                .unsqueeze(0)
                .to(device),
                coach_idx=torch.tensor([away_coach_idx], dtype=torch.long).to(device),
                roster_continuity=torch.tensor([[away_cont]], dtype=torch.float32).to(
                    device
                ),
                coach_games=torch.tensor([away_coach_games], dtype=torch.float32).to(
                    device
                ),
            )

            # L4 forward
            preds = self._l4_model(
                team_home=home_repr,
                team_away=away_repr,
                l2_home=home_l2,
                l2_away=away_l2,
                context=torch.tensor(game_ctx, dtype=torch.float32)
                .unsqueeze(0)
                .to(device),
            )

        # Extract predictions
        spread_mu = preds["spread_mu"].item()
        # Win probability is P(margin > 0) under N(spread_mu, SPREAD_SIGMA). The
        # model's own spread_sigma is under-dispersed (mean ~10.7 vs a 15.0 RMSE;
        # 52% of outcomes within 1 sigma), which made win % overconfident.
        # 2025-26 pre-game-roster backtest Brier: 0.234 fixed sigma, 0.239 model
        # sigma, 0.241 classifier head.
        win_prob = 0.5 * (1.0 + math.erf(spread_mu / (SPREAD_SIGMA * math.sqrt(2))))
        # No total or team scores: the total head is untrained (w_total=0 in
        # L4Config; val total MAE ~123), so total_mu and any scores derived
        # from it are meaningless.

        # Determine if this is a live prediction (before game) or backtest
        from src.database import get_db

        with get_db(str(DB_PATH)) as conn:
            row = conn.execute(
                "SELECT status FROM Games WHERE game_id = ?",
                (game_id,),
            ).fetchone()
        game_status = row[0] if row else None
        is_live = game_status == 1  # status=1 means scheduled (not yet played)

        return {
            "pred_home_win_pct": round(win_prob, 4),
            "pred_spread": round(spread_mu, 2),
            "pred_spread_sigma": SPREAD_SIGMA,
            "prediction_type": "live" if is_live else "backtest",
            "roster_confidence": min(
                roster.get("home_confidence", 1.0),
                roster.get("away_confidence", 1.0),
            ),
        }

    def _coach_inputs(self, features: dict, side: str) -> tuple[int, float]:
        """(coach_idx, coach_games) for the L3 coach embedding."""
        coach_id, games = features.get(f"{side}_coach", (None, 0))
        if coach_id is None:
            return 0, 0.0
        idx = self._coach_to_idx.get(str(coach_id), 0)
        # Coaches unseen in training have no learned embedding (index 0)
        return idx, float(games) if idx else 0.0

    def _load_player_data(self, player_ids: list[int], before_date: str) -> dict:
        """L2 inputs for a roster (ordered by projected minutes): the first
        max_players players with an L1 state before the date, as tensors."""
        arrays = self._l1_store.roster_arrays(
            [player_ids], [before_date], self._player_to_idx, self._max_players
        )
        return {
            "abilities": torch.tensor(arrays["abilities"][0]),
            "uncertainties": torch.tensor(arrays["uncertainties"][0]),
            "archetypes": torch.tensor(arrays["archetypes"][0]),
            "mask": torch.tensor(arrays["mask"][0]),
            "player_idx": torch.tensor(arrays["player_idx"][0]),
        }

    def _compute_roster_summary(self, player_data: dict) -> np.ndarray:
        """Compute the 12-d roster summary exactly as the training cache does."""
        return compute_roster_summary(
            player_data["abilities"].numpy(),
            player_data["uncertainties"].numpy(),
            player_data["mask"].numpy(),
        )

    @staticmethod
    def _normalize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        """Z-score normalization."""
        return (x - mean) / (std + 1e-8)
