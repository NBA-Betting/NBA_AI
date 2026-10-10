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

from src.database import DB_PATH
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

L1_VECTORS_DIR = PROJECT_ROOT / "data" / "l2_cache" / "l1_vectors"
MODELS_DIR = PROJECT_ROOT / "models" / "phase5"
PHASE_B_CACHE = PROJECT_ROOT / "data" / "phase_b_cache"

# Maximum players per team for L2 (must match training)
MAX_PLAYERS = 13

# Players actually fed to the model per team. Training rosters were the players
# who logged minutes in the game itself (~10.7/team), and that count tracks the
# final margin (benches empty in blowouts), so the model reads roster size as a
# margin signal. On 2025-26 validation games with pre-game rosters, spread MAE
# was 11.90 / 11.92 / 12.57 / 13.03 at caps 10 / 11 / 12 / 15 (Vegas 10.91).
# Retraining on pre-game rosters is the real fix.
ROSTER_CAP = 11
# A team with fewer players that have L1 vectors looks weaker to the model
# (roster-size signal), so its prediction would be misleading: skip the game
MIN_L1_PLAYERS = 8
# Spread uncertainty for win probability: residual RMSE of the 2025-26 backtest
SPREAD_SIGMA = 15.0


class Phase5Predictor(BasePredictor):
    """Phase 5 Hierarchical model predictor."""

    def __init__(self, model_paths=None):
        super().__init__(model_paths)
        self._models_loaded = False
        self._device = None
        self._l2_model = None
        self._l3_model = None
        self._l4_model = None
        self._l2_cfg = None
        self._roster_assembler = RosterAssembler()
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
        self._archetype_centroids = None

    def _ensure_models_loaded(self):
        """Lazy-load models on first prediction call."""
        if self._models_loaded:
            return

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load configs
        self._l2_cfg = L2Config()
        l3_cfg = L3Config()
        l4_cfg = L4Config()

        # Normalization: Phase C (l3l4.pt) used training-split stats, not the
        # all-data stats in the Phase B metadata (val spread MAE 10.37 vs 10.47)
        norm_path = MODELS_DIR / "norm_stats.json"
        if norm_path.exists():
            norm = json.loads(norm_path.read_text())
        else:
            logger.warning(
                f"{norm_path} missing; falling back to Phase B normalization stats"
            )
            norm = json.loads((PHASE_B_CACHE / "metadata.json").read_text())[
                "normalization"
            ]
        self._l3_mean = np.array(norm["l3_mean"], dtype=np.float32)
        self._l3_std = np.array(norm["l3_std"], dtype=np.float32)
        self._l4_mean = np.array(norm["l4_mean"], dtype=np.float32)
        self._l4_std = np.array(norm["l4_std"], dtype=np.float32)
        self._rs_mean = np.array(norm["roster_summary_mean"], dtype=np.float32)
        self._rs_std = np.array(norm["roster_summary_std"], dtype=np.float32)

        # Player-to-index mapping for L2 FM residual embeddings
        idx_path = PHASE_B_CACHE / "player_to_idx.json"
        if idx_path.exists():
            self._player_to_idx = {
                int(k): v for k, v in json.loads(idx_path.read_text()).items()
            }
        else:
            raise FileNotFoundError(
                f"{idx_path} missing: without the training-time player index every "
                "player would map to the padding embedding. Re-run "
                "scripts/build_phase_b_cache.py or restore the file."
            )

        # Coach-to-index mapping for the L3 coach embedding. Training falls back
        # to index 0 with 0 games when the coach cache is absent; do the same.
        coach_path = PHASE_B_CACHE / "coach_mapping.json"
        self._coach_to_idx = (
            json.loads(coach_path.read_text())["coach_to_idx"]
            if coach_path.exists()
            else None
        )

        # Ensure L2 has enough embedding slots
        if self._player_to_idx:
            self._l2_cfg.n_players = max(
                self._l2_cfg.n_players, max(self._player_to_idx.values()) + 1
            )

        # Load archetype centroids for soft assignment
        arch_path = PROJECT_ROOT / "data" / "phase5_cache" / "archetypes.npz"
        if arch_path.exists():
            self._archetype_centroids = np.load(str(arch_path))["centroids"]  # (K, 32)
        else:
            self._archetype_centroids = None

        # Create models
        self._l2_model = PlayerSynergyNetwork(self._l2_cfg).to(self._device)
        self._l3_model = TeamModel(l3_cfg).to(self._device)
        self._l4_model = GamePredictor(l4_cfg).to(self._device)

        # Load checkpoints
        l2_ckpt = torch.load(
            str(MODELS_DIR / "l2.pt"),
            map_location=self._device,
            weights_only=False,
        )
        self._l2_model.load_state_dict(l2_ckpt["model_state_dict"], strict=False)

        c_ckpt = torch.load(
            str(MODELS_DIR / "l3l4.pt"),
            map_location=self._device,
            weights_only=False,
        )
        self._l3_model.load_state_dict(c_ckpt["l3_state_dict"], strict=False)
        self._l4_model.load_state_dict(c_ckpt["l4_state_dict"], strict=False)
        # L2 is fine-tuned jointly with L3/L4; use the weights L3/L4 were trained
        # against rather than the standalone l2.pt.
        if "l2_state_dict" in c_ckpt:
            self._l2_model.load_state_dict(c_ckpt["l2_state_dict"], strict=False)

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
                predictions[game_id] = pred
            except Exception as e:
                logger.warning(f"Error predicting game {game_id}: {e}")

        return predictions

    def _predict_single_game(self, game_id: str, roster: dict, features: dict) -> dict:
        """Run the full L2→L3→L4 forward pass for a single game."""
        device = self._device

        # Load L1 vectors for all players
        home_data = self._load_player_data(roster["home_players"])
        away_data = self._load_player_data(roster["away_players"])
        n_home, n_away = int(home_data["mask"].sum()), int(away_data["mask"].sum())
        if min(n_home, n_away) < MIN_L1_PLAYERS:
            raise ValueError(
                f"too few players with L1 vectors (home {n_home}, away {n_away})"
            )

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
        if self._coach_to_idx is None or coach_id is None:
            return 0, 0.0
        idx = self._coach_to_idx.get(str(coach_id), 0)
        # Coaches unseen in training have no learned embedding (index 0)
        return idx, float(games) if idx else 0.0

    def _load_player_data(self, player_ids: list[int]) -> dict:
        """
        Load L1 vectors for a list of player IDs.

        Returns dict with abilities, uncertainties, archetypes, mask, player_idx
        all as tensors of shape (MAX_PLAYERS, ...).
        """
        abilities = np.zeros((MAX_PLAYERS, 32), dtype=np.float32)
        uncertainties = np.ones((MAX_PLAYERS, 32), dtype=np.float32)
        archetypes = np.zeros((MAX_PLAYERS, 10), dtype=np.float32)
        mask = np.zeros(MAX_PLAYERS, dtype=bool)
        player_idx = np.zeros(MAX_PLAYERS, dtype=np.int64)

        # Top ROSTER_CAP players (sorted by projected minutes) that have L1
        # vectors; players without them would only be masked out anyway
        j = 0
        for pid in player_ids:
            if j == ROSTER_CAP:
                break
            npz_path = L1_VECTORS_DIR / f"{pid}.npz"
            if not npz_path.exists():
                continue
            data = np.load(str(npz_path), allow_pickle=True)
            # Use the most recent ability vector
            abilities[j] = data["ability"][-1]
            uncertainties[j] = data["uncertainty"][-1]
            # Archetype weights are produced by the L1 model (static per player)
            archetypes[j] = data["archetype_weights"]
            mask[j] = True
            player_idx[j] = self._player_to_idx.get(pid, 0)
            j += 1

        return {
            "abilities": torch.tensor(abilities, dtype=torch.float32),
            "uncertainties": torch.tensor(uncertainties, dtype=torch.float32),
            "archetypes": torch.tensor(archetypes, dtype=torch.float32),
            "mask": torch.tensor(mask, dtype=torch.bool),
            "player_idx": torch.tensor(player_idx, dtype=torch.long),
        }

    def _compute_archetypes(self, ability: np.ndarray) -> np.ndarray:
        """Compute soft archetype assignment from ability vector."""
        if self._archetype_centroids is None:
            return np.ones(10, dtype=np.float32) / 10.0

        # Euclidean distance to each centroid
        dists = np.linalg.norm(
            self._archetype_centroids - ability[np.newaxis, :], axis=1
        )
        # Soft assignment via negative distance softmax
        logits = -dists
        logits = logits - logits.max()  # numerical stability
        weights = np.exp(logits)
        weights /= weights.sum() + 1e-8
        return weights.astype(np.float32)

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
