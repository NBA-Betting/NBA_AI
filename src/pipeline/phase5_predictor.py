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

        # Load Phase B metadata for normalization and player mapping
        meta_path = PHASE_B_CACHE / "metadata.json"
        meta = json.loads(meta_path.read_text())
        norm = meta["normalization"]
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
            self._player_to_idx = {}

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
            pred_home_score, pred_away_score, pred_home_win_pct,
            pred_spread, pred_spread_sigma, pred_total, pred_total_sigma,
            pred_ats_prob, roster_confidence
        }}
        """
        self._ensure_l1_fresh()
        self._ensure_models_loaded()

        # Step 1: Assemble rosters
        rosters = self._roster_assembler.get_projected_rosters(game_ids)

        # Step 2: Compute team features and game context
        team_feats = self._team_features.compute_for_games(game_ids)

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
                coach_idx=torch.tensor([0], dtype=torch.long).to(device),
                roster_continuity=torch.tensor([[home_cont]], dtype=torch.float32).to(
                    device
                ),
                coach_games=torch.tensor([0.0], dtype=torch.float32).to(device),
            )
            away_repr = self._l3_model(
                l2_team=away_l2,
                team_features=torch.tensor(away_tf, dtype=torch.float32)
                .unsqueeze(0)
                .to(device),
                roster_summary=torch.tensor(away_rs_norm, dtype=torch.float32)
                .unsqueeze(0)
                .to(device),
                coach_idx=torch.tensor([0], dtype=torch.long).to(device),
                roster_continuity=torch.tensor([[away_cont]], dtype=torch.float32).to(
                    device
                ),
                coach_games=torch.tensor([0.0], dtype=torch.float32).to(device),
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
        spread_sigma = preds["spread_sigma"].item()
        # The classifier head is badly mis-calibrated (mean 0.81 home win prob
        # vs a 0.55 base rate over 2025-26). Derive win probability from the
        # spread head and its own predicted uncertainty instead: P(margin > 0)
        # under N(spread_mu, spread_sigma). Brier 0.232 vs 0.253 for the head
        # on 2025-26 live games.
        sigma_for_prob = spread_sigma if spread_sigma and spread_sigma > 1.0 else 20.0
        win_prob = 0.5 * (1.0 + math.erf(spread_mu / (sigma_for_prob * math.sqrt(2))))
        total_mu = preds["total_mu"].item()
        total_sigma = preds["total_sigma"].item()

        # Convert spread/total to home/away scores
        home_score = (total_mu + spread_mu) / 2.0
        away_score = (total_mu - spread_mu) / 2.0

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
            "pred_home_score": round(home_score, 1),
            "pred_away_score": round(away_score, 1),
            "pred_home_win_pct": round(win_prob, 4),
            "pred_spread": round(spread_mu, 2),
            "pred_spread_sigma": round(spread_sigma, 2),
            "pred_total": round(total_mu, 1),
            "pred_total_sigma": round(total_sigma, 2),
            "prediction_type": "live" if is_live else "backtest",
            "roster_confidence": min(
                roster.get("home_confidence", 1.0),
                roster.get("away_confidence", 1.0),
            ),
        }

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

        for j, pid in enumerate(player_ids[:MAX_PLAYERS]):
            npz_path = L1_VECTORS_DIR / f"{pid}.npz"
            if npz_path.exists():
                data = np.load(str(npz_path), allow_pickle=True)
                # Use the most recent ability vector
                abilities[j] = data["ability"][-1]
                uncertainties[j] = data["uncertainty"][-1]
                # Archetype weights are produced by the L1 model (static per player)
                archetypes[j] = data["archetype_weights"]
                mask[j] = True
                player_idx[j] = self._player_to_idx.get(pid, 0)
            else:
                # Player has no L1 vectors — use zero (masked out)
                mask[j] = False

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
        """Compute 12-d roster summary from player abilities."""
        abilities = player_data["abilities"].numpy()  # (A, 32)
        mask = player_data["mask"].numpy()  # (A,)
        uncertainties = player_data["uncertainties"].numpy()  # (A, 32)

        n_valid = mask.sum()
        if n_valid == 0:
            return np.zeros(12, dtype=np.float32)

        valid_ab = abilities[mask]  # (N, 32)
        valid_unc = uncertainties[mask]

        summary = np.zeros(12, dtype=np.float32)
        # Mean ability across dims (take first 6 principal dims as summary)
        summary[0:6] = valid_ab.mean(axis=0)[:6]
        # Std of ability norms (spread of talent)
        norms = np.linalg.norm(valid_ab, axis=1)
        summary[6] = norms.mean()
        summary[7] = norms.std() if len(norms) > 1 else 0.0
        # Mean uncertainty
        summary[8] = valid_unc.mean()
        # Number of valid players (normalized)
        summary[9] = n_valid / 13.0
        # Max ability norm (star power)
        summary[10] = norms.max() if len(norms) > 0 else 0.0
        # Min ability norm (depth)
        summary[11] = norms.min() if len(norms) > 0 else 0.0

        return summary

    @staticmethod
    def _normalize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        """Z-score normalization."""
        return (x - mean) / (std + 1e-8)
