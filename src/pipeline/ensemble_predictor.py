"""
Ensemble Predictor: combines predictions from multiple individual models.

Live, it reads the predictions the individual predictors just saved (it runs
LAST in the pipeline); in historical mode (backtests) it runs the components
itself, each as of tip-off. Spreads are averaged with equal weights and win
probabilities in log-odds.

Usage:
    from src.pipeline.ensemble_predictor import EnsemblePredictor
    predictor = EnsemblePredictor()
    preds = predictor.make_pre_game_predictions(["0022500900", "0022500901"])
"""

from __future__ import annotations

import json
import logging
import math

from src.database import get_db
from src.predictions.prediction_engines.base_predictor import BasePredictor

logger = logging.getLogger(__name__)

# Models that produce spread directly (pred_spread field)
SPREAD_MODELS = {"Phase5", "Phase3"}

# Models that produce individual scores (pred_home_score, pred_away_score)
SCORE_MODELS = {"Linear", "Tree", "MLP"}

# Minimum number of component predictions required to produce an ensemble
MIN_COMPONENTS = 2


class EnsemblePredictor(BasePredictor):
    """Ensemble predictor that combines predictions from individual models."""

    COMPONENT_MODELS = ["Phase5", "Phase3", "Linear", "Tree", "MLP"]

    def __init__(self, model_paths=None, historical: bool = False):
        super().__init__(model_paths)
        self.historical = historical

    def make_pre_game_predictions(self, game_ids: list[str]) -> dict:
        """
        Generate ensemble predictions by combining individual model predictions.

        Reads each component model's stored prediction (in historical mode,
        runs the components), then combines them using:
          - Spread: average of all available spreads (direct or derived from scores)
          - Win prob: log-odds (logit) averaging
          - Scores: the average total of every model that predicts scores
            (Linear, Tree, MLP, Phase3), split by the ensemble spread

        Args:
            game_ids: List of game IDs to generate ensemble predictions for.

        Returns:
            Dict mapping game_id to prediction dict with keys:
                pred_home_score, pred_away_score, pred_home_win_pct,
                pred_spread, prediction_type, model_contributions
        """
        if not game_ids:
            return {}

        # Step 1: Component predictions
        component_preds = (
            self._run_components(game_ids)
            if self.historical
            else self._load_component_predictions(game_ids)
        )

        # Step 2: Look up game statuses for prediction_type
        game_statuses = self._get_game_statuses(game_ids)

        # Step 3: Combine predictions for each game
        predictions = {}
        for game_id in game_ids:
            available = component_preds.get(game_id, {})

            if len(available) < MIN_COMPONENTS:
                logger.debug(
                    f"Game {game_id}: only {len(available)} component predictions "
                    f"(need {MIN_COMPONENTS}), skipping ensemble"
                )
                continue

            try:
                pred = self._combine_predictions(game_id, available, game_statuses)
                predictions[game_id] = pred
            except Exception as e:
                logger.warning(f"Error combining predictions for game {game_id}: {e}")

        logger.info(
            f"Ensemble: {len(predictions)}/{len(game_ids)} games predicted "
            f"(min {MIN_COMPONENTS} components required)"
        )
        return predictions

    def _load_component_predictions(
        self, game_ids: list[str]
    ) -> dict[str, dict[str, dict]]:
        """
        Load component model predictions from the Predictions table.

        Returns:
            {game_id: {predictor_name: prediction_dict}}
        """
        placeholders_games = ",".join("?" * len(game_ids))
        placeholders_models = ",".join("?" * len(self.COMPONENT_MODELS))

        with get_db() as conn:
            # For each game+predictor combo, get the most recent prediction
            rows = conn.execute(
                f"""
                SELECT game_id, predictor, prediction_set
                FROM Predictions
                WHERE game_id IN ({placeholders_games})
                AND predictor IN ({placeholders_models})
                ORDER BY prediction_datetime DESC
                """,
                game_ids + self.COMPONENT_MODELS,
            ).fetchall()

        # Build nested dict, keeping only most recent per game+predictor
        result: dict[str, dict[str, dict]] = {}
        seen: set[tuple[str, str]] = set()

        for game_id, predictor, prediction_set_json in rows:
            key = (game_id, predictor)
            if key in seen:
                continue  # Already have the most recent one (ORDER BY DESC)
            seen.add(key)

            try:
                pred = json.loads(prediction_set_json)
            except (json.JSONDecodeError, TypeError):
                logger.warning(f"Invalid JSON for {predictor}/{game_id}, skipping")
                continue

            # Validate that prediction has win probability (required for ensemble)
            if "pred_home_win_pct" not in pred:
                logger.debug(
                    f"{predictor}/{game_id}: missing pred_home_win_pct, skipping"
                )
                continue

            result.setdefault(game_id, {})[predictor] = pred

        return result

    def _run_components(self, game_ids: list[str]) -> dict[str, dict[str, dict]]:
        """Run each component as of tip-off (no stored rows are used).
        Returns {game_id: {predictor_name: prediction_dict}}."""
        from src.predictions.prediction_manager import (
            _drop_implausible_predictions,
            create_predictor,
        )

        result: dict[str, dict[str, dict]] = {}
        for name in self.COMPONENT_MODELS:
            try:
                preds = _drop_implausible_predictions(
                    create_predictor(name, historical=True).make_pre_game_predictions(
                        game_ids
                    ),
                    name,
                )
            except Exception as e:
                logger.warning(f"Ensemble component {name} failed: {e}")
                continue
            for game_id, pred in preds.items():
                if "pred_home_win_pct" in pred:
                    result.setdefault(game_id, {})[name] = pred
        return result

    def _get_game_statuses(self, game_ids: list[str]) -> dict[str, int]:
        """Look up game status (1=scheduled, 3=final) for prediction_type."""
        placeholders = ",".join("?" * len(game_ids))
        with get_db() as conn:
            rows = conn.execute(
                f"SELECT game_id, status FROM Games WHERE game_id IN ({placeholders})",
                game_ids,
            ).fetchall()
        return {row[0]: row[1] for row in rows}

    def _combine_predictions(
        self,
        game_id: str,
        available: dict[str, dict],
        game_statuses: dict[str, int],
    ) -> dict:
        """
        Combine predictions from available component models for a single game.

        Args:
            game_id: The game ID.
            available: {predictor_name: prediction_dict} for this game.
            game_statuses: {game_id: status} lookup.

        Returns:
            Combined prediction dict.
        """
        # --- Spread: collect from all models ---
        spreads = []
        spread_sources = []
        for model, pred in available.items():
            if model in SPREAD_MODELS and "pred_spread" in pred:
                spreads.append(pred["pred_spread"])
                spread_sources.append(model)
            elif model in SCORE_MODELS:
                home = pred.get("pred_home_score")
                away = pred.get("pred_away_score")
                if home is not None and away is not None:
                    spreads.append(home - away)
                    spread_sources.append(model)

        ensemble_spread = sum(spreads) / len(spreads) if spreads else 0.0

        # --- Win probability: log-odds averaging ---
        win_probs = []
        win_sources = []
        for model, pred in available.items():
            wp = pred.get("pred_home_win_pct")
            if wp is not None and not math.isnan(wp):
                win_probs.append(wp)
                win_sources.append(model)

        ensemble_win_pct = _logodds_average(win_probs) if win_probs else 0.5

        # --- Scores: split the average total of the models that predict
        # scores (Phase3 included) by the ensemble spread, so the scores agree
        # with it; none if no model predicts scores ---
        totals = []
        for model, pred in available.items():
            home = pred.get("pred_home_score")
            away = pred.get("pred_away_score")
            if home is not None and away is not None:
                totals.append(home + away)
        avg_total = sum(totals) / len(totals) if totals else None

        # --- Prediction type ---
        game_status = game_statuses.get(game_id)
        prediction_type = "live" if game_status == 1 else "backtest"

        # --- Model contributions ---
        model_contributions = {}
        for model, pred in available.items():
            contrib = {"pred_home_win_pct": pred.get("pred_home_win_pct")}

            # Include spread (direct or derived)
            if model in SPREAD_MODELS and "pred_spread" in pred:
                contrib["pred_spread"] = pred["pred_spread"]
            elif model in SCORE_MODELS:
                home = pred.get("pred_home_score")
                away = pred.get("pred_away_score")
                if home is not None and away is not None:
                    contrib["pred_spread"] = home - away
                    contrib["pred_home_score"] = home
                    contrib["pred_away_score"] = away

            model_contributions[model] = contrib

        return {
            "pred_home_score": (
                round((avg_total + ensemble_spread) / 2, 1) if totals else None
            ),
            "pred_away_score": (
                round((avg_total - ensemble_spread) / 2, 1) if totals else None
            ),
            "pred_home_win_pct": round(ensemble_win_pct, 4),
            "pred_spread": round(ensemble_spread, 2),
            "prediction_type": prediction_type,
            "model_contributions": model_contributions,
            "n_components": len(available),
        }


def _logodds_average(probs: list[float]) -> float:
    """
    Average probabilities in log-odds (logit) space with equal weights.

    Clamps probabilities to [0.001, 0.999] to avoid infinities.
    """
    EPS = 1e-3
    logits = []
    for p in probs:
        p_clamped = max(EPS, min(1.0 - EPS, p))
        logits.append(math.log(p_clamped / (1.0 - p_clamped)))

    avg_logit = sum(logits) / len(logits)
    return 1.0 / (1.0 + math.exp(-avg_logit))
