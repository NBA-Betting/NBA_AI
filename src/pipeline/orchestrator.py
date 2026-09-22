"""
Pipeline Orchestrator: daily entry point for the NBA_AI production pipeline.

Two modes of operation:
  - post-game (~1:30am ET): update DB with yesterday's results, run L1 incremental updates
  - pre-game  (~4pm ET):    refresh injuries/betting, generate predictions for today's games

Usage:
    python -m src.pipeline.orchestrator --mode=post-game --season=Current
    python -m src.pipeline.orchestrator --mode=pre-game --season=Current --dry-run
    python -m src.pipeline.orchestrator --mode=full
"""

from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime
from pathlib import Path

from src.config import config
from src.database import DB_PATH, get_db

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
from src.logging_config import setup_logging
from src.utils import determine_current_season, get_current_eastern_datetime

logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """Coordinates post-game and pre-game pipeline stages."""

    def __init__(self, season: str = "Current", db_path: str | None = None):
        self.db_path = db_path or DB_PATH
        self.season = season
        self._resolved_season: str | None = None

    @property
    def resolved_season(self) -> str:
        """Resolve 'Current' to actual season string, cached."""
        if self._resolved_season is None:
            if self.season == "Current":
                self._resolved_season = determine_current_season()
            else:
                self._resolved_season = self.season
        return self._resolved_season

    # ------------------------------------------------------------------
    # Post-Game Mode
    # ------------------------------------------------------------------

    def run_post_game(self, dry_run: bool = False) -> dict:
        """
        Post-game pipeline: update DB with completed game data, run L1 updates.

        Stages:
          1. Full database update (schedule, players, injuries, betting, PBP,
             GameStates, boxscores, prior states, feature sets)
          2. L1 incremental vector updates for players with new games

        Args:
            dry_run: If True, report what would happen without making changes.

        Returns:
            Summary dict with stages, timing, counts, errors.
        """
        from src.pipeline.monitoring import PipelineMonitor

        monitor = PipelineMonitor(db_path=self.db_path)
        run_id = monitor.start_run("post_game", self.resolved_season)

        summary = {
            "mode": "post_game",
            "season": self.resolved_season,
            "dry_run": dry_run,
            "stages": {},
            "errors": [],
            "warnings": [],
            "total_time": 0.0,
        }
        t_start = time.time()

        # Stage 1: Database update
        stage_result = self._run_stage(
            "database_update",
            lambda: self._stage_database_update(dry_run),
        )
        summary["stages"]["database_update"] = stage_result
        if stage_result.get("error"):
            summary["errors"].append(f"database_update: {stage_result['error']}")

        # Stage 2: PBP enrichment (parse raw PBP into player-level stats)
        # Only runs if the enrichment script/data is available
        stage_result = self._run_stage(
            "pbp_enrichment",
            lambda: self._stage_pbp_enrichment(dry_run),
        )
        summary["stages"]["pbp_enrichment"] = stage_result

        # Stage 3: L1 incremental updates (requires Phase 5 checkpoint)
        l1_checkpoint = PROJECT_ROOT / "models" / "phase5" / "l1.pt"
        if l1_checkpoint.exists():
            stage_result = self._run_stage(
                "l1_update",
                lambda: self._stage_l1_update(dry_run),
            )
            summary["stages"]["l1_update"] = stage_result
            if stage_result.get("error"):
                summary["errors"].append(f"l1_update: {stage_result['error']}")
        else:
            logger.debug("Skipping L1 update (no Phase 5 checkpoint)")

        summary["total_time"] = round(time.time() - t_start, 1)

        # Determine overall status
        status = self._determine_status(summary["errors"], summary["stages"])
        games_processed = sum(
            s.get("games_processed", 0) for s in summary["stages"].values()
        )

        monitor.complete_run(
            run_id,
            status=status,
            games=games_processed,
            predictions=0,
            errors=summary["errors"],
            warnings=summary["warnings"],
            metrics={"total_time": summary["total_time"]},
        )

        logger.info(
            f"Post-game pipeline {status}: "
            f"{summary['total_time']}s, {len(summary['errors'])} errors"
        )
        return summary

    # ------------------------------------------------------------------
    # Pre-Game Mode
    # ------------------------------------------------------------------

    def run_pre_game(self, dry_run: bool = False) -> dict:
        """
        Pre-game pipeline: refresh live data, generate predictions for today's games.

        Stages:
          1. Refresh injury reports
          2. Refresh betting lines
          3. Find today's upcoming games
          4. Run Phase5 predictor and save predictions

        Args:
            dry_run: If True, report what would happen without making changes.

        Returns:
            Summary dict with stages, timing, counts, errors.
        """
        from src.pipeline.monitoring import PipelineMonitor

        monitor = PipelineMonitor(db_path=self.db_path)
        run_id = monitor.start_run("pre_game", self.resolved_season)

        summary = {
            "mode": "pre_game",
            "season": self.resolved_season,
            "dry_run": dry_run,
            "stages": {},
            "errors": [],
            "warnings": [],
            "total_time": 0.0,
            "predictions_generated": 0,
        }
        t_start = time.time()

        # Stage 1: Refresh injuries
        stage_result = self._run_stage(
            "refresh_injuries",
            lambda: self._stage_refresh_injuries(dry_run),
        )
        summary["stages"]["refresh_injuries"] = stage_result
        if stage_result.get("error"):
            summary["errors"].append(f"refresh_injuries: {stage_result['error']}")

        # Stage 2: Refresh betting lines
        stage_result = self._run_stage(
            "refresh_betting",
            lambda: self._stage_refresh_betting(dry_run),
        )
        summary["stages"]["refresh_betting"] = stage_result
        if stage_result.get("error"):
            summary["errors"].append(f"refresh_betting: {stage_result['error']}")

        # Stage 3: Find today's games
        stage_result = self._run_stage(
            "find_games",
            lambda: self._stage_find_todays_games(),
        )
        summary["stages"]["find_games"] = stage_result
        if stage_result.get("error"):
            summary["errors"].append(f"find_games: {stage_result['error']}")

        game_ids = stage_result.get("game_ids", [])

        if not game_ids:
            summary["warnings"].append("No upcoming games found for today")
            logger.info("No upcoming games found for today — skipping prediction stage")
        else:
            # Stage 4: Generate predictions
            stage_result = self._run_stage(
                "predictions",
                lambda: self._stage_generate_predictions(game_ids, dry_run),
            )
            summary["stages"]["predictions"] = stage_result
            summary["predictions_generated"] = stage_result.get(
                "predictions_generated", 0
            )
            if stage_result.get("error"):
                summary["errors"].append(f"predictions: {stage_result['error']}")

        summary["total_time"] = round(time.time() - t_start, 1)

        # Determine overall status
        status = self._determine_status(summary["errors"], summary["stages"])

        monitor.complete_run(
            run_id,
            status=status,
            games=len(game_ids),
            predictions=summary["predictions_generated"],
            errors=summary["errors"],
            warnings=summary["warnings"],
            metrics={"total_time": summary["total_time"]},
        )

        logger.info(
            f"Pre-game pipeline {status}: "
            f"{len(game_ids)} games, {summary['predictions_generated']} predictions, "
            f"{summary['total_time']}s, {len(summary['errors'])} errors"
        )
        return summary

    # ------------------------------------------------------------------
    # Full Mode
    # ------------------------------------------------------------------

    def run_full(self, dry_run: bool = False) -> dict:
        """Run post-game then pre-game sequentially."""
        post = self.run_post_game(dry_run=dry_run)
        pre = self.run_pre_game(dry_run=dry_run)
        return {
            "mode": "full",
            "post_game": post,
            "pre_game": pre,
            "total_time": round(post["total_time"] + pre["total_time"], 1),
        }

    # ------------------------------------------------------------------
    # Individual stage implementations
    # ------------------------------------------------------------------

    def _stage_database_update(self, dry_run: bool) -> dict:
        """Stage: full database update via database_update_manager."""
        if dry_run:
            logger.info("[dry-run] Would run full database update")
            return {"status": "skipped", "dry_run": True}

        from src.database_updater.database_update_manager import update_database

        update_database(
            season=self.resolved_season, predictor=None, db_path=self.db_path
        )
        return {"status": "ok"}

    def _stage_pbp_enrichment(self, dry_run: bool) -> dict:
        """Stage: Parse raw PBP into per-player enriched stats (PBPPlayerGameStatsV2)."""
        if dry_run:
            logger.info("[dry-run] Would enrich PBP data")
            return {"status": "skipped", "dry_run": True}

        try:
            import sys

            sys.path.insert(
                0, str(Path(__file__).resolve().parent.parent.parent / "scripts")
            )
            from parse_pbp_stats_v2 import (
                get_all_game_ids,
                get_processed_game_ids,
                process_games,
            )

            from src.database import get_db

            with get_db(self.db_path) as conn:
                all_ids = get_all_game_ids(conn)
                already = get_processed_game_ids(conn)
                new_ids = [gid for gid in all_ids if gid not in already]

                if not new_ids:
                    return {
                        "status": "ok",
                        "games_processed": 0,
                        "message": "up to date",
                    }

                logger.info(f"Enriching PBP for {len(new_ids)} new games")
                process_games(new_ids, conn)

            return {"status": "ok", "games_processed": len(new_ids)}

        except Exception as e:
            logger.warning(f"PBP enrichment failed (non-critical): {e}")
            return {"status": "warning", "error": str(e), "games_processed": 0}

    def _stage_l1_update(self, dry_run: bool) -> dict:
        """Stage: L1 incremental vector updates for players with new games."""
        from src.pipeline.l1_updater import L1IncrementalUpdater

        updater = L1IncrementalUpdater(db_path=str(self.db_path))
        results = updater.update_all_new(dry_run=dry_run)
        total_games = sum(results.values())
        return {
            "status": "ok",
            "players_updated": len(results),
            "games_processed": total_games,
        }

    def _stage_refresh_injuries(self, dry_run: bool) -> dict:
        """Stage: refresh injury reports from NBA official source."""
        if dry_run:
            logger.info("[dry-run] Would refresh injury data")
            return {"status": "skipped", "dry_run": True}

        from src.database_updater.database_update_manager import update_injury_data

        update_injury_data(season=self.resolved_season, db_path=self.db_path)
        return {"status": "ok"}

    def _stage_refresh_betting(self, dry_run: bool) -> dict:
        """Stage: refresh betting lines from ESPN/Covers."""
        if dry_run:
            logger.info("[dry-run] Would refresh betting lines")
            return {"status": "skipped", "dry_run": True}

        from src.database_updater.database_update_manager import update_betting_lines

        update_betting_lines(season=self.resolved_season, db_path=self.db_path)
        return {"status": "ok"}

    def _stage_find_todays_games(self) -> dict:
        """Stage: query the Games table for today's upcoming games (status=1).

        Uses an ET-aware UTC window (5am-5am UTC) to correctly handle
        late-night ET games that cross the UTC date boundary.
        """
        now_et = get_current_eastern_datetime()
        today_str = now_et.strftime("%Y-%m-%d")

        # ET date → UTC window: games on an ET date span from ~5am UTC to ~5am UTC next day
        # (EST=UTC-5, EDT=UTC-4). Using 5am covers both DST cases.
        utc_start = f"{today_str}T05:00:00Z"
        # Next day
        from datetime import timedelta

        tomorrow = now_et + timedelta(days=1)
        tomorrow_str = tomorrow.strftime("%Y-%m-%d")
        utc_end = f"{tomorrow_str}T05:00:00Z"

        with get_db(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT game_id FROM Games
                WHERE date_time_utc >= ? AND date_time_utc < ?
                AND status = 1
                ORDER BY date_time_utc
                """,
                (utc_start, utc_end),
            )
            game_ids = [row[0] for row in cursor.fetchall()]

        logger.info(f"Found {len(game_ids)} upcoming games for {today_str}")
        return {"status": "ok", "game_ids": game_ids, "date": today_str}

    def _stage_generate_predictions(self, game_ids: list[str], dry_run: bool) -> dict:
        """Stage: run all configured predictors and save predictions."""
        if dry_run:
            logger.info(
                f"[dry-run] Would generate predictions for {len(game_ids)} games"
            )
            return {
                "status": "skipped",
                "dry_run": True,
                "game_ids": game_ids,
                "predictions_generated": 0,
            }

        from src.predictions.prediction_manager import (
            make_pre_game_predictions,
            save_predictions,
        )

        # Determine which predictors are available based on model files
        predictors_to_run = ["Baseline"]  # Always available (formula-based)

        # Legacy ML models (need .joblib/.pth files)
        if (PROJECT_ROOT / "models" / "linear" / "model.joblib").exists():
            predictors_to_run.append("Linear")
        if (PROJECT_ROOT / "models" / "tree" / "home_model.json").exists():
            predictors_to_run.append("Tree")
        if (PROJECT_ROOT / "models" / "mlp" / "model.pth").exists():
            predictors_to_run.append("MLP")

        # Deep learning models (need checkpoints)
        if (PROJECT_ROOT / "models" / "phase5" / "l3l4.pt").exists():
            predictors_to_run.append("Phase5")
        if (PROJECT_ROOT / "models" / "phase3" / "model.pt").exists():
            predictors_to_run.append("Phase3")

        # Ensemble (needs at least 2 component models to have predictions)
        if len(predictors_to_run) >= 3:  # Baseline + at least 2 others
            predictors_to_run.append("Ensemble")

        logger.info(f"Available predictors: {predictors_to_run}")
        total_preds = 0
        results_by_predictor = {}

        for predictor_name in predictors_to_run:
            try:
                # Check which games still need predictions for this predictor
                from src.database import get_db

                with get_db(self.db_path) as conn:
                    placeholders = ",".join("?" * len(game_ids))
                    existing = conn.execute(
                        f"""
                        SELECT game_id FROM Predictions
                        WHERE predictor = ? AND game_id IN ({placeholders})
                        """,
                        [predictor_name] + game_ids,
                    ).fetchall()
                already_predicted = {row[0] for row in existing}
                needs_prediction = [
                    gid for gid in game_ids if gid not in already_predicted
                ]

                if not needs_prediction:
                    logger.info(
                        f"  {predictor_name}: all {len(game_ids)} games already predicted"
                    )
                    results_by_predictor[predictor_name] = 0
                    continue

                predictions = make_pre_game_predictions(
                    needs_prediction, predictor_name, save=True
                )
                n = len(predictions) if predictions else 0
                total_preds += n
                results_by_predictor[predictor_name] = n
                logger.info(f"  {predictor_name}: {n} predictions generated")

            except Exception as e:
                logger.warning(f"  {predictor_name} failed: {e}")
                results_by_predictor[predictor_name] = 0

        return {
            "status": "ok",
            "predictions_generated": total_preds,
            "game_ids": game_ids,
            "by_predictor": results_by_predictor,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _run_stage(self, name: str, fn) -> dict:
        """
        Execute a pipeline stage, catching exceptions so one failure
        doesn't block subsequent stages.

        Returns the stage result dict, augmented with timing and error info.
        """
        t0 = time.time()
        try:
            result = fn()
            result["duration"] = round(time.time() - t0, 1)
            return result
        except Exception as e:
            logger.error(f"Stage '{name}' failed: {e}", exc_info=True)
            return {
                "status": "error",
                "error": str(e),
                "duration": round(time.time() - t0, 1),
            }

    @staticmethod
    def _determine_status(errors: list, stages: dict) -> str:
        """Determine overall run status from error list and stage results."""
        if not errors:
            return "success"
        # If all stages failed, it's a full failure
        stage_statuses = [s.get("status") for s in stages.values()]
        if all(s == "error" for s in stage_statuses):
            return "failed"
        return "partial"


def main():
    parser = argparse.ArgumentParser(
        description="NBA_AI Pipeline Orchestrator — daily prediction pipeline"
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["post-game", "pre-game", "full"],
        help="Pipeline mode: post-game (DB update + L1), pre-game (predictions), or full (both)",
    )
    parser.add_argument(
        "--season",
        type=str,
        default="Current",
        help="Season to process (e.g. '2025-2026' or 'Current'). Default: Current",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would happen without making changes",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level. Default: INFO",
    )
    args = parser.parse_args()

    setup_logging(log_level=args.log_level)

    orchestrator = PipelineOrchestrator(season=args.season)

    mode_map = {
        "post-game": orchestrator.run_post_game,
        "pre-game": orchestrator.run_pre_game,
        "full": orchestrator.run_full,
    }

    result = mode_map[args.mode](dry_run=args.dry_run)

    # Print summary
    if args.mode == "full":
        post_errors = len(result["post_game"]["errors"])
        pre_errors = len(result["pre_game"]["errors"])
        preds = result["pre_game"].get("predictions_generated", 0)
        print(
            f"\nFull pipeline complete in {result['total_time']}s "
            f"— {post_errors + pre_errors} errors, {preds} predictions"
        )
    else:
        errors = len(result.get("errors", []))
        preds = result.get("predictions_generated", 0)
        total = result.get("total_time", 0)
        print(
            f"\n{args.mode} pipeline complete in {total}s — {errors} errors, {preds} predictions"
        )

    # Print any errors
    for err in result.get("errors", []):
        print(f"  ERROR: {err}")

    # Print any warnings
    for warn in result.get("warnings", []):
        print(f"  WARNING: {warn}")


if __name__ == "__main__":
    main()
