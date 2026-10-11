"""
Pipeline monitoring: tracks runs and logs results.

Creates and manages the PipelineRuns table in the project database.
Used by the PipelineOrchestrator to record each run's outcome.

Usage:
    from src.pipeline.monitoring import PipelineMonitor
    monitor = PipelineMonitor()
    run_id = monitor.start_run("pre_game", "2025-2026")
    # ... do work ...
    monitor.complete_run(run_id, "success", games=8, predictions=8,
                         errors=[], warnings=[], metrics={"mae": 10.1})
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime

from src.database import DB_PATH, get_db

logger = logging.getLogger(__name__)


class PipelineMonitor:
    """Records and queries pipeline run history in the PipelineRuns table."""

    def __init__(self, db_path: str | None = None):
        self.db_path = db_path or DB_PATH
        self._ensure_table()

    def _ensure_table(self):
        """Create PipelineRuns table if it doesn't exist."""
        with get_db(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS PipelineRuns (
                    run_id TEXT PRIMARY KEY,
                    run_type TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    completed_at TEXT,
                    status TEXT NOT NULL,
                    season TEXT,
                    games_processed INTEGER DEFAULT 0,
                    predictions_generated INTEGER DEFAULT 0,
                    errors TEXT,
                    metrics TEXT,
                    warnings TEXT
                )
            """)

    def start_run(self, run_type: str, season: str) -> str:
        """
        Create a run record with status 'running'.

        Args:
            run_type: "post_game" or "pre_game"
            season: e.g. "2025-2026"

        Returns:
            run_id (UUID string)
        """
        run_id = str(uuid.uuid4())
        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

        with get_db(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO PipelineRuns
                    (run_id, run_type, started_at, status, season, errors, warnings, metrics)
                VALUES (?, ?, ?, 'running', ?, '[]', '[]', '{}')
                """,
                (run_id, run_type, now, season),
            )

        logger.info(f"Pipeline run started: {run_type} [{run_id[:8]}]")
        return run_id

    def complete_run(
        self,
        run_id: str,
        status: str,
        games: int = 0,
        predictions: int = 0,
        errors: list | None = None,
        warnings: list | None = None,
        metrics: dict | None = None,
    ):
        """
        Update a run record on completion.

        Args:
            run_id: The UUID returned by start_run().
            status: "success", "partial", or "failed".
            games: Number of games processed.
            predictions: Number of predictions generated.
            errors: List of error message strings.
            warnings: List of warning message strings.
            metrics: Arbitrary metrics dict (timings, counts, etc.).
        """
        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

        with get_db(self.db_path) as conn:
            conn.execute(
                """
                UPDATE PipelineRuns
                SET completed_at = ?,
                    status = ?,
                    games_processed = ?,
                    predictions_generated = ?,
                    errors = ?,
                    warnings = ?,
                    metrics = ?
                WHERE run_id = ?
                """,
                (
                    now,
                    status,
                    games,
                    predictions,
                    json.dumps(errors or []),
                    json.dumps(warnings or []),
                    json.dumps(metrics or {}),
                    run_id,
                ),
            )

        logger.info(
            f"Pipeline run completed: {status} [{run_id[:8]}] "
            f"— {games} games, {predictions} predictions"
        )
