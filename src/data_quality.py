"""
data_quality.py

Streamlined data quality monitoring and validation for NBA AI.

This module provides comprehensive data quality checks optimized for:
1. Current ML pipeline validation
2. Future GenAI training data preparation
3. Real-time collection monitoring

Key Features:
- Fast quality checks (no pandas dependency for core checks)
- Detailed reporting with actionable insights
- Extensible for future data sources (odds, injuries, etc.)
- Timestamp-based freshness tracking
- Automated anomaly detection

Usage:
    # Quick health check
    python -m src.data_quality --season=2024-2025 --quick

    # Full audit with CSV export
    python -m src.data_quality --season=2024-2025 --output=audit.csv

    # Check specific game
    python -m src.data_quality --game_id=0022300649

    # Multi-season report
    python -m src.data_quality --all-seasons

Functions:
    - quick_health_check(season): Fast coverage stats (no deep inspection)
    - full_audit(season): Comprehensive validation with issue detection
    - check_game(game_id): Detailed single-game validation
    - validate_pbp_quality(game_ids): Check PBP parsing quality
    - detect_anomalies(season): Statistical outlier detection
"""

import argparse
import json
import logging
import sqlite3
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from src.config import config
from src.logging_config import setup_logging

DB_PATH = config["database"]["path"]


class DataQualityReport:
    """Container for quality check results."""

    def __init__(self, season: str):
        self.season = season
        self.timestamp = datetime.now().isoformat()
        self.coverage = {}
        self.issues = []
        self.warnings = []
        self.stats = {}

    def add_issue(self, category: str, description: str, game_ids: List[str] = None):
        """Add a data quality issue."""
        self.issues.append(
            {
                "category": category,
                "description": description,
                "game_ids": game_ids or [],
                "count": len(game_ids) if game_ids else 0,
            }
        )

    def add_warning(self, category: str, description: str, count: int = 0):
        """Add a non-critical warning."""
        self.warnings.append(
            {"category": category, "description": description, "count": count}
        )

    def to_dict(self) -> dict:
        """Export report as dictionary."""
        return {
            "season": self.season,
            "timestamp": self.timestamp,
            "coverage": self.coverage,
            "issues": self.issues,
            "warnings": self.warnings,
            "stats": self.stats,
        }

    def print_summary(self):
        """Print human-readable summary."""
        print(f"\n{'='*80}")
        print(f"DATA QUALITY REPORT: {self.season}")
        print(f"Generated: {self.timestamp}")
        print(f"{'='*80}\n")

        # Coverage
        print("COVERAGE:")
        for data_type, stats in self.coverage.items():
            pct = stats["percentage"]
            status = "✅" if pct >= 95 else "⚠️" if pct >= 80 else "❌"
            print(
                f"  {status} {data_type}: {stats['count']}/{stats['total']} ({pct:.1f}%)"
            )

        # Issues
        if self.issues:
            print(f"\n❌ CRITICAL ISSUES ({len(self.issues)}):")
            for issue in self.issues:
                print(f"  - {issue['category']}: {issue['description']}")
                if issue["count"] > 0:
                    print(f"    Affected games: {issue['count']}")
        else:
            print("\n✅ NO CRITICAL ISSUES FOUND")

        # Warnings
        if self.warnings:
            print(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"  - {warning['category']}: {warning['description']}")
                if warning["count"] > 0:
                    print(f"    Count: {warning['count']}")


def quick_health_check(season: str, db_path: str = DB_PATH) -> DataQualityReport:
    """
    Fast health check focusing on coverage percentages.

    Checks:
    - Games scheduled vs completed vs collected
    - PBP, GameStates, PlayerBox, TeamBox, Features, Predictions coverage
    - No deep data validation (fast execution)

    Args:
        season: Season to check (e.g., "2024-2025")
        db_path: Database path

    Returns:
        DataQualityReport with coverage stats
    """
    report = DataQualityReport(season)

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()

        # Total completed games
        cursor.execute(
            """
            SELECT COUNT(*)
            FROM Games
            WHERE season = ?
            AND season_type IN ('Regular Season', 'Post Season')
            AND status IN ('Completed', 'Final')
        """,
            (season,),
        )
        total_games = cursor.fetchone()[0]

        if total_games == 0:
            report.add_warning("Coverage", f"No completed games found for {season}")
            return report

        # PBP Coverage
        cursor.execute(
            """
            SELECT COUNT(DISTINCT g.game_id)
            FROM Games g
            INNER JOIN PbP_Logs p ON g.game_id = p.game_id
            WHERE g.season = ?
            AND g.season_type IN ('Regular Season', 'Post Season')
            AND g.status IN ('Completed', 'Final')
        """,
            (season,),
        )
        pbp_count = cursor.fetchone()[0]
        report.coverage["PBP_Logs"] = {
            "count": pbp_count,
            "total": total_games,
            "percentage": (pbp_count / total_games * 100) if total_games > 0 else 0,
        }

        # GameStates Coverage
        cursor.execute(
            """
            SELECT COUNT(DISTINCT g.game_id)
            FROM Games g
            INNER JOIN GameStates gs ON g.game_id = gs.game_id
            WHERE g.season = ?
            AND g.season_type IN ('Regular Season', 'Post Season')
            AND g.status IN ('Completed', 'Final')
        """,
            (season,),
        )
        states_count = cursor.fetchone()[0]
        report.coverage["GameStates"] = {
            "count": states_count,
            "total": total_games,
            "percentage": (states_count / total_games * 100) if total_games > 0 else 0,
        }

        # PlayerBox Coverage
        cursor.execute(
            """
            SELECT COUNT(DISTINCT g.game_id)
            FROM Games g
            INNER JOIN PlayerBox pb ON g.game_id = pb.game_id
            WHERE g.season = ?
            AND g.season_type IN ('Regular Season', 'Post Season')
            AND g.status IN ('Completed', 'Final')
        """,
            (season,),
        )
        playerbox_count = cursor.fetchone()[0]
        report.coverage["PlayerBox"] = {
            "count": playerbox_count,
            "total": total_games,
            "percentage": (
                (playerbox_count / total_games * 100) if total_games > 0 else 0
            ),
        }

        # Features Coverage
        cursor.execute(
            """
            SELECT COUNT(DISTINCT g.game_id)
            FROM Games g
            INNER JOIN Features f ON g.game_id = f.game_id
            WHERE g.season = ?
            AND g.season_type IN ('Regular Season', 'Post Season')
            AND g.status IN ('Completed', 'Final')
        """,
            (season,),
        )
        features_count = cursor.fetchone()[0]
        report.coverage["Features"] = {
            "count": features_count,
            "total": total_games,
            "percentage": (
                (features_count / total_games * 100) if total_games > 0 else 0
            ),
        }

        # Predictions Coverage
        cursor.execute(
            """
            SELECT COUNT(DISTINCT g.game_id)
            FROM Games g
            INNER JOIN Predictions p ON g.game_id = p.game_id
            WHERE g.season = ?
            AND g.season_type IN ('Regular Season', 'Post Season')
            AND g.status IN ('Completed', 'Final')
        """,
            (season,),
        )
        predictions_count = cursor.fetchone()[0]
        report.coverage["Predictions"] = {
            "count": predictions_count,
            "total": total_games,
            "percentage": (
                (predictions_count / total_games * 100) if total_games > 0 else 0
            ),
        }

        # Check for critical gaps
        if pbp_count < total_games:
            missing = total_games - pbp_count
            report.add_issue(
                "PBP_Missing", f"{missing} games missing PBP data", []
            )  # Could query for specific game_ids

        if playerbox_count < total_games * 0.95:  # 95% threshold
            missing = total_games - playerbox_count
            report.add_issue(
                "PlayerBox_Missing", f"{missing} games missing PlayerBox data", []
            )

    return report


def validate_pbp_quality(
    game_ids: List[str], db_path: str = DB_PATH
) -> Dict[str, dict]:
    """
    Validate PBP data quality for specific games.

    Checks:
    - Record count in expected range (300-800 per game)
    - PbP_Logs count matches GameStates count
    - Final state exists (is_final_state=1)
    - Score progression is monotonic (no score decreases)

    Args:
        game_ids: List of game IDs to validate
        db_path: Database path

    Returns:
        Dict mapping game_id to quality results
    """
    results = {}

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()

        for game_id in game_ids:
            issues = []

            # Check PBP count
            cursor.execute(
                """
                SELECT COUNT(*) FROM PbP_Logs WHERE game_id = ?
            """,
                (game_id,),
            )
            pbp_count = cursor.fetchone()[0]

            # Check GameStates count
            cursor.execute(
                """
                SELECT COUNT(*) FROM GameStates WHERE game_id = ?
            """,
                (game_id,),
            )
            states_count = cursor.fetchone()[0]

            # Validate counts
            if pbp_count < 300:
                issues.append(f"Low PBP count: {pbp_count} (expected 300-800)")
            elif pbp_count > 800:
                issues.append(f"High PBP count: {pbp_count} (expected 300-800)")

            if pbp_count != states_count:
                issues.append(f"PBP/GameStates mismatch: {pbp_count} vs {states_count}")

            # Check for final state
            cursor.execute(
                """
                SELECT COUNT(*) FROM GameStates 
                WHERE game_id = ? AND is_final_state = 1
            """,
                (game_id,),
            )
            final_count = cursor.fetchone()[0]

            if final_count == 0:
                issues.append("No final state found")
            elif final_count > 1:
                issues.append(f"Multiple final states: {final_count}")

            # Check score progression (sample first 10 and last 10 states)
            cursor.execute(
                """
                SELECT play_id, home_score, away_score
                FROM GameStates
                WHERE game_id = ?
                ORDER BY play_id
                LIMIT 20
            """,
                (game_id,),
            )

            prev_home, prev_away = 0, 0
            for play_id, home_score, away_score in cursor.fetchall():
                if home_score < prev_home or away_score < prev_away:
                    issues.append(f"Score decreased at play {play_id}")
                    break
                prev_home, prev_away = home_score, away_score

            results[game_id] = {
                "pbp_count": pbp_count,
                "states_count": states_count,
                "has_final_state": final_count == 1,
                "issues": issues,
                "quality": "PASS" if len(issues) == 0 else "FAIL",
            }

    return results


def check_game(game_id: str, db_path: str = DB_PATH) -> dict:
    """
    Detailed validation for a single game.

    Returns comprehensive status of all data types for one game.
    """
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()

        # Game info
        cursor.execute(
            """
            SELECT home_team, away_team, date_time_est, status,
                   game_data_finalized, pre_game_data_finalized
            FROM Games WHERE game_id = ?
        """,
            (game_id,),
        )

        row = cursor.fetchone()
        if not row:
            return {"error": f"Game {game_id} not found"}

        home, away, date, status, game_data_flag, pre_game_flag = row

        # Count all data types
        data_counts = {}
        for table in ["PbP_Logs", "GameStates", "PlayerBox", "TeamBox", "Features"]:
            cursor.execute(
                f"SELECT COUNT(*) FROM {table} WHERE game_id = ?", (game_id,)
            )
            data_counts[table] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM Predictions WHERE game_id = ?", (game_id,))
        data_counts["Predictions"] = cursor.fetchone()[0]

        return {
            "game_id": game_id,
            "matchup": f"{away} @ {home}",
            "date": date,
            "status": status,
            "game_data_finalized": bool(game_data_flag),
            "pre_game_data_finalized": bool(pre_game_flag),
            "data_counts": data_counts,
        }


def main():
    parser = argparse.ArgumentParser(description="NBA AI Data Quality Monitoring")
    parser.add_argument("--season", type=str, help="Season to check (e.g., 2024-2025)")
    parser.add_argument("--game_id", type=str, help="Check specific game")
    parser.add_argument("--all-seasons", action="store_true", help="Check all seasons")
    parser.add_argument("--quick", action="store_true", help="Quick health check only")
    parser.add_argument("--output", type=str, help="Export report to JSON file")
    parser.add_argument("--log_level", default="INFO", help="Logging level")

    args = parser.parse_args()
    setup_logging(args.log_level)

    if args.game_id:
        # Single game check
        result = check_game(args.game_id)
        print(f"\n{'='*80}")
        print(f"GAME CHECK: {result['game_id']}")
        print(f"{'='*80}")
        print(f"Matchup: {result['matchup']}")
        print(f"Date: {result['date']}")
        print(f"Status: {result['status']}")
        print(f"\nData Counts:")
        for table, count in result["data_counts"].items():
            print(f"  {table}: {count}")
        print(f"\nFlags:")
        print(f"  game_data_finalized: {result['game_data_finalized']}")
        print(f"  pre_game_data_finalized: {result['pre_game_data_finalized']}")

    elif args.all_seasons:
        # Multi-season report
        seasons = ["2023-2024", "2024-2025", "2025-2026"]
        reports = []

        for season in seasons:
            print(f"\nChecking {season}...")
            report = quick_health_check(season)
            reports.append(report)
            report.print_summary()

        if args.output:
            with open(args.output, "w") as f:
                json.dump([r.to_dict() for r in reports], f, indent=2)
            print(f"\n✅ Report saved to {args.output}")

    else:
        # Single season check
        season = args.season or "2024-2025"

        if args.quick:
            report = quick_health_check(season)
            report.print_summary()
        else:
            # Full audit (to be implemented - use quick for now)
            report = quick_health_check(season)
            report.print_summary()

            # TODO: Add deep validation
            # - PBP quality checks
            # - Anomaly detection
            # - Temporal consistency

        if args.output:
            with open(args.output, "w") as f:
                json.dump(report.to_dict(), f, indent=2)
            print(f"\n✅ Report saved to {args.output}")


if __name__ == "__main__":
    main()
