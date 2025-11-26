"""
Database Validator CLI Tool.

Comprehensive validation suite for NBA_AI database with 50+ checks across 9 categories.
Validates flag logic, data integrity, scores, volumes, temporal constraints, and more.

Usage:
    python -m src.database_validator --categories flag,integrity
    python -m src.database_validator --fix --check-id FLAG-001
    python -m src.database_validator --output json > validation_report.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

from src.config import config
from src.logging_config import setup_logging
from src.validators import (
    AlignmentValidator,
    FeatureValidator,
    FlagValidator,
    IntegrityValidator,
    PriorStateValidator,
    ScoreValidator,
    TeamValidator,
    TemporalValidator,
    VolumeValidator,
)

VALIDATOR_MAP = {
    "flag": FlagValidator,
    "integrity": IntegrityValidator,
    "score": ScoreValidator,
    "volume": VolumeValidator,
    "temporal": TemporalValidator,
    "alignment": AlignmentValidator,
    "prior_state": PriorStateValidator,
    "feature": FeatureValidator,
    "team": TeamValidator,
}


class DatabaseValidator:
    """Main validator orchestrator."""

    def __init__(self, db_path: str):
        """
        Initialize database validator.

        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self.validators = {}

    def validate(self, categories: List[str] = None) -> Dict[str, Any]:
        """
        Run validation checks.

        Args:
            categories: List of validator categories to run (default: all)

        Returns:
            Validation report dictionary
        """
        if categories is None:
            categories = list(VALIDATOR_MAP.keys())

        self.logger.info(f"Running validation for categories: {', '.join(categories)}")

        all_issues = []
        stats = {
            "total_checks": 0,
            "issues_found": 0,
            "critical": 0,
            "warning": 0,
            "info": 0,
        }

        for category in categories:
            if category not in VALIDATOR_MAP:
                self.logger.warning(f"Unknown validator category: {category}")
                continue

            validator_class = VALIDATOR_MAP[category]
            validator = validator_class(self.db_path)
            self.validators[category] = validator

            self.logger.info(f"Running {category} validator...")
            issues = validator.validate()

            all_issues.extend(issues)
            stats["total_checks"] += len(
                [m for m in dir(validator) if m.startswith("_check_")]
            )
            stats["issues_found"] += len(issues)

            for issue in issues:
                stats[issue.severity] = stats.get(issue.severity, 0) + 1

        report = {
            "database": self.db_path,
            "categories": categories,
            "stats": stats,
            "issues": [self._issue_to_dict(issue) for issue in all_issues],
        }

        self._print_summary(report)
        return report

    def fix(self, check_id: str = None, categories: List[str] = None) -> Dict[str, Any]:
        """
        Attempt to auto-fix validation issues.

        Args:
            check_id: Specific check to fix (e.g., "FLAG-001")
            categories: Categories to fix (default: all)

        Returns:
            Fix report dictionary
        """
        if categories is None:
            categories = list(VALIDATOR_MAP.keys())

        self.logger.info(
            f"Attempting to fix issues in categories: {', '.join(categories)}"
        )

        fixed_count = 0
        fix_details = []

        for category in categories:
            if category not in VALIDATOR_MAP:
                continue

            validator_class = VALIDATOR_MAP[category]
            validator = validator_class(self.db_path)

            try:
                count = validator.fix(check_id=check_id)
                if count > 0:
                    fixed_count += count
                    fix_details.append(
                        {
                            "category": category,
                            "check_id": check_id or "all",
                            "fixed": count,
                        }
                    )
                    self.logger.info(f"{category}: Fixed {count} issues")
            except Exception as e:
                self.logger.error(f"Error fixing {category}: {e}")
                fix_details.append({"category": category, "error": str(e)})

        report = {"total_fixed": fixed_count, "details": fix_details}

        self.logger.info(f"Fixed {fixed_count} total issues")
        return report

    def _issue_to_dict(self, issue) -> Dict[str, Any]:
        """Convert ValidationIssue to dictionary."""
        return {
            "check_id": issue.check_id,
            "severity": issue.severity,
            "category": issue.category,
            "message": issue.message,
            "count": issue.count,
            "sample_data": issue.sample_data[:5] if issue.sample_data else [],
            "fixable": issue.fixable,
        }

    def _print_summary(self, report: Dict[str, Any]):
        """Print validation summary to console."""
        print("\n" + "=" * 80)
        print("DATABASE VALIDATION REPORT")
        print("=" * 80)
        print(f"Database: {report['database']}")
        print(f"Categories: {', '.join(report['categories'])}")
        print(f"\nTotal Checks Run: {report['stats']['total_checks']}")
        print(f"Issues Found: {report['stats']['issues_found']}")
        print(f"  - Critical: {report['stats'].get('critical', 0)}")
        print(f"  - Warning:  {report['stats'].get('warning', 0)}")
        print(f"  - Info:     {report['stats'].get('info', 0)}")

        if report["issues"]:
            print(f"\n{'-' * 80}")
            print("ISSUES DETAILS")
            print("-" * 80)

            for issue in report["issues"]:
                icon = (
                    "🔴"
                    if issue["severity"] == "critical"
                    else "🟡" if issue["severity"] == "warning" else "🔵"
                )
                fix_marker = "✓ Fixable" if issue["fixable"] else "✗ Manual"
                print(f"\n{icon} [{issue['check_id']}] {issue['message']}")
                print(f"   Count: {issue['count']:,} | {fix_marker}")

                if issue.get("sample_data"):
                    print(f"   Sample: {issue['sample_data'][0]}")

        print("\n" + "=" * 80 + "\n")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Validate NBA_AI database integrity and quality",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all validators
  python -m src.database_validator

  # Run specific categories
  python -m src.database_validator --categories flag,integrity,score

  # Fix all auto-fixable issues
  python -m src.database_validator --fix

  # Fix specific check
  python -m src.database_validator --fix --check-id FLAG-001

  # Output as JSON
  python -m src.database_validator --output json > report.json
        """,
    )

    parser.add_argument(
        "--categories",
        type=str,
        help="Comma-separated list of validator categories (flag,integrity,score,volume,temporal,alignment,prior_state,feature,team)",
    )
    parser.add_argument(
        "--fix", action="store_true", help="Attempt to auto-fix validation issues"
    )
    parser.add_argument(
        "--check-id", type=str, help="Specific check ID to fix (e.g., FLAG-001)"
    )
    parser.add_argument(
        "--output",
        type=str,
        choices=["console", "json"],
        default="console",
        help="Output format",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    # Get database path from config
    project_root = Path(config["project"]["root"])
    db_path = project_root / config["database"]["path"]

    if not db_path.exists():
        logger.error(f"Database not found: {db_path}")
        sys.exit(1)

    # Parse categories
    categories = None
    if args.categories:
        categories = [c.strip() for c in args.categories.split(",")]
        invalid = [c for c in categories if c not in VALIDATOR_MAP]
        if invalid:
            logger.error(f"Invalid categories: {', '.join(invalid)}")
            logger.info(f"Valid categories: {', '.join(VALIDATOR_MAP.keys())}")
            sys.exit(1)

    # Run validator
    validator = DatabaseValidator(str(db_path))

    if args.fix:
        report = validator.fix(check_id=args.check_id, categories=categories)
        if args.output == "json":
            print(json.dumps(report, indent=2))
        else:
            print(f"\nFixed {report['total_fixed']} issues")
    else:
        report = validator.validate(categories=categories)
        if args.output == "json":
            print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
