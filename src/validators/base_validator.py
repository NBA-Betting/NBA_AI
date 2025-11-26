"""Base validator class with common functionality."""

import logging
import sqlite3
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ValidationIssue:
    """Represents a single validation issue found."""

    check_id: str
    severity: str  # 'critical', 'warning', 'info'
    category: str
    message: str
    count: int
    sample_data: Optional[List[Dict[str, Any]]] = None
    fixable: bool = False


class BaseValidator:
    """Base class for all validators."""

    def __init__(self, db_path: str):
        """
        Initialize validator.

        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path
        self.logger = logging.getLogger(self.__class__.__name__)
        self.issues: List[ValidationIssue] = []

    def run_query(self, query: str, params: tuple = ()) -> List[tuple]:
        """Execute a query and return results."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return cursor.fetchall()

    def run_query_dict(self, query: str, params: tuple = ()) -> List[Dict[str, Any]]:
        """Execute query and return results as list of dicts."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def execute_update(self, query: str, params: tuple = ()) -> int:
        """Execute an UPDATE/INSERT/DELETE query."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            conn.commit()
            return cursor.rowcount

    def add_issue(
        self,
        check_id: str,
        severity: str,
        message: str,
        count: int,
        sample_data: Optional[List[Dict[str, Any]]] = None,
        fixable: bool = False,
    ):
        """
        Add a validation issue.

        Args:
            check_id: Unique check identifier (e.g., "FLAG-001")
            severity: Issue severity level
            message: Human-readable description
            count: Number of records affected
            sample_data: Sample of affected records
            fixable: Whether issue can be auto-fixed
        """
        issue = ValidationIssue(
            check_id=check_id,
            severity=severity,
            category=self.__class__.__name__.replace("Validator", ""),
            message=message,
            count=count,
            sample_data=sample_data or [],
            fixable=fixable,
        )
        self.issues.append(issue)
        self.logger.log(
            logging.ERROR if severity == "critical" else logging.WARNING,
            f"[{check_id}] {message} (count={count})",
        )

    def validate(self) -> List[ValidationIssue]:
        """
        Run all validation checks for this validator.

        Returns:
            List of validation issues found
        """
        raise NotImplementedError("Subclasses must implement validate()")

    def fix(self, check_id: Optional[str] = None) -> int:
        """
        Attempt to auto-fix validation issues.

        Args:
            check_id: Specific check to fix, or None for all fixable issues

        Returns:
            Number of issues fixed
        """
        raise NotImplementedError("Subclasses must implement fix()")
