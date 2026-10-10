"""Tests for dashboard data formatting."""

from src.web_app import dashboard as dashboard_module


class _DashboardConnection:
    def __init__(self):
        self.rows = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def cursor(self):
        return self

    def execute(self, query, params=None):
        self.rows = [
            (
                "0022500579",
                "2026-04-16T00:00:00Z",
                "DET",
                "PHX",
                110,
                105,
                -3.5,
                "Phase5",
                '{"pred_spread": 4.0, "pred_home_win_pct": 0.6}',
            )
        ]
        return self

    def fetchall(self):
        return self.rows


def test_dashboard_dates_are_displayed_in_eastern_time(monkeypatch):
    """An evening Eastern game must not be displayed under the next UTC day."""
    monkeypatch.setattr(dashboard_module, "get_db", _DashboardConnection)

    games, _ = dashboard_module.load_live_record("2025-2026")

    assert games["0022500579"]["date"] == "2026-04-15"
