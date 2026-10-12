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


def test_ats_result_against_closing_line():
    ats = dashboard_module.ats_result
    # Home favored by 5.5 (closing -5.5): won by 8 -> covered
    assert ats(pred_spread=7.0, closing_spread=-5.5, margin=8) == "W"  # picked home
    assert ats(pred_spread=3.0, closing_spread=-5.5, margin=8) == "L"  # picked away
    # Home won by 3, line home -5.5 -> away covered
    assert ats(pred_spread=3.0, closing_spread=-5.5, margin=3) == "W"
    # Margin on the line is a push, whatever the pick
    assert ats(pred_spread=10.0, closing_spread=-4.0, margin=4) == "P"


def _game(margin, closing=-3.0):
    return {"date": "2026-01-01", "home": "BOS", "away": "NYK",
            "home_score": 100 + margin, "away_score": 100, "closing_spread": closing}


def test_comparison_uses_common_games_and_drops_sparse_predictors():
    games = {f"g{i}": _game(margin=i) for i in range(10)}
    predictions = {
        "A": {g: 0.0 for g in games},
        "B": {g: 0.0 for g in list(games)[:8]},
        "Sparse": {"g0": 0.0, "g1": 0.0},  # under half of A's games
    }
    rows, n_common, excluded = dashboard_module.compare_predictors(games, predictions)
    assert excluded == ["Sparse"]
    assert n_common == 8
    assert {r["predictor"] for r in rows} == {"A", "B"}
    assert all(r["n_games"] == 8 for r in rows)


def test_saved_backtest_loads_as_a_record(tmp_path, monkeypatch):
    import argparse
    import importlib.util

    spec = importlib.util.spec_from_file_location("backtest", "scripts/backtest.py")
    backtest = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(backtest)

    games = [
        {"game_id": "g1", **_game(margin=5, closing=-2.5)},
        {"game_id": "g2", **_game(margin=-4, closing=None)},  # no line: not graded
    ]
    args = argparse.Namespace(
        output=tmp_path / "2025-2026.json", season="2025-2026", start=None,
        season_type="Regular Season", note="n",
    )
    backtest.save(args, games, {"Tree": {"g1": (4.0, 0.6), "g2": (-1.0, 0.45)}})

    monkeypatch.setattr(dashboard_module, "BACKTESTS_DIR", tmp_path)
    loaded_games, predictions = dashboard_module.load_backtest_record("2025-2026")
    assert set(loaded_games) == {"g1"}
    assert predictions == {"Tree": {"g1": 4.0}}
    monkeypatch.setattr(dashboard_module, "_live_seasons", lambda: [])
    assert dashboard_module.list_records() == [
        {"id": "backtest-2025-2026", "label": "2025-26 Backtest", "note": "n",
         "season_type": "Regular Season"}
    ]
