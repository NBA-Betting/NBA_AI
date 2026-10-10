"""
dashboard.py

Flask blueprint for the prediction results dashboard.

Results are shown per record, each an honest set of past predictions:
- Live: predictions stored before tip-off, one record per season since live
  tracking began (LIVE_PREDICTIONS_START).
- Backtest: a season replayed through each predictor's real code with
  pre-game inputs only (scripts/backtest.py --output), saved under
  models/backtests/. Older stored predictions were made by models trained on
  those games, so they aren't shown.

Every prediction is graded against the closing line (ATS at -110) and the
actual margin. The model comparison scores all predictors on the games they
all predicted.

Routes:
- /dashboard: Renders the dashboard page (filled in by static/js/dashboard.js)
- /dashboard/data: JSON for one record and predictor
"""

import json
import logging
from pathlib import Path

from flask import Blueprint, current_app, jsonify, render_template, request

from src.database import get_db
from src.database_updater.betting import CLOSING_SPREAD_SQL
from src.utils import to_eastern_date

dashboard = Blueprint("dashboard", __name__)

# First live prediction date — predictions from here onward are leakage-free
LIVE_PREDICTIONS_START = "2026-04-01"
BACKTESTS_DIR = Path(__file__).resolve().parents[2] / "models" / "backtests"

WIN_PAYOUT = 100 / 1.10  # profit on a winning $100 bet at -110


def _season_label(season):
    """'2025-2026' -> '2025-26'."""
    return f"{season[:4]}-{season[7:]}"


def _live_seasons():
    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT DISTINCT g.season FROM Predictions p
            JOIN Games g ON g.game_id = p.game_id
            WHERE g.status = 3 AND g.date_time_utc >= ?
            ORDER BY g.season DESC
            """,
            (LIVE_PREDICTIONS_START,),
        ).fetchall()
    return [row[0] for row in rows]


def list_records():
    """Available records, newest live season first, then backtests."""
    records = [
        {"id": f"live-{s}", "label": f"{_season_label(s)} Live"}
        for s in _live_seasons()
    ]
    for path in sorted(BACKTESTS_DIR.glob("*.json"), reverse=True):
        data = json.loads(path.read_text())
        records.append(
            {
                "id": f"backtest-{path.stem}",
                "label": data["label"],
                "note": data.get("note", ""),
            }
        )
    return records


def load_live_record(season=None):
    """
    Live predictions (made before tip-off) for finished games with a closing
    line, for one season or all since live tracking began.

    Returns (games, predictions): games is {game_id: {date, home, away,
    home_score, away_score, closing_spread}}; predictions is
    {predictor: {game_id: pred_spread}} (positive = home favored).
    """
    # prediction_datetime is "YYYY-MM-DD HH:MM:SS" (UTC); date_time_utc is
    # "YYYY-MM-DDTHH:MM:SSZ", so normalize before comparing
    query = f"""
        SELECT g.game_id, g.date_time_utc, g.home_team, g.away_team,
               gs.home_score, gs.away_score, {CLOSING_SPREAD_SQL},
               p.predictor, p.prediction_set
        FROM Games g
        JOIN Predictions p ON g.game_id = p.game_id
        JOIN Betting b ON g.game_id = b.game_id
        JOIN GameStates gs ON g.game_id = gs.game_id AND gs.is_final_state = 1
        WHERE g.status = 3
          AND g.date_time_utc >= ?
          AND p.prediction_datetime
              < replace(replace(g.date_time_utc, 'T', ' '), 'Z', '')
          AND {CLOSING_SPREAD_SQL} IS NOT NULL
          {"AND g.season = ?" if season else ""}
        ORDER BY g.date_time_utc
    """
    params = [LIVE_PREDICTIONS_START] + ([season] if season else [])
    with get_db() as conn:
        rows = conn.execute(query, params).fetchall()

    games, predictions = {}, {}
    for gid, dt, home, away, home_score, away_score, closing, predictor, pset in rows:
        prediction_set = json.loads(pset)
        spread = prediction_set.get("pred_spread")
        # Legacy predictors output scores, not a spread
        if spread is None:
            home_pts = prediction_set.get("pred_home_score")
            away_pts = prediction_set.get("pred_away_score")
            if home_pts is None or away_pts is None:
                continue
            spread = home_pts - away_pts
        games[gid] = {
            # NBA schedule dates are Eastern; an evening tip-off is already
            # the next day in UTC
            "date": to_eastern_date(dt),
            "home": home,
            "away": away,
            "home_score": home_score,
            "away_score": away_score,
            "closing_spread": closing,
        }
        predictions.setdefault(predictor, {})[gid] = spread
    return games, predictions


def load_backtest_record(name):
    """A saved backtest, in the same form as load_live_record."""
    data = json.loads((BACKTESTS_DIR / f"{name}.json").read_text())
    games = {
        gid: g for gid, g in data["games"].items() if g["closing_spread"] is not None
    }
    predictions = {
        predictor: {gid: p[0] for gid, p in by_game.items() if gid in games}
        for predictor, by_game in data["predictions"].items()
    }
    return games, predictions


def ats_result(pred_spread, closing_spread, margin):
    """
    "W", "L" or "P" for a predicted home margin against the closing line
    (Vegas convention, negative = home favored) given the final home margin.
    The pick is home when the prediction beats the line; a final margin
    within 0.25 of the line is a push.
    """
    margin_vs_line = margin + closing_spread  # > 0 means home covered
    if abs(margin_vs_line) < 0.25:
        return "P"
    return "W" if (pred_spread > -closing_spread) == (margin_vs_line > 0) else "L"


def grade_games(games, spreads):
    """
    Grade a predictor's spreads against the closing line (ats_result) and
    the final margin, in date order.
    """
    graded = []
    for gid in sorted(spreads, key=lambda g: (games[g]["date"], g)):
        g = games[gid]
        spread = spreads[gid]
        margin = g["home_score"] - g["away_score"]
        result = ats_result(spread, g["closing_spread"], margin)
        margin_vs_line = margin + g["closing_spread"]
        graded.append(
            {
                "game_id": gid,
                "date": g["date"],
                "home": g["home"],
                "away": g["away"],
                "actual_margin": margin,
                "pred_spread": round(spread, 1),
                "closing_spread": round(g["closing_spread"], 1),
                "ats_result": result,
                "home_covered": None if result == "P" else margin_vs_line > 0,
                "abs_error": round(abs(spread - margin), 1),
                "vegas_abs_error": abs(g["closing_spread"] + margin),
            }
        )
    return graded


def summarize(graded):
    """ATS record, ROI at -110 and spread MAE (with baselines) for graded games."""
    wins = sum(g["ats_result"] == "W" for g in graded)
    losses = sum(g["ats_result"] == "L" for g in graded)
    decided = wins + losses
    n = len(graded)

    def pct(x, of):
        return round(x / of * 100, 1) if of else 0.0

    return {
        "n_games": n,
        "ats_wins": wins,
        "ats_losses": losses,
        "ats_pushes": n - decided,
        "ats_pct": pct(wins, decided),
        "always_home_ats_pct": pct(
            sum(bool(g["home_covered"]) for g in graded), decided
        ),
        "profit": round(wins * WIN_PAYOUT - losses * 100, 2),
        "roi": pct(wins * WIN_PAYOUT - losses * 100, decided * 100),
        "mae": round(sum(g["abs_error"] for g in graded) / n, 2) if n else 0.0,
        "vegas_mae": (
            round(sum(g["vegas_abs_error"] for g in graded) / n, 2) if n else 0.0
        ),
    }


def _cumulative_pl(graded):
    running, points = 0.0, []
    for g in graded:
        running += (
            WIN_PAYOUT
            if g["ats_result"] == "W"
            else -100 if g["ats_result"] == "L" else 0
        )
        points.append({"date": g["date"], "pl": round(running, 2)})
    return points


def fetch_dashboard_data(record_id=None, predictor=None):
    """
    Everything the dashboard shows for one record (default: the first
    listed) and predictor (default: the app's, else the first available).
    """
    records = list_records()
    record = next((r for r in records if r["id"] == record_id), None) or (
        records[0] if records else None
    )
    games, predictions = {}, {}
    if record:
        kind, name = record["id"].split("-", 1)
        games, predictions = (
            load_live_record(name) if kind == "live" else load_backtest_record(name)
        )
    if not predictions:
        return {"records": records, "record": record and record["id"], "summary": None}

    predictors = sorted(predictions)
    if predictor not in predictions:
        default = current_app.config.get("PREDICTOR")
        predictor = default if default in predictions else predictors[0]

    # The comparison uses only games every predictor has, so it's like for like
    common = set.intersection(*(set(p) for p in predictions.values()))
    comparison = [
        {
            "predictor": p,
            **summarize(
                grade_games(games, {g: s for g, s in spreads.items() if g in common})
            ),
        }
        for p, spreads in predictions.items()
    ]
    comparison.sort(key=lambda row: row["mae"])

    graded = grade_games(games, predictions[predictor])
    return {
        "records": records,
        "record": record["id"],
        "is_backtest": kind == "backtest",
        "predictors": predictors,
        "predictor": predictor,
        "summary": summarize(graded),
        "comparison": {"n_games": len(common), "rows": comparison},
        "cumulative_pl": _cumulative_pl(graded),
        "games": graded[::-1],  # most recent first
    }


@dashboard.route("/dashboard")
def dashboard_page():
    """Render the dashboard page; its data comes from /dashboard/data."""
    return render_template("dashboard.html")


@dashboard.route("/dashboard/data")
def dashboard_data():
    """Dashboard data as JSON. Accepts ?record= and ?predictor=."""
    try:
        return jsonify(
            fetch_dashboard_data(
                request.args.get("record"), request.args.get("predictor")
            )
        )
    except Exception:
        logging.exception("Error fetching dashboard data")
        return jsonify({"error": "Internal server error"}), 500
