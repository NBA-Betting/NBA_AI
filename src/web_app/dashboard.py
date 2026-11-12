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

Predictions are scored first on accuracy against Vegas on the same games:
spread MAE minus the closing line's MAE, and win-probability Brier score
against the no-vig closing moneyline. Against the spread (ATS at -110) is
shown too, with a 95% range: every model, like the closing line itself, sits
near 50%, so short ATS runs are mostly noise. The model comparison scores all
predictors on the games they all predicted.

Routes:
- /dashboard: Renders the dashboard page (filled in by static/js/dashboard.js)
- /dashboard/data: JSON for one record and predictor
"""

import json
import logging
import math
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


# Finished games with a closing line and a prediction made before tip-off.
# prediction_datetime is "YYYY-MM-DD HH:MM:SS" (UTC); date_time_utc is
# "YYYY-MM-DDTHH:MM:SSZ", so normalize before comparing
LIVE_GAMES_SQL = f"""
    FROM Games g
    JOIN Predictions p ON g.game_id = p.game_id
    JOIN Betting b ON g.game_id = b.game_id
    JOIN GameStates gs ON g.game_id = gs.game_id AND gs.is_final_state = 1
    WHERE g.status = 3
      AND g.date_time_utc >= ?
      AND p.prediction_datetime
          < replace(replace(g.date_time_utc, 'T', ' '), 'Z', '')
      AND {CLOSING_SPREAD_SQL} IS NOT NULL
"""


def _live_seasons():
    """Seasons with a graded live game, newest first."""
    with get_db() as conn:
        rows = conn.execute(
            f"SELECT DISTINCT g.season {LIVE_GAMES_SQL} ORDER BY g.season DESC",
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
                "season_type": data.get("season_type", "Regular Season"),
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
    query = f"""
        SELECT g.game_id, g.date_time_utc, g.home_team, g.away_team,
               gs.home_score, gs.away_score, {CLOSING_SPREAD_SQL},
               p.predictor, p.prediction_set
        {LIVE_GAMES_SQL}
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
        game = games.setdefault(
            gid,
            {
                # NBA schedule dates are Eastern; an evening tip-off is already
                # the next day in UTC
                "date": to_eastern_date(dt),
                "home": home,
                "away": away,
                "home_score": home_score,
                "away_score": away_score,
                "closing_spread": closing,
                "win_probs": {},
            },
        )
        if prediction_set.get("pred_home_win_pct") is not None:
            game["win_probs"][predictor] = prediction_set["pred_home_win_pct"]
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
    for predictor, by_game in data["predictions"].items():
        for gid, p in by_game.items():
            if gid in games and len(p) > 1 and p[1] is not None:
                games[gid].setdefault("win_probs", {})[predictor] = p[1]
    return games, predictions


def _no_vig_home_prob(ml_home, ml_away):
    """Home win probability from American moneylines, vig removed."""

    def implied(ml):
        return 100 / (ml + 100) if ml > 0 else -ml / (-ml + 100)

    home, away = implied(ml_home), implied(ml_away)
    return home / (home + away)


def attach_vegas_probs(games):
    """Add each game's no-vig closing-moneyline home win probability (ESPN
    closing moneylines; Covers has none) as games[gid]["vegas_prob"]."""
    if not games:
        return
    ids = list(games)
    with get_db() as conn:
        rows = conn.execute(
            f"""SELECT game_id, espn_closing_ml_home, espn_closing_ml_away FROM Betting
            WHERE game_id IN ({",".join("?" * len(ids))})
              AND espn_closing_ml_home IS NOT NULL AND espn_closing_ml_away IS NOT NULL""",
            ids,
        ).fetchall()
    for gid, ml_home, ml_away in rows:
        games[gid]["vegas_prob"] = _no_vig_home_prob(ml_home, ml_away)


def ats_result(pred_spread, closing_spread, margin):
    """
    "W", "L" or "P" for a predicted home margin against the closing line
    (Vegas convention, negative = home favored) given the final home margin.
    The pick is home when the prediction beats the line; a final margin
    within 0.25 of the line is a push, and so is a prediction equal to the
    line (no pick, so no bet).
    """
    margin_vs_line = margin + closing_spread  # > 0 means home covered
    if abs(margin_vs_line) < 0.25 or pred_spread == -closing_spread:
        return "P"
    return "W" if (pred_spread > -closing_spread) == (margin_vs_line > 0) else "L"


def grade_games(games, spreads, predictor=None):
    """
    Grade a predictor's spreads against the closing line (ats_result) and
    the final margin, in date order, with its win probability and Vegas's
    where known.
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
                "win_prob": g.get("win_probs", {}).get(predictor),
                "vegas_prob": g.get("vegas_prob"),
                "home_won": margin > 0,
            }
        )
    return graded


def summarize(graded):
    """Accuracy vs Vegas (spread MAE, win-prob Brier), plus the ATS record with
    its 95% range and ROI at -110, for graded games."""
    wins = sum(g["ats_result"] == "W" for g in graded)
    losses = sum(g["ats_result"] == "L" for g in graded)
    decided = wins + losses
    n = len(graded)

    def pct(x, of):
        return round(x / of * 100, 1) if of else 0.0

    mae = sum(g["abs_error"] for g in graded) / n if n else 0.0
    vegas_mae = sum(g["vegas_abs_error"] for g in graded) / n if n else 0.0
    # Brier on the games with both a model and a Vegas probability
    probs = [g for g in graded if g["win_prob"] is not None and g["vegas_prob"] is not None]

    def brier(key):
        if not probs:
            return None
        return round(sum((g[key] - g["home_won"]) ** 2 for g in probs) / len(probs), 4)

    return {
        "mae_vs_vegas": round(mae - vegas_mae, 2),
        "brier": brier("win_prob"),
        "vegas_brier": brier("vegas_prob"),
        "n_brier": len(probs),
        # Half-width of the 95% range of the ATS win rate, percentage points
        "ats_range": (
            round(1.96 * math.sqrt(wins * losses / decided**3) * 100, 1)
            if decided
            else 0.0
        ),
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
        "mae": round(mae, 2),
        "vegas_mae": round(vegas_mae, 2),
    }


# A predictor with fewer games than this share of the best-covered one is
# left out of the comparison, so it can't shrink everyone's common games
MIN_COMPARISON_COVERAGE = 0.5


def compare_predictors(games, predictions):
    """
    Score predictors on the games they all predicted, best MAE first.

    Returns (rows, n_common_games, excluded predictor names).
    """
    most = max(len(spreads) for spreads in predictions.values())
    included = {
        p: spreads
        for p, spreads in predictions.items()
        if len(spreads) >= MIN_COMPARISON_COVERAGE * most
    }
    common = set.intersection(*(set(spreads) for spreads in included.values()))
    rows = [
        {
            "predictor": p,
            **summarize(
                grade_games(
                    games, {g: s for g, s in spreads.items() if g in common}, p
                )
            ),
        }
        for p, spreads in included.items()
    ]
    rows.sort(key=lambda row: row["mae"])
    return rows, len(common), sorted(predictions.keys() - included.keys())


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
    attach_vegas_probs(games)

    predictors = sorted(predictions)
    if predictor not in predictions:
        default = current_app.config.get("PREDICTOR")
        predictor = default if default in predictions else predictors[0]

    comparison, n_common, excluded = compare_predictors(games, predictions)
    graded = grade_games(games, predictions[predictor], predictor)
    return {
        "records": records,
        "record": record["id"],
        "is_backtest": kind == "backtest",
        "predictors": predictors,
        "predictor": predictor,
        "summary": summarize(graded),
        "comparison": {"n_games": n_common, "rows": comparison, "excluded": excluded},
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
