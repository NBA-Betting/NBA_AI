"""
dashboard.py

Flask blueprint for the prediction results dashboard.
Provides routes to display ATS performance, P&L, rolling metrics,
and per-game results for Phase5 predictions.

Routes:
- /dashboard: Renders the dashboard HTML page
- /dashboard/data: Returns JSON with summary, per-game results, and rolling metrics
"""

import json
import logging

from flask import Blueprint, jsonify, render_template, request

from src.database import get_db
from src.utils import get_eastern_tz, parse_utc_datetime

dashboard = Blueprint("dashboard", __name__)


def _get_available_predictors():
    """Get list of predictors that have predictions in the DB."""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT DISTINCT predictor FROM Predictions ORDER BY predictor"
        ).fetchall()
    return [row[0] for row in rows]

    # First live prediction date — predictions from here onward are leakage-free


LIVE_PREDICTIONS_START = "2026-04-01"


def _fetch_dashboard_data(predictor="Phase5", live_only=False):
    """
    Query completed games with predictions and closing lines.
    Computes ATS results, prediction errors, and rolling metrics.

    Args:
        predictor: Name of predictor to show results for.
        live_only: If True, only show predictions from April 1, 2026 onward
                   (leakage-free live predictions).

    Returns:
        dict with keys: summary, games, rolling, predictor, available_predictors, live_only
    """
    date_filter = (
        f"AND g.date_time_utc >= '{LIVE_PREDICTIONS_START}'" if live_only else ""
    )

    query = f"""
        SELECT
            g.game_id,
            g.date_time_utc,
            g.home_team,
            g.away_team,
            gs.home_score,
            gs.away_score,
            p.prediction_set,
            b.espn_opening_spread,
            b.espn_closing_spread AS closing_spread
        FROM Games g
        JOIN Predictions p ON g.game_id = p.game_id
        JOIN Betting b ON g.game_id = b.game_id
        JOIN GameStates gs ON g.game_id = gs.game_id AND gs.is_final_state = 1
        WHERE g.status = 3
          AND p.predictor = ?
          AND b.espn_closing_spread IS NOT NULL
          {date_filter}
        ORDER BY g.date_time_utc ASC
    """

    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(query, (predictor,))
        rows = cursor.fetchall()

    games = []
    for row in rows:
        (
            game_id,
            date_time_utc,
            home_team,
            away_team,
            home_score,
            away_score,
            prediction_set_json,
            opening_spread,
            closing_spread,
        ) = row

        prediction_set = json.loads(prediction_set_json)
        pred_spread = prediction_set.get("pred_spread")
        pred_home_win_pct = prediction_set.get("pred_home_win_pct")

        # Legacy predictors (Baseline, Tree, etc.) output scores, not spread.
        # Derive spread from predicted scores if pred_spread is missing.
        if pred_spread is None:
            pred_home = prediction_set.get("pred_home_score")
            pred_away = prediction_set.get("pred_away_score")
            if pred_home is not None and pred_away is not None:
                pred_spread = pred_home - pred_away
            else:
                continue

        if closing_spread is None:
            continue

        actual_margin = home_score - away_score

        # ATS logic:
        # pred_spread is model's raw output: positive = home advantage
        # closing_spread is Vegas convention: negative = home favored
        #
        # Our pick: home covers when pred_spread > -closing_spread
        # Home actually covers when actual_margin > -closing_spread
        # Push when |actual_margin - (-closing_spread)| < 0.25
        #   i.e. |actual_margin + closing_spread| < 0.25

        vegas_line = -closing_spread  # convert to home-favored-positive
        margin_vs_line = actual_margin - vegas_line  # >0 means home covered

        # Check for push
        if abs(margin_vs_line) < 0.25:
            ats_result = "P"
        else:
            home_covered = margin_vs_line > 0
            we_picked_home = pred_spread > vegas_line
            ats_result = "W" if (we_picked_home == home_covered) else "L"

        # Prediction error (our predicted margin vs actual margin)
        prediction_error = pred_spread - actual_margin

        # Extract date for display. NBA schedule dates are Eastern, and an
        # evening tip-off has already rolled into the next day in UTC, so the
        # raw timestamp would date most games a day late.
        game_date = (
            parse_utc_datetime(date_time_utc)
            .astimezone(get_eastern_tz())
            .strftime("%Y-%m-%d")
            if date_time_utc
            else ""
        )

        games.append(
            {
                "game_id": game_id,
                "date": game_date,
                "home": home_team,
                "away": away_team,
                "actual_margin": actual_margin,
                "pred_spread": round(pred_spread, 1),
                "pred_home_win_pct": (
                    round(pred_home_win_pct, 3)
                    if pred_home_win_pct is not None
                    else None
                ),
                "opening_spread": (
                    round(opening_spread, 1) if opening_spread is not None else None
                ),
                "closing_spread": round(closing_spread, 1),
                "ats_result": ats_result,
                "error": round(prediction_error, 1),
                "abs_error": round(abs(prediction_error), 1),
            }
        )

    # Compute summary
    ats_wins = sum(1 for g in games if g["ats_result"] == "W")
    ats_losses = sum(1 for g in games if g["ats_result"] == "L")
    ats_pushes = sum(1 for g in games if g["ats_result"] == "P")
    n_decided = ats_wins + ats_losses
    ats_pct = (ats_wins / n_decided * 100) if n_decided > 0 else 0.0

    abs_errors = [g["abs_error"] for g in games]
    mae = sum(abs_errors) / len(abs_errors) if abs_errors else 0.0

    # ROI: flat $100 bets at -110 odds
    # Win pays $90.91 profit, loss costs $100
    win_payout = 100 / 1.10  # ~90.91
    total_profit = (ats_wins * win_payout) - (ats_losses * 100)
    total_wagered = n_decided * 100
    roi = (total_profit / total_wagered * 100) if total_wagered > 0 else 0.0

    summary = {
        "ats_wins": ats_wins,
        "ats_losses": ats_losses,
        "ats_pushes": ats_pushes,
        "ats_pct": round(ats_pct, 1),
        "roi": round(roi, 1),
        "mae": round(mae, 2),
        "n_games": len(games),
    }

    # Compute rolling metrics
    window = 20
    rolling_ats = []
    rolling_mae = []
    cumulative_pl = []
    running_pl = 0.0

    for i, g in enumerate(games):
        # Cumulative P&L
        if g["ats_result"] == "W":
            running_pl += win_payout
        elif g["ats_result"] == "L":
            running_pl -= 100

        cumulative_pl.append(
            {
                "date": g["date"],
                "pl": round(running_pl, 2),
                "game_index": i,
            }
        )

        # Rolling ATS % (window of last N decided games)
        if i >= window - 1:
            window_games = games[i - window + 1 : i + 1]
            window_decided = [
                wg for wg in window_games if wg["ats_result"] in ("W", "L")
            ]
            window_wins = sum(1 for wg in window_decided if wg["ats_result"] == "W")
            window_ats_pct = (
                (window_wins / len(window_decided) * 100) if window_decided else 0
            )

            window_errors = [wg["abs_error"] for wg in window_games]
            window_mae = sum(window_errors) / len(window_errors)

            rolling_ats.append(
                {
                    "date": g["date"],
                    "ats_pct": round(window_ats_pct, 1),
                    "game_index": i,
                }
            )
            rolling_mae.append(
                {
                    "date": g["date"],
                    "mae": round(window_mae, 2),
                    "game_index": i,
                }
            )

    rolling = {
        "ats": rolling_ats,
        "mae": rolling_mae,
        "cumulative_pl": cumulative_pl,
    }

    return {
        "summary": summary,
        "games": games,
        "rolling": rolling,
        "predictor": predictor,
        "live_only": live_only,
        "available_predictors": _get_available_predictors(),
    }


@dashboard.route("/dashboard")
def dashboard_page():
    """Render the dashboard HTML page."""
    predictor = request.args.get("predictor", "Phase5")
    live_only = request.args.get("live_only", "true").lower() == "true"
    available = _get_available_predictors()
    return render_template(
        "dashboard.html",
        predictor=predictor,
        available_predictors=available,
        live_only=live_only,
    )


@dashboard.route("/dashboard/data")
def dashboard_data():
    """Return dashboard data as JSON. Accepts ?predictor= and ?live_only= params."""
    try:
        predictor = request.args.get("predictor", "Phase5")
        live_only = request.args.get("live_only", "true").lower() == "true"
        data = _fetch_dashboard_data(predictor=predictor, live_only=live_only)
        return jsonify(data)
    except Exception:
        logging.exception("Error fetching dashboard data")
        return jsonify({"error": "Internal server error"}), 500
