#!/usr/bin/env python3
"""
Backtest predictors on past games through their real prediction code.

Phase5 and Phase3 run in historical mode (pre-game rosters inferred from
appearances, features and L1 states as of each tip-off); legacy predictors use
their stored pre-game features. Each model is compared on the same games with
the Vegas closing line and a naive baseline (the previous season's mean home
margin).

With --output, each predictor's per-game predictions are saved as JSON; the
results dashboard shows such a file (kept in models/backtests/) as a backtest
record next to the live ones.

Usage:
    python scripts/backtest.py --predictors Phase5 Tree --season 2025-2026
    python scripts/backtest.py --predictors Phase5 --start 2026-01-01 --end 2026-02-28
    python scripts/backtest.py --predictors Phase5 Phase3 Ensemble Tree MLP Linear Baseline \
        --season 2025-2026 --output models/backtests/2025-2026.json

Ensemble in a backtest replays its components itself (it would otherwise
combine whatever is stored in the Predictions table).
"""

import argparse
import json
import logging
import math
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.database import DB_PATH, get_db  # noqa: E402
from src.database_updater.betting import CLOSING_SPREAD_SQL  # noqa: E402
from src.predictions.prediction_manager import (  # noqa: E402
    _drop_implausible_predictions,
    create_predictor,
)
from src.utils import to_eastern_date  # noqa: E402


def load_games(args) -> list[dict]:
    query = f"""
        SELECT g.game_id, g.date_time_utc, g.season, g.home_team, g.away_team,
               gs.home_score, gs.away_score, {CLOSING_SPREAD_SQL}
        FROM Games g
        JOIN GameStates gs ON gs.game_id = g.game_id AND gs.is_final_state = 1
        LEFT JOIN Betting b ON b.game_id = g.game_id
        WHERE g.status = 3 AND g.season_type = ?
    """
    params = [args.season_type]
    if args.season:
        query += " AND g.season = ?"
        params.append(args.season)
    # --start/--end are Eastern (schedule) dates; evening games fall on the
    # next UTC day, so widen the UTC bounds and filter on the Eastern date
    if args.start:
        query += " AND g.date_time_utc >= ?"
        params.append(args.start)
    if args.end:
        query += " AND g.date_time_utc < ?"
        params.append(
            (datetime.strptime(args.end, "%Y-%m-%d") + timedelta(days=2)).strftime(
                "%Y-%m-%d"
            )
        )
    with get_db(DB_PATH) as conn:
        rows = [
            r
            for r in conn.execute(query + " ORDER BY g.date_time_utc", params)
            if (not args.start or to_eastern_date(r[1]) >= args.start)
            and (not args.end or to_eastern_date(r[1]) <= args.end)
        ]
        if not rows:
            raise SystemExit("No finished games in that range")
        seasons = sorted({r[2] for r in rows})
        prev = f"{int(seasons[0][:4]) - 1}-{seasons[0][:4]}"
        naive = conn.execute(
            """
            SELECT AVG(gs.home_score - gs.away_score) FROM Games g
            JOIN GameStates gs ON gs.game_id = g.game_id AND gs.is_final_state = 1
            WHERE g.season = ? AND g.season_type = 'Regular Season' AND g.status = 3
            """,
            (prev,),
        ).fetchone()[0]
    return [
        {
            "game_id": gid,
            "date": to_eastern_date(dt),
            "month": to_eastern_date(dt)[:7],
            "home": home,
            "away": away,
            "home_score": home_score,
            "away_score": away_score,
            "closing_spread": line,
            "margin": float(home_score - away_score),
            # Closing spreads are home lines (negative = home favored)
            "vegas": -float(line) if line is not None else math.nan,
            "naive": float(naive or 0.0),
        }
        for gid, dt, _, home, away, home_score, away_score, line in rows
    ]


def predict(name: str, game_ids: list[str]) -> dict[str, tuple[float, float | None]]:
    """{game_id: (spread, home win prob)} from the predictor's own code."""
    preds = _drop_implausible_predictions(
        create_predictor(name, historical=True).make_pre_game_predictions(game_ids), name
    )
    out = {}
    for gid, p in preds.items():
        spread = p.get("pred_spread")
        if spread is None and p.get("pred_home_score") is not None:
            spread = p["pred_home_score"] - p["pred_away_score"]
        if spread is not None:
            out[gid] = (float(spread), p.get("pred_home_win_pct"))
    return out


def summarize(name, games, preds, ref_errors=None, rng=None):
    rows = [g for g in games if g["game_id"] in preds]
    spread = np.array([preds[g["game_id"]][0] for g in rows])
    margin = np.array([g["margin"] for g in rows])
    vegas = np.array([g["vegas"] for g in rows])
    naive = np.array([g["naive"] for g in rows])
    err = np.abs(spread - margin)
    lined = ~np.isnan(vegas)

    covered = margin[lined] - vegas[lined]
    decided = np.abs(covered) >= 0.25
    picks_home = spread[lined] > vegas[lined]
    wins = int(((picks_home == (covered > 0)) & decided).sum())
    losses = int(decided.sum()) - wins

    print(f"\n{name}: {len(rows)}/{len(games)} games predicted")
    print(
        f"  spread MAE {err.mean():.3f} | Vegas {np.abs(vegas[lined] - margin[lined]).mean():.3f}"
        f" (model on same games {err[lined].mean():.3f}) | naive {np.abs(naive - margin).mean():.3f}"
    )
    print(
        f"  vs closing line {wins}-{losses} ({wins / max(wins + losses, 1):.1%})"
        f" | prediction std {spread.std():.2f} (Vegas {np.nanstd(vegas):.2f})"
    )
    probs = [preds[g["game_id"]][1] for g in rows]
    if all(p is not None for p in probs):
        p = np.clip(np.array(probs, dtype=float), 1e-6, 1 - 1e-6)
        y = (margin > 0).astype(float)
        print(
            f"  win prob Brier {np.mean((p - y) ** 2):.4f}"
            f" | log-loss {-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)):.4f}"
        )
    months = sorted({g["month"] for g in rows})
    by_month = []
    for m in months:
        sel = np.array([g["month"] == m for g in rows])
        by_month.append(
            f"{m} {err[sel].mean():.2f}/{np.abs(vegas[sel & lined] - margin[sel & lined]).mean():.2f}"
        )
    print("  MAE model/Vegas by month: " + ", ".join(by_month))

    errors = {g["game_id"]: e for g, e in zip(rows, err)}
    if ref_errors:
        common = sorted(errors.keys() & ref_errors.keys())
        diff = np.array([errors[g] - ref_errors[g] for g in common])
        boot = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(2000)]
        print(
            f"  vs reference on {len(common)} common games: MAE {diff.mean():+.3f}"
            f" (95% CI {np.percentile(boot, 2.5):+.3f} to {np.percentile(boot, 97.5):+.3f})"
        )
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--predictors", nargs="+", required=True)
    parser.add_argument("--season", help='e.g. "2025-2026"')
    parser.add_argument("--start", help="YYYY-MM-DD")
    parser.add_argument("--end", help="YYYY-MM-DD (inclusive)")
    parser.add_argument("--season-type", default="Regular Season")
    parser.add_argument("--output", type=Path, help="Save predictions as JSON")
    parser.add_argument("--note", default="", help="Caveat shown with the saved record")
    args = parser.parse_args()
    if not (args.season or args.start):
        parser.error("give --season or --start/--end")
    logging.basicConfig(level=logging.WARNING)

    games = load_games(args)
    print(f"{len(games)} games")
    rng = np.random.default_rng(0)
    ref_errors = None
    saved = {}
    for i, name in enumerate(args.predictors):
        preds = predict(name, [g["game_id"] for g in games])
        errors = summarize(name, games, preds, ref_errors, rng)
        if i == 0:
            ref_errors = errors
            print("  (reference for the comparisons below)")
        if args.output:
            saved[name] = preds
            save(args, games, saved)


def save(args, games, preds):
    """Write the games and each predictor's {game_id: [spread, win prob]}."""
    keys = ("date", "home", "away", "home_score", "away_score", "closing_spread")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "label": (
                    f"{args.season[:4]}-{args.season[7:]}" if args.season else args.start
                )
                + " Backtest",
                "note": args.note,
                "season_type": args.season_type,
                "generated_at": time.strftime("%Y-%m-%d"),
                "games": {g["game_id"]: {k: g[k] for k in keys} for g in games},
                "predictions": {
                    name: {
                        gid: [round(spread, 2), None if p is None else round(p, 4)]
                        for gid, (spread, p) in p_by_game.items()
                    }
                    for name, p_by_game in preds.items()
                },
            }
        )
    )


if __name__ == "__main__":
    main()
