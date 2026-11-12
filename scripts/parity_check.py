#!/usr/bin/env python3
"""
Live-vs-backtest parity: re-predict a day's finished games through the
backtest path and compare with the live pre-tip predictions.

An engine is flagged when its live spreads differ from the replay by more
than the tolerances below. Normal differences (rosters
are projected live but inferred from appearances in a replay): Nov 2026, 72
games, Phase5 mean |diff| 0.24, Phase3 0.56; legacy models exactly 0.

Prints one line per flagged engine; exit 1 if any, 0 if none, 2 if the check failed.

Usage:
    python scripts/parity_check.py [--date YYYY-MM-DD]   # Eastern date, default yesterday
"""

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.database import DB_PATH, get_db  # noqa: E402
from src.utils import to_eastern_date  # noqa: E402

MIN_GAMES = 3
MAX_MEAN_DIFF = 1.0  # systematic bias, points
MAX_MEAN_ABS_DIFF = 2.0

_spec = importlib.util.spec_from_file_location(
    "backtest", PROJECT_ROOT / "scripts" / "backtest.py"
)
backtest = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(backtest)


def live_spreads(game_date: str) -> dict[str, dict[str, float]]:
    """{predictor: {game_id: spread}} from pre-tip live predictions of finished games."""
    with get_db(DB_PATH) as conn:
        rows = conn.execute(
            """
            SELECT p.game_id, p.predictor, p.prediction_datetime, g.date_time_utc, p.prediction_set
            FROM Predictions p JOIN Games g ON g.game_id = p.game_id
            WHERE g.status = 3 AND g.date_time_utc >= ? AND g.date_time_utc < ?
            """,
            (
                game_date,
                (datetime.fromisoformat(game_date) + timedelta(days=2)).strftime(
                    "%Y-%m-%d"
                ),
            ),
        ).fetchall()
    out: dict[str, dict[str, float]] = {}
    for game_id, predictor, made_at, tip, prediction_set in rows:
        made = datetime.fromisoformat(made_at).replace(tzinfo=ZoneInfo("UTC"))
        if to_eastern_date(tip) != game_date or made >= datetime.fromisoformat(
            tip.replace("Z", "+00:00")
        ):
            continue
        p = json.loads(prediction_set)
        spread = p.get("pred_spread")
        if spread is None and p.get("pred_home_score") is not None:
            spread = p["pred_home_score"] - p["pred_away_score"]
        if spread is not None:
            out.setdefault(predictor, {})[game_id] = float(spread)
    return out


def parity_alerts(game_date: str) -> list[str]:
    alerts = []
    for predictor, live in sorted(live_spreads(game_date).items()):
        replay = backtest.predict(predictor, list(live))
        diffs = [live[g] - replay[g][0] for g in live if g in replay]
        if len(diffs) < MIN_GAMES:
            continue
        mean = sum(diffs) / len(diffs)
        mean_abs = sum(abs(d) for d in diffs) / len(diffs)
        if abs(mean) > MAX_MEAN_DIFF or mean_abs > MAX_MEAN_ABS_DIFF:
            alerts.append(
                f"[WARN] parity/{predictor} {game_date}: live - backtest spread mean {mean:+.2f}, "
                f"mean |diff| {mean_abs:.2f} over {len(diffs)} games"
            )
    return alerts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    yesterday = (
        datetime.now(ZoneInfo("America/New_York")) - timedelta(days=1)
    ).strftime("%Y-%m-%d")
    parser.add_argument("--date", default=yesterday)
    args = parser.parse_args()
    try:
        alerts = parity_alerts(args.date)
    except Exception as e:  # exit 2: the check itself failed (1 means engines flagged)
        print(f"Parity {args.date}: check failed: {e!r}")
        sys.exit(2)
    for line in alerts:
        print(line)
    print(f"Parity {args.date}: {len(alerts)} engine(s) flagged")
    sys.exit(1 if alerts else 0)


if __name__ == "__main__":
    main()
