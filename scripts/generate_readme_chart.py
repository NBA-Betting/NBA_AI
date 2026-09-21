#!/usr/bin/env python3
"""Generate prediction model performance chart for README.

Queries the database for live predictions (April 1, 2026 onward) and computes
ATS win rate, ROI at -110, and Spread MAE for all predictors. Produces a
grouped bar chart saved to docs/images/predictor_performance.png.

Usage:
    python scripts/generate_readme_chart.py
"""

import json
import sqlite3
import sys
from pathlib import Path

import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Project setup — resolve paths relative to repo root
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.database import get_db  # noqa: E402

LIVE_START = "2026-04-01"
OUTPUT_PATH = REPO_ROOT / "docs" / "images" / "predictor_performance.png"

# Breakeven ATS% at -110 odds
BREAKEVEN_ATS = 52.4

# All predictors in display order
PREDICTORS = ["Phase5", "Phase3", "Ensemble", "MLP", "Tree", "Linear", "Baseline"]

# Display-friendly labels
LABELS = {
    "Phase5": "Phase 5\n(Hierarchical)",
    "Phase3": "Phase 3\n(Transformer)",
    "Ensemble": "Ensemble",
    "MLP": "MLP",
    "Tree": "Tree",
    "Linear": "Linear",
    "Baseline": "Baseline",
}

# Color palette — distinct, colorblind-friendly
COLORS = {
    "Phase5": "#2563eb",  # blue
    "Phase3": "#7c3aed",  # purple
    "Ensemble": "#0891b2",  # teal
    "MLP": "#ea580c",  # orange
    "Tree": "#16a34a",  # green
    "Linear": "#d97706",  # amber
    "Baseline": "#6b7280",  # gray
}


def compute_predictor_metrics(predictor: str, conn: sqlite3.Connection) -> dict | None:
    """
    Compute ATS win rate, ROI, and spread MAE for a single predictor.

    Uses the same logic as src/web_app/dashboard.py:_fetch_dashboard_data().
    """
    query = """
        SELECT
            g.game_id,
            gs.home_score,
            gs.away_score,
            p.prediction_set,
            COALESCE(b.espn_closing_spread, b.covers_closing_spread) AS closing_spread
        FROM Games g
        JOIN Predictions p ON g.game_id = p.game_id
        JOIN Betting b ON g.game_id = b.game_id
        JOIN GameStates gs ON g.game_id = gs.game_id AND gs.is_final_state = 1
        WHERE g.status = 3
          AND p.predictor = ?
          AND COALESCE(b.espn_closing_spread, b.covers_closing_spread) IS NOT NULL
          AND g.date_time_utc >= ?
        ORDER BY g.date_time_utc ASC
    """

    rows = conn.execute(query, (predictor, LIVE_START)).fetchall()

    ats_wins = 0
    ats_losses = 0
    ats_pushes = 0
    abs_errors = []

    for game_id, home_score, away_score, prediction_set_json, closing_spread in rows:
        prediction_set = json.loads(prediction_set_json)
        pred_spread = prediction_set.get("pred_spread")

        # Legacy predictors output scores, not spread — derive spread.
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

        # ATS logic (mirrors dashboard.py exactly):
        # pred_spread: positive = home advantage (model convention)
        # closing_spread: negative = home favored (Vegas convention)
        #
        # Our pick: home covers when pred_spread > -closing_spread
        # Home actually covers when actual_margin > -closing_spread
        vegas_line = -closing_spread  # convert to home-favored-positive
        margin_vs_line = actual_margin - vegas_line  # >0 means home covered

        # Push check
        if abs(margin_vs_line) < 0.25:
            ats_pushes += 1
        else:
            home_covered = margin_vs_line > 0
            we_picked_home = pred_spread > vegas_line
            if we_picked_home == home_covered:
                ats_wins += 1
            else:
                ats_losses += 1

        # Spread MAE
        abs_errors.append(abs(pred_spread - actual_margin))

    n_decided = ats_wins + ats_losses
    if n_decided == 0:
        return None

    ats_pct = ats_wins / n_decided * 100

    # ROI: flat $100 bets at -110 odds
    win_payout = 100 / 1.10  # ~$90.91
    total_profit = (ats_wins * win_payout) - (ats_losses * 100)
    total_wagered = n_decided * 100
    roi = total_profit / total_wagered * 100

    mae = sum(abs_errors) / len(abs_errors) if abs_errors else 0.0

    return {
        "predictor": predictor,
        "ats_wins": ats_wins,
        "ats_losses": ats_losses,
        "ats_pushes": ats_pushes,
        "n_decided": n_decided,
        "n_games": len(rows),
        "ats_pct": round(ats_pct, 1),
        "roi": round(roi, 1),
        "mae": round(mae, 2),
    }


def print_table(metrics: list[dict]) -> None:
    """Print a formatted table of results to console."""
    header = f"{'Predictor':<20} {'Games':>5} {'ATS W-L-P':>12} {'ATS%':>6} {'ROI%':>7} {'MAE':>7}"
    print()
    print("=" * len(header))
    print(f"  Live Prediction Performance (since {LIVE_START})")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for m in metrics:
        wlp = f"{m['ats_wins']}-{m['ats_losses']}-{m['ats_pushes']}"
        print(
            f"{m['predictor']:<20} {m['n_games']:>5} {wlp:>12} "
            f"{m['ats_pct']:>5.1f}% {m['roi']:>6.1f}% {m['mae']:>7.2f}"
        )
    print("-" * len(header))
    print(f"  Breakeven at -110: {BREAKEVEN_ATS}%")
    print()


def create_chart(metrics: list[dict]) -> None:
    """Create a grouped bar chart and save to OUTPUT_PATH."""
    predictors = [m["predictor"] for m in metrics]
    ats_vals = [m["ats_pct"] for m in metrics]
    roi_vals = [m["roi"] for m in metrics]
    mae_vals = [m["mae"] for m in metrics]
    labels = [LABELS.get(p, p) for p in predictors]
    colors = [COLORS.get(p, "#888888") for p in predictors]

    fig, axes = plt.subplots(1, 3, figsize=(16, 6), gridspec_kw={"wspace": 0.35})
    fig.patch.set_facecolor("white")

    # -----------------------------------------------------------------------
    # Panel 1: ATS Win Rate (%)
    # -----------------------------------------------------------------------
    ax1 = axes[0]
    bars1 = ax1.barh(labels, ats_vals, color=colors, edgecolor="white", height=0.6)
    ax1.axvline(
        x=BREAKEVEN_ATS,
        color="#dc2626",
        linestyle="--",
        linewidth=1.5,
        label=f"Breakeven ({BREAKEVEN_ATS}%)",
    )
    ax1.set_xlabel("ATS Win Rate (%)", fontsize=11, fontweight="bold")
    ax1.set_title("ATS Win Rate", fontsize=13, fontweight="bold", pad=10)
    ax1.legend(loc="lower right", fontsize=9)
    ax1.set_xlim(0, max(ats_vals) * 1.2)

    for bar, val in zip(bars1, ats_vals):
        color = "#16a34a" if val > BREAKEVEN_ATS else "#dc2626"
        ax1.text(
            bar.get_width() + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%",
            va="center",
            ha="left",
            fontsize=10,
            fontweight="bold",
            color=color,
        )

    # -----------------------------------------------------------------------
    # Panel 2: ROI at -110 (%)
    # -----------------------------------------------------------------------
    ax2 = axes[1]
    roi_colors = ["#16a34a" if v > 0 else "#dc2626" for v in roi_vals]
    bars2 = ax2.barh(labels, roi_vals, color=roi_colors, edgecolor="white", height=0.6)
    ax2.axvline(x=0, color="#374151", linestyle="-", linewidth=1)
    ax2.set_xlabel("ROI (%)", fontsize=11, fontweight="bold")
    ax2.set_title("ROI at -110 Odds", fontsize=13, fontweight="bold", pad=10)

    for bar, val in zip(bars2, roi_vals):
        # Always place label to the right of the bar end, pushing into whitespace
        x_pos = bar.get_width() + (0.8 if val >= 0 else -0.8)
        ha = "left" if val >= 0 else "right"
        color = "#16a34a" if val > 0 else "#dc2626"
        ax2.text(
            x_pos,
            bar.get_y() + bar.get_height() / 2,
            f"{val:+.1f}%",
            va="center",
            ha=ha,
            fontsize=10,
            fontweight="bold",
            color=color,
        )

    # Set symmetric x limits with room for labels
    max_abs_roi = max(abs(v) for v in roi_vals)
    ax2.set_xlim(-max_abs_roi * 1.5, max_abs_roi * 1.5)

    # -----------------------------------------------------------------------
    # Panel 3: Spread MAE
    # -----------------------------------------------------------------------
    ax3 = axes[2]
    bars3 = ax3.barh(labels, mae_vals, color=colors, edgecolor="white", height=0.6)
    ax3.set_xlabel("Spread MAE (points)", fontsize=11, fontweight="bold")
    ax3.set_title("Spread MAE", fontsize=13, fontweight="bold", pad=10)

    min_mae = min(mae_vals)
    for bar, val in zip(bars3, mae_vals):
        fontweight = "bold" if val == min_mae else "normal"
        ax3.text(
            bar.get_width() + 0.1,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}",
            va="center",
            ha="left",
            fontsize=10,
            fontweight=fontweight,
            color="#374151",
        )

    ax3.set_xlim(0, max(mae_vals) * 1.18)

    # -----------------------------------------------------------------------
    # Global styling
    # -----------------------------------------------------------------------
    for ax in axes:
        ax.invert_yaxis()  # best predictor at top
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="y", labelsize=10)
        ax.tick_params(axis="x", labelsize=9)

    fig.suptitle(
        f"Live Prediction Performance  (since {LIVE_START})",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )

    fig.subplots_adjust(left=0.08, right=0.97, top=0.88, bottom=0.10, wspace=0.45)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=100, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Chart saved to {OUTPUT_PATH}")


def main():
    metrics = []
    with get_db() as conn:
        for predictor in PREDICTORS:
            m = compute_predictor_metrics(predictor, conn)
            if m is not None:
                metrics.append(m)
            else:
                print(f"WARNING: {predictor} — no decided games, skipping.")

    if not metrics:
        print("ERROR: No metrics computed. Is the database populated?")
        sys.exit(1)

    print_table(metrics)
    create_chart(metrics)


if __name__ == "__main__":
    main()
