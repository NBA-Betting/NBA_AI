#!/usr/bin/env python3
"""Generate prediction model performance chart for README.

Scores live predictions (made before tip-off, April 1, 2026 onward) with the
results dashboard's model comparison: ATS win rate, ROI at -110, and Spread
MAE for all predictors on the games they all predicted. Produces a
grouped bar chart saved to docs/images/predictor_performance.png.

Usage:
    python scripts/generate_readme_chart.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Project setup — resolve paths relative to repo root
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.web_app.dashboard import (  # noqa: E402
    LIVE_PREDICTIONS_START as LIVE_START,
    compare_predictors,
    load_live_record,
)

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
    # Same comparison as the dashboard: every model on the games they all
    # predicted (models with too few games are left out)
    games, predictions = load_live_record()
    rows, n_common, excluded = compare_predictors(games, predictions)
    by_name = {row["predictor"]: row for row in rows}
    metrics = [by_name[p] for p in PREDICTORS if p in by_name]
    print(f"{n_common} games every model predicted")
    for p in excluded:
        print(f"WARNING: {p} — too few games, left out.")

    if not metrics:
        print("ERROR: No metrics computed. Is the database populated?")
        sys.exit(1)

    print_table(metrics)
    create_chart(metrics)


if __name__ == "__main__":
    main()
