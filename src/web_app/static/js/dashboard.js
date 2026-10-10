/**
 * @module dashboard.js
 *
 * Results dashboard: summary cards, cumulative P&L and rolling ATS charts,
 * and the per-game results table, all from /dashboard/data.
 */

const GAMES_PAGE_SIZE = 20;
let allGames = [];
let gamesShown = GAMES_PAGE_SIZE;

/** Formats a number with an explicit sign, e.g. +3.5 / -2.0. */
function signed(value, digits = 1) {
  return (value >= 0 ? "+" : "") + value.toFixed(digits);
}

/** Formats dollars with a sign, e.g. +$120 / -$45. */
function dollars(value, digits = 0) {
  return (value >= 0 ? "+$" : "-$") + Math.abs(value).toFixed(digits);
}

function currentFilters() {
  return new URLSearchParams({
    predictor: document.getElementById("predictorSelect").value,
    live_only: document.getElementById("liveOnlyToggle").checked,
  });
}

function showError(message) {
  const alert = document.getElementById("errorAlert");
  alert.textContent = message;
  alert.hidden = false;
}

function loadDashboard() {
  fetch(`/dashboard/data?${currentFilters()}`)
    .then((response) =>
      response.json().then((data) => {
        if (!response.ok) {
          throw new Error(data.error || "Failed to load data");
        }
        return data;
      }),
    )
    .then((data) => {
      if (data.summary.n_games === 0) {
        document.getElementById("emptyState").hidden = false;
        return;
      }
      document.getElementById("dashboardContent").hidden = false;
      populateSummary(data.summary);
      drawPLChart(data.rolling.cumulative_pl);
      drawATSChart(data.rolling.ats);
      allGames = [...data.games].reverse(); // most recent first
      renderGamesTable();
    })
    .catch((error) => showError(`Error loading dashboard: ${error.message}`))
    .finally(() => {
      document.getElementById("loadingSpinner").hidden = true;
    });
}

function populateSummary(summary) {
  const setText = (id, text) => {
    document.getElementById(id).textContent = text;
  };

  setText(
    "atsRecord",
    `${summary.ats_wins}-${summary.ats_losses}` +
      (summary.ats_pushes > 0 ? `-${summary.ats_pushes}` : ""),
  );
  setText(
    "atsPct",
    `${summary.ats_pct}% win rate (always home: ${summary.always_home_ats_pct}%)`,
  );

  const roi = document.getElementById("roi");
  roi.textContent = `${summary.roi >= 0 ? "+" : ""}${summary.roi}%`;
  roi.classList.add(summary.roi >= 0 ? "text-profit" : "text-loss");

  // $100 flat bets at -110: a win pays $90.91
  const profit = summary.ats_wins * (100 / 1.1) - summary.ats_losses * 100;
  setText("roiDetail", `${dollars(profit)} profit`);

  setText("mae", summary.mae.toFixed(2));
  setText(
    "maeDetail",
    `vs actual margin (Vegas: ${summary.vegas_mae.toFixed(2)})`,
  );
  setText("nGames", summary.n_games);
  setText(
    "nGamesDetail",
    `${summary.ats_wins + summary.ats_losses} decided, ${summary.ats_pushes} pushes`,
  );
}

/**
 * Draws a line chart over a date series, with each point's date as its
 * tooltip title.
 */
function drawLineChart(canvasId, points, datasets, options) {
  new Chart(document.getElementById(canvasId), {
    type: "line",
    data: { labels: points.map((d) => d.date), datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { intersect: false, mode: "index" },
      plugins: {
        legend: options.legend || { display: false },
        tooltip: { callbacks: { label: options.tooltipLabel } },
      },
      scales: {
        x: {
          ticks: { maxRotation: 45, autoSkip: true, maxTicksLimit: 8 },
          grid: { display: false },
        },
        y: options.y,
      },
    },
  });
}

function drawPLChart(plData) {
  if (plData.length === 0) return;
  drawLineChart(
    "plChart",
    plData,
    [
      {
        label: "Cumulative P&L ($)",
        data: plData.map((d) => d.pl),
        borderColor: plData[plData.length - 1].pl >= 0 ? "#28a745" : "#dc3545",
        borderWidth: 2,
        pointRadius: 0,
        tension: 0.1,
      },
    ],
    {
      tooltipLabel: (context) => dollars(context.parsed.y, 2),
      y: { ticks: { callback: (val) => dollars(val) } },
    },
  );
}

function drawATSChart(atsData) {
  if (atsData.length === 0) {
    document.getElementById("atsChartEmpty").hidden = false;
    return;
  }
  const constant = (label, value, color, dash, width) => ({
    label,
    data: atsData.map(() => value),
    borderColor: color,
    borderDash: dash,
    borderWidth: width,
    pointRadius: 0,
  });
  drawLineChart(
    "atsChart",
    atsData,
    [
      {
        label: "20-Game ATS %",
        data: atsData.map((d) => d.ats_pct),
        borderColor: "#0d6efd",
        borderWidth: 2,
        pointRadius: 0,
        tension: 0.1,
      },
      constant("Breakeven (52.4%)", 52.4, "#ffc107", [6, 4], 1.5),
      constant("Coin Flip (50%)", 50, "#adb5bd", [3, 3], 1),
    ],
    {
      legend: {
        position: "bottom",
        labels: { boxWidth: 12, padding: 15 },
      },
      tooltipLabel: (context) =>
        `${context.dataset.label}: ${context.parsed.y.toFixed(1)}%`,
      y: { min: 20, max: 80, ticks: { callback: (val) => `${val}%` } },
    },
  );
}

function renderGamesTable() {
  const tbody = document.getElementById("gamesTableBody");
  const atsClass = { W: "ats-win", L: "ats-loss", P: "ats-push" };

  tbody.replaceChildren(
    ...allGames.slice(0, gamesShown).map((game) => {
      const cells = [
        [game.date, ""],
        [game.home, ""],
        [game.away, ""],
        [signed(game.actual_margin, 0), "text-center"],
        // Our spread in Vegas convention (negative = home favored)
        [signed(-game.pred_spread), "text-center"],
        [signed(game.closing_spread), "text-center"],
        [game.ats_result, `text-center ${atsClass[game.ats_result]}`],
        [game.abs_error.toFixed(1), "text-center"],
      ];
      const tr = document.createElement("tr");
      for (const [text, className] of cells) {
        const td = document.createElement("td");
        td.textContent = text;
        td.className = className;
        tr.appendChild(td);
      }
      return tr;
    }),
  );

  const remaining = allGames.length - gamesShown;
  const btn = document.getElementById("showMoreBtn");
  btn.hidden = remaining <= 0;
  btn.textContent = `Show More (${Math.min(remaining, GAMES_PAGE_SIZE)} more)`;
}

document.addEventListener("DOMContentLoaded", function () {
  loadDashboard();

  // Filters reload the page so the URL can be shared
  for (const id of ["predictorSelect", "liveOnlyToggle"]) {
    document.getElementById(id).addEventListener("change", function () {
      window.location.search = currentFilters();
    });
  }

  document.getElementById("showMoreBtn").addEventListener("click", function () {
    gamesShown += GAMES_PAGE_SIZE;
    renderGamesTable();
  });
});
