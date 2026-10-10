/**
 * @module dashboard.js
 *
 * Results dashboard for one record (a live season, or a saved backtest) and
 * predictor: summary cards, cumulative P&L, a comparison of all models on the
 * same games, and the graded games. Data comes from /dashboard/data.
 */

const GAMES_PAGE_SIZE = 20;
const BREAKEVEN_ATS = 52.4; // ATS win rate that breaks even at -110
const RECORD_NOTES = {
  live: "Live: predictions made and stored before tip-off.",
  backtest:
    "Backtest: the season replayed through each model's live code with pre-game information only.",
};
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

function setText(id, text) {
  document.getElementById(id).textContent = text;
}

/** Reloads the page for a record and predictor, so the URL can be shared. */
function navigate(record, predictor) {
  window.location.search = new URLSearchParams({ record, predictor });
}

function fillSelect(id, options, selected) {
  const select = document.getElementById(id);
  select.replaceChildren(
    ...options.map(
      ({ value, label }) => new Option(label, value, false, value === selected),
    ),
  );
}

function loadDashboard() {
  const params = new URLSearchParams(window.location.search);
  fetch(`/dashboard/data?${params}`)
    .then((response) =>
      response.json().then((data) => {
        if (!response.ok) {
          throw new Error(data.error || "Failed to load data");
        }
        return data;
      }),
    )
    .then((data) => {
      fillSelect(
        "recordSelect",
        data.records.map((r) => ({ value: r.id, label: r.label })),
        data.record,
      );
      if (!data.summary) {
        document.getElementById("emptyState").hidden = false;
        return;
      }
      fillSelect(
        "predictorSelect",
        data.predictors.map((p) => ({ value: p, label: p })),
        data.predictor,
      );
      const record = data.records.find((r) => r.id === data.record);
      setText(
        "recordNote",
        [RECORD_NOTES[data.is_backtest ? "backtest" : "live"], record.note]
          .filter(Boolean)
          .join(" "),
      );

      document.getElementById("dashboardContent").hidden = false;
      populateSummary(data.summary);
      drawPLChart(data.cumulative_pl);
      renderComparison(data.comparison, data.record, data.predictor);
      allGames = data.games;
      renderGamesTable();
    })
    .catch((error) => {
      const alert = document.getElementById("errorAlert");
      alert.textContent = `Error loading dashboard: ${error.message}`;
      alert.hidden = false;
    })
    .finally(() => {
      document.getElementById("loadingSpinner").hidden = true;
    });
}

function populateSummary(summary) {
  setText(
    "atsRecord",
    `${summary.ats_wins}-${summary.ats_losses}` +
      (summary.ats_pushes > 0 ? `-${summary.ats_pushes}` : ""),
  );
  setText(
    "atsPct",
    `${summary.ats_pct}% (breakeven ${BREAKEVEN_ATS}%, always home ${summary.always_home_ats_pct}%)`,
  );

  const roi = document.getElementById("roi");
  roi.textContent = `${summary.roi >= 0 ? "+" : ""}${summary.roi}%`;
  roi.classList.add(summary.roi >= 0 ? "text-profit" : "text-loss");
  setText("roiDetail", `${dollars(summary.profit)} on $100 bets`);

  setText("mae", summary.mae.toFixed(2));
  setText("maeDetail", `points (Vegas: ${summary.vegas_mae.toFixed(2)})`);
  setText("nGames", summary.n_games);
  setText(
    "nGamesDetail",
    `${summary.ats_wins + summary.ats_losses} decided, ${summary.ats_pushes} pushes`,
  );
}

function drawPLChart(plData) {
  if (plData.length === 0) return;
  new Chart(document.getElementById("plChart"), {
    type: "line",
    data: {
      labels: plData.map((d) => d.date),
      datasets: [
        {
          data: plData.map((d) => d.pl),
          borderColor:
            plData[plData.length - 1].pl >= 0 ? "#1a7f37" : "#cf222e",
          borderWidth: 2,
          pointRadius: 0,
          tension: 0.1,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { intersect: false, mode: "index" },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: { label: (context) => dollars(context.parsed.y, 2) },
        },
      },
      scales: {
        x: {
          ticks: {
            maxRotation: 0,
            autoSkip: true,
            maxTicksLimit: 6,
            // "2026-01-15" -> "Jan 15"
            callback: function (value) {
              const [y, m, d] = this.getLabelForValue(value).split("-").map(Number);
              return new Date(y, m - 1, d).toLocaleDateString("en-US", {
                month: "short",
                day: "numeric",
              });
            },
          },
          grid: { display: false },
        },
        y: { ticks: { callback: (val) => dollars(val) } },
      },
    },
  });
}

/** All models on the games they all predicted; a row selects that model. */
function renderComparison(comparison, record, selected) {
  setText(
    "comparisonNote",
    `The ${comparison.n_games} games every model predicted, best MAE first`,
  );
  document.getElementById("comparisonBody").replaceChildren(
    ...comparison.rows.map((row) => {
      const tr = document.createElement("tr");
      tr.className = "comparison-row";
      if (row.predictor === selected) {
        tr.classList.add("table-active");
      }
      const cells = [
        [row.predictor, ""],
        [`${row.ats_pct.toFixed(1)}%`, "text-end"],
        [
          `${row.roi >= 0 ? "+" : ""}${row.roi.toFixed(1)}%`,
          `text-end ${row.roi >= 0 ? "text-profit" : "text-loss"}`,
        ],
        [row.mae.toFixed(2), "text-end"],
      ];
      for (const [text, className] of cells) {
        const td = document.createElement("td");
        td.textContent = text;
        td.className = className;
        tr.appendChild(td);
      }
      tr.addEventListener("click", () => navigate(record, row.predictor));
      return tr;
    }),
  );
}

function renderGamesTable() {
  const tbody = document.getElementById("gamesTableBody");
  const atsClass = { W: "ats-win", L: "ats-loss", P: "ats-push" };

  tbody.replaceChildren(
    ...allGames.slice(0, gamesShown).map((game) => {
      const cells = [
        [game.date, "text-nowrap"],
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

  const recordSelect = document.getElementById("recordSelect");
  const predictorSelect = document.getElementById("predictorSelect");
  recordSelect.addEventListener("change", () =>
    navigate(recordSelect.value, predictorSelect.value),
  );
  predictorSelect.addEventListener("change", () =>
    navigate(recordSelect.value, predictorSelect.value),
  );

  document.getElementById("showMoreBtn").addEventListener("click", function () {
    gamesShown += GAMES_PAGE_SIZE;
    renderGamesTable();
  });
});
