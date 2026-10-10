/**
 * @module main.js
 *
 * This module provides functions to fetch and display game-related data, including:
 * - Fetching and displaying a list of games for a specific date.
 * - Populating player details in a specified container.
 * - Displaying play-by-play data for a game.
 * - Fetching and showing detailed information for a specific game.
 * - Choosing a date from the calendar picker.
 *
 * API text is always inserted with textContent, never as HTML.
 */

/**
 * Returns a table cell with the given text and classes.
 * Text with newlines renders as separate lines (see .multiline).
 */
function textCell(text, className = "") {
  const td = document.createElement("td");
  td.className = className;
  td.textContent = text;
  return td;
}

/**
 * Returns a table cell with a team logo and the team's display name.
 */
function teamCell(logoUrl, fullName, displayName) {
  const td = document.createElement("td");
  const wrapper = document.createElement("div");
  wrapper.className = "d-flex align-items-center";
  const logo = document.createElement("img");
  logo.src = logoUrl;
  logo.alt = fullName;
  logo.title = fullName;
  logo.className = "team-logo-row";
  // Names are hidden on phones; the logo carries the name there
  const name = document.createElement("div");
  name.className = "multiline d-none d-sm-block";
  name.textContent = displayName;
  wrapper.append(logo, name);
  td.appendChild(wrapper);
  return td;
}

/**
 * Replaces a table body's rows with one full-width message row.
 */
function showTableMessage(tableBody, colspan, message) {
  const row = document.createElement("tr");
  row.appendChild(textCell(message, "text-center text-muted"));
  row.firstChild.colSpan = colspan;
  tableBody.replaceChildren(row);
}

/**
 * Fetches JSON for one or more games from /get-game-data, passing the
 * browser's timezone so times and "Today"/"Tomorrow" are local.
 * @param {Object} params - Either {date} or {game_id}.
 * @returns {Promise<Array>} The processed games.
 */
function fetchGameData(params) {
  const query = new URLSearchParams({
    ...params,
    user_tz: Intl.DateTimeFormat().resolvedOptions().timeZone,
  });
  return fetch(`/get-game-data?${query}`).then((response) =>
    response.json().then((data) => {
      if (!response.ok) {
        throw new Error(data.error || "Failed to load games");
      }
      return data;
    }),
  );
}

/**
 * Fetches games for the page's date (body `data-query-date`) and fills the
 * games table, one clickable row per game.
 */
function fetchAndUpdateGames() {
  const tableBody = document.querySelector("#gamesTableBody");

  fetchGameData({ date: document.body.dataset.queryDate })
    .then((games) => {
      if (games.length === 0) {
        showTableMessage(tableBody, 8, "No games on this date");
        return;
      }

      tableBody.replaceChildren(
        ...games.map((game) => {
          const row = document.createElement("tr");
          row.className = "game-row";
          row.dataset.gameId = game.game_id;
          row.tabIndex = 0;

          const isPostponed = game.game_status === "PPD";
          const unavailable = game.prediction_unavailable_reason;
          const predWinner = game.pred_winner
            ? `${game.pred_winner} ${game.pred_win_pct}`
            : "-";

          // Graded colors for completed games; backtests (made after
          // tip-off) are shown muted and not graded
          const grade = (correct) =>
            game.pred_is_backtest
              ? " pred-backtest"
              : correct === true
                ? " pred-correct"
                : correct === false
                  ? " pred-wrong"
                  : "";

          const spreadCell = textCell(
            isPostponed ? "-" : unavailable || game.pred_spread || "-",
            "text-center" + grade(game.spread_closer_than_vegas),
          );
          const winnerCell = textCell(
            isPostponed ? "-" : unavailable ? "" : predWinner,
            "text-center" + grade(game.pred_winner_correct),
          );
          if (unavailable) {
            spreadCell.colSpan = 2;
          }
          if (game.pred_is_backtest) {
            spreadCell.title = winnerCell.title =
              "Backtest prediction, made after tip-off";
          }

          row.append(
            textCell(
              isPostponed ? "PPD" : game.datetime_display.replace(" - ", "\n"),
              "multiline",
            ),
            teamCell(
              game.home_logo_url,
              game.home_full_name,
              game.home_team_display,
            ),
            teamCell(
              game.away_logo_url,
              game.away_full_name,
              game.away_team_display,
            ),
            textCell(
              isPostponed ? "-" : game.opening_spread || "-",
              "text-center d-none d-sm-table-cell",
            ),
            textCell(isPostponed ? "-" : game.home_score, "text-center"),
            textCell(isPostponed ? "-" : game.away_score, "text-center"),
            spreadCell,
          );
          if (!unavailable) {
            row.appendChild(winnerCell);
          }
          return row;
        }),
      );
    })
    .catch((error) => {
      console.error("Error fetching games:", error);
      showTableMessage(tableBody, 8, `Could not load games: ${error.message}`);
    });
}

/**
 * Fills a container with a team's top players.
 *
 * For completed/in-progress games (status 2 or 3): shows actual points scored.
 * For upcoming games: shows predicted points if available, otherwise just the roster.
 *
 * @param {Array} players - Objects with `player_headshot_url`, `player_name`, `points`, and `pred_points`.
 * @param {HTMLElement} container - The container to fill.
 * @param {number} gameStatusCode - 1=scheduled, 2=in-progress, 3=final.
 * @param {number} [limit=5] - The maximum number of players to show.
 */
function populatePlayerDetails(players, container, gameStatusCode, limit = 5) {
  if (players.length === 0) {
    const empty = document.createElement("p");
    empty.className = "text-muted text-center";
    empty.textContent = "No player data available";
    container.replaceChildren(empty);
    return;
  }

  const hasActualStats = gameStatusCode === 2 || gameStatusCode === 3;

  container.replaceChildren(
    ...players.slice(0, limit).map((player) => {
      let stats = "";
      if (hasActualStats && player.points != null) {
        stats = `${player.points} PTS`;
      } else if (player.pred_points != null) {
        stats = `${player.pred_points} PTS (pred)`;
      }

      const detail = document.createElement("div");
      detail.className = "d-flex align-items-center mb-2";
      const headshot = document.createElement("img");
      headshot.src = player.player_headshot_url;
      headshot.alt = "";
      headshot.className = "player-headshot me-2";
      const text = document.createElement("div");
      const name = document.createElement("p");
      name.className = "mb-0 fw-bold";
      name.textContent = player.player_name;
      text.appendChild(name);
      if (stats) {
        const statLine = document.createElement("p");
        statLine.className = "mb-0";
        statLine.textContent = stats;
        text.appendChild(statLine);
      }
      detail.append(headshot, text);
      return detail;
    }),
  );
}

/**
 * Fills the modal's play-by-play table, most recent play first.
 *
 * @param {DocumentFragment} root - The game details being built.
 * @param {string} homeTeam - Home team abbreviation (column header).
 * @param {string} awayTeam - Away team abbreviation (column header).
 * @param {Array} pbp - Records with `time_info`, `description`, `home_score`, `away_score`.
 * @param {number} [limit=Infinity] - The maximum number of records to show.
 */
function populatePlayByPlay(root, homeTeam, awayTeam, pbp, limit = Infinity) {
  root.querySelector("#homeTeamHeader").textContent = homeTeam;
  root.querySelector("#awayTeamHeader").textContent = awayTeam;
  const body = root.querySelector("#playByPlayBody");

  if (pbp.length === 0) {
    showTableMessage(body, 4, "No Play By Play Logs Available");
    return;
  }
  body.replaceChildren(
    ...pbp.slice(0, limit).map((record) => {
      const row = document.createElement("tr");
      row.append(
        textCell(record.time_info, "text-nowrap"),
        textCell(record.description),
        textCell(record.home_score),
        textCell(record.away_score),
      );
      return row;
    }),
  );
}

/**
 * Fills the modal title: "HOME [logo] score [logo] AWAY" over the date/status.
 */
function setModalTitle(game) {
  const logo = (url) => {
    const img = document.createElement("img");
    img.src = url;
    img.alt = "";
    img.className = "team-logo-title mx-1";
    return img;
  };
  const hasScore =
    game.game_status_code === 2 || game.game_status_code === 3;
  const status = document.createElement("small");
  status.className = "text-muted d-block";
  status.textContent = game.datetime_display;

  document
    .querySelector("#gameDetailsModalLabel")
    .replaceChildren(
      `${game.home} `,
      logo(game.home_logo_url),
      hasScore ? ` ${game.home_score} - ${game.away_score} ` : " vs ",
      logo(game.away_logo_url),
      ` ${game.away}`,
      status,
    );
}

/**
 * Fetches a game and shows its details in the modal.
 *
 * @param {string} gameId - The game to show.
 */
function showGameDetails(gameId) {
  const modalElement = document.getElementById("gameDetailsModal");
  const modalBody = modalElement.querySelector(".modal-body");

  fetchGameData({ game_id: gameId })
    .then((data) => {
      const game = data[0];
      const { home, away, home_score: homeScore, away_score: awayScore } = game;
      const isFinal =
        game.game_status_code === 3 && homeScore !== "" && awayScore !== "";

      setModalTitle(game);

      const details = document
        .querySelector("#gameDetailsTemplate")
        .content.cloneNode(true);
      details.querySelector("#templateHomeTeam").textContent =
        game.home_full_name;
      details.querySelector("#templateAwayTeam").textContent =
        game.away_full_name;
      details.querySelector("#templateHomeLogo").src = game.home_logo_url;
      details.querySelector("#templateAwayLogo").src = game.away_logo_url;
      details.querySelector("#templateOpenSpread").textContent =
        game.opening_spread || "-";
      details.querySelector("#templatePredictedSpread").textContent =
        game.prediction_unavailable_reason || game.pred_spread || "-";

      // Actual result as "TEAM by X"
      if (isFinal) {
        const margin = homeScore - awayScore;
        details.querySelector("#templateActualMargin").textContent =
          margin > 0
            ? `${home} by ${margin}`
            : margin < 0
              ? `${away} by ${-margin}`
              : "Tie";
        details.querySelector("#templateResultSection").hidden = false;
      }

      details.querySelector("#templatePredictedWinPct").textContent =
        game.pred_win_pct || "-";

      // Arrow toward the predicted winner: dark before the game, then
      // green/red once it's decided (backtests aren't graded)
      const arrowId =
        game.pred_winner === home
          ? "#winnerLeftIcon"
          : game.pred_winner === away
            ? "#winnerRightIcon"
            : "";
      const arrow = arrowId && details.querySelector(arrowId);
      if (arrow) {
        arrow.style.visibility = "visible";
        if (isFinal && !game.pred_is_backtest) {
          const actualWinner = homeScore > awayScore ? home : away;
          arrow.classList.add(
            game.pred_winner === actualWinner ? "pred-correct" : "pred-wrong",
          );
        }
      }

      populatePlayerDetails(
        game.home_players,
        details.querySelector("#homeTeamPlayers"),
        game.game_status_code,
      );
      populatePlayerDetails(
        game.away_players,
        details.querySelector("#awayTeamPlayers"),
        game.game_status_code,
      );
      populatePlayByPlay(details, home, away, game.condensed_pbp, 100);

      modalBody.replaceChildren(details);
    })
    .catch((error) => {
      console.error("Error fetching game details:", error);
      document.querySelector("#gameDetailsModalLabel").textContent =
        "Game Details";
      const alert = document.createElement("div");
      alert.className = "alert alert-danger mb-0";
      alert.textContent = `Could not load this game: ${error.message}`;
      modalBody.replaceChildren(alert);
    })
    .finally(() => bootstrap.Modal.getOrCreateInstance(modalElement).show());
}

/**
 * Sets up the calendar date picker attached to the current-date button.
 *
 * Days that have games are marked with a dot, days without are dimmed, so the
 * whole month can be scanned at a glance. Counts come from /game-dates and are
 * cached per month.
 */
function initDatePicker() {
  const toggle = document.getElementById("datePickerToggle");
  const panel = document.getElementById("datePickerPanel");
  const grid = document.getElementById("calGrid");
  const monthLabel = document.getElementById("calMonthLabel");

  if (!toggle || !panel || !grid) {
    return;
  }

  const MONTH_NAMES = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
  ];

  const monthCache = {};
  const selectedDateStr = document.body.dataset.queryDate;

  /**
   * Splits a "YYYY-MM-DD" string into numeric parts.
   *
   * Done by hand because new Date("YYYY-MM-DD") is parsed as UTC, which lands
   * on the previous day for anyone west of Greenwich.
   */
  function parseDateStr(str) {
    const parts = str.split("-").map(Number);
    return { year: parts[0], month: parts[1] - 1, day: parts[2] };
  }

  function formatDateStr(year, month, day) {
    const paddedMonth = String(month + 1).padStart(2, "0");
    const paddedDay = String(day).padStart(2, "0");
    return `${year}-${paddedMonth}-${paddedDay}`;
  }

  function monthKey(year, month) {
    return `${year}-${String(month + 1).padStart(2, "0")}`;
  }

  const selected = parseDateStr(selectedDateStr);
  const now = new Date();
  const todayStr = formatDateStr(
    now.getFullYear(),
    now.getMonth(),
    now.getDate(),
  );

  let viewYear = selected.year;
  let viewMonth = selected.month;

  /**
   * Loads game counts for a month, caching the result.
   * A failed lookup resolves to no counts, so the calendar still opens.
   */
  function fetchMonthCounts(year, month) {
    const key = monthKey(year, month);

    if (monthCache[key]) {
      return Promise.resolve(monthCache[key]);
    }

    return fetch(`/game-dates?month=${key}`)
      .then((response) => (response.ok ? response.json() : {}))
      .then((counts) => {
        monthCache[key] = counts || {};
        return monthCache[key];
      })
      .catch((error) => {
        console.error("Error fetching game dates:", error);
        return {};
      });
  }

  function renderCalendar() {
    monthLabel.textContent = `${MONTH_NAMES[viewMonth]} ${viewYear}`;

    const counts = monthCache[monthKey(viewYear, viewMonth)] || {};
    const firstWeekday = new Date(viewYear, viewMonth, 1).getDay();
    const daysInMonth = new Date(viewYear, viewMonth + 1, 0).getDate();

    grid.innerHTML = "";

    // Blank cells so the 1st lands on the right weekday
    for (let i = 0; i < firstWeekday; i++) {
      const filler = document.createElement("span");
      filler.className = "date-picker-day is-empty";
      grid.appendChild(filler);
    }

    for (let day = 1; day <= daysInMonth; day++) {
      const dateStr = formatDateStr(viewYear, viewMonth, day);
      const gameCount = counts[dateStr] || 0;

      const cell = document.createElement("a");
      cell.className = "date-picker-day";
      cell.href = `/?date=${dateStr}`;
      cell.textContent = day;

      if (gameCount > 0) {
        cell.classList.add("has-games");
        cell.title = `${gameCount} game${gameCount === 1 ? "" : "s"}`;
        cell.setAttribute("aria-label", `${dateStr}, ${gameCount} games`);
      } else {
        cell.classList.add("no-games");
        cell.setAttribute("aria-label", `${dateStr}, no games`);
      }

      if (dateStr === todayStr) {
        cell.classList.add("is-today");
      }

      if (dateStr === selectedDateStr) {
        cell.classList.add("is-selected");
        cell.setAttribute("aria-current", "date");
      }

      grid.appendChild(cell);
    }
  }

  /**
   * Shows a month, painting immediately and repainting once counts arrive.
   * The repaint is skipped if the user moved on, so a slow response for an
   * old month cannot overwrite the month now on screen.
   */
  function showMonth(year, month) {
    viewYear = year;
    viewMonth = month;
    renderCalendar();

    fetchMonthCounts(year, month).then(() => {
      if (viewYear === year && viewMonth === month) {
        renderCalendar();
      }
    });
  }

  function openPanel() {
    panel.hidden = false;
    toggle.setAttribute("aria-expanded", "true");
    showMonth(selected.year, selected.month);
  }

  function closePanel() {
    panel.hidden = true;
    toggle.setAttribute("aria-expanded", "false");
  }

  toggle.addEventListener("click", function (event) {
    event.stopPropagation();
    if (panel.hidden) {
      openPanel();
    } else {
      closePanel();
    }
  });

  document
    .getElementById("calPrevMonth")
    .addEventListener("click", function () {
      const month = viewMonth - 1;
      showMonth(month < 0 ? viewYear - 1 : viewYear, (month + 12) % 12);
    });

  document
    .getElementById("calNextMonth")
    .addEventListener("click", function () {
      const month = viewMonth + 1;
      showMonth(month > 11 ? viewYear + 1 : viewYear, month % 12);
    });

  document.getElementById("calToday").addEventListener("click", function () {
    window.location.href = `/?date=${todayStr}`;
  });

  // Close when clicking away or pressing Escape
  document.addEventListener("click", function (event) {
    if (
      !panel.hidden &&
      !panel.contains(event.target) &&
      !toggle.contains(event.target)
    ) {
      closePanel();
    }
  });

  document.addEventListener("keydown", function (event) {
    if (event.key === "Escape" && !panel.hidden) {
      closePanel();
      toggle.focus();
    }
  });

  // Warm the cache so the first open already shows the dots
  fetchMonthCounts(selected.year, selected.month);
}

// Initialize event listeners after DOM content is fully loaded
document.addEventListener("DOMContentLoaded", function () {
  fetchAndUpdateGames();
  initDatePicker();

  const tableBody = document.querySelector("#gamesTableBody");
  tableBody.addEventListener("click", function (event) {
    const row = event.target.closest(".game-row");
    if (row) {
      showGameDetails(row.dataset.gameId);
    }
  });
  tableBody.addEventListener("keydown", function (event) {
    const row = event.target.closest(".game-row");
    if (row && (event.key === "Enter" || event.key === " ")) {
      event.preventDefault();
      showGameDetails(row.dataset.gameId);
    }
  });
});
