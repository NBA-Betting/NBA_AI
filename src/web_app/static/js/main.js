/**
 * @module main.js
 *
 * This module provides functions to fetch and display game-related data, including:
 * - Fetching and displaying a list of games for a specific date.
 * - Populating player details in a specified container.
 * - Displaying play-by-play data for a game.
 * - Fetching and showing detailed information for a specific game.
 *
 * Functions:
 * - fetchAndUpdateGames(): Fetches games for a specific date and updates the games table.
 * - populatePlayerDetails(players, container, limit): Populates a container with player details.
 * - populatePlayByPlay(home_team, away_team, pbp, limit): Populates a table with play-by-play data.
 * - showGameDetails(gameId): Fetches and displays the details of a specific game.
 *
 * Each function handles specific aspects of game data presentation, aiming to provide a seamless
 * user experience by dynamically updating the UI with fetched data.
 */

/**
 * Fetches games for a specific date and updates the games table.
 * The date is read from the body's `data-query-date` attribute.
 * Each game is added as a new row in the games table.
 * @throws {Error} If the fetch operation fails.
 */
function fetchAndUpdateGames() {
  // Retrieve the query date from the body's dataset
  const queryDate = document.body.dataset.queryDate;

  // Get user's timezone from browser (IANA format, e.g., "America/New_York")
  const userTz = Intl.DateTimeFormat().resolvedOptions().timeZone;

  // Fetch games data for the specified date, passing user timezone
  fetch(
    `/get-game-data?date=${queryDate}&user_tz=${encodeURIComponent(userTz)}`,
  )
    .then((response) => {
      if (!response.ok) {
        return response.json().then((error) => {
          throw new Error(error.error);
        });
      }
      return response.json();
    })
    .then((games) => {
      const tableBody = document.querySelector("#gamesTableBody");
      tableBody.innerHTML = ""; // Clear the table body

      if (games.length > 0) {
        games.forEach((game) => {
          const row = document.createElement("tr");
          row.className = "game-row custom-vertical-align-middle";
          row.setAttribute("data-game-id", game.game_id);

          // Check if game is postponed
          const isPostponed = game.game_status === "PPD";
          const datetimeDisplay = isPostponed
            ? "PPD"
            : game.datetime_display.split("-").join("<br>");
          const homeScore = isPostponed ? "-" : game.home_score;
          const awayScore = isPostponed ? "-" : game.away_score;
          const openSpread = isPostponed ? "-" : game.opening_spread || "-";
          const predSpreadDisplay = isPostponed ? "-" : game.pred_spread || "-";
          const predWinnerDisplay = isPostponed
            ? "-"
            : `${game.pred_winner} ${game.pred_win_pct}`;

          // Color-coding classes for completed games
          const winnerClass =
            game.pred_winner_correct === true
              ? " pred-correct"
              : game.pred_winner_correct === false
                ? " pred-wrong"
                : "";
          const spreadClass =
            game.spread_closer_than_vegas === true
              ? " pred-correct"
              : game.spread_closer_than_vegas === false
                ? " pred-wrong"
                : "";

          row.innerHTML = `
                        <td class="text-left custom-vertical-align-middle">${datetimeDisplay}</td>
                        <td class="custom-vertical-align-middle">
                            <div class="custom-display-flex custom-align-items-center">
                                <img src="${game.home_logo_url}" alt="Logo of ${
                                  game.home_team_display
                                }" class="custom-team-logo">
                                <div class="custom-text-align-left">${
                                  game.home_team_display
                                }</div>
                            </div>
                        </td>
                        <td class="custom-vertical-align-middle">
                            <div class="custom-display-flex custom-align-items-center">
                                <img src="${game.away_logo_url}" alt="Logo of ${
                                  game.away_team_display
                                }" class="custom-team-logo">
                                <div class="custom-text-align-left">${
                                  game.away_team_display
                                }</div>
                            </div>
                        </td>
                        <td class="text-center custom-vertical-align-middle">${openSpread}</td>
                        <td class="text-center custom-vertical-align-middle">${homeScore}</td>
                        <td class="text-center custom-vertical-align-middle">${awayScore}</td>
                        <td class="text-center custom-vertical-align-middle${spreadClass}">${predSpreadDisplay}</td>
                        <td class="text-center custom-vertical-align-middle${winnerClass}">${predWinnerDisplay}</td>
                    `;
          tableBody.appendChild(row);
        });
      } else {
        tableBody.innerHTML =
          '<tr><td colspan="8" class="text-center">No Games for the selected date</td></tr>';
      }
    })
    .catch((error) => {
      console.error("Error fetching games:", error);
      const tableBody = document.querySelector("#gamesTableBody");
      tableBody.innerHTML = `<tr><td colspan="8" class="text-center">${error.message}</td></tr>`;
    });
}

/**
 * Populates a container with player details.
 *
 * For completed/in-progress games (status 2 or 3): shows actual points scored.
 * For upcoming games (status 1): shows predicted points if available, otherwise just the roster.
 *
 * @param {Array} players - An array of player objects with `player_headshot_url`, `player_name`, `points`, and `pred_points`.
 * @param {HTMLElement} container - The container to populate with player details.
 * @param {number} gameStatusCode - The game status code (1=scheduled, 2=in-progress, 3=final).
 * @param {number} [limit=5] - The maximum number of players to display. Defaults to 5.
 */
function populatePlayerDetails(players, container, gameStatusCode, limit = 5) {
  container.innerHTML = ""; // Clear previous content

  if (players.length === 0) {
    container.innerHTML =
      '<p class="text-muted text-center">No player data available</p>';
    return;
  }

  const hasActualStats = gameStatusCode === 2 || gameStatusCode === 3;
  const hasPredictions = players.some(
    (p) => p.pred_points !== null && p.pred_points !== undefined,
  );

  players.slice(0, limit).forEach((player) => {
    const playerDetailDiv = document.createElement("div");
    playerDetailDiv.className = "player-detail d-flex align-items-center mb-2";

    let statsDisplay;
    if (
      hasActualStats &&
      player.points !== null &&
      player.points !== undefined
    ) {
      statsDisplay = `${player.points} PTS`;
    } else if (
      hasPredictions &&
      player.pred_points !== null &&
      player.pred_points !== undefined
    ) {
      statsDisplay = `${player.pred_points} PTS (pred)`;
    } else {
      statsDisplay = "";
    }

    playerDetailDiv.innerHTML = `
            <img src="${player.player_headshot_url}" alt="${player.player_name}" class="player-headshot me-2">
            <div>
                <p class="mb-0"><strong>${player.player_name}</strong></p>
                ${statsDisplay ? `<p class="mb-0">${statsDisplay}</p>` : ""}
            </div>
        `;
    container.appendChild(playerDetailDiv);
  });
}

/**
 * Populates a table with play-by-play data.
 *
 * @param {string} home_team - The name of the home team.
 * @param {string} away_team - The name of the away team.
 * @param {Array} pbp - An array of play-by-play records. Each record should have `time_info`, `description`, `home_score`, and `away_score` properties.
 * @param {number} [limit=Infinity] - The maximum number of records to display. Defaults to Infinity.
 */
function populatePlayByPlay(home_team, away_team, pbp, limit = Infinity) {
  const homeTeamHeader = document.getElementById("homeTeamHeader");
  const awayTeamHeader = document.getElementById("awayTeamHeader");
  const playByPlayBody = document.getElementById("playByPlayBody");

  homeTeamHeader.textContent = home_team;
  awayTeamHeader.textContent = away_team;

  playByPlayBody.innerHTML = ""; // Clear existing content

  if (pbp.length === 0) {
    playByPlayBody.innerHTML =
      '<tr><td colspan="4" class="text-center no-pbp-data">No Play By Play Logs Available</td></tr>';
  } else {
    pbp.slice(0, limit).forEach((record) => {
      const row = document.createElement("tr");
      row.innerHTML = `
                <td>${record.time_info}</td>
                <td>${record.description}</td>
                <td>${record.home_score}</td>
                <td>${record.away_score}</td>
            `;
      playByPlayBody.appendChild(row);
    });
  }
}

/**
 * Fetches and displays the details of a specific game.
 *
 * @param {string} gameId - The ID of the game to fetch details for.
 */
function showGameDetails(gameId) {
  // Get user's timezone from browser (IANA format, e.g., "America/New_York")
  const userTz = Intl.DateTimeFormat().resolvedOptions().timeZone;

  fetch(
    `/get-game-data?game_id=${gameId}&user_tz=${encodeURIComponent(userTz)}`,
  )
    .then((response) => {
      if (!response.ok) {
        return response.json().then((error) => {
          throw new Error(error.error);
        });
      }
      return response.json();
    })
    .then((data) => {
      const game = data[0];

      const {
        home,
        away,
        home_full_name: homeFullName,
        away_full_name: awayFullName,
        home_logo_url: homeLogoUrl,
        away_logo_url: awayLogoUrl,
        home_score: homeScore,
        away_score: awayScore,
        game_status_code: gameStatusCode,
        datetime_display: dateTimeDisplay,
        condensed_pbp: playByPlay,
        home_players: homePlayers,
        away_players: awayPlayers,
        pred_winner: predictedWinner,
        pred_win_pct: predictedWinPercentage,
      } = game;

      // Modal title — show score for completed games, time for upcoming
      const modalTitle = document.querySelector("#gameDetailsModalLabel");
      const scoreDisplay =
        gameStatusCode === 3
          ? `${homeScore} - ${awayScore}`
          : gameStatusCode === 2
            ? `${homeScore} - ${awayScore}`
            : "vs";
      modalTitle.innerHTML = `
                ${home} <img src="${homeLogoUrl}" alt="${home}" class="team-logo">
                ${scoreDisplay}
                <img src="${awayLogoUrl}" alt="${away}" class="team-logo"> ${away}
                <br><small class="text-muted">${dateTimeDisplay}</small>
            `;

      const template = document
        .querySelector("#gameDetailsTemplate")
        .content.cloneNode(true);
      template.querySelector("#templateHomeTeam").textContent = homeFullName;
      template.querySelector("#templateAwayTeam").textContent = awayFullName;
      template.querySelector("#templateHomeLogo").src = homeLogoUrl;
      template.querySelector("#templateAwayLogo").src = awayLogoUrl;

      // Spreads
      template.querySelector("#templateOpenSpread").textContent =
        game.opening_spread || "-";
      template.querySelector("#templatePredictedSpread").textContent =
        game.pred_spread || "-";

      // Show actual margin for completed games as "TEAM by X"
      if (gameStatusCode === 3 && homeScore !== "" && awayScore !== "") {
        const margin = homeScore - awayScore;
        let resultStr;
        if (margin > 0) resultStr = `${home} by ${margin}`;
        else if (margin < 0) resultStr = `${away} by ${Math.abs(margin)}`;
        else resultStr = "Tie";
        template.querySelector("#templateActualMargin").textContent = resultStr;
        template.querySelector("#templateResultSection").style.display =
          "block";
      }

      // Winner
      template.querySelector("#templatePredictedWinPct").textContent =
        predictedWinPercentage || "";

      populatePlayerDetails(
        homePlayers,
        template.querySelector("#homeTeamPlayers"),
        gameStatusCode,
      );
      populatePlayerDetails(
        awayPlayers,
        template.querySelector("#awayTeamPlayers"),
        gameStatusCode,
      );

      const modalBody = document.querySelector("#gameDetailsModal .modal-body");
      modalBody.innerHTML = "";
      modalBody.appendChild(template);

      const winnerLeftIcon = document.getElementById("winnerLeftIcon");
      const winnerRightIcon = document.getElementById("winnerRightIcon");

      // Determine arrow color: black for upcoming, green/red for completed
      let arrowColor = "#2C3E50"; // default black
      if (gameStatusCode === 3 && homeScore !== "" && awayScore !== "") {
        const actualWinner =
          homeScore > awayScore ? home : awayScore > homeScore ? away : "";
        arrowColor = predictedWinner === actualWinner ? "#1a7f37" : "#cf222e";
      }

      if (predictedWinner === home) {
        winnerLeftIcon.style.visibility = "visible";
        winnerLeftIcon.style.color = arrowColor;
        winnerRightIcon.style.visibility = "hidden";
      } else if (predictedWinner === away) {
        winnerRightIcon.style.visibility = "visible";
        winnerRightIcon.style.color = arrowColor;
        winnerLeftIcon.style.visibility = "hidden";
      } else {
        winnerLeftIcon.style.visibility = "hidden";
        winnerRightIcon.style.visibility = "hidden";
      }

      populatePlayByPlay(home, away, playByPlay, 100);

      var gameDetailsModal = new bootstrap.Modal(
        document.getElementById("gameDetailsModal"),
        {},
      );
      gameDetailsModal.show();
    })
    .catch((error) => {
      console.error("Error fetching game details:", error);
    });
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

  document
    .querySelector("#gamesTableBody")
    .addEventListener("click", function (event) {
      let target = event.target;

      while (target && !target.classList.contains("game-row")) {
        target = target.parentElement;
      }

      if (target && target.classList.contains("game-row")) {
        const gameId = target.getAttribute("data-game-id");
        showGameDetails(gameId);
      }
    });
});
