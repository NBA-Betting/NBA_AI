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

    // Fetch games data for the specified date
    fetch(`/get-game-data?date=${queryDate}`)
        .then(response => {
            if (!response.ok) {
                return response.json().then(error => {
                    throw new Error(error.error);
                });
            }
            return response.json();
        })
        .then(games => {
            const tableBody = document.querySelector('#gamesTableBody');
            tableBody.innerHTML = ''; // Clear the table body

            if (games.length > 0) {
                games.forEach(game => {
                    console.log('Game:', game);
                    const row = document.createElement('tr');
                    row.className = 'game-row custom-vertical-align-middle';
                    row.setAttribute('data-game-id', game.game_id);
                    row.innerHTML = `
                        <td class="text-left custom-vertical-align-middle">${(game.datetime_display).split('-').join('<br>')}</td>
                        <td class="custom-vertical-align-middle">
                            <div class="custom-display-flex custom-align-items-center">
                                <img src="${game.home_logo_url}" alt="Logo of ${game.home_team_display}" class="custom-team-logo">
                                <div class="custom-text-align-left">${game.home_team_display}</div>
                            </div>
                        </td>
                        <td class="custom-vertical-align-middle">
                            <div class="custom-display-flex custom-align-items-center">
                                <img src="${game.away_logo_url}" alt="Logo of ${game.away_team_display}" class="custom-team-logo">
                                <div class="custom-text-align-left">${game.away_team_display}</div>
                            </div>
                        </td>
                        <td class="text-center custom-vertical-align-middle">${game.home_score}</td>
                        <td class="text-center custom-vertical-align-middle">${game.away_score}</td>
                        <td class="text-center custom-vertical-align-middle">${game.pred_home_score}</td>
                        <td class="text-center custom-vertical-align-middle">${game.pred_away_score}</td>
                        <td class="text-center custom-vertical-align-middle">${game.pred_winner} ${game.pred_win_pct}</td>
                    `;
                    tableBody.appendChild(row);
                });
            } else {
                tableBody.innerHTML = '<tr><td colspan="8" class="text-center">No Games for the selected date</td></tr>';
            }
        })
        .catch(error => {
            console.error('Error fetching games:', error);
            const tableBody = document.querySelector('#gamesTableBody');
            tableBody.innerHTML = `<tr><td colspan="8" class="text-center">${error.message}</td></tr>`;
        });
}

/**
 * Populates a container with player details.
 * 
 * @param {Array} players - An array of player objects. Each object should have `player_headshot_url`, `player_name`, and `pred_points` properties.
 * @param {HTMLElement} container - The container to populate with player details.
 * @param {number} [limit=5] - The maximum number of players to display. Defaults to 5.
 */
function populatePlayerDetails(players, container, limit = 5) {
    container.innerHTML = ''; // Clear previous content

    players.slice(0, limit).forEach(player => {
        const playerDetailDiv = document.createElement('div');
        playerDetailDiv.className = 'player-detail row d-flex align-items-center mb-3';
        playerDetailDiv.innerHTML = `
            <div class="col-auto">
                <img src="${player.player_headshot_url}" alt="${player.player_name}" class="img-fluid mb-2 player-headshot">
            </div>
            <div class="col">
                <p class="mb-0"><strong>${player.player_name}</strong></p>
                <p class="mb-0">${player.pred_points} PTS</p>
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
    const homeTeamHeader = document.getElementById('homeTeamHeader');
    const awayTeamHeader = document.getElementById('awayTeamHeader');
    const playByPlayBody = document.getElementById('playByPlayBody');

    homeTeamHeader.textContent = home_team;
    awayTeamHeader.textContent = away_team;

    playByPlayBody.innerHTML = ''; // Clear existing content

    if (pbp.length === 0) {
        playByPlayBody.innerHTML = '<tr><td colspan="4" class="text-center no-pbp-data">No Play By Play Logs Available</td></tr>';
    } else {
        pbp.slice(0, limit).forEach((record) => {
            const row = document.createElement('tr');
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
    fetch(`/get-game-data?game_id=${gameId}`)
        .then(response => {
            if (!response.ok) {
                return response.json().then(error => {
                    throw new Error(error.error);
                });
            }
            return response.json();
        })
        .then(data => {
            const game = data[0];

            console.log('Game details:', game);

            const {
                home,
                away,
                home_full_name: homeFullName,
                away_full_name: awayFullName,
                home_logo_url: homeLogoUrl,
                away_logo_url: awayLogoUrl,
                home_score: homeScore,
                away_score: awayScore,
                datetime_display: dateTimeDisplay,
                condensed_pbp: playByPlay,
                home_players: homePlayers,
                away_players: awayPlayers,
                pred_home_score: predictedHomeScore,
                pred_away_score: predictedAwayScore,
                pred_winner: predictedWinner,
                pred_win_pct: predictedWinPercentage
            } = game;

            const modalTitle = document.querySelector('#gameDetailsModalLabel');
            modalTitle.innerHTML = `
                ${homeFullName} <img src="${homeLogoUrl}" alt="${homeFullName}" class="team-logo"> 
                ${homeScore} - ${awayScore} 
                <img src="${awayLogoUrl}" alt="${awayFullName}" class="team-logo"> ${awayFullName} 
                <span class="breakpoint"> - <wbr></span>${dateTimeDisplay}
            `;

            const template = document.querySelector('#gameDetailsTemplate').content.cloneNode(true);
            template.querySelector('#templateHomeTeam').textContent += `${home}`;
            template.querySelector('#templateAwayTeam').textContent += `${away}`;
            template.querySelector('#templateHomeLogo').src = homeLogoUrl;
            template.querySelector('#templateAwayLogo').src = awayLogoUrl;
            template.querySelector('#templatePredictedHomeScore').textContent += `${predictedHomeScore}`;
            template.querySelector('#templatePredictedAwayScore').textContent += `${predictedAwayScore}`;
            template.querySelector('#templatePredictedWinPct').textContent += `${predictedWinPercentage}`;

            populatePlayerDetails(homePlayers, template.querySelector('#homeTeamPlayers'));
            populatePlayerDetails(awayPlayers, template.querySelector('#awayTeamPlayers'));

            const modalBody = document.querySelector('#gameDetailsModal .modal-body');
            modalBody.innerHTML = '';
            modalBody.appendChild(template);

            const winnerLeftIcon = document.getElementById('winnerLeftIcon');
            const winnerRightIcon = document.getElementById('winnerRightIcon');

            if (predictedWinner === home) {
                winnerLeftIcon.style.visibility = 'visible';
                winnerRightIcon.style.visibility = 'hidden';
            } else if (predictedWinner === away) {
                winnerRightIcon.style.visibility = 'visible';
                winnerLeftIcon.style.visibility = 'hidden';
            } else {
                winnerLeftIcon.style.visibility = 'hidden';
                winnerRightIcon.style.visibility = 'hidden';
            }

            populatePlayByPlay(home, away, playByPlay, 100);

            var gameDetailsModal = new bootstrap.Modal(document.getElementById('gameDetailsModal'), {});
            gameDetailsModal.show();
        })
        .catch(error => {
            console.error('Error fetching game details:', error);
        });
}

// Initialize event listeners after DOM content is fully loaded
document.addEventListener("DOMContentLoaded", function () {
    fetchAndUpdateGames();

    document.querySelector('#gamesTableBody').addEventListener('click', function (event) {
        let target = event.target;

        while (target && !target.classList.contains('game-row')) {
            target = target.parentElement;
        }

        if (target && target.classList.contains('game-row')) {
            const gameId = target.getAttribute('data-game-id');
            showGameDetails(gameId);
        }
    });
});
