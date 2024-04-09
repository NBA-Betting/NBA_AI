/**
 * Fetches games for a specific date and updates the games table.
 * The date is read from the body's `data-query-date` attribute.
 * Each game is added as a new row in the games table.
 * @throws Will throw an error if the fetch operation fails.
 */
function fetchAndUpdateGames() {
    // Get the query date from the body's dataset
    const queryDate = document.body.dataset.queryDate;

    // Fetch games for the query date
    fetch(`/get-games?date=${queryDate}`)
        .then(response => {
            // If the response is not ok, throw an error
            if (!response.ok) {
                return response.json().then(error => {
                    throw new Error(error.error);
                });
            }
            // Otherwise, parse the response as JSON
            return response.json();
        })
        .then(games => {
            // Get the table body element
            const tableBody = document.querySelector('#gamesTableBody');

            // Clear the table body
            tableBody.innerHTML = '';

            // If there are games, create a new row for each game
            if (games.length > 0) {
                games.forEach(game => {
                    console.log('Game:', game);
                    const row = document.createElement('tr');
                    row.className = 'game-row custom-vertical-align-middle';
                    row.setAttribute('data-game-id', game.game_id);

                    // Set the row's inner HTML
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

                    // Append the row to the table body
                    tableBody.appendChild(row);
                });
            } else {
                // If there are no games, display a message
                tableBody.innerHTML = '<tr><td colspan="8" class="text-center">No Games for the selected date</td></tr>';
            }
        })
        .catch(error => {
            // Log any errors
            console.error('Error fetching games:', error);

            // Get the table body element
            const tableBody = document.querySelector('#gamesTableBody');

            // Clear the table body
            tableBody.innerHTML = '';

            // Display the error message
            tableBody.innerHTML = `<tr><td colspan="8" class="text-center">${error.message}</td></tr>`;
        });
}

/**
 * Populates a container with player details.
 * 
 * @param {Array} players - An array of player objects. Each player object should have `player_headshot_url`, `player_name`, and `pred_points` properties.
 * @param {HTMLElement} container - The container to populate with player details.
 * @param {number} [limit=5] - The maximum number of players to display. Defaults to 5.
 */
function populatePlayerDetails(players, container, limit = 5) {
    // Clear the container's previous content
    container.innerHTML = '';

    // Loop through the players, but only up to the specified limit
    players.slice(0, limit).forEach(player => {
        // Create a new div for the player details
        const playerDetailDiv = document.createElement('div');
        playerDetailDiv.className = 'player-detail row d-flex align-items-center mb-3';

        // Populate the div with the player's details
        playerDetailDiv.innerHTML = `
                <div class="col-auto">
                    <img src="${player.player_headshot_url}" alt="${player.player_name}" class="img-fluid mb-2 player-headshot">
                </div>
                <div class="col">
                    <p class="mb-0"><strong>${player.player_name}</strong></p>
                    <p class="mb-0">${player.pred_points} PTS</p>
                </div>
                `;

        // Append the player details div to the container
        container.appendChild(playerDetailDiv);
    });
}

/**
 * Populates a table with play-by-play data.
 * 
 * @param {string} home_team - The name of the home team.
 * @param {string} away_team - The name of the away team.
 * @param {Array} pbp - An array of play-by-play records. Each record should have `time_info`, `description`, `home_score`, and `away_score` properties.
 * @param {number} [limit=Infinity] - The maximum number of records to display. Defaults to Infinity, which means no limit.
 */
function populatePlayByPlay(home_team, away_team, pbp, limit = Infinity) {
    // Get the headers and body of the play-by-play table
    const homeTeamHeader = document.getElementById('homeTeamHeader');
    const awayTeamHeader = document.getElementById('awayTeamHeader');
    const playByPlayBody = document.getElementById('playByPlayBody');

    // Set the team names in the headers
    homeTeamHeader.textContent = home_team;
    awayTeamHeader.textContent = away_team;

    // Clear the table body's existing content
    playByPlayBody.innerHTML = '';

    // Check if the play-by-play array is empty
    if (pbp.length === 0) {
        // If it is, display a message indicating that no data is available
        playByPlayBody.innerHTML = '<tr><td colspan="4" class="text-center no-pbp-data">No Play By Play Logs Available</td></tr>';
    } else {
        // If it's not, add a row to the table for each record, up to the specified limit
        pbp.slice(0, limit).forEach((record) => {
            // Extract the necessary data from the record
            const formattedTime = record.time_info;
            const description = record.description;
            const homeScore = record.home_score;
            const awayScore = record.away_score;

            // Create a new row and populate it with the record's data
            const row = document.createElement('tr');
            row.innerHTML = `
                        <td>${formattedTime}</td>
                        <td>${description}</td>
                        <td>${homeScore}</td>
                        <td>${awayScore}</td>
                    `;

            // Append the row to the table body
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
    // Fetch the game details from the server
    fetch(`/game-details/${gameId}`)
        .then(response => {
            // If the response was not OK, throw an error
            if (!response.ok) {
                return response.json().then(error => {
                    throw new Error(error.error);
                });
            }
            // Otherwise, parse the response as JSON
            return response.json();
        })
        .then(data => {
            // Log the game details
            console.log('Game details:', data);

            // Destructure the necessary data from the response
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
            } = data;

            // Construct the modal header content
            const modalTitle = document.querySelector('#gameDetailsModalLabel');
            modalTitle.innerHTML = `
                        ${homeFullName} <img src="${homeLogoUrl}" alt="${homeFullName}" class="team-logo"> 
                        ${homeScore} - ${awayScore} 
                        <img src="${awayLogoUrl}" alt="${awayFullName}" class="team-logo"> ${awayFullName} 
                        <span class="breakpoint"> - <wbr></span>${dateTimeDisplay}
                    `;

            // Use the template for the modal body
            const template = document.querySelector('#gameDetailsTemplate').content.cloneNode(true);
            template.querySelector('#templateHomeTeam').textContent += `${home}`;
            template.querySelector('#templateAwayTeam').textContent += `${away}`;
            template.querySelector('#templateHomeLogo').src = homeLogoUrl;
            template.querySelector('#templateAwayLogo').src = awayLogoUrl;
            template.querySelector('#templatePredictedHomeScore').textContent += `${predictedHomeScore}`;
            template.querySelector('#templatePredictedAwayScore').textContent += `${predictedAwayScore}`;
            template.querySelector('#templatePredictedWinPct').textContent += `${predictedWinPercentage}`;

            // Populate player details for both teams
            const homePlayersContainer = template.querySelector('#homeTeamPlayers');
            const awayPlayersContainer = template.querySelector('#awayTeamPlayers');
            populatePlayerDetails(homePlayers, homePlayersContainer);
            populatePlayerDetails(awayPlayers, awayPlayersContainer);

            // Clear existing content in the modal body and append the new content
            const modalBody = document.querySelector('#gameDetailsModal .modal-body');
            modalBody.innerHTML = '';
            modalBody.appendChild(template);

            // Determine which icon to show based on the predicted winner
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

            // Populate the play-by-play data
            populatePlayByPlay(home, away, playByPlay, 100);

            // Show the modal
            var gameDetailsModal = new bootstrap.Modal(document.getElementById('gameDetailsModal'), {});
            gameDetailsModal.show();
        })
        .catch(error => {
            // Log any errors that occurred during the fetch
            console.error('Error fetching game details:', error);
        });
}

// This event listener waits for the DOM content to be fully loaded before executing the function
document.addEventListener("DOMContentLoaded", function () {
    // Fetches and updates the games once the DOM is fully loaded
    fetchAndUpdateGames();

    // Selects the body of the games table
    const tableBody = document.querySelector('#gamesTableBody');

    // Adds a click event listener to the table body
    tableBody.addEventListener('click', function (event) {
        // Initializes the target as the element that was clicked
        let target = event.target;

        // Traverses up the DOM tree until it finds an element with the 'game-row' class or runs out of parent elements
        while (target != null && !target.classList.contains('game-row')) {
            target = target.parentElement;
        }

        // If a 'game-row' element was found, it gets the game ID from the element's data attributes and shows the game details
        if (target && target.classList.contains('game-row')) {
            const gameId = target.getAttribute('data-game-id');
            showGameDetails(gameId);
        }
    });
});