import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

from utils import validate_game_ids

logging.basicConfig(level=logging.WARNING)


import requests


def get_pbp_3(game_ids):
    if isinstance(game_ids, str):
        game_ids = [game_ids]

    validate_game_ids(game_ids)

    headers = {
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Accept-Language": "en-US,en;q=0.9",
        "Cache-Control": "max-age=0",
        "Connection": "keep-alive",
        "Host": "cdn.nba.com",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36",
    }

    base_url = "https://cdn.nba.com/static/json/liveData/playbyplay/playbyplay_{}.json"
    results = {}
    session = requests.Session()  # Use a session for connection pooling

    for game_id in game_ids:
        try:
            response = session.get(
                base_url.format(game_id), headers=headers, timeout=30
            )
            response.raise_for_status()
            data = response.json()

            actions = data.get("game", {}).get("actions", [])
            actions_sorted = sorted(actions, key=lambda x: x["actionNumber"])

            results[game_id] = actions_sorted
        except requests.exceptions.HTTPError as http_err:
            if http_err.response.status_code == 403:
                logging.warning(
                    f"Game Id: {game_id} - HTTP error occurred: {http_err}. Game may not have started yet or is from the distant past."
                )
            else:
                logging.warning(
                    f"Game Id: {game_id} - HTTP error occurred: {http_err}."
                )
            results[game_id] = []
        except Exception as e:
            logging.warning(f"Game Id: {game_id} - PBP API call error: {str(e)}.")
            results[game_id] = []

    return results


def get_pbp_stats_endpoint(game_ids):
    if isinstance(game_ids, str):
        game_ids = [game_ids]  # Convert to list if only a single ID is provided

    # Validate all game_ids before processing
    validate_game_ids(game_ids)

    results = {}

    for game_id in game_ids:
        try:
            # Use the PlayByPlay class to retrieve the play-by-play logs for the game
            pbp = playbyplayv3.PlayByPlayV3(game_id=game_id).get_dict()["game"][
                "actions"
            ]
            pbp_logs_sorted = sorted(pbp, key=lambda x: x["actionNumber"])
            results[game_id] = pbp_logs_sorted
        except Exception as e:
            # If an error occurs, log the error and include an empty list for this game_id
            logging.warning(
                f"Game Id {game_id} - PBP API call error. Game may not have started yet."
            )
            results[game_id] = []

    return results


def create_game_states_2(pbp_logs, home, away, game_id, game_date):
    try:
        if not pbp_logs:
            return []

        pbp_logs = sorted(pbp_logs, key=lambda x: x["actionNumber"])
        pbp_logs = [log for log in pbp_logs if "description" in log]

        game_states = []
        players = {"home": {}, "away": {}}

        # Initialize current scores
        current_home_score = 0
        current_away_score = 0

        for i, row in enumerate(pbp_logs):
            if row.get("personId") not in [None, ""] and row.get("playerNameI") not in [
                None,
                "",
            ]:
                team = "home" if row["teamTricode"] == home else "away"
                player_id = row["personId"]
                player_name = row["playerNameI"]

                if player_id not in players[team]:
                    players[team][player_id] = {"name": player_name, "points": 0}

                # Extract points from description
                match = re.search(r"\((\d+) PTS\)", row.get("description", ""))
                if match:
                    points = int(match.group(1))
                    players[team][player_id]["points"] = points

            # Update current scores if new scores are available
            if row.get("scoreHome") not in ["", "0"]:
                current_home_score = int(row["scoreHome"])
            if row.get("scoreAway") not in ["", "0"]:
                current_away_score = int(row["scoreAway"])

            current_game_state = {
                "game_id": game_id,
                "play_id": int(row["actionNumber"]),
                "game_date": game_date,
                "home": home,
                "away": away,
                "remaining_time": row["clock"],
                "period": int(row["period"]),
                "home_score": current_home_score,
                "away_score": current_away_score,
                "total": current_home_score + current_away_score,
                "home_margin": current_home_score - current_away_score,
                "is_final_state": i
                == len(pbp_logs) - 1,  # Set is_final_state for the last log
                "players_data": deepcopy(players),
            }

            game_states.append(current_game_state)

        return game_states

    except Exception as e:
        logging.error(f"Game Id {game_id} - Failed to create game states. {e}")
        return []


if __name__ == "__main__":
    # g1 = get_pbp_3("0022000001")

    # g2 = get_pbp_3("0022000919")
    # print(g2)

    # g3 = get_pbp_3("0022001170")
    # print(g3)

    games = ["0022000001", "0022000919", "0022001170", "0011000017"]
    more_games = [
        "0022200560",
        "0022200561",
        "0022200562",
        "0022200563",
        "0022200564",
        "0022200565",
        "0022200566",
        "0022200567",
        "0022200568",
        "0022200569",
    ] + games

    start_time = time.time()
    pbp = get_pbp_3c(more_games)
    end_time = time.time()

    for game_id, pbp_data in pbp.items():
        print(f"Game ID: {game_id}")
        print(f"Number of actions: {len(pbp_data)}")
        print()

    print(f"Total execution time: {end_time - start_time:.2f} seconds")
    print(
        f"Excecution time per game: {(end_time - start_time) / len(more_games):.2f} seconds"
    )
