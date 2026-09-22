"""Travel context features (game_context dims 8-11).

Shared by the Phase B cache builder (training) and the live TeamFeatures
computation (inference) so both fill the same values.
"""

import sqlite3

from src.phase5.arena_data import HISTORICAL_TO_MODERN, haversine_miles, resolve_arena

# Team abbreviation fallback for historical + special games
KNOWN_TEAM_ABBREVS = set(
    [
        "ATL",
        "BOS",
        "BKN",
        "CHA",
        "CHI",
        "CLE",
        "DAL",
        "DEN",
        "DET",
        "GSW",
        "HOU",
        "IND",
        "LAC",
        "LAL",
        "MEM",
        "MIA",
        "MIL",
        "MIN",
        "NOP",
        "NYK",
        "OKC",
        "ORL",
        "PHI",
        "PHX",
        "POR",
        "SAC",
        "SAS",
        "TOR",
        "UTA",
        "WAS",
        "NJN",
        "SEA",
        "NOH",
        "NOK",
        "CHH",
        "VAN",
    ]
)


def compute_travel_features(
    conn: sqlite3.Connection,
    game_id: str,
    home_team: str,
    away_team: str,
    date_time_utc: str,
) -> dict[str, float]:
    """Compute travel distance and timezone crossing for home and away teams.

    Looks up each team's previous game location to compute travel.
    """
    result = {
        "travel_dist_home": 0.0,
        "travel_dist_away": 0.0,
        "tz_crossings_home": 0.0,
        "tz_crossings_away": 0.0,
    }

    for team_abbr, prefix in [(home_team, "home"), (away_team, "away")]:
        # Resolve historical abbreviation
        modern = HISTORICAL_TO_MODERN.get(team_abbr, team_abbr)
        if modern not in KNOWN_TEAM_ABBREVS:
            continue

        # Find this team's previous game (home or away)
        prev = conn.execute(
            """
            SELECT game_id, home_team, date_time_utc
            FROM Games
            WHERE status = 3
              AND (home_team = ? OR away_team = ?)
              AND date_time_utc < ?
            ORDER BY date_time_utc DESC
            LIMIT 1
            """,
            (team_abbr, team_abbr, date_time_utc),
        ).fetchone()

        if prev is None:
            continue

        prev_game_id, prev_home, _ = prev
        # The venue is the home team's arena
        prev_venue = prev_home
        current_venue = home_team  # current game's venue = home team arena

        try:
            prev_arena = resolve_arena(prev_venue)
            curr_arena = resolve_arena(current_venue)
            dist = haversine_miles(
                prev_arena.latitude,
                prev_arena.longitude,
                curr_arena.latitude,
                curr_arena.longitude,
            )
            tz_diff = abs(prev_arena.utc_offset - curr_arena.utc_offset)
            result[f"travel_dist_{prefix}"] = dist
            result[f"tz_crossings_{prefix}"] = tz_diff
        except KeyError:
            pass  # Unknown arena, leave as 0

    return result
