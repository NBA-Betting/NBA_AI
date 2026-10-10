"""
Static lookup table of NBA arena coordinates, timezones, and altitudes.

Used for computing travel distance and timezone-crossing features in the
L4 game-context model.  Every team abbreviation that appears in
Games.home_team / Games.away_team (regular season + playoffs) is covered
either directly or via HISTORICAL_TO_MODERN mapping.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

# ── Historical → modern franchise mapping (consistent with cache_builder.py) ──

HISTORICAL_TO_MODERN: Dict[str, str] = {
    "NJN": "BKN",
    "SEA": "OKC",
    "NOH": "NOP",
    "NOK": "NOP",
    "CHH": "CHA",
    "VAN": "MEM",
    "WSB": "WAS",
    "SDC": "LAC",
    "KCK": "SAC",
    "BUF": "LAC",
    "NOJ": "NOP",
    "CLT": "CHA",
}


def franchise_abbrs(team_abbr: str) -> list[str]:
    """Every abbreviation a franchise has had, modern first. Games keeps the
    abbreviation of the day (SEA, NJN), so a team's games across a relocation
    need all of them."""
    modern = HISTORICAL_TO_MODERN.get(team_abbr, team_abbr)
    return [modern, *(h for h, m in HISTORICAL_TO_MODERN.items() if m == modern)]


# ── Arena dataclass ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ArenaInfo:
    abbreviation: str  # Team code (e.g. "BOS")
    arena_name: str  # Current arena name
    latitude: float  # Degrees north
    longitude: float  # Degrees east (negative for western hemisphere)
    timezone: str  # IANA timezone string
    utc_offset: float  # Standard-time UTC offset in hours (e.g. -5 for ET)
    altitude_ft: int  # Arena altitude in feet above sea level


# ── Arena lookup (one entry per modern franchise) ────────────────────────────
# Coordinates are arena locations; utc_offset is *standard* time (not DST).
# PHX is the exception: Arizona does not observe DST, so utc_offset == -7 year
# round.

_ARENA_LIST = [
    ArenaInfo("ATL", "State Farm Arena", 33.757, -84.396, "America/New_York", -5, 1050),
    ArenaInfo("BOS", "TD Garden", 42.366, -71.062, "America/New_York", -5, 141),
    ArenaInfo("BKN", "Barclays Center", 40.683, -73.975, "America/New_York", -5, 33),
    ArenaInfo("CHA", "Spectrum Center", 35.225, -80.839, "America/New_York", -5, 751),
    ArenaInfo("CHI", "United Center", 41.881, -87.674, "America/Chicago", -6, 594),
    ArenaInfo(
        "CLE",
        "Rocket Mortgage FieldHouse",
        41.497,
        -81.688,
        "America/New_York",
        -5,
        653,
    ),
    ArenaInfo(
        "DAL", "American Airlines Center", 32.791, -96.810, "America/Chicago", -6, 430
    ),
    ArenaInfo("DEN", "Ball Arena", 39.749, -105.008, "America/Denver", -7, 5280),
    ArenaInfo(
        "DET", "Little Caesars Arena", 42.341, -83.055, "America/Detroit", -5, 600
    ),
    ArenaInfo("GSW", "Chase Center", 37.768, -122.388, "America/Los_Angeles", -8, 0),
    ArenaInfo("HOU", "Toyota Center", 29.751, -95.362, "America/Chicago", -6, 80),
    ArenaInfo(
        "IND",
        "Gainbridge Fieldhouse",
        39.764,
        -86.156,
        "America/Indiana/Indianapolis",
        -5,
        715,
    ),
    ArenaInfo("LAC", "Intuit Dome", 33.944, -118.341, "America/Los_Angeles", -8, 233),
    ArenaInfo(
        "LAL", "Crypto.com Arena", 34.043, -118.267, "America/Los_Angeles", -8, 233
    ),
    ArenaInfo("MEM", "FedExForum", 35.138, -90.051, "America/Chicago", -6, 337),
    ArenaInfo("MIA", "Kaseya Center", 25.781, -80.187, "America/New_York", -5, 6),
    ArenaInfo("MIL", "Fiserv Forum", 43.045, -87.917, "America/Chicago", -6, 617),
    ArenaInfo("MIN", "Target Center", 44.980, -93.276, "America/Chicago", -6, 830),
    ArenaInfo("NOP", "Smoothie King Center", 29.949, -90.082, "America/Chicago", -6, 7),
    ArenaInfo(
        "NYK", "Madison Square Garden", 40.751, -73.994, "America/New_York", -5, 33
    ),
    ArenaInfo("OKC", "Paycom Center", 35.463, -97.515, "America/Chicago", -6, 1201),
    ArenaInfo("ORL", "Amway Center", 28.539, -81.384, "America/New_York", -5, 82),
    ArenaInfo("PHI", "Wells Fargo Center", 39.901, -75.172, "America/New_York", -5, 39),
    ArenaInfo("PHX", "Footprint Center", 33.446, -112.071, "America/Phoenix", -7, 1086),
    ArenaInfo("POR", "Moda Center", 45.532, -122.667, "America/Los_Angeles", -8, 50),
    ArenaInfo(
        "SAC", "Golden 1 Center", 38.580, -121.500, "America/Los_Angeles", -8, 30
    ),
    ArenaInfo("SAS", "Frost Bank Center", 29.427, -98.438, "America/Chicago", -6, 650),
    ArenaInfo("TOR", "Scotiabank Arena", 43.643, -79.379, "America/Toronto", -5, 249),
    ArenaInfo("UTA", "Delta Center", 40.768, -111.901, "America/Denver", -7, 4226),
    ArenaInfo("WAS", "Capital One Arena", 38.898, -77.021, "America/New_York", -5, 0),
]

# Historical-only arenas (franchise has since relocated)
_HISTORICAL_ARENA_LIST = [
    ArenaInfo("NJN", "Izod Center", 40.782, -74.072, "America/New_York", -5, 33),
    ArenaInfo("SEA", "KeyArena", 47.622, -122.354, "America/Los_Angeles", -8, 0),
    ArenaInfo("NOH", "Smoothie King Center", 29.949, -90.082, "America/Chicago", -6, 7),
    ArenaInfo("NOK", "Smoothie King Center", 29.949, -90.082, "America/Chicago", -6, 7),
    ArenaInfo(
        "CHH", "Charlotte Coliseum", 35.205, -80.850, "America/New_York", -5, 751
    ),
    ArenaInfo("VAN", "GM Place", 49.278, -123.109, "America/Vancouver", -8, 0),
]


# ── Primary lookup dict ──────────────────────────────────────────────────────

ARENA_DATA: Dict[str, ArenaInfo] = {}
for _a in _ARENA_LIST + _HISTORICAL_ARENA_LIST:
    ARENA_DATA[_a.abbreviation] = _a


def resolve_arena(team_abbr: str) -> ArenaInfo:
    """Look up arena info, resolving historical abbreviations if needed.

    Raises KeyError if the abbreviation is unknown.
    """
    if team_abbr in ARENA_DATA:
        return ARENA_DATA[team_abbr]
    modern = HISTORICAL_TO_MODERN.get(team_abbr)
    if modern and modern in ARENA_DATA:
        return ARENA_DATA[modern]
    raise KeyError(f"Unknown team abbreviation: {team_abbr!r}")


# ── Distance & timezone helpers ──────────────────────────────────────────────


def haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two points in miles."""
    R = 3959.0  # Earth radius in miles
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    return R * 2 * math.asin(math.sqrt(a))


def timezone_crossing(team_a: str, team_b: str) -> float:
    """Absolute difference in standard-time UTC offsets (hours).

    Useful as a fatigue/jet-lag proxy.  Returns 0 when both teams are
    in the same standard-time zone.
    """
    a = resolve_arena(team_a)
    b = resolve_arena(team_b)
    return abs(a.utc_offset - b.utc_offset)


def get_travel_features(
    prev_game_team: str,
    current_game_team: str,
) -> Dict[str, float]:
    """Compute travel-related features between two arena locations.

    Parameters
    ----------
    prev_game_team : str
        Team abbreviation of the venue of the *previous* game
        (i.e. the home_team of that game).
    current_game_team : str
        Team abbreviation of the venue of the *current* game.

    Returns
    -------
    dict with keys:
        travel_miles : float   — great-circle distance in miles
        tz_cross     : float   — absolute timezone offset difference (hours)
        altitude_delta_ft : float — altitude change (current - previous, signed)
    """
    prev = resolve_arena(prev_game_team)
    curr = resolve_arena(current_game_team)

    dist = haversine_miles(prev.latitude, prev.longitude, curr.latitude, curr.longitude)
    tz = abs(prev.utc_offset - curr.utc_offset)
    alt_delta = curr.altitude_ft - prev.altitude_ft

    return {
        "travel_miles": dist,
        "tz_cross": tz,
        "altitude_delta_ft": float(alt_delta),
    }


# ── Convenience: altitude map compatible with cache_builder.ALTITUDE_MAP ─────

ALTITUDE_MAP: Dict[str, int] = {
    a.abbreviation: a.altitude_ft for a in _ARENA_LIST + _HISTORICAL_ARENA_LIST
}
