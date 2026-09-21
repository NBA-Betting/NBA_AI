"""
Betting data collection module with 3-tier data source strategy.

=============================================================================
3-TIER FETCHING STRATEGY
=============================================================================

Tier 1 - ESPN API (Live Window: -7 to +2 days):
  - Primary source for real-time and recent data
  - Provides DraftKings odds via summary endpoint
  - Includes spreads, totals, moneylines with odds (-110, etc.)
  - Used automatically by web app for current games

Tier 2 - Covers.com Matchups (On-Demand Finalization):
  - Used for games >7 days old that need closing line finalization
  - Fetches by date (1 API call per date with unfinalized games)
  - Provides closing spreads and totals (no odds)
  - Triggered automatically when ESPN window expires

Tier 3 - Covers.com Team Schedules (Historical Backfill):
  - Bulk backfill via CLI with --backfill flag
  - Fetches all home games per team (30 API calls per season)
  - Best for populating historical seasons
  - Rate-limited to respect Covers.com

=============================================================================
SMART CACHING STRATEGY
=============================================================================

Tier 1 ESPN fetches use intelligent caching based on game status:

1. Scheduled Games (not started):
   - Cache for 1 hour
   - Rationale: Lines update slowly pre-game, but we want fresh data
   - Balances freshness with API efficiency for day-of games

2. In-Progress Games (live):
   - Cache for 6 hours
   - Rationale: Closing lines LOCK at tipoff - no more changes during game
   - One fetch per game captures final closing line

3. Completed Games (final):
   - Cache permanently once closing lines fetched (lines_finalized=1)
   - Rationale: Closing lines never change after game ends
   - Verify once post-game, then cache forever

=============================================================================
SINGLE ROW PER GAME DESIGN
=============================================================================

Each game has ONE row in the Betting table that gets updated over time:
  1. Created with opening lines (Tier 1 ESPN, ~1-2 days before game)
  2. Updated with current lines (Tier 1 ESPN, as game approaches)
  3. Finalized with closing lines (Tier 1 ESPN or Tier 2 Covers)

The lines_finalized flag indicates we have the final closing line and
won't update this game anymore.
"""

import logging
import sqlite3
from datetime import date, datetime, timedelta, timezone
from typing import Optional

import requests
from tqdm import tqdm

from src.config import config
from src.database import create_connection, get_db
from src.utils import (
    NBATeamConverter,
    StageLogger,
    get_eastern_tz,
    log_execution_time,
)

logger = logging.getLogger(__name__)

DB_PATH = config["database"]["path"]

# ESPN API configuration
ESPN_SCOREBOARD_URL = (
    "http://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
)
ESPN_SUMMARY_URL = "http://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary"

# Fetching thresholds
FUTURE_CUTOFF_DAYS = 2  # Don't fetch for games more than 2 days out
ESPN_LOOKBACK_DAYS = 7  # ESPN API has data for ~7 days back
COVERS_FINALIZATION_DELAY_DAYS = 1  # Wait 1 day after ESPN window before using Covers

# Caching thresholds
CACHE_HOURS = 1  # Cache all non-finalized games for 1 hour, then re-check for new lines


def _filter_failed_covers_dates(
    dates: list[str], conn: sqlite3.Connection
) -> list[str]:
    """
    Filter out dates that were recently attempted but had no matches to avoid repeated API calls.

    Args:
        dates: List of date strings to potentially fetch from Covers
        conn: Database connection

    Returns:
        Filtered list of dates that should be attempted
    """
    # Create table to track failed attempts if it doesn't exist
    conn.execute("""
        CREATE TABLE IF NOT EXISTS CoversAttempts (
            date_str TEXT PRIMARY KEY,
            last_attempt_datetime TEXT NOT NULL,
            match_count INTEGER DEFAULT 0
        )
    """)

    # Filter out dates attempted recently:
    # - Zero matches: cache for 6 hours (date mismatch, try again later)
    # - Successful matches: cache for 24 hours (data collected, avoid redundant calls)
    filtered_dates = []
    for date_str in dates:
        cursor = conn.execute(
            """
            SELECT match_count, last_attempt_datetime 
            FROM CoversAttempts 
            WHERE date_str = ?
        """,
            (date_str,),
        )

        result = cursor.fetchone()
        if result:
            match_count = result[0]
            last_attempt = result[1]

            # Cache successful attempts (>0 matches) for 24 hours
            if match_count > 0:
                cursor = conn.execute(
                    """
                    SELECT 1 FROM CoversAttempts 
                    WHERE date_str = ? 
                    AND last_attempt_datetime > datetime('now', '-24 hours')
                """,
                    (date_str,),
                )
                if cursor.fetchone():
                    logger.debug(
                        f"Skipping Covers fetch for {date_str} - {match_count} matches found in recent attempt"
                    )
                    continue

            # Cache failed attempts (0 matches) for 6 hours
            elif match_count == 0:
                cursor = conn.execute(
                    """
                    SELECT 1 FROM CoversAttempts 
                    WHERE date_str = ? 
                    AND last_attempt_datetime > datetime('now', '-6 hours')
                """,
                    (date_str,),
                )
                if cursor.fetchone():
                    logger.debug(
                        f"Skipping Covers fetch for {date_str} - no matches found in recent attempt"
                    )
                    continue

        filtered_dates.append(date_str)

    return filtered_dates


def _record_covers_attempt(date_str: str, match_count: int, conn: sqlite3.Connection):
    """Record a Covers fetch attempt with match count for future filtering."""
    conn.execute(
        """
        INSERT OR REPLACE INTO CoversAttempts (date_str, last_attempt_datetime, match_count)
        VALUES (?, datetime('now'), ?)
    """,
        (date_str, match_count),
    )
    conn.commit()


def _eastern_date(date_time_utc: str) -> str:
    """Return the US/Eastern calendar date (YYYY-MM-DD) of a Games.date_time_utc value.

    Covers matchups pages are keyed by Eastern date; most games tip off on the
    following day in UTC.
    """
    utc_dt = datetime.strptime(date_time_utc, "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=timezone.utc
    )
    return utc_dt.astimezone(get_eastern_tz()).date().isoformat()


def _get_current_season(now: datetime) -> str:
    """
    Determine current NBA season based on date.
    NBA season runs Oct-June, so Oct-Dec is start year, Jan-Sept is end year.

    Args:
        now: Current datetime

    Returns:
        Season string like "2024-2025"
    """
    year = now.year
    month = now.month

    # Oct-Dec: current year is start of season (e.g., Oct 2024 → 2024-2025)
    if month >= 10:
        return f"{year}-{year + 1}"
    # Jan-Sept: current year is end of season (e.g., Mar 2025 → 2024-2025)
    else:
        return f"{year - 1}-{year}"


# =============================================================================
# Schema Definition - Opening/Current/Closing Separation
# =============================================================================

BETTING_SCHEMA = """
CREATE TABLE IF NOT EXISTS Betting (
    -- Primary key
    game_id TEXT PRIMARY KEY,
    
    -- ESPN mapping
    espn_event_id TEXT,
    
    -- ESPN Opening Lines (from ESPN 'open' field)
    espn_opening_spread REAL,
    espn_opening_spread_home_odds INTEGER,
    espn_opening_spread_away_odds INTEGER,
    espn_opening_total REAL,
    espn_opening_over_odds INTEGER,
    espn_opening_under_odds INTEGER,
    espn_opening_ml_home INTEGER,
    espn_opening_ml_away INTEGER,
    
    -- ESPN Current Lines (from ESPN 'close' field when game not completed)
    espn_current_spread REAL,
    espn_current_spread_home_odds INTEGER,
    espn_current_spread_away_odds INTEGER,
    espn_current_total REAL,
    espn_current_over_odds INTEGER,
    espn_current_under_odds INTEGER,
    espn_current_ml_home INTEGER,
    espn_current_ml_away INTEGER,
    
    -- ESPN Closing Lines (from ESPN 'close' field ONLY when game completed)
    espn_closing_spread REAL,
    espn_closing_spread_home_odds INTEGER,
    espn_closing_spread_away_odds INTEGER,
    espn_closing_total REAL,
    espn_closing_over_odds INTEGER,
    espn_closing_under_odds INTEGER,
    espn_closing_ml_home INTEGER,
    espn_closing_ml_away INTEGER,
    
    -- Covers Closing Lines (always closing, from Covers.com)
    covers_closing_spread REAL,
    covers_closing_total REAL,
    
    -- Results (from Covers or calculated)
    spread_result TEXT,  -- 'W', 'L', 'P' (home team perspective)
    ou_result TEXT,      -- 'O', 'U', 'P'
    
    -- Metadata
    lines_finalized INTEGER NOT NULL DEFAULT 0,  -- 1 = have verified closing lines
    
    -- Timestamps
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    
    FOREIGN KEY (game_id) REFERENCES Games(game_id)
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_betting_lines_finalized ON Betting(lines_finalized);
"""


def create_betting_tables(conn: Optional[sqlite3.Connection] = None) -> None:
    """Create Betting table (single row per game schema)."""
    close_conn = False
    if conn is None:
        conn = create_connection()
        close_conn = True

    try:
        conn.executescript(BETTING_SCHEMA)
        conn.commit()
        logger.debug("Betting table created/verified")
    finally:
        if close_conn:
            conn.close()


# =============================================================================
# ESPN API Functions
# =============================================================================


def get_espn_event_id(
    game_id: str,
    game_date: str,
    home_team: str,
    away_team: str,
    game_time_utc: str = None,
) -> Optional[str]:
    """
    Get ESPN event ID for an NBA game.

    Uses ESPNGameMapping cache if available, otherwise fetches from ESPN scoreboard.

    Note: ESPN uses US Eastern dates, but our DB stores UTC. Games in early UTC hours
    (00:00-08:00) are typically the previous day's evening games in US time.
    This function tries both the given date and the previous day.

    Args:
        game_id: NBA game ID
        game_date: Game date in YYYY-MM-DD format (UTC)
        home_team: Home team tricode (e.g., "BOS")
        away_team: Away team tricode (e.g., "LAL")
        game_time_utc: Optional game time (HH:MM:SS) to determine if previous day check needed

    Returns:
        ESPN event ID or None if not found
    """
    # Check cache first
    with get_db() as conn:
        cursor = conn.execute(
            "SELECT espn_event_id FROM ESPNGameMapping WHERE nba_game_id = ?",
            (game_id,),
        )
        row = cursor.fetchone()
        if row:
            return row[0]

    # Determine which dates to try
    # Games at 00:00-08:00 UTC are likely previous day's evening games in US time
    dates_to_try = [game_date]
    if game_time_utc:
        hour = int(game_time_utc.split(":")[0])
        if hour < 8:  # Early UTC = previous day in US
            prev_date = (
                datetime.strptime(game_date, "%Y-%m-%d") - timedelta(days=1)
            ).strftime("%Y-%m-%d")
            dates_to_try = [prev_date, game_date]  # Try previous day first

    for try_date in dates_to_try:
        date_formatted = try_date.replace("-", "")
        url = f"{ESPN_SCOREBOARD_URL}?dates={date_formatted}"

        try:
            resp = requests.get(url, timeout=5)
            resp.raise_for_status()
            data = resp.json()

            events = data.get("events", [])
            for event in events:
                competitors = event.get("competitions", [{}])[0].get("competitors", [])
                if len(competitors) >= 2:
                    # ESPN: competitors[0] is home, competitors[1] is away
                    espn_home = competitors[0].get("team", {}).get("abbreviation", "")
                    espn_away = competitors[1].get("team", {}).get("abbreviation", "")

                    # Normalize team abbreviations (ESPN uses different codes sometimes)
                    if _teams_match(home_team, espn_home) and _teams_match(
                        away_team, espn_away
                    ):
                        espn_id = event["id"]

                        # Cache the mapping
                        _cache_espn_mapping(
                            game_id, espn_id, game_date, home_team, away_team
                        )
                        return espn_id

        except requests.RequestException as e:
            logger.error(f"ESPN scoreboard request failed for {try_date}: {e}")
            continue

    logger.warning(f"No ESPN match found for {away_team}@{home_team} on {game_date}")
    return None


def _teams_match(nba_code: str, espn_code: str) -> bool:
    """Check if team codes match using comprehensive NBATeamConverter."""
    try:
        # Use NBATeamConverter for comprehensive team normalization
        nba_normalized = NBATeamConverter.get_abbreviation(nba_code)
        espn_normalized = NBATeamConverter.get_abbreviation(espn_code)
        return nba_normalized == espn_normalized
    except ValueError:
        # Fallback to simple string comparison if normalization fails
        return nba_code.upper() == espn_code.upper()


def _cache_espn_mapping(
    game_id: str, espn_id: str, game_date: str, home_team: str, away_team: str
) -> None:
    """Cache ESPN game mapping for future lookups."""
    try:
        with get_db() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO ESPNGameMapping 
                (nba_game_id, espn_event_id, game_date, home_team, away_team)
                VALUES (?, ?, ?, ?, ?)
            """,
                (game_id, espn_id, game_date, home_team, away_team),
            )
            conn.commit()
    except sqlite3.Error as e:
        logger.warning(f"Failed to cache ESPN mapping: {e}")


def fetch_espn_betting_data(espn_event_id: str, home_team: str) -> Optional[dict]:
    """
    Fetch betting data from ESPN summary endpoint.

    ESPN API provides structured opening and closing/current lines:
    - Pre-game: Uses nested structure with 'open' and 'close' fields
    - Post-game: Uses flat structure with 'spread', 'overUnder', 'homeTeamOdds', etc.

    Args:
        espn_event_id: ESPN event ID
        home_team: Home team tricode (for spread conversion to home perspective)

    Returns:
        Dict with opening and current/closing lines, or None if unavailable
        Example:
        {
            "opening": {"spread": -7.5, "spread_home_odds": -110, ...},
            "current_or_closing": {"spread": -8.0, "spread_home_odds": -115, ...}
        }
    """
    url = f"{ESPN_SUMMARY_URL}?event={espn_event_id}"

    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        data = resp.json()

        pickcenter = data.get("pickcenter", [])
        if not pickcenter:
            return None

        # Prefer DraftKings, fall back to first available
        dk_odds = next(
            (
                p
                for p in pickcenter
                if p.get("provider", {}).get("name") == "Draft Kings"
            ),
            None,
        )
        odds_data = dk_odds if dk_odds else pickcenter[0]

        result = {}

        # Extract OPENING lines (from 'open' field in nested structure)
        opening = _extract_espn_lines(odds_data, home_team, line_type="open")
        if opening:
            result["opening"] = opening

        # Extract CURRENT/CLOSING lines
        # First try nested 'close' structure (pre-game format)
        current_or_closing = _extract_espn_lines(
            odds_data, home_team, line_type="close"
        )

        # If nested structure is empty, try flat structure (post-game format)
        if not current_or_closing:
            current_or_closing = _extract_espn_flat_lines(odds_data)

        if current_or_closing:
            result["current_or_closing"] = current_or_closing

        return result if result else None

    except requests.RequestException as e:
        logger.error(f"ESPN summary request failed for event {espn_event_id}: {e}")
        return None


def _extract_espn_lines(
    odds_data: dict, home_team: str, line_type: str
) -> Optional[dict]:
    """
    Extract betting lines from ESPN odds data for a specific line type (open or close).

    ESPN structure:
    {
        "pointSpread": {
            "home": {"open": {...}, "close": {...}},
            "away": {"open": {...}, "close": {...}}
        },
        "total": {
            "home": {"open": {...}, "close": {...}},
            "away": {"open": {...}, "close": {...}}
        },
        "moneyline": {
            "home": {"open": {...}, "close": {...}},
            "away": {"open": {...}, "close": {...}}
        }
    }

    Args:
        odds_data: ESPN pickcenter odds object
        home_team: Home team tricode for spread conversion
        line_type: "open" or "close"

    Returns:
        Dict with spread, total, moneyline, and odds, or None
    """
    lines = {}

    # Extract spread
    point_spread = odds_data.get("pointSpread", {})
    home_spread_data = point_spread.get("home", {}).get(line_type, {})
    away_spread_data = point_spread.get("away", {}).get(line_type, {})

    if home_spread_data.get("line") is not None:
        # ESPN provides spread from home perspective (negative = favored)
        lines["spread"] = float(home_spread_data["line"])
        lines["spread_home_odds"] = _convert_odds(home_spread_data.get("odds"))
        lines["spread_away_odds"] = _convert_odds(away_spread_data.get("odds"))

    # Extract total
    total_data = odds_data.get("total", {})
    over_data = total_data.get("home", {}).get(line_type, {})  # home = over
    under_data = total_data.get("away", {}).get(line_type, {})  # away = under

    if over_data.get("line") is not None:
        lines["total"] = float(over_data["line"])
        lines["over_odds"] = _convert_odds(over_data.get("odds"))
        lines["under_odds"] = _convert_odds(under_data.get("odds"))

    # Extract moneylines
    moneyline = odds_data.get("moneyline", {})
    home_ml_data = moneyline.get("home", {}).get(line_type, {})
    away_ml_data = moneyline.get("away", {}).get(line_type, {})

    if home_ml_data.get("odds") is not None:
        lines["ml_home"] = _convert_odds(home_ml_data.get("odds"))
    if away_ml_data.get("odds") is not None:
        lines["ml_away"] = _convert_odds(away_ml_data.get("odds"))

    return lines if lines else None


def _extract_espn_flat_lines(odds_data: dict) -> Optional[dict]:
    """
    Extract betting lines from ESPN flat structure (used for completed games).

    Post-game ESPN API uses a flat structure instead of nested open/close:
    {
        "spread": -4.5,
        "overUnder": 241.5,
        "overOdds": -112.0,
        "underOdds": -108.0,
        "homeTeamOdds": {"spreadOdds": -102.0, "moneyLine": -180, ...},
        "awayTeamOdds": {"spreadOdds": -118.0, "moneyLine": 150, ...}
    }

    Args:
        odds_data: ESPN pickcenter odds object

    Returns:
        Dict with spread, total, moneyline, and odds, or None if unavailable
    """
    lines = {}

    # Extract spread (already in home perspective: negative = home favored)
    if odds_data.get("spread") is not None:
        lines["spread"] = float(odds_data["spread"])

    # Extract spread odds from team odds
    home_odds = odds_data.get("homeTeamOdds", {})
    away_odds = odds_data.get("awayTeamOdds", {})

    if home_odds.get("spreadOdds") is not None:
        lines["spread_home_odds"] = _convert_odds(home_odds["spreadOdds"])
    if away_odds.get("spreadOdds") is not None:
        lines["spread_away_odds"] = _convert_odds(away_odds["spreadOdds"])

    # Extract total (over/under)
    if odds_data.get("overUnder") is not None:
        lines["total"] = float(odds_data["overUnder"])

    if odds_data.get("overOdds") is not None:
        lines["over_odds"] = _convert_odds(odds_data["overOdds"])
    if odds_data.get("underOdds") is not None:
        lines["under_odds"] = _convert_odds(odds_data["underOdds"])

    # Extract moneylines
    if home_odds.get("moneyLine") is not None:
        lines["ml_home"] = _convert_odds(home_odds["moneyLine"])
    if away_odds.get("moneyLine") is not None:
        lines["ml_away"] = _convert_odds(away_odds["moneyLine"])

    return lines if lines else None


def _parse_spread_from_details(
    details: Optional[str], home_team: str
) -> Optional[float]:
    """
    Parse spread from ESPN 'details' field and convert to home team perspective.

    ESPN provides spread as: 'TEAM_ABBREV +/-POINTS' (e.g., 'BOS -8.5', 'LAL +3')
    We convert to home team perspective: negative = home favored

    Args:
        details: Spread string from ESPN (e.g., 'BOS -8.5')
        home_team: Home team tricode (e.g., 'BOS')

    Returns:
        Float spread from home perspective, or None if parsing fails

    Examples:
        details='BOS -8.5', home_team='BOS' → -8.5 (home favored by 8.5)
        details='BOS -8.5', home_team='LAL' → +8.5 (away favored, home underdog)
        details='LAL +3', home_team='LAL' → +3 (home underdog by 3)
        details='LAL +3', home_team='BOS' → -3 (away underdog, home favored)
    """
    if not details:
        return None

    try:
        # Parse 'BOS -8.5' format
        parts = details.strip().split()
        if len(parts) != 2:
            logger.warning(f"Unexpected spread format: '{details}'")
            return None

        team_abbrev = parts[0]
        spread_str = parts[1]  # e.g., '-8.5' or '+3'

        # Convert to float (handles both '+3' and '-8.5')
        spread_value = float(spread_str)

        # Convert to home team perspective
        # If the team in the details is the home team, use value as-is
        # If it's the away team, flip the sign
        if team_abbrev == home_team:
            return spread_value
        else:
            return -spread_value

    except (ValueError, IndexError) as e:
        logger.warning(f"Failed to parse spread from '{details}': {e}")
        return None


def _convert_odds(odds_value) -> Optional[int]:
    """Convert odds to integer (ESPN sometimes returns floats)."""
    if odds_value is None:
        return None
    try:
        return int(float(odds_value))
    except (ValueError, TypeError):
        return None


# =============================================================================
# Core Betting Data Functions
# =============================================================================


def should_fetch_betting(
    game_datetime: datetime, game_status: str, now: Optional[datetime] = None
) -> tuple[bool, str]:
    """
    Determine if we should fetch betting data for a game.

    Strategy:
    - Future > 2 days: Skip (odds not available yet)
    - Future 1-2 days: Fetch from ESPN (opening lines)
    - Game started or completed: Fetch closing lines
    - After game > 7 days old: Skip (ESPN no longer has data)

    Args:
        game_datetime: Game scheduled start time (UTC)
        game_status: Game status ("Scheduled", "In Progress", "Final", etc.)
        now: Current time (for testing), defaults to datetime.utcnow()

    Returns:
        Tuple of (should_fetch: bool, source: str)
        source can be: "espn", "too_far_future", "too_old", "skip"
    """
    if now is None:
        now = datetime.now(timezone.utc)

    # Make game_datetime timezone-aware if it isn't
    if game_datetime.tzinfo is None:
        game_datetime = game_datetime.replace(tzinfo=timezone.utc)

    delta = game_datetime - now
    days_away = delta.total_seconds() / 86400  # Convert to days

    # Game is too far in the future
    if days_away > FUTURE_CUTOFF_DAYS:
        return False, "too_far_future"

    # Game is too old - ESPN no longer has data (Covers.com scraping implemented separately)
    if game_status == 3 and days_away < -ESPN_LOOKBACK_DAYS:  # Final
        return False, "too_old"

    # Game is within ESPN's data window
    if (
        game_status in (2, 3)  # In Progress or Final
        or days_away >= -ESPN_LOOKBACK_DAYS
    ):
        return True, "espn"

    # Default: skip
    return False, "skip"


def fetch_betting_for_game(
    game_id: str,
    game_date: str,
    home_team: str,
    away_team: str,
    game_datetime: datetime,
    game_status: str,
) -> Optional[dict]:
    """
    Fetch betting data for a single game from ESPN API (Tier 1).

    Returns structured data with opening and current/closing lines.
    The 'current_or_closing' field contains:
    - Current lines (for scheduled/in-progress games)
    - Closing lines (for completed games)

    Data availability:
    - Upcoming games (1-2 days): Opening/current lines
    - Recent completed games (< 7 days): Opening/closing lines
    - Older games (> 7 days): Returns None (use Tier 2/3 Covers)

    Args:
        game_id: NBA game ID
        game_date: Game date (YYYY-MM-DD)
        home_team: Home team tricode
        away_team: Away team tricode
        game_datetime: Game start time (UTC)
        game_status: Game status (used to determine if 'close' = current or closing)

    Returns:
        Dict with game_id, espn_event_id, game_status, opening, current_or_closing
    """
    # Check if we should fetch from ESPN
    should_fetch, source = should_fetch_betting(game_datetime, game_status)
    if not should_fetch:
        logger.debug(f"Skipping ESPN betting for {game_id}: {source}")
        return None

    # Get ESPN event ID
    # Extract time as string (HH:MM:SS) from datetime object for timezone checking
    game_time_str = (
        game_datetime.strftime("%H:%M:%S")
        if isinstance(game_datetime, datetime)
        else game_datetime
    )
    espn_id = get_espn_event_id(
        game_id, game_date, home_team, away_team, game_time_utc=game_time_str
    )
    if not espn_id:
        logger.warning(f"No ESPN ID found for {game_id}")
        return None

    # Fetch from ESPN
    espn_lines = fetch_espn_betting_data(espn_id, home_team)

    if not espn_lines:
        logger.debug(f"No betting data available for {game_id} (ESPN: {espn_id})")
        return None

    # Build result with metadata
    result = {
        "game_id": game_id,
        "espn_event_id": espn_id,
        "game_status": game_status,  # Used by save logic to determine field mapping
    }

    # Add opening lines if available
    if "opening" in espn_lines:
        result["opening"] = espn_lines["opening"]

    # Add current/closing lines if available
    if "current_or_closing" in espn_lines:
        result["current_or_closing"] = espn_lines["current_or_closing"]

    # Determine if lines should be finalized
    # Only finalized when game completed AND we have closing lines
    if game_status == 3 and "current_or_closing" in result:  # Final
        result["lines_finalized"] = 1
    else:
        result["lines_finalized"] = 0

    return result


@log_execution_time(average_over="betting_data_list")
def save_betting_data(
    betting_data_list: list[dict], conn: Optional[sqlite3.Connection] = None
) -> int:
    """
    Save betting data to Betting table with opening/current/closing separation.

    Logic:
    - ESPN data with 'opening' → espn_opening_* fields
    - ESPN data with 'current_or_closing' + game not completed → espn_current_* fields
    - ESPN data with 'current_or_closing' + game completed → espn_closing_* fields
    - Covers data → covers_closing_* fields

    Uses UPSERT pattern: inserts new rows, updates existing ones.

    Args:
        betting_data_list: List of betting data dicts
          ESPN format: {game_id, espn_event_id, game_status, opening: {...}, current_or_closing: {...}}
          Covers format: {game_id, covers_closing_spread, covers_closing_total, spread_result, ou_result}
        conn: Optional database connection

    Returns:
        Number of rows inserted/updated
    """
    if not betting_data_list:
        return 0

    close_conn = False
    if conn is None:
        conn = create_connection()
        close_conn = True

    saved = 0
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    try:
        for data in betting_data_list:
            game_id = data["game_id"]

            # Check if row exists
            cursor = conn.execute(
                "SELECT lines_finalized FROM Betting WHERE game_id = ?",
                (game_id,),
            )
            existing = cursor.fetchone()

            # Determine data source and build field mappings
            is_espn = "espn_event_id" in data
            is_covers = (
                "covers_closing_spread" in data or "covers_closing_total" in data
            )
            is_placeholder = (
                not is_espn and not is_covers
            )  # No data found, just timestamp

            if is_espn:
                # ESPN data - separate opening, current, closing
                game_status = data.get("game_status", 0)
                is_completed = game_status == 3  # Final

                # Build UPDATE/INSERT fields
                fields = {}
                fields["espn_event_id"] = data.get("espn_event_id")

                # Opening lines
                if "opening" in data:
                    opening = data["opening"]
                    fields["espn_opening_spread"] = opening.get("spread")
                    fields["espn_opening_spread_home_odds"] = opening.get(
                        "spread_home_odds"
                    )
                    fields["espn_opening_spread_away_odds"] = opening.get(
                        "spread_away_odds"
                    )
                    fields["espn_opening_total"] = opening.get("total")
                    fields["espn_opening_over_odds"] = opening.get("over_odds")
                    fields["espn_opening_under_odds"] = opening.get("under_odds")
                    fields["espn_opening_ml_home"] = opening.get("ml_home")
                    fields["espn_opening_ml_away"] = opening.get("ml_away")

                # Current or Closing lines
                if "current_or_closing" in data:
                    lines = data["current_or_closing"]

                    if is_completed:
                        # Game completed → save to espn_closing_* fields
                        fields["espn_closing_spread"] = lines.get("spread")
                        fields["espn_closing_spread_home_odds"] = lines.get(
                            "spread_home_odds"
                        )
                        fields["espn_closing_spread_away_odds"] = lines.get(
                            "spread_away_odds"
                        )
                        fields["espn_closing_total"] = lines.get("total")
                        fields["espn_closing_over_odds"] = lines.get("over_odds")
                        fields["espn_closing_under_odds"] = lines.get("under_odds")
                        fields["espn_closing_ml_home"] = lines.get("ml_home")
                        fields["espn_closing_ml_away"] = lines.get("ml_away")
                    else:
                        # Game not completed → save to espn_current_* fields
                        fields["espn_current_spread"] = lines.get("spread")
                        fields["espn_current_spread_home_odds"] = lines.get(
                            "spread_home_odds"
                        )
                        fields["espn_current_spread_away_odds"] = lines.get(
                            "spread_away_odds"
                        )
                        fields["espn_current_total"] = lines.get("total")
                        fields["espn_current_over_odds"] = lines.get("over_odds")
                        fields["espn_current_under_odds"] = lines.get("under_odds")
                        fields["espn_current_ml_home"] = lines.get("ml_home")
                        fields["espn_current_ml_away"] = lines.get("ml_away")

                fields["lines_finalized"] = data.get("lines_finalized", 0)
                fields["spread_result"] = data.get("spread_result")
                fields["ou_result"] = data.get("ou_result")
                fields["updated_at"] = now

            elif is_covers:
                # Covers data - always closing
                fields = {}
                fields["covers_closing_spread"] = data.get("covers_closing_spread")
                fields["covers_closing_total"] = data.get("covers_closing_total")
                fields["spread_result"] = data.get("spread_result")
                fields["ou_result"] = data.get("ou_result")
                fields["lines_finalized"] = data.get("lines_finalized", 1)
                fields["updated_at"] = now

            elif is_placeholder:
                # Placeholder row - no data found, just update timestamp for cache
                fields = {}
                fields["updated_at"] = data.get("updated_at", now)

            else:
                logger.warning(f"Unknown data format for {game_id}")
                continue

            if existing:
                # UPDATE existing row - use COALESCE to keep existing non-NULL values
                set_clauses = []
                values = []
                for field, value in fields.items():
                    if field == "updated_at":
                        set_clauses.append(f"{field} = ?")
                        values.append(value)
                    elif field == "lines_finalized":
                        set_clauses.append(f"{field} = MAX({field}, ?)")
                        values.append(value)
                    else:
                        set_clauses.append(f"{field} = COALESCE(?, {field})")
                        values.append(value)

                values.append(game_id)

                query = f"UPDATE Betting SET {', '.join(set_clauses)} WHERE game_id = ?"
                conn.execute(query, values)

            else:
                # INSERT new row
                fields["game_id"] = game_id
                fields["created_at"] = now
                if "updated_at" not in fields:
                    fields["updated_at"] = now

                columns = list(fields.keys())
                placeholders = ["?" * len(columns)]
                values = [fields[col] for col in columns]

                query = f"INSERT INTO Betting ({', '.join(columns)}) VALUES ({', '.join(['?'] * len(columns))})"
                conn.execute(query, values)

            saved += 1

        conn.commit()
    finally:
        if close_conn:
            conn.close()

    return saved


# Legacy alias for backwards compatibility
save_betting_lines = save_betting_data


# =============================================================================
# 3-Tier Batch Operations
# =============================================================================


@log_execution_time()
def update_betting_data(
    game_ids: Optional[list[str]] = None,
    date_range: Optional[tuple[str, str]] = None,
    season: Optional[str] = None,
    skip_finalized: bool = True,
    use_covers: bool = True,
    stage_logger=None,
) -> dict:
    """
    Update betting data using 3-tier strategy.

    TIER 1 - ESPN API (games within -7 to +2 days):
        - Used for real-time and recent data
        - Provides full odds (spreads, totals, moneylines with vig)
        - Primary source for automatic web app updates

    TIER 2 - Covers Matchups (completed games >7 days old):
        - Used for finalizing older games that ESPN no longer serves
        - Fetches by date (efficient: 1 call per date)
        - Only triggered for dates with unfinalized completed games
        - Provides closing spreads/totals (no odds)

    TIER 3 - Covers Team Schedules (historical backfill):
        - Used via CLI with --backfill flag
        - Not triggered by this function (use update_betting_backfill)

    Args:
        game_ids: Specific game IDs to update, or None for auto-detection
        date_range: Tuple of (start_date, end_date) in YYYY-MM-DD format
        season: Specific season to update (e.g., "2024-2025")
                If None, uses current season (automatic mode for web app)
        skip_finalized: If True, skip games with finalized lines (default: True)
        use_covers: If True, use Tier 2 Covers for games outside ESPN window
        stage_logger: Optional StageLogger for tracking API calls

    Returns:
        Dict with stats: {"espn_fetched": int, "covers_fetched": int,
                         "saved": int, "skipped": int, "cached": int, "errors": int}
    """
    stats = {
        "espn_fetched": 0,
        "covers_fetched": 0,
        "saved": 0,
        "skipped": 0,
        "cached": 0,
        "already_finalized": 0,
        "errors": 0,
    }

    now = datetime.now(timezone.utc)

    with get_db() as conn:
        conn.row_factory = sqlite3.Row

        # Ensure table exists
        create_betting_tables(conn)

        # Determine season filter
        if season:
            target_season = season
        elif game_ids or date_range:
            target_season = None
        else:
            target_season = _get_current_season(now)

        # Get games that need betting data
        games = _get_games_needing_betting_data(
            conn, game_ids, date_range, target_season, now
        )

        if not games:
            logger.debug("No games need betting data updates")
            return stats

        logger.debug(
            f"Processing betting data for {len(games)} games "
            f"(season: {target_season or 'mixed'})"
        )

        # Get existing finalized game_ids
        existing_finalized = set()
        if skip_finalized:
            cursor = conn.execute(
                """SELECT game_id FROM Betting 
                   WHERE lines_finalized = 1 
                   AND (espn_closing_spread IS NOT NULL OR covers_closing_spread IS NOT NULL)"""
            )
            existing_finalized = {row[0] for row in cursor.fetchall()}

        # Partition games into ESPN window vs Covers needed
        espn_games = []
        covers_dates_needed = set()

        for game in games:
            game_id = game["game_id"]

            if skip_finalized and game_id in existing_finalized:
                stats["already_finalized"] += 1
                continue

            game_datetime = datetime.strptime(
                game["date_time_utc"], "%Y-%m-%dT%H:%M:%SZ"
            )
            days_ago = (now - game_datetime.replace(tzinfo=timezone.utc)).days

            # ESPN window: -FUTURE_CUTOFF_DAYS to +ESPN_LOOKBACK_DAYS
            if -FUTURE_CUTOFF_DAYS <= days_ago <= ESPN_LOOKBACK_DAYS:
                espn_games.append(game)
            elif (
                use_covers
                and game["status"] == 3  # Final
                and days_ago > ESPN_LOOKBACK_DAYS
            ):
                # Game is outside ESPN window, use Covers (pages are per Eastern date)
                covers_dates_needed.add(_eastern_date(game["date_time_utc"]))

        # TIER 1: Fetch from ESPN
        if espn_games:
            espn_stats = _fetch_espn_batch(espn_games, conn, stage_logger)
            stats["espn_fetched"] += espn_stats["fetched"]
            stats["saved"] += espn_stats["saved"]
            stats["skipped"] += espn_stats["skipped"]
            stats["cached"] += espn_stats.get("cached", 0)
            stats["errors"] += espn_stats["errors"]

        # TIER 2: Fetch from Covers for dates outside ESPN window
        covers_stats = {"fetched": 0, "saved": 0, "errors": 0}  # Initialize
        if covers_dates_needed and use_covers:
            # Filter out dates that were recently attempted but had no successful matches
            # to avoid repeated API calls for mismatched dates
            filtered_dates = _filter_failed_covers_dates(
                list(covers_dates_needed), conn
            )

            if filtered_dates:
                covers_stats = _fetch_covers_batch(
                    filtered_dates, conn, target_season, stage_logger
                )
                stats["covers_fetched"] += covers_stats["fetched"]
                stats["saved"] += covers_stats["saved"]
        stats["errors"] += covers_stats["errors"]

    logger.debug(f"Betting update complete: {stats}")
    return stats


def _get_games_needing_betting_data(
    conn: sqlite3.Connection,
    game_ids: Optional[list[str]],
    date_range: Optional[tuple[str, str]],
    season: Optional[str],
    now: datetime,
) -> list[sqlite3.Row]:
    """
    Get games that need betting data from the database.

    Returns games that are:
    1. Completed without finalized betting data
    2. Upcoming within +2 days of now
    """
    if game_ids:
        placeholders = ",".join(["?"] * len(game_ids))
        query = f"""
            SELECT g.game_id, g.date_time_utc, g.home_team, g.away_team, g.status
            FROM Games g
            WHERE g.game_id IN ({placeholders})
        """
        return conn.execute(query, game_ids).fetchall()

    if date_range:
        query = """
            SELECT g.game_id, g.date_time_utc, g.home_team, g.away_team, g.status
            FROM Games g
            WHERE date(g.date_time_utc) BETWEEN ? AND ?
              AND g.season_type IN ('Regular Season', 'Post Season')
            ORDER BY g.date_time_utc
        """
        return conn.execute(query, date_range).fetchall()

    # Season mode - get games needing betting data
    future_cutoff = (now + timedelta(days=FUTURE_CUTOFF_DAYS)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )

    query = """
        SELECT g.game_id, g.date_time_utc, g.home_team, g.away_team, g.status
        FROM Games g
        LEFT JOIN Betting b ON g.game_id = b.game_id
        WHERE g.season = ?
          AND g.season_type IN ('Regular Season', 'Post Season')
          AND (g.status = 3 OR g.date_time_utc <= ?)  -- Final or upcoming
          AND (b.lines_finalized IS NULL OR b.lines_finalized = 0 
               OR (b.espn_closing_spread IS NULL AND b.covers_closing_spread IS NULL))
        ORDER BY g.date_time_utc
    """
    return conn.execute(query, (season, future_cutoff)).fetchall()


def _should_use_cache(
    existing: dict,
    game_status: int,
    now: datetime,
    game_datetime: Optional[datetime] = None,
) -> bool:
    """
    Determine if we should use cached betting data instead of re-fetching.

    Caching rules:
    1. Finalized games (closing lines exist + lines_finalized=1): Cache forever
    2. Games with no ESPN ID after previous attempt: Cache for 24 hours (fall through to Covers)
    3. Completed games with current_spread: Re-fetch once to get closing (then finalize)
    4. Future games >24h away: Cache for 6 hours
    5. Future games <24h away: Cache for 1 hour (lines move more)
    6. In-progress games: Cache for 6 hours (lines locked at tipoff)

    Args:
        existing: Dict with keys: updated_at, lines_finalized, espn_closing_spread,
                  covers_closing_spread, espn_current_spread, espn_event_id
        game_status: Current game status (1=scheduled, 2=in-progress, 3=final)
        now: Current datetime (UTC)
        game_datetime: Game start time (optional, for smarter caching)

    Returns:
        True if cached data should be used, False if we should re-fetch
    """
    # No existing data = can't use cache (first fetch)
    if not existing or not existing.get("updated_at"):
        return False

    # Parse last update time
    try:
        updated_at = datetime.fromisoformat(
            existing["updated_at"].replace("Z", "+00:00")
        )
    except (ValueError, AttributeError):
        return False  # Invalid timestamp = re-fetch

    # Rule 1: Finalized games = cache forever
    has_closing = (
        existing.get("espn_closing_spread") is not None
        or existing.get("covers_closing_spread") is not None
    )
    if existing.get("lines_finalized") == 1 and has_closing:
        return True  # Cache forever

    hours_since_update = (now - updated_at).total_seconds() / 3600

    # Rule 2: Games without ESPN ID = cache for 24 hours (ESPN won't have it, use Covers)
    # Only applies if we've tried before (has updated_at) but no ESPN ID or lines
    if (
        existing.get("espn_event_id") is None
        and existing.get("espn_current_spread") is None
        and existing.get("espn_closing_spread") is None
    ):
        return hours_since_update < 24.0

    # Rule 3: Completed games - need one re-fetch to get closing lines
    if game_status == 3:  # Final
        if existing.get("espn_current_spread") is not None:
            # Has current lines, give 1 hour to re-fetch for closing format
            return hours_since_update < 1.0
        else:
            # No data yet, re-fetch immediately
            return False

    # Rule 4: In-progress games - lines locked at tipoff, cache for 6 hours
    if game_status == 2:  # In Progress
        return hours_since_update < 6.0

    # Rule 5 & 6: Scheduled games - cache based on time until game
    if game_datetime:
        hours_until_game = (
            game_datetime.replace(tzinfo=timezone.utc) - now
        ).total_seconds() / 3600
        if hours_until_game > 24:
            # >24h away: cache for 6 hours
            return hours_since_update < 6.0
        else:
            # <24h away: cache for 1 hour
            return hours_since_update < 1.0

    # Fallback: 1 hour cache
    return hours_since_update < 1.0


@log_execution_time(average_over="games")
def _fetch_espn_batch(games: list, conn: sqlite3.Connection, stage_logger=None) -> dict:
    """
    Fetch betting data from ESPN API for a batch of games (Tier 1).

    Caching strategy:
    - Finalized games: Cache forever
    - Completed games: Re-fetch once to get closing, then finalize
    - In-progress games: Cache for 6 hours (lines locked at tipoff)
    - Future games >24h: Cache for 6 hours
    - Future games <24h: Cache for 1 hour
    """
    stats = {"fetched": 0, "saved": 0, "skipped": 0, "errors": 0, "cached": 0}
    betting_data_batch = []
    BATCH_SIZE = 50
    now = datetime.now(timezone.utc)

    # Get existing betting data with timestamps for cache checks
    game_ids = [g["game_id"] for g in games]
    placeholders = ",".join(["?"] * len(game_ids))
    cursor = conn.execute(
        f"""
        SELECT game_id, updated_at, lines_finalized, 
               espn_closing_spread, covers_closing_spread, espn_current_spread, espn_event_id
        FROM Betting
        WHERE game_id IN ({placeholders})
        """,
        game_ids,
    )
    existing_betting = {row[0]: dict(row) for row in cursor.fetchall()}

    # Pre-check if all games are cached to avoid showing unnecessary progress bar
    all_cached = True
    for game in games:
        game_id = game["game_id"]
        game_status = game["status"]
        game_datetime_str = game["date_time_utc"]
        game_datetime = datetime.strptime(game_datetime_str, "%Y-%m-%dT%H:%M:%SZ")
        existing = existing_betting.get(game_id)
        if not (
            existing and _should_use_cache(existing, game_status, now, game_datetime)
        ):
            all_cached = False
            break

    # Only show progress bar if we're fetching (not all cached)
    pbar = (
        None
        if all_cached
        else tqdm(games, desc="Tier 1: ESPN API", unit="game", leave=False)
    )

    for game in games:
        game_id = game["game_id"]
        game_status = game["status"]
        if pbar:
            pbar.set_postfix(
                fetched=stats["fetched"], cached=stats["cached"], errors=stats["errors"]
            )
            pbar.update(1)

        try:
            # Parse game datetime for cache decision
            game_datetime_str = game["date_time_utc"]
            game_datetime = datetime.strptime(game_datetime_str, "%Y-%m-%dT%H:%M:%SZ")

            # Check cache before fetching
            existing = existing_betting.get(game_id)
            if existing and _should_use_cache(
                existing, game_status, now, game_datetime
            ):
                stats["cached"] += 1
                stats["skipped"] += 1
                continue

            game_date = game_datetime_str.split("T")[0]

            betting_data = fetch_betting_for_game(
                game_id=game_id,
                game_date=game_date,
                home_team=game["home_team"],
                away_team=game["away_team"],
                game_datetime=game_datetime,
                game_status=game_status,
            )

            # Track API call if stage_logger provided
            if stage_logger:
                stage_logger.log_api_call()

            if betting_data:
                betting_data_batch.append(betting_data)
                stats["fetched"] += 1

                if len(betting_data_batch) >= BATCH_SIZE:
                    stats["saved"] += save_betting_data(betting_data_batch, conn)
                    betting_data_batch = []
            else:
                # No data found - save placeholder row with updated_at timestamp
                # This enables cache to work: we won't re-check for 1 hour
                placeholder = {
                    "game_id": game_id,
                    "updated_at": now.isoformat().replace("+00:00", "Z"),
                }
                betting_data_batch.append(placeholder)
                stats["skipped"] += 1

                if len(betting_data_batch) >= BATCH_SIZE:
                    stats["saved"] += save_betting_data(betting_data_batch, conn)
                    betting_data_batch = []

        except Exception as e:
            logger.error(f"ESPN error for {game_id}: {e}")
            stats["errors"] += 1

    if pbar:
        pbar.close()

    if betting_data_batch:
        stats["saved"] += save_betting_data(betting_data_batch, conn)

    return stats


@log_execution_time(average_over="dates")
def _fetch_covers_batch(
    dates: list[str], conn: sqlite3.Connection, season: Optional[str], stage_logger=None
) -> dict:
    """
    Fetch betting data from Covers.com matchups pages (Tier 2).

    Args:
        dates: List of dates (YYYY-MM-DD) to fetch
        conn: Database connection
        season: Season string for context
        stage_logger: Optional StageLogger for tracking
    """
    from src.database_updater.covers import fetch_matchups_for_date
    from src.utils import NBATeamConverter

    stats = {"fetched": 0, "saved": 0, "errors": 0}

    # `dates` are Eastern dates (Covers pages are per Eastern date). Key each game
    # by its own Eastern date so same-venue games a day or two apart stay distinct.
    game_lookup = {}
    games_by_teams = {}  # (home, away) -> {game_id: eastern_date}, for the fallback

    for date_str in dates:
        # Eastern date D spans UTC dates D..D+1; widen by a day for the fallback
        cursor = conn.execute(
            """
            SELECT game_id, home_team, away_team, date_time_utc
            FROM Games
            WHERE date(date_time_utc) BETWEEN date(?, '-1 day') AND date(?, '+2 day')
              AND season_type IN ('Regular Season', 'Post Season')
        """,
            (date_str, date_str),
        )
        for row in cursor.fetchall():
            # Normalize team names using comprehensive NBATeamConverter
            try:
                home_team = NBATeamConverter.get_abbreviation(row["home_team"])
                away_team = NBATeamConverter.get_abbreviation(row["away_team"])
            except ValueError:
                # Fallback to original if normalization fails
                home_team = row["home_team"]
                away_team = row["away_team"]

            eastern_date = _eastern_date(row["date_time_utc"])

            # Primary lookup: exact Eastern date match
            game_lookup[(eastern_date, home_team, away_team)] = row["game_id"]

            # Secondary lookup: team-based for the ±1 day fallback
            games_by_teams.setdefault((home_team, away_team), {})[
                row["game_id"]
            ] = eastern_date

    pbar = tqdm(dates, desc="Tier 2: Covers matchups", unit="date", leave=False)
    for date_str in pbar:
        pbar.set_postfix(fetched=stats["fetched"], errors=stats["errors"])

        try:
            game_date = date.fromisoformat(date_str)
            covers_games = fetch_matchups_for_date(game_date)

            # Track API call if stage_logger provided
            if stage_logger:
                stage_logger.log_api_call()

            betting_data_batch = []
            for cg in covers_games:
                # Normalize team names using comprehensive NBATeamConverter
                try:
                    home_team = NBATeamConverter.get_abbreviation(cg.home_team)
                    away_team = NBATeamConverter.get_abbreviation(cg.away_team)
                except ValueError:
                    # Fallback to original if normalization fails
                    home_team = cg.home_team
                    away_team = cg.away_team

                # Try exact match first
                key = (date_str, home_team, away_team)
                game_id = game_lookup.get(key)

                # If no exact match, accept a game within ±1 day only when it is the
                # sole candidate; never choose between two games of the same matchup
                if not game_id:
                    candidates = [
                        (candidate_id, candidate_date)
                        for candidate_id, candidate_date in games_by_teams.get(
                            (home_team, away_team), {}
                        ).items()
                        if abs((date.fromisoformat(candidate_date) - game_date).days)
                        <= 1
                    ]
                    if len(candidates) == 1:
                        game_id, candidate_date = candidates[0]
                        logger.debug(
                            f"Fuzzy match for Covers game: {cg.away_team}@{cg.home_team} "
                            f"on {date_str} -> DB date {candidate_date}"
                        )

                if not game_id:
                    logger.debug(
                        f"No DB match for Covers game: {cg.away_team}@{cg.home_team} on {date_str} "
                        f"(normalized: {away_team}@{home_team})"
                    )
                    continue

                # Use results from Covers directly (already parsed from page)
                # Covers only provides closing lines (no opening, no current)
                betting_data = {
                    "game_id": game_id,
                    "covers_closing_spread": cg.spread,
                    "covers_closing_total": cg.total,
                    "spread_result": cg.spread_result,
                    "ou_result": cg.ou_result,
                    "lines_finalized": 1 if cg.spread is not None else 0,
                }
                betting_data_batch.append(betting_data)
                stats["fetched"] += 1

            if betting_data_batch:
                stats["saved"] += save_betting_data(betting_data_batch, conn)

            # Record the attempt with match count for future caching
            _record_covers_attempt(date_str, len(betting_data_batch), conn)

        except Exception as e:
            logger.error(f"Covers error for {date_str}: {e}")
            stats["errors"] += 1
            # Record failed attempt to prevent immediate retry
            _record_covers_attempt(date_str, 0, conn)

    pbar.close()
    return stats


def update_betting_backfill(season: str) -> dict:
    """
    Tier 3: Historical backfill using Covers.com team schedule pages.

    This fetches all home games for all 30 teams in a season from Covers.
    Use this for bulk historical data collection, not for regular updates.

    Makes 30 requests with rate limiting (~90 seconds total).

    Args:
        season: Season to backfill (e.g., "2024-2025")

    Returns:
        Dict with stats: {"fetched": int, "matched": int, "saved": int, "errors": int}
    """
    from src.database_updater.covers import NBA_TO_COVERS_SLUG, fetch_team_schedule

    stats = {"fetched": 0, "matched": 0, "saved": 0, "errors": 0}

    logger.info(f"Tier 3: Backfilling season {season} from Covers team schedules")

    with get_db() as conn:
        conn.row_factory = sqlite3.Row

        # Ensure table exists
        create_betting_tables(conn)

        # Build lookup: (date, home_team, away_team) -> game_id
        cursor = conn.execute(
            """
            SELECT game_id, date(date_time_utc) as game_date, 
                   home_team, away_team
            FROM Games
            WHERE season = ?
              AND season_type IN ('Regular Season', 'Post Season')
        """,
            (season,),
        )

        game_lookup = {}
        for row in cursor.fetchall():
            key = (row["game_date"], row["home_team"], row["away_team"])
            game_lookup[key] = row["game_id"]

        logger.info(f"Found {len(game_lookup)} games in DB for {season}")

        # Fetch each team's schedule
        pbar = tqdm(
            NBA_TO_COVERS_SLUG.items(),
            desc="Tier 3: Covers team schedules",
            unit="team",
            leave=False,
        )
        for tricode, slug in pbar:
            pbar.set_postfix(team=tricode, fetched=stats["fetched"])

            try:
                covers_games = fetch_team_schedule(slug, season)

                betting_data_batch = []
                for cg in covers_games:
                    # Fill in home team
                    cg.home_team = tricode

                    if not cg.away_team:
                        continue

                    # Format date for lookup
                    game_date_str = cg.game_date.strftime("%Y-%m-%d")
                    key = (game_date_str, tricode, cg.away_team)
                    game_id = game_lookup.get(key)

                    if not game_id:
                        logger.debug(
                            f"No DB match: {cg.away_team}@{tricode} on {game_date_str}"
                        )
                        continue

                    stats["matched"] += 1

                    # Use results from Covers directly (already parsed from page)
                    # Covers only provides closing lines
                    betting_data = {
                        "game_id": game_id,
                        "covers_closing_spread": cg.spread,
                        "covers_closing_total": cg.total,
                        "spread_result": cg.spread_result,
                        "ou_result": cg.ou_result,
                        "lines_finalized": 1 if cg.spread is not None else 0,
                    }
                    betting_data_batch.append(betting_data)
                    stats["fetched"] += 1

                if betting_data_batch:
                    stats["saved"] += save_betting_data(betting_data_batch, conn)

            except Exception as e:
                logger.error(f"Covers error for {tricode}: {e}")
                stats["errors"] += 1

        pbar.close()

    logger.info(f"Tier 3 backfill complete: {stats}")
    return stats


# =============================================================================
# Query Functions
# =============================================================================


def get_betting_data(game_id: str) -> Optional[dict]:
    """
    Get betting data for a game.

    Args:
        game_id: NBA game ID

    Returns:
        Betting data dict or None
    """
    with get_db() as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(
            "SELECT * FROM Betting WHERE game_id = ?",
            (game_id,),
        )
        row = cursor.fetchone()
        return dict(row) if row else None


# =============================================================================
# CLI Interface
# =============================================================================

if __name__ == "__main__":
    import argparse

    from src.logging_config import setup_logging

    parser = argparse.ArgumentParser(
        description="Fetch and store NBA betting data using 3-tier strategy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Update current season (Tier 1 ESPN + Tier 2 Covers auto)
  python -m src.database_updater.betting
  
  # Update specific season
  python -m src.database_updater.betting --season=2025-2026

  # Backfill historical season (Tier 3 Covers team schedules)
  python -m src.database_updater.betting --backfill --season=2024-2025

  # Force re-fetch even if finalized
  python -m src.database_updater.betting --season=2025-2026 --force
  
  # Update specific games
  python -m src.database_updater.betting --game_ids=0022400123,0022400124
        """,
    )
    parser.add_argument("--game_ids", type=str, help="Comma-separated game IDs")
    parser.add_argument("--date", type=str, help="Single date (YYYY-MM-DD)")
    parser.add_argument("--start_date", type=str, help="Start date for range")
    parser.add_argument("--end_date", type=str, help="End date for range")
    parser.add_argument(
        "--season",
        type=str,
        help="Season to process (e.g., 2024-2025). Uses current season if not specified.",
    )
    parser.add_argument(
        "--backfill",
        action="store_true",
        help="Use Tier 3 (Covers team schedules) for bulk historical backfill. Requires --season.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-fetch data even if already finalized",
    )
    parser.add_argument(
        "--no-covers",
        action="store_true",
        dest="no_covers",
        help="Disable Tier 2 Covers fetching (ESPN only)",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    parser.add_argument(
        "--create_tables", action="store_true", help="Create Betting table and exit"
    )

    args = parser.parse_args()
    setup_logging(args.log_level)

    if args.create_tables:
        create_betting_tables()
        print("Betting table created successfully")
        exit(0)

    # Validate backfill requires season
    if args.backfill:
        if not args.season:
            parser.error("--backfill requires --season to be specified")

        # Run Tier 3 backfill
        stats = update_betting_backfill(args.season)
        print(f"\nTier 3 Backfill complete for {args.season}:")
        print(f"  Fetched from Covers:  {stats['fetched']}")
        print(f"  Matched to DB games:  {stats['matched']}")
        print(f"  Saved:                {stats['saved']}")
        print(f"  Errors:               {stats['errors']}")
        exit(0)

    # Determine what to fetch
    game_ids = None
    date_range = None
    season = args.season

    if args.game_ids:
        game_ids = [g.strip() for g in args.game_ids.split(",")]
    elif args.date:
        date_range = (args.date, args.date)
    elif args.start_date and args.end_date:
        date_range = (args.start_date, args.end_date)

    # Run 3-tier update
    stats = update_betting_data(
        game_ids=game_ids,
        date_range=date_range,
        season=season,
        skip_finalized=not args.force,
        use_covers=not args.no_covers,
    )

    print(f"\nBetting data update complete:")
    print(f"  Tier 1 - ESPN:        {stats['espn_fetched']}")
    print(f"  Tier 2 - Covers:      {stats['covers_fetched']}")
    print(f"  Saved:                {stats['saved']}")
    print(f"  Already finalized:    {stats['already_finalized']}")
    print(f"  Skipped (no data):    {stats['skipped']}")
    print(f"  Errors:               {stats['errors']}")
