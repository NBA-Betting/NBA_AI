"""
utils.py

This module provides utility functions and classes for managing and processing NBA data, including
database interactions, HTTP request handling, and data validation. It includes functions for looking
up game information, validating game IDs and dates, and converting between different NBA team identifiers.

Core Functions:
- lookup_basic_game_info(game_ids, db_path=DB_PATH): Retrieves basic game information for given game IDs from the database.
- log_execution_time(average_over=None): A decorator to log the execution time of functions.
- requests_retry_session(retries=3, backoff_factor=0.3, status_forcelist=(500, 502, 504), session=None, timeout=(10, 30)): Creates an HTTP session with retry logic and a default timeout.
- game_id_to_season(game_id, abbreviate=False): Converts a game ID to a season string.
- validate_game_ids(game_ids): Validates game IDs.
- validate_date_format(date): Validates that a date string is in the format "YYYY-MM-DD".
- validate_season_format(season, abbreviated=False): Validates the format of a season string.
- date_to_season(date_str): Converts a date to the corresponding NBA season.
- determine_current_season(): Determines the current NBA season based on the current date.
- get_player_image(player_id): Retrieves a player's image from the NBA website or a local cache.

Classes:
- NBATeamConverter: A class for converting between various identifiers of NBA teams such as team ID, abbreviation, short name, and full name.

Usage:
- This module can be used to support data validation and transformation tasks in an NBA data pipeline.
- Functions are typically called to validate inputs, fetch data from the database, or format data for display.
"""

import logging
import os
import re
import time
from datetime import datetime
from functools import wraps
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

from src.config import config
from src.database import get_db

# Configuration values
DB_PATH = config["database"]["path"]
PROJECT_ROOT = Path(config["project"]["root"])

# Timezone constants
EASTERN_TZ_OFFSET_HOURS = -5  # EST (standard time)
EASTERN_TZ_OFFSET_DST_HOURS = -4  # EDT (daylight saving time)


# =============================================================================
# DATETIME STRATEGY
# =============================================================================
# This project uses a consistent datetime strategy:
#
# 1. STORAGE: All timestamps stored in UTC (ISO 8601 format with 'Z' suffix)
#    - Games.date_time_utc: "2024-10-22T19:30:00Z"
#    - Betting.updated_at: "2024-10-22T19:30:00Z"
#    - Cache timestamps: UTC via get_utc_now()
#
# 2. QUERIES: NBA schedule dates are in Eastern Time (NBA's operating timezone)
#    - When user asks for "games on Dec 26", convert ET day boundaries to UTC
#    - Use get_current_eastern_date() for season determination
#
# 3. DISPLAY: Convert to user's timezone for frontend display
#    - Pass user_tz from browser to backend
#    - Use utc_to_user_tz() for display formatting
#
# Key functions:
#   get_utc_now()              - Current time in UTC (for timestamps)
#   get_current_eastern_date() - Current date in ET (for NBA schedule logic)
#   get_eastern_tz()           - Get pytz Eastern timezone object
#   utc_to_user_tz()           - Convert UTC to user's timezone for display
# =============================================================================


def get_utc_now() -> datetime:
    """
    Get current datetime in UTC (timezone-aware).

    Use this for:
    - Storing timestamps in database
    - Cache expiration calculations
    - API rate limiting

    Returns:
        datetime: Current UTC time with tzinfo set
    """
    from datetime import timezone

    return datetime.now(timezone.utc)


def get_eastern_tz():
    """
    Get the US/Eastern timezone object.

    Returns:
        pytz timezone for US/Eastern (handles DST automatically)
    """
    import pytz

    return pytz.timezone("US/Eastern")


def get_current_eastern_datetime() -> datetime:
    """
    Get current datetime in Eastern Time (timezone-aware).

    Use this for:
    - NBA schedule date calculations (NBA operates in ET)
    - Season boundary determination
    - Game scheduling logic

    Returns:
        datetime: Current Eastern time with tzinfo set
    """
    import pytz

    utc_now = datetime.now(pytz.UTC)
    return utc_now.astimezone(get_eastern_tz())


def get_current_eastern_date():
    """
    Get current date in Eastern Time.

    Use this for:
    - Determining "today's games" in NBA schedule terms
    - Season boundary checks (June 30 cutoff)

    Returns:
        date: Current date in Eastern timezone
    """
    return get_current_eastern_datetime().date()


def utc_to_user_tz(utc_dt: datetime, user_tz: str = None) -> datetime:
    """
    Convert UTC datetime to user's timezone.

    Args:
        utc_dt: datetime in UTC (can be naive or aware)
        user_tz: IANA timezone string from browser (e.g., "America/New_York")
                 If None, falls back to server's local timezone

    Returns:
        datetime: Timezone-aware datetime in user's timezone
    """
    from datetime import timezone

    import pytz

    # If naive, assume UTC
    if utc_dt.tzinfo is None:
        utc_dt = utc_dt.replace(tzinfo=timezone.utc)

    if user_tz:
        try:
            tz = pytz.timezone(user_tz)
            return utc_dt.astimezone(tz)
        except pytz.UnknownTimeZoneError:
            logging.warning(
                f"Unknown timezone '{user_tz}', falling back to server local"
            )

    # Fallback to server's local timezone
    try:
        from tzlocal import get_localzone

        local_tz = get_localzone()
        return utc_dt.astimezone(local_tz)
    except ImportError:
        return utc_dt  # Return UTC if tzlocal not available


def parse_utc_datetime(utc_string: str) -> datetime:
    """
    Parse a UTC datetime string from the database.

    Args:
        utc_string: ISO 8601 UTC string, e.g., "2024-10-22T00:30:00Z"

    Returns:
        datetime: Timezone-aware datetime in UTC
    """
    from datetime import timezone

    # Handle both "Z" suffix and no suffix
    if utc_string.endswith("Z"):
        utc_string = utc_string[:-1]

    # Handle space separator (SQLite datetime output) vs T separator
    if " " in utc_string:
        dt = datetime.strptime(utc_string, "%Y-%m-%d %H:%M:%S")
    else:
        dt = datetime.strptime(utc_string, "%Y-%m-%dT%H:%M:%S")

    return dt.replace(tzinfo=timezone.utc)


def lookup_basic_game_info(game_ids, db_path=DB_PATH):
    """
    Looks up basic game information given a game_id or a list of game_ids from the Games table in the SQLite database.

    Args:
        game_ids (str or list): The ID of the game or a list of game IDs to look up.
        db_path (str): The path to the SQLite database. Defaults to the value in the config file.

    Returns:
        dict: A dictionary with game IDs as keys and each value being a dictionary representing a game.
              Each game dictionary contains the home team, away team, date/time (UTC), status, season, and season type.
    """
    if not isinstance(game_ids, list):
        game_ids = [game_ids]

    validate_game_ids(game_ids)

    sql = f"""
    SELECT game_id, home_team, away_team, date_time_utc, status, season, season_type
    FROM Games
    WHERE game_id IN ({','.join(['?'] * len(game_ids))})
    """

    with get_db(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(sql, game_ids)
        games = cursor.fetchall()

    game_ids_set = set(game_ids)
    game_info_dict = {}
    for game_id, home, away, date_time_utc, status, season, season_type in games:
        game_ids_set.remove(game_id)
        game_info_dict[game_id] = {
            "home": home,
            "away": away,
            "date_time_utc": date_time_utc,
            "status": status,
            "season": season,
            "season_type": season_type,
        }

    if game_ids_set:
        logging.warning(f"Game IDs not found in the database: {game_ids_set}")

    return game_info_dict


def determine_current_season():
    """
    Determines the current NBA season based on the current date in Eastern Time.

    Uses Eastern Time because NBA operates in ET and season boundaries
    (June 30th cutoff) are defined in ET.

    Returns:
        str: The current NBA season in 'XXXX-XXXX' format.
    """
    # Use Eastern time since NBA operates in ET
    current_date = get_current_eastern_datetime()
    current_year = current_date.year

    # Determine the season based on the league year cutoff (June 30th ET)
    # Using timezone-aware comparison
    eastern = get_eastern_tz()
    league_year_cutoff = eastern.localize(datetime(current_year, 6, 30, 23, 59, 59))

    if current_date > league_year_cutoff:
        season = f"{current_year}-{current_year + 1}"
    else:
        season = f"{current_year - 1}-{current_year}"

    return season


def get_season_start_date(season: str, db_path: str = DB_PATH) -> datetime:
    """
    Gets the actual start date of a season from the Games table.
    Falls back to Oct 22 if no games found.

    Args:
        season: Season string in 'XXXX-XXXX' format
        db_path: Path to database

    Returns:
        datetime: Date of first game in season (or Oct 22 fallback)
    """
    with get_db(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT MIN(date_time_utc)
            FROM Games
            WHERE season = ?
            AND season_type IN ('Regular Season', 'Post Season')
            """,
            (season,),
        )
        result = cursor.fetchone()[0]

        if result:
            # Parse ISO format datetime and return just the date part
            return datetime.fromisoformat(result.replace("Z", "+00:00")).replace(
                hour=0, minute=0, second=0, microsecond=0, tzinfo=None
            )
        else:
            # Fallback to Oct 22 if no games found (typical season start)
            season_start_year = int(season.split("-")[0])
            return datetime(season_start_year, 10, 22)


def get_player_image(player_id):
    """
    Returns the NBA CDN URL for a player's headshot image.

    Args:
        player_id (str): The ID of the player whose image is to be retrieved.

    Returns:
        str: URL to the player's headshot on the NBA CDN, or a local default fallback.
    """
    return f"https://cdn.nba.com/headshots/nba/latest/260x190/{player_id}.png"


def log_execution_time(average_over=None):
    """
    Decorator that tracks execution time silently (no logging).
    Use StageLogger for actual logging output.

    Args:
        average_over (str or None): Deprecated, kept for compatibility.

    Returns:
        function: The wrapped function with execution time tracking.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            duration = time.time() - start_time

            # Store duration in function for access by StageLogger if needed
            wrapper.last_duration = duration

            return result

        return wrapper

    return decorator


class StageLogger:
    """
    Unified logger for pipeline stages - outputs ONE line per stage.

    Format: [Stage] season: +added ~updated -removed (total, validation) | duration | api_calls
    Example: [Schedule] 2025-2026: +3 ~47 -0 (1230 total, WARN: 18 TBD) | 1.2s | 1 API call
    """

    def __init__(self, stage_name: str):
        self.stage_name = stage_name
        self.logger = logging.getLogger(__name__)
        self.start_time = time.time()
        self.api_calls = 0
        self.added = 0
        self.updated = 0
        self.removed = 0
        self.total = 0
        self.validation_suffix = ""
        self.extra_info = ""

    def log_api_call(self):
        """Increment API call counter."""
        self.api_calls += 1

    def set_counts(self, added=0, updated=0, removed=0, total=0):
        """Set update counts."""
        self.added = added
        self.updated = updated
        self.removed = removed
        self.total = total

    def set_validation(self, validation_result):
        """Set validation suffix from ValidationResult."""
        if hasattr(validation_result, "log_suffix"):
            suffix = validation_result.log_suffix()
            if suffix:
                self.validation_suffix = f", {suffix}"

    def set_extra_info(self, info: str):
        """Set extra information to display."""
        self.extra_info = info

    def log_cache_hit(self, season: str = None, cache_age_minutes: float = None):
        """Log cache hit (no updates needed)."""
        duration = time.time() - self.start_time
        parts = [f"[{self.stage_name}]"]

        if season:
            parts.append(f"{season}:")

        if cache_age_minutes is not None:
            parts.append(f"cached ({cache_age_minutes:.0f}m ago)")
        else:
            parts.append("cached")

        parts.append(f"| {duration:.1f}s")
        self.logger.info(" ".join(parts))

    def log_skip(self, season: str, reason: str):
        """Log skipped stage."""
        self.logger.info(f"[{self.stage_name}] {season}: skipped - {reason}")

    def log_complete(self, season: str = None):
        """Log stage completion with all metrics in ONE line."""
        duration = time.time() - self.start_time

        # Build the log line
        parts = [f"[{self.stage_name}]"]

        if season:
            parts.append(f"{season}:")

        # Update counts - always show if set (even if zero)
        if (
            self.added is not None
            or self.updated is not None
            or self.removed is not None
        ):
            # Only show non-zero counts to save space
            count_parts = []
            if self.added > 0:
                count_parts.append(f"+{self.added}")
            if self.updated > 0:
                count_parts.append(f"~{self.updated}")
            if self.removed > 0:
                count_parts.append(f"-{self.removed}")

            if count_parts:
                parts.append(" ".join(count_parts))
            else:
                parts.append("no changes")

        # Total count (compact format)
        if self.total:
            parts.append(f"({self.total}{self.validation_suffix})")

        # Extra info
        if self.extra_info:
            parts.append(self.extra_info)

        # Duration and API calls combined
        metrics = [f"{duration:.1f}s"]
        if self.api_calls > 0:
            metrics.append(f"{self.api_calls} api")
        parts.append("| " + " | ".join(metrics))

        self.logger.info(" ".join(parts))

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - auto-log if not already logged."""
        # If exception occurred, log error instead
        if exc_type is not None:
            duration = time.time() - self.start_time
            self.logger.error(
                f"[{self.stage_name}] FAILED: {exc_val} | {duration:.1f}s"
            )
        return False  # Don't suppress exceptions


class TimeoutHTTPAdapter(HTTPAdapter):
    """
    HTTPAdapter that applies a default timeout to every request sent through it.

    requests only honors a timeout passed to the individual call. Assigning
    `session.timeout` sets an attribute that requests never reads, which leaves
    the request with no timeout at all. Injecting it at the adapter gives every
    request a bound, so an endpoint that accepts the connection and then stops
    responding raises instead of blocking forever.
    """

    def __init__(self, *args, timeout=None, **kwargs):
        self.timeout = timeout
        super().__init__(*args, **kwargs)

    def send(self, request, **kwargs):
        """Apply the default timeout unless the caller passed one explicitly."""
        if kwargs.get("timeout") is None:
            kwargs["timeout"] = self.timeout
        return super().send(request, **kwargs)


def requests_retry_session(
    retries=3,
    backoff_factor=0.3,
    status_forcelist=(500, 502, 504),
    session=None,
    timeout=(10, 30),
):
    """
    Creates a session with retry logic and a default timeout.

    Args:
        retries (int): The number of retry attempts.
        backoff_factor (float): The backoff factor for retries.
        status_forcelist (tuple): A set of HTTP status codes to trigger a retry.
        session (requests.Session): An existing session to use, or None to create a new one.
        timeout (float or tuple): Default timeout for every request, either a
            number of seconds or a (connect, read) pair. Individual calls can
            still override it. NBA endpoints regularly accept a connection and
            then stall, so without this the caller blocks indefinitely.

    Returns:
        requests.Session: A session configured with retry logic and a timeout.
    """
    session = session or requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
    )
    adapter = TimeoutHTTPAdapter(max_retries=retry, timeout=timeout)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def game_id_to_season(game_id, abbreviate=False):
    """
    Converts a game ID to a season.

    The season is determined based on the third and fourth characters of the game ID.
    If these characters represent a number less than 40, the season is in the 2000s, otherwise it's in the 1900s.

    Args:
        game_id (str): The game ID to convert.
        abbreviate (bool): Whether to abbreviate the second year of the season.

    Returns:
        str: The season corresponding to the game ID.
    """
    # Validate the game ID
    validate_game_ids(game_id)

    # Extract the season from the game ID
    season = game_id[3:5]

    # Determine the prefix based on the season
    prefix = "20" if int(season) < 40 else "19"

    # Construct the years for the season
    year1 = prefix + season
    year2 = str(int(year1) + 1)

    # Return the season in the appropriate format
    if abbreviate:
        return year1 + "-" + year2[2:]
    return year1 + "-" + year2


def validate_game_ids(game_ids):
    """
    Validates a game ID or a list of game IDs.

    Each game ID must be a 10-character string that starts with '00'.

    Args:
        game_ids (str or list): The game ID(s) to validate.

    Raises:
        ValueError: If any game ID is not valid.
    """
    # Ensure game_ids is a list
    if isinstance(game_ids, str):
        game_ids = [game_ids]

    invalid_game_ids = []

    # Validate the game_ids
    for game_id in game_ids:
        if not (
            game_id
            and isinstance(game_id, str)
            and len(game_id) == 10
            and game_id.startswith("00")
        ):
            invalid_game_ids.append(game_id)
            logging.warning(
                f"Invalid game ID {game_id}. Game ID must be a 10-digit string starting with '00'. Example: '0022100001'."
            )

    if invalid_game_ids:
        raise ValueError(
            f"Invalid game IDs: {invalid_game_ids}. Each game ID must be a 10-digit string starting with '00'. Example: '0022100001'."
        )


def validate_date_format(date):
    """
    Validates if the given date is in the format "YYYY-MM-DD".

    Args:
        date (str): The date string to validate.

    Raises:
        ValueError: If the date is not in the correct format or if the month or day is not valid.
    """
    # Check the overall format
    if len(date) != 10 or date[4] != "-" or date[7] != "-":
        raise ValueError("Invalid date format. Please use YYYY-MM-DD format.")

    year, month, day = date.split("-")

    # Check if year, month and day are all digits
    if not year.isdigit() or not month.isdigit() or not day.isdigit():
        raise ValueError("Invalid date format. Please use YYYY-MM-DD format.")

    year, month, day = int(year), int(month), int(day)

    # Check if month is between 1 and 12
    if month < 1 or month > 12:
        raise ValueError(
            "Invalid month. Please use MM format with a value between 01 and 12."
        )

    # Check if day is between 1 and the maximum day of the month
    if month in [4, 6, 9, 11] and day > 30:
        raise ValueError(
            "Invalid day. Please use DD format with a value between 01 and 30 for this month."
        )
    elif month == 2 and day > 29:
        raise ValueError(
            "Invalid day. Please use DD format with a value between 01 and 29 for this month."
        )
    elif day < 1 or day > 31:
        raise ValueError(
            "Invalid day. Please use DD format with a value between 01 and 31."
        )


def validate_season_format(season, abbreviated=False):
    """
    Validates the format of a season string.

    Args:
        season (str): The season string to validate, formatted as 'XXXX-XX' or 'XXXX-XXXX'.
        abbreviated (bool): Whether the second year in the season string is abbreviated.

    Raises:
        ValueError: If the season string does not match the required format or if the second year does not logically follow the first year.
    """
    FULL_SEASON_PATTERN = r"^(\d{4})-(\d{4})$"
    ABBREVIATED_SEASON_PATTERN = r"^(\d{4})-(\d{2})$"

    # Define the regex pattern based on abbreviated flag
    pattern = ABBREVIATED_SEASON_PATTERN if abbreviated else FULL_SEASON_PATTERN

    # Attempt to match the pattern to the season string
    match = re.match(pattern, season)
    if not match:
        raise ValueError(
            "Season does not match the required format. Please use 'XXXX-XX' or 'XXXX-XXXX'."
        )

    year1, year2_suffix = map(int, match.groups())

    # Handle the year2 based on whether it's abbreviated or not
    year2 = year2_suffix if not abbreviated else year1 // 100 * 100 + year2_suffix

    # Check if year2 logically follows year1
    if year1 + 1 != year2:
        raise ValueError(
            f"Second year {year2} does not logically follow the first year {year1}."
        )

    # Check if years are within a valid range
    if year1 < 1900 or year2 > 2100:
        raise ValueError(
            f"Season years must be between 1900 and 2100. {year1}-{year2} is not a valid season."
        )


def date_to_season(date_str):
    """
    Converts a date to the NBA season.

    The typical cutoff date between seasons is June 30th.
    Special cases are handled for seasons affected by lockouts and COVID-19.

    Args:
        date_str (str): The date in YYYY-MM-DD format.

    Returns:
        str: The season in YYYY-YYYY format.
    """
    # Validate the date format
    validate_date_format(date_str)

    date = datetime.strptime(date_str, "%Y-%m-%d")

    # Special cases for lockout and COVID-19 seasons (full league year)
    special_cases = [
        ("2011-2012", datetime(2011, 7, 1), datetime(2012, 6, 30)),
        ("2019-2020", datetime(2019, 7, 1), datetime(2020, 10, 11)),
        ("2020-2021", datetime(2020, 10, 12), datetime(2021, 7, 20)),
    ]

    for season, start, end in special_cases:
        if start <= date <= end:
            return season

    # General case
    year = date.year
    if date.month > 6 or (
        date.month == 6 and date.day > 30
    ):  # After June 30th, it's the next season
        return f"{year}-{year + 1}"
    else:  # Before July 1st, it's the previous season
        return f"{year - 1}-{year}"


class NBATeamConverter:
    """
    A class to convert between various identifiers of NBA teams such as team ID,
    abbreviation, short name, and full name along with any historical identifiers.
    """

    project_root = config["project"]["root"]
    relative_db_path = config["database"]["path"]
    absolute_db_path = os.path.join(project_root, relative_db_path)

    @staticmethod
    def __get_team_id(identifier):
        """
        Get the team ID corresponding to the given identifier.
        If the identifier is unknown, raise a ValueError.

        Args:
            identifier (str): The identifier of the team.

        Returns:
            int: The team ID corresponding to the identifier.

        Raises:
            ValueError: If the identifier is unknown.
        """
        # Normalize the identifier
        identifier_normalized = str(identifier).lower().replace("-", " ")

        # Open a new database connection
        with get_db(NBATeamConverter.absolute_db_path) as conn:
            cursor = conn.cursor()

            # Execute the SQL query
            cursor.execute(
                """
                SELECT team_id FROM Teams
                WHERE abbreviation_normalized = ? OR full_name_normalized = ? OR short_name_normalized = ? OR
                json_extract(alternatives_normalized, '$') LIKE ?
                """,
                (
                    identifier_normalized,
                    identifier_normalized,
                    identifier_normalized,
                    f'%"{identifier_normalized}"%',
                ),
            )

            # Fetch the result of the query
            result = cursor.fetchone()

            # If the result is None, raise a ValueError
            if result is None:
                raise ValueError(f"Unknown team identifier: {identifier}")

            # Return the team ID
            return result[0]

    @staticmethod
    def get_abbreviation(identifier):
        """
        Get the abbreviation of the team corresponding to the given identifier.

        Args:
            identifier (str): The identifier of the team.

        Returns:
            str: The abbreviation of the team.
        """
        # Get the team ID corresponding to the identifier
        team_id = NBATeamConverter.__get_team_id(identifier)

        # Open a new database connection
        with get_db(NBATeamConverter.absolute_db_path) as conn:
            cursor = conn.cursor()

            # Execute the SQL query
            cursor.execute(
                "SELECT abbreviation FROM Teams WHERE team_id = ?", (team_id,)
            )

            # Return the abbreviation of the team
            return cursor.fetchone()[0].upper()

    @staticmethod
    def get_short_name(identifier):
        """
        Get the short name of the team corresponding to the given identifier.

        Args:
            identifier (str): The identifier of the team.

        Returns:
            str: The short name of the team.
        """
        # Get the team ID corresponding to the identifier
        team_id = NBATeamConverter.__get_team_id(identifier)

        # Open a new database connection
        with get_db(NBATeamConverter.absolute_db_path) as conn:
            cursor = conn.cursor()

            # Execute the SQL query
            cursor.execute("SELECT short_name FROM Teams WHERE team_id = ?", (team_id,))

            # Return the short name of the team
            return cursor.fetchone()[0].title()

    @staticmethod
    def get_full_name(identifier):
        """
        Get the full name of the team corresponding to the given identifier.

        Args:
            identifier (str): The identifier of the team.

        Returns:
            str: The full name of the team.
        """
        # Get the team ID corresponding to the identifier
        team_id = NBATeamConverter.__get_team_id(identifier)

        # Open a new database connection
        with get_db(NBATeamConverter.absolute_db_path) as conn:
            cursor = conn.cursor()

            # Execute the SQL query
            cursor.execute("SELECT full_name FROM Teams WHERE team_id = ?", (team_id,))

            # Return the full name of the team
            return cursor.fetchone()[0].title()
