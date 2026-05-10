"""
nba_official_injuries.py

Fetches and parses NBA's official daily injury report PDFs.
Source: https://ak-static.cms.nba.com/referee/injury/Injury-Report_{YYYY-MM-DD}_{time}.pdf

The NBA publishes injury reports multiple times daily (9AM, 12PM, 3PM, 5PM, 5:30PM, 7PM ET).
Later reports contain more complete data as teams submit updates throughout the day.
This module fetches the latest available report (7PM preferred) for most complete coverage.

This provides granular injury data including:
- body_part: Ankle, Knee, Hamstring, etc.
- injury_type: Sprain, Strain, Soreness, Surgery, etc.
- injury_side: Left, Right
- status: Out, Questionable, Doubtful, Probable, Available

Used for: Historical backfill and daily updates to complement ESPN real-time data.

Functions:
    - update_nba_official_injuries(days_back=1): Updates recent injury reports
    - fetch_injury_report(date): Fetches and parses a single day's PDF (latest time)
    - parse_injury_pdf(pdf_content): Parses PDF content to extract injuries
"""

import io
import logging
import re
import sqlite3
import time
from datetime import datetime, timedelta
from typing import List, Optional

import pandas as pd
import pdfplumber
import requests
from tqdm import tqdm

from src.config import config
from src.database import create_connection, get_db

# Suppress verbose DEBUG logging from pdfminer (used by pdfplumber)
logging.getLogger("pdfminer").setLevel(logging.WARNING)

DB_PATH = config["database"]["path"]
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

# NBA changed URL format around Dec 22, 2025 - try new format first, fallback to old
# Report times available: 9AM, 10AM, 11AM, 12PM, 1PM, 2PM, 3PM, 4PM, 4:30PM, 5PM,
# 5:30PM, 6PM, 6:30PM, 7PM ET (later reports contain more complete data).
# We fetch latest available first, falling back through earlier times so the script
# still finds a parseable PDF when run before 5PM ET.
#
# NOTE: NBA's CDN returns 403 Forbidden for non-existent files (not 404 as expected).
# This means 403 can indicate either:
#   1. File doesn't exist yet (e.g., today's report before 9 AM ET)
#   2. URL format has changed
# We distinguish these cases by checking the time of day in Eastern timezone.
PDF_TIMES_NEW = [
    "07_00PM",
    "06_30PM",
    "06_00PM",
    "05_30PM",
    "05_00PM",
    "04_30PM",
    "04_00PM",
    "03_00PM",
    "02_00PM",
    "01_00PM",
    "12_00PM",
    "11_00AM",
    "10_00AM",
    "09_00AM",
]  # Post Dec 22, 2025 format
PDF_TIMES_OLD = [
    "07PM",
    "0630PM",
    "06PM",
    "0530PM",
    "05PM",
    "0430PM",
    "04PM",
    "03PM",
    "02PM",
    "01PM",
    "12PM",
    "11AM",
    "10AM",
    "09AM",
]  # Pre Dec 22, 2025 format
PDF_URL_BASE = (
    "https://ak-static.cms.nba.com/referee/injury/Injury-Report_{date}_{time}.pdf"
)

# Cache configuration
INJURY_CACHE_TODAY_HOURS = 2  # Refetch today's injuries every 2 hours
INJURY_CACHE_NOT_SUBMITTED_HOURS = 1  # Retry "not yet submitted" after 1 hour
FIRST_REPORT_HOUR_ET = 9  # First injury report published at 9 AM Eastern


def parse_injury_reason(reason: str) -> tuple:
    """
    Parse injury reason text to extract body_part, injury_type, injury_side, and category.

    Returns:
        tuple: (body_part, injury_type, injury_side, category)
    """
    if not reason or pd.isna(reason):
        return None, None, None, None

    reason = reason.upper()

    # Filter out non-injury related absences
    non_injury_keywords = [
        "GLEAGUE",
        "G LEAGUE",
        "G-LEAGUE",
        "TWO-WAY",
        "TRADE",
        "PERSONAL",
        "REST",
        "COACH",
        "NOT WITH TEAM",
        "SUSPENSION",
        "RETURN TO COMPETITION",
        "RECONDITIONING",
    ]
    if any(keyword in reason for keyword in non_injury_keywords):
        return None, None, None, "Non-Injury"

    # Extract side
    side = None
    if "LEFT" in reason:
        side = "Left"
    elif "RIGHT" in reason:
        side = "Right"

    # Map body parts
    body_parts_map = {
        "ANKLE": "Ankle",
        "KNEE": "Knee",
        "HAMSTRING": "Hamstring",
        "FOOT": "Foot",
        "BACK": "Back",
        "HIP": "Hip",
        "SHOULDER": "Shoulder",
        "HAND": "Hand",
        "FINGER": "Finger",
        "WRIST": "Wrist",
        "ELBOW": "Elbow",
        "CALF": "Calf",
        "THIGH": "Thigh",
        "GROIN": "Groin",
        "RIB": "Ribs",
        "ACHILLES": "Achilles",
        "QUAD": "Quad",
        "TOE": "Toe",
        "HEAD": "Head",
        "NECK": "Neck",
        "CONCUSSION": "Head",
        "ABDOMINAL": "Abdomen",
        "ABDOMEN": "Abdomen",
        "ILLNESS": "Illness",
        "COVID": "Illness",
        "LEG": "Leg",
        "ARM": "Arm",
        "PATELLAR": "Knee",
        "ACL": "Knee",
        "MCL": "Knee",
        "MENISCUS": "Knee",
        "PLANTAR": "Foot",
        "LUMBAR": "Back",
        "FACE": "Face",
        "EYE": "Eye",
        "NOSE": "Face",
        "JAW": "Face",
        "THUMB": "Hand",
        "FOREARM": "Arm",
        "BICEP": "Arm",
        "TRICEP": "Arm",
        "PELVIS": "Hip",
        "GLUTE": "Hip",
        "ADDUCTOR": "Groin",
        "OBLIQUE": "Abdomen",
    }

    body_part = None
    for key, val in body_parts_map.items():
        if key in reason:
            body_part = val
            break

    # Map injury types
    injury_types_map = {
        "SPRAIN": "Sprain",
        "STRAIN": "Strain",
        "SORENESS": "Soreness",
        "SURGERY": "Surgery",
        "FRACTURE": "Fracture",
        "CONTUSION": "Contusion",
        "TENDINITIS": "Tendinitis",
        "TENDONITIS": "Tendinitis",
        "TORN": "Tear",
        "TEAR": "Tear",
        "INFLAMMATION": "Inflammation",
        "ILLNESS": "Illness",
        "DISLOCATION": "Dislocation",
        "IMPINGEMENT": "Impingement",
        "BRUISE": "Contusion",
        "BROKEN": "Fracture",
        "BONE": "Fracture",
    }

    injury_type = None
    for key, val in injury_types_map.items():
        if key in reason:
            injury_type = val
            break

    return body_part, injury_type, side, "Injury"


def parse_injury_pdf(pdf_content: bytes) -> pd.DataFrame:
    """Parse NBA injury report PDF content.

    Handles multiple PDF formats:
    - Lines with date/time/matchup prefix: "MM/DD/YYYY HH:MM(ET) ABC@XYZ TeamName Player,Name Status Reason"
    - Lines with time/matchup prefix: "HH:MM(ET) ABC@XYZ TeamName Player,Name Status Reason"
    - Lines with team name prefix: "TeamName Player,Name Status Reason"
    - Lines with matchup embedded: "ABC@XYZ TeamName Player,Name Status Reason"
    - Simple player lines: "Player,Name Status Reason"
    - Player lines without reason: "Player,Name Status" (reason on next line)
    """
    try:
        pdf = pdfplumber.open(io.BytesIO(pdf_content))
        all_text = ""
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                all_text += text + "\n"
        pdf.close()
    except Exception:
        return pd.DataFrame()

    records = []
    lines = all_text.split("\n")

    current_date = None
    current_time = None
    current_matchup = None

    # Known team name patterns (CamelCase without spaces as they appear in PDFs)
    team_names = [
        "AtlantaHawks",
        "BostonCeltics",
        "BrooklynNets",
        "CharlotteHornets",
        "ChicagoBulls",
        "ClevelandCavaliers",
        "DallasMavericks",
        "DenverNuggets",
        "DetroitPistons",
        "GoldenStateWarriors",
        "HoustonRockets",
        "IndianaPacers",
        "LosAngelesClippers",
        "LosAngelesLakers",
        "LAClippers",
        "LALakers",
        "MemphisGrizzlies",
        "MiamiHeat",
        "MilwaukeeBucks",
        "MinnesotaTimberwolves",
        "NewOrleansPelicans",
        "NewYorkKnicks",
        "OklahomaCityThunder",
        "OrlandoMagic",
        "Philadelphia76ers",
        "PhoenixSuns",
        "PortlandTrailBlazers",
        "SacramentoKings",
        "SanAntonioSpurs",
        "TorontoRaptors",
        "UtahJazz",
        "WashingtonWizards",
    ]
    team_pattern = "|".join(team_names)

    for line in lines:
        line = line.strip()
        if (
            not line
            or line.startswith("Page")
            or line.startswith("Injury Report:")
            or line.startswith("GameDate")
        ):
            continue

        rest = line

        # Match: "MM/DD/YYYY HH:MM(ET) ABC@XYZ ..."
        date_match = re.match(
            r"(\d{2}/\d{2}/\d{4})\s+(\d{2}:\d{2})\(ET\)\s+([A-Z]{3}@[A-Z]{3})\s*(.*)",
            rest,
        )
        if date_match:
            current_date = date_match.group(1)
            current_time = date_match.group(2)
            current_matchup = date_match.group(3)
            rest = date_match.group(4)
        else:
            # Match: "HH:MM(ET) ABC@XYZ ..."
            time_match = re.match(
                r"(\d{2}:\d{2})\(ET\)\s+([A-Z]{3}@[A-Z]{3})\s*(.*)", rest
            )
            if time_match:
                current_time = time_match.group(1)
                current_matchup = time_match.group(2)
                rest = time_match.group(3)
            else:
                # Match: "ABC@XYZ ..." (matchup at start of line)
                matchup_match = re.match(r"([A-Z]{3}@[A-Z]{3})\s+(.*)", rest)
                if matchup_match:
                    current_matchup = matchup_match.group(1)
                    rest = matchup_match.group(2)

        # Strip team name prefix if present (e.g., "DallasMavericks Cisse,Moussa...")
        team_strip = re.match(rf"^({team_pattern})\s+(.*)", rest)
        if team_strip:
            rest = team_strip.group(2)

        # Match player line with reason: "Player,Name Status Reason"
        player_match = re.match(
            r"^([A-Za-z\'\-]+,\s*[A-Za-z\'\-]+(?:\s*(?:Jr\.?|Sr\.?|III|IV|II|V))?)\s+"
            r"(Out|Available|Questionable|Doubtful|Probable)\s+(.+)$",
            rest,
        )
        if not player_match:
            # Match player line without reason: "Player,Name Status"
            player_match = re.match(
                r"^([A-Za-z\'\-]+,\s*[A-Za-z\'\-]+(?:\s*(?:Jr\.?|Sr\.?|III|IV|II|V))?)\s+"
                r"(Out|Available|Questionable|Doubtful|Probable)$",
                rest,
            )
            if player_match:
                # Create a fake match with empty reason
                player_name = player_match.group(1).strip()
                status = player_match.group(2)
                reason = ""
            else:
                continue
        else:
            player_name = player_match.group(1).strip()
            status = player_match.group(2)
            reason = player_match.group(3).strip()

        if current_date:
            body_part, injury_type, side, category = parse_injury_reason(reason)

            # Include ALL absences (injuries + rest/personal/etc.)
            records.append(
                {
                    "game_date": current_date,
                    "game_time": current_time,
                    "matchup": current_matchup,
                    "player_name": player_name,
                    "status": status,
                    "reason": reason,
                    "body_part": body_part,
                    "injury_type": injury_type,
                    "injury_side": side,
                    "category": category
                    or "Injury",  # Default to Injury if not classified
                }
            )

    return pd.DataFrame(records)


def fetch_injury_report(date: datetime) -> tuple[pd.DataFrame, str]:
    """Fetch and parse injury report for a specific date.

    Tries to fetch the latest available report (7PM > 5:30PM > 5PM) for most complete data.
    Handles both new URL format (post Dec 2025) and old format.

    Returns:
        tuple: (DataFrame with injuries, status string)
            status can be: "success", "not_found", "not_yet_submitted", "parse_empty", "forbidden", "error"
    """
    date_str = date.strftime("%Y-%m-%d")

    # Determine which URL time formats to try based on date
    # New format uses underscores (07_00PM), old format doesn't (07PM)
    # Try new format first for recent dates, old format for historical
    format_cutover = datetime(2025, 12, 22)
    if date.replace(tzinfo=None) >= format_cutover:
        time_formats = PDF_TIMES_NEW + PDF_TIMES_OLD  # Try new first, fallback to old
    else:
        time_formats = PDF_TIMES_OLD + PDF_TIMES_NEW  # Try old first, fallback to new

    last_status_code = None
    for time_fmt in time_formats:
        url = PDF_URL_BASE.format(date=date_str, time=time_fmt)
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            last_status_code = resp.status_code
            if resp.status_code == 200:
                # Always parse first — a PDF can contain NOTYETSUBMITTED markers for
                # individual teams while still carrying complete data for others.
                df = parse_injury_pdf(resp.content)
                if len(df) > 0:
                    df["report_date"] = date_str
                    logging.debug(
                        f"Injury PDF for {date_str}: fetched {time_fmt} with {len(df)} records"
                    )
                    return df, "success"
                else:
                    # 0 rows: try earlier time (teams may not have submitted yet)
                    logging.debug(
                        f"Injury PDF for {date_str} at {time_fmt}: parsed 0 records, trying earlier time"
                    )
                    continue
            elif resp.status_code == 403:
                # 403 means this time slot doesn't exist - try next
                continue
            elif resp.status_code == 404:
                # 404 means no report at this time - try next
                continue
        except requests.exceptions.RequestException as e:
            logging.debug(f"Request error fetching {date_str} at {time_fmt}: {e}")
            continue
        except Exception as e:
            logging.debug(f"Error fetching {date_str} at {time_fmt}: {e}")
            continue

    # If we exhausted all times and last status was 403/404, check if it's truly not found
    if last_status_code in (403, 404):
        return pd.DataFrame(), "not_found"

    # If we got NOTYETSUBMITTED from all times, report that
    return pd.DataFrame(), "not_yet_submitted"


def normalize_player_name(name: str) -> str:
    """Normalize player name for matching to Players table."""
    if not name:
        return ""
    # Split attached suffixes (WalkerIV -> Walker IV)
    name = re.sub(r"([a-z])(II|III|IV|Jr|Sr)([,\s]|$)", r"\1 \2\3", name)
    # Remove suffixes entirely for matching
    name = re.sub(
        r"\s+(Jr\.?|Sr\.?|III|II|IV)(\s|$|,)", r"\2", name, flags=re.IGNORECASE
    )
    # Remove periods, apostrophes and extra spaces
    name = name.replace(".", "").replace("'", "").strip()
    # Handle special chars (ć -> c, etc)
    replacements = {
        "ć": "c",
        "č": "c",
        "ž": "z",
        "š": "s",
        "đ": "d",
        "ö": "o",
        "ü": "u",
        "ä": "a",
    }
    for old, new in replacements.items():
        name = name.replace(old, new)
    return name.lower()


def _ensure_injury_cache_table(db_path: str = DB_PATH):
    """Create InjuryCache table if it doesn't exist."""
    with get_db(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS InjuryCache (
                report_date TEXT PRIMARY KEY,
                last_fetched_at TEXT NOT NULL,
                status TEXT DEFAULT 'success'
            )
            """)
        # Add status column if it doesn't exist (migration for existing tables)
        cursor.execute("PRAGMA table_info(InjuryCache)")
        columns = [row[1] for row in cursor.fetchall()]
        if "status" not in columns:
            cursor.execute(
                "ALTER TABLE InjuryCache ADD COLUMN status TEXT DEFAULT 'success'"
            )
        conn.commit()


def _get_injury_fetch_time(
    report_date: str, db_path: str = DB_PATH
) -> Optional[datetime]:
    """Get the last fetch time for a specific injury report date."""
    _ensure_injury_cache_table(db_path)
    try:
        with get_db(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT last_fetched_at FROM InjuryCache WHERE report_date = ?",
                (report_date,),
            )
            result = cursor.fetchone()
            if result:
                return datetime.fromisoformat(result[0])
            return None
    except sqlite3.OperationalError:
        return None


def _update_injury_cache(
    report_date: str, db_path: str = DB_PATH, status: str = "success"
):
    """Update the injury cache with current UTC fetch timestamp and status."""
    _ensure_injury_cache_table(db_path)
    with get_db(db_path) as conn:
        cursor = conn.cursor()
        from datetime import timezone

        fetch_time = datetime.now(timezone.utc).isoformat()
        cursor.execute(
            """
            INSERT INTO InjuryCache (report_date, last_fetched_at, status)
            VALUES (?, ?, ?)
            ON CONFLICT(report_date) DO UPDATE SET
                last_fetched_at = excluded.last_fetched_at,
                status = excluded.status
            """,
            (report_date, fetch_time, status),
        )
        conn.commit()


def _get_injury_cache_status(report_date: str, db_path: str = DB_PATH) -> Optional[str]:
    """Get the cache status for a specific injury report date."""
    _ensure_injury_cache_table(db_path)
    try:
        with get_db(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT status FROM InjuryCache WHERE report_date = ?",
                (report_date,),
            )
            result = cursor.fetchone()
            if result:
                return result[0]
            return None
    except sqlite3.OperationalError:
        return None


def _should_fetch_injury_date(report_date: datetime, db_path: str = DB_PATH) -> bool:
    """
    Determine if an injury report date should be fetched.

    Cache strategy:
    - Today's date: Refetch if last fetch was >2 hours ago AND status indicates data may exist
    - Past dates: Once fetched, never refetch (permanent cache)
    - Dates with status='not_yet_submitted' for today: Don't refetch within 2 hours
    """
    from datetime import timezone

    from src.utils import get_current_eastern_datetime, get_utc_now

    date_str = report_date.strftime("%Y-%m-%d")
    # Use Eastern time for "today" since NBA injury reports use ET
    today_eastern = get_current_eastern_datetime().replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    is_today = report_date.date() == today_eastern.date()

    last_fetch = _get_injury_fetch_time(date_str, db_path)

    if last_fetch is None:
        # Never fetched - fetch it
        return True

    # Get cached status
    cached_status = _get_injury_cache_status(date_str, db_path)

    if is_today:
        # Today: check if cache expired (2 hours)
        # Use UTC for cache comparison
        now_utc = get_utc_now()
        if last_fetch.tzinfo is None:
            last_fetch = last_fetch.replace(tzinfo=timezone.utc)
        hours_since_fetch = (now_utc - last_fetch).total_seconds() / 3600

        if hours_since_fetch <= INJURY_CACHE_TODAY_HOURS:
            # Cache is still fresh - skip
            logging.debug(
                f"Today's injury cache fresh ({hours_since_fetch:.1f}h old) - skipping"
            )
            return False

        # Cache expired - check status to determine retry behavior
        # For 'not_yet_submitted', use shorter retry interval since teams submit throughout the day
        if cached_status == "not_yet_submitted":
            if hours_since_fetch <= INJURY_CACHE_NOT_SUBMITTED_HOURS:
                logging.debug(
                    f"Today's injury cache 'not_yet_submitted' and only {hours_since_fetch:.1f}h old - skipping"
                )
                return False
            logging.debug(
                f"Today's injury cache 'not_yet_submitted' but {hours_since_fetch:.1f}h old - retrying"
            )
            return True

        logging.debug(
            f"Today's injury cache expired ({hours_since_fetch:.1f}h old) - refetching"
        )
        return True
    else:
        # Past date: permanent cache UNLESS it was 'not_found' within the last 7 days
        # (the NBA publishes injury PDFs at 7pm ET, so a 10am fetch may miss them)
        from src.utils import get_utc_now

        if cached_status == "not_found":
            now_utc = get_utc_now()
            if last_fetch.tzinfo is None:
                last_fetch = last_fetch.replace(tzinfo=timezone.utc)
            days_ago = (now_utc - last_fetch).total_seconds() / 86400
            if days_ago < 7:
                logging.debug(
                    f"Past date {date_str} was 'not_found' {days_ago:.1f} days ago - retrying"
                )
                return True
        return False


def _find_dates_missing_data(dates: list, db_path: str = DB_PATH) -> list:
    """
    Find dates that are in the cache but have no actual injury data in the database,
    AND where the cached status indicates data should exist.

    This catches cases where:
    - A previous fetch got a 403 error and was incorrectly cached
    - The PDF was empty/unparseable
    - Some other silent failure occurred

    We DON'T retry dates where:
    - status='not_found' (404 - legitimate off-day, no games)
    - status='not_yet_submitted' (teams haven't submitted yet)

    Args:
        dates: List of datetime objects to check
        db_path: Path to database

    Returns:
        List of datetime objects that should be retried
    """
    if not dates:
        return []

    # Ensure table has status column (migration)
    _ensure_injury_cache_table(db_path)

    with get_db(db_path) as conn:
        cursor = conn.cursor()

        # Get dates that are in cache with status='success' (expected to have data)
        # We only retry dates that were marked as "success" but actually have no data
        date_strs = [dt.strftime("%Y-%m-%d") for dt in dates]
        placeholders = ",".join("?" * len(date_strs))

        cursor.execute(
            f"""
            SELECT report_date FROM InjuryCache
            WHERE report_date IN ({placeholders})
            AND (status = 'success' OR status IS NULL)
            """,
            date_strs,
        )
        cached_success_dates = {row[0] for row in cursor.fetchall()}

        if not cached_success_dates:
            return []

        # Get dates that have actual injury data
        cursor.execute(
            f"""
            SELECT DISTINCT DATE(report_timestamp) as report_date
            FROM InjuryReports
            WHERE source = 'NBA_Official'
            AND DATE(report_timestamp) IN ({placeholders})
            """,
            date_strs,
        )
        dates_with_data = {row[0] for row in cursor.fetchall()}

        # Find dates that are cached as success but have no data
        missing_data_dates = cached_success_dates - dates_with_data

        if missing_data_dates:
            logging.debug(
                f"Found {len(missing_data_dates)} cached dates without data: {sorted(missing_data_dates)[:5]}..."
            )

        # Convert back to datetime objects
        from src.utils import get_eastern_tz

        eastern = get_eastern_tz()

        result = []
        for date_str in missing_data_dates:
            try:
                dt = datetime.strptime(date_str, "%Y-%m-%d")
                dt = eastern.localize(dt)
                result.append(dt)
            except ValueError:
                continue

        return result


def build_player_lookup(db_path: str = DB_PATH) -> dict:
    """Build a lookup dict mapping normalized names to NBA player IDs."""
    with get_db(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT person_id, first_name, last_name, full_name FROM Players"
        )
        players = cursor.fetchall()

    player_lookup = {}
    for person_id, first_name, last_name, full_name in players:
        if last_name and first_name:
            key1 = normalize_player_name(f"{last_name}, {first_name}")
            player_lookup[key1] = person_id
        if full_name:
            key2 = normalize_player_name(full_name)
            player_lookup[key2] = person_id

    return player_lookup


def _ensure_injury_unique_constraint(db_path: str = DB_PATH):
    """Add unique constraint to prevent duplicate injury records.

    Uses (player_name, report_timestamp, source, team) as the semantic key
    since nba_player_id can be NULL for unmatched players.
    """
    with get_db(db_path) as conn:
        cursor = conn.cursor()

        # Check if new constraint already exists
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_injury_semantic_unique'"
        )
        if cursor.fetchone():
            return  # Already exists

        # Remove old constraint if it exists (the one that included nba_player_id)
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_injury_unique'"
        )
        if cursor.fetchone():
            cursor.execute("DROP INDEX idx_injury_unique")
            logging.debug("Dropped old idx_injury_unique index")

        # Remove existing semantic duplicates before adding constraint
        # Keep the record with an nba_player_id if possible, otherwise keep the first
        logging.debug(
            "Removing duplicate injury records before adding unique constraint..."
        )
        cursor.execute("""
            DELETE FROM InjuryReports
            WHERE id NOT IN (
                SELECT MIN(CASE WHEN nba_player_id IS NOT NULL THEN id ELSE id + 1000000000 END)
                FROM InjuryReports
                GROUP BY player_name, report_timestamp, source, COALESCE(team, '')
            )
        """)
        removed = cursor.rowcount
        if removed > 0:
            logging.info(f"Removed {removed} duplicate injury records")

        # Add semantic unique constraint (without nba_player_id which can be NULL)
        cursor.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_injury_semantic_unique 
            ON InjuryReports(player_name, report_timestamp, source, COALESCE(team, ''))
        """)
        conn.commit()


def save_injury_records(df: pd.DataFrame, db_path: str = DB_PATH) -> dict:
    """Save injury records to database with player ID matching and UPSERT logic.

    Returns:
        dict: {"added": int, "updated": int, "total": int}
    """
    if df.empty:
        return {"added": 0, "updated": 0, "total": 0}

    # Ensure unique constraint exists
    _ensure_injury_unique_constraint(db_path)

    # Build player lookup for matching
    player_lookup = build_player_lookup(db_path)

    conn = create_connection(db_path)

    db_records = []
    for _, row in df.iterrows():
        matchup = row.get("matchup", "")
        away_team = matchup.split("@")[0] if "@" in matchup else None
        home_team = matchup.split("@")[1] if "@" in matchup else None

        # Format player name
        player_name = row["player_name"]
        if "," in player_name and ", " not in player_name:
            player_name = player_name.replace(",", ", ")

        # Match to NBA player ID
        normalized_name = normalize_player_name(player_name)
        nba_player_id = player_lookup.get(normalized_name)

        # Determine injury location category
        leg_parts = [
            "Ankle",
            "Knee",
            "Hamstring",
            "Calf",
            "Thigh",
            "Foot",
            "Toe",
            "Achilles",
            "Quad",
            "Groin",
            "Leg",
        ]
        injury_location = "Leg" if row["body_part"] in leg_parts else "Other"

        # Derive season from report_date (Oct-Dec = current year, Jan-Sep = previous year)
        report_date = row["report_date"]
        if report_date:
            year = int(report_date[:4])
            month = int(report_date[5:7])
            if month >= 10:  # Oct-Dec = start of new season
                season = f"{year}-{year + 1}"
            else:  # Jan-Sep = end of previous season
                season = f"{year - 1}-{year}"
        else:
            season = None

        db_records.append(
            {
                "nba_player_id": nba_player_id,
                "player_name": player_name,
                "team": away_team or home_team,
                "status": row["status"],
                "injury_type": row["injury_type"],
                "body_part": row["body_part"],
                "injury_location": injury_location,
                "injury_side": row["injury_side"],
                "category": row.get("category", "Injury"),
                "report_timestamp": row["report_date"],
                "source": "NBA_Official",
                "season": season,
            }
        )

    # Use semantic key (player_name, report_timestamp, source, team) for deduplication
    # This handles NULL nba_player_id correctly
    cursor = conn.cursor()

    # Check which records already exist using semantic key
    existing_keys = set()
    if db_records:
        # Build semantic keys to check
        semantic_keys = [
            (r["player_name"], r["report_timestamp"], r["source"], r["team"] or "")
            for r in db_records
        ]

        # Query in batches if needed
        for i in range(0, len(semantic_keys), 500):
            batch = semantic_keys[i : i + 500]
            placeholders = ",".join(["(?,?,?,?)"] * len(batch))
            flat_params = [item for key in batch for item in key]

            cursor.execute(
                f"""
                SELECT player_name, report_timestamp, source, COALESCE(team, '')
                FROM InjuryReports
                WHERE (player_name, report_timestamp, source, COALESCE(team, '')) IN (VALUES {placeholders})
            """,
                flat_params,
            )
            existing_keys.update(cursor.fetchall())

    # Count added vs updated
    added_count = 0
    updated_count = 0

    for record in db_records:
        key = (
            record["player_name"],
            record["report_timestamp"],
            record["source"],
            record["team"] or "",
        )
        if key in existing_keys:
            updated_count += 1
        else:
            added_count += 1

    # Use INSERT OR REPLACE with semantic unique index
    records_df = pd.DataFrame(db_records)

    # Pandas to_sql doesn't support OR REPLACE, so use executemany
    columns = list(records_df.columns)
    placeholders_str = ",".join(["?"] * len(columns))

    cursor.executemany(
        f"""
        INSERT OR REPLACE INTO InjuryReports ({','.join(columns)})
        VALUES ({placeholders_str})
        """,
        records_df.values.tolist(),
    )

    conn.commit()
    conn.close()

    return {"added": added_count, "updated": updated_count, "total": len(records_df)}


def update_nba_official_injuries(
    days_back: int = 1, season: str = None, db_path: str = DB_PATH, stage_logger=None
) -> dict:
    """
    Update NBA Official injury reports for recent days or entire season.

    This is meant to be called as part of the daily pipeline to fetch
    the latest injury report PDFs.

    Args:
        days_back: Number of days to look back (default 1 = yesterday + today)
        season: Season string (e.g., "2024-2025") for season-wide gap filling
        db_path: Path to database
        stage_logger: Optional StageLogger for tracking

    Returns:
        dict: {"added": int, "updated": int, "total": int}
    """
    from src.utils import (
        determine_current_season,
        get_current_eastern_datetime,
        get_eastern_tz,
        get_season_start_date,
    )

    # Use Eastern time for "today" since NBA operates in ET
    eastern = get_eastern_tz()
    today = get_current_eastern_datetime().replace(
        hour=0, minute=0, second=0, microsecond=0
    )

    # If season provided, fetch all missing dates in season
    if season:
        current_season = determine_current_season()

        # Determine season date range
        if season == current_season:
            # Current season: from actual season start to today
            season_start = get_season_start_date(season, db_path)
            # Make timezone-aware if naive
            if season_start.tzinfo is None:
                season_start = eastern.localize(season_start)
            season_end = today
        else:
            # Historical season: from actual season start to May 31 next year
            season_start = get_season_start_date(season, db_path)
            if season_start.tzinfo is None:
                season_start = eastern.localize(season_start)
            season_end_year = int(season.split("-")[1])
            season_end = eastern.localize(datetime(season_end_year, 5, 31))

        # Generate all dates in season
        all_dates = []
        current = season_start
        while current <= season_end:
            all_dates.append(current)
            current += timedelta(days=1)
    else:
        # Generate dates to check (recent days mode)
        all_dates = [today - timedelta(days=i) for i in range(days_back + 1)]
        all_dates.reverse()  # Process oldest first

    conn = create_connection(db_path)

    # Filter dates using smart caching:
    # - Today: refetch if >2 hours old
    # - Past dates: permanent cache (only fetch if never fetched)
    dates_to_fetch = [dt for dt in all_dates if _should_fetch_injury_date(dt, db_path)]

    # Also check for dates that were cached but have NO data in the database
    # This catches cases where fetching failed silently (e.g., 403 errors)
    dates_missing_data = _find_dates_missing_data(all_dates, db_path)

    # Combine and deduplicate
    dates = list(set(dates_to_fetch + dates_missing_data))
    dates.sort()  # Process in chronological order

    if len(dates) == 0:
        logging.debug(
            f"All {len(all_dates)} days already cached with data, nothing to fetch"
        )
        conn.close()

        # Get total count for reporting
        with get_db(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT COUNT(*) FROM InjuryReports WHERE source='NBA_Official'"
            )
            total = cursor.fetchone()[0]

        return {"added": 0, "updated": 0, "total": total}

    if dates_missing_data:
        logging.info(
            f"Found {len(dates_missing_data)} cached dates with no injury data - will retry"
        )

    logging.debug(
        f"Checking NBA Official injury reports for {len(dates)} days ({len(all_dates) - len(dates)} cached with data)..."
    )

    total_added = 0
    total_updated = 0
    api_calls = 0
    forbidden_count = 0

    # Use tqdm for progress bar only if fetching more than 7 days
    iterator = (
        tqdm(dates, desc="Fetching injury reports", unit="day")
        if len(dates) > 7
        else dates
    )

    for dt in iterator:
        date_str = dt.strftime("%Y-%m-%d")

        # Fetch the report (now returns tuple with status)
        df, status = fetch_injury_report(dt)
        api_calls += 1

        if stage_logger:
            stage_logger.log_api_call()

        if status == "success" and not df.empty:
            counts = save_injury_records(df, db_path)
            total_added += counts["added"]
            total_updated += counts["updated"]

            logging.debug(
                f"NBA Official injuries for {date_str}: +{counts['added']} ~{counts['updated']}"
            )
            if isinstance(iterator, tqdm):
                iterator.set_postfix({"status": "saved", "records": counts["total"]})

            # Cache successful fetches with data
            _update_injury_cache(date_str, db_path, status="success")

        elif status == "not_found":
            logging.debug(
                f"NBA Official injuries for {date_str}: no report available (off-day)"
            )
            if isinstance(iterator, tqdm):
                iterator.set_postfix({"status": "not found"})
            # Cache 404s with status - these are legitimate "no game day" dates
            _update_injury_cache(date_str, db_path, status="not_found")

        elif status == "not_yet_submitted":
            logging.debug(
                f"NBA Official injuries for {date_str}: teams have not yet submitted data"
            )
            if isinstance(iterator, tqdm):
                iterator.set_postfix({"status": "not submitted"})
            # Cache with status - teams didn't submit for this date (normal for early reports)
            _update_injury_cache(date_str, db_path, status="not_yet_submitted")

        elif status == "forbidden":
            forbidden_count += 1
            logging.debug(f"NBA Official injuries for {date_str}: 403 Forbidden")
            if isinstance(iterator, tqdm):
                iterator.set_postfix({"status": "forbidden"})
            # DON'T cache 403s - we should retry these later

        else:
            logging.debug(
                f"NBA Official injuries for {date_str}: fetch failed ({status})"
            )
            if isinstance(iterator, tqdm):
                iterator.set_postfix({"status": status})
            # DON'T cache errors - we should retry these later

        time.sleep(0.1)  # Be nice to NBA servers

    conn.close()

    # Log warning if we got 403 errors - but suppress if it's just today before 9 AM ET
    if forbidden_count > 0:
        # Check if all 403s are from today and it's before first report time
        # NBA CDN returns 403 for non-existent files (not 404), so this is expected
        is_before_first_report = (
            today.hour < FIRST_REPORT_HOUR_ET
        )  # today is already in Eastern time
        today_str = today.strftime("%Y-%m-%d")
        all_403s_are_today = (
            forbidden_count == 1
            and len(dates) == 1
            and dates[0].strftime("%Y-%m-%d") == today_str
        )

        if all_403s_are_today and is_before_first_report:
            logging.debug(
                f"Injury reports: Today's report not yet published "
                f"(before {FIRST_REPORT_HOUR_ET} AM ET, NBA CDN returns 403 for non-existent files)"
            )
        else:
            logging.warning(
                f"Injury reports: {forbidden_count} dates returned 403 Forbidden. "
                f"NBA CDN uses 403 for non-existent files, but this many 403s may indicate "
                f"a URL format change. Check PDF_TIMES_NEW and PDF_TIMES_OLD in nba_official_injuries.py"
            )

    # Log warning if we attempted fetches but collected nothing (skip if before first report)
    if len(dates) > 0 and total_added == 0 and total_updated == 0:
        is_before_first_report = today.hour < FIRST_REPORT_HOUR_ET
        today_str = today.strftime("%Y-%m-%d")
        all_dates_are_today = (
            len(dates) == 1 and dates[0].strftime("%Y-%m-%d") == today_str
        )

        if all_dates_are_today and is_before_first_report:
            # Don't warn - this is expected before first report time
            logging.debug(
                f"Injury reports: No data for today yet (before {FIRST_REPORT_HOUR_ET} AM ET)"
            )
        else:
            logging.warning(
                f"Injury reports: Attempted {len(dates)} dates but collected 0 records. "
                f"This may indicate an API or parsing issue."
            )

    # Get total count
    with get_db(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM InjuryReports WHERE source='NBA_Official'")
        total = cursor.fetchone()[0]

    logging.debug(
        f"NBA Official injury update complete: +{total_added} ~{total_updated} (total: {total})"
    )

    return {"added": total_added, "updated": total_updated, "total": total}


def backfill_injury_reports(
    start_date: str, end_date: str, db_path: str = DB_PATH, batch_size: int = 50
) -> int:
    """
    Backfill NBA Official injury reports for a date range.

    This is for historical data collection. Uses batch saving and progress bars
    for efficient processing of large date ranges.

    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        db_path: Path to database
        batch_size: Number of days to process before saving (default: 50)

    Returns:
        Number of records inserted
    """
    from datetime import datetime, timedelta

    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    if start_dt > end_dt:
        raise ValueError("start_date must be before end_date")

    # Generate all dates in range
    current_dt = start_dt
    dates = []
    while current_dt <= end_dt:
        dates.append(current_dt)
        current_dt += timedelta(days=1)

    logging.info(
        f"Backfilling NBA Official injury reports for {len(dates)} days ({start_date} to {end_date})..."
    )

    conn = create_connection(db_path)
    total_inserted = 0
    total_cached = 0
    total_not_found = 0

    # Batch collection for efficient saving
    batch_dfs = []

    # Progress bar for large backfills
    with tqdm(dates, desc="Backfilling injury reports", unit="day") as pbar:
        for dt in pbar:
            date_str = dt.strftime("%Y-%m-%d")

            # Check if we already have data for this date
            existing = pd.read_sql(
                "SELECT COUNT(*) as cnt FROM InjuryReports WHERE source = 'NBA_Official' AND report_timestamp = ?",
                conn,
                params=(date_str,),
            )["cnt"].iloc[0]

            if existing > 0:
                logging.debug(f"{date_str}: already have {existing} records")
                total_cached += 1
                pbar.set_postfix(
                    {
                        "cached": total_cached,
                        "inserted": total_inserted,
                        "not_found": total_not_found,
                    }
                )
                continue

            # Fetch the report
            df = fetch_injury_report(dt)
            if not df.empty:
                batch_dfs.append(df)
                pbar.set_postfix(
                    {
                        "cached": total_cached,
                        "batch": len(batch_dfs),
                        "pending": total_inserted,
                    }
                )

                # Save batch if it reaches batch_size
                if len(batch_dfs) >= batch_size:
                    combined_df = pd.concat(batch_dfs, ignore_index=True)
                    count = save_injury_records(combined_df, db_path)
                    logging.info(
                        f"Saved batch: {count} records from {len(batch_dfs)} days"
                    )
                    total_inserted += count
                    batch_dfs = []  # Clear batch
            else:
                logging.debug(f"{date_str}: no report available")
                total_not_found += 1
                pbar.set_postfix(
                    {
                        "cached": total_cached,
                        "inserted": total_inserted,
                        "not_found": total_not_found,
                    }
                )

            # Rate limiting - be respectful to NBA servers
            time.sleep(0.2)

    # Save any remaining records in the last batch
    if batch_dfs:
        combined_df = pd.concat(batch_dfs, ignore_index=True)
        count = save_injury_records(combined_df, db_path)
        logging.info(f"Saved final batch: {count} records from {len(batch_dfs)} days")
        total_inserted += count

    conn.close()

    logging.info(
        f"Backfill complete: {total_inserted} new records, {total_cached} cached, {total_not_found} not found ({len(dates)} days total)"
    )
    return total_inserted


def main():
    """CLI entry point for injury data collection."""
    import argparse

    from src.logging_config import setup_logging

    parser = argparse.ArgumentParser(
        description="Collect NBA Official injury reports",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Update recent reports (default: yesterday + today)
  python -m src.database_updater.nba_official_injuries
  
  # Update last 7 days
  python -m src.database_updater.nba_official_injuries --days-back 7
  
  # Backfill historical range
  python -m src.database_updater.nba_official_injuries --backfill --start 2024-10-01 --end 2024-12-01
        """,
    )

    parser.add_argument(
        "--days-back",
        type=int,
        default=1,
        help="Number of days to look back (default: 1)",
    )
    parser.add_argument(
        "--backfill",
        action="store_true",
        help="Backfill historical data (requires --start and --end)",
    )
    parser.add_argument(
        "--start", type=str, help="Start date for backfill (YYYY-MM-DD)"
    )
    parser.add_argument("--end", type=str, help="End date for backfill (YYYY-MM-DD)")
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )

    args = parser.parse_args()
    setup_logging(log_level=args.log_level.upper())

    if args.backfill:
        if not args.start or not args.end:
            parser.error("--backfill requires both --start and --end dates")

        count = backfill_injury_reports(args.start, args.end)
        print(f"✓ Backfill complete: {count} new records")
    else:
        count = update_nba_official_injuries(days_back=args.days_back)
        print(f"✓ Update complete: {count} new records")


if __name__ == "__main__":
    main()
