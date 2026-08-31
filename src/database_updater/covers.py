"""
Covers.com scraping module for historical betting data.

This module provides two scraping approaches:
1. Matchups page (by date) - for on-demand finalization of recent games
2. Team schedule pages - for bulk historical backfill

Data collected:
- Closing spread (home team perspective)
- Spread result (W/L/P)
- Over/Under total
- O/U result (O/U/P)
- Final scores (for verification)
"""

import logging
import re
import time
from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# Rate limiting - be respectful to Covers.com
REQUEST_DELAY_SECONDS = 3.0

# User agent to mimic a real browser
# Note: Do NOT include brotli (br) in Accept-Encoding as requests can't decompress it
HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate",
}

# =============================================================================
# Team Mappings
# =============================================================================

# NBA tricode → Covers URL slug
NBA_TO_COVERS_SLUG = {
    "ATL": "atlanta-hawks",
    "BOS": "boston-celtics",
    "BKN": "brooklyn-nets",
    "CHA": "charlotte-hornets",
    "CHI": "chicago-bulls",
    "CLE": "cleveland-cavaliers",
    "DAL": "dallas-mavericks",
    "DEN": "denver-nuggets",
    "DET": "detroit-pistons",
    "GSW": "golden-state-warriors",
    "HOU": "houston-rockets",
    "IND": "indiana-pacers",
    "LAC": "los-angeles-clippers",
    "LAL": "los-angeles-lakers",
    "MEM": "memphis-grizzlies",
    "MIA": "miami-heat",
    "MIL": "milwaukee-bucks",
    "MIN": "minnesota-timberwolves",
    "NOP": "new-orleans-pelicans",
    "NYK": "new-york-knicks",
    "OKC": "oklahoma-city-thunder",
    "ORL": "orlando-magic",
    "PHI": "philadelphia-76ers",
    "PHX": "phoenix-suns",
    "POR": "portland-trail-blazers",
    "SAC": "sacramento-kings",
    "SAS": "san-antonio-spurs",
    "TOR": "toronto-raptors",
    "UTA": "utah-jazz",
    "WAS": "washington-wizards",
}

# Covers abbreviation → NBA tricode (Covers uses slightly different codes)
COVERS_ABBREV_TO_NBA = {
    "ATL": "ATL",
    "BOS": "BOS",
    "BK": "BKN",
    "BKN": "BKN",
    "CHA": "CHA",
    "CHAR": "CHA",
    "CHI": "CHI",
    "CLE": "CLE",
    "DAL": "DAL",
    "DEN": "DEN",
    "DET": "DET",
    "GS": "GSW",
    "GSW": "GSW",
    "HOU": "HOU",
    "IND": "IND",
    "LAC": "LAC",
    "LAL": "LAL",
    "MEM": "MEM",
    "MIA": "MIA",
    "MIL": "MIL",
    "MIN": "MIN",
    "NO": "NOP",
    "NOP": "NOP",
    "NY": "NYK",
    "NYK": "NYK",
    "OKC": "OKC",
    "ORL": "ORL",
    "PHI": "PHI",
    "PHO": "PHX",
    "PHX": "PHX",
    "POR": "POR",
    "SAC": "SAC",
    "SA": "SAS",
    "SAS": "SAS",
    "TOR": "TOR",
    "UTA": "UTA",
    "WAS": "WAS",
}


def normalize_team_abbrev(covers_abbrev: str) -> Optional[str]:
    """Convert Covers team abbreviation to NBA tricode."""
    abbrev = covers_abbrev.strip().upper()
    return COVERS_ABBREV_TO_NBA.get(abbrev)


def get_team_slug(nba_tricode: str) -> Optional[str]:
    """Get Covers URL slug for an NBA team."""
    return NBA_TO_COVERS_SLUG.get(nba_tricode.upper())


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class CoversGameData:
    """Parsed game data from Covers.com."""

    game_date: date
    home_team: str  # NBA tricode
    away_team: str  # NBA tricode
    home_score: Optional[int]
    away_score: Optional[int]
    spread: Optional[float]  # Home team perspective (negative = home favored)
    spread_result: Optional[str]  # 'W', 'L', 'P' (from home team perspective)
    total: Optional[float]  # Over/under line
    ou_result: Optional[str]  # 'O', 'U', 'P'


# =============================================================================
# Parsing Helpers
# =============================================================================


def _parse_spread(spread_text: str) -> Optional[float]:
    """
    Parse spread value from Covers text.

    Examples:
        "-6" → -6.0
        "2.5" → 2.5
        "-10.5" → -10.5
        "PK" → 0.0 (pick'em)
    """
    if not spread_text:
        return None

    spread_text = spread_text.strip()

    if spread_text.upper() == "PK":
        return 0.0

    try:
        return float(spread_text)
    except ValueError:
        logger.debug(f"Could not parse spread: {spread_text}")
        return None


def _parse_spread_result(result_text: str) -> Optional[str]:
    """Parse spread result (W/L/P) from text."""
    if not result_text:
        return None

    result = result_text.strip().upper()
    if result in ("W", "L", "P"):
        return result
    return None


def _parse_ou_result(result_text: str) -> Optional[str]:
    """Parse over/under result (O/U/P) from text."""
    if not result_text:
        return None

    result = result_text.strip().upper()
    if result in ("O", "U", "P"):
        return result
    return None


def _parse_total(total_text: str) -> Optional[float]:
    """Parse over/under total from text."""
    if not total_text:
        return None

    try:
        return float(total_text.strip())
    except ValueError:
        logger.debug(f"Could not parse total: {total_text}")
        return None


def _parse_score(score_text: str) -> tuple[Optional[int], Optional[int]]:
    """
    Parse score from format like "132-109" or "W 132-109".

    Returns (winner_score, loser_score) from the text order.
    """
    if not score_text:
        return None, None

    # Find pattern like "132-109"
    match = re.search(r"(\d+)-(\d+)", score_text)
    if match:
        try:
            return int(match.group(1)), int(match.group(2))
        except ValueError:
            pass

    return None, None


def _invert_spread_result(result: Optional[str]) -> Optional[str]:
    """Convert an ATS result from one team's perspective to the other team's."""
    return {"W": "L", "L": "W", "P": "P"}.get(result)


def _to_home_spread_perspective(
    subject_team: Optional[str],
    home_team: str,
    away_team: str,
    spread: Optional[float],
    spread_result: Optional[str],
) -> tuple[Optional[float], Optional[str]]:
    """Normalize a team-specific spread and result to the home team's perspective."""
    if subject_team == home_team:
        return spread, spread_result
    if subject_team == away_team:
        home_spread = -spread if spread is not None else None
        if home_spread == 0:
            home_spread = 0.0
        return home_spread, _invert_spread_result(spread_result)

    logger.debug(
        "Ignoring spread with unknown subject team %s for %s at %s",
        subject_team,
        away_team,
        home_team,
    )
    return None, None


def _team_text_aliases(team: str) -> list[str]:
    """Return Covers names and abbreviations that can identify an NBA team."""
    aliases = {team}
    slug = NBA_TO_COVERS_SLUG.get(team)
    if slug:
        full_name = slug.replace("-", " ")
        aliases.add(full_name)
        aliases.add("trail blazers" if team == "POR" else full_name.split()[-1])
    aliases.update(
        covers_code
        for covers_code, nba_code in COVERS_ABBREV_TO_NBA.items()
        if nba_code == team
    )
    return sorted(aliases, key=len, reverse=True)


def _parse_summary_spread(
    text: str, home_team: str, away_team: str
) -> tuple[Optional[float], Optional[str]]:
    """Parse a Covers summary sentence and return home-perspective ATS data."""
    for subject_team in (home_team, away_team):
        for alias in _team_text_aliases(subject_team):
            match = re.search(
                rf"\b{re.escape(alias)}\b\s+"
                r"(covered|did not cover)\s+the spread of\s+"
                r"(PK|[-+]?\d+(?:\.\d+)?)\b",
                text,
                re.I,
            )
            if not match:
                continue

            spread = _parse_spread(match.group(2))
            result = "W" if match.group(1).lower() == "covered" else "L"
            return _to_home_spread_perspective(
                subject_team, home_team, away_team, spread, result
            )

    return None, None


def _parse_fallback_spread(
    text: str, home_team: str, away_team: str
) -> Optional[float]:
    """Parse an abbreviated team line and return its home-perspective spread."""
    match = re.search(
        r"\b([A-Z]{2,4})\s*(PK|[-+]?\d+(?:\.\d+)?)\b", text, re.I
    )
    if not match:
        return None

    subject_team = normalize_team_abbrev(match.group(1))
    spread = _parse_spread(match.group(2))
    home_spread, _ = _to_home_spread_perspective(
        subject_team, home_team, away_team, spread, None
    )
    return home_spread


# =============================================================================
# Matchups Page Scraping (Tier 2)
# =============================================================================


def fetch_matchups_for_date(
    game_date: date, delay: float = REQUEST_DELAY_SECONDS
) -> list[CoversGameData]:
    """
    Fetch all NBA game betting data for a specific date from Covers matchups page.

    Args:
        game_date: The date to fetch games for
        delay: Seconds to wait before making request (rate limiting)

    Returns:
        List of CoversGameData for all games on that date
    """
    url = f"https://www.covers.com/sports/NBA/matchups?selectedDate={game_date.strftime('%Y-%m-%d')}"

    logger.debug(f"Fetching Covers matchups for {game_date}: {url}")

    # Rate limiting
    if delay > 0:
        time.sleep(delay)

    try:
        response = requests.get(url, headers=HEADERS, timeout=30)
        response.raise_for_status()
    except requests.RequestException as e:
        logger.error(f"Failed to fetch Covers matchups for {game_date}: {e}")
        return []

    return _parse_matchups_page(response.text, game_date)


def _parse_matchups_page(html: str, game_date: date) -> list[CoversGameData]:
    """
    Parse the Covers matchups page HTML.

    The matchups page uses <article class="gamebox"> elements with:
    - data-home-team-shortname: Home team abbreviation (lowercase)
    - data-away-team-shortname: Away team abbreviation (lowercase)
    - Score elements: <strong class="team-score home/away">
    - Summary box: Contains spread, total, and results text
    """
    soup = BeautifulSoup(html, "html.parser")
    games = []

    # Find all game boxes - now using article.gamebox instead of the old cmg classes
    game_boxes = soup.find_all("article", class_="gamebox")

    for box in game_boxes:
        try:
            # Extract data attributes (lowercase on Covers)
            home_abbrev = box.get("data-home-team-shortname", "")
            away_abbrev = box.get("data-away-team-shortname", "")

            # Convert team abbreviations
            home_team = normalize_team_abbrev(home_abbrev)
            away_team = normalize_team_abbrev(away_abbrev)

            if not home_team or not away_team:
                logger.debug(f"Could not map teams: {home_abbrev} vs {away_abbrev}")
                continue

            # Extract scores from team-score elements
            home_score = None
            away_score = None
            score_elements = box.find_all(class_="team-score")
            for elem in score_elements:
                classes = elem.get("class", [])
                text = elem.get_text(strip=True)
                if text.isdigit():
                    if "away" in classes:
                        away_score = int(text)
                    elif "home" in classes:
                        home_score = int(text)

            # Extract spread, total, and results from summary box
            spread = None
            spread_result = None
            total = None
            ou_result = None

            summary = box.find(class_="summary-box")
            if summary:
                text = summary.get_text(" ", strip=True)

                # Extract total: "was over 217" or "was under 217"
                total_match = re.search(r"was\s+(over|under)\s+(\d+\.?\d*)", text, re.I)
                if total_match:
                    ou_result = "O" if total_match.group(1).lower() == "over" else "U"
                    total = float(total_match.group(2))

                # Covers describes the line from the named team's perspective.
                # Preserve that subject so away-team lines can be inverted.
                spread, spread_result = _parse_summary_spread(
                    text, home_team, away_team
                )

            # Fallback: get spread from spread container if not in summary
            if spread is None:
                spread_container = box.find(class_="trending-and-cover-by-container")
                if spread_container:
                    # Look for pattern like "MIA -3.5" in span elements
                    span = spread_container.find(
                        string=re.compile(
                            r"\b[A-Z]{2,4}\s*(?:PK|[-+]?\d+(?:\.\d+)?)\b", re.I
                        )
                    )
                    if span:
                        spread = _parse_fallback_spread(
                            span.strip(), home_team, away_team
                        )

            game_data = CoversGameData(
                game_date=game_date,
                home_team=home_team,
                away_team=away_team,
                home_score=home_score,
                away_score=away_score,
                spread=spread,
                spread_result=spread_result,
                total=total,
                ou_result=ou_result,
            )
            games.append(game_data)

        except Exception as e:
            logger.warning(f"Error parsing game box: {e}")
            continue

    logger.debug(f"Parsed {len(games)} games from Covers matchups for {game_date}")
    return games


# =============================================================================
# Team Schedule Page Scraping (Tier 3)
# =============================================================================


def fetch_team_schedule(
    team: str, season: str, delay: float = REQUEST_DELAY_SECONDS
) -> list[CoversGameData]:
    """
    Fetch all home games for a team from their Covers schedule page.

    Args:
        team: NBA tricode (e.g., "BOS") or Covers slug (e.g., "boston-celtics")
        season: Season string (e.g., "2024-2025")
        delay: Seconds to wait before making request (rate limiting)

    Returns:
        List of CoversGameData for all HOME games (to avoid duplicates)
    """
    # Convert tricode to slug if needed
    home_team_tricode = None
    if len(team) <= 3:
        home_team_tricode = team.upper()
        slug = get_team_slug(home_team_tricode)
        if not slug:
            logger.error(f"Unknown team tricode: {team}")
            return []
    else:
        slug = team
        # Try to reverse-lookup the tricode from the slug
        for tricode, s in NBA_TO_COVERS_SLUG.items():
            if s == slug:
                home_team_tricode = tricode
                break

    url = f"https://www.covers.com/sport/basketball/nba/teams/main/{slug}/{season}"

    logger.debug(f"Fetching Covers team schedule: {url}")

    # Rate limiting
    if delay > 0:
        time.sleep(delay)

    try:
        response = requests.get(url, headers=HEADERS, timeout=30)
        response.raise_for_status()
    except requests.RequestException as e:
        logger.error(f"Failed to fetch Covers schedule for {team}: {e}")
        return []

    games = _parse_team_schedule_page(response.text, season)

    # Fill in the home team for all parsed games
    if home_team_tricode:
        for game in games:
            game.home_team = home_team_tricode

    return games


def _parse_team_schedule_page(html: str, season: str) -> list[CoversGameData]:
    """
    Parse the Covers team schedule page HTML.

    The team schedule page has tables with class 'covers-CoversResults-Table'.
    Structure (5 cells per row):
    - Cell 0: Combined data (often contains duplicate text)
    - Cell 1: Opponent (e.g., "HOU" for home, "@ POR" for away)
    - Cell 2: Score result (e.g., "W 140-109" or "L 81-109")
    - Cell 3: ATS result + spread (e.g., "W-13.5" or "L6")
    - Cell 4: O/U result + total (e.g., "O228" or "U216")

    We only return HOME games to avoid duplicates.
    """
    soup = BeautifulSoup(html, "html.parser")
    games = []

    # Determine season year for date parsing
    # Season "2024-2025" starts in Oct 2024
    try:
        start_year = int(season.split("-")[0])
    except (ValueError, IndexError):
        logger.error(f"Invalid season format: {season}")
        return []

    # Find the Regular Season results table
    # Tables have class 'covers-CoversResults-Table'
    results_tables = soup.find_all("table", class_="covers-CoversResults-Table")

    target_table = None
    for table in results_tables:
        first_th = table.find("th")
        if first_th and "Regular Season" in first_th.get_text():
            target_table = table
            break

    if not target_table:
        logger.warning("Could not find Regular Season results table")
        return games

    # Find all data rows (skip header row)
    rows = target_table.find_all("tr")[1:]

    for row in rows:
        try:
            cells = row.find_all("td")
            if len(cells) < 5:
                continue

            # Cell 1: Opponent (e.g., "HOU" for home, "@ POR" for away)
            opponent_cell = cells[1].get_text(strip=True)
            is_away = opponent_cell.startswith("@")
            opponent_abbrev = opponent_cell.replace("@", "").strip()

            # Skip away games - only store home games to avoid duplicates
            if is_away:
                continue

            # Cell 0: Extract date from combined data (e.g., "Apr 11HOUW 140-109...")
            # The date is at the beginning: "Apr 11" or "Oct 22"
            combined_text = cells[0].get_text(strip=True)
            date_match = re.match(r"([A-Z][a-z]{2}\s*\d{1,2})", combined_text)
            if not date_match:
                logger.debug(f"Could not parse date from: {combined_text[:20]}")
                continue
            date_text = date_match.group(1)

            # Cell 2: Result (e.g., "W 140-109" or "L 81-109")
            result_text = cells[2].get_text(strip=True)
            score1, score2 = _parse_score(result_text)

            # Determine home/away scores based on result
            # If "W", first score is the team's (home) score
            # If "L", first score is the team's (home) score (they lost)
            if result_text.startswith("W"):
                home_score = score1
                away_score = score2
            elif result_text.startswith("L"):
                home_score = score1
                away_score = score2
            else:
                home_score = score1
                away_score = score2

            # Cell 3: ATS result + spread (e.g., "W-13.5" or "L6")
            spread_cell = cells[3].get_text(strip=True)
            spread_match = re.match(r"([WLP])([-+]?\d+\.?\d*|PK)?", spread_cell)
            if spread_match:
                spread_result = spread_match.group(1)
                spread_str = spread_match.group(2) if spread_match.group(2) else None
                spread = _parse_spread(spread_str) if spread_str else None
            else:
                spread_result = None
                spread = None

            # Cell 4: O/U result + total (e.g., "O228" or "U216")
            ou_cell = cells[4].get_text(strip=True)
            ou_match = re.match(r"([OUP])(\d+\.?\d*)?", ou_cell)
            if ou_match:
                ou_result = ou_match.group(1)
                total = _parse_total(ou_match.group(2)) if ou_match.group(2) else None
            else:
                ou_result = None
                total = None

            # Parse game date
            game_date = _parse_game_date(date_text, start_year)
            if not game_date:
                continue

            # Get away team tricode
            away_team = normalize_team_abbrev(opponent_abbrev)
            if not away_team:
                logger.debug(f"Could not map opponent: {opponent_abbrev}")
                continue

            # Home team would need to be passed in or extracted from page
            # For now, we'll set it to None and fill in during processing
            home_team = None  # Will be set by caller based on which team page

            game_data = CoversGameData(
                game_date=game_date,
                home_team=home_team,  # To be filled by caller
                away_team=away_team,
                home_score=home_score,
                away_score=away_score,
                spread=spread,
                spread_result=spread_result,
                total=total,
                ou_result=ou_result,
            )
            games.append(game_data)

        except Exception as e:
            logger.warning(f"Error parsing schedule row: {e}")
            continue

    logger.debug(f"Parsed {len(games)} home games from Covers team schedule")
    return games


def _parse_game_date(date_text: str, season_start_year: int) -> Optional[date]:
    """
    Parse date from Covers format like "Oct 22" or "Jan 15".

    Args:
        date_text: Date string like "Oct 22"
        season_start_year: Year the season starts (e.g., 2024 for 2024-2025 season)

    Returns:
        date object or None if parsing fails
    """
    try:
        # Parse month and day
        parsed = datetime.strptime(date_text.strip(), "%b %d")
        month = parsed.month
        day = parsed.day

        # Determine year based on month
        # Oct-Dec = season start year, Jan-Jun = season start year + 1
        if month >= 10:
            year = season_start_year
        else:
            year = season_start_year + 1

        return date(year, month, day)
    except ValueError:
        logger.debug(f"Could not parse date: {date_text}")
        return None


# =============================================================================
# Bulk Operations
# =============================================================================


def fetch_season_all_teams(
    season: str, delay: float = REQUEST_DELAY_SECONDS
) -> list[CoversGameData]:
    """
    Fetch betting data for all teams in a season (Tier 3 backfill).

    This makes 30 requests (one per team) with rate limiting.

    Args:
        season: Season string (e.g., "2024-2025")
        delay: Seconds to wait between requests

    Returns:
        List of all home games for the season
    """
    all_games = []

    for tricode, slug in NBA_TO_COVERS_SLUG.items():
        logger.info(f"Fetching {tricode} ({slug}) for {season}...")

        games = fetch_team_schedule(slug, season, delay=delay)

        # Fill in home team for all games
        for game in games:
            game.home_team = tricode

        all_games.extend(games)

    logger.info(f"Fetched {len(all_games)} total home games for {season}")
    return all_games


def fetch_dates_with_unfinalized_games(
    dates: list[date], delay: float = REQUEST_DELAY_SECONDS
) -> list[CoversGameData]:
    """
    Fetch betting data for multiple specific dates (Tier 2 smart update).

    Args:
        dates: List of dates to fetch
        delay: Seconds to wait between requests

    Returns:
        List of all games from those dates
    """
    all_games = []

    for game_date in dates:
        games = fetch_matchups_for_date(game_date, delay=delay)
        all_games.extend(games)

    logger.info(f"Fetched {len(all_games)} games from {len(dates)} dates")
    return all_games
