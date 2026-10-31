"""
player_attributes.py

Adds PlayerAttributes rows (height, weight, draft position, birth year) for
rostered players who have none, e.g. incoming rookies, so Phase5's L1 player
profiles aren't built from defaults. The table was filled historically by
scripts/collect_player_attributes.py; existing rows are never changed here.
"""

import logging
import time

from nba_api.stats.endpoints import commonplayerinfo

from src.database import get_db
from src.utils import StageLogger

REQUEST_TIMEOUT = 30
REQUEST_PAUSE = 1.0  # seconds between calls

POSITION_CODES = {"Guard": "G", "Forward": "F", "Center": "C"}


def _to_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _height_inches(height):
    feet, _, inches = (height or "").partition("-")
    if _to_int(feet) is None or _to_int(inches) is None:
        return None
    return int(feet) * 12 + int(inches)


def update_player_attributes(db_path):
    """Fetch CommonPlayerInfo for rostered players missing from PlayerAttributes."""
    stage_logger = StageLogger("PlayerAttributes")
    with get_db(db_path) as conn:
        missing = [row[0] for row in conn.execute("""
                SELECT p.person_id FROM Players p
                LEFT JOIN PlayerAttributes pa ON pa.person_id = p.person_id
                WHERE p.roster_status = 1 AND pa.person_id IS NULL
                """)]

    records = []
    failures = 0
    for person_id in missing:
        time.sleep(REQUEST_PAUSE)  # before every call, failed ones included
        try:
            info = commonplayerinfo.CommonPlayerInfo(
                player_id=person_id, timeout=REQUEST_TIMEOUT
            ).get_data_frames()[0]
            stage_logger.log_api_call()
            failures = 0
        except Exception as e:
            logging.warning(f"PlayerAttributes fetch failed for {person_id}: {e}")
            # Consecutive failures mean the API is throttling; the rest are
            # still missing, so the next run picks them up
            failures += 1
            if failures == 3:
                logging.warning(
                    "PlayerAttributes stopped after 3 consecutive failures "
                    "(likely NBA API throttling)"
                )
                break
            continue
        if info.empty:
            continue
        r = info.iloc[0]
        draft_number = _to_int(r["DRAFT_NUMBER"])
        position = "-".join(
            POSITION_CODES.get(p, "") for p in (r["POSITION"] or "").split("-")
        ).strip("-")
        birth_year = _to_int(str(r["BIRTHDATE"] or "")[:4])
        records.append(
            (
                person_id,
                r["FIRST_NAME"],
                r["LAST_NAME"],
                position,
                r["HEIGHT"] or None,
                _height_inches(r["HEIGHT"]),
                _to_int(r["WEIGHT"]),
                r["COUNTRY"],
                r["SCHOOL"],
                _to_int(r["DRAFT_YEAR"]),
                _to_int(r["DRAFT_ROUND"]),
                draft_number,
                0 if draft_number else 1,
                _to_int(r["FROM_YEAR"]),
                _to_int(r["TO_YEAR"]),
                birth_year,
            )
        )

    if records:
        with get_db(db_path) as conn:
            conn.executemany(
                """
                INSERT OR IGNORE INTO PlayerAttributes (
                    person_id, first_name, last_name, position, height,
                    height_inches, weight, country, college, draft_year,
                    draft_round, draft_number, undrafted, from_year, to_year,
                    birth_year
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                records,
            )
            conn.commit()

    stage_logger.set_counts(added=len(records), updated=0, removed=0)
    stage_logger.log_complete()
    return {"added": len(records), "checked": len(missing)}
