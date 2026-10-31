"""
coaches.py

Keeps the Coaches table current for Phase5's coach features. The table was
filled historically by scripts/collect_coaching_data.py; this fetches head coaches
(CommonTeamRoster) only for teams missing one in the current season, so it makes
at most a handful of API calls, about once a season. Past seasons are left alone:
the endpoint can return today's staff for them.
"""

import logging
import time

from nba_api.stats.endpoints import commonteamroster

from src.database import get_db
from src.utils import StageLogger

REQUEST_TIMEOUT = 30
REQUEST_PAUSE = 2.0  # seconds between calls


def update_coaches(season, db_path):
    """Fetch coaching staffs for teams with no head coach in `season`.
    `season` is "YYYY-YYYY"; Coaches stores "YYYY-YY"."""
    stage_logger = StageLogger("Coaches")
    api_season = f"{season[:4]}-{season[-2:]}"

    with get_db(db_path) as conn:
        missing = conn.execute(
            """
            SELECT t.team_id FROM Teams t
            WHERE NOT EXISTS (
                SELECT 1 FROM Coaches c WHERE c.team_id = t.team_id
                AND c.season = ? AND c.is_head_coach = 1
            )
            """,
            (api_season,),
        ).fetchall()

    added = 0
    failures = 0
    for (team_id,) in missing:
        time.sleep(REQUEST_PAUSE)  # before every call, failed ones included
        try:
            dfs = commonteamroster.CommonTeamRoster(
                team_id=str(team_id), season=api_season, timeout=REQUEST_TIMEOUT
            ).get_data_frames()
            stage_logger.log_api_call()
            failures = 0
        except Exception as e:
            logging.warning(f"Coaches fetch failed for {team_id} {api_season}: {e}")
            # Consecutive failures mean the API is throttling; the rest are
            # still missing, so the next run picks them up
            failures += 1
            if failures == 3:
                logging.warning(
                    "Coaches stopped after 3 consecutive failures "
                    "(likely NBA API throttling)"
                )
                break
            continue
        if len(dfs) >= 2 and not dfs[1].empty:
            records = [
                (
                    str(team_id),
                    api_season,
                    str(row["COACH_ID"]),
                    row["COACH_NAME"],
                    # IS_ASSISTANT: 1=Head Coach, 2=Assistant, 3/4=Trainers
                    1 if row.get("IS_ASSISTANT") == 1 else 0,
                )
                for _, row in dfs[1].iterrows()
            ]
            with get_db(db_path) as conn:
                # OR IGNORE keeps existing rows (and their order: the first head
                # coach per team-season is the one the features use)
                cur = conn.executemany(
                    """
                    INSERT OR IGNORE INTO Coaches
                    (team_id, season, coach_id, coach_name, is_head_coach)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    records,
                )
                added += cur.rowcount
                conn.commit()

    stage_logger.set_counts(added=added, updated=0, removed=0)
    stage_logger.log_complete()
    return {"added": added, "checked": len(missing)}
