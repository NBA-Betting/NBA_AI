"""
Main game processing: iterate events, track lineup/possessions, accumulate stats.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict

from .constants import (
    DEAD_BALL_TO_SUBTYPES,
    FOUL_DRAWN_COLUMNS,
    HEAVE_DISTANCE,
    HEAVE_TIME,
    LIVE_BALL_TO_SUBTYPES,
    OVERTIME_PERIOD_SECONDS,
    REGULAR_PERIOD_SECONDS,
    SHOT_CLOCK_SUBTYPES,
    STAT_COLUMNS,
    STEP_OOB_SUBTYPES,
    THREE_SEC_SUBTYPES,
    TRAVEL_SUBTYPES,
)
from .event_parser import Event, parse_events
from .lineup_tracker import LineupTracker

logger = logging.getLogger(__name__)


def _resolve_assist_names(events: list[Event], tracker: LineupTracker):
    """Resolve assist_player_name → assist_player_id for older format events."""
    for evt in events:
        if evt.assist_player_name and not evt.assist_player_id:
            name = evt.assist_player_name.lower()
            # Try team-specific lookup (assister is on same team as scorer)
            pid = tracker._name_to_id.get((evt.team_id, name), 0)
            if not pid:
                pid = tracker._name_to_id_global.get(name, 0)
            if not pid:
                # Partial match
                for (t, n), p in tracker._name_to_id.items():
                    if t == evt.team_id and name in n:
                        pid = p
                        break
            evt.assist_player_id = pid


def _new_player_stats() -> dict[str, float]:
    """Create a zeroed stat dict for one player."""
    return {col: 0.0 for col in STAT_COLUMNS}


def process_game(
    events_raw: list[dict],
    home_team_id: int,
    away_team_id: int,
    game_id: str = "",
    player_names: dict[int, tuple[int, list[str]]] | None = None,
) -> dict[int, dict]:
    """
    player_names: optional {player_id: (team_id, [name forms])} for the
    game's roster (box score), to resolve substitutions of players with no
    other actions.

    Process all PBP events for one game and return per-player stats.

    Returns:
        {player_id: {"team_id": int, "stats": {stat_name: value, ...}}}
    """
    if not events_raw:
        return {}

    events = parse_events(events_raw)
    if not events:
        return {}

    tracker = LineupTracker(events, home_team_id, away_team_id, player_names)

    # Resolve assist player names → IDs for older format events
    _resolve_assist_names(events, tracker)

    # Per-player stat accumulators
    player_stats: dict[int, dict[str, float]] = defaultdict(_new_player_stats)
    player_teams: dict[int, int] = {}  # player_id → team_id

    # Track offense team for possession logic. The live feed states it on each
    # event; the stats feed doesn't, so each event's own facts decide it there
    offense_team_id = 0
    states_possession = any(evt.possession_team_id for evt in events)

    # Build action_number → event index for linking (rebounds → shots, etc.)
    action_num_to_idx: dict[int, int] = {}
    for evt in events:
        if evt.action_number:
            action_num_to_idx[evt.action_number] = evt.idx

    # Process period by period
    periods = sorted(set(evt.period for evt in events if evt.period > 0))

    for period in periods:
        tracker.initialize_period(period)

        period_start_seconds = (
            REGULAR_PERIOD_SECONDS if period <= 4 else OVERTIME_PERIOD_SECONDS
        )

        period_events = [e for e in events if e.period == period]
        prev_clock = period_start_seconds
        pending_seconds: list[tuple[dict[int, set[int]], float]] = []
        # Stats feed: team with the ball after the last event (0 = not known
        # yet, e.g. after the tip) and whether its possession has had an event
        ball = 0
        possession_open = False

        for evt_i, evt in enumerate(period_events):
            # --- Record player team ---
            if evt.player_id and evt.team_id:
                player_teams[evt.player_id] = evt.team_id

            # --- Update offense team (possession tracking) ---
            if states_possession:
                if evt.possession_team_id:
                    offense_team_id = evt.possession_team_id
                ends_possession = None  # decided by _is_possession_ending_event
            else:
                # Who had the ball up to this event, and who has it after
                offense_team_id, ball = _older_possession(evt, events, tracker, ball)
                ends_possession = bool(offense_team_id) and ball != offense_team_id

            # --- Seconds attribution ---
            # Before the first possession is known (older format learns it at
            # the first make, free throw or turnover) seconds are held with the
            # lineup on court, then credited once that team is known: it had
            # the ball all along
            seconds_elapsed = prev_clock - evt.clock_seconds
            if seconds_elapsed > 0:
                on_court = {
                    tid: set(players)
                    for tid, players in tracker.get_current_players().items()
                }
                pending_seconds.append((on_court, seconds_elapsed))
            if offense_team_id:
                _credit_seconds(player_stats, pending_seconds, offense_team_id)
            prev_clock = evt.clock_seconds

            # --- Substitution handling ---
            if evt.is_substitution:
                tracker.process_substitution(evt)
                continue

            # --- Period start/end ---
            if evt.is_period_start or evt.is_period_end:
                if evt.is_period_start:
                    prev_clock = period_start_seconds
                elif ends_possession is not None and possession_open:
                    # Stats feed: the period ends the possession in progress
                    # (the live feed credits it at its last event)
                    _credit_possession(tracker, offense_team_id, player_stats)
                    possession_open = False
                continue

            if ends_possession is not None:
                possession_open = (possession_open or bool(offense_team_id)) and not ends_possession

            # Team events (team rebounds, team turnovers) have no player but
            # can still end a possession
            if not evt.player_id:
                if _ends(evt, events, offense_team_id, tracker, ends_possession):
                    _credit_possession(tracker, offense_team_id, player_stats)
                continue

            pid = evt.player_id
            stats = player_stats[pid]

            # --- Field goal ---
            if evt.is_field_goal:
                _process_field_goal(evt, stats, player_stats, tracker, events)

            # --- Rebound ---
            elif evt.is_rebound:
                _process_rebound(
                    evt, stats, player_stats, tracker, events, action_num_to_idx
                )

            # --- Turnover ---
            elif evt.is_turnover:
                _process_turnover(evt, stats, player_stats)

            # --- Foul ---
            elif evt.is_foul:
                _process_foul(evt, stats, player_stats)

            # --- Free throw ---
            elif evt.is_free_throw:
                _process_free_throw(evt, stats, player_stats, tracker, offense_team_id)

            # --- Possession end check ---
            if _ends(evt, events, offense_team_id, tracker, ends_possession):
                _credit_possession(tracker, offense_team_id, player_stats)

        # Offense never known this period: still count the playing time
        _credit_seconds(player_stats, pending_seconds, offense_team_id)

    # Build result. Players who only ever appear through lineups (no own
    # events) take their team from the court or the box score; anyone never
    # placed on a team (coaches charged with technicals) is not a player row.
    box_teams = {pid: tid for pid, (tid, _) in (player_names or {}).items()}
    result = {}
    for pid, stats in player_stats.items():
        team_id = player_teams.get(pid) or next(
            (tid for tid, ever in tracker._ever_on_court.items() if pid in ever),
            box_teams.get(pid, 0),
        )
        if pid == 0 or not team_id:
            continue
        result[pid] = {
            "team_id": team_id,
            "stats": stats,
            "lineup_starters_found": tracker.all_starters_found,
        }

    return result


def _credit_seconds(player_stats, pending_seconds, offense_team_id):
    """Credit held (lineup, seconds) spans as offensive or defensive time,
    then clear them. offense_team_id 0 (unknown) counts as defense for both."""
    for on_court, seconds in pending_seconds:
        for tid, players in on_court.items():
            is_offense = offense_team_id and tid == offense_team_id
            key = "seconds_played_off" if is_offense else "seconds_played_def"
            for pid in players:
                player_stats[pid][key] += seconds
    pending_seconds.clear()


# ---------------------------------------------------------------------------
# Event processing helpers
# ---------------------------------------------------------------------------


def _process_field_goal(
    evt: Event,
    stats: dict,
    player_stats: dict[int, dict],
    tracker: LineupTracker,
    events: list[Event],
):
    """Process a field goal event (made or missed)."""
    zone = evt.shot_zone
    if not zone:
        zone = "long_mid" if not evt.is_3pt else "arc3"  # fallback

    # FGA always
    stats[f"{zone}_fga"] += 1

    # Shot distance tracking
    if evt.shot_distance is not None:
        if evt.is_3pt:
            stats["total_3pt_shot_distance"] += evt.shot_distance
            stats["total_3pt_shots_with_distance"] += 1
        else:
            stats["total_2pt_shot_distance"] += evt.shot_distance
            stats["total_2pt_shots_with_distance"] += 1

    # Heave detection
    is_heave = (
        evt.shot_distance is not None
        and evt.shot_distance > HEAVE_DISTANCE
        and evt.clock_seconds < HEAVE_TIME
    )

    if evt.is_made:
        stats[f"{zone}_fgm"] += 1

        if is_heave:
            stats["heave_makes"] += 1

        # Assisted vs unassisted
        if evt.assist_player_id:
            stats[f"assisted_{zone}"] += 1
            # Credit assister
            assister_stats = player_stats[evt.assist_player_id]
            assister_stats[f"ast_{zone}"] += 1
        else:
            stats[f"unassisted_{zone}"] += 1
            # Putback check
            if _is_putback(evt, events):
                stats["putbacks"] += 1

        # Plus/minus for all 10 on-court players
        on_court = tracker.get_current_players()
        for tid, players in on_court.items():
            pm_val = evt.shot_value if tid == evt.team_id else -evt.shot_value
            for p in players:
                player_stats[p]["plus_minus"] += pm_val

    else:
        # Missed shot
        if is_heave:
            stats["heave_misses"] += 1

        if evt.block_player_id:
            # Blocked shot
            stats[f"blocked_{zone}"] += 1
            blocker_stats = player_stats[evt.block_player_id]
            blocker_stats[f"block_{zone}"] += 1
        else:
            # Missed unblocked
            stats[f"missed_{zone}"] += 1


def _is_putback(evt: Event, events: list[Event]) -> bool:
    """Check if a made shot is a putback (unassisted 2PT within 2s of own OREB)."""
    if evt.is_3pt or evt.assist_player_id:
        return False

    # Check descriptor (v3)
    if evt.descriptor and "putback" in evt.descriptor.lower():
        return True

    # Scan backward for own offensive rebound within 2 seconds
    idx = evt.idx
    for i in range(idx - 1, max(idx - 10, -1), -1):
        if i < 0 or i >= len(events):
            break
        prev = events[i]
        time_diff = prev.clock_seconds - evt.clock_seconds
        if time_diff > 2.0:
            break
        if time_diff < 0:
            continue
        if (
            prev.is_rebound
            and prev.reb_is_offensive
            and prev.player_id == evt.player_id
        ):
            return True
        # Skip fouls/violations at same time
        if prev.is_foul or prev.action_type == "violation":
            continue
    return False


def _process_rebound(
    evt: Event,
    stats: dict,
    player_stats: dict[int, dict],
    tracker: LineupTracker,
    events: list[Event],
    action_num_to_idx: dict[int, int],
):
    """Process a rebound event."""
    # Skip team rebounds (personId = 0)
    if evt.player_id == 0:
        return

    # Determine the missed shot's zone
    shot_zone = _get_rebound_shot_zone(evt, events, action_num_to_idx)

    if evt.reb_is_offensive:
        prefix = "oreb"
    else:
        prefix = "dreb"

    # Zone-specific rebound
    if shot_zone:
        stats[f"{prefix}_{shot_zone}"] += 1
    else:
        # FT rebound or unknown zone
        stats[f"{prefix}_ft"] += 1

    # Self-rebound check
    missed_shot = _find_missed_shot_for_rebound(evt, events, action_num_to_idx)
    if missed_shot and missed_shot.player_id == evt.player_id and evt.reb_is_offensive:
        stats["self_oreb"] += 1

    # Rebound opportunities for all on-court players
    on_court = tracker.get_current_players()
    for tid, players in on_court.items():
        is_off_team = tid == evt.team_id
        if evt.reb_is_offensive:
            opp_key = "oreb_opportunities" if is_off_team else "dreb_opportunities"
        else:
            opp_key = "dreb_opportunities" if is_off_team else "oreb_opportunities"
        for p in players:
            player_stats[p][opp_key] += 1

    # On-floor offensive rebound tracking
    if evt.reb_is_offensive:
        for p in on_court.get(evt.team_id, set()):
            player_stats[p]["on_floor_oreb"] += 1


def _get_rebound_shot_zone(
    evt: Event,
    events: list[Event],
    action_num_to_idx: dict[int, int],
) -> str:
    """Get the shot zone of the missed shot that this rebound came from."""
    missed = _find_missed_shot_for_rebound(evt, events, action_num_to_idx)
    if missed:
        return missed.shot_zone
    return ""


def _find_missed_shot_for_rebound(
    evt: Event,
    events: list[Event],
    action_num_to_idx: dict[int, int],
) -> Event | None:
    """Find the missed shot event that preceded this rebound."""
    # V3: use shotActionNumber linkage
    if evt.reb_shot_action_number is not None:
        idx = action_num_to_idx.get(evt.reb_shot_action_number)
        if idx is not None and idx < len(events):
            shot = events[idx]
            if shot.is_field_goal and not shot.is_made:
                return shot

    # Fallback: scan backward for the most recent missed shot or FT
    for i in range(evt.idx - 1, max(evt.idx - 15, -1), -1):
        if i < 0:
            break
        prev = events[i]
        if prev.is_field_goal and not prev.is_made:
            return prev
        if prev.is_free_throw and not prev.is_ft_made:
            return prev
        # Don't scan past the previous scoring play or possession end
        if prev.is_field_goal and prev.is_made:
            break
        if prev.is_period_start or prev.is_period_end:
            break
    return None


def _process_turnover(
    evt: Event,
    stats: dict,
    player_stats: dict[int, dict],
):
    """Process a turnover event."""
    subtype = evt.to_subtype

    has_steal = evt.steal_player_id != 0

    if has_steal:
        # Live ball turnover
        if "bad" in subtype or "pass" in subtype:
            stats["bad_pass_turnovers"] += 1
            player_stats[evt.steal_player_id]["bad_pass_steals"] += 1
        else:
            # Default to lost ball for other steal-accompanied turnovers
            stats["lost_ball_turnovers"] += 1
            player_stats[evt.steal_player_id]["lost_ball_steals"] += 1
    else:
        # Dead ball turnover
        stats["deadball_turnovers"] += 1

        # Specific sub-categories
        if subtype in TRAVEL_SUBTYPES:
            stats["travels"] += 1
        elif subtype in THREE_SEC_SUBTYPES:
            stats["three_sec_violations"] += 1
        elif subtype in SHOT_CLOCK_SUBTYPES:
            stats["shot_clock_violations"] += 1
        elif subtype in STEP_OOB_SUBTYPES:
            stats["step_out_of_bounds"] += 1
        elif "offensive" in subtype and "foul" in subtype:
            stats["offensive_fouls_to"] += 1


def _process_foul(
    evt: Event,
    stats: dict,
    player_stats: dict[int, dict],
):
    """Process a foul event."""
    foul_col = evt.foul_type
    if not foul_col:
        foul_col = "personal_fouls"  # fallback

    # Don't count technicals in personal foul stats for committing player
    # (technicals are separate)
    stats[foul_col] += 1

    # Foul drawn
    if evt.foul_drawn_player_id and not evt.is_technical:
        drawn_col = FOUL_DRAWN_COLUMNS.get(foul_col)
        if drawn_col:
            player_stats[evt.foul_drawn_player_id][drawn_col] += 1


def _process_free_throw(
    evt: Event,
    stats: dict,
    player_stats: dict[int, dict],
    tracker: LineupTracker,
    offense_team_id: int,
):
    """Process a free throw event."""
    if evt.is_technical_ft:
        if evt.is_ft_made:
            stats["tech_fts_made"] += 1
        # Technical FTs don't count as regular FT makes/misses
        return

    if evt.is_ft_made:
        stats["fts_made"] += 1
        # Plus/minus for all 10 on-court players (1 point)
        on_court = tracker.get_current_players()
        for tid, players in on_court.items():
            pm_val = 1 if tid == evt.team_id else -1
            for p in players:
                player_stats[p]["plus_minus"] += pm_val
    else:
        stats["fts_missed"] += 1


# ---------------------------------------------------------------------------
# Possession logic
# ---------------------------------------------------------------------------


def _ends(evt, events, offense_team_id, tracker, ends_possession) -> bool:
    """Whether the event ends the possession: stated for the stats feed
    (ends_possession), looked up for the live feed."""
    if ends_possession is not None:
        return ends_possession
    return _is_possession_ending_event(evt, events, offense_team_id, tracker)


def _is_possession_ending_event(
    evt: Event,
    events: list[Event],
    offense_team_id: int,
    tracker: LineupTracker,
) -> bool:
    """Live feed: the event ends a possession if the next event with a
    possession field names the other team (or the period ends)."""
    if not evt.possession_team_id:
        return False
    for i in range(evt.idx + 1, min(evt.idx + 10, len(events))):
        nxt = events[i]
        if nxt.is_period_end:
            return True
        if nxt.possession_team_id and nxt.possession_team_id != evt.possession_team_id:
            return True
        if nxt.possession_team_id:
            return False  # same team still has it
    return False


_FT_NUMBER_RE = re.compile(r"(\d) of (\d)")


def _is_last_free_throw(evt: Event) -> bool:
    """True for the last of a free-throw set ("2 of 2"); technicals have no count."""
    m = _FT_NUMBER_RE.search(evt.description)
    return bool(m) and m.group(1) == m.group(2)


def _is_and_one(evt: Event, events: list[Event]) -> bool:
    """A made shot followed, at the same clock, by a (non-technical) foul."""
    for nxt in events[evt.idx + 1 : evt.idx + 5]:
        if abs(nxt.clock_seconds - evt.clock_seconds) > 0.1:
            break
        if nxt.is_foul and not nxt.is_technical:
            return True
    return False


def _older_possession(
    evt: Event, events: list[Event], tracker: LineupTracker, ball: int
) -> tuple[int, int]:
    """Stats feed: (team with the ball up to this event, team with it after).

    Shots, free throws and turnovers belong to the team with the ball and
    rebounds show which side had it, so every such event re-anchors the
    offense (an inference error can't carry on). Other events keep `ball`.
    """
    team = evt.team_id
    if not team:
        return ball, ball
    other = tracker._opposite_team(team)
    if evt.is_rebound:
        return (team if evt.reb_is_offensive else other), team
    if evt.is_field_goal:
        keeps = not evt.is_made or _is_and_one(evt, events)
        return team, team if keeps else other
    if evt.is_free_throw and not evt.is_technical_ft:
        # Flagrant and clear-path free throws are followed by the same team's ball
        keeps_ball = "Flagrant" in evt.description or "Clear Path" in evt.description
        if evt.is_ft_made and _is_last_free_throw(evt) and not keeps_ball:
            return team, other
        return team, team
    if evt.is_turnover:
        return team, other
    return ball, ball


def _credit_possession(
    tracker: LineupTracker,
    offense_team_id: int,
    player_stats: dict[int, dict],
):
    """Credit OffPoss/DefPoss to all 5 on-court players per team."""
    if not offense_team_id:
        return
    on_court = tracker.get_current_players()
    for tid, players in on_court.items():
        key = "off_poss" if tid == offense_team_id else "def_poss"
        for pid in players:
            player_stats[pid][key] += 1
