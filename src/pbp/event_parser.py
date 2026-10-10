"""
Normalize raw PBP JSON events into a common Event dataclass.
Handles both the older format (pre-2022) and v3 format (2022+).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Optional

from .constants import (
    AT_RIM_CUTOFF,
    CORNER_3_Y_CUTOFF,
    PAINT_DEPTH,
    PAINT_HALF_WIDTH,
    RESTRICTED_AREA_RADIUS,
    SHORT_MID_CUTOFF,
    V3_AREA_TO_ZONE,
)

# Regex for parsing assist from older format descriptions
# e.g., "(Irving 1 AST)" or "(James 12 AST)"
_ASSIST_RE = re.compile(r"\(([^)]+?)\s+\d+\s+AST\)")

# Regex for parsing "SUB: IncomingPlayer FOR OutgoingPlayer"
_SUB_RE = re.compile(r"SUB:\s*(.+?)\s+FOR\s+(.+?)$", re.IGNORECASE)

# Regex for parsing shot distance from description, e.g., "23' Jump Shot"
_DISTANCE_RE = re.compile(r"(\d+)'\s")


@dataclass
class Event:
    """Normalized PBP event with a common interface for both format eras."""

    idx: int  # position in event list
    action_type: str  # 'field_goal', 'free_throw', 'rebound', 'turnover',
    # 'foul', 'substitution', 'period_start', 'period_end',
    # 'jump_ball', 'timeout', 'violation', 'ejection', 'other'
    player_id: int  # primary player (personId)
    team_id: int
    period: int
    clock_seconds: float  # seconds remaining in period
    description: str

    # Shot-specific
    is_field_goal: bool = False
    is_made: bool = False
    is_3pt: bool = False
    shot_zone: str = ""  # 'at_rim', 'short_mid', 'long_mid', 'corner3', 'arc3'
    shot_distance: Optional[float] = None
    loc_x: Optional[int] = None  # legacy x coordinate
    loc_y: Optional[int] = None  # legacy y coordinate
    shot_value: int = 0  # 2 or 3

    # Assist/block attribution
    assist_player_id: int = 0
    assist_player_name: str = ""  # for older format: name parsed from description
    block_player_id: int = 0

    # Rebound-specific
    is_rebound: bool = False
    reb_is_offensive: bool = False
    reb_shot_action_number: Optional[int] = None  # v3: links rebound to missed shot

    # Turnover-specific
    is_turnover: bool = False
    to_subtype: str = ""  # 'bad_pass', 'lost_ball', 'travel', etc.
    steal_player_id: int = 0

    # Foul-specific
    is_foul: bool = False
    foul_type: str = ""  # mapped column name from constants
    foul_drawn_player_id: int = 0
    is_technical: bool = False

    # Free throw specific
    is_free_throw: bool = False
    is_ft_made: bool = False
    is_technical_ft: bool = False

    # Substitution-specific
    is_substitution: bool = False
    sub_in_player_id: int = 0
    sub_out_player_id: int = 0
    sub_direction: str = ""  # 'in', 'out', 'both' (older format has both in one event)

    # Period markers
    is_period_start: bool = False
    is_period_end: bool = False

    # Jump ball (v3): winner and loser, who are on court even when the tip
    # goes to the team (personId 0)
    jump_ball_player_ids: list = field(default_factory=list)

    # V3-specific
    possession_team_id: int = 0  # from v3's 'possession' field
    qualifiers: list = field(default_factory=list)
    descriptor: str = ""

    # Raw data for debugging
    action_number: int = 0
    order_number: int = 0


def parse_clock(clock_str: str) -> float:
    """Parse 'PT12M34.00S' or 'PT09M52.00S' format to seconds remaining."""
    if not clock_str:
        return 0.0
    s = clock_str.replace("PT", "").replace("S", "")
    parts = s.split("M")
    if len(parts) == 2:
        try:
            minutes = float(parts[0])
            seconds = float(parts[1]) if parts[1] else 0.0
            return minutes * 60 + seconds
        except (ValueError, TypeError):
            return 0.0
    return 0.0


def _safe_int(val, default=0) -> int:
    if val is None or val == "":
        return default
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


def _safe_float(val, default=None) -> Optional[float]:
    if val is None or val == "":
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def _compute_distance(loc_x: Optional[int], loc_y: Optional[int]) -> Optional[float]:
    """Compute shot distance in feet from legacy coordinates."""
    if loc_x is None or loc_y is None:
        return None
    if loc_x == 0 and loc_y == 0:
        return None  # no real coordinates
    return round(math.sqrt(loc_x**2 + loc_y**2) / 10.0, 1)


def _classify_zone_by_coords(
    distance: Optional[float],
    loc_x: Optional[int],
    loc_y: Optional[int],
    is_3pt: bool,
) -> str:
    """Classify shot zone from coordinates (older format), matching the live
    feed's area labels; falls back to distance when coordinates are missing."""
    if is_3pt:
        if loc_y is not None and abs(loc_y) <= CORNER_3_Y_CUTOFF:
            return "corner3"
        return "arc3"
    if loc_x is not None and loc_y is not None and (loc_x, loc_y) != (0, 0):
        if math.hypot(loc_x, loc_y) <= RESTRICTED_AREA_RADIUS:
            return "at_rim"
        if abs(loc_x) <= PAINT_HALF_WIDTH and loc_y <= PAINT_DEPTH:
            return "short_mid"
        return "long_mid"
    if (loc_x, loc_y) == (0, 0) and not distance:
        return "at_rim"  # tips and dunks are logged at the hoop with no distance
    if distance is None:
        return ""
    if distance < AT_RIM_CUTOFF:
        return "at_rim"
    if distance < SHORT_MID_CUTOFF:
        return "short_mid"
    return "long_mid"


def _classify_zone_v3(area: str, is_3pt: bool) -> str:
    """Classify shot zone from v3 area field."""
    zone = V3_AREA_TO_ZONE.get(area, "")
    if zone:
        return zone
    # Fallback: if area not recognized but we know it's 3PT
    if is_3pt:
        return "arc3"
    return ""


# ---------------------------------------------------------------------------
# Older format normalization
# ---------------------------------------------------------------------------

_OLDER_FG_TYPES = {"Made Shot", "Missed Shot"}
_OLDER_ACTION_MAP = {
    "Rebound": "rebound",
    "Turnover": "turnover",
    "Foul": "foul",
    "Free Throw": "free_throw",
    "Substitution": "substitution",
    "Jump Ball": "jump_ball",
    "Timeout": "timeout",
    "Violation": "violation",
    "Ejection": "ejection",
    "Instant Replay": "other",
    "period": "period_marker",
}


def from_older(raw: dict, idx: int) -> Event:
    """Create an Event from the older PBP format (pre-2022)."""
    action_type_raw = raw.get("actionType", "")
    sub_type = raw.get("subType", "")
    desc = raw.get("description", "") or ""
    person_id = _safe_int(raw.get("personId"))
    team_id = _safe_int(raw.get("teamId"))
    period = _safe_int(raw.get("period"))
    clock_seconds = parse_clock(raw.get("clock", ""))
    loc_x = _safe_int(raw.get("xLegacy"), default=None)
    loc_y = _safe_int(raw.get("yLegacy"), default=None)

    evt = Event(
        idx=idx,
        action_type="other",
        player_id=person_id,
        team_id=team_id,
        period=period,
        clock_seconds=clock_seconds,
        description=desc,
        loc_x=loc_x,
        loc_y=loc_y,
        action_number=_safe_int(raw.get("actionNumber")),
        order_number=_safe_int(raw.get("actionId")),
    )

    # --- Field goal ---
    if action_type_raw in _OLDER_FG_TYPES:
        evt.action_type = "field_goal"
        evt.is_field_goal = True
        evt.is_made = action_type_raw == "Made Shot"
        evt.is_3pt = "3PT" in desc or sub_type == "3PT"
        evt.shot_value = 3 if evt.is_3pt else 2

        # Distance
        raw_dist = _safe_float(raw.get("shotDistance"))
        computed_dist = _compute_distance(loc_x, loc_y)
        evt.shot_distance = raw_dist if raw_dist else computed_dist
        if evt.shot_distance is None and desc:
            m = _DISTANCE_RE.search(desc)
            if m:
                evt.shot_distance = float(m.group(1))

        # Zone
        evt.shot_zone = _classify_zone_by_coords(
            evt.shot_distance, loc_x, loc_y, evt.is_3pt
        )

        # Assist — check for assistPersonId field first, then parse description
        assist_pid = _safe_int(raw.get("assistPersonId"))
        if assist_pid:
            evt.assist_player_id = assist_pid
        elif evt.is_made and "AST" in desc:
            # Parse assister name from description: "(Walker 1 AST)"
            m = _ASSIST_RE.search(desc)
            if m:
                evt.assist_player_name = m.group(1).strip()

        # Block
        block_pid = _safe_int(raw.get("blockPersonId"))
        if block_pid:
            evt.block_player_id = block_pid

    # --- Rebound ---
    elif action_type_raw == "Rebound":
        evt.action_type = "rebound"
        evt.is_rebound = True
        # Offensive vs defensive is resolved in _link_older_events
        if 1610612737 <= person_id <= 1610612766:
            # Team rebound: personId holds the team id
            evt.team_id, evt.player_id = person_id, 0

    # --- Turnover ---
    elif action_type_raw == "Turnover":
        evt.action_type = "turnover"
        evt.is_turnover = True
        evt.to_subtype = sub_type.lower().strip() if sub_type else ""
        # Steal player
        steal_pid = _safe_int(raw.get("stealPersonId"))
        if steal_pid:
            evt.steal_player_id = steal_pid

    # --- Foul ---
    elif action_type_raw == "Foul":
        evt.action_type = "foul"
        evt.is_foul = True
        from .constants import OLDER_FOUL_MAP

        evt.foul_type = OLDER_FOUL_MAP.get(sub_type, "personal_fouls")
        evt.is_technical = "Technical" in sub_type or "Defense 3 Second" in sub_type
        foul_drawn = _safe_int(raw.get("foulDrawnPersonId")) or _safe_int(
            raw.get("player3Id")
        )
        if foul_drawn:
            evt.foul_drawn_player_id = foul_drawn

    # --- Free throw ---
    elif action_type_raw == "Free Throw":
        evt.action_type = "free_throw"
        evt.is_free_throw = True
        shot_result = raw.get("shotResult", "")
        evt.is_ft_made = (
            shot_result == "Made"
            or "Makes" in desc
            or ("MISS" not in desc.upper() and raw.get("pointsTotal", 0))
        )
        evt.is_technical_ft = "Technical" in sub_type or "technical" in desc.lower()

    # --- Substitution ---
    elif action_type_raw == "Substitution":
        evt.action_type = "substitution"
        evt.is_substitution = True
        evt.sub_out_player_id = person_id  # personId = outgoing player
        evt.sub_direction = "both"
        # Incoming player: parse from description "SUB: IncomingName FOR OutgoingName"
        m = _SUB_RE.search(desc)
        if m:
            # Store incoming name — stat_computer resolves to ID via lookup
            evt.descriptor = m.group(
                1
            ).strip()  # reuse descriptor field for incoming name

    # --- Period markers ---
    elif action_type_raw == "period":
        if sub_type == "start":
            evt.action_type = "period_start"
            evt.is_period_start = True
        elif sub_type == "end":
            evt.action_type = "period_end"
            evt.is_period_end = True
        else:
            evt.action_type = "period_marker"

    # --- Other ---
    elif action_type_raw in _OLDER_ACTION_MAP:
        evt.action_type = _OLDER_ACTION_MAP[action_type_raw]
    elif action_type_raw == "" and "BLOCK" in desc.upper():
        # Older format: standalone block events have empty actionType
        evt.action_type = "block_event"
        evt.block_player_id = person_id
    elif action_type_raw == "" and "STEAL" in desc.upper():
        evt.action_type = "steal_event"
        evt.steal_player_id = person_id

    return evt


# ---------------------------------------------------------------------------
# V3 format normalization
# ---------------------------------------------------------------------------

_V3_FG_TYPES = {"2pt", "3pt"}
_V3_ACTION_MAP = {
    "rebound": "rebound",
    "turnover": "turnover",
    "foul": "foul",
    "freethrow": "free_throw",
    "substitution": "substitution",
    "jumpball": "jump_ball",
    "timeout": "timeout",
    "violation": "violation",
    "ejection": "ejection",
    "period": "period_marker",
    "game": "other",
    "block": "block_event",
    "steal": "steal_event",
}


def from_v3(raw: dict, idx: int) -> Event:
    """Create an Event from the v3 PBP format (2022+)."""
    action_type_raw = raw.get("actionType", "")
    sub_type = raw.get("subType", "")
    descriptor = raw.get("descriptor", "") or ""
    desc = raw.get("description", "") or ""
    person_id = _safe_int(raw.get("personId"))
    team_id = _safe_int(raw.get("teamId"))
    period = _safe_int(raw.get("period"))
    clock_seconds = parse_clock(raw.get("clock", ""))
    qualifiers = raw.get("qualifiers", []) or []
    possession_tid = _safe_int(raw.get("possession"))
    loc_x = _safe_int(raw.get("xLegacy"), default=None)
    loc_y = _safe_int(raw.get("yLegacy"), default=None)

    evt = Event(
        idx=idx,
        action_type="other",
        player_id=person_id,
        team_id=team_id,
        period=period,
        clock_seconds=clock_seconds,
        description=desc,
        loc_x=loc_x,
        loc_y=loc_y,
        possession_team_id=possession_tid,
        qualifiers=qualifiers,
        descriptor=descriptor,
        action_number=_safe_int(raw.get("actionNumber")),
        order_number=_safe_int(raw.get("orderNumber")),
    )

    if action_type_raw == "jumpball":
        evt.jump_ball_player_ids = [
            pid
            for pid in (
                _safe_int(raw.get("jumpBallWonPersonId")),
                _safe_int(raw.get("jumpBallLostPersonId")),
            )
            if pid
        ]

    # --- Field goal ---
    if action_type_raw in _V3_FG_TYPES:
        evt.action_type = "field_goal"
        evt.is_field_goal = True
        evt.is_3pt = action_type_raw == "3pt"
        evt.shot_value = 3 if evt.is_3pt else 2
        shot_result = raw.get("shotResult", "")
        evt.is_made = shot_result == "Made"

        # Distance
        evt.shot_distance = _safe_float(raw.get("shotDistance"))
        if evt.shot_distance is None:
            evt.shot_distance = _compute_distance(loc_x, loc_y)

        # Zone: prefer v3 area field
        area = raw.get("area", "")
        if area:
            evt.shot_zone = _classify_zone_v3(area, evt.is_3pt)
        if not evt.shot_zone:
            evt.shot_zone = _classify_zone_by_coords(
                evt.shot_distance, loc_x, loc_y, evt.is_3pt
            )

        # Assist
        evt.assist_player_id = _safe_int(raw.get("assistPersonId"))

        # Block
        evt.block_player_id = _safe_int(raw.get("blockPersonId"))

    # --- Block event (v3 has explicit block events) ---
    elif action_type_raw == "block":
        evt.action_type = "block_event"
        evt.block_player_id = person_id

    # --- Steal event (v3 has explicit steal events) ---
    elif action_type_raw == "steal":
        evt.action_type = "steal_event"
        evt.steal_player_id = person_id

    # --- Rebound ---
    elif action_type_raw == "rebound":
        evt.action_type = "rebound"
        evt.is_rebound = True
        sub_lower = sub_type.lower() if sub_type else ""
        evt.reb_is_offensive = sub_lower == "offensive"
        evt.reb_shot_action_number = _safe_int(
            raw.get("shotActionNumber"), default=None
        )

    # --- Turnover ---
    elif action_type_raw == "turnover":
        evt.action_type = "turnover"
        evt.is_turnover = True
        evt.to_subtype = sub_type.lower().strip() if sub_type else ""
        evt.steal_player_id = _safe_int(raw.get("stealPersonId"))

    # --- Foul ---
    elif action_type_raw == "foul":
        evt.action_type = "foul"
        evt.is_foul = True
        from .constants import V3_FOUL_MAP

        sub_lower = sub_type.lower().strip() if sub_type else ""
        desc_lower = descriptor.lower().strip()
        key = (sub_lower, desc_lower)
        evt.foul_type = V3_FOUL_MAP.get(key, "")
        if not evt.foul_type:
            # Try with just subType
            evt.foul_type = V3_FOUL_MAP.get((sub_lower, ""), "personal_fouls")
        evt.is_technical = sub_lower == "technical"
        evt.foul_drawn_player_id = _safe_int(raw.get("foulDrawnPersonId"))

    # --- Free throw ---
    elif action_type_raw == "freethrow":
        evt.action_type = "free_throw"
        evt.is_free_throw = True
        shot_result = raw.get("shotResult", "")
        evt.is_ft_made = shot_result == "Made"
        evt.is_technical_ft = "technical" in (sub_type or "").lower()

    # --- Substitution ---
    elif action_type_raw == "substitution":
        evt.action_type = "substitution"
        evt.is_substitution = True
        sub_lower = sub_type.lower() if sub_type else ""
        if sub_lower == "out":
            evt.sub_out_player_id = person_id
            evt.sub_direction = "out"
        elif sub_lower == "in":
            evt.sub_in_player_id = person_id
            evt.sub_direction = "in"

    # --- Period markers ---
    elif action_type_raw == "period":
        sub_lower = sub_type.lower() if sub_type else ""
        if sub_lower == "start":
            evt.action_type = "period_start"
            evt.is_period_start = True
        elif sub_lower == "end":
            evt.action_type = "period_end"
            evt.is_period_end = True
        else:
            evt.action_type = "period_marker"
    elif action_type_raw == "game":
        sub_lower = sub_type.lower() if sub_type else ""
        if sub_lower == "end":
            evt.action_type = "period_end"
            evt.is_period_end = True
        else:
            evt.action_type = "other"

    # --- Other mapped types ---
    elif action_type_raw in _V3_ACTION_MAP:
        evt.action_type = _V3_ACTION_MAP[action_type_raw]

    return evt


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------

# Only types UNIQUE to each format (not shared like "period", "ejection")
V3_UNIQUE_TYPES = {
    "2pt",
    "3pt",
    "freethrow",
    "jumpball",
    "block",
    "steal",
    "game",
}

OLDER_UNIQUE_TYPES = {
    "Made Shot",
    "Missed Shot",
    "Rebound",
    "Turnover",
    "Foul",
    "Free Throw",
    "Substitution",
    "Jump Ball",
    "Instant Replay",
}


def detect_format(events_raw: list[dict]) -> str:
    """Detect PBP format: 'v3' or 'older'. Sample first 20 events."""
    for raw in events_raw[:20]:
        at = raw.get("actionType", "")
        if at in V3_UNIQUE_TYPES:
            return "v3"
        if at in OLDER_UNIQUE_TYPES:
            return "older"
    # Fallback: check for v3-specific fields
    for raw in events_raw[:20]:
        if "timeActual" in raw or "qualifiers" in raw or "orderNumber" in raw:
            return "v3"
    return "older"


_REB_COUNTS_RE = re.compile(r"\(Off:(\d+) Def:(\d+)\)")


def _link_older_events(events: list[Event]) -> None:
    """Recover from the stats-format feed what the live feed states directly:
    block/steal rows share the shot's/turnover's actionNumber, rebound type is
    only given as running per-player counts "(Off:N Def:M)", and there is no
    fouled-player field (taken from the free throws that follow the foul)."""
    by_number = {e.action_number: e for e in events if e.is_field_goal or e.is_turnover}
    for evt in events:
        target = by_number.get(evt.action_number)
        if evt.action_type == "block_event" and target and target.is_field_goal:
            target.block_player_id = evt.block_player_id
        elif evt.action_type == "steal_event" and target and target.is_turnover:
            target.steal_player_id = evt.steal_player_id

    reb_counts: dict[int, tuple[int, int]] = {}
    last_miss_team = 0
    for i, evt in enumerate(events):
        if (evt.is_field_goal and not evt.is_made) or (
            evt.is_free_throw and not evt.is_ft_made
        ):
            last_miss_team = evt.team_id
        elif evt.is_rebound:
            m = _REB_COUNTS_RE.search(evt.description)
            if m and evt.player_id:
                counts = (int(m.group(1)), int(m.group(2)))
                evt.reb_is_offensive = counts[0] > reb_counts.get(evt.player_id, (0, 0))[0]
                reb_counts[evt.player_id] = counts
            else:
                evt.reb_is_offensive = evt.team_id == last_miss_team
        elif evt.is_foul and not evt.is_technical and not evt.foul_drawn_player_id:
            for nxt in events[i + 1 : i + 8]:
                if (nxt.period, nxt.clock_seconds) != (evt.period, evt.clock_seconds):
                    break
                if nxt.is_free_throw and not nxt.is_technical_ft and nxt.team_id != evt.team_id:
                    evt.foul_drawn_player_id = nxt.player_id
                    break


def parse_events(events_raw: list[dict]) -> list[Event]:
    """Parse a list of raw PBP JSON dicts into normalized Event objects."""
    fmt = detect_format(events_raw)
    if fmt == "v3":
        return [from_v3(raw, idx) for idx, raw in enumerate(events_raw)]
    events = [from_older(raw, idx) for idx, raw in enumerate(events_raw)]
    _link_older_events(events)
    return events
