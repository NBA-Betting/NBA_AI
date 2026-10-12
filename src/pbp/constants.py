"""
Constants for PBP stat computation, matching pbpstats package values.
"""

# ---------------------------------------------------------------------------
# Shot zone boundaries (feet)
# ---------------------------------------------------------------------------
AT_RIM_CUTOFF = 4  # < 4 ft = at rim
SHORT_MID_CUTOFF = 14  # 4-14 ft = short mid-range
# 14+ ft non-3PT = long mid-range
CORNER_3_Y_CUTOFF = 90  # |locY| <= 90 = corner 3 (legacy coords)
# Legacy coordinates (0.1 ft, hoop at origin): restricted area and paint bounds,
# fitted to the live feed's area labels (97.6% of 2PT shots agree)
RESTRICTED_AREA_RADIUS = 44
PAINT_HALF_WIDTH = 78
PAINT_DEPTH = 137
HEAVE_DISTANCE = 40  # > 40 ft = heave
HEAVE_TIME = 2.0  # < 2 seconds remaining = heave

SHOT_ZONES = ("at_rim", "short_mid", "long_mid", "corner3", "arc3")

# ---------------------------------------------------------------------------
# V3 area field → zone mapping
# ---------------------------------------------------------------------------
V3_AREA_TO_ZONE = {
    "Restricted Area": "at_rim",
    "In The Paint (Non-RA)": "short_mid",
    "Mid-Range": "long_mid",
    "Left Corner 3": "corner3",
    "Right Corner 3": "corner3",
    "Above the Break 3": "arc3",
    "Backcourt": "arc3",  # heaves from backcourt
}

# ---------------------------------------------------------------------------
# Turnover subtypes → live vs dead ball
# ---------------------------------------------------------------------------
LIVE_BALL_TO_SUBTYPES = {
    "bad pass",
    "badpass",
    "bad_pass",
    "lost ball",
    "lostball",
    "lost_ball",
    "stolen",
}

DEAD_BALL_TO_SUBTYPES = {
    "traveling",
    "travel",
    "3 second violation",
    "3secviolation",
    "3sec",
    "backcourt",
    "backcourt turnover",
    "double dribble",
    "discontinued dribble",
    "5 second violation",
    "5sec inbound",
    "8 second violation",
    "8sec",
    "shot clock",
    "shotclock",
    "shot clock violation",
    "step out of bounds",
    "outofbounds",
    "palming",
    "offensive goaltending",
    "offensivegoaltending",
    "lane violation",
    "laneviolation",
    "kicked ball",
    "kickedball",
    "swinging elbows",
    "basket from below",
    "illegal screen",
    "offensive foul",
    "punched ball",
    "out of bounds lost ball turnover",
    "jump ball violation",
}

# Specific dead-ball subtypes we track individually
TRAVEL_SUBTYPES = {"traveling", "travel"}
THREE_SEC_SUBTYPES = {"3 second violation", "3secviolation", "3sec"}
SHOT_CLOCK_SUBTYPES = {"shot clock", "shotclock", "shot clock violation"}
STEP_OOB_SUBTYPES = {"step out of bounds"}

# ---------------------------------------------------------------------------
# Foul type mapping — v3 format (subType + descriptor → our column name)
# ---------------------------------------------------------------------------
V3_FOUL_MAP = {
    # (subType, descriptor) → column_name
    ("personal", ""): "personal_fouls",
    ("personal", "shooting"): "shooting_fouls",
    ("personal", "loose ball"): "loose_ball_fouls",
    ("personal", "looseball"): "loose_ball_fouls",
    ("personal", "block"): "personal_fouls",  # personal block foul
    ("personal", "take"): "personal_fouls",  # personal take foul
    ("personal", "transition"): "transition_take_fouls",
    ("personal", "away from play"): "away_from_play_fouls",
    ("personal", "awayfromplay"): "away_from_play_fouls",
    ("personal", "clear path"): "clear_path_fouls",
    ("personal", "clearpath"): "clear_path_fouls",
    ("personal", "flagrant type 1"): "flagrant1_fouls",
    ("personal", "flagranttype1"): "flagrant1_fouls",
    ("personal", "flagrant type 2"): "flagrant2_fouls",
    ("personal", "flagranttype2"): "flagrant2_fouls",
    ("personal", "inbound"): "away_from_play_fouls",
    ("offensive", ""): "offensive_fouls",
    ("offensive", "charge"): "charge_fouls",
    ("offensive", "offensive"): "offensive_fouls",
    ("technical", ""): "technical_fouls",
    ("technical", "defensive 3 second"): "technical_fouls",
    ("technical", "defensive3second"): "technical_fouls",
    ("technical", "double"): "technical_fouls",
    ("technical", "delay"): "technical_fouls",
    ("technical", "taunting"): "technical_fouls",
    ("loose ball", ""): "loose_ball_fouls",
}

# Foul type mapping — older format (subType → our column name)
OLDER_FOUL_MAP = {
    "Personal": "personal_fouls",
    "Shooting": "shooting_fouls",
    "Loose Ball": "loose_ball_fouls",
    "Offensive": "offensive_fouls",
    "Offensive Charge": "charge_fouls",
    "Away From Play": "away_from_play_fouls",
    "Clear Path": "clear_path_fouls",
    "Flagrant Type 1": "flagrant1_fouls",
    "Flagrant Type 1 ": "flagrant1_fouls",
    "Flagrant Type 2": "flagrant2_fouls",
    "Technical": "technical_fouls",
    "Defense 3 Second": "technical_fouls",
    "Delay": "technical_fouls",
    "Taunting": "technical_fouls",
    "Personal Block": "personal_fouls",
    "Personal Take": "personal_fouls",
    "Shooting Block": "shooting_fouls",
    "Transition Take": "transition_take_fouls",
    "Double Personal": "personal_fouls",
    "Inbound": "away_from_play_fouls",
}

# Fouls that have a "drawn" counterpart we track
FOUL_DRAWN_COLUMNS = {
    "personal_fouls": "personal_fouls_drawn",
    "shooting_fouls": "shooting_fouls_drawn",
    "charge_fouls": "charge_fouls_drawn",
}

# ---------------------------------------------------------------------------
# Period durations (seconds)
# ---------------------------------------------------------------------------
REGULAR_PERIOD_SECONDS = 720.0  # 12 minutes
OVERTIME_PERIOD_SECONDS = 300.0  # 5 minutes

# ---------------------------------------------------------------------------
# All stat columns for the DB table
# ---------------------------------------------------------------------------
STAT_COLUMNS = [
    # Seconds and possessions
    "seconds_played_off",
    "seconds_played_def",
    "off_poss",
    "def_poss",
    # FG by zone
    "at_rim_fgm",
    "at_rim_fga",
    "short_mid_fgm",
    "short_mid_fga",
    "long_mid_fgm",
    "long_mid_fga",
    "corner3_fgm",
    "corner3_fga",
    "arc3_fgm",
    "arc3_fga",
    # Assisted/unassisted
    "assisted_at_rim",
    "assisted_short_mid",
    "assisted_long_mid",
    "assisted_corner3",
    "assisted_arc3",
    "unassisted_at_rim",
    "unassisted_short_mid",
    "unassisted_long_mid",
    "unassisted_corner3",
    "unassisted_arc3",
    # Assists given by zone
    "ast_at_rim",
    "ast_short_mid",
    "ast_long_mid",
    "ast_corner3",
    "ast_arc3",
    # Blocked (shooter was blocked) by zone
    "blocked_at_rim",
    "blocked_short_mid",
    "blocked_long_mid",
    "blocked_corner3",
    "blocked_arc3",
    # Blocks made by zone
    "block_at_rim",
    "block_short_mid",
    "block_long_mid",
    "block_corner3",
    "block_arc3",
    # Missed unblocked by zone
    "missed_at_rim",
    "missed_short_mid",
    "missed_long_mid",
    "missed_corner3",
    "missed_arc3",
    # Shot distance tracking
    "total_2pt_shot_distance",
    "total_2pt_shots_with_distance",
    "total_3pt_shot_distance",
    "total_3pt_shots_with_distance",
    # Special shots
    "putbacks",
    "heave_makes",
    "heave_misses",
    # Plus/minus
    "plus_minus",
    # Rebounds by zone
    "oreb_at_rim",
    "oreb_short_mid",
    "oreb_long_mid",
    "oreb_corner3",
    "oreb_arc3",
    "dreb_at_rim",
    "dreb_short_mid",
    "dreb_long_mid",
    "dreb_corner3",
    "dreb_arc3",
    "oreb_ft",
    "dreb_ft",
    "self_oreb",
    "on_floor_oreb",
    "oreb_opportunities",
    "dreb_opportunities",
    # Free throws
    "fts_made",
    "fts_missed",
    "tech_fts_made",
    # Turnovers
    "bad_pass_turnovers",
    "lost_ball_turnovers",
    "bad_pass_steals",
    "lost_ball_steals",
    "deadball_turnovers",
    "travels",
    "three_sec_violations",
    "shot_clock_violations",
    "step_out_of_bounds",
    "offensive_fouls_to",
    # Fouls committed
    "personal_fouls",
    "shooting_fouls",
    "loose_ball_fouls",
    "offensive_fouls",
    "charge_fouls",
    "away_from_play_fouls",
    "clear_path_fouls",
    "flagrant1_fouls",
    "flagrant2_fouls",
    "technical_fouls",
    "transition_take_fouls",
    # Fouls drawn
    "personal_fouls_drawn",
    "shooting_fouls_drawn",
    "charge_fouls_drawn",
]

# Total column count: 4 + 10 + 10 + 5 + 5 + 5 + 5 + 4 + 3 + 1 + 12 + 4 + 3 + 10 + 11 + 3 = 95
