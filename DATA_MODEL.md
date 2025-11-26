# NBA AI Data Model Reference
**Last Updated**: November 24, 2025  
**Purpose**: Comprehensive reference for all data structures, schemas, and API endpoints used in the project

---

## Table of Contents
1. [Database Schema](#database-schema)
2. [External API Endpoints (NBA)](#external-api-endpoints-nba)
3. [Internal API Endpoints (Games API)](#internal-api-endpoints-games-api)
4. [Data Flow Pipeline](#data-flow-pipeline)
5. [JSON Data Structures](#json-data-structures)

---

## Database Schema

### Overview
- **Database**: SQLite
- **Active DB**: `data/NBA_AI_2023_2025.sqlite` (2.1GB working database)
- **Master Archive**: `data/NBA_AI_ALL_SEASONS.sqlite` (24GB, read-only)
- **Key Design**: TEXT-based (game_id, team names) for simplicity

### Tables (9 total)

#### 1. Games (Master Schedule Table)
**Purpose**: Central table tracking all NBA games and their collection status

```sql
CREATE TABLE IF NOT EXISTS "Games" (
    game_id TEXT PRIMARY KEY,              -- Format: 00223XXXXX (season/type/game#)
    date_time_est TEXT NOT NULL,           -- ISO 8601: "2024-10-22T19:30:00Z"
    home_team TEXT NOT NULL,               -- 3-letter abbreviation: "BOS", "LAL"
    away_team TEXT NOT NULL,               -- 3-letter abbreviation: "NYK", "MIA"
    status TEXT NOT NULL,                  -- "Scheduled", "In Progress", "Completed", "Final"
    season TEXT NOT NULL,                  -- "2023-2024", "2024-2025"
    season_type TEXT NOT NULL,             -- "Regular Season", "Post Season", "Pre Season", "All-Star"
    pre_game_data_finalized BOOLEAN NOT NULL DEFAULT 0,  -- Features/predictions ready
    game_data_finalized BOOLEAN NOT NULL DEFAULT 0       -- PBP/GameStates/Boxscores complete
);
```

**Key Fields**:
- `game_id`: Encodes season (chars 2-5) and game type (char 1)
  - `002` = Regular Season, `004` = Playoffs, `001` = Pre-Season, `003` = All-Star
- `game_data_finalized`: Set to 1 when PbP_Logs, GameStates, PlayerBox, TeamBox all collected
- `pre_game_data_finalized`: Set to 1 when Features and Predictions created

**Status Values**: 
- "Scheduled" → "In Progress" → "Completed"/"Final"

---

#### 2. PbP_Logs (Raw Play-by-Play Data)
**Purpose**: Stores raw JSON play-by-play data from NBA API

```sql
CREATE TABLE IF NOT EXISTS "PbP_Logs" (
    game_id TEXT NOT NULL,
    play_id INTEGER NOT NULL,              -- Action number/order from NBA API
    log_data TEXT,                         -- Raw JSON from NBA API
    PRIMARY KEY (game_id, play_id)
);
```

**Data Volume**: ~492 plays per game average  
**Source**: NBA CDN (live) or stats.nba.com (stats) endpoints

**log_data JSON Structure** (key fields):
```json
{
    "actionNumber": 1,
    "period": 1,
    "clock": "PT11M59.00S",           // ISO 8601 duration
    "scoreHome": "0",
    "scoreAway": "0",
    "actionType": "jumpball",
    "subType": "",
    "description": "Jump Ball...",
    "personId": 1630162,
    "playerName": "...",
    "teamTricode": "BOS"
}
```

---

#### 3. GameStates (Parsed Game Snapshots)
**Purpose**: Structured game state at each play (parsed from PbP_Logs)

```sql
CREATE TABLE IF NOT EXISTS "GameStates" (
    game_id TEXT NOT NULL,
    play_id INTEGER NOT NULL,
    game_date TEXT,                        -- "2024-10-22"
    home TEXT,                             -- Home team tricode
    away TEXT,                             -- Away team tricode
    clock TEXT,                            -- "PT11M59.00S"
    period INTEGER,                        -- 1-4 (reg), 5+ (OT)
    home_score INTEGER,
    away_score INTEGER,
    total INTEGER,                         -- home_score + away_score
    home_margin INTEGER,                   -- home_score - away_score
    is_final_state BOOLEAN,                -- 1 if final play of game
    players_data TEXT,                     -- JSON: player stats at this moment
    PRIMARY KEY (game_id, play_id)
);
```

**Data Volume**: ~492 states per game (one per play)  
**Generated**: Parsed from PbP_Logs by `game_states.py`

**players_data JSON Structure**:
```json
{
    "BOS": {
        "12345": {  // player_id
            "name": "Jayson Tatum",
            "pts": 15,
            "reb": 5,
            "ast": 3,
            ...
        }
    },
    "NYK": { ... }
}
```

---

#### 4. PlayerBox (Player Boxscore Stats)
**Purpose**: Traditional boxscore statistics for each player per game

```sql
CREATE TABLE PlayerBox (
    player_id INTEGER NOT NULL,
    game_id TEXT NOT NULL,
    team_id TEXT NOT NULL,                 -- Team tricode: "BOS", "LAL"
    player_name TEXT,
    position TEXT,                         -- "F", "G", "C", "G-F"
    min REAL,                              -- Minutes played (float)
    pts INTEGER,                           -- Points
    reb INTEGER,                           -- Total rebounds
    ast INTEGER,                           -- Assists
    stl INTEGER,                           -- Steals
    blk INTEGER,                           -- Blocks
    tov INTEGER,                           -- Turnovers
    pf INTEGER,                            -- Personal fouls
    oreb INTEGER,                          -- Offensive rebounds
    dreb INTEGER,                          -- Defensive rebounds
    fga INTEGER,                           -- Field goals attempted
    fgm INTEGER,                           -- Field goals made
    fg_pct REAL,                           -- Field goal percentage
    fg3a INTEGER,                          -- 3-pointers attempted
    fg3m INTEGER,                          -- 3-pointers made
    fg3_pct REAL,                          -- 3-point percentage
    fta INTEGER,                           -- Free throws attempted
    ftm INTEGER,                           -- Free throws made
    ft_pct REAL,                           -- Free throw percentage
    plus_minus INTEGER,                    -- Plus/minus
    PRIMARY KEY (player_id, game_id),
    FOREIGN KEY (game_id) REFERENCES Games(game_id)
);
```

**Data Volume**: ~26 players per game  
**Source**: BoxScoreTraditionalV3 from nba_api

---

#### 5. TeamBox (Team Boxscore Stats)
**Purpose**: Team-level aggregate statistics per game

```sql
CREATE TABLE TeamBox (
    team_id TEXT NOT NULL,                 -- Team tricode
    game_id TEXT NOT NULL,
    pts INTEGER,
    pts_allowed INTEGER,                   -- Opponent's points
    reb INTEGER,
    ast INTEGER,
    stl INTEGER,
    blk INTEGER,
    tov INTEGER,
    pf INTEGER,
    fga INTEGER,
    fgm INTEGER,
    fg_pct REAL,
    fg3a INTEGER,
    fg3m INTEGER,
    fg3_pct REAL,
    fta INTEGER,
    ftm INTEGER,
    ft_pct REAL,
    plus_minus INTEGER,
    PRIMARY KEY (team_id, game_id),
    FOREIGN KEY (game_id) REFERENCES Games(game_id)
);
```

**Data Volume**: 2 records per game (home + away)  
**Source**: BoxScoreTraditionalV3 from nba_api

---

#### 6. Features (ML Feature Sets)
**Purpose**: Engineered features for machine learning models

```sql
CREATE TABLE IF NOT EXISTS "Features" (
    game_id TEXT PRIMARY KEY,
    feature_set TEXT,                      -- JSON: all features for this game
    save_datetime TEXT                     -- When features were created
);
```

**feature_set JSON Structure** (34 features total):
```json
{
    "home_win_pct_10": 0.7,
    "away_win_pct_10": 0.6,
    "home_ppg_10": 112.5,
    "away_ppg_10": 108.3,
    "home_opp_ppg_10": 105.2,
    "away_opp_ppg_10": 110.1,
    "home_fg_pct_10": 0.467,
    "away_fg_pct_10": 0.452,
    ...  // 17 features x 2 (home/away) = 34 total
}
```

**Generated**: `features.py` using rolling averages from prior final GameStates  
**Dependencies**: Requires prior game data for both teams

---

#### 7. Predictions (Model Predictions)
**Purpose**: Store predictions from various prediction engines

```sql
CREATE TABLE IF NOT EXISTS "Predictions" (
    game_id TEXT NOT NULL,
    predictor TEXT NOT NULL,               -- "Baseline", "Linear", "Tree", "MLP"
    prediction_datetime TEXT NOT NULL,     -- When prediction was made
    prediction_set TEXT NOT NULL,          -- JSON: all prediction outputs
    PRIMARY KEY (game_id, predictor)
);
```

**prediction_set JSON Structure**:
```json
{
    "home_score": 112.5,
    "away_score": 108.3,
    "home_margin": 4.2,
    "home_win_prob": 0.62,
    "prediction_type": "pre_game",         // or "live_updated"
    "confidence": 0.75                     // Model-specific
}
```

**Predictors**:
- `Baseline`: Simple PPG-based formula
- `Linear`: Ridge Regression on features
- `Tree`: XGBoost on features
- `MLP`: PyTorch neural network
- `Random`: Random predictor for testing

---

#### 8. Players (Reference Table)
**Purpose**: Master list of all NBA players

```sql
CREATE TABLE Players (
    person_id INTEGER PRIMARY KEY,
    first_name TEXT,
    last_name TEXT,
    full_name TEXT,
    from_year INTEGER,                     -- First season
    to_year INTEGER,                       -- Last season
    roster_status BOOLEAN,                 -- Currently active?
    team TEXT                              -- Current team tricode
);
```

**Data Volume**: 5,114 players  
**Source**: NBA Stats API commonallplayers endpoint  
**Updated**: When schedule is updated for a season

---

#### 9. Teams (Reference Table)
**Purpose**: Master list of all NBA teams with name variations

```sql
CREATE TABLE IF NOT EXISTS "Teams" (
    team_id TEXT PRIMARY KEY,              -- NBA team ID: "1610612738"
    abbreviation TEXT NOT NULL,            -- "BOS"
    abbreviation_normalized TEXT NOT NULL, -- "bos"
    full_name TEXT NOT NULL,               -- "Boston Celtics"
    full_name_normalized TEXT NOT NULL,    -- "boston celtics"
    short_name TEXT NOT NULL,              -- "Celtics"
    short_name_normalized TEXT NOT NULL,   -- "celtics"
    alternatives TEXT,                     -- JSON: ["BOS", "Celts", ...]
    alternatives_normalized TEXT           -- JSON: ["bos", "celts", ...]
);
```

**Data Volume**: 30 teams  
**Purpose**: Handle name variations in text matching

---

## External API Endpoints (NBA)

### 1. Schedule Endpoint
**URL**: `https://stats.nba.com/stats/scheduleleaguev2?Season={season}&LeagueID=00`  
**Module**: `src/database_updater/schedule.py`  
**Purpose**: Fetch all games for a season

**Request**:
- Method: GET
- Season format: "2023-24" (abbreviated)
- Headers: See `config.yaml` → `nba_api.schedule_headers`

**Response Structure**:
```json
{
    "leagueSchedule": {
        "gameDates": [
            {
                "gameDate": "2024-10-22",
                "games": [
                    {
                        "gameId": "0022400061",
                        "gameStatus": 3,                    // 1=scheduled, 2=in progress, 3=final
                        "gameDateTimeEst": "2024-10-22T19:30:00Z",
                        "homeTeam": {"teamTricode": "BOS"},
                        "awayTeam": {"teamTricode": "NYK"}
                    }
                ]
            }
        ]
    }
}
```

**Rate Limiting**: None observed, but uses retry logic  
**Saved To**: `Games` table

---

### 2. Play-by-Play Endpoints (Dual Source)

#### Primary: NBA CDN (Live Endpoint)
**URL**: `https://cdn.nba.com/static/json/liveData/playbyplay/playbyplay_{game_id}.json`  
**Module**: `src/database_updater/pbp.py`  
**Purpose**: Fetch real-time play-by-play data

**Advantages**: 
- Fast, reliable CDN
- Real-time updates during games
- More detailed player tracking

**Disadvantages**:
- Only available for recent games
- May not have historical data

**Response Structure**:
```json
{
    "game": {
        "gameId": "0022400061",
        "actions": [
            {
                "actionNumber": 1,
                "orderNumber": 1,
                "period": 1,
                "clock": "PT11M59.00S",
                "scoreHome": "0",
                "scoreAway": "0",
                "actionType": "jumpball",
                "subType": "recovered",
                "description": "Jump Ball Adams vs. Tatum...",
                "personId": 1630162,
                "playerName": "J. Adams",
                "teamTricode": "BOS"
            }
        ]
    }
}
```

#### Fallback: NBA Stats (Historical Endpoint)
**URL**: `https://stats.nba.com/stats/playbyplayv3?GameID={game_id}&StartPeriod=0&EndPeriod=0`  
**Module**: `src/database_updater/pbp.py`  
**Purpose**: Fetch historical play-by-play data

**Advantages**:
- Available for all historical games (back to 2000-2001)
- Official NBA Stats API

**Disadvantages**:
- Slower
- Different JSON structure (uses `actionId` instead of `orderNumber`)

**Response Structure**: Similar to CDN but with `actionId` field

**Saved To**: `PbP_Logs` table (raw JSON in `log_data` column)

---

### 3. BoxScore Endpoint (nba_api Library)
**Endpoint Class**: `BoxScoreTraditionalV3` from nba_api  
**Module**: `src/database_updater/boxscores.py`  
**Purpose**: Fetch player and team boxscore statistics

**Parameters**:
- `game_id`: Required (TEXT format game ID)
- `end_period`: 0 (default, means all periods)
- `start_period`: 0 (default)
- `timeout`: 30 seconds

**Response Structure**:
```json
{
    "boxScoreTraditional": {
        "homeTeam": {
            "teamId": 1610612738,
            "teamTricode": "BOS",
            "statistics": {
                "points": 112,
                "reboundsTotal": 45,
                "assists": 28,
                "fieldGoalsMade": 42,
                "fieldGoalsAttempted": 88,
                "fieldGoalsPercentage": 0.477,
                ...
            },
            "players": [
                {
                    "personId": 1627759,
                    "firstName": "Jayson",
                    "familyName": "Tatum",
                    "position": "F",
                    "statistics": {
                        "minutes": "36:24",
                        "points": 28,
                        "reboundsTotal": 9,
                        ...
                    }
                }
            ]
        },
        "awayTeam": { ... }
    }
}
```

**Rate Limiting**: 0.6 second sleep between requests  
**Saved To**: `PlayerBox` and `TeamBox` tables

---

### 4. Players Endpoint
**URL**: `https://stats.nba.com/stats/commonallplayers?LeagueID=00&Season={season}`  
**Module**: `src/database_updater/players.py`  
**Purpose**: Fetch all players for a season

**Saved To**: `Players` table

---

## Internal API Endpoints (Games API)

### Overview
**Module**: `src/games_api/api.py`  
**Framework**: Flask  
**Base URL**: `http://127.0.0.1:5000` (development)  
**Purpose**: Serve game data to web app with live updates

### Endpoints

#### 1. GET /api/games
**Purpose**: Fetch games by date or game_ids with predictions

**Query Parameters**:
- `date` (optional): ISO date "YYYY-MM-DD"
- `game_ids` (optional): Comma-separated game IDs
- `predictor` (optional): Predictor name (default from config)
- `update_predictions` (optional): "true" or "false" (default: true)

**Example Requests**:
```bash
# Get games for a specific date
GET /api/games?date=2024-10-22&predictor=Baseline

# Get specific games by ID
GET /api/games?game_ids=0022400061,0022400062

# Skip prediction updates (faster)
GET /api/games?date=2024-10-22&update_predictions=false
```

**Response Structure**:
```json
{
    "0022400061": {
        "date_time_est": "2024-10-22T19:30:00Z",
        "home_team": "BOS",
        "away_team": "NYK",
        "status": "Final",
        "season": "2024-2025",
        "season_type": "Regular Season",
        "pre_game_data_finalized": true,
        "game_data_finalized": true,
        "play_by_play": [ ... ],           // Up to 500 most recent plays
        "game_states": [                   // Latest game state only
            {
                "play_id": 492,
                "period": 4,
                "clock": "PT00M00.00S",
                "home_score": 112,
                "away_score": 108,
                "is_final_state": true,
                "players_data": { ... }
            }
        ],
        "predictions": {
            "pre_game": {
                "prediction_datetime": "2024-10-22T18:00:00",
                "prediction_set": {
                    "home_score": 110.5,
                    "away_score": 107.2,
                    "home_win_prob": 0.58
                }
            },
            "current": {                   // Only if game in progress or final
                "home_score": 112.3,       // Blended with actual score
                "away_score": 108.1,
                "home_win_prob": 0.95
            }
        }
    }
}
```

**Processing**:
1. Query `Games` table for matching games
2. Join with `PbP_Logs`, `GameStates`, `Predictions`
3. If `update_predictions=true` and game in progress/final:
   - Call `make_current_predictions()` to blend pre-game with live data
4. Return formatted JSON

**Rate Limits**: Max 20 game_ids per request (configurable in `config.yaml`)

---

## Data Flow Pipeline

### Full Pipeline (database_update_manager.py)

```
Stage 1: Schedule Update
├─ Fetch: scheduleleaguev2 API
├─ Parse: game_id, teams, date, status
└─ Save: Games table

Stage 2: Players Update
├─ Fetch: commonallplayers API
├─ Parse: player names, IDs, team
└─ Save: Players table

Stage 3: Game Data Collection (for games with game_data_finalized=0)
├─ 3a: PbP Collection
│   ├─ Fetch: playbyplay CDN or stats API
│   ├─ Parse: ~492 plays per game
│   └─ Save: PbP_Logs table
├─ 3b: GameStates Parsing
│   ├─ Read: PbP_Logs
│   ├─ Parse: game states at each play
│   └─ Save: GameStates table
└─ 3c: Boxscores Collection
    ├─ Fetch: BoxScoreTraditionalV3 via nba_api
    ├─ Parse: player stats (~26 per game), team stats (2 per game)
    ├─ Save: PlayerBox and TeamBox tables
    └─ Update: Games.game_data_finalized = 1

Stage 4: Pre-Game Data Preparation (for games with pre_game_data_finalized=0)
├─ 4a: Determine Prior States
│   ├─ Read: Games table (find prior games for each team)
│   └─ Read: GameStates (final states from prior games)
├─ 4b: Create Features
│   ├─ Compute: Rolling averages (win%, PPG, FG%, etc.) over last N games
│   ├─ Aggregate: 34 features (17 home + 17 away)
│   └─ Save: Features table
└─ Update: Games.pre_game_data_finalized = 1

Stage 5: Predictions
├─ Read: Features table
├─ Load: ML models (Ridge, XGBoost, MLP)
├─ Predict: home_score, away_score, win_prob
├─ Save: Predictions table
└─ Update: prediction_datetime timestamp
```

### Live Prediction Updates (games_api/games.py)

```
When game in progress or completed:
├─ Read: Predictions (pre-game prediction)
├─ Read: GameStates (current score, clock, period)
├─ Calculate: time_remaining_factor
├─ Blend: pre_game_score * time_factor + current_score * (1 - time_factor)
└─ Return: current prediction with updated win_prob
```

---

## JSON Data Structures

### 1. PbP_Logs.log_data
**Source**: NBA CDN or Stats API  
**Size**: ~10-50 KB per play

```json
{
    "actionNumber": 145,
    "period": 2,
    "clock": "PT05M23.00S",
    "scoreHome": "52",
    "scoreAway": "48",
    "actionType": "2pt",
    "subType": "layup",
    "qualifiers": ["fastbreak"],
    "description": "Tatum 2' Driving Layup (15 PTS) (Brown 3 AST)",
    "personId": 1627759,
    "playerName": "J. Tatum",
    "playerNameI": "Tatum, J.",
    "teamTricode": "BOS",
    "teamId": 1610612738,
    "descriptor": "made",
    "shotDistance": 2,
    "shotResult": "Made",
    "pointsTotal": 15,                    // Player's total points so far
    "assistTotal": 0,                     // Player's total assists so far
    "reboundTotal": 5,                    // etc.
    "x": 5,                               // Court coordinates
    "y": 25
}
```

### 2. GameStates.players_data
**Source**: Parsed from PbP_Logs  
**Size**: ~50-100 KB per game state

```json
{
    "BOS": {
        "1627759": {
            "name": "Jayson Tatum",
            "status": "on_court",
            "pts": 15,
            "reb": 5,
            "ast": 2,
            "stl": 1,
            "blk": 0,
            "tov": 1,
            "pf": 2,
            "fgm": 6,
            "fga": 12,
            "fg3m": 1,
            "fg3a": 4,
            "ftm": 2,
            "fta": 2
        },
        "1628369": { ... }
    },
    "NYK": { ... }
}
```

### 3. Features.feature_set
**Source**: Generated from prior GameStates  
**Size**: ~2-3 KB per game

```json
{
    "home_win_pct_10": 0.7000,
    "home_ppg_10": 112.5000,
    "home_opp_ppg_10": 105.2000,
    "home_net_ppg_10": 7.3000,
    "home_fg_pct_10": 0.4670,
    "home_fg3_pct_10": 0.3560,
    "home_ft_pct_10": 0.8120,
    "home_reb_10": 44.5000,
    "home_ast_10": 25.3000,
    "home_stl_10": 7.8000,
    "home_blk_10": 5.2000,
    "home_tov_10": 12.1000,
    "home_pace_10": 98.5000,
    "home_off_rating_10": 114.2000,
    "home_def_rating_10": 106.9000,
    "home_plus_minus_10": 7.3000,
    "home_rest_days": 2,
    "away_win_pct_10": 0.6000,
    "away_ppg_10": 108.3000,
    ... // 17 more away features
}
```

### 4. Predictions.prediction_set
**Source**: Prediction engines  
**Size**: ~0.5-1 KB per prediction

```json
{
    "home_score": 112.5,
    "away_score": 108.3,
    "total_score": 220.8,
    "home_margin": 4.2,
    "away_margin": -4.2,
    "home_win_prob": 0.6234,
    "away_win_prob": 0.3766,
    "spread": 4.2,
    "confidence": 0.75,                   // Model-specific
    "features_used": 34,                  // Model-specific
    "model_version": "Ridge_2024-07-29"   // Model-specific
}
```

---

## Data Types & Conventions

### Date/Time Formats
- **Game DateTime**: ISO 8601 with Z suffix: `"2024-10-22T19:30:00Z"`
- **Game Date**: ISO date only: `"2024-10-22"`
- **Clock**: ISO 8601 duration: `"PT11M59.00S"` (11 minutes 59 seconds)
- **Prediction DateTime**: ISO 8601: `"2024-10-22T18:00:00"`

### ID Formats
- **game_id**: 10-digit TEXT: `"0022400061"`
  - Char 1: Season type (001=pre, 002=reg, 003=all-star, 004=playoffs)
  - Chars 2-5: Season year (2024 = 2024-2025 season)
  - Chars 6-10: Game number
- **player_id**: INTEGER: `1627759`
- **team_id**: TEXT tricode: `"BOS"` or numeric: `"1610612738"`

### Team Name Conventions
- **Tricode**: 3 letters, uppercase: `"BOS"`, `"LAL"`, `"NYK"`
- **Full Name**: `"Boston Celtics"`, `"Los Angeles Lakers"`
- **Short Name**: `"Celtics"`, `"Lakers"`

### Boolean Values
- SQLite: 0 (false), 1 (true)
- JSON: `true`, `false`

---

## Key Relationships

```
Games (1) ──< (N) PbP_Logs
Games (1) ──< (N) GameStates
Games (1) ──< (N) PlayerBox
Games (1) ──< (2) TeamBox
Games (1) ──< (1) Features
Games (1) ──< (N) Predictions

Teams (1) ──< (N) Games (as home_team or away_team)
Players (1) ──< (N) PlayerBox
```

---

## Notes & Gotchas

1. **TEXT vs INTEGER IDs**: This project uses TEXT for game_id and team_id for simplicity, unlike the Custom_Model branch which used INTEGER foreign keys.

2. **Two PbP Sources**: Always try CDN first (faster, more reliable), fallback to Stats API for historical games.

3. **Clock Format**: NBA uses ISO 8601 duration (`PT11M59.00S`). Convert to seconds: `11*60 + 59 = 719 seconds`.

4. **Minutes Played**: Stored as REAL (float) in minutes. Convert from "MM:SS" format: `36:24` → `36.4` minutes.

5. **game_data_finalized Flag**: Only set to 1 when ALL of PbP_Logs, GameStates, PlayerBox, TeamBox are complete. Prevents partial updates.

6. **Features Dependency**: Cannot create features until both teams have prior game data. New season starts require games to complete first.

7. **Prediction Blending**: Live predictions blend pre-game prediction with current score based on time remaining. Formula: `blend_factor = (time_remaining / total_time)^2`

8. **Rate Limiting**: 0.6s sleep between BoxScore API calls to avoid connection pool warnings.

9. **Season Format**: APIs use abbreviated (`"2023-24"`) but database stores full (`"2023-2024"`).

10. **Status Values**: Games progress: `Scheduled` → `In Progress` → `Completed`/`Final`. Collection only happens when status is `Completed` or `Final`.

---

## Database Validation

### Overview

The database validation suite (`src/database_validator.py`) provides comprehensive automated checks across 9 categories to ensure data quality, logical consistency, and referential integrity. The suite includes 25+ validation checks with auto-fix capabilities for common issues.

### Usage

```bash
# Run all validators
python -m src.database_validator

# Run specific categories
python -m src.database_validator --categories flag,integrity,score

# Auto-fix issues
python -m src.database_validator --fix --categories flag

# Fix specific check
python -m src.database_validator --fix --check-id FLAG-001

# Output as JSON
python -m src.database_validator --output json > validation_report.json
```

### Validation Categories

#### 1. Flag Validator (`--categories flag`)
Validates finalization flag logic and consistency.

| Check ID | Severity | Description | Fixable |
|----------|----------|-------------|---------|
| FLAG-001 | Critical | Games marked `game_data_finalized=1` without final GameState | ✓ |
| FLAG-002 | Critical | Games marked `game_data_finalized=1` without PBP data | ✓ |
| FLAG-003 | Critical | Games marked `pre_game_data_finalized=1` without Features | ✗ |
| FLAG-004 | Warning | Pre-game finalized but teams have no prior finalized games | ✗ |
| FLAG-005 | Critical | `pre_game_data_finalized=1` but `game_data_finalized=0` (logic error) | ✓ |
| FLAG-006 | Warning | Completed games with final state but `game_data_finalized=0` | ✓ |

**Common Issues**: Pre-season games marked finalized without actual data (FLAG-001, FLAG-002). Logic errors where pre-game finalized before game data (FLAG-005). First games of season have no prior states (FLAG-004 - expected).

#### 2. Team Validator (`--categories team`)
Validates team code consistency across tables.

| Check ID | Severity | Description | Fixable |
|----------|----------|-------------|---------|
| TEAM-002 | Critical | GameStates home/away don't match Games table | ✗ |
| TEAM-003 | Warning | Team codes in Games not found in Teams reference table | ✗ |
| TEAM-004 | Critical | Active NBA teams missing from Teams reference table | ✗ |
| TEAM-005 | Critical | TeamBox team codes don't match Games table | ✗ |

**Verification**: NBA API is internally consistent - all PBP teamTricode values match Games table (verified Nov 2025). International teams from pre-season games (TEAM-003 - expected).

#### 3. Integrity Validator (`--categories integrity`)
Validates referential integrity and NULL values.

| Check ID | Severity | Description | Fixable |
|----------|----------|-------------|---------|
| INTEGRITY-001 | Critical | PBP_Logs without matching Games record | ✓ |
| INTEGRITY-002 | Critical | GameStates without matching Games record | ✓ |
| INTEGRITY-003 | Critical | Features without matching Games record | ✓ |
| INTEGRITY-004 | Warning | Predictions without matching Games record | ✓ |
| INTEGRITY-005 | Critical | NULL values in critical fields | ✗ |
| INTEGRITY-006 | Critical | Duplicate GameStates (same game_id + play_id) | ✓ |

**Auto-fix behavior**: Deletes orphaned records and duplicate GameStates (keeps first occurrence).

#### 4. Score Validator (`--categories score`)
Validates score consistency and monotonicity.

| Check ID | Severity | Description | Fixable |
|----------|----------|-------------|---------|
| SCORE-001 | Critical | Scores decreased within same period (non-monotonic) | ✗ |
| SCORE-002 | Critical | Negative scores detected | ✗ |
| SCORE-003 | Critical | Games with multiple different final scores | ✗ |
| SCORE-004 | Warning | Unrealistic score jumps (>10 points in one play) | ✗ |

#### 5. Volume & Temporal Validators
Additional checks for play counts (VOL-001, VOL-002), future games (TEMP-001), and chronological ordering (TEMP-002).

### Validation Workflow

**Pre-Data Collection**:
```bash
python -m src.database_validator --categories flag,integrity
```

**Post-Pipeline Run**:
```bash
python -m src.database_validator --categories flag,team,score
python -m src.database_validator --fix
```

**Expected Issues** (safe to ignore):
- FLAG-004: First games of season have no prior states
- TEAM-003: International teams from pre-season games

**Critical Issues** (require investigation):
- FLAG-005: Logic error in flag setting
- TEAM-002: GameStates don't match Games (data corruption)
- SCORE-001: Non-monotonic scores (API issue or parsing error)
- INTEGRITY-005: NULL values in critical fields
