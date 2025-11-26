# NBA AI - Copilot Instructions

## Project North Star

**Core Goal**: Build a GenAI-based prediction engine that uses play-by-play (PBP) data as its primary source, minimizing human-engineered features and data collection complexity.

**Strategic Vision**:

- **Time Series Focus**: Leverage sequential nature of events in games and across seasons
- **Minimal Data Collection**: Maximum impact from minimal data sources (primarily PBP)
- **Wider Applicability**: Predict comprehensive outcomes beyond spreads/over-unders (player props, score distributions, etc.)
- **Advanced Modeling**: Novel GenAI/DL approach vs. traditional feature engineering
- **Minimal Human Decisions**: Reduce manual feature engineering and domain assumptions

**Current Supporting Infrastructure** (maintenance only until GenAI engine ready):

- Web app (Flask frontend for displaying predictions)
- Traditional ML predictors (Ridge, XGBoost, MLP - placeholder models)
- Data collection pipeline (schedule, PBP, game states, features)
- Games API (serves predictions, updates live scores)

**What This Means for Development**:

1. **High Priority**: Anything directly advancing GenAI prediction engine
2. **Medium Priority**: Bug fixes/refactoring enabling GenAI work or critical stability
3. **Low Priority**: Enhancements to supporting infrastructure (defer until post-GenAI)

---

## Project Architecture

This is an NBA game prediction system with a Flask web app, built around a **5-stage data pipeline**:

1. **Schedule Update** → Fetches game schedule from NBA API
2. **Game Data Update** → Fetches play-by-play logs, generates game states, collects boxscores (via `database_updater/pbp.py`, `game_states.py`, `boxscores.py`)
3. **Pre-Game Data** → Determines prior states, creates feature sets (via `prior_states.py`, `predictions/features.py`)
4. **Predictions** → Generates pre-game predictions using pluggable predictors
5. **Web App / Games API** → Serves current predictions with live score updates

**Core orchestrator**: `src/database_updater/database_update_manager.py` runs this pipeline for a given season.

**Recent Improvements** (Nov 2025):

- Enhanced player data collection with height/age parsing and progress bars
- Added live boxscore endpoint support for in-progress games
- Improved logging with rotating file handlers and JSON support
- Chunked `update_pre_game_data()` to handle large batches (100 games at a time)
- Archived `database_updater_CM/` - features merged into main system

### Component Boundaries

- **`src/database_updater/`**: ETL pipeline from NBA API → SQLite (`game_states.py` parses PBP into structured states)
- **`src/predictions/`**: Feature engineering (`features.py`) + pluggable predictor classes in `prediction_engines/`
- **`src/games_api/`**: REST API (`api.py` + `games.py`) that updates predictions for in-progress games
- **`src/web_app/`**: Flask frontend (`app.py`, templates in `templates/`, static assets in `static/`)

## Configuration & Environment

### Virtual Environment

**CRITICAL: Always activate the virtual environment before running any Python commands**

```bash
source venv/bin/activate
```

The project uses a Python virtual environment (`venv/`) with all dependencies isolated. Common issues:

- `ModuleNotFoundError`: Virtual environment not activated
- Package version conflicts: System Python vs venv Python
- Installation location confusion: Use `pip install` (not `pip3` or system pip)

**All Python commands must run inside the venv:**

```bash
# ✅ Correct
source venv/bin/activate
python -m src.database_updater.database_update_manager

# ❌ Wrong - uses system Python
python -m src.database_updater.database_update_manager
```

### Critical Setup Pattern

All config lives in **`config.yaml`** with environment variable interpolation via `${VAR_NAME}`. The `src/config.py` loader:

1. Reads `config.yaml`
2. Replaces `${PROJECT_ROOT}`, `${DATABASE_PATH}`, etc. from `.env` file
3. Computes absolute paths (e.g., `database.path` becomes `/full/path/to/data/NBA_AI_BASE.sqlite`)
4. Auto-generates `WEB_APP_SECRET_KEY` if missing

**Never hardcode paths** - always use `config["database"]["path"]` or `config["project"]["root"]`.

### Database Selection

The active database is set via `.env` or defaults to `data/NBA_AI_2023_2025.sqlite` (working DB). Available databases:

- `NBA_AI_2023_2025.sqlite` (2.1GB) → Current working database (2023-2025 seasons, TEXT schema with PlayerBox/TeamBox)
- `NBA_AI_ALL_SEASONS.sqlite` (24GB) → Master archive, read-only (all historical data, PROTECTED)

Change via `.env`: `DATABASE_PATH=data/NBA_AI_ALL_SEASONS.sqlite`

## Running the Application

### Web App Entry Point

```bash
python start_app.py --predictor=Baseline --log_level=INFO --debug
```

- Valid predictors: `Baseline`, `Linear`, `Tree`, `MLP`
- Default predictor set in `config.yaml` under `default_predictor`
- Flask debug mode enabled with `--debug`

### Database Updates

**Auto-Update (via Web App):**
The web app automatically triggers the full data pipeline when you view any date:

1. Fetches latest schedule for that season
2. Collects PBP for completed games missing data
3. Generates GameStates from PBP
4. Collects PlayerBox/TeamBox boxscores
5. Determines prior states and creates feature sets
6. Generates predictions with specified predictor

Smart updates only process games with `game_data_finalized=0` to avoid redundant API calls.

**Manual Batch Update:**
For collecting all games in a season at once:

```bash
python -m src.database_updater.database_update_manager --season=2024-2025 --predictor=Linear --log_level=DEBUG
```

First run for a season makes ~1500 NBA API calls (one per game) and requires up to 2GB memory. Processes in 100-game chunks to manage memory.

### Predictions

Standalone prediction generation:

```bash
python -m src.predictions.prediction_manager --game_ids=0042300401,0022300649 --predictor=Linear --save
```

### Games API Testing

Fetch game data directly:

```bash
python -m src.games_api.games --date="2024-04-01" --predictor=Baseline --output=screen
```

## Predictor Pattern

All predictors implement two methods:

- `make_pre_game_predictions(game_ids)` → Uses feature sets from DB, returns `{game_id: {home_score, away_score, ...}}`
- `make_current_predictions(game_ids)` → Blends pre-game prediction with current score + time remaining

New predictor checklist:

1. Create class in `src/predictions/prediction_engines/` (see `baseline_predictor.py` for minimal example)
2. Add to `PREDICTOR_MAP` in `prediction_manager.py`
3. Add config entry in `config.yaml` under `predictors:` with any `model_paths`
4. Predictions saved to `Predictions` table via `save_predictions()` (JSON blob in `prediction_set` column)

## Database Schema Highlights

- **`Games`**: Master table with `pre_game_data_finalized` flag (true when prior states available)
- **`GameStates`**: Parsed PBP snapshots (one per play), latest state fetched via CTE in `games.py`
- **`PbP_Logs`**: Raw play-by-play JSON (`log_data` column)
- **`Predictions`**: `(game_id, predictor, prediction_datetime, prediction_set)` - JSON stores all prediction metrics
- **`FeatureSets`**: Pre-game features derived from prior game states (used by ML predictors)
- **`PlayerBox`, `TeamBox`, `Players`, `Teams`**: Boxscore and reference data (Custom_Model branch additions)

Query pattern: `games.py::get_normal_data()` uses CTE to get latest `GameStates` per game, avoiding full table joins.

## Season & Date Utilities

All season/date validation lives in `src/utils.py`:

- `game_id_to_season("0022300649")` → `"2023-2024"` (NBA game IDs encode season/type)
- `date_to_season("2024-03-15")` → `"2023-2024"` (uses Oct 1 cutoff)
- `validate_game_ids()`, `validate_date_format()` → Raise `ValueError` on invalid inputs

**API restrictions**: `config.yaml` limits valid seasons (`valid_seasons: ["2023-2024", "2024-2025", "2025-2026"]`) to avoid accidental bulk updates. Remove to support back to 2000-2001.

## Feature Engineering

`predictions/features.py::create_feature_sets()`:

1. Loads prior final game states for home/away teams (via `prior_states_dict`)
2. Computes **17 rolling averages** (points, FG%, pace, etc.) over last N games
3. Aggregates home/away features into single vector (34 features per game)
4. Saves to `FeatureSets` table for ML model consumption

**Missing prior states**: Games marked `pre_game_data_finalized=0` until both teams have prior games.

## Dependencies & Environment

**Core Dependencies** (requirements.txt has 46 packages organized by purpose):

- Web: Flask, python-dotenv, PyYAML, requests
- Database: SQLAlchemy
- Data: numpy, pandas
- ML: scikit-learn, scipy, xgboost, joblib
- DL: torch (CPU version)
- NBA API: nba_api==1.11.3 (critical - BoxScoreTraditionalV3, PbP, Schedule)
- Optional: python-json-logger

**Never add transitive dependencies** to requirements.txt - pip installs them automatically. Only direct project imports.

**Data Directory Structure**:

```
data/
  NBA_AI_2023_2025.sqlite (2.1GB working DB)
  NBA_AI_ALL_SEASONS.sqlite (24GB master, PROTECTED)
  archive/
    all_player_data.csv
    teams.json
    eval_data/
    old_schemas/
```

## Code Quality Patterns

- **Logging**: Use `logging_config.py::setup_logging(log_level)` at entry points. All modules log via `logging.getLogger()`
- **Execution timing**: Decorate functions with `@log_execution_time(average_over="game_ids")` to log duration per game
- **Database connections**: Always use `with sqlite3.connect(DB_PATH) as conn:` context managers
- **Chunked processing**: `database_update_manager.py::update_game_data()` processes games in 100-game chunks to avoid memory issues
- **Retry logic**: `utils.py::requests_retry_session()` wraps all NBA API calls with exponential backoff

## Common Pitfalls

1. **Path resolution**: Scripts run from project root assume `config.yaml` is in CWD. Always use `python -m src.module` notation.
2. **Predictor name mismatch**: Database stores predictor names as strings. Ensure `PREDICTOR_MAP` keys match `config.yaml` predictor names exactly.
3. **Season type filtering**: Queries must include `AND season_type IN ('Regular Season', 'Post Season')` to exclude All-Star games.
4. **JSON serialization**: NumPy types (`np.float32`, `np.int64`) must be cast to native Python types before `json.dumps()` (see `prediction_manager.py::save_predictions()`)
5. **API limits**: `max_game_ids: 20` in config prevents accidental overload of Games API endpoint.

## Development Workflow

The full data pipeline is orchestrated by `database_update_manager.py::update_database()`:

**Stage 1: Schedule** → `update_schedule(season)` - Fetches game schedule from NBA API
**Stage 2: Players** → `update_players()` - Updates player reference data  
**Stage 3: Game Data** → `update_game_data()`:

- `get_pbp(game_ids)` → `save_pbp()` - Play-by-play logs
- `create_game_states()` → `save_game_states()` - Parsed snapshots (~492 per game)
- `get_boxscores()` → `save_boxscores()` - PlayerBox/TeamBox stats
  **Stage 4: Pre-Game Data** → `update_pre_game_data()`:
- `determine_prior_states_needed()` → `load_prior_states()` - Last game state for each team
- `create_feature_sets()` → `save_feature_sets()` - 34 rolling average features per game
  **Stage 5: Predictions** → `update_prediction_data()`:
- `make_pre_game_predictions()` → `save_predictions()` - ML/GenAI predictions

Each stage checks completion flags (`game_data_finalized`, `pre_game_data_finalized`) to skip already-processed games. Games are processed in 100-game chunks to manage memory.
