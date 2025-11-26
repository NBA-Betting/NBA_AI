# NBA AI TODO

> **Last Updated**: November 26, 2025  
> **Current Sprint**: (None - infrastructure cleanup complete, ready for data collection)
>
> **Project Management Approach**: Active sprint contains expansive planning with phases and subtasks. Backlog items remain high-level (goal + why + approach). Completed sprints condensed to title, date, and key achievements only.
>
> **⚠️ Document Maintenance Rules**:
> - **Active Sprint**: Only ONE active sprint at a time with full detail (goal, phases, insights, challenges)
> - **Backlog**: High-level only (goal + why + 3-5 approach bullets). NO phases, NO detailed subtasks
> - **Completed Work**: 2-5 key bullets per sprint. NO phases, NO bug fixes, NO file paths, NO verbose details
> - **When completing sprint**: Immediately condense from expansive → succinct format (remove ~80% of content)
> - **Archive old sprints**: Keep only last 10-15 completed sprints, move older to ARCHIVE.md if needed

---

## 🎯 Active Sprint

**No active sprint** - Ready for next phase of work

---

## 📋 Backlog

### 🔧 Infrastructure & Data Collection (Priority 1 - Complete Before GenAI)

#### 1. Complete 2024-2025 Season Data Collection
**Goal**: Collect all available data for current season (~1200 games remaining)  
**Why**: Need complete dataset before infrastructure validation and GenAI work  
**Approach**:
- Run: `python -m src.database_updater.database_update_manager --season=2024-2025`
- Monitor: Verify PBP, GameStates, PlayerBox, TeamBox all collecting
- Validate: Check for missing games, incomplete data, API errors
- Document: Final data coverage statistics

#### 2. Data Quality & Pipeline Validation
**Goal**: Verify all data collection pipelines work correctly and produce quality data  
**Why**: Can't train models on broken/incomplete data  
**Approach**:
- Audit: PBP coverage across all seasons (check for gaps, parsing errors)
- Audit: PlayerBox/TeamBox completeness (verify all games have boxscores)
- Test: Run full pipeline on new games (schedule → PBP → states → features → predictions)
- Validate: GameStates parsing quality (~400 snapshots/game, player tracking accurate)
- Document: Data quality report with coverage statistics

#### 2. Code Quality & Testing Infrastructure
**Goal**: Add basic testing and improve code quality  
**Why**: Tests catch bugs early, improve confidence in changes  
**Status**: ✅ Database validation complete, pytest deferred
**Approach**:
- ✅ Database validator with 25+ checks across 9 categories
- Defer: Unit tests until clear testing strategy emerges
- Defer: pytest configuration (validation suite sufficient for now)

#### 3. Environment & Setup Validation
**Goal**: Ensure all systems functional after branch consolidation  
**Why**: Need stable foundation before GenAI development  
**Status**: ✅ Complete - all imports verified, dead code removed
**Approach**:
- ✅ Verified all module imports work correctly
- ✅ Removed test files, redundant scripts, cache files
- ✅ Web app tested end-to-end (Sprint 7)
- ✅ Database update pipeline validated (Sprint 6)

#### 4. Critical Bug Fixes & Refactoring
**Goal**: Fix blocking issues and clean up technical debt  
**Why**: Stable foundation needed before GenAI development  
**Status**: ✅ Major cleanup complete
**Approach**:
- ✅ Removed dead code (test files, empty models/, cache files)
- ✅ Fixed flag inconsistencies (180 games)
- Continue: Monitor for issues during data collection

#### 5. Logging & Monitoring Improvements
**Goal**: Improve observability of data collection and predictions  
**Why**: Better logging helps debug issues and monitor production system  
**Approach**:
- Audit: Review current logging configuration (logging_config.py)
- Implement: Structured logging with consistent format across modules
- Add: Performance metrics logging (API call times, processing duration)
- Add: Error tracking and alerting for pipeline failures
- Consider: Add Sentry or similar for error monitoring

---

### 🎨 Frontend & User Experience (Priority 2 - Before GenAI)

#### 6. Web App UX Improvements
**Goal**: Enhance UI with better usability and mobile support  
**Why**: Better UX makes predictions more useful and accessible  
**Approach**:
- Implement: Auto-refresh for live games (WebSocket or polling)
- Implement: Mobile responsive design
- Implement: Confidence intervals and probability distributions display
- Implement: Historical prediction accuracy charts
- Test: Cross-browser, mobile devices

#### 7. Baseline Metrics Framework
**Goal**: Establish rigorous evaluation comparing predictors vs. Vegas lines  
**Why**: Need objective performance measurement before GenAI development  
**Approach**:
- Implement: Betting table + data collection from odds API
- Implement: Evaluation metrics (MAE, Brier score, ROI, calibration)
- Implement: Test harness comparing predictors on held-out games
- Run: Benchmark current ML models vs. Vegas to set baseline
- Document: Performance targets for GenAI to beat

---

### 📈 Feature Enhancements (Priority 3 - Before GenAI)

#### 8. Player Prediction Model (Traditional ML)
**Goal**: Build ML model predicting player points/props using PlayerBox data  
**Why**: Player-level predictions add value, test infrastructure before GenAI  
**Approach**:
- Design: Feature engineering for player predictions (recent performance, matchups, etc.)
- Implement: Train Ridge/XGBoost models on PlayerBox historical data
- Evaluate: Compare player prop predictions vs. Vegas lines
- Deploy: Integrate into web app as new prediction type
- Note: This will be replaced by GenAI eventually, but validates data collection

#### 9. Injury Status Tracking
**Goal**: Track player injuries and incorporate into predictions  
**Why**: Important signal that impacts game outcomes  
**Approach**:
- Research: Find reliable injury data API (ESPN, NBA official, etc.)
- Implement: Injury table in database with status tracking
- Implement: Update pipeline to collect injury reports daily
- Integrate: Add injury flags to feature engineering
- Display: Show injury status in web app

---

### 🚀 GenAI Engine Development (Priority 4 - Final Phase)

#### 10. PBP Data Quality Assessment for GenAI
**Goal**: Validate PBP data completeness and quality for GenAI training  
**Why**: GenAI engine depends on rich, sequential PBP - need to confirm readiness  
**Approach**:
- Audit: Final coverage check across all seasons
- Analyze: Determine if parsed GameStates sufficient or need raw PBP
- Test: Load sample games into prototype model input format
- Document: Data quality report with GenAI-specific requirements

#### 11. GenAI Architecture Design & Planning
**Goal**: Define technical architecture for DL/GenAI prediction engine  
**Why**: Core project goal - need concrete plan before implementation  
**Approach**:
- Research: Review transformer models for time series (TFT, Informer, etc.)
- Research: Evaluate RAG architectures for game context (vector DB + LLM)
- Design: Choose between end-to-end transformer, RAG hybrid, or staged approach
- Design: Define input format, training strategy, evaluation metrics
- Document: Complete architecture spec

#### 12. GenAI Prototype v1 - Baseline Model
**Goal**: Build minimal viable GenAI predictor proving concept works  
**Why**: De-risk approach before full implementation  
**Approach**:
- Implement: Simple transformer or LSTM ingesting game state sequences
- Train: On 2023-2024 season predicting final scores
- Evaluate: Compare vs. current ML predictors and Vegas lines
- Iterate: Identify improvements needed
- Decide: Proceed or pivot based on results

#### 13. GenAI Production Model
**Goal**: Build full production-quality GenAI predictor  
**Why**: Achieve project's core vision  
**Approach**:
- Implement: Scale architecture across all seasons
- Implement: Player-level predictions and probability distributions
- Train: Full training run with proper train/val/test splits
- Deploy: Replace traditional ML predictors
- Integrate: Vector store if using RAG approach

#### 14. GenAI Chat Interface
**Goal**: Add conversational interface for prediction exploration  
**Why**: README mentions future goal for interactive predictions  
**Approach**:
- Design: Chat UI for querying predictions, adjusting parameters
- Implement: LLM integration for natural language interaction
- Deploy: Add to web app as new feature

---

## ✅ Completed Sprints

### Sprint 8: Database Validation & Code Cleanup (Nov 26, 2025)
- Built comprehensive database validator (25+ checks across 9 categories)
- Auto-fixed 180 flag inconsistencies, deleted 4,256 old predictions
- Removed dead code (test files, redundant scripts, empty directories)
- Verified all imports work correctly after cleanup
- Added validation documentation to DATA_MODEL.md

### Sprint 7: Web App End-to-End Testing (Nov 25, 2025)
- Tested modal functionality (player headshots, play-by-play, predictions display)
- Verified date navigation and game data loading across multiple dates
- Fixed timezone conversion bugs (UTC↔local) and empty game_states IndexError
- Added player enrichment skip option (5+ min → <2 sec improvement)

### Sprint 6: Database Architecture Consolidation (Nov 25, 2025)
- Deleted database_updater_CM/ archive - fully integrated into main system
- PlayerBox/TeamBox in src/database_updater/boxscores.py (adapted from CM)
- TEXT-based game_id schema unified across all tables
- Single clear data pipeline via database_update_manager.py

### Sprint 5: Data Lineage Enhancements (Nov 25, 2025)
- Added ScheduleCache table for tracking schedule update timestamps
- Schedule caching prevents redundant NBA API calls (5-min cache for historical seasons)
- Prediction timestamp validation prevents post-game predictions
- Timezone-aware datetime handling (UTC) eliminates tz-naive comparison errors
- Force flag for manual cache override (--force)

### Sprint 4: Web App Testing & Live Data Collection (Nov 25, 2025)
- Implemented live game data collection pipeline (Step 3.5 in main pipeline)
- Integrated live/stats endpoint selection in boxscores module
- Verified web app functionality: modals, navigation, player images, play-by-play
- Fixed timezone bugs (UTC → local time) and empty game_states IndexError
- Added player enrichment skip option (major performance improvement)

### Sprint 3: Prediction Engine Refactoring (Nov 25, 2025)
- Created base predictor classes eliminating 95% code duplication
- Refactored all 4 predictors to use inheritance
- Built unified training script (train.py) consolidating 3 separate scripts
- Verified end-to-end pipeline: 1,304/1,400 games with complete predictions

### Sprint 2: Infrastructure Cleanup & Simplification (Nov 24-25, 2025)
- Removed 4 subsystems (Archive/CM, Wandb, Airflow, GPT4Mini)
- Created modern data quality system (data_quality.py - 30x faster)
- Implemented schedule caching (ScheduleCache table - 316x speedup)
- Added prediction timestamp validation (prevents post-game predictions)
- Requirements cleanup (87→46 packages), .env.template documentation

### Sprint 1: Data Pipeline Validation & Infrastructure (Nov 23-24, 2025)
- Unified database setup (NBA_AI_2023_2025.sqlite)
- Integrated PlayerBox/TeamBox into main pipeline
- Verified full pipeline works end-to-end
- Created timing analysis preventing train/test contamination

---

## 🗄️ Archive

(Older completed sprints will be moved here when list grows beyond 10-15 items)
