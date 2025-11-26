<img src='images/nba_ai_header_15x6.png' alt='NBA AI'/>

# NBA AI

## Table of Contents
* [Project Overview](#project-overview)
    * [Current State](#current-state)
    * [Future Goals](#future-goals)
    * [Guiding Principles](#guiding-principles)
* [Web App](#web-app)
* [Prediction Engines](#prediction-engines)
* [Development Status](#development-status)

## Project Overview

#### Using AI to predict the outcomes of NBA games.

This project aims to streamline the process of predicting NBA game outcomes by focusing on advanced AI prediction models rather than extensive data collection and management. Unlike my previous project, [NBA Betting](https://github.com/NBA-Betting/NBA_Betting/tree/main), which aimed to create a comprehensive feature set for predicting NBA games through extensive data collection, this project simplifies the process. While the previous approach benefited from various industry-derived metrics, the cost and complexity of managing the data collection were too high. This project focuses on a core data set, such as play-by-play data, and leverages deep learning and GenAI to predict game outcomes.

### Current State

The project is in active development with a complete data collection pipeline and basic prediction engines. **Recent infrastructure cleanup (Nov 2025)** removed unnecessary complexity (Airflow orchestration, Wandb experiment tracking) to focus on the core GenAI prediction engine development.

The current system processes 3 seasons of data (2023-2026) with complete PBP → GameStates → PlayerBox/TeamBox → Features → Predictions pipeline. The web app provides a simple interface for displaying games with current scores and predictions.

![Project Flowchart](images/project_flowchart.png)

The project is built around a few key components:
* **Database Updater**: This component is responsible for updating the database with the latest NBA game data. It fetches data from the NBA Stats API, performs ETL operations, generates features, creates predictions, and stores the data in a SQLite database. It consists of a few modules:
    * `database_update_manager.py`: The main module that orchestrates the entire process.
    * `schedule.py`: Fetches the schedule from the NBA API and updates the database.
    * `pbp.py`: Fetches play-by-play data for games and updates the database.
    * `game_states.py`: Parses play-by-play data to generate game states and updates the database.
    * `prior_states.py`: Determines prior final game states for teams.
    * `features.py`: Uses prior final game states to generate features for the prediction engine.
    * `predictions.py`: Generates predictions for games using the chosen prediction engine.

* **Games API**: This component is responsible for updating predictions for ongoing or completed games and providing the data to the web app. It fetches data from the database, generates predictions, and serves the data to the web app.
    * `games.py`: Fetches game data from the database, manages prediction updating and data formatting.
    * `api.py`: Defines the API endpoints.
    * `games_api.md`: API documentation.

* **Web App**: This component is the front end of the project, providing a simple interface for users to view games and predictions. It is built using Flask.
    * `start_app.py`: The main entry point for the web app found in the root directory.
    * `app.py`: The main module that defines the Flask app and routes.
    * `game_data_processor.py`: Formats game data from the API for the web app.
    * `templates/`: Contains the HTML templates for the web app.
    * `static/`: Contains the CSS and JavaScript files for the web app.

### Future Goals

![Foundational Model Outline](images/foundational_model_outline.png)

1. **Data Sourcing**: Focus on a minimal number of data sources that fundamentally describe basketball. Currently, we use play-by-play data from the NBA API. In the future, incorporating video and tracking data would be interesting, though these require considerably more resources and access.

2. **Prediction Engine**: This is the core of the project and will be the development focus until the 2024-2025 season begins. The current prediction engine options will be replaced with a DL and GenAI-based engine, allowing for decreased data parsing and feature engineering while also scaling to predict more complex outcomes, including individual player performance.

3. **Data Storage**: Future data storage will more seamlessly integrate with the prediction engine. The storage requirements will combine the current SQL-based data used for the API and web app with more advanced vector-based storage for RAG-based GenAI models.

4. **Web App**: This is the project's front end, displaying the games for the selected date along with current scores and predictions. The interface will remain simple while usability is gradually improved. A separate GenAI chat will be added in the future to allow users to interact with the prediction engine and modify individual predictions based on their preferences.

### Guiding Principles

![Project Guiding Principles](images/guiding_principles.png)

- **Time Series Data Inclusive:** A focus on incorporating the sequential nature of events in games and across seasons, recognizing the significance of order and timing in the NBA.
- **Minimal Data Collection:** Streamlining data sourcing to the essentials, aiming for maximum impact with minimal data, thereby reducing time and resource investment.
- **Wider Applicability:** Extending the scope to cover more comprehensive outcomes, moving beyond standard predictions like point spreads or over/unders.
- **Advanced Modeling System:** Developing a system that is not only a learning tool but also potentially novel compared to the methods used by odds setters.
- **Minimal Human Decisions:** Reducing the reliance on human decision-making to minimize errors and the limitations of individual expertise.

## Web App

![Web App Home Page](images/web_app_homepage.png)
![Web App Game Details](images/web_app_game_details.png)

## Prediction Engines

Currently, there are a few basic prediction engines used to predict the outcomes of NBA games. These serve as placeholders for the more advanced DL and GenAI engines that will be implemented in the future. The current engines make pre-game predictions for home and away scores using ML models. These predictions are then used to calculate the win percentage and margin for the home team. Updated (after game start) predictions are based on a combination of the current game score, time remaining, and the pre-game predictions.

### Current Prediction Engines

- **Linear**: Based on a Ridge Regression model, using features generated by aggregating prior final game states for the teams involved.
- **Tree**: Based on an XGBoost model, using the same features as the Linear model.
- **MLP**: Based on a PyTorch MLP model, using the same features as the Linear model.
- **Random**: A random predictor that generates random scores for each game.
- **Baseline**: A simple predictor that predicts the home and away scores based on the home and away teams' PPG and their opponents' PPG. The formula for the baseline predictor is:
  - Home Score: (Home PPG + Away Opponent PPG) / 2
  - Away Score: (Away PPG + Home Opponent PPG) / 2
  - Home Margin: Home Score - Away Score
  - Home Win Probability: Sigmoid(-0.2504 + 0.1949 * Home Margin)


### Performance Metrics

The current metrics are based on pre-game predictions for the home and away team scores, along with downstream metrics such as win percentage and margin. These simple predictors currently outperform the baseline predictor.

In the future, a more challenging baseline based on the Vegas spread will be added when the DL and GenAI models are implemented.

![Prediction Engine Performance Metrics](images/predictor_performance.png)

## Development Status

**⚠️ This project is in active development and not ready for public use.**

The project is currently being refactored and improved. Public installation instructions and sample databases will be provided once the core prediction engine and infrastructure are stable.

### For Developers

If you're interested in contributing or following development progress:

* Monitor the [Issues](https://github.com/NBA-Betting/NBA_AI/issues) page for active work
* Check the TODO.md file for current development priorities
* The project focuses on building DL/GenAI prediction engines using play-by-play data

### Technical Notes

* Current focus: 2023-2024, 2024-2025, 2025-2026 seasons (configurable in config.yaml)
* Database: SQLite with TEXT-based schema (2.1GB working DB, 24GB historical archive)
* Data pipeline: Schedule → PBP → GameStates → PlayerBox/TeamBox → Features → Predictions
* Initial season updates require ~1,500 NBA API calls and may use up to 2GB memory
* Modern data quality monitoring via `data_quality.py` (30x faster than legacy audit)
* Streamlined architecture - removed Airflow, Wandb, CM archive (Nov 2025 cleanup)
* Project built with Python, Flask, SQLite, PyTorch, and nba_api


