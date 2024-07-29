# NBA Game Data API Documentation

## Overview

The NBA Game Data API provides endpoints to retrieve comprehensive game data based on specific game IDs or dates. It supports both basic and detailed data retrieval, including game states, play-by-play logs, and predictive analysis.

## Endpoints

**`/games`**

This endpoint provides game data based on a list of game IDs or a specific date.

**Query Parameters**

* Required:

    * `game_ids` (string): Comma-separated list of game IDs. For example, `game_ids=0042300401,0022300649`.  
    OR
    * `date` (string): Date in the format `YYYY-MM-DD`. For example, `date=2023-11-15`.
* Optional:

    * `predictor` (string): Specifies the predictive model. Must be one of the valid predictors. Defaults to `Best`.
    * `detail_level` (string): Specifies the level of detail. Must be `Basic` or `Normal`. Defaults to `Basic`.
    * `update_predictions` (string): Indicates whether to update predictions. Must be `True` or `False`. Defaults to `True`.

## Data Structure

The response structure varies based on the `detail_level` but includes comprehensive details about the games. Here's a detailed example for the "Normal" detail level, which includes all possible data.

### Normal Detail Level

* **game_id** (string): Unique identifier for the game.
* **date_time_est** (string): Game date and time in Eastern Standard Time.
* **home_team** (string): Name of the home team.
* **away_team** (string): Name of the away team.
* **status** (string): Game status (e.g., scheduled, completed).
* **season** (string): Season identifier.
* **season_type** (string): Type of the season (e.g., regular, playoffs).
* **pre_game_data_finalized** (bool): Indicates if pre-game data is finalized.
* **game_data_finalized** (bool): Indicates if game data is finalized.
* **game_states** (list): A list containing the latest game state.
    * **play_id** (string): Identifier for the play.
    * **game_date** (string): Date of the game state.
    * **home** (string): Home team's identifier.
    * **away** (string): Away team's identifier.
    * **clock** (string): Game clock.
    * **period** (int): Current period.
    * **home_score** (int): Score of the home team.
    * **away_score** (int): Score of the away team.
    * **total** (int): Total score.
    * **home_margin** (int): Score margin for the home team.
    * **is_final_state** (bool): Indicates if it is the final state.
    * **players_data** (dict): Detailed player data, organized by team and player ID.
* **play_by_play** (list): Detailed play-by-play logs.
    * **play_id** (string): Identifier for the play.
    * **period** (int): Period of the play.
    * **clock** (string): Game clock at the time of the play.
    * **scoreHome** (int): Home team score at the time of the play.
    * **scoreAway** (int): Away team score at the time of the play.
    * **description** (string): Description of the play.
* **predictions** (dict): Contains predictions based on the specified predictor.
    * **pre_game** (dict): Predictions made before the game.
        * **model_id** (string): Identifier for the prediction model.
        * **prediction_datetime** (string): Datetime of the prediction.
        * **prediction_set** (dict): Prediction details, including:
            * **pred_home_score** (float): Predicted score for the home team.
            * **pred_away_score** (float): Predicted score for the away team.
            * **pred_home_win_pct** (float): Predicted probability of home team winning.
            * **pred_players** (dict): Predicted statistics for players, organized by team and player ID.
    * **current** (dict): Updated predictions based on the current game state.
        * **pred_home_score** (float): Updated predicted score for the home team.
        * **pred_away_score** (float): Updated predicted score for the away team.
        * **pred_home_win_pct** (float): Updated predicted probability of home team winning.
        * **pred_players** (dict): Updated predicted statistics for players, organized by team and player ID.

## Example Usage

### Example: Retrieve Basic Data by Date

```
GET /api/games?date=2024-04-02
```

### Example: Retrieve Normal Data by Game IDs using the Best Predictor

```
GET /api/games?game_ids=0042300401,0022300649&detail_level=Normal&predictor=Best
```

## Example Response (Normal Detail Level)

Basic Detail Level does not include play-by-play logs as they are often extensive. The response is in JSON format.

```json
{
  "0022300589": {
    "away_team": "DAL",
    "date_time_est": "2024-04-02T22:00:00Z",
    "game_data_finalized": 1,
    "game_states": [
      {
        "away": "DAL",
        "away_score": 100,
        "clock": "PT00M00.00S",
        "game_date": "2024-04-02",
        "home": "GSW",
        "home_margin": 4,
        "home_score": 104,
        "is_final_state": 1,
        "period": 4,
        "play_id": 6820000,
        "players_data": {
          "away": {
            "1627884": {"name": "D. Jones Jr.", "points": 0},
            "1628467": {"name": "M. Kleber", "points": 0},
            "1629023": {"name": "P. Washington", "points": 20},
            ...
          },
          "home": {
            "101108": {"name": "C. Paul", "points": 14},
            "1626172": {"name": "K. Looney", "points": 0},
            "1627780": {"name": "G. Payton II", "points": 8},
            ...
          }
        },
        "total": 204
      }
    ],
    "home_team": "GSW",
    "pre_game_data_finalized": 1,
    "play_by_play": [
      {
        "clock": "PT12M00.00S",
        "description": "Period Start",
        "period": 1,
        "play_id": 20000,
        "scoreAway": 0,
        "scoreHome": 0
      },
      {
        "clock": "PT11M57.00S",
        "description": "Jump Ball T. Jackson-Davis vs. D. Gafford: Tip to A. Wiggins",
        "period": 1,
        "play_id": 40000,
        "scoreAway": 0,
        "scoreHome": 0
      },
      {
        "clock": "PT11M45.00S",
        "description": "MISS A. Wiggins 6' turnaround Hook",
        "period": 1,
        "play_id": 70000,
        "scoreAway": 0,
        "scoreHome": 0
      },
      ...
    ],
    "predictions": {
      "current": {
        "pred_home_score": 104,
        "pred_away_score": 100,
        "pred_home_win_pct": 1.0,
        "pred_players": {
          "away": {
            "1627884": {"name": "D. Jones Jr.", "points": 0},
            "1628467": {"name": "M. Kleber", "points": 0},
            "1629023": {"name": "P. Washington", "points": 20},
            ...
          },
          "home": {
            "101108": {"name": "C. Paul", "points": 14},
            "1626172": {"name": "K. Looney", "points": 0},
            "1627780": {"name": "G. Payton II", "points": 8},
            ...
          }
        }
      },
      "pre_game": {
        "model_id": "Random",
        "prediction_datetime": "2024-07-26 16:43:04",
        "prediction_set": {
          "pred_away_score": 106.0,
          "pred_home_score": 106.0,
          "pred_home_win_pct": 0.43772504793095957,
          "pred_players": {
            "away": {},
            "home": {}
          }
        }
      }
    },
    "season": "2023-2024",
    "season_type": "Regular Season",
    "status": "Completed"
  }
}
...
```
 

