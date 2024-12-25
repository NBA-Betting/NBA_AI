# NBA Database Structure

## Overview

This document describes the NBA project's database schema. It stores comprehensive data for NBA games, teams, players, play-by-play actions, game states, betting information, injuries, player box scores, and team box scores.

### Tables

- [Games](#games): Stores metadata for NBA games.
- [Teams](#teams): Stores data for NBA teams.
- [Players](#players): Stores data for NBA players.
- [Betting](#betting): Stores game-level betting data.
- [Injuries](#injuries): Tracks player injuries.
- [PBP (Play-by-Play)](#pbp-play-by-play): Tracks each action within a game.
- [GameStates](#gamestates): Captures a running summary of the game state.
- [PlayerBox](#playerbox): Stores individual player statistics for each game.
- [TeamBox](#teambox): Stores team-level statistics for each game.
- [WinProbability](#winprobability): Stores win probability data for each game.

## **Games**

Core table of the database structure. Includes the schedule and basic game metadata.

- **Columns**:
  | Column Name              | Data Type | Description                                   | Constraints                      |
  |--------------------------|-----------|-----------------------------------------------|----------------------------------|
  | `game_id`                | TEXT      | Unique identifier for the game.               | Primary Key                      |
  | `gamecode`               | TEXT      | Unique game code.                             | Not Null                         |
  | `date_time_est`          | TEXT      | Date and time of the game in ISO-8601 format. | Not Null                         |
  | `home_team_id`           | INTEGER   | Identifier for the home team.                 | Foreign Key (Teams)              |
  | `away_team_id`           | INTEGER   | Identifier for the away team.                 | Foreign Key (Teams)              |
  | `status`                 | TEXT      | Status of the game (e.g., "Scheduled", "Completed"). | Nullable                       |
  | `season`                 | TEXT      | NBA season (e.g., "2024-2025").               | Nullable                         |
  | `season_type`            | TEXT      | Type of season (e.g., "Regular Season", "Playoffs"). | Nullable                      |
  | `boxscores_finalized`    | INTEGER   | Flag indicating if boxscores are finalized (1 for Yes, 0 for No). | Default 0         |
  | `game_states_finalized`  | INTEGER   | Flag indicating if game states are finalized (1 for Yes, 0 for No). | Default 0           |

- **Primary Key**: `game_id`.
- **Foreign Keys**:
  - `home_team_id` references `Teams(team_id)`.
  - `away_team_id` references `Teams(team_id)`.

## **Teams**

Stores data for NBA teams, both current and historical.

- **Columns**:
  | Column Name              | Data Type | Description                                   | Constraints                      |
  |--------------------------|-----------|-----------------------------------------------|----------------------------------|
  | `team_id`                | INTEGER   | Unique identifier for the team (e.g., NBA-assigned ID). | Primary Key |
  | `abbreviation`           | TEXT      | Current abbreviation for the team (e.g., "ATL"). | Not Null, Unique |
  | `abbreviation_normalized`| TEXT      | Lowercase normalized version of abbreviation. | Not Null |
  | `full_name`              | TEXT      | Full name of the team (e.g., "Atlanta Hawks"). | Not Null |
  | `full_name_normalized`   | TEXT      | Lowercase normalized version of full name.    | Not Null |
  | `short_name`             | TEXT      | Short name of the team (e.g., "Hawks").       | Nullable |
  | `short_name_normalized`  | TEXT      | Lowercase normalized version of short name.   | Nullable |
  | `alternatives`           | TEXT      | JSON array of alternative names and abbreviations (e.g., ["STL", "Tri-Cities Blackhawks"]). | Nullable |
  | `alternatives_normalized`| TEXT      | JSON array of normalized alternative names.   | Nullable |
  | `logo_file`              | TEXT      | Path to the team's logo file (if applicable). | Nullable |

- **Primary Key**: `team_id`.

## **Players**

Stores basic biographical data for current and historical NBA players.

- **Columns**:
  | Column Name    | Data Type | Description                                 | Constraints    |
  |----------------|-----------|---------------------------------------------|----------------|
  | `player_id`    | INTEGER   | Unique identifier for the player.           | Primary Key    |
  | `first_name`   | TEXT      | First name of the player.                   | Nullable       |
  | `last_name`    | TEXT      | Last name of the player.                    | Nullable       |
  | `full_name`    | TEXT      | Full name of the player (e.g., "Byron Scott"). | Nullable    |
  | `from_year`    | INTEGER   | Year the player started their career.       | Nullable       |
  | `to_year`      | INTEGER   | Year the player ended their career (or NULL if active). | Nullable |
  | `roster_status`| INTEGER   | Indicates if the player is on a roster (1 for active, 0 for inactive). | Nullable |
  | `team_id`      | INTEGER   | Current team ID for active players.         | Foreign Key (Teams) |
  | `height`       | INTEGER   | Height of the player in centimeters.        | Nullable       |
  | `weight`       | INTEGER   | Weight of the player in kilograms.          | Nullable       |
  | `position`     | TEXT      | Primary position of the player (e.g., "G", "F", "C"). | Nullable |
  | `age`          | INTEGER   | Current age of the player (calculated or stored). | Nullable |
  | `is_active`    | INTEGER   | Current status flag (1 for active, 0 for inactive). | Default 0 |

- **Primary Key**: `player_id`.
- **Foreign Keys**:
  - `team_id` references `Teams(team_id)`.

## **Betting**

Stores betting data for each game, including opening, closing, and current odds for over/under, spread, and moneyline.

- **Columns**:
  | Column Name              | Data Type | Description                                   | Constraints                      |
  |--------------------------|-----------|-----------------------------------------------|----------------------------------|
  | `betting_id`             | INTEGER   | Unique identifier for the betting record.     | Primary Key                      |
  | `game_id`                | INTEGER   | Identifier for the associated game in the Schedule table. | Unique, Foreign Key (Games)          |
  | `source`                 | TEXT      | Source of line/betting info.                  | Not Null                         |
  | `ou_open`                | REAL      | Opening over/under value.                     | Nullable                         |
  | `ou_close`               | REAL      | Closing over/under value.                     | Nullable                         |
  | `ou_current`             | REAL      | Current over/under value, for real-time updates. | Nullable                      |
  | `ou_updated_at`          | TEXT      | Timestamp for when over/under was last updated. | ISO-8601 Format               |
  | `spread_open`            | REAL      | Opening point spread.                         | Nullable                         |
  | `spread_close`           | REAL      | Closing point spread.                         | Nullable                         |
  | `spread_current`         | REAL      | Current point spread, for real-time updates.  | Nullable                         |
  | `spread_updated_at`      | TEXT      | Timestamp for when the point spread was last updated. | ISO-8601 Format               |
  | `moneyline_home_open`    | REAL      | Opening moneyline for the home team.          | Nullable                         |
  | `moneyline_home_close`   | REAL      | Closing moneyline for the home team.          | Nullable                         |
  | `moneyline_home_current` | REAL      | Current moneyline for the home team, for real-time updates. | Nullable                   |
  | `moneyline_home_updated_at` | TEXT   | Timestamp for when the home team's moneyline was last updated. | ISO-8601 Format          |
  | `moneyline_away_open`    | REAL      | Opening moneyline for the away team.          | Nullable                         |
  | `moneyline_away_close`   | REAL      | Closing moneyline for the away team.          | Nullable                         |
  | `moneyline_away_current` | REAL      | Current moneyline for the away team, for real-time updates. | Nullable                   |
  | `moneyline_away_updated_at` | TEXT   | Timestamp for when the away team's moneyline was last updated. | ISO-8601 Format           |

- **Primary Key**: `betting_id`.
- **Foreign Keys**:
  - `game_id` references `Games(game_id)`.

## **Injuries**

Stores information about player injuries, including the type, status, and recovery timeline.

- **Columns**:
  | Column Name            | Data Type | Description                                   | Constraints                      |
  |------------------------|-----------|-----------------------------------------------|----------------------------------|
  | `injury_id`            | INTEGER   | Unique identifier for the injury record.      | Primary Key                      |
  | `player_id`            | INTEGER   | Identifier for the injured player.            | Foreign Key (Players)                     |
  | `game_id`              | INTEGER   | Identifier for the game during which the injury occurred (if applicable). | Foreign Key (Games)       |
  | `injury_type`          | TEXT      | Description of the injury (e.g., "ACL Tear", "Ankle Sprain"). | Not Null         |
  | `status`               | TEXT      | Current injury status (e.g., "Out", "Probable", "Day-to-Day"). | Not Null         |
  | `recovery_timeline`    | TEXT      | Estimated timeline for recovery (e.g., "2-4 weeks", "Indefinite"). | Nullable         |
  | `estimated_return_date`| TEXT      | Estimated return date for the player in ISO-8601 format (e.g., "2024-12-15"). | Nullable     |
  | `reported_at`          | TEXT      | Date when the injury was first reported.      | Not Null                         |
  | `updated_at`           | TEXT      | Timestamp for the last update to the injury record. | Not Null                    |

- **Primary Key**: `injury_id`.
- **Foreign Keys**:
  - `player_id` references `Players(player_id)`.
  - `game_id` references `Games(game_id)`.

## **PBP (Play-by-Play)**

Stores detailed actions that occur during each game, capturing important events such as shots, fouls, turnovers, and more.

- **Columns**:
  | Column Name            | Data Type | Description                                   | Constraints                      |
  |------------------------|-----------|-----------------------------------------------|----------------------------------|
  | `game_id`              | INTEGER   | Identifier for the game in which the action took place. | Not Null, Foreign Key (Games) |
  | `action_id`            | INTEGER   | Unique identifier for the action within the game. | Not Null |
  | `clock`                | TEXT      | Time remaining in the period, represented in ISO-8601 duration format (e.g., "PT11M47S"). | Not Null |
  | `period`               | INTEGER   | Period number in which the action occurred (e.g., 1 for Q1, 2 for Q2). | Not Null |
  | `team_id`              | INTEGER   | Identifier for the team involved in the action. | Foreign Key (Teams) |
  | `team_tricode`         | TEXT      | Abbreviation of the team (e.g., "TOR").          | Nullable       |
  | `player_id`            | INTEGER   | Identifier for the player involved in the action (if applicable). | Foreign Key (Players) |
  | `players_on_court_home` | TEXT     | JSON representation of player IDs on the court for the home team at the time of the action. | Not Null   |
  | `players_on_court_away` | TEXT     | JSON representation of player IDs on the court for the away team at the time of the action. | Not Null   |
  | `description`          | TEXT      | Description of the action (e.g., "Made Shot", "Turnover"). | Not Null       |
  | `action_type`          | TEXT      | Type of action that took place (e.g., "Shot", "Foul", "Rebound"). | Not Null   |
  | `sub_type`             | TEXT      | Subtype of the action (e.g., "3-Point Shot", "Offensive Rebound"). | Nullable |
  | `home_score`           | INTEGER   | Home team's score after the action.            | Nullable |
  | `away_score`           | INTEGER   | Away team's score after the action.            | Nullable |

- **Primary Key**: Composite key (`game_id`, `action_id`).
- **Foreign Keys**:
  - `game_id` references `Games(game_id)`.
  - `team_id` references `Teams(team_id)`.
  - `player_id` references `Players(player_id)`.

## **GameStates**

Stores detailed information about the state of the game at each play, providing a snapshot of the game progression after each action.

- **Columns**:
  | Column Name            | Data Type | Description                                   | Constraints                      |
  |------------------------|-----------|-----------------------------------------------|----------------------------------|
  | `game_id`              | INTEGER   | Identifier for the game.                      | Primary Key, Foreign Key (Games)    |
  | `action_id`            | INTEGER   | Sequential action ID within the game (matches PBP table). | Primary Key, Foreign Key (PBP)  |
  | `clock`                | TEXT      | Time remaining in the period.                 | Not Null                         |
  | `period`               | INTEGER   | Period number (e.g., 1 for Q1, 2 for Q2).     | Not Null                         |
  | `home_score`           | INTEGER   | Total score for the home team.                | Not Null                         |
  | `away_score`           | INTEGER   | Total score for the away team.                | Not Null                         |
  | `total_score`          | INTEGER   | Combined score of both teams.                 | Not Null                         |
  | `home_margin`          | INTEGER   | Home team's lead or deficit (home_score - away_score). | Not Null                    |
  | `is_final_state`       | INTEGER   | Flag indicating if this is the game's final state (1 for Yes, 0 for No). | Default 0                |
  | `players_data`         | TEXT      | JSON representation of player stats for both teams. | Not Null                    |

- **Primary Key**: Composite key (`game_id`, `action_id`).
- **Foreign Keys**:
  - (`game_id`, `action_id`) references `PBP(game_id, action_id)`.

## **PlayerBox**

Stores game-level statistics for individual players, capturing essential performance metrics for each game.

- **Columns**:
  | Column Name            | Data Type | Description                                   | Constraints                      |
  |------------------------|-----------|-----------------------------------------------|----------------------------------|
  | `player_id`            | INTEGER   | Identifier for the player.                    | Primary Key, Foreign Key (Players) |
  | `game_id`              | INTEGER   | Identifier for the game.                      | Primary Key, Foreign Key (Games)   |
  | `team_id`              | INTEGER   | Identifier for the player's team.             | Foreign Key (Teams)              |
  | `player_name`          | TEXT      | Name of the player.                           | Nullable                         |
  | `start_position`       | TEXT      | Starting position of the player.              | Nullable                         |
  | `min`                  | REAL      | Minutes played by the player.                 | Nullable                         |
  | `pts`                  | INTEGER   | Total points scored by the player.            | Nullable                         |
  | `reb`                  | INTEGER   | Total rebounds by the player.                 | Nullable                         |
  | `ast`                  | INTEGER   | Total assists by the player.                  | Nullable                         |
  | `stl`                  | INTEGER   | Total steals by the player.                   | Nullable                         |
  | `blk`                  | INTEGER   | Total blocks by the player.                   | Nullable                         |
  | `tov`                  | INTEGER   | Total turnovers by the player.                | Nullable                         |
  | `pf`                   | INTEGER   | Total personal fouls committed by the player. | Nullable                         |
  | `oreb`                 | INTEGER   | Total offensive rebounds by the player.       | Nullable                         |
  | `dreb`                 | INTEGER   | Total defensive rebounds by the player.       | Nullable                         |
  | `fga`                  | INTEGER   | Number of field goal attempts.                | Nullable                         |
  | `fgm`                  | INTEGER   | Number of field goals made.                   | Nullable                         |
  | `fg_pct`               | REAL      | Field goal percentage.                        | Nullable                         |
  | `fg3a`                 | INTEGER   | Number of three-point attempts.               | Nullable                         |
  | `fg3m`                 | INTEGER   | Number of three-point shots made.             | Nullable                         |
  | `fg3_pct`              | REAL      | Three-point percentage.                       | Nullable                         |
  | `fta`                  | INTEGER   | Number of free throw attempts.                | Nullable                         |
  | `ftm`                  | INTEGER   | Number of free throws made.                   | Nullable                         |
  | `ft_pct`               | REAL      | Free throw percentage.                        | Nullable                         |
  | `plus_minus`           | INTEGER   | Plus-minus statistic for the player.          | Nullable                         |

- **Primary Key**: Composite key (`player_id`, `game_id`).
- **Foreign Keys**:
  - `player_id` references `Players(player_id)`.
  - `game_id` references `Games(game_id)`.
  - `team_id` references `Teams(team_id)`.

## **TeamBox**

Stores game-level statistics for teams, summarizing essential performance metrics for each game.

- **Columns**:
  | Column Name            | Data Type | Description                                   | Constraints                      |
  |------------------------|-----------|-----------------------------------------------|----------------------------------|
  | `team_id`              | INTEGER   | Identifier for the team.                      | Primary Key, Foreign Key (Teams) |
  | `game_id`              | INTEGER   | Identifier for the game.                      | Primary Key, Foreign Key (Games) |
  | `pts`                  | INTEGER   | Total points scored by the team.              | Nullable                         |
  | `pts_allowed`          | INTEGER   | Total points allowed by the team.             | Nullable                         |
  | `reb`                  | INTEGER   | Total rebounds by the team.                   | Nullable                         |
  | `ast`                  | INTEGER   | Total assists by the team.                    | Nullable                         |
  | `stl`                  | INTEGER   | Total steals by the team.                     | Nullable                         |
  | `blk`                  | INTEGER   | Total blocks by the team.                     | Nullable                         |
  | `tov`                  | INTEGER   | Total turnovers committed by the team.        | Nullable                         |
  | `pf`                   | INTEGER   | Total fouls committed by the team.            | Nullable                         |
  | `fga`                  | INTEGER   | Number of field goal attempts by the team.    | Nullable                         |
  | `fgm`                  | INTEGER   | Number of field goals made by the team.       | Nullable                         |
  | `fg_pct`               | REAL      | Field goal percentage.                        | Nullable                         |
  | `fg3a`                 | INTEGER   | Number of three-point attempts by the team.   | Nullable                         |
  | `fg3m`                 | INTEGER   | Number of three-point shots made by the team. | Nullable                         |
  | `fg3_pct`              | REAL      | Three-point percentage.                       | Nullable                         |
  | `fta`                  | INTEGER   | Number of free throw attempts by the team.    | Nullable                         |
  | `ftm`                  | INTEGER   | Number of free throws made by the team.       | Nullable                         |
  | `ft_pct`               | REAL      | Free throw percentage.                        | Nullable                         |
  | `plus_minus`           | INTEGER   | Plus-minus statistic for the team.            | Nullable                         |

- **Primary Key**: Composite key (`team_id`, `game_id`).
- **Foreign Keys**:
  - `team_id` references `Teams(team_id)`.
  - `game_id` references `Games(game_id)`.


