-- Create NBA Database Tables

-- Create Teams Table
CREATE TABLE Teams (
    team_id INTEGER PRIMARY KEY,
    abbreviation TEXT NOT NULL UNIQUE,
    abbreviation_normalized TEXT NOT NULL,
    full_name TEXT NOT NULL,
    full_name_normalized TEXT NOT NULL,
    short_name TEXT,
    short_name_normalized TEXT,
    alternatives TEXT,
    alternatives_normalized TEXT,
    logo_file TEXT
);

-- Create Players Table
CREATE TABLE Players (
    player_id INTEGER PRIMARY KEY,
    first_name TEXT NOT NULL,
    last_name TEXT NOT NULL,
    full_name TEXT NOT NULL,
    from_year INTEGER,
    to_year INTEGER,
    roster_status INTEGER,
    team_id INTEGER,
    height REAL,
    weight REAL,
    position TEXT,
    age INTEGER,
    is_active INTEGER DEFAULT 0,
    FOREIGN KEY (team_id) REFERENCES Teams(team_id)
);

-- Create Games Table
CREATE TABLE Games (
    game_id INTEGER PRIMARY KEY,
    date_time_est TEXT NOT NULL,
    home_team TEXT NOT NULL,
    away_team TEXT NOT NULL,
    status TEXT,
    season TEXT,
    season_type TEXT,
    pre_game_data_finalized INTEGER,
    game_data_finalized INTEGER,
    FOREIGN KEY (home_team) REFERENCES Teams(team_id),
    FOREIGN KEY (away_team) REFERENCES Teams(team_id)
);

-- Create Betting Table
CREATE TABLE Betting (
    betting_id INTEGER PRIMARY KEY,
    game_id INTEGER UNIQUE,
    source TEXT NOT NULL,
    ou_open REAL,
    ou_close REAL,
    ou_current REAL,
    ou_updated_at TEXT,
    spread_open REAL,
    spread_close REAL,
    spread_current REAL,
    spread_updated_at TEXT,
    moneyline_home_open REAL,
    moneyline_home_close REAL,
    moneyline_home_current REAL,
    moneyline_home_updated_at TEXT,
    moneyline_away_open REAL,
    moneyline_away_close REAL,
    moneyline_away_current REAL,
    moneyline_away_updated_at TEXT,
    FOREIGN KEY (game_id) REFERENCES Games(game_id)
);

-- Create Injuries Table
CREATE TABLE Injuries (
    injury_id INTEGER PRIMARY KEY,
    player_id INTEGER NOT NULL,
    game_id INTEGER,
    injury_type TEXT NOT NULL,
    status TEXT NOT NULL,
    recovery_timeline TEXT,
    estimated_return_date TEXT,
    reported_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (player_id) REFERENCES Players(player_id),
    FOREIGN KEY (game_id) REFERENCES Games(game_id)
);

-- Create PBP (Play-by-Play) Table
CREATE TABLE PBP (
    game_id INTEGER NOT NULL,
    action_id INTEGER NOT NULL,
    clock TEXT NOT NULL,
    period INTEGER NOT NULL,
    team_id INTEGER,
    team_tricode TEXT,
    player_id INTEGER,
    players_on_court_home TEXT NOT NULL,
    players_on_court_away TEXT NOT NULL,
    description TEXT NOT NULL,
    action_type TEXT NOT NULL,
    sub_type TEXT,
    home_score INTEGER,
    away_score INTEGER,
    PRIMARY KEY (game_id, action_id),
    FOREIGN KEY (game_id) REFERENCES Games(game_id),
    FOREIGN KEY (team_id) REFERENCES Teams(team_id),
    FOREIGN KEY (player_id) REFERENCES Players(player_id)
);

-- Create GameStates Table
CREATE TABLE GameStates (
    game_id INTEGER NOT NULL,
    action_id INTEGER NOT NULL,
    clock TEXT NOT NULL,
    period INTEGER NOT NULL,
    home_score INTEGER NOT NULL,
    away_score INTEGER NOT NULL,
    total_score INTEGER NOT NULL,
    home_margin INTEGER NOT NULL,
    is_final_state INTEGER DEFAULT 0,
    players_data TEXT NOT NULL,
    PRIMARY KEY (game_id, action_id),
    FOREIGN KEY (game_id) REFERENCES Games(game_id),
    FOREIGN KEY (action_id) REFERENCES PBP(action_id)
);

-- Create PlayerBox Table
CREATE TABLE PlayerBox (
    player_id INTEGER NOT NULL,
    game_id INTEGER NOT NULL,
    team_id INTEGER,
    minutes REAL,
    points INTEGER,
    rebounds INTEGER,
    assists INTEGER,
    steals INTEGER,
    blocks INTEGER,
    turnovers INTEGER,
    fouls INTEGER,
    offensive_rebounds INTEGER,
    defensive_rebounds INTEGER,
    field_goals_attempted INTEGER,
    field_goals_made INTEGER,
    three_points_attempted INTEGER,
    three_points_made INTEGER,
    free_throws_attempted INTEGER,
    free_throws_made INTEGER,
    plus_minus INTEGER,
    PRIMARY KEY (player_id, game_id),
    FOREIGN KEY (player_id) REFERENCES Players(player_id),
    FOREIGN KEY (game_id) REFERENCES Games(game_id),
    FOREIGN KEY (team_id) REFERENCES Teams(team_id)
);

-- Create TeamBox Table
CREATE TABLE TeamBox (
    team_id INTEGER NOT NULL,
    game_id INTEGER NOT NULL,
    points INTEGER,
    points_allowed INTEGER,
    rebounds INTEGER,
    assists INTEGER,
    steals INTEGER,
    blocks INTEGER,
    turnovers INTEGER,
    fouls INTEGER,
    field_goals_attempted INTEGER,
    field_goals_made INTEGER,
    three_points_attempted INTEGER,
    three_points_made INTEGER,
    free_throws_attempted INTEGER,
    free_throws_made INTEGER,
    PRIMARY KEY (team_id, game_id),
    FOREIGN KEY (team_id) REFERENCES Teams(team_id),
    FOREIGN KEY (game_id) REFERENCES Games(game_id)
);

-- Create WinProbability Table
CREATE TABLE WinProbability (
    game_id INTEGER NOT NULL,
    action_id INTEGER NOT NULL,
    period INTEGER,
    clock TEXT,
    home_win_percentage REAL,
    away_win_percentage REAL,
    tie_percentage REAL,
    home_score INTEGER,
    away_score INTEGER,
    home_team TEXT,
    away_team TEXT,
    PRIMARY KEY (game_id, action_id),
    FOREIGN KEY (game_id) REFERENCES Games(game_id),
    FOREIGN KEY (action_id) REFERENCES PBP(action_id),
    FOREIGN KEY (home_team) REFERENCES Teams(team_id),
    FOREIGN KEY (away_team) REFERENCES Teams(team_id)
);
