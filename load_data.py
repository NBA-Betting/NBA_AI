import sqlite3

import pandas as pd

from src.config import config

DB_PATH = config["database"]["path"]


def get_player_data(conn, player_id):
    """
    Returns the final query's results for a given player_id
    from a SQLite connection as a pandas DataFrame.
    """
    query = """
    SELECT
        -----------------------------------------------------------------------------
        -- 1) Player Identifiers
        -----------------------------------------------------------------------------
        p_box.player_id,
        p_box.player_name,

        -----------------------------------------------------------------------------
        -- 2) Player Profile
        -----------------------------------------------------------------------------
        ply.position AS player_position,
        ply.height,
        ply.weight,

        -----------------------------------------------------------------------------
        -- 3) Game Info
        -----------------------------------------------------------------------------
        gms.game_id,
        gms.date_time_est,
        gms.season,
        CASE 
            WHEN gms.season_type = 'Play In' THEN 'Post Season'
            ELSE gms.season_type
        END AS season_type,

        -----------------------------------------------------------------------------
        -- 4) Sequence Info
        -----------------------------------------------------------------------------
        ROW_NUMBER() OVER (
            PARTITION BY p_box.player_id
            ORDER BY gms.date_time_est
        ) AS game_in_career,

        ROW_NUMBER() OVER (
            PARTITION BY p_box.player_id, gms.season
            ORDER BY gms.date_time_est
        ) AS game_in_season,

        COALESCE(
            JULIANDAY(gms.date_time_est)
            - LAG(JULIANDAY(gms.date_time_est)) OVER (
                PARTITION BY p_box.player_id, gms.season
                ORDER BY gms.date_time_est
            ),
            0
        ) AS days_since_last_game,

        -----------------------------------------------------------------------------
        -- 5) Player’s Team vs. Opponent
        -----------------------------------------------------------------------------
        p_box.team_id AS player_team_id,
        tm_player.abbreviation AS player_team_abbr,
        CASE 
            WHEN p_box.team_id = gms.home_team_id THEN gms.away_team_id
            ELSE gms.home_team_id
        END AS opponent_team_id,
        tm_opp.abbreviation AS opponent_team_abbr,
        CASE 
            WHEN p_box.team_id = gms.home_team_id THEN 1
            ELSE 0
        END AS was_home,

        -----------------------------------------------------------------------------
        -- 6) Team Performance
        -----------------------------------------------------------------------------
        t_box.pts         AS team_pts,
        t_box.pts_allowed AS team_pts_allowed,
        t_box.plus_minus  AS team_plus_minus,
        CASE 
            WHEN t_box.pts > t_box.pts_allowed THEN 1
            ELSE 0
        END AS team_won,

        -----------------------------------------------------------------------------
        -- 7) Player Performance
        -----------------------------------------------------------------------------
        p_box.position AS game_position,
        p_box.min,
        p_box.pts,
        p_box.reb,
        p_box.ast,
        p_box.stl,
        p_box.blk,
        p_box.tov,
        p_box.pf,
        p_box.oreb,
        p_box.dreb,
        p_box.fga,
        p_box.fgm,
        p_box.fg_pct,
        p_box.fg3a,
        p_box.fg3m,
        p_box.fg3_pct,
        p_box.fta,
        p_box.ftm,
        p_box.ft_pct,
        p_box.plus_minus AS player_plus_minus

    FROM PlayerBox AS p_box
    JOIN Games     AS gms
        ON p_box.game_id = gms.game_id
    JOIN Players   AS ply
        ON p_box.player_id = ply.player_id
    JOIN TeamBox   AS t_box
        ON t_box.game_id = gms.game_id
       AND t_box.team_id = p_box.team_id
    LEFT JOIN Teams AS tm_player
           ON p_box.team_id = tm_player.team_id
    LEFT JOIN Teams AS tm_opp
           ON CASE 
                WHEN p_box.team_id = gms.home_team_id THEN gms.away_team_id
                ELSE gms.home_team_id
              END = tm_opp.team_id

    -- Filter for just one player's data and only include Regular Season, Play In, Post Season
    WHERE p_box.player_id = ?
      AND gms.season_type IN ('Regular Season', 'Play In', 'Post Season')

    -- Order by player, then chronological within each player
    ORDER BY 
        p_box.player_id,
        gms.date_time_est;
    """

    # Read the query into a DataFrame, passing the player_id as a parameter
    df = pd.read_sql_query(query, conn, params=(player_id,))
    return df


if __name__ == "__main__":

    player = 203507

    with sqlite3.connect(DB_PATH) as conn:
        player_df = get_player_data(conn, player_id=player)

        # Print the DataFrame
        print(player_df.head())
        print(player_df.info())
