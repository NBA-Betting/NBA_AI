import json

import pandas as pd


def load_featurized_modeling_data(seasons, db_path):
    """
    Load featurized modeling data from the Games and PriorStates tables.

    This function selects all fields from both tables where the game_id matches and the feature_set field in the
    PriorStates table is not an empty JSON object. The feature_set field is expanded into separate columns in the
    resulting DataFrame.

    Args:
        seasons (list or tuple): A list or tuple of seasons to load data for.
        db_path (str): The path to the SQLite database.

    Returns:
        pandas.DataFrame: A DataFrame containing all the original fields from the Games and PriorStates tables,
        plus the expanded fields from feature_set.
    """
    # Convert seasons list to tuple for IN clause compatibility, if not already a tuple
    seasons_tuple = tuple(seasons) if isinstance(seasons, list) else seasons

    # SQL query, leveraging placeholders for parameterization
    # Selects all fields from both tables and filters out empty 'feature_set' JSON objects
    query = """
        SELECT g.game_id, g.date_time_est, g.home_team, g.away_team, g.season, g.season_type, 
               gs.home_score, gs.away_score, gs.total, gs.home_margin, gs.players_data, ps.feature_set
        FROM Games g
        INNER JOIN PriorStates ps ON g.game_id = ps.game_id
        INNER JOIN GameStates gs ON g.game_id = gs.game_id
        WHERE g.season IN ({placeholders})
        AND ps.feature_set <> '{{}}'
        AND ps.are_prior_states_finalized = 1
        AND g.status = 'Completed'
        AND g.season_type IN ('Regular Season', 'Post Season')
        AND gs.is_final_state = 1
    """.format(
        placeholders=",".join("?" for _ in seasons_tuple)
    )

    # Load the data into a DataFrame
    # The 'params' argument is used to pass the seasons to the query, preventing SQL injection
    df = pd.read_sql_query(query, "sqlite:///" + db_path, params=seasons_tuple)

    # Normalize 'feature_set' into a separate DataFrame
    # This step expands the JSON strings in 'feature_set' into separate columns
    df_feature_set = pd.json_normalize(df["feature_set"].apply(json.loads))

    # Drop the 'feature_set' column from the original DataFrame to avoid duplication
    df.drop(columns=["feature_set"], inplace=True)

    # Concatenate the original DataFrame (without 'feature_set') with the normalized 'feature_set' DataFrame
    # This results in a single DataFrame with all the original fields plus the expanded fields from 'feature_set'
    df_final = pd.concat([df, df_feature_set], axis=1)

    return df_final
