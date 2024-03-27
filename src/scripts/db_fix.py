import sqlite3

import pandas as pd
from tqdm import tqdm

# Replace 'your_database.db' with the path to your SQLite database
db_path = "data/NBA_AI.sqlite"

# Connect to the SQLite database
conn = sqlite3.connect(db_path)

pd.set_option("future.no_silent_downcasting", True)

# Assuming your table is named 'GameStates'
table_name = "GameStates"


def update_game_states(conn, game_id, updated_df):
    """
    Update the game states in the database for a given game.
    """
    for _, row in updated_df.iterrows():
        conn.execute(
            f"""
            UPDATE {table_name}
            SET home_score = ?, away_score = ?, total = ?, home_margin = ?
            WHERE game_id = ? AND play_id = ?;
        """,
            (
                row["home_score"],
                row["away_score"],
                row["total"],
                row["home_margin"],
                game_id,
                row["play_id"],
            ),
        )
    conn.commit()


def correct_scores_for_game(conn, game_id):
    """
    Correct the scores for a specific game by forward filling the correct values.
    """
    df = pd.read_sql_query(
        f"SELECT * FROM {table_name} WHERE game_id = '{game_id}' ORDER BY play_id ASC",
        conn,
    )

    # Only forward fill if there's a non-zero score, to avoid carrying over from the previous game
    df["home_score"] = (
        df["home_score"]
        .replace(0, pd.NA)
        .ffill()
        .fillna(0)
        .infer_objects(copy=False)
        .astype(int)
    )
    df["away_score"] = (
        df["away_score"]
        .replace(0, pd.NA)
        .ffill()
        .fillna(0)
        .infer_objects(copy=False)
        .astype(int)
    )

    # Recalculate 'total' and 'home_margin' for each play based on the updated scores
    df["total"] = df["home_score"] + df["away_score"]
    df["home_margin"] = df["home_score"] - df["away_score"]

    update_game_states(conn, game_id, df)


def main():
    # Get a distinct list of game_ids to process each game separately
    game_ids = pd.read_sql_query(f"SELECT DISTINCT game_id FROM {table_name}", conn)

    for game_id in tqdm(game_ids["game_id"]):
        correct_scores_for_game(conn, game_id)

    print("All game states have been corrected.")


if __name__ == "__main__":
    main()

# Close the connection to the database
conn.close()
