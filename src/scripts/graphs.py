import os

import pandas as pd
from dotenv import load_dotenv

load_dotenv()
PROJECT_ROOT = os.getenv("PROJECT_ROOT")

database_location = "sqlite:///" + os.path.join(PROJECT_ROOT, "data", "NBA_AI.sqlite")

query = """
SELECT * 
FROM (
    SELECT * 
    FROM Games 
    WHERE season = "2023-2024" 
    AND season_type = "Regular Season"
) AS FilteredGames 
INNER JOIN (
    SELECT * 
    FROM GameStates 
    WHERE is_final_state = 1
) AS FinalGameStates ON FilteredGames.game_id = FinalGameStates.game_id;"""

df = pd.read_sql_query(query, database_location)

print(df.head())

print(df.info())
