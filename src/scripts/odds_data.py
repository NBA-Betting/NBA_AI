import pandas as pd
from sportsdataverse.nba.nba_pbp import espn_nba_pbp
from sportsdataverse.nba.nba_schedule import espn_nba_schedule

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)

if __name__ == "__main__":
    test_game_ids = ["0022200001", "0022200919", "0022301170"]

    schedule_columns = [
        "id",
        "date",
        "home_abbreviation",
        "away_abbreviation",
        "season",
    ]

    schedule = espn_nba_schedule(dates="2019-01-03", return_as_pandas=True, limit=1100)
    print(schedule)
    schedule = schedule[schedule["season_type"].isin([2, 3])]
    schedule = schedule[schedule_columns]
    ordered_schedule = schedule.sort_values(by="date", ascending=True)

    print(ordered_schedule)
    print(len(ordered_schedule))

    # idk = espn_nba_pbp(401307514)

    # print(idk)
