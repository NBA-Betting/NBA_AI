import glob
import os
import re
from datetime import datetime

import pandas as pd
import tqdm
from dotenv import load_dotenv

load_dotenv()
PROJECT_ROOT = os.getenv("PROJECT_ROOT")
import traceback

from player_mapper import NBAPlayerMapper

# Initialize NBAPlayerMapper
csv_path = os.path.join(PROJECT_ROOT, "data", "all_players.csv")
player_mapper = NBAPlayerMapper(csv_path)

pd.set_option("display.max_columns", None)


def initialize_game_base_state(df):
    # Assuming df is already sorted by play_id; if not, do so
    df_sorted = df.sort_values(by="play_id")
    first_row = df_sorted.iloc[0]  # Get the first row to identify starting lineups

    # Initialize base state with starting lineup column identifiers and player IDs
    base_state = {
        "play_id": first_row["play_id"],
        "home_score": first_row["home_score"],
        "away_score": first_row["away_score"],
        "period": first_row["period"],
        "remaining_time": first_row["remaining_time"],
        "home_margin": first_row["home_score"] - first_row["away_score"],
        "total": first_row["home_score"] + first_row["away_score"],
    }

    # Assign starting players to their respective column identifiers and fetch their IDs
    for i, position in enumerate(["h1", "h2", "h3", "h4", "h5"], start=1):
        player_name = first_row[position]
        player_id = player_mapper.name_to_id(
            player_name
        )  # Use player_mapper to get player ID
        base_state[f"p{i}h_name"] = player_name
        base_state[f"p{i}h_id"] = player_id  # Include player ID
        base_state[f"p{i}h_pts"] = 0  # Initialize points
        base_state[f"p{i}h_pm"] = 0  # Initialize plus-minus

    for i, position in enumerate(["a1", "a2", "a3", "a4", "a5"], start=1):
        player_name = first_row[position]
        player_id = player_mapper.name_to_id(
            player_name
        )  # Use player_mapper to get player ID
        base_state[f"p{i}a_name"] = player_name
        base_state[f"p{i}a_id"] = player_id  # Include player ID
        base_state[f"p{i}a_pts"] = 0  # Initialize points
        base_state[f"p{i}a_pm"] = 0  # Initialize plus-minus

    return base_state


def update_game_state_iteratively(df, base_state):
    # Initialize the DataFrame to store the updated game states
    updated_game_states = [base_state]  # Start with the base state

    # Initialize dictionaries to track mappings and next available identifiers
    # Maps player names to their column identifiers (e.g., 'LeBron James': 'p1h')
    # Initialized with starting lineup mappings from the base state
    player_column_mappings = {}
    for i in range(1, 6):
        player_column_mappings[base_state[f"p{i}h_name"]] = f"p{i}h"
        player_column_mappings[base_state[f"p{i}a_name"]] = f"p{i}a"

    # Tracks the next available column identifier for home and away.
    # Starts at 6 since 1 thru 5 are assigned to starting lineups in the base state
    next_column_identifiers = {
        "h": 6,
        "a": 6,
    }

    # Iterate through each row in the DataFrame (excluding the first row, already in base_state)
    for index, row in df.iloc[1:].iterrows():
        previous_state = updated_game_states[-1].copy()  # Get the previous state
        current_state = previous_state.copy()  # Start with the previous state

        # Part 1: Game State Metrics
        # Update the current state with the game state metrics
        current_state.update(
            {
                "play_id": row["play_id"],
                "home_score": row["home_score"],
                "away_score": row["away_score"],
                "period": row["period"],
                "remaining_time": row["remaining_time"],
                "home_margin": row["home_score"] - row["away_score"],
                "total": row["home_score"] + row["away_score"],
            }
        )

        # Part 2: Player Names and IDs along with Column Assigning

        # Iterate through home and away team player positions
        for team_prefix in ["h", "a"]:
            for pos in range(1, 6):  # For positions 1-5
                player_name = row[f"{team_prefix}{pos}"]
                if pd.notna(player_name):
                    # Check if the player already has a column identifier
                    if player_name not in player_column_mappings:
                        # Assign next available column identifier
                        column_identifier = next_column_identifiers[team_prefix]
                        player_column_mappings[player_name] = (
                            f"p{column_identifier}{team_prefix}"
                        )
                        next_column_identifiers[team_prefix] += 1

                        # Get player ID from the mapper
                        player_id = player_mapper.name_to_id(player_name)

                        # Update current state with player info
                        current_state[player_column_mappings[player_name] + "_name"] = (
                            player_name
                        )
                        current_state[player_column_mappings[player_name] + "_id"] = (
                            player_id
                        )
                        current_state[player_column_mappings[player_name] + "_pts"] = 0
                        current_state[player_column_mappings[player_name] + "_pm"] = 0
                    else:
                        # Ensure existing players are represented even if not active in the current play
                        column_identifier = player_column_mappings[player_name]
                        current_state[column_identifier + "_name"] = player_name
                        current_state[column_identifier + "_id"] = previous_state[
                            column_identifier + "_id"
                        ]

        # Part 3: Player Cumulative Points
        if pd.notna(row["player"]) and pd.notna(row["points"]) and row["points"] > 0:
            scoring_player = row["player"]
            points_scored = row["points"]
            scoring_player_points_column = (
                player_column_mappings[scoring_player] + "_pts"
            )
            current_state[scoring_player_points_column] += points_scored

        # Part 4: Player Cumulative Plus-Minus
        previous_score_diff = (
            previous_state["home_score"] - previous_state["away_score"]
        )
        current_score_diff = row["home_score"] - row["away_score"]
        score_diff_change = current_score_diff - previous_score_diff
        home_player_pm_adjustment = score_diff_change
        away_player_pm_adjustment = -score_diff_change

        players_on_court_home = [
            row[f"h{i}"] for i in range(1, 6) if pd.notna(row[f"h{i}"])
        ]
        players_on_court_away = [
            row[f"a{i}"] for i in range(1, 6) if pd.notna(row[f"a{i}"])
        ]

        players_on_court_home_column_identifiers = [
            player_column_mappings[player] for player in players_on_court_home
        ]
        players_on_court_away_column_identifiers = [
            player_column_mappings[player] for player in players_on_court_away
        ]

        for column_identifier in players_on_court_home_column_identifiers:
            current_state[column_identifier + "_pm"] += home_player_pm_adjustment
        for column_identifier in players_on_court_away_column_identifiers:
            current_state[column_identifier + "_pm"] += away_player_pm_adjustment

        # Append the current_state to the list of updated game states
        updated_game_states.append(current_state)

    # Convert the list of game states into a DataFrame for final output
    final_game_state_df = pd.DataFrame(updated_game_states)
    return final_game_state_df


def create_game_states(df):
    # Initialize the base game state
    base_state = initialize_game_base_state(df)

    # Update the game state iteratively
    final_game_state_df = update_game_state_iteratively(df, base_state)

    return final_game_state_df


def create_game_state_files(input_csv_path):
    # Ensure the input path is absolute
    input_csv_path = os.path.abspath(input_csv_path)

    # Load the CSV file into a DataFrame
    df = pd.read_csv(input_csv_path)

    # Extract year1 and year2 from the folder name
    folder_path = os.path.dirname(input_csv_path)
    parent_folder_name = os.path.basename(folder_path)
    year_pattern = re.compile(r"(\d{4})-(\d{4})_.*")
    match = year_pattern.search(parent_folder_name)
    if match:
        year1, year2 = match.groups()
    else:
        # If year1 and year2 are not in the folder name, raise an error
        raise ValueError("Could not extract year1 and year2 from the folder name.")

    # Extract game_id, dataset, and date from the input DataFrame
    game_id = df["game_id"].iloc[0]
    dataset = df["data_set"].iloc[0]
    date = df["date"].iloc[0]

    # Extract home_team and away_team from the filename
    filename_pattern = re.compile(r"-(\w+)@(\w+)\.csv$")
    match = filename_pattern.search(input_csv_path)
    if match:
        away_team, home_team = match.groups()
    else:
        away_team, home_team = "Unknown", "Unknown"

    # Assume calculate_game_state is a function you've defined elsewhere
    detailed_game_state_df = create_game_states(df)

    # Directory handling for Game_States
    parent_dir = os.path.dirname(folder_path)
    game_states_folder = os.path.join(parent_dir, f"{year1}-{year2}_Game_States")
    os.makedirs(game_states_folder, exist_ok=True)
    base_filename = os.path.splitext(os.path.basename(input_csv_path))[0]
    game_states_csv_file = os.path.join(
        game_states_folder, f"{base_filename}_game_states.csv"
    )
    detailed_game_state_df.to_csv(game_states_csv_file, index=False)

    # Preparing the final game state DataFrame
    final_state_df = detailed_game_state_df.iloc[[-1]].drop(
        ["play_id", "period", "remaining_time"], axis=1
    )
    new_columns_df = pd.DataFrame(
        {
            "game_id": [game_id],
            "data_set": [dataset],
            "date": [date],
            "home_team": [home_team],
            "away_team": [away_team],
        }
    )
    final_state_ordered_df = pd.concat(
        [new_columns_df, final_state_df.reset_index(drop=True)], axis=1
    )

    # Directory handling for Final_States
    final_states_folder = os.path.join(parent_dir, f"{year1}-{year2}_Final_States")
    os.makedirs(final_states_folder, exist_ok=True)
    final_state_csv_file = os.path.join(
        final_states_folder, f"{base_filename}_final_state.csv"
    )
    final_state_ordered_df.to_csv(final_state_csv_file, index=False)

    # New logic for Prior_States
    # Directory handling for Prior_States
    prior_states_folder = os.path.join(parent_dir, f"{year1}-{year2}_Prior_States")
    os.makedirs(prior_states_folder, exist_ok=True)
    prior_state_csv_file = os.path.join(
        prior_states_folder, f"{base_filename}_prior_state.csv"
    )

    final_state_filename_pattern = re.compile(
        r"\[(\d{4}-\d{2}-\d{2})\]-(\d{10})-(\w{3})@(\w{3})(?:_final_state)?\.csv$"
    )

    pattern = os.path.join(final_states_folder, "*.csv")

    # Convert current game date string to datetime object for comparison
    try:
        current_game_date_dt = datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        current_game_date_dt = datetime.strptime(date, "%m/%d/%Y")

    all_prior_states_df = pd.DataFrame()

    for team in [home_team, away_team]:
        team_prior_games_final_states = []
        for file in glob.glob(pattern):
            filename = os.path.basename(file)
            match = final_state_filename_pattern.match(filename)
            if match:
                # Extract the components from the filename
                date_str, game_id, away, home = match.groups()
                game_date_dt = datetime.strptime(date_str, "%Y-%m-%d")

                # Check if the file includes the specified team and predates the current game
                if (team in [home, away]) and (game_date_dt < current_game_date_dt):
                    df = pd.read_csv(file)
                    df["current_focus_team"] = team
                    team_prior_games_final_states.append(df)

        if len(team_prior_games_final_states) == 0:
            team_prior_states_df = pd.DataFrame()
        else:
            team_prior_states_df = pd.concat(
                team_prior_games_final_states, ignore_index=True
            )

        all_prior_states_df = pd.concat([all_prior_states_df, team_prior_states_df])

    all_prior_states_df.to_csv(prior_state_csv_file, index=False)

    return game_states_csv_file, final_state_csv_file, prior_state_csv_file


def create_game_state_files_by_folder(folder_path):
    # List all CSV files in the given folder, excluding any with "combined-stats" in the filename
    csv_files = [
        file
        for file in os.listdir(folder_path)
        if file.endswith(".csv") and "combined-stats" not in file
    ]

    # Sort files by date extracted from the filename
    csv_files_sorted = sorted(
        csv_files, key=lambda x: datetime.strptime(x.split("]")[0][1:], "%Y-%m-%d")
    )

    # Total number of files to process
    total_files = len(csv_files_sorted)
    print(f"Total files to process: {total_files}")

    # Iterate over each CSV file and process it
    for csv_file in tqdm.tqdm(csv_files_sorted, desc="Processing files", unit="file"):
        full_file_path = os.path.join(folder_path, csv_file)
        try:
            create_game_state_files(
                full_file_path
            )  # Assume this is a predefined function elsewhere
        except Exception as e:
            print(f"Failed to process file: {csv_file}")
            print(f"Error: {str(e)}")
            traceback.print_exc()

    if total_files == 0:
        print("No files to process.")
    else:
        print("All files processed successfully.")


if __name__ == "__main__":
    # game_state_df = create_game_states(
    #     pd.read_csv(
    #         "../data/2021-2022/2021-2022_NBA_PbP_Logs/[2021-10-19]-0022100001-BKN@MIL.csv"
    #     )
    # )

    # print(game_state_df.info(verbose=True))
    # print(game_state_df.head())
    # print(game_state_df.tail())

    # create_game_state_files(
    #     "../data/2021-2022/2021-2022_NBA_PbP_Logs/[2021-10-23]-0022100029-DAL@TOR.csv"
    # )

    create_game_state_files_by_folder("../data/2022-2023/2022-2023_NBA_PbP_Logs")
