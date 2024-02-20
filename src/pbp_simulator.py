import csv
import os
import shutil
import time

import pandas as pd
from dotenv import load_dotenv

load_dotenv()
PROJECT_ROOT = os.getenv("PROJECT_ROOT")


class NBASimulator:
    def __init__(self, season, game_id):
        self.season = self._validate_season_format(season)
        self.game_id = self._validate_game_id_format(game_id)
        self.load_data()
        self.prepare_data()
        self.current_play_index = 0
        self.plays_seen = []

    def __del__(self):
        """Cleanup method to remove temporary and output files when the instance is destroyed."""
        try:
            if os.path.exists(self.temp_file_path):
                os.remove(self.temp_file_path)
            if os.path.exists(self.output_file_path):
                os.remove(self.output_file_path)
            print("Temporary and output files have been removed.")
        except Exception as e:
            print(f"Error removing files: {e}")

    def _validate_season_format(self, season):
        if (
            len(season) != 9
            or season[4] != "-"
            or not season[:4].isdigit()
            or not season[5:].isdigit()
        ):
            raise ValueError(
                "Invalid season format. Please use the format 'XXXX-XXXX', e.g. 2021-2022"
            )
        return season

    def _validate_game_id_format(self, game_id):
        if not isinstance(game_id, int) or (
            len(str(game_id)) != 8
            and (len(str(game_id)) != 10 or str(game_id)[:2] != "00")
        ):
            raise ValueError(
                "Invalid game_id format. Please use an 8-digit integer or a 10-digit integer starting with '00'"
            )
        if len(str(game_id)) == 8:
            game_id = f"00{game_id}"
        return game_id

    def _set_file_names(self):
        self.base_file_name = f"{self.game_id}_{self.home}_{self.away}.csv"
        self.output_file_path = self.base_file_name
        self.temp_file_path = f"temp_{self.base_file_name}"

    def load_data(self):
        """Load and merge data from various sources into a single DataFrame."""
        # Define paths to data folders
        season_folder = f"{PROJECT_ROOT}/data/{self.season}/"
        pbp_folder = f"{season_folder}{self.season}_NBA_PbP_Logs/"
        game_state_folder = f"{season_folder}{self.season}_Game_States/"
        final_state_folder = f"{season_folder}{self.season}_Final_States/"
        prior_state_folder = f"{season_folder}{self.season}_Prior_States/"

        # Lookup filenames for the specified game_id
        game_id_str = str(self.game_id)
        matching_files = [
            filename
            for filename in os.listdir(pbp_folder)
            if f"-{game_id_str}-" in filename
        ]
        if len(matching_files) == 0:
            raise FileNotFoundError(
                f"No matching PbP file found for game_id {game_id_str}"
            )
        elif len(matching_files) > 1:
            raise ValueError(
                f"Multiple matching PbP files found for game_id {game_id_str}. Please specify a unique game_id."
            )
        else:
            pbp_filename = os.path.join(pbp_folder, matching_files[0])

        pbp_basename = os.path.basename(pbp_filename)

        # Define file paths
        game_states_filename = os.path.join(
            game_state_folder, pbp_basename.replace(".csv", "_game_states.csv")
        )
        final_state_filename = os.path.join(
            final_state_folder, pbp_basename.replace(".csv", "_final_state.csv")
        )
        prior_state_filename = os.path.join(
            prior_state_folder, pbp_basename.replace(".csv", "_prior_state.csv")
        )

        # Load each DataFrame, setting to None if file not found
        self.pbp_df = self._load_dataframe(pbp_filename)
        self.game_states_df = self._load_dataframe(game_states_filename)
        self.final_state_df = self._load_dataframe(final_state_filename)
        self.prior_state_df = self._load_dataframe(prior_state_filename)

        self.date = self.pbp_df.iloc[0]["date"]
        self.home = pbp_filename[
            pbp_filename.index("@") + 1 : pbp_filename.index("@") + 4
        ]
        self.away = pbp_filename[pbp_filename.index("@") - 3 : pbp_filename.index("@")]

        self._set_file_names()

    def _load_dataframe(self, filename):
        """Helper function to load a dataframe from a CSV file.
        Sets to None if file not found."""
        try:
            df = pd.read_csv(filename)
            df = df.fillna("")
            return df
        except pd.errors.EmptyDataError:
            print(f"Warning: {filename} is empty")
            return None
        except FileNotFoundError:
            print(f"Warning: {filename} Not Found")
            return None

    def prepare_data(self):
        # Step 1: Merge self.pbp_df and self.game_states_df on play_id, keeping self.pbp_df's columns in case of duplicates.
        merged_df = pd.merge(
            self.pbp_df, self.game_states_df, on="play_id", suffixes=("", "_drop")
        )
        merged_df = merged_df[
            [col for col in merged_df.columns if not col.endswith("_drop")]
        ]

        # Predefined columns to keep
        columns_to_keep = [
            "game_id",
            "date",
            "play_id",
            "home_score",
            "away_score",
            "period",
            "remaining_time",
            "home_margin",
            "total",
            "description",
        ]

        # Dynamically find columns that end with the specified suffixes
        dynamic_columns = [
            col
            for col in merged_df.columns
            if col.endswith(("_name", "_id", "_pts", "_pm"))
        ]

        # Combine both lists, ensuring no duplicates
        all_columns_to_keep = list(dict.fromkeys(columns_to_keep + dynamic_columns))

        # Select and reorder columns in the DataFrame, dropping all others
        merged_df = merged_df[all_columns_to_keep]

        # Step 3: Add 'home' and 'away' columns between 'date' and 'play_id'.
        # This involves reordering columns as well.
        merged_df.insert(2, "home", self.home)
        merged_df.insert(3, "away", self.away)

        self.main_df = merged_df.sort_values("play_id")

    def get_next_play(self):
        """Return the next play in the sequence and update self.plays_seen."""
        if self.current_play_index < len(self.main_df):
            play = self.main_df.iloc[self.current_play_index]
            self.current_play_index += 1
            self.plays_seen.append(play.to_dict())
            return self.plays_seen, play
        else:
            print("End of game reached.")
            return self.plays_seen, None

    def simulate_game(self, delay=1, print_plays=True, save_plays=False):
        """Simulate the game, allowing for printing play-by-play data, saving to a file, or both."""
        for i in range(self.current_play_index, len(self.main_df)):
            _, play = (
                self.get_next_play()
            )  # Assuming this updates self.plays_seen and returns index and play data
            if print_plays:
                remaining_time = play["remaining_time"][-5:]
                play_data_formatted = (
                    f"{play['play_id']:<3} | {remaining_time:<5} {play['period']}Q | "
                    f"{play['home']:<3} {play['home_score']:<3} - "
                    f"{play['away']:<3} {play['away_score']:<3} | "
                    f"{play['description']}"
                )
                print(play_data_formatted)

            if save_plays:
                # Sort self.plays_seen by play_id in descending order before saving
                plays_seen_sorted = sorted(
                    self.plays_seen, key=lambda x: x["play_id"], reverse=True
                )

                # Determine column headers from the keys of the first play
                headers = list(plays_seen_sorted[0].keys())

                with open(self.temp_file_path, "w", newline="") as file:
                    writer = csv.DictWriter(file, fieldnames=headers)
                    writer.writeheader()
                    for play in plays_seen_sorted:
                        writer.writerow(play)

                # Atomic move to ensure the file is not partially written
                shutil.move(self.temp_file_path, self.output_file_path)

            time.sleep(delay)


if __name__ == "__main__":
    season = "2021-2022"
    game_id = 22100001
    simulator = NBASimulator(season, game_id)
    simulator.simulate_game(delay=0.1, save_plays=True)
