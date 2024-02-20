import re
import warnings

import pandas as pd


class NBAPlayerMapper:
    def __init__(self, csv_path):
        self.data = pd.read_csv(csv_path)
        # Normalize names and create mappings
        self.data["normalized_name"] = self.data["full_name"].apply(self.normalize_name)
        # Mapping player names and normalized names to IDs, considering possible duplicates
        self.name_to_ids_map = (
            self.data.groupby("full_name")["id"].apply(list).to_dict()
        )
        self.normalized_name_to_ids_map = (
            self.data.groupby("normalized_name")["id"].apply(list).to_dict()
        )
        self.id_to_names_map = (
            self.data.groupby("id")["full_name"].apply(list).to_dict()
        )

    @staticmethod
    def normalize_name(name):
        """Normalize a name for matching."""
        return re.sub(r"\W", "", name).lower()

    def name_to_id(self, player_name):
        """Find player ID(s) using a multi-level matching system, with error handling for duplicates."""
        # Exact match
        if player_name in self.name_to_ids_map:
            ids = self.name_to_ids_map[player_name]
            if len(ids) > 1:
                print(f"Duplicate IDs found for '{player_name}': {ids}")
                return None
            return ids[0]

        # Normalized match
        normalized_name = self.normalize_name(player_name)
        if normalized_name in self.normalized_name_to_ids_map:
            ids = self.normalized_name_to_ids_map[normalized_name]
            if len(ids) > 1:
                print(
                    f"Duplicate IDs found for '{player_name} > {normalized_name}': {ids}"
                )
                return None
            return ids[0]

        # If no match found
        return None

    def id_to_name(self, player_id):
        """Return player's full name(s) given their ID, with error handling for inconsistencies."""
        if player_id in self.id_to_names_map:
            names = self.id_to_names_map[player_id]
            if len(names) > 1:
                raise ValueError(f"Duplicate names found for ID '{player_id}': {names}")
            return names[0]
        raise ValueError(f"No player found for ID '{player_id}'")
