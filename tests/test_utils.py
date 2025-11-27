"""
test_utils.py

Tests for utility functions in src/utils.py.
These are critical validation functions used throughout the pipeline.
"""

import pytest

from src.utils import (
    date_to_season,
    determine_current_season,
    game_id_to_season,
    validate_date_format,
    validate_game_ids,
    validate_season_format,
)


class TestValidateGameIds:
    """Tests for validate_game_ids function."""

    def test_valid_single_game_id(self):
        """Valid single game ID should not raise."""
        validate_game_ids("0022400001")  # Should not raise

    def test_valid_list_game_ids(self):
        """Valid list of game IDs should not raise."""
        validate_game_ids(["0022400001", "0022400002", "0022400003"])

    def test_invalid_too_short(self):
        """Game ID that is too short should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid game IDs"):
            validate_game_ids("00224")

    def test_invalid_too_long(self):
        """Game ID that is too long should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid game IDs"):
            validate_game_ids("00224000011")

    def test_invalid_wrong_prefix(self):
        """Game ID with wrong prefix should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid game IDs"):
            validate_game_ids("1022400001")

    def test_invalid_non_string(self):
        """Non-string game ID should raise TypeError (not iterable)."""
        with pytest.raises(TypeError):
            validate_game_ids(22400001)

    def test_mixed_valid_invalid(self):
        """Mix of valid and invalid IDs should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid game IDs"):
            validate_game_ids(["0022400001", "invalid", "0022400003"])


class TestValidateDateFormat:
    """Tests for validate_date_format function."""

    def test_valid_date(self):
        """Valid date should not raise."""
        validate_date_format("2024-10-22")  # Should not raise

    def test_invalid_format_slashes(self):
        """Date with slashes should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid date format"):
            validate_date_format("2024/10/22")

    def test_invalid_format_no_dashes(self):
        """Date without dashes should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid date format"):
            validate_date_format("20241022")

    def test_invalid_month_13(self):
        """Month 13 should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid month"):
            validate_date_format("2024-13-01")

    def test_invalid_day_32(self):
        """Day 32 should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid day"):
            validate_date_format("2024-01-32")

    def test_invalid_day_feb_30(self):
        """February 30th should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid day"):
            validate_date_format("2024-02-30")


class TestValidateSeasonFormat:
    """Tests for validate_season_format function."""

    def test_valid_full_season(self):
        """Valid full season format should not raise."""
        validate_season_format("2024-2025")  # Should not raise

    def test_valid_abbreviated_season(self):
        """Valid abbreviated season format should not raise."""
        validate_season_format("2024-25", abbreviated=True)

    def test_invalid_not_consecutive(self):
        """Non-consecutive years should raise ValueError."""
        with pytest.raises(ValueError, match="does not logically follow"):
            validate_season_format("2024-2026")

    def test_invalid_format(self):
        """Invalid format should raise ValueError."""
        with pytest.raises(ValueError, match="does not match the required format"):
            validate_season_format("2024")


class TestGameIdToSeason:
    """Tests for game_id_to_season function."""

    def test_2024_season(self):
        """2024-2025 season game ID should return correct season."""
        assert game_id_to_season("0022400001") == "2024-2025"

    def test_2023_season(self):
        """2023-2024 season game ID should return correct season."""
        assert game_id_to_season("0022300001") == "2023-2024"

    def test_abbreviated(self):
        """Abbreviated flag should return shortened second year."""
        assert game_id_to_season("0022400001", abbreviate=True) == "2024-25"

    def test_playoff_game(self):
        """Playoff game ID should return correct season."""
        # Playoff games start with 004
        assert game_id_to_season("0042300401") == "2023-2024"


class TestDateToSeason:
    """Tests for date_to_season function."""

    def test_october_start(self):
        """October date should return upcoming season."""
        assert date_to_season("2024-10-22") == "2024-2025"

    def test_june_end(self):
        """June date should return current season."""
        assert date_to_season("2024-06-15") == "2023-2024"

    def test_july_start(self):
        """July date should return upcoming season."""
        assert date_to_season("2024-07-15") == "2024-2025"


class TestDetermineCurrentSeason:
    """Tests for determine_current_season function."""

    def test_returns_valid_format(self):
        """Should return a valid season format."""
        season = determine_current_season()
        # Should match pattern YYYY-YYYY
        assert len(season) == 9
        assert season[4] == "-"
        year1, year2 = season.split("-")
        assert int(year2) == int(year1) + 1
