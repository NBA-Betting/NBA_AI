"""
test_api.py

Tests for the Games API endpoints.
Tests validation, error handling, and response structure.
"""

import pytest


class TestGamesAPIValidation:
    """Tests for /api/games endpoint validation."""

    def test_no_params_returns_error(self, flask_test_client):
        """Request without game_ids or date should return 400."""
        response = flask_test_client.get("/api/games")
        assert response.status_code == 400
        data = response.get_json()
        assert "error" in data
        assert "game_ids" in data["error"] or "date" in data["error"]

    def test_both_params_returns_error(self, flask_test_client):
        """Request with both game_ids and date should return 400."""
        response = flask_test_client.get(
            "/api/games?game_ids=0022400001&date=2024-10-22"
        )
        assert response.status_code == 400
        data = response.get_json()
        assert "error" in data
        assert "not both" in data["error"]

    def test_invalid_game_id_returns_error(self, flask_test_client):
        """Invalid game ID should return 400."""
        response = flask_test_client.get("/api/games?game_ids=invalid")
        assert response.status_code == 400
        data = response.get_json()
        assert "error" in data

    def test_invalid_date_format_returns_error(self, flask_test_client):
        """Invalid date format should return 400."""
        response = flask_test_client.get("/api/games?date=10-22-2024")
        assert response.status_code == 400
        data = response.get_json()
        assert "error" in data

    def test_invalid_predictor_returns_error(self, flask_test_client):
        """Invalid predictor should return 400."""
        response = flask_test_client.get(
            "/api/games?date=2024-10-22&predictor=InvalidPredictor"
        )
        assert response.status_code == 400
        data = response.get_json()
        assert "error" in data
        assert "Invalid predictor" in data["error"]

    def test_too_many_game_ids_returns_error(self, flask_test_client):
        """Too many game IDs should return 400."""
        from src.config import config

        max_ids = config["api"]["max_game_ids"]

        # Generate more than max_game_ids
        game_ids = ",".join([f"002240{str(i).zfill(4)}" for i in range(max_ids + 5)])

        response = flask_test_client.get(f"/api/games?game_ids={game_ids}")
        assert response.status_code == 400
        data = response.get_json()
        assert "error" in data
        assert "Too many game IDs" in data["error"]

    def test_invalid_season_returns_error(self, flask_test_client):
        """Game ID from invalid season should return 400."""
        # 2023-2024 season game (may not be in valid_seasons for public release)
        response = flask_test_client.get("/api/games?game_ids=0021900001")
        assert response.status_code == 400
        data = response.get_json()
        assert "error" in data
        assert "valid seasons" in data["error"]


class TestGamesAPIResponses:
    """Tests for /api/games endpoint responses (requires database)."""

    def test_valid_date_returns_dict(self, flask_test_client, sample_date):
        """Valid date should return a dict of games keyed by game_id or season error."""
        response = flask_test_client.get(f"/api/games?date={sample_date}")

        # May return 200 with games, 200 with empty dict, or 400 if season not valid
        # The season restriction is configurable, so both are acceptable
        assert response.status_code in (200, 400)
        data = response.get_json()
        if response.status_code == 200:
            assert isinstance(data, dict)
        else:
            assert "error" in data

    def test_response_structure(self, flask_test_client, sample_date):
        """Response should have expected game structure if data available."""
        response = flask_test_client.get(f"/api/games?date={sample_date}")

        if response.status_code == 200:
            data = response.get_json()
            if len(data) > 0:
                # Games are keyed by game_id
                game_id = list(data.keys())[0]
                game = data[game_id]
                # Check required fields
                assert "home_team" in game
                assert "away_team" in game


class TestWebAppRoutes:
    """Tests for web app routes."""

    def test_home_page_renders(self, flask_test_client):
        """Home page should render successfully."""
        response = flask_test_client.get("/")
        assert response.status_code == 200
        # Check it returns HTML
        assert b"<!DOCTYPE html>" in response.data or b"<html" in response.data

    def test_home_page_with_date(self, flask_test_client, sample_date):
        """Home page with date param should render."""
        response = flask_test_client.get(f"/?date={sample_date}")
        assert response.status_code == 200

    def test_home_page_invalid_date_flashes(self, flask_test_client):
        """Invalid date should still render (with flash message)."""
        response = flask_test_client.get("/?date=invalid")
        assert response.status_code == 200  # Should render with error message

    def test_get_game_data_no_params(self, flask_test_client):
        """get-game-data without params should return 400."""
        response = flask_test_client.get("/get-game-data")
        assert response.status_code == 400

    def test_get_game_data_with_valid_game_id(self, flask_test_client):
        """get-game-data with valid game_id should return 200."""
        # Use a known game_id from 2024-2025 season
        response = flask_test_client.get("/get-game-data?game_id=0022400415")
        assert response.status_code == 200
        data = response.get_json()
        assert isinstance(data, list)
        assert len(data) == 1

    def test_get_game_data_with_invalid_game_id(self, flask_test_client):
        """get-game-data with invalid game_id should return 400."""
        response = flask_test_client.get("/get-game-data?game_id=invalid")
        assert response.status_code == 400
        data = response.get_json()
        assert "error" in data

    def test_get_game_data_with_empty_game_id(self, flask_test_client):
        """get-game-data with empty game_id should return 400."""
        response = flask_test_client.get("/get-game-data?game_id=")
        assert response.status_code == 400


class TestValidSeasons:
    """Tests for season validation surviving a season rollover."""

    def test_current_season_is_valid(self):
        """The current season should be valid even if config.yaml predates it."""
        from src.games_api.api import get_valid_seasons
        from src.utils import determine_current_season

        assert determine_current_season() in get_valid_seasons()

    def test_previous_season_is_valid(self):
        """The season before the current one should stay valid."""
        from src.games_api.api import get_valid_seasons
        from src.utils import determine_current_season

        previous_start_year = int(determine_current_season().split("-")[0]) - 1
        assert f"{previous_start_year}-{previous_start_year + 1}" in get_valid_seasons()

    def test_configured_seasons_are_preserved(self):
        """Seasons listed in config.yaml should remain valid."""
        from src.config import config
        from src.games_api.api import get_valid_seasons

        valid_seasons = get_valid_seasons()
        for season in config["api"]["valid_seasons"]:
            assert season in valid_seasons

    def test_date_in_new_season_accepted_after_rollover(
        self, monkeypatch, flask_test_client
    ):
        """A date in a newly started season should not be rejected as invalid.

        Regression test: valid_seasons was read from config.yaml only, so every
        date in a new season returned 400 until the file was edited by hand.
        """
        import src.games_api.api as api_module

        monkeypatch.setattr(api_module, "determine_current_season", lambda: "2030-2031")

        response = flask_test_client.get("/api/games?date=2030-12-25")
        assert response.status_code == 200

    def test_game_id_in_new_season_accepted_after_rollover(
        self, monkeypatch, flask_test_client
    ):
        """A game ID from a newly started season should not be rejected."""
        import src.games_api.api as api_module

        monkeypatch.setattr(api_module, "determine_current_season", lambda: "2030-2031")

        response = flask_test_client.get("/api/games?game_ids=0023000001")
        assert response.status_code == 200

    def test_season_outside_window_still_rejected(self, flask_test_client):
        """Seasons with no data should still return a 400 rather than empty results."""
        response = flask_test_client.get("/api/games?date=2019-12-25")
        assert response.status_code == 400
        data = response.get_json()
        assert "valid seasons" in data["error"]
class TestGameDatesEndpoint:
    """Tests for /game-dates, which drives the calendar date picker."""

    def test_returns_counts_for_month(self, flask_test_client):
        """Should return a mapping of dates in the month to game counts."""
        response = flask_test_client.get("/game-dates?month=2026-01")
        assert response.status_code == 200

        data = response.get_json()
        assert isinstance(data, dict)
        for date_str, count in data.items():
            assert date_str.startswith("2026-01")
            assert len(date_str) == 10
            assert isinstance(count, int)
            assert count > 0

    def test_defaults_to_current_month(self, flask_test_client):
        """Omitting the month should fall back to the current month."""
        response = flask_test_client.get("/game-dates")
        assert response.status_code == 200
        assert isinstance(response.get_json(), dict)

    def test_month_without_games_is_empty(self, flask_test_client):
        """A month with no games should return an empty mapping, not an error."""
        response = flask_test_client.get("/game-dates?month=1990-01")
        assert response.status_code == 200
        assert response.get_json() == {}

    @pytest.mark.parametrize("bad_month", ["bogus", "2026-13", "2026", "2026-01-15"])
    def test_invalid_month_returns_error(self, flask_test_client, bad_month):
        """A malformed month should return 400 rather than a stack trace."""
        response = flask_test_client.get(f"/game-dates?month={bad_month}")
        assert response.status_code == 400
        assert "error" in response.get_json()

    def test_counts_match_games_for_date(self, flask_test_client):
        """The calendar must agree with the table it sends the user to.

        Both bucket games by Eastern Time date; if they ever diverge, a day
        would show a dot and then load an empty table.
        """
        from src.games_api.games import get_games_for_date

        counts = flask_test_client.get("/game-dates?month=2026-01").get_json()

        for date_str, count in sorted(counts.items())[:3]:
            games = get_games_for_date(date_str, predictor="Baseline")
            assert len(games) == count, f"mismatch on {date_str}"
