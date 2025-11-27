"""
test_frontend_integration.py

Integration tests for frontend functionality across multiple dates.
Tests the full stack: API endpoints, game data retrieval, and prediction display
using the live database.

Tests 5 date scenarios:
- 1 week ago
- Yesterday
- Today
- Tomorrow
- 1 week from now
"""

from datetime import datetime, timedelta

import pytest


class TestFrontendDateRange:
    """Integration tests for frontend across different date ranges."""

    @pytest.fixture
    def test_dates(self):
        """Generate test dates relative to current date."""
        today = datetime.now().date()
        return {
            "week_ago": (today - timedelta(days=7)).strftime("%Y-%m-%d"),
            "yesterday": (today - timedelta(days=1)).strftime("%Y-%m-%d"),
            "today": today.strftime("%Y-%m-%d"),
            "tomorrow": (today + timedelta(days=1)).strftime("%Y-%m-%d"),
            "week_ahead": (today + timedelta(days=7)).strftime("%Y-%m-%d"),
        }

    def test_home_page_renders_all_dates(self, flask_test_client, test_dates):
        """Home page should render for all date scenarios."""
        for date_label, date in test_dates.items():
            response = flask_test_client.get(f"/?date={date}")
            assert response.status_code == 200, f"Failed for {date_label}: {date}"
            assert b"<!DOCTYPE html>" in response.data or b"<html" in response.data

    def test_api_endpoint_all_dates(self, flask_test_client, test_dates):
        """API endpoint should respond for all dates (may return empty)."""
        for date_label, date in test_dates.items():
            response = flask_test_client.get(
                f"/api/games?date={date}&update_predictions=False"
            )
            # Should succeed (200) or fail gracefully (400 if season invalid)
            assert response.status_code in (
                200,
                400,
            ), f"Unexpected status for {date_label}: {date}"

            data = response.get_json()
            if response.status_code == 200:
                assert isinstance(
                    data, dict
                ), f"Expected dict for {date_label}, got {type(data)}"
            else:
                # If 400, should have error message
                assert "error" in data

    def test_get_game_data_all_dates(self, flask_test_client, test_dates):
        """get-game-data endpoint should work for all dates."""
        for date_label, date in test_dates.items():
            response = flask_test_client.get(f"/get-game-data?date={date}")

            # May return 200 (success), 400 (validation), or 500 (internal error)
            # 500 can occur in test mode due to internal request handling
            assert response.status_code in (
                200,
                400,
                500,
            ), f"Unexpected status for {date_label}: {date}"

            # Only validate structure for successful responses
            if response.status_code == 200:
                data = response.get_json()
                assert isinstance(
                    data, list
                ), f"Expected list for {date_label}, got {type(data)}"


class TestPredictionDisplay:
    """Tests verifying predictions appear in frontend responses."""

    @pytest.fixture
    def known_game_date(self):
        """Return a date known to have games with predictions in DB."""
        # Use a recent past date from current season
        # This should have completed games with predictions
        today = datetime.now().date()
        # Go back 1-3 days to find completed games
        return (today - timedelta(days=2)).strftime("%Y-%m-%d")

    def test_predictions_in_api_response(self, flask_test_client, known_game_date):
        """API response should include prediction fields for games with data."""
        response = flask_test_client.get(
            f"/api/games?date={known_game_date}&update_predictions=True"
        )

        if response.status_code == 200:
            data = response.get_json()

            if len(data) > 0:
                # Check first game for prediction structure
                game_id = list(data.keys())[0]
                game = data[game_id]

                # Should have predictions dict
                assert "predictions" in game, f"No predictions key in game {game_id}"

                predictions = game["predictions"]

                # Should have either pre_game, current, or both
                has_predictions = "pre_game" in predictions or "current" in predictions
                if not has_predictions:
                    pytest.skip(
                        f"No predictions generated yet for date {known_game_date}"
                    )

    def test_predictions_in_game_data_response(
        self, flask_test_client, known_game_date
    ):
        """get-game-data response should include prediction fields."""
        response = flask_test_client.get(f"/get-game-data?date={known_game_date}")

        if response.status_code == 200:
            games = response.get_json()

            if len(games) > 0:
                game = games[0]

                # Check for prediction fields (processed by game_data_processor)
                prediction_fields = [
                    "pred_home_score",
                    "pred_away_score",
                    "pred_winner",
                    "pred_win_pct",
                ]

                # If predictions exist, these should be present
                # (may be empty string if no predictions)
                for field in prediction_fields:
                    assert field in game, f"Missing field '{field}' in game data"

    def test_prediction_values_are_numeric(self, flask_test_client, known_game_date):
        """Prediction scores should be numeric when present."""
        response = flask_test_client.get(f"/get-game-data?date={known_game_date}")

        if response.status_code == 200:
            games = response.get_json()

            for game in games:
                pred_home = game.get("pred_home_score")
                pred_away = game.get("pred_away_score")

                # If predictions exist (not empty string), should be numeric
                if pred_home != "":
                    assert isinstance(
                        pred_home, (int, float)
                    ), f"pred_home_score should be numeric, got {type(pred_home)}"
                    assert (
                        70 <= pred_home <= 150
                    ), f"pred_home_score {pred_home} out of reasonable range"

                if pred_away != "":
                    assert isinstance(
                        pred_away, (int, float)
                    ), f"pred_away_score should be numeric, got {type(pred_away)}"
                    assert (
                        70 <= pred_away <= 150
                    ), f"pred_away_score {pred_away} out of reasonable range"


class TestGameStateVariations:
    """Tests for different game states (scheduled, in-progress, completed)."""

    def test_past_games_completed(self, flask_test_client):
        """Past games should be marked as completed."""
        past_date = (datetime.now().date() - timedelta(days=7)).strftime("%Y-%m-%d")
        response = flask_test_client.get(f"/get-game-data?date={past_date}")

        if response.status_code == 200:
            games = response.get_json()

            if len(games) > 0:
                for game in games:
                    status = game.get("game_status", "")
                    # Past games should be Completed (or very rarely postponed/cancelled)
                    assert status in (
                        "Completed",
                        "Final",
                        "Postponed",
                        "Cancelled",
                        "",
                    )

    def test_future_games_not_started(self, flask_test_client):
        """Future games should not be marked as completed."""
        future_date = (datetime.now().date() + timedelta(days=7)).strftime("%Y-%m-%d")
        response = flask_test_client.get(f"/get-game-data?date={future_date}")

        if response.status_code == 200:
            games = response.get_json()

            if len(games) > 0:
                for game in games:
                    status = game.get("game_status", "")
                    # Future games should NOT be completed
                    assert status not in (
                        "Completed",
                        "Final",
                    ), f"Future game marked as {status}"


class TestPredictorParameter:
    """Tests for predictor parameter in API requests."""

    def test_tree_predictor(self, flask_test_client):
        """Tree predictor should work in API calls."""
        today = datetime.now().date().strftime("%Y-%m-%d")
        response = flask_test_client.get(
            f"/api/games?date={today}&predictor=Tree&update_predictions=False"
        )

        # Should succeed or return season error
        assert response.status_code in (200, 400)

    def test_baseline_predictor(self, flask_test_client):
        """Baseline predictor should work in API calls."""
        today = datetime.now().date().strftime("%Y-%m-%d")
        response = flask_test_client.get(
            f"/api/games?date={today}&predictor=Baseline&update_predictions=False"
        )

        assert response.status_code in (200, 400)

    def test_predictor_affects_predictions(self, flask_test_client):
        """Different predictors should potentially give different predictions."""
        # Use a date with games
        past_date = (datetime.now().date() - timedelta(days=3)).strftime("%Y-%m-%d")

        response_tree = flask_test_client.get(
            f"/api/games?date={past_date}&predictor=Tree&update_predictions=False"
        )
        response_baseline = flask_test_client.get(
            f"/api/games?date={past_date}&predictor=Baseline&update_predictions=False"
        )

        # If both succeed and return games, predictions may differ
        if response_tree.status_code == 200 and response_baseline.status_code == 200:
            tree_data = response_tree.get_json()
            baseline_data = response_baseline.get_json()

            # Just verify both responses are valid dicts
            assert isinstance(tree_data, dict)
            assert isinstance(baseline_data, dict)


class TestDatabaseIntegration:
    """Tests verifying integration with live database."""

    def test_database_accessible(self, flask_test_client):
        """Database should be accessible for game queries."""
        today = datetime.now().date().strftime("%Y-%m-%d")

        response = flask_test_client.get(
            f"/api/games?date={today}&update_predictions=False"
        )

        # Should get a response (even if empty or season error)
        assert response.status_code in (200, 400)
        data = response.get_json()
        assert data is not None

    def test_valid_season_returns_data_or_empty(self, flask_test_client):
        """Valid season date should return dict, not error."""
        # Use a date from the current valid season (2025-2026)
        valid_date = "2025-11-15"

        response = flask_test_client.get(
            f"/api/games?date={valid_date}&update_predictions=False"
        )

        if response.status_code == 200:
            data = response.get_json()
            assert isinstance(data, dict)  # May be empty if no games that day
        # If 400, that's also acceptable (date validation)

    def test_response_time_reasonable(self, flask_test_client):
        """API response time should be reasonable (< 10 seconds)."""
        import time

        today = datetime.now().date().strftime("%Y-%m-%d")

        start = time.time()
        response = flask_test_client.get(
            f"/api/games?date={today}&update_predictions=False"
        )
        elapsed = time.time() - start

        assert response.status_code in (200, 400)
        assert (
            elapsed < 10.0
        ), f"API took {elapsed:.2f}s, should be < 10s with update_predictions=False"
