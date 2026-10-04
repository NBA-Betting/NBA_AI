"""Preseason games stay visible without presenting unsupported forecasts."""

import pytest

from src.web_app.game_data_processor import process_game_data


def _game(season_type):
    return {
        "date_time_utc": "2026-10-03T23:00:00Z",
        "home_team": "BOS",
        "away_team": "NYK",
        "status": 1,
        "status_text": "7:00 pm ET",
        "season_type": season_type,
        "predictions": {
            "pre_game": {
                "prediction_set": {
                    "pred_home_score": 115,
                    "pred_away_score": 109,
                    "pred_home_win_pct": 0.7,
                    "pred_players": {"home": {"123": {"pred_points": 20}}},
                }
            }
        },
    }


def test_preseason_hides_existing_forecasts_and_explains_unavailability():
    displayed = process_game_data(
        {"0012600001": _game("Pre Season")}, user_tz="America/New_York"
    )[0]

    assert displayed["home"] == "BOS"
    assert displayed["away"] == "NYK"
    assert displayed["season_type"] == "Pre Season"
    assert displayed["prediction_unavailable_reason"] == "Preseason predictions unavailable"
    for field in (
        "pred_home_score", "pred_away_score", "pred_winner", "pred_win_pct", "pred_spread"
    ):
        assert displayed[field] == ""
    assert displayed["home_players"] == []
    assert displayed["pred_winner_correct"] is None


@pytest.mark.parametrize("season_type", ["Regular Season", "Post Season"])
def test_supported_games_keep_their_forecasts(season_type):
    displayed = process_game_data(
        {"0022600001": _game(season_type)}, user_tz="America/New_York"
    )[0]

    assert displayed["prediction_unavailable_reason"] == ""
    assert displayed["pred_home_score"] == 115
    assert displayed["pred_away_score"] == 109
    assert displayed["pred_winner"] == "BOS"
    assert displayed["pred_win_pct"] == "70%"
    assert displayed["pred_spread"] == "BOS by 6.0"
    assert displayed["home_players"][0]["pred_points"] == 20
