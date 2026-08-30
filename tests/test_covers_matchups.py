"""Regression tests for Covers matchup spread perspective."""

from datetime import date

import pytest

from src.database_updater.covers import _parse_matchups_page


def _matchup_html(summary: str = "", fallback: str = "") -> str:
    summary_html = f'<div class="summary-box">{summary}</div>' if summary else ""
    fallback_html = (
        f'<div class="trending-and-cover-by-container"><span>{fallback}</span></div>'
        if fallback
        else ""
    )
    return f"""
        <article class="gamebox"
                 data-home-team-shortname="mia"
                 data-away-team-shortname="bos">
            <strong class="team-score away">101</strong>
            <strong class="team-score home">105</strong>
            {summary_html}
            {fallback_html}
        </article>
    """


@pytest.mark.parametrize(
    ("summary", "expected_spread", "expected_result"),
    [
        ("The Miami Heat covered the spread of -3.5", -3.5, "W"),
        ("The Boston Celtics covered the spread of -3.5", 3.5, "L"),
        ("Boston Celtics did not cover the spread of +4.5", -4.5, "W"),
        ("The Celtics covered the spread of PK", 0.0, "L"),
    ],
)
def test_summary_spread_is_stored_from_home_perspective(
    summary, expected_spread, expected_result
):
    games = _parse_matchups_page(_matchup_html(summary=summary), date(2026, 1, 1))

    assert len(games) == 1
    assert games[0].spread == expected_spread
    assert games[0].spread_result == expected_result


@pytest.mark.parametrize(
    ("fallback", "expected_spread"),
    [("MIA -3.5", -3.5), ("BOS -3.5", 3.5), ("BOS PK", 0.0)],
)
def test_fallback_spread_retains_team_subject(fallback, expected_spread):
    games = _parse_matchups_page(_matchup_html(fallback=fallback), date(2026, 1, 1))

    assert len(games) == 1
    assert games[0].spread == expected_spread
    assert games[0].spread_result is None


def test_unknown_fallback_subject_is_not_stored():
    games = _parse_matchups_page(
        _matchup_html(fallback="LAL -3.5"), date(2026, 1, 1)
    )

    assert len(games) == 1
    assert games[0].spread is None
    assert games[0].spread_result is None
