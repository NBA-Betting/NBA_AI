"""Tests for the web app entry point."""

import sys
from unittest.mock import Mock


def test_port_argument_is_passed_to_flask(monkeypatch):
    """The CLI port option should override Flask's default port."""
    import start_app

    app = Mock()
    monkeypatch.setattr(start_app, "create_app", lambda predictor: app)
    monkeypatch.setattr(start_app, "setup_logging", lambda log_level: None)
    monkeypatch.setattr(
        sys, "argv", ["start_app.py", "--predictor", "Baseline", "--port", "5001"]
    )

    start_app.main()

    app.run.assert_called_once_with(debug=False, port=5001)
