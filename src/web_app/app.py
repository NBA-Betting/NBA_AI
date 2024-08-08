"""
app.py

This module sets up a Flask web application to display NBA game data, including game schedules, team details, and predictions. 
It integrates with external APIs to fetch and process data, and utilizes a machine learning predictor for game predictions.

Core Functions:
- create_app(predictor): Initializes and configures the Flask application, including setting up routes and the app secret key.

Routes:
- home(): Renders the home page with the NBA game schedule for a specific date.
- get_game_data(): Fetches game data for a given date or game ID and processes it for display.

Helper Functions:
- add_header(response): Adds headers to the response to prevent caching of the pages.

Usage:
Typically run via a entry point in the root directory of the project.
"""

from datetime import datetime, timedelta

import requests
from flask import Flask, flash, jsonify, render_template, request, url_for

from src.config import config
from src.games_api.api import api as api_blueprint
from src.utils import validate_date_format
from src.web_app.game_data_processor import get_user_datetime, process_game_data

# Configuration variables
DB_PATH = config["database"]["path"]
WEB_APP_SECRET_KEY = config["web_app"]["secret_key"]


def create_app(predictor):
    """
    Initializes and configures the Flask application.

    Args:
        predictor (str): A predictor used for generating game predictions.

    Returns:
        Flask: The configured Flask application instance.
    """
    app = Flask(__name__)
    app.secret_key = WEB_APP_SECRET_KEY

    # Store the predictor in the app configuration
    app.config["PREDICTOR"] = predictor

    # Register the API blueprint
    app.register_blueprint(api_blueprint, url_prefix="/api")

    @app.route("/")
    def home():
        """
        Renders the home page with NBA game schedule and details for a specific date.

        - Defaults to the current date if no date is provided or if an invalid date is entered.
        - Displays a list of games, including links to detailed game information.

        Returns:
            str: Rendered HTML page of the home screen with games table.
        """
        current_date_local = get_user_datetime(as_eastern_tz=False)
        current_date_str = current_date_local.strftime("%Y-%m-%d")
        query_date_str = request.args.get("date", current_date_str)

        try:
            validate_date_format(query_date_str)
            query_date = datetime.strptime(query_date_str, "%Y-%m-%d")
        except Exception as e:
            flash("Invalid date format. Showing games for today.", "error")
            query_date_str = current_date_str
            query_date = current_date_local

        query_date_display_str = query_date.strftime("%b %d")
        next_date = query_date + timedelta(days=1)
        prev_date = query_date - timedelta(days=1)
        next_date_str = next_date.strftime("%Y-%m-%d")
        prev_date_str = prev_date.strftime("%Y-%m-%d")

        return render_template(
            "index.html",
            query_date_str=query_date_str,
            query_date_display_str=query_date_display_str,
            prev_date=prev_date_str,
            next_date=next_date_str,
        )

    @app.route("/get-game-data")
    def get_game_data():
        """
        Fetches and processes game data for a given date or game ID.

        - Supports querying by either 'date' or 'game_id'.
        - Retrieves game data from an external API and processes it for display.

        Returns:
            Response: JSON response containing processed game data or error message.
        """
        try:
            # Determine the type of input (date or game_id)
            if "date" in request.args:
                # Use provided date or default to the current date if not provided
                inbound_query_date_str = request.args.get("date")
                if inbound_query_date_str is None or inbound_query_date_str == "":
                    current_date_local = get_user_datetime(as_eastern_tz=False)
                    query_date_str = current_date_local.strftime("%Y-%m-%d")
                else:
                    query_date_str = inbound_query_date_str

                api_param_kv = {"date": query_date_str, "detail_level": "Basic"}

            elif "game_id" in request.args:
                game_id = request.args.get("game_id")
                api_param_kv = {"game_ids": game_id, "detail_level": "Normal"}

            else:
                return (
                    jsonify({"error": "Either 'date' or 'game_id' must be provided."}),
                    400,
                )

            # Fetch data from the API endpoint using url_for
            api_url = url_for("api.games", _external=True)
            predictor = app.config["PREDICTOR"]  # Get the predictor from the config
            params = {
                "predictor": predictor,
                "update_predictions": "True",
            }
            params.update(api_param_kv)

            response = requests.get(api_url, params=params)

            # Check if the response indicates an error
            if response.status_code != 200:
                return (
                    jsonify({"error": response.json().get("error", "Unknown error")}),
                    response.status_code,
                )

            game_data = response.json()
            outbound_game_data = process_game_data(game_data)

            return jsonify(outbound_game_data)

        except requests.RequestException as e:
            return (
                jsonify(
                    {
                        "error": f"Unable to fetch game data due to a request error: {str(e)}"
                    }
                ),
                500,
            )

    @app.after_request
    def add_header(response):
        """
        Adds headers to the response to prevent caching of the pages.

        Args:
            response (Response): The HTTP response object.

        Returns:
            Response: The modified response object with added headers.
        """
        response.headers["Cache-Control"] = "no-store"
        return response

    return app
