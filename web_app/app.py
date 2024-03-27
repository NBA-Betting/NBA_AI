import logging
import os
from datetime import datetime, timedelta

from dotenv import load_dotenv
from flask import Flask, flash, jsonify, render_template, request

from src.game_info import get_games_info
from src.predictions import get_predictions
from src.utils import get_games_for_date, update_scheduled_games, validate_date_format

from .web_app_utils import get_user_datetime, load_models, process_games_data

load_dotenv()
PROJECT_ROOT = os.getenv("PROJECT_ROOT")
WEB_APP_SECRET_KEY = os.getenv("WEB_APP_SECRET_KEY")
CURRENT_SEASON = os.getenv("CURRENT_SEASON")


def create_app(prediction_engine="Random"):
    db_path = os.path.join(PROJECT_ROOT, "data", "NBA_AI.sqlite")
    update_scheduled_games(CURRENT_SEASON, db_path)

    app = Flask(__name__)
    app.secret_key = WEB_APP_SECRET_KEY
    app.config["PREDICTION_ENGINE"] = prediction_engine
    app.config["MODEL"] = load_models(prediction_engine)

    @app.route("/")
    def home():
        """
        Home route handler. It handles the display of games based on the date provided in the query parameters.
        If no date is provided, it defaults to the current date. If an invalid date is provided, it flashes an error message
        and defaults to the current date. It also calculates the previous and next dates for navigation links.
        """

        # Get the current date in Eastern Time Zone
        current_date_et = get_user_datetime(as_eastern_tz=True)
        # Format the current date as a string
        current_date_str = current_date_et.strftime("%Y-%m-%d")
        # Get the date from the query parameters, default to the current date if not provided
        query_date_str = request.args.get("date", current_date_str)

        # Attempt to validate the date provided in the query parameters
        try:
            validate_date_format(query_date_str)
            # If the date is valid, parse it into a datetime object
            query_date = datetime.strptime(query_date_str, "%Y-%m-%d")
        except Exception as e:
            # If the date is invalid, log a warning with the invalid date and the error
            logging.warning(
                f"Date validation error for input date string: {query_date_str}. Error: {e}"
            )
            # Flash an error message to the user
            flash("Invalid date format. Showing games for today.", "error")
            # Default to the current date
            query_date_str = current_date_str
            query_date = current_date_et

        # Format the query date as a string for display
        query_date_display_str = query_date.strftime("%b %d")

        # Calculate the previous and next dates for the navigation links
        next_date = query_date + timedelta(days=1)
        prev_date = query_date - timedelta(days=1)
        # Format the previous and next dates as strings
        next_date_str = next_date.strftime("%Y-%m-%d")
        prev_date_str = prev_date.strftime("%Y-%m-%d")

        # Render the index.html template, passing the necessary data for initial page setup
        return render_template(
            "index.html",
            query_date_str=query_date_str,
            query_date_display_str=query_date_display_str,
            prev_date=prev_date_str,
            next_date=next_date_str,
        )

    @app.route("/get-games")
    def get_games():
        """
        Fetch and process games for a given date.

        This function fetches games for a given date, gets game info and predictions for each game,
        and processes the game data. If any of these steps fail, it logs an error and includes an error
        message in the response.

        The date is provided as a 'date' parameter in the request. If no date is provided, it defaults
        to the current date.

        Returns:
            A JSON response containing the processed game data for each game.
        """
        # Get the current date in Eastern Time Zone
        current_date_est = get_user_datetime(as_eastern_tz=True)

        # Fetch the date parameter from the request, default to current date if not provided
        inbound_query_date_str = request.args.get("date")
        if inbound_query_date_str is None or inbound_query_date_str == "":
            query_date_str = current_date_est.strftime("%Y-%m-%d")
        else:
            query_date_str = inbound_query_date_str

        # Before loading games, validate the query date format
        try:
            validate_date_format(query_date_str)
        except Exception as e:  # Adjust to catch specific exceptions as needed
            # Log an error if the date format is invalid
            logging.error(f"Invalid date format provided for get-games: {e}")
            return jsonify({"error": "Invalid date format provided"}), 400

        # Load Games
        try:
            scheduled_games = get_games_for_date(query_date_str, db_path)
        except Exception as e:
            # Log an error if loading games fails
            logging.error(f"Error loading games for date {query_date_str}: {e}")
            return (
                jsonify({"error": f"Error loading games for date {query_date_str}"}),
                500,
            )

        scheduled_game_ids = [game["game_id"] for game in scheduled_games]

        # Get the game info for all games
        games = get_games_info(
            scheduled_game_ids,
            db_path=db_path,
            include_prior_states=False,
            save_to_database=False,
        )

        # Get predictions for all games
        predictions = get_predictions(games, prediction_engine)

        # Process the game data for all games
        outbound_game_data = process_games_data(games, predictions)

        # Return the outbound game data as a JSON response
        return jsonify(outbound_game_data)

    @app.route("/game-details/<game_id>")
    def game_details(game_id):
        """
        Endpoint to get game details.

        This function is mapped to the "/game-details/<game_id>" route. It loads game data, creates predictions,
        processes the game data, and returns it as a JSON response.

        Args:
            game_id (str): The ID of the game.

        Returns:
            Response: A response object containing the game data in JSON format.
        """
        # Load game data using the get_game_info function
        # The game data is not saved to the database and prior states are not included or updated
        game_info = get_games_info(
            game_id,
            db_path=db_path,
            include_prior_states=False,
            save_to_database=False,
        )

        # Create predictions for the game using the get_predictions function
        predictions = get_predictions(game_info, prediction_engine)

        # Process the game data using the process_game_data function
        # The processed game data includes the game info and predictions
        outbound_game_data = process_games_data(game_info, predictions)[0]

        # Return the processed game data as a JSON response
        return jsonify(outbound_game_data)

    @app.after_request
    def add_header(response):
        response.headers["Cache-Control"] = "no-store"
        return response

    return app
