import argparse

from web_app.app import create_app

"""
Prediction Engine Options
-------------------------
Random
    - Prior to Game Start: Randomly predicts the results within a broad range 
    - After Game Start: No updates are made based on the current game state
LinearModel 
    - Prior to Game Start: Predicts the results using a Ridge Regression model
    - After Game Start: Predictions are a combination of the model's pregame predictions 
                        and the current game state.   
TreeModel
    - Prior to Game Start: Predicts the results using a XGBoost model
    - After Game Start: Predictions are a combination of the model's pregame predictions 
                        and the current game state.
"""

# Default prediction engine
default_prediction_engine = "Random"

# Create the parser
parser = argparse.ArgumentParser(
    description="Launch the web app with a specified prediction engine"
)

# Add the arguments
parser.add_argument(
    "PredictionEngine",
    metavar="prediction_engine",
    type=str,
    nargs="?",
    default=default_prediction_engine,
    help="The prediction engine to use",
)

# Parse the arguments
args = parser.parse_args()

# Create the app with the specified prediction engine
app = create_app(prediction_engine=args.PredictionEngine)

if __name__ == "__main__":
    app.run(debug=True)
