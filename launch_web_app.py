import argparse

from src.web_app.app import create_app

"""
Prediction Engine Options
-------------------------
Best
    - Prior to Game Start: Predicts the results using the best model set in the config file
    - After Game Start: Predictions are a combination of the best model's pregame predictions
                        and the current game state.
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
MLPModel
    - Prior to Game Start: Predicts the results using a Multi-Layer Perceptron model
    - After Game Start: Predictions are a combination of the model's pregame predictions 
                        and the current game state.
"""

# Create the parser
parser = argparse.ArgumentParser(
    description="Launch the web app with a specified prediction engine"
)

# Add the arguments
parser.add_argument(
    "--predictor",
    default="Random",
    type=str,
    help="The predictor to use for predictions.",
)

# Parse the arguments
args = parser.parse_args()

# Create the app with the specified prediction engine
app = create_app(predictor=args.predictor)

if __name__ == "__main__":
    app.run(debug=True)
