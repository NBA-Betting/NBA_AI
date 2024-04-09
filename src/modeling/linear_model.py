import os
from datetime import datetime

import numpy as np
import pandas as pd
import wandb
from dotenv import load_dotenv
from evaluation import create_evaluations
from joblib import dump
from modeling_utils import load_featurized_modeling_data
from sklearn.linear_model import Ridge
from sklearn.metrics import log_loss
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

load_dotenv()
PROJECT_ROOT = os.getenv("PROJECT_ROOT")

if __name__ == "__main__":
    db_path = f"{PROJECT_ROOT}/data/NBA_AI.sqlite"

    # ------------------------
    # Section 1: Data Loading
    # ------------------------
    # Define the seasons for training and testing
    training_seasons = ["2020-2021", "2021-2022"]
    testing_seasons = ["2022-2023"]

    # Load featurized modeling data for the defined seasons
    print("Loading featurized modeling data...")
    print(f"Training seasons: {training_seasons}")
    print(f"Testing seasons: {testing_seasons}")
    training_df = load_featurized_modeling_data(training_seasons, db_path)
    testing_df = load_featurized_modeling_data(testing_seasons, db_path)
    print(f"Training data shape: {training_df.shape}")
    print(f"Testing data shape: {testing_df.shape}")

    # Drop rows with NaN values to ensure data quality
    print("\nDropping rows with NaN values...")
    training_df = training_df.dropna()
    testing_df = testing_df.dropna()
    print(f"Training data shape after dropping NaNs: {training_df.shape}")
    print(f"Testing data shape after dropping NaNs: {testing_df.shape}")

    # ----------------------------------------
    # Section 2: Feature and Target Selection
    # ----------------------------------------
    # Define features (X) by dropping target and non-predictive columns
    # Define targets (y) for the model
    game_info_columns = [
        "game_id",
        "date_time_est",
        "home_team",
        "away_team",
        "season",
        "season_type",
    ]
    game_results_columns = [
        "home_score",
        "away_score",
        "total",
        "home_margin",
        "players_data",
    ]

    X_train = training_df.drop(columns=game_info_columns + game_results_columns)
    y_train = training_df[["home_score", "away_score"]]
    X_test = testing_df.drop(columns=game_info_columns + game_results_columns)
    y_test = testing_df[["home_score", "away_score"]]

    # Keep a list of feature names for later use (e.g., for model interpretation)
    feature_names = X_train.columns.tolist()

    print("\nX_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape, y_train.columns.tolist())
    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape, y_test.columns.tolist())

    # ---------------------------
    # Section 3: Data Preprocessing
    # ---------------------------
    # Initialize and fit a scaler to standardize features
    print("\nStandardizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # -------------------------------------
    # Section 4: Hyperparameter Tuning
    # -------------------------------------
    print("\nPerforming hyperparameter tuning...")
    # Define the hyperparameter space and setup RandomizedSearchCV
    param_distributions = {
        "alpha": np.logspace(-4, 4, 20),  # Defines a range of values for alpha
        "fit_intercept": [
            True,
            False,
        ],  # Whether to calculate the intercept for this model
    }

    # Perform hyperparameter tuning using Randomized Search
    random_search = RandomizedSearchCV(
        Ridge(), param_distributions, n_iter=10, cv=5, random_state=42
    )
    random_search.fit(X_train_scaled, y_train)
    best_params = random_search.best_params_

    # ------------------------------------
    # Section 5: Model Training and Logging
    # ------------------------------------
    # Initialize wandb for experiment tracking
    run = wandb.init(project="NBA AI", config=best_params)

    # Log configuration and model details
    model_type = "Ridge_Regression"
    run_datetime = datetime.now().isoformat()
    wandb.config.update(
        {
            "model_type": model_type,
            "train_seasons": training_seasons,
            "test_seasons": testing_seasons,
            "train_shape": X_train_scaled.shape,
            "test_shape": X_test_scaled.shape,
            "targets": ["home_score", "away_score"],
            "features": feature_names,
            "run_datetime": run_datetime,
        }
    )

    # Setup and fit the Ridge Regression model with the best hyperparameters
    print("\nTraining the Ridge Regression model...")
    model = Ridge(**best_params)
    model.fit(X_train_scaled, y_train)

    # -----------------------------
    # Section 6: Making Predictions
    # -----------------------------
    # Predict on training and testing data
    print("\nMaking predictions...")
    y_pred = model.predict(X_test_scaled)
    y_pred_train = model.predict(X_train_scaled)

    # Extract specific prediction components for analysis
    y_pred_home_score = y_pred[:, 0]
    y_pred_away_score = y_pred[:, 1]
    y_pred_home_margin = y_pred_home_score - y_pred_away_score

    y_pred_train_home_score = y_pred_train[:, 0]
    y_pred_train_away_score = y_pred_train[:, 1]
    y_pred_train_home_margin = y_pred_train_home_score - y_pred_train_away_score

    # Apply a sigmoid function for probability estimation (e.g., home team win probability)
    def sigmoid(x):
        sensitivity_factor = 0.1  # Adjust this value as needed
        return 1 / (1 + np.exp(-x * sensitivity_factor))

    y_pred_home_win_prob = sigmoid(y_pred_home_margin)
    y_pred_train_home_win_prob = sigmoid(y_pred_train_home_margin)

    # -----------------------------------
    # Section 7: Evaluation and Logging
    # -----------------------------------
    # Log Core Metrics
    home_score_mae = np.mean(np.abs(y_test["home_score"] - y_pred_home_score))
    away_score_mae = np.mean(np.abs(y_test["away_score"] - y_pred_away_score))
    home_margin_mae = np.mean(
        np.abs(y_test["home_score"] - y_test["away_score"] - y_pred_home_margin)
    )
    home_win_prob_log_loss = log_loss(
        (y_test["home_score"] > y_test["away_score"]).astype(int), y_pred_home_win_prob
    )

    wandb.log(
        {
            "home_score_mae": home_score_mae,
            "away_score_mae": away_score_mae,
            "home_margin_mae": home_margin_mae,
            "home_win_prob_log_loss": home_win_prob_log_loss,
        }
    )

    # Run full evaluation suite
    # Prepare correct and predicted values for evaluation
    train_correct = {
        "home_score": y_train["home_score"],
        "away_score": y_train["away_score"],
        "home_margin_derived": y_train["home_score"] - y_train["away_score"],
        "total_points_derived": y_train["home_score"] + y_train["away_score"],
        "home_win_prob": (y_train["home_score"] > y_train["away_score"]).astype(int),
    }
    train_pred = {
        "home_score": y_pred_train_home_score,
        "away_score": y_pred_train_away_score,
        "home_margin_derived": y_pred_train_home_margin,
        "total_points_derived": y_pred_train_home_score + y_pred_train_away_score,
        "home_win_prob": y_pred_train_home_win_prob,
    }

    test_correct = {
        "home_score": y_test["home_score"],
        "away_score": y_test["away_score"],
        "home_margin_derived": y_test["home_score"] - y_test["away_score"],
        "total_points_derived": y_test["home_score"] + y_test["away_score"],
        "home_win_prob": (y_test["home_score"] > y_test["away_score"]).astype(int),
    }

    test_pred = {
        "home_score": y_pred_home_score,
        "away_score": y_pred_away_score,
        "home_margin_derived": y_pred_home_margin,
        "total_points_derived": y_pred_home_score + y_pred_away_score,
        "home_win_prob": y_pred_home_win_prob,
    }

    train_evaluations = pd.DataFrame(
        [
            {
                "train_" + k: v
                for k, v in create_evaluations(train_correct, train_pred).items()
            }
        ]
    )

    test_evaluations = pd.DataFrame(
        [
            {
                "test_" + k: v
                for k, v in create_evaluations(test_correct, test_pred).items()
            }
        ]
    )

    # Convert the dataframes to wandb Tables
    train_evaluations_table = wandb.Table(dataframe=train_evaluations)
    test_evaluations_table = wandb.Table(dataframe=test_evaluations)

    # Log the tables to wandb
    wandb.log({"Train Evals": train_evaluations_table})
    wandb.log({"Test Evals": test_evaluations_table})

    # Create a DataFrame with the intercept and coefficients
    model_details = pd.DataFrame(
        [
            {
                "feature_" + feature: value
                for feature, value in zip(
                    ["intercept"] + feature_names,
                    [model.intercept_] + list(model.coef_),
                )
            }
        ]
    )

    # Convert the DataFrame to a wandb Table
    model_details_table = wandb.Table(dataframe=model_details)

    # Log the table to wandb
    wandb.log({"Model Details": model_details_table})

    # ------------------------------------------
    # Section 8: Recreating Model on Full Data
    # ------------------------------------------

    print("\nRecreating the model on full data...")
    X_all = np.concatenate((X_train, X_test))
    y_all = np.concatenate((y_train, y_test))

    final_scaler = StandardScaler()
    X_all_scaled = final_scaler.fit_transform(X_all)

    final_random_search = RandomizedSearchCV(
        Ridge(), param_distributions, n_iter=10, cv=5, random_state=42
    )
    final_random_search.fit(X_all_scaled, y_all)
    final_params = final_random_search.best_params_

    final_model = Ridge(**final_params)
    final_model.fit(X_all_scaled, y_all)

    pipeline = Pipeline([("scaler", final_scaler), ("model", final_model)])

    # ----------------------------------
    # Section 9: Saving the Model
    # ----------------------------------

    # Construct filename and save the pipeline
    print("\nSaving the model...")
    model_filename = f"{PROJECT_ROOT}/models/{model_type}_{run_datetime}.joblib"
    dump(pipeline, model_filename)
    wandb.save(model_filename, base_path=PROJECT_ROOT)
    print(f"Model saved to {model_filename}")

    # End the wandb run
    run.finish()
