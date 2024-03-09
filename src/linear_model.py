import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from dotenv import load_dotenv
from joblib import dump
from sklearn.linear_model import Ridge
from sklearn.metrics import brier_score_loss, mean_absolute_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from wandb.sklearn import plot_regressor

import wandb
from utils import load_featurized_data

load_dotenv()
PROJECT_ROOT = os.getenv("PROJECT_ROOT")

pd.set_option("display.max_columns", None)
sns.set_context("notebook")

if __name__ == "__main__":
    # Choose Seasons
    training_seasons = ["2021-2022"]
    testing_seasons = ["2022-2023"]

    # Load data
    training_df = load_featurized_data(training_seasons)
    testing_df = load_featurized_data(testing_seasons)

    # Remove nans
    training_df = training_df.dropna()
    testing_df = testing_df.dropna()

    # Create X and y
    X_train = training_df.drop(
        columns=[
            "game_id",
            "game_date",
            "home_margin",
            "home_score",
            "away_score",
            "total_score",
        ]
    )
    y_train = training_df[["home_score", "away_score"]]
    X_test = testing_df.drop(
        columns=[
            "game_id",
            "game_date",
            "home_margin",
            "home_score",
            "away_score",
            "total_score",
        ]
    )
    y_test = testing_df[["home_score", "away_score"]]

    # Save the feature names for later use
    feature_names = X_train.columns.tolist()

    # Initialize the scaler
    scaler = StandardScaler()

    # Fit on the training data and transform both the training and testing data
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Hyperparameter Tuning
    # Define the hyperparameters and their distributions
    param_distributions = {
        "alpha": np.logspace(-4, 4, 20),  # Logarithmically spaced values
        "fit_intercept": [True, False],
    }

    # Initialize the RandomizedSearchCV for Ridge Regression
    random_search = RandomizedSearchCV(
        Ridge(), param_distributions, n_iter=10, cv=5, random_state=42
    )

    # Fit the RandomizedSearchCV
    random_search.fit(X_train, y_train)

    # Get the best parameters
    best_params = random_search.best_params_

    # Initialize wandb
    run = wandb.init(project="NBA AI", config=best_params)

    # Update the configuration
    model_type = "Ridge Regression"
    run_datetime = datetime.now().isoformat()
    wandb.config.update(
        {
            "model_type": model_type,
            "train_seasons": training_seasons,
            "test_seasons": testing_seasons,
            "train_shape": X_train.shape,
            "test_shape": X_test.shape,
            "targets": ["home_score", "away_score"],
            "features": feature_names,
            "run_datetime": run_datetime,
        }
    )

    # Setup the model with the best parameters found
    model = Ridge(**best_params)

    # Fit the model
    model.fit(X_train, y_train)

    # Create predictions
    y_pred = model.predict(X_test)
    y_pred_train = model.predict(X_train)

    y_pred_home_score = y_pred[:, 0]
    y_pred_away_score = y_pred[:, 1]
    y_pred_home_margin = y_pred_home_score - y_pred_away_score

    y_pred_train_home_score = y_pred_train[:, 0]
    y_pred_train_away_score = y_pred_train[:, 1]
    y_pred_train_home_margin = y_pred_train_home_score - y_pred_train_away_score

    def sigmoid(x):
        sensitivity_factor = (
            0.1  # Adjust this value to change the steepness of the curve
        )
        return 1 / (1 + np.exp(-x * sensitivity_factor))

    y_pred_home_win_prob = sigmoid(y_pred_home_margin)
    y_pred_train_home_win_prob = sigmoid(y_pred_train_home_margin)

    # Compute metrics
    train_mae_home_score = mean_absolute_error(
        y_train["home_score"], y_pred_train_home_score
    )
    test_mae_home_score = mean_absolute_error(y_test["home_score"], y_pred_home_score)
    train_r2_home_score = r2_score(y_train["home_score"], y_pred_train_home_score)
    test_r2_home_score = r2_score(y_test["home_score"], y_pred_home_score)

    train_mae_away_score = mean_absolute_error(
        y_train["away_score"], y_pred_train_away_score
    )
    test_mae_away_score = mean_absolute_error(y_test["away_score"], y_pred_away_score)
    train_r2_away_score = r2_score(y_train["away_score"], y_pred_train_away_score)
    test_r2_away_score = r2_score(y_test["away_score"], y_pred_away_score)

    y_train_home_margin = y_train["home_score"] - y_train["away_score"]
    y_test_home_margin = y_test["home_score"] - y_test["away_score"]

    train_mae_home_margin = mean_absolute_error(
        y_train_home_margin, y_pred_train_home_margin
    )
    test_mae_home_margin = mean_absolute_error(y_test_home_margin, y_pred_home_margin)
    train_r2_home_margin = r2_score(y_train_home_margin, y_pred_train_home_margin)
    test_r2_home_margin = r2_score(y_test_home_margin, y_pred_home_margin)

    y_train_home_win = y_train_home_margin > 0
    y_test_home_win = y_test_home_margin > 0

    train_brier_score = brier_score_loss(y_train_home_win, y_pred_train_home_win_prob)
    test_brier_score = brier_score_loss(y_test_home_win, y_pred_home_win_prob)

    # Log the metrics
    wandb.log(
        {
            "train_mae_home_score": train_mae_home_score,
            "test_mae_home_score": test_mae_home_score,
            "train_r2_home_score": train_r2_home_score,
            "test_r2_home_score": test_r2_home_score,
            "train_mae_away_score": train_mae_away_score,
            "test_mae_away_score": test_mae_away_score,
            "train_r2_away_score": train_r2_away_score,
            "test_r2_away_score": test_r2_away_score,
            "train_mae_home_margin": train_mae_home_margin,
            "test_mae_home_margin": test_mae_home_margin,
            "train_r2_home_margin": train_r2_home_margin,
            "test_r2_home_margin": test_r2_home_margin,
            "train_brier_score": train_brier_score,
            "test_brier_score": test_brier_score,
        }
    )

    # Log model info
    wandb.log({"intercept": model.intercept_})
    # Log the coefficients as a dictionary
    wandb.log({"coefficients": dict(zip(feature_names, model.coef_))})

    # Create and log the plots
    plot_regressor(model, X_train, X_test, y_train, y_test, model_name=model_type)

    # Feature importances
    feature_importances = dict(zip(feature_names, model.coef_))
    sorted_features = sorted(
        feature_importances.items(), key=lambda x: abs(x[1]), reverse=True
    )
    features, importances = zip(*sorted_features)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances, y=features, palette="viridis")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Feature Importances")

    wandb.log({"feature_importances": plt})

    # Save the model
    model_filename = f"{PROJECT_ROOT}/models/{model_type}_{run_datetime}_train_mae_{train_mae:.2f}_test_mae_{test_mae:.2f}.joblib"

    # Save the model using joblib
    dump(model, model_filename)
    wandb.save(model_filename, base_path=PROJECT_ROOT)

    # Finish the wandb run
    run.finish()
