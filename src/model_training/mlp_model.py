from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import wandb
from src.config import config
from src.model_training.evaluation import create_evaluations
from src.model_training.modeling_utils import load_featurized_modeling_data

# Configuration
DB_PATH = config["database"]["path"]
PROJECT_ROOT = config["project"]["root"]


class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    log_to_wandb = True  # Set to False to disable logging to Weights & Biases
    if log_to_wandb:
        run = wandb.init(project="NBA AI")
    if not log_to_wandb:
        print("\nLogging to Weights & Biases disabled.")

    model_type = "MLP_Regression"
    run_datetime = datetime.now().isoformat()

    # -------------------------
    # Section 1: Data Loading
    # -------------------------

    # Define the seasons for training and testing
    available_training_seasons = [
        "2001-2002",
        "2002-2003",
        "2003-2004",
        "2004-2005",
        "2005-2006",
        "2006-2007",
        "2007-2008",
        "2008-2009",
        "2009-2010",
        "2010-2011",
        "2011-2012",
        "2012-2013",
        "2013-2014",
        "2014-2015",
        "2015-2016",
        "2016-2017",
        "2017-2018",
        "2018-2019",
        "2019-2020",
        "2020-2021",
        "2021-2022",
        "2022-2023",
    ]

    training_seasons = available_training_seasons
    testing_seasons = ["2023-2024"]

    # Load featurized modeling data for the defined seasons
    print("\nLoading featurized modeling data...")
    print(f"Training seasons: {training_seasons}")
    print(f"Testing seasons: {testing_seasons}")
    training_df = load_featurized_modeling_data(training_seasons, DB_PATH)
    testing_df = load_featurized_modeling_data(testing_seasons, DB_PATH)
    print(f"Training data shape: {training_df.shape}")
    print(f"Testing data shape: {testing_df.shape}")

    # Drop rows with NaN values to ensure data quality
    print("\nDropping rows with NaN values...")
    training_df = training_df.dropna()
    testing_df = testing_df.dropna()
    print(f"Training data shape after dropping NaNs: {training_df.shape}")
    print(f"Testing data shape after dropping NaNs: {testing_df.shape}")

    # -----------------------------------------
    # Section 2: Feature and Target Selection
    # -----------------------------------------

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

    # -----------------------------
    # Section 3: Data Preparation
    # -----------------------------

    print("\nPreparing data for model training...")
    # Convert the DataFrames to PyTorch tensors
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

    # Initialize and fit a scaler to standardize features
    mean = X_train_tensor.mean(dim=0)
    std = X_train_tensor.std(dim=0)
    X_train_scaled = (X_train_tensor - mean) / std
    X_test_scaled = (X_test_tensor - mean) / std

    # Define PyTorch Datasets
    train_dataset = TensorDataset(X_train_scaled, y_train_tensor)
    test_dataset = TensorDataset(X_test_scaled, y_test_tensor)

    # Define PyTorch DataLoaders
    batch_size = 64  # Number of samples in each batch
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # -----------------------------
    # Section 4: Model Definition
    # -----------------------------

    # See above for the definition of the MLP class

    # Define the size of each layer
    input_size = X_train_scaled.shape[1]  # Number of features

    # Initialize the model
    model = MLP(input_size)

    if log_to_wandb:
        wandb.watch(model)

    print("\nModel Architecture:")
    print(model)

    # --------------------------------
    # Section 5: Model Training Loop
    # --------------------------------

    print("\nTraining the model...")
    print(
        f"{'Epoch':<10}{'Training Loss':<15}{'Test Loss':<10}{'Train Home Margin MAE':<20}{'Train Home Score MAE':<20}{'Train Away Score MAE':<20}{'Test Home Margin MAE':<20}{'Test Home Score MAE':<20}{'Test Away Score MAE':<20}"
    )
    # Define loss function and optimizer
    criterion = nn.MSELoss()  # Mean Squared Error Loss for regression tasks
    learning_rate = 0.001  # Learning rate for the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # Adam optimizer

    # Define MAE loss function for metrics
    mae = nn.L1Loss()

    # Number of epochs
    epochs = 100

    # Training loop
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        running_mae_home = 0.0
        running_mae_away = 0.0
        running_mae_margin = 0.0
        for inputs, targets in train_loader:
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Calculate metrics
            running_loss += loss.item() * inputs.size(0)
            mae_loss_home = mae(outputs[:, 0], targets[:, 0])
            mae_loss_away = mae(outputs[:, 1], targets[:, 1])
            mae_loss_margin = mae(
                outputs[:, 0] - outputs[:, 1], targets[:, 0] - targets[:, 1]
            )
            running_mae_home += mae_loss_home.item() * inputs.size(0)
            running_mae_away += mae_loss_away.item() * inputs.size(0)
            running_mae_margin += mae_loss_margin.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_mae_home = running_mae_home / len(train_loader.dataset)
        epoch_mae_away = running_mae_away / len(train_loader.dataset)
        epoch_mae_margin = running_mae_margin / len(train_loader.dataset)

        # Validation phase
        model.eval()
        test_loss = 0.0
        test_mae_home = 0.0
        test_mae_away = 0.0
        test_mae_margin = 0.0
        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item() * inputs.size(0)
                mae_loss_home = mae(outputs[:, 0], targets[:, 0])
                mae_loss_away = mae(outputs[:, 1], targets[:, 1])
                mae_loss_margin = mae(
                    outputs[:, 0] - outputs[:, 1], targets[:, 0] - targets[:, 1]
                )
                test_mae_home += mae_loss_home.item() * inputs.size(0)
                test_mae_away += mae_loss_away.item() * inputs.size(0)
                test_mae_margin += mae_loss_margin.item() * inputs.size(0)

        test_loss /= len(test_loader.dataset)
        test_mae_home /= len(test_loader.dataset)
        test_mae_away /= len(test_loader.dataset)
        test_mae_margin /= len(test_loader.dataset)

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch+1}/{epochs}, Train Loss: {epoch_loss:.2f}, Test Loss: {test_loss:.2f}\nTrain Home Margin MAE: {epoch_mae_margin:.2f}, Train Home Score MAE: {epoch_mae_home:.2f}, Train Away Score MAE: {epoch_mae_away:.2f}\nTest Home Margin MAE: {test_mae_margin:.2f}, Test Home Score MAE: {test_mae_home:.2f}, Test Away Score MAE: {test_mae_away:.2f}"
            )
        else:
            print(
                f"Epoch {epoch+1}/{epochs}, Train Loss: {epoch_loss:.2f}, Test Loss: {test_loss:.2f}"
            )

        # Log to Weights and Biases
        if log_to_wandb:
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "train_loss": epoch_loss,
                    "test_loss": test_loss,
                    "train_home_margin_mae": epoch_mae_margin,
                    "train_home_score_mae": epoch_mae_home,
                    "train_away_score_mae": epoch_mae_away,
                    "test_home_margin_mae": test_mae_margin,
                    "test_home_score_mae": test_mae_home,
                    "test_away_score_mae": test_mae_away,
                }
            )

    # -------------------------------
    # Section 6: Making Predictions
    # -------------------------------

    # Predict on training and testing data
    print("\nMaking predictions...")
    # Set the model to evaluation mode
    model.eval()

    # Initialize lists to store predictions
    y_pred_home_score, y_pred_away_score = [], []
    y_pred_train_home_score, y_pred_train_away_score = [], []

    # Predict on the test and training sets
    with torch.no_grad():
        for (
            inputs_test,
            _,
        ) in test_loader:  # Assuming test_loader returns (inputs, targets)
            outputs_test = model(inputs_test)

            # Store predictions
            y_pred_home_score.append(outputs_test[:, 0])
            y_pred_away_score.append(outputs_test[:, 1])

        for (
            inputs_train,
            _,
        ) in train_loader:  # Assuming train_loader returns (inputs, targets)
            outputs_train = model(inputs_train)

            # Store predictions
            y_pred_train_home_score.append(outputs_train[:, 0])
            y_pred_train_away_score.append(outputs_train[:, 1])

    # Concatenate all the predictions
    y_pred_home_score = torch.cat(y_pred_home_score)
    y_pred_away_score = torch.cat(y_pred_away_score)
    y_pred_train_home_score = torch.cat(y_pred_train_home_score)
    y_pred_train_away_score = torch.cat(y_pred_train_away_score)

    # Calculate home margin
    y_pred_home_margin = y_pred_home_score - y_pred_away_score
    y_pred_train_home_margin = y_pred_train_home_score - y_pred_train_away_score

    # Calculate win probabilities using the logistic (sigmoid) function
    def win_prob(score_diff):
        a = -0.2504  # Intercept term
        b = 0.1949  # Slope term
        win_prob = 1 / (1 + torch.exp(-(a + b * score_diff)))
        return win_prob

    y_pred_home_win_prob = win_prob(y_pred_home_margin)
    y_pred_train_home_win_prob = win_prob(y_pred_train_home_margin)

    # -----------------------------
    # Section 7: Model Evaluation
    # -----------------------------

    print("\nEvaluating the model predictions...")

    # Convert numpy arrays to PyTorch tensors
    y_test_home_score = torch.tensor(
        y_test["home_score"].values.astype(np.float32), requires_grad=False
    )
    y_test_away_score = torch.tensor(
        y_test["away_score"].values.astype(np.float32), requires_grad=False
    )

    # Core Metrics
    home_score_mae = torch.mean(torch.abs(y_test_home_score - y_pred_home_score))
    away_score_mae = torch.mean(torch.abs(y_test_away_score - y_pred_away_score))
    home_margin_mae = torch.mean(
        torch.abs(y_test_home_score - y_test_away_score - y_pred_home_margin)
    )

    # Define the loss function
    loss_fn = nn.BCELoss()

    # Calculate log loss (binary cross entropy loss)
    home_win_prob_log_loss = loss_fn(
        y_pred_home_win_prob, (y_test_home_score > y_test_away_score).float()
    )

    print("\nCore Metrics for the Test Set:")
    print(f"Home Score MAE: {home_score_mae:.2f}")
    print(f"Away Score MAE: {away_score_mae:.2f}")
    print(f"Home Margin MAE: {home_margin_mae:.2f}")
    print(f"Home Win Probability Log Loss: {home_win_prob_log_loss:.4f}")

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

    # ------------------------------------------
    # Section 8: Recreating Model on Full Data
    # ------------------------------------------

    print("\nRecreating the model on full data...")
    # Combine the training and testing data
    X_full = pd.concat([X_train, X_test], axis=0)
    y_full = pd.concat([y_train, y_test], axis=0)

    # Convert the DataFrames to PyTorch tensors
    X_full_tensor = torch.tensor(X_full.values, dtype=torch.float32)
    y_full_tensor = torch.tensor(y_full.values, dtype=torch.float32)

    # Standardize the features
    final_mean = X_full_tensor.mean(dim=0)
    final_std = X_full_tensor.std(dim=0)
    X_full_scaled = (X_full_tensor - final_mean) / final_std

    # Define the PyTorch Dataset and DataLoader
    full_dataset = TensorDataset(X_full_scaled, y_full_tensor)
    full_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True)

    # Initialize the model and optimizer
    final_model = MLP(input_size)
    final_optimizer = torch.optim.Adam(final_model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(epochs):
        final_model.train()
        final_running_loss = 0.0
        for inputs, targets in full_loader:
            final_optimizer.zero_grad()
            outputs = final_model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            final_optimizer.step()

            final_running_loss += loss.item() * inputs.size(0)

        final_epoch_loss = final_running_loss / len(full_loader.dataset)
        print(f"Epoch {epoch+1}/{epochs}, Final Training Loss: {final_epoch_loss:.2f}")

    # -----------------------------
    # Section 9: Saving the Model
    # -----------------------------

    # Construct filename and save the pipeline
    print("\nSaving the model...")
    model_filename = f"{PROJECT_ROOT}/models/{model_type}_{run_datetime}.pth"
    torch.save(
        {
            "model_state_dict": final_model.state_dict(),
            "mean": final_mean,
            "std": final_std,
            "input_size": input_size,
        },
        model_filename,
    )

    print(f"Model saved to {model_filename}\n")

    # ----------------------------------------
    # Section 10: Logging to Weights & Biases
    # ----------------------------------------

    if not log_to_wandb:
        exit()

    # Log configuration and model details
    wandb.config.update(
        {
            "model_type": model_type,
            "train_seasons": training_seasons,
            "train_season_count": len(training_seasons),
            "test_seasons": testing_seasons,
            "train_shape": X_train.shape,
            "test_shape": X_test.shape,
            "targets": ["home_score", "away_score"],
            "features": feature_names,
            "run_datetime": run_datetime,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "epochs": epochs,
            "optimizer": str(optimizer),
            "criterion": str(criterion),
        }
    )

    # Log core metrics
    wandb.summary.update(
        {
            "home_score_mae": home_score_mae,
            "away_score_mae": away_score_mae,
            "home_margin_mae": home_margin_mae,
            "home_win_prob_log_loss": home_win_prob_log_loss,
        }
    )

    # Log the full evaluation suite
    train_evaluations_table = wandb.Table(dataframe=train_evaluations)
    test_evaluations_table = wandb.Table(dataframe=test_evaluations)
    wandb.summary.update({"Train Evals": train_evaluations_table})
    wandb.summary.update({"Test Evals": test_evaluations_table})

    # Save the model to wandb
    wandb.save(model_filename, base_path=PROJECT_ROOT)

    # End the wandb run
    run.finish()
