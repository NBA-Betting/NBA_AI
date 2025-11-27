"""
models.py

Shared neural network model definitions for training and inference.

Classes:
- MLP: Multi-layer perceptron for NBA score prediction.

Usage:
    from src.model_training.models import MLP

    model = MLP(input_size=43, hidden_sizes=[64, 32], dropout=0.2)
    predictions = model(features_tensor)
"""

import torch
from torch import nn


class MLP(nn.Module):
    """
    Multi-layer Perceptron for NBA score prediction.

    Architecture:
        input → hidden_1 → ReLU → Dropout → ... → hidden_n → ReLU → Dropout → 2

    Predicts [home_score, away_score] from game features.

    Args:
        input_size: Number of input features (default: 43)
        hidden_sizes: List of hidden layer sizes (default: [64, 32])
        dropout: Dropout probability (default: 0.2)

    Example:
        >>> model = MLP(input_size=43)
        >>> x = torch.randn(32, 43)  # batch of 32 games
        >>> predictions = model(x)  # shape: (32, 2)
    """

    def __init__(self, input_size, hidden_sizes=None, dropout=0.2):
        super(MLP, self).__init__()

        if hidden_sizes is None:
            hidden_sizes = [64, 32]

        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.dropout_rate = dropout

        # Build network layers
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, 2))  # Output: [home_score, away_score]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, input_size)

        Returns:
            Tensor of shape (batch_size, 2) with [home_score, away_score]
        """
        return self.network(x)
