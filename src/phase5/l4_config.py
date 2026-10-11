"""
L4 Game Context and Prediction configuration.
"""

from dataclasses import dataclass


@dataclass
class L4Config:
    """Configuration for Level 4: Game Context + Prediction."""

    # --- Input dimensions (from lower levels) ---
    d_team: int = 128  # L3 team representation per team
    d_l2: int = 134  # L2 team vector per team

    # --- Context features ---
    n_context: int = 14  # game-specific context features
    d_context_proj: int = 64  # projected context dimension

    # --- Matchup construction ---
    # matchup = concat(home, away) + diff + hadamard = 4 * d_team = 512
    # l4_input = concat(matchup, context_proj) = 512 + 64 = 576

    # --- L2 skip connection ---
    d_l2_skip: int = 256  # projected L2 diff dimension (additive after first layer)

    # --- Prediction MLP ---
    d_hidden: int = 256  # hidden dimension throughout MLP
    d_ff: int = 512  # feed-forward expansion in residual blocks
    n_residual_blocks: int = 2  # number of residual blocks
    dropout: float = 0.1

    # --- Prediction heads ---
    # spread_mu: Linear(d_hidden, 1)
    # spread_sigma: Softplus(Linear(d_hidden, 1))
    # win_logit: Linear(d_hidden, 1)

    # --- Loss weights ---
    w_spread: float = 1.0  # Gaussian NLL for spread
    w_win: float = 0.3  # BCE for win probability
    w_consistency: float = 0.1  # MSE between Phi(mu/sigma) and win_prob

    # --- Training ---
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    batch_size: int = 128
    gradient_clip: float = 1.0
    epochs: int = 60
    patience: int = 15

    # --- Spread loss ---
    spread_sigma_min: float = 1.0  # floor on spread sigma to prevent collapse

    @property
    def d_matchup(self) -> int:
        """Total matchup dimension: concat + diff + hadamard."""
        return 4 * self.d_team  # 512

    @property
    def d_l4_input(self) -> int:
        """Input to the prediction MLP."""
        return self.d_matchup + self.d_context_proj  # 576
