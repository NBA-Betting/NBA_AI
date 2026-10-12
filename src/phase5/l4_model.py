"""
L4 Game Context and Prediction model.

Architecture:
  1. MatchupConstructor: concat + diff + hadamard from two L3 team vectors → 512-d
  2. ContextEncoder: project 14-d game context features → 64-d
  3. L2SkipConnection: project L2 home-away diff → 256-d (additive after first layer)
  4. Prediction MLP: Linear(576, 256) → 2x ResidualBlock(256, 512) → 256-d
  5. PredictionHeads: spread (mu, sigma), win_prob

Input: L3 team_home (128-d), L3 team_away (128-d), L2 home (134-d), L2 away (134-d),
       game context (14-d)
Output: spread_mu, spread_sigma, win_prob
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .l4_config import L4Config


class MatchupConstructor(nn.Module):
    """
    Build 512-d matchup representation from two 128-d team vectors.

    Uses the NLI triple representation:
      - concat(home, away) = 256-d: preserves absolute team strength
      - home - away = 128-d: relative advantage (natural spread form)
      - home * away = 128-d: interaction/style matchup effects
    """

    def forward(self, team_home: torch.Tensor, team_away: torch.Tensor) -> torch.Tensor:
        """
        Args:
            team_home: (B, d_team) L3 home team representation
            team_away: (B, d_team) L3 away team representation
        Returns:
            matchup: (B, 4 * d_team) = (B, 512)
        """
        concat = torch.cat([team_home, team_away], dim=-1)  # (B, 256)
        diff = team_home - team_away  # (B, 128)
        hadamard = team_home * team_away  # (B, 128)
        return torch.cat([concat, diff, hadamard], dim=-1)  # (B, 512)


class ContextEncoder(nn.Module):
    """
    Project 14-d game context features to 64-d embedding.

    Context features (14 total):
      home_flag (1), arena_altitude (1),
      rest_days_home/away (2), is_b2b_home/away (2), games_7d_home/away (2),
      travel_dist_home/away (2), tz_crossings_home/away (2),
      season_progress (1), is_playoffs (1)
    """

    def __init__(self, cfg: L4Config):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(cfg.n_context, cfg.d_context_proj),
            nn.GELU(),
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            context: (B, n_context) = (B, 14)
        Returns:
            context_proj: (B, d_context_proj) = (B, 64)
        """
        return self.proj(context)


class L2SkipConnection(nn.Module):
    """
    Project L2 home-away difference for additive injection into the hidden layer.

    This skip connection lets L4 directly access L2-level team composition
    signals (player aggregation + synergy + meta) without them being
    compressed through L3's residual bottleneck.
    """

    def __init__(self, cfg: L4Config):
        super().__init__()
        self.proj = nn.Linear(cfg.d_l2, cfg.d_l2_skip)

    def forward(self, l2_home: torch.Tensor, l2_away: torch.Tensor) -> torch.Tensor:
        """
        Args:
            l2_home: (B, d_l2) = (B, 134)
            l2_away: (B, d_l2) = (B, 134)
        Returns:
            skip: (B, d_l2_skip) = (B, 256)
        """
        l2_diff = l2_home - l2_away  # (B, 134)
        return self.proj(l2_diff)  # (B, 256)


class ResidualBlock(nn.Module):
    """
    Pre-norm residual block: LayerNorm → Linear → GELU → Dropout → Linear → add.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ff(self.norm(x))


class PredictionHeads(nn.Module):
    """
    Multi-task prediction heads from shared 256-d representation.

    Outputs:
      - spread_mu: predicted margin (positive = home advantage)
      - spread_sigma: uncertainty (Softplus)
      - win_prob: P(home win) via sigmoid
    """

    def __init__(self, cfg: L4Config):
        super().__init__()
        d = cfg.d_hidden
        self.sigma_min_spread = cfg.spread_sigma_min

        # Spread
        self.spread_mu_head = nn.Linear(d, 1)
        self.spread_sigma_head = nn.Linear(d, 1)

        # Win probability
        self.win_logit_head = nn.Linear(d, 1)

    def forward(self, h: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            h: (B, d_hidden) shared L4 representation
        Returns:
            dict with spread_mu, spread_sigma, win_logit, win_prob, all (B,)
        """
        spread_mu = self.spread_mu_head(h).squeeze(-1)  # (B,)
        spread_sigma = (
            F.softplus(self.spread_sigma_head(h).squeeze(-1)) + self.sigma_min_spread
        )  # (B,)

        # Win probability
        win_logit = self.win_logit_head(h).squeeze(-1)  # (B,)
        win_prob = torch.sigmoid(win_logit)  # (B,)

        return {
            "spread_mu": spread_mu,
            "spread_sigma": spread_sigma,
            "win_logit": win_logit,
            "win_prob": win_prob,
        }


class GamePredictor(nn.Module):
    """
    Level 4: Full game prediction model.

    Takes L3 team representations (128-d each), L2 team vectors (134-d each),
    and game context (14-d) to produce spread and win probability predictions
    with calibrated uncertainty.

    Architecture:
      matchup (512-d) + context_proj (64-d) → Linear(576, 256)
      + L2 skip (additive 256-d)
      → 2x ResidualBlock(256, 512) → PredictionHeads
    """

    def __init__(self, cfg: L4Config | None = None):
        super().__init__()
        if cfg is None:
            cfg = L4Config()
        self.cfg = cfg

        # Components
        self.matchup = MatchupConstructor()
        self.context_encoder = ContextEncoder(cfg)
        self.l2_skip = L2SkipConnection(cfg)
        # Learnable scaling factor for L2 skip — init 0.3 so L3 pathway dominates
        self.skip_scale = nn.Parameter(torch.tensor(0.3))

        # Prediction MLP
        self.input_proj = nn.Linear(cfg.d_l4_input, cfg.d_hidden)
        self.input_norm = nn.LayerNorm(cfg.d_hidden)
        self.input_dropout = nn.Dropout(cfg.dropout)

        self.residual_blocks = nn.ModuleList(
            [
                ResidualBlock(cfg.d_hidden, cfg.d_ff, cfg.dropout)
                for _ in range(cfg.n_residual_blocks)
            ]
        )

        self.output_norm = nn.LayerNorm(cfg.d_hidden)

        # Heads
        self.heads = PredictionHeads(cfg)

    def forward(
        self,
        team_home: torch.Tensor,
        team_away: torch.Tensor,
        l2_home: torch.Tensor,
        l2_away: torch.Tensor,
        context: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Full L4 forward pass.

        Args:
            team_home: (B, 128) L3 home team representation
            team_away: (B, 128) L3 away team representation
            l2_home: (B, 134) L2 home team vector
            l2_away: (B, 134) L2 away team vector
            context: (B, 14) game-specific context features
        Returns:
            dict with spread_mu, spread_sigma, win_logit, win_prob — all (B,)
        """
        # 1. Build matchup representation
        matchup = self.matchup(team_home, team_away)  # (B, 512)

        # 2. Encode context
        ctx = self.context_encoder(context)  # (B, 64)

        # 3. Combine matchup + context
        l4_input = torch.cat([matchup, ctx], dim=-1)  # (B, 576)

        # 4. Project to hidden dim
        h = self.input_proj(l4_input)  # (B, 256)

        # 5. Additive L2 skip connection after first projection (scaled down)
        l2_skip = self.l2_skip(l2_home, l2_away)  # (B, 256)
        h = h + self.skip_scale * l2_skip

        # 6. Normalize and activate
        h = self.input_dropout(F.gelu(self.input_norm(h)))

        # 7. Residual blocks
        for block in self.residual_blocks:
            h = block(h)

        # 8. Output normalization
        h = self.output_norm(h)

        # 9. Prediction heads
        return self.heads(h)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
