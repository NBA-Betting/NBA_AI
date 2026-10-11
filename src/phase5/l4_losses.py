"""
Loss functions for L4 Game Prediction training.

Multi-task composite loss:
  Primary (w=1.0): Gaussian NLL for spread prediction
  Auxiliary (w=0.3): BCE for win probability (label smoothing 0.05)
  Consistency (w=0.1): MSE between Phi(mu/sigma) and win_prob
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from .l4_config import L4Config

# Standard normal CDF via error function
_LOG_SQRT_2 = 0.5 * math.log(2.0)
_INV_SQRT_2 = 1.0 / math.sqrt(2.0)


def _normal_cdf(x: torch.Tensor) -> torch.Tensor:
    """Standard normal CDF: Phi(x) = 0.5 * (1 + erf(x / sqrt(2)))."""
    return 0.5 * (1.0 + torch.erf(x * _INV_SQRT_2))


def gaussian_nll(
    mu: torch.Tensor,
    sigma: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    Gaussian negative log-likelihood: -log N(target | mu, sigma^2).

    Args:
        mu: (B,) predicted mean
        sigma: (B,) predicted standard deviation (must be > 0)
        target: (B,) actual value
    Returns:
        scalar mean loss
    """
    # -log N(x|mu,sigma^2) = 0.5*log(2*pi) + log(sigma) + 0.5*((x-mu)/sigma)^2
    variance = sigma.pow(2)
    nll = (
        0.5 * math.log(2.0 * math.pi)
        + torch.log(sigma)
        + 0.5 * (target - mu).pow(2) / variance
    )
    return nll.mean()


def spread_loss(
    spread_mu: torch.Tensor,
    spread_sigma: torch.Tensor,
    actual_margin: torch.Tensor,
) -> torch.Tensor:
    """
    Gaussian NLL for spread prediction.

    Args:
        spread_mu: (B,) predicted margin (positive = home advantage)
        spread_sigma: (B,) predicted uncertainty
        actual_margin: (B,) actual home margin (home_score - away_score)
    Returns:
        scalar loss
    """
    return gaussian_nll(spread_mu, spread_sigma, actual_margin)


def win_loss(
    win_logit: torch.Tensor,
    actual_home_win: torch.Tensor,
    label_smoothing: float = 0.05,
) -> torch.Tensor:
    """
    Binary cross-entropy for win probability with label smoothing.

    Label smoothing (eps=0.05) maps hard 0/1 targets to [0.025, 0.975],
    preventing overconfident logits and improving calibration.

    Args:
        win_logit: (B,) raw logit (before sigmoid)
        actual_home_win: (B,) 1.0 if home won, 0.0 otherwise
        label_smoothing: smoothing factor (default 0.05)
    Returns:
        scalar loss
    """
    smoothed = actual_home_win * (1.0 - label_smoothing) + 0.5 * label_smoothing
    return F.binary_cross_entropy_with_logits(win_logit, smoothed)


def consistency_loss(
    spread_mu: torch.Tensor,
    spread_sigma: torch.Tensor,
    win_prob: torch.Tensor,
) -> torch.Tensor:
    """
    Consistency regularization between Gaussian-derived and direct win probabilities.

    The Gaussian spread model implies P(home_win) = Phi(spread_mu / spread_sigma).
    This should agree with the directly predicted win_prob. We stop gradient on
    the Gaussian side so it acts as a soft target for the win head.

    Args:
        spread_mu: (B,) predicted margin
        spread_sigma: (B,) predicted uncertainty
        win_prob: (B,) directly predicted win probability (sigmoid output)
    Returns:
        scalar MSE loss
    """
    gaussian_win_prob = _normal_cdf(spread_mu / spread_sigma)
    return F.mse_loss(win_prob, gaussian_win_prob.detach())


def game_prediction_loss(
    predictions: dict[str, torch.Tensor],
    actual_margin: torch.Tensor,
    actual_home_win: torch.Tensor,
    cfg: L4Config | None = None,
) -> dict[str, torch.Tensor]:
    """
    Combined L4 multi-task loss.

    Args:
        predictions: dict from GamePredictor.forward() with keys:
            spread_mu, spread_sigma, win_logit, win_prob
        actual_margin: (B,) home_score - away_score
        actual_home_win: (B,) 1.0 if home won, 0.0 otherwise
        cfg: L4Config for loss weights (uses defaults if None)
    Returns:
        dict with individual components and total:
            spread, win, consistency, total_loss
    """
    if cfg is None:
        cfg = L4Config()

    L_spread = spread_loss(
        predictions["spread_mu"],
        predictions["spread_sigma"],
        actual_margin,
    )

    L_win = win_loss(
        predictions["win_logit"],
        actual_home_win,
    )

    L_consist = consistency_loss(
        predictions["spread_mu"],
        predictions["spread_sigma"],
        predictions["win_prob"],
    )

    L_combined = (
        cfg.w_spread * L_spread
        + cfg.w_win * L_win
        + cfg.w_consistency * L_consist
    )

    return {
        "total_loss": L_combined,
        "spread": L_spread,
        "win": L_win,
        "consistency": L_consist,
    }


def prediction_metrics(
    all_preds: dict[str, list],
    all_targets: dict[str, list],
) -> dict[str, float]:
    """Evaluation metrics from accumulated per-batch predictions and targets
    (keys spread_mu, spread_sigma, win_prob / margin, win)."""
    spread_mu = np.concatenate(all_preds["spread_mu"])
    spread_sigma = np.concatenate(all_preds["spread_sigma"])
    win_prob = np.concatenate(all_preds["win_prob"])
    actual_margin = np.concatenate(all_targets["margin"])
    actual_win = np.concatenate(all_targets["win"])
    n = len(actual_margin)

    # Win AUC (Wilcoxon-Mann-Whitney, no sklearn dependency)
    order = np.argsort(win_prob)
    sorted_labels = actual_win[order]
    n_pos = int(actual_win.sum())
    n_neg = n - n_pos
    if n_pos > 0 and n_neg > 0:
        cum_pos = np.cumsum(sorted_labels)
        auc = 1.0 - cum_pos[sorted_labels == 0].sum() / (n_pos * n_neg)
    else:
        auc = 0.5

    # Expected calibration error, 10 bins
    edges = np.linspace(0, 1, 11)
    ece = 0.0
    for j in range(10):
        mask = (win_prob >= edges[j]) & (
            (win_prob <= edges[j + 1]) if j == 9 else (win_prob < edges[j + 1])
        )
        if mask.any():
            ece += mask.sum() / n * abs(win_prob[mask].mean() - actual_win[mask].mean())

    return {
        "spread_mae": float(np.abs(spread_mu - actual_margin).mean()),
        "win_acc": float(((win_prob > 0.5) == actual_win).mean()),
        "win_auc": float(auc),
        "ece": float(ece),
        "mean_sigma": float(spread_sigma.mean()),
        "n_games": n,
    }
