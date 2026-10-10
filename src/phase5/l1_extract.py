"""Run the L1 (NKE-H) Kalman model over a player's game sequence.

Shared by scripts/extract_l1_vectors.py (all players) and the live L1 updater
(new players, and new games for existing ones). Sequences come from
src.phase5.cache_builder.build_player_cache, so inputs are built one way.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from src.phase5.dataset import Normalizer
from src.phase5.model import NKEH

# Maximum sequence length per forward pass (must match training)
MAX_LEN = 300


def _age_and_gap(ctx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Raw (unnormalized) context col 0 (age, scaled as in training) and col 8
    (days since last game)."""
    return (ctx[:, 0:1] - 8.0) / 8.0, ctx[:, 8:9].clone()


def _normalized_inputs(seq: dict, normalizer: Normalizer, start: int, end: int):
    box = normalizer.normalize_box(
        torch.tensor(seq["box_stats"][start:end], dtype=torch.float32)
    )
    pbp = normalizer.normalize_pbp(
        torch.tensor(seq["pbp_stats"][start:end], dtype=torch.float32)
    )
    # Games without PBP: zero after normalization, as in training
    pbp[~torch.tensor(seq["has_pbp"][start:end], dtype=torch.bool)] = 0.0
    ctx_raw = torch.tensor(seq["context"][start:end], dtype=torch.float32)
    return box, pbp, ctx_raw, normalizer.normalize_context(ctx_raw)


def prepare_player_inputs(
    seq: dict,
    normalizer: Normalizer,
    profile_vec: torch.Tensor,
    start: int,
    end: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Model inputs for games [start:end], padded to MAX_LEN, batch dim 1."""
    T = end - start
    box, pbp, ctx_raw, ctx = _normalized_inputs(seq, normalizer, start, end)
    age, days_gap = _age_and_gap(ctx_raw)
    init_ctx = normalizer.normalize_context(
        torch.tensor(seq["context"][0], dtype=torch.float32)
    )

    mask = torch.zeros(MAX_LEN, dtype=torch.bool)
    mask[:T] = True
    pad = MAX_LEN - T
    if pad > 0:
        box, pbp, ctx, age, days_gap = (
            torch.cat([x, torch.zeros(pad, x.shape[1])], dim=0)
            for x in (box, pbp, ctx, age, days_gap)
        )

    return {
        "box_stats_seq": box.unsqueeze(0).to(device),
        "pbp_stats_seq": pbp.unsqueeze(0).to(device),
        "context_seq": ctx.unsqueeze(0).to(device),
        "profile": normalizer.normalize_profile(profile_vec).unsqueeze(0).to(device),
        "age_seq": age.unsqueeze(0).to(device),
        "seq_mask": mask.unsqueeze(0).to(device),
        "days_gap_seq": days_gap.unsqueeze(0).to(device),
        "init_context": init_ctx.unsqueeze(0).to(device),
    }


def extract_player(
    seq: dict,
    model: NKEH,
    normalizer: Normalizer,
    profile_vec: torch.Tensor,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Kalman-filter a player's whole career: ability and uncertainty after
    each game, plus archetype weights.

    Careers longer than MAX_LEN use windows that step by MAX_LEN // 2; each
    game's vector comes from the first window covering it beyond the overlap,
    so every game has at least MAX_LEN // 2 games of context.
    """
    N = len(seq["game_ids"])
    ability = np.zeros((N, 32), dtype=np.float32)
    P = np.zeros((N, 32), dtype=np.float32)
    done = np.zeros(N, dtype=bool)
    step = MAX_LEN // 2
    arch = None

    window_start = 0
    while window_start < N:
        window_end = min(window_start + MAX_LEN, N)
        inputs = prepare_player_inputs(
            seq, normalizer, profile_vec, window_start, window_end, device
        )
        with torch.no_grad():
            out = model.forward_sequence(**inputs)
        first = 0 if window_start == 0 else step
        for local in range(first, window_end - window_start):
            g = window_start + local
            if not done[g]:
                ability[g] = out["ability"][0, local].cpu().numpy()
                P[g] = out["P"][0, local].cpu().numpy()
                done[g] = True
        arch = out["archetype_weights"][0].cpu().numpy()
        if window_end == N:
            break
        window_start += step

    return {"ability": ability, "uncertainty": P, "archetype_weights": arch}


def kalman_steps(
    seq: dict,
    start: int,
    end: int,
    mu: np.ndarray,
    P: np.ndarray,
    model: NKEH,
    normalizer: Normalizer,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Continue the filter from state (mu, P) through games [start:end]:
    one predict + update per game. Returns ability and P after each game."""
    box, pbp, ctx_raw, ctx = _normalized_inputs(seq, normalizer, start, end)
    age, days_gap = _age_and_gap(ctx_raw)
    mu_t = torch.tensor(mu, dtype=torch.float32, device=device).unsqueeze(0)
    P_t = torch.tensor(P, dtype=torch.float32, device=device).unsqueeze(0)
    abilities, Ps = [], []
    with torch.no_grad():
        for k in range(end - start):
            mu_pred, P_pred = model.kalman_predict(
                mu_t, P_t, age[k : k + 1].to(device), days_gap[k : k + 1].to(device)
            )
            mu_obs, log_sigma_obs = model.game_encoder(
                box[k : k + 1].to(device),
                pbp[k : k + 1].to(device),
                ctx[k : k + 1].to(device),
                mu_pred,
            )
            mu_t, P_t = model.kalman_update(
                mu_pred, P_pred, mu_obs, F.softplus(log_sigma_obs)
            )
            abilities.append(mu_t[0].cpu().numpy())
            Ps.append(P_t[0].cpu().numpy())
    return np.array(abilities, dtype=np.float32), np.array(Ps, dtype=np.float32)
