"""
L2 Player Synergy Network.

Architecture (4 stages):
  Stage 1: Archetype Interaction Matrix — dense K×K prior for all pairs
  Stage 2: FM Pairwise Residual — hybrid MLP + gated residual per player
  Stage 3: GATv2 Message Passing — contextualizes each player within lineup
  Stage 4: Gated Attention Pooling — aggregates to team-level representation

Input: frozen L1 ability vectors for active players on each team
Output: 134-d team vector (64 player + 64 synergy + 6 meta scalars)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .l2_config import L2Config


class ArchetypeInteractionMatrix(nn.Module):
    """
    Stage 1: Learned K×K symmetric interaction matrix.
    Captures archetype-level synergy priors (e.g., "two ball-dominant guards = negative").
    """

    def __init__(self, cfg: L2Config):
        super().__init__()
        K = cfg.n_archetypes
        # Store upper triangle + diagonal as parameters, reconstruct symmetric matrix
        self.K = K
        n_params = K * (K + 1) // 2  # 55 for K=10
        self.triu_params = nn.Parameter(torch.zeros(n_params))

    def _get_matrix(self) -> torch.Tensor:
        """Reconstruct symmetric K×K matrix from upper triangle params."""
        M = torch.zeros(self.K, self.K, device=self.triu_params.device)
        idx = torch.triu_indices(self.K, self.K)
        M[idx[0], idx[1]] = self.triu_params
        M = M + M.T - torch.diag(M.diagonal())  # symmetrize
        return M

    def forward(self, archetypes: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            archetypes: (B, A, K) soft archetype weights for A players
            mask: (B, A) True = valid player
        Returns:
            pairwise_scores: (B, A, A) archetype synergy scores
        """
        M = self._get_matrix()  # (K, K)
        # arch_i @ M @ arch_j.T for all pairs
        # (B, A, K) @ (K, K) = (B, A, K)
        projected = torch.einsum("bak,kl->bal", archetypes, M)
        # (B, A, K) @ (B, K, A) = (B, A, A)
        scores = torch.einsum("bak,bck->bac", projected, archetypes)

        # Mask invalid pairs
        pair_mask = mask.unsqueeze(-1) & mask.unsqueeze(-2)  # (B, A, A)
        scores = scores * pair_mask.float()
        return scores


class FMSynergyVectors(nn.Module):
    """
    Stage 2: Hybrid FM synergy vectors.
    MLP(ability) as base + gated per-player residual lookup.
    New players get MLP-only; data-rich players get MLP + residual.
    """

    def __init__(self, cfg: L2Config):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_ability, cfg.fm_hidden),
            nn.GELU(),
            nn.Linear(cfg.fm_hidden, cfg.d_synergy),
        )
        # Residual lookup (initialized to zero — MLP dominates at start)
        self.residual = nn.Embedding(cfg.n_players, cfg.d_synergy)
        nn.init.zeros_(self.residual.weight)

        # Per-player confidence gate (starts low so MLP dominates)
        self.alpha = nn.Embedding(cfg.n_players, 1)
        nn.init.constant_(self.alpha.weight, cfg.fm_residual_init_gate)

    def forward(
        self, ability: torch.Tensor, player_idx: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            ability: (B, A, d_ability) L1 ability vectors
            player_idx: (B, A) player embedding indices, or None for MLP-only
        Returns:
            synergy_vectors: (B, A, d_synergy)
        """
        base = self.mlp(ability)  # (B, A, d_syn)

        if player_idx is not None:
            res = self.residual(player_idx)  # (B, A, d_syn)
            gate = torch.sigmoid(self.alpha(player_idx))  # (B, A, 1)
            return base + gate * res

        return base

    def pairwise_scores(
        self, synergy_vectors: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute FM inner product for all pairs.
        Args:
            synergy_vectors: (B, A, d_syn)
            mask: (B, A)
        Returns:
            pairwise_scores: (B, A, A)
        """
        # (B, A, d) @ (B, d, A) = (B, A, A)
        scores = torch.bmm(synergy_vectors, synergy_vectors.transpose(1, 2))
        # Normalize by sqrt(d) for stable gradients
        scores = scores / math.sqrt(synergy_vectors.shape[-1])

        pair_mask = mask.unsqueeze(-1) & mask.unsqueeze(-2)
        scores = scores * pair_mask.float()
        return scores


class SynergyGATv2Layer(nn.Module):
    """
    Stage 3: GATv2 message passing layer (within-team only).
    Each player's representation is updated based on teammates,
    weighted by attention that considers edge features (synergy scores,
    shared minutes, positional overlap, etc.).
    """

    def __init__(self, cfg: L2Config):
        super().__init__()
        d = cfg.d_ability
        n_heads = cfg.n_gat_heads
        d_head = d // n_heads
        assert (
            d % n_heads == 0
        ), f"d_ability ({d}) must be divisible by n_heads ({n_heads})"

        self.n_heads = n_heads
        self.d_head = d_head

        # Per-head projections
        self.W_q = nn.Linear(d, d, bias=False)
        self.W_k = nn.Linear(d, d, bias=False)
        self.W_v = nn.Linear(d, d, bias=False)

        # Edge feature projection into attention space
        self.edge_proj = nn.Linear(cfg.n_edge_features, d)

        # Attention scoring (GATv2: apply nonlinearity BEFORE dot product)
        self.attn_proj = nn.Linear(d_head, 1, bias=False)

        # Output projection
        self.out_proj = nn.Linear(d, d)
        self.layer_norm = nn.LayerNorm(d)
        self.dropout = nn.Dropout(cfg.gat_dropout)

    def forward(
        self,
        h: torch.Tensor,
        edge_features: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            h: (B, A, d) player representations
            edge_features: (B, A, A, n_edge) pairwise features
            mask: (B, A) True = valid player
        Returns:
            h_new: (B, A, d) updated representations (residual)
        """
        B, A, d = h.shape
        H = self.n_heads
        d_h = self.d_head

        # Project queries, keys, values
        q = self.W_q(h).view(B, A, H, d_h)  # (B, A, H, d_h)
        k = self.W_k(h).view(B, A, H, d_h)
        v = self.W_v(h).view(B, A, H, d_h)

        # Edge features → per-head contribution
        edge_h = self.edge_proj(edge_features).view(
            B, A, A, H, d_h
        )  # (B, A, A, H, d_h)

        # GATv2 attention: a(LeakyReLU(q_i + k_j + edge_ij))
        # Expand q and k for pairwise computation
        q_exp = q.unsqueeze(2).expand(B, A, A, H, d_h)  # (B, A, A, H, d_h)
        k_exp = k.unsqueeze(1).expand(B, A, A, H, d_h)

        attn_input = F.leaky_relu(q_exp + k_exp + edge_h, 0.2)
        attn_scores = self.attn_proj(attn_input).squeeze(-1)  # (B, A, A, H)

        # Mask: invalid players get -inf attention
        pair_mask = mask.unsqueeze(-1) & mask.unsqueeze(-2)  # (B, A, A)
        attn_scores = attn_scores.masked_fill(~pair_mask.unsqueeze(-1), float("-inf"))

        # Softmax over neighbors (dim=2: attending over j for each i)
        attn_weights = F.softmax(attn_scores, dim=2)  # (B, A, A, H)
        # NaN fix: invalid players have all -inf neighbors, softmax gives NaN.
        # Replace NaN with 0 (these positions are masked out downstream anyway).
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        attn_weights = self.dropout(attn_weights)

        # Weighted sum of values
        # (B, A, A, H) x (B, A, H, d_h) → (B, A, H, d_h)
        v_exp = v.unsqueeze(1).expand(B, A, A, H, d_h)
        context = (attn_weights.unsqueeze(-1) * v_exp).sum(dim=2)  # (B, A, H, d_h)

        # Reshape and project
        context = context.reshape(B, A, d)
        out = self.out_proj(context)
        out = self.dropout(out)

        # Residual + LayerNorm
        h_new = self.layer_norm(h + out)
        return h_new


class GatedAttentionPooling(nn.Module):
    """
    Stage 4a: Aggregate player-level representations into team vector.
    Uses gated attention (Ilse et al., 2018) — not all players contribute equally.
    """

    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(d_in, 1), nn.Sigmoid())
        self.attn = nn.Linear(d_in, 1)
        self.proj = nn.Linear(d_in, d_out)

    @property
    def d_out(self) -> int:
        return self.proj.out_features

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: (B, A, d_in) player representations
            mask: (B, A) True = valid player
        Returns:
            team_vec: (B, d_out)
        """
        # NaN guard: if entire batch has no valid players, return zeros
        if not mask.any():
            return torch.zeros(h.shape[0], self.d_out, device=h.device)

        gate = self.gate(h)  # (B, A, 1)
        attn_logits = self.attn(h)  # (B, A, 1)

        # Mask invalid players
        attn_logits = attn_logits.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        attn_weights = F.softmax(attn_logits, dim=1)  # (B, A, 1)

        # Per-sample NaN guard: samples where ALL players are masked get
        # all-inf logits, producing NaN after softmax.  Zero them out so the
        # downstream projection returns zeros for those samples.
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        # Gated attention: element-wise gate * normalized attention
        combined = gate * attn_weights  # (B, A, 1)
        pooled = (combined * h).sum(dim=1)  # (B, d_in)

        return self.proj(pooled)  # (B, d_out)


class SynergyAggregation(nn.Module):
    """
    Stage 4b: Aggregate pairwise synergy scores into team synergy vector.
    """

    def __init__(self, cfg: L2Config):
        super().__init__()
        # Input: sum of pairwise features + statistics
        # For each pair: archetype_syn (1) + fm_syn (1) + total_syn (1) = 3
        # Aggregate: sum, mean, max of each → 9 features
        # Plus: n_pairs, synergy_std → 11 total
        self.mlp = nn.Sequential(
            nn.Linear(11, cfg.synergy_mlp_hidden),
            nn.GELU(),
            nn.Linear(cfg.synergy_mlp_hidden, cfg.d_team_synergy),
        )

    def forward(
        self,
        archetype_scores: torch.Tensor,
        fm_scores: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            archetype_scores: (B, A, A) from Stage 1
            fm_scores: (B, A, A) from Stage 2
            mask: (B, A) True = valid
        Returns:
            team_synergy: (B, d_team_synergy)
        """
        total = archetype_scores + fm_scores  # (B, A, A)
        pair_mask = mask.unsqueeze(-1) & mask.unsqueeze(-2)  # (B, A, A)

        # Zero diagonal (self-interaction is meaningless)
        diag_mask = ~torch.eye(
            total.shape[1], dtype=torch.bool, device=total.device
        ).unsqueeze(0)
        valid = pair_mask & diag_mask

        # Aggregate statistics
        n_pairs = valid.float().sum(dim=(1, 2)).clamp(min=1)  # (B,)

        def masked_stat(x, m):
            x_masked = x.masked_fill(~m, 0.0)
            s = x_masked.sum(dim=(1, 2))
            mean = s / n_pairs
            max_val = x_masked.masked_fill(~m, float("-inf")).amax(dim=(1, 2))
            max_val = max_val.clamp(min=-10.0)  # handle all-masked
            return s, mean, max_val

        arch_sum, arch_mean, arch_max = masked_stat(archetype_scores, valid)
        fm_sum, fm_mean, fm_max = masked_stat(fm_scores, valid)
        total_sum, total_mean, total_max = masked_stat(total, valid)

        # Variance of total synergy
        total_var = (
            (total - total_mean.unsqueeze(1).unsqueeze(2)).pow(2) * valid.float()
        ).sum(dim=(1, 2)) / n_pairs

        features = torch.stack(
            [
                arch_mean,
                arch_max,
                fm_mean,
                fm_max,
                total_mean,
                total_max,
                total_sum,
                # clamp: sqrt'(0) = inf -> NaN grads when var is 0 (<= 2 players)
                total_var.clamp(min=1e-8).sqrt(),
                n_pairs / 100.0,  # normalized pair count
                fm_sum,
                arch_sum,
            ],
            dim=-1,
        )  # (B, 11)

        return self.mlp(features)  # (B, d_team_synergy)


class PlayerSynergyNetwork(nn.Module):
    """
    Level 2: Complete Player Synergy Network.

    Takes frozen L1 vectors for a team's active roster, models pairwise
    interactions, and produces a team-level representation.

    Output: 134-d per team (64 player + 64 synergy + 6 meta scalars)
    """

    def __init__(self, cfg: L2Config):
        super().__init__()
        self.cfg = cfg

        # Stage 1: Archetype interaction
        self.archetype_interaction = ArchetypeInteractionMatrix(cfg)

        # Stage 2: FM pairwise
        self.fm_synergy = FMSynergyVectors(cfg)

        # Stage 3: GATv2 message passing
        self.gat = SynergyGATv2Layer(cfg)

        # Stage 4a: Player aggregation
        self.player_pool = GatedAttentionPooling(cfg.d_ability, cfg.d_team_player)

        # Stage 4b: Synergy aggregation
        self.synergy_agg = SynergyAggregation(cfg)

    def forward(
        self,
        ability: torch.Tensor,
        uncertainty: torch.Tensor,
        archetypes: torch.Tensor,
        mask: torch.Tensor,
        player_idx: torch.Tensor | None = None,
        edge_features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            ability: (B, A, d_ability) L1 ability vectors
            uncertainty: (B, A, d_ability) L1 uncertainty (P diagonal)
            archetypes: (B, A, K) soft archetype weights
            mask: (B, A) True = valid player
            player_idx: (B, A) embedding indices for FM residual lookup
            edge_features: (B, A, A, n_edge) precomputed edge features
                           (shared_minutes, years_together, etc.)
        Returns:
            dict with:
                team_vector: (B, d_l2_output) = (B, 134)
                team_player: (B, d_team_player) = (B, 64)
                team_synergy: (B, d_team_synergy) = (B, 64)
                team_meta: (B, n_meta) = (B, 6)
                player_vectors: (B, A, d_ability) context-adjusted per-player
                archetype_scores: (B, A, A) pairwise archetype synergy
                fm_scores: (B, A, A) pairwise FM synergy
                pairwise_total: (B, A, A) total pairwise synergy
        """
        B, A, d = ability.shape

        # Stage 1: Archetype interaction
        arch_scores = self.archetype_interaction(archetypes, mask)  # (B, A, A)

        # Stage 2: FM pairwise
        syn_vectors = self.fm_synergy(ability, player_idx)  # (B, A, d_syn)
        fm_scores = self.fm_synergy.pairwise_scores(syn_vectors, mask)  # (B, A, A)

        # Build edge features for GATv2
        if edge_features is None:
            # Minimal edge features: archetype + fm scores only
            edge_features = torch.stack(
                [arch_scores, fm_scores], dim=-1
            )  # (B, A, A, 2)
            # Pad to n_edge_features
            pad = torch.zeros(
                B, A, A, self.cfg.n_edge_features - 2, device=ability.device
            )
            edge_features = torch.cat([edge_features, pad], dim=-1)
        else:
            # Inject computed archetype and fm scores into first 2 positions
            edge_features = edge_features.clone()
            edge_features[:, :, :, 0] = arch_scores
            edge_features[:, :, :, 1] = fm_scores

        # Stage 3: GATv2 message passing
        h = self.gat(ability, edge_features, mask)  # (B, A, d_ability)

        # Stage 4a: Player aggregation
        team_player = self.player_pool(h, mask)  # (B, d_team_player)

        # Stage 4b: Synergy aggregation
        team_synergy = self.synergy_agg(
            arch_scores, fm_scores, mask
        )  # (B, d_team_synergy)

        # Meta features (computed, not learned)
        team_meta = self._compute_meta(ability, uncertainty, archetypes, mask)  # (B, 6)

        # Combined output
        team_vector = torch.cat(
            [team_player, team_synergy, team_meta], dim=-1
        )  # (B, 134)

        return {
            "team_vector": team_vector,
            "team_player": team_player,
            "team_synergy": team_synergy,
            "team_meta": team_meta,
            "player_vectors": h,
            "archetype_scores": arch_scores,
            "fm_scores": fm_scores,
            "pairwise_total": arch_scores + fm_scores,
        }

    def _compute_meta(
        self,
        ability: torch.Tensor,
        uncertainty: torch.Tensor,
        archetypes: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute 6 meta scalar features from player data."""
        B = ability.shape[0]
        mask_f = mask.float()  # (B, A)
        n_players = mask_f.sum(dim=1, keepdim=True).clamp(min=1)  # (B, 1)

        # 1. Team uncertainty mean
        unc_norms = uncertainty.mean(dim=-1)  # (B, A) mean across dims
        unc_mean = (unc_norms * mask_f).sum(dim=1, keepdim=True) / n_players  # (B, 1)

        # 2. Number of active players (normalized)
        n_active = n_players / 13.0  # normalize to ~[0.6, 1.0]

        # 3. Ability std (star-heavy vs balanced)
        ability_norms = ability.norm(dim=-1)  # (B, A)
        ability_mean = (ability_norms * mask_f).sum(dim=1, keepdim=True) / n_players
        ability_var = ((ability_norms - ability_mean).pow(2) * mask_f).sum(
            dim=1, keepdim=True
        ) / n_players
        ability_std = ability_var.sqrt()  # (B, 1)

        # 4. Archetype entropy
        avg_arch = (archetypes * mask_f.unsqueeze(-1)).sum(dim=1) / n_players  # (B, K)
        arch_entropy = -(avg_arch * (avg_arch + 1e-8).log()).sum(
            dim=-1, keepdim=True
        )  # (B, 1)

        # 5. Dominant archetype weight
        arch_dominant = avg_arch.max(dim=-1, keepdim=True).values  # (B, 1)

        # 6. Max player ability norm (star power)
        max_ability = (ability_norms * mask_f).max(dim=1, keepdim=True).values  # (B, 1)
        max_ability = max_ability / 10.0  # normalize

        meta = torch.cat(
            [unc_mean, n_active, ability_std, arch_entropy, arch_dominant, max_ability],
            dim=-1,
        )  # (B, 6)
        return meta

    def predict_pairwise_synergy(
        self,
        ability: torch.Tensor,
        archetypes: torch.Tensor,
        player_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Predict pairwise synergy for a pair of players (for 2-man loss).
        Args:
            ability: (B, 2, d_ability) two players' abilities
            archetypes: (B, 2, K) two players' archetypes
            player_idx: (B, 2) embedding indices
        Returns:
            synergy: (B,) predicted pairwise synergy score
        """
        mask = torch.ones(ability.shape[0], 2, dtype=torch.bool, device=ability.device)

        # Archetype synergy
        arch_scores = self.archetype_interaction(archetypes, mask)  # (B, 2, 2)
        arch_syn = arch_scores[:, 0, 1]  # (B,) — the off-diagonal element

        # FM synergy
        syn_vecs = self.fm_synergy(ability, player_idx)  # (B, 2, d_syn)
        fm_syn = (syn_vecs[:, 0] * syn_vecs[:, 1]).sum(dim=-1) / math.sqrt(
            syn_vecs.shape[-1]
        )  # (B,)

        return arch_syn + fm_syn


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
