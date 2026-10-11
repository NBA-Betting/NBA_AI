"""
Phase 2 Model — Top-level wiring.

Combines all Phase 2 components into a single forward pass:
GameStates encoder → Per-game encoder → Temporal attention → Roster → Fusion → Prediction.
"""

import torch
import torch.nn as nn

from src.transformer.models.prediction_heads import PredictionHeads, GamePrediction
from src.transformer.phase2.config import Phase2ModelConfig
from src.transformer.phase2.models.gamestates_encoder import Phase2GameStatesEncoder
from src.transformer.phase2.models.temporal_attention import Phase2TemporalAttention
from src.transformer.phase2.models.per_game_encoder import PerGameEncoder
from src.transformer.phase2.models.roster_encoder import Phase2RosterEncoder
from src.transformer.phase2.models.fusion import Phase2Fusion
from src.transformer.phase2.models.team_gat import TeamInteractionGAT


class Phase2Model(nn.Module):
    """
    Full Phase 2 NBA prediction model.

    Forward pass per team:
    1. gs_encoder(gs_data, gs_lengths) -> (B, N_recent, 512) dynamics
    2. Scatter dynamics into full game sequence at is_recent positions
    3. per_game_encoder(scores, opp, loc, players, dynamics, is_recent) -> (B, G, 512)
    4. temporal_attention(game_reprs, days_before, game_mask) -> (B, 512)
    5. roster_encoder(roster_ids, form_vectors) -> (B, 512)
    6. rest_embed(rest_days) -> (B, 512)
    7. team_combine: concat([season, roster, rest]) -> Linear(1536, 512) -> LN
    8. fusion(home_repr, away_repr) -> (B, 512)
    9. prediction_heads(matchup) -> GamePrediction
    """

    def __init__(self, config: Phase2ModelConfig):
        super().__init__()
        self.config = config
        h = config.hidden_dim

        # Shared player embedding
        self.player_embed = nn.Embedding(
            config.n_players, config.player_embed_dim, padding_idx=0
        )

        # GameStates encoder
        self.gs_encoder = Phase2GameStatesEncoder(
            hidden_dim=h,
            num_layers=config.gs_encoder_layers,
            num_heads=config.gs_encoder_heads,
            ff_dim=config.gs_encoder_ff_dim,
            dropout=config.gs_encoder_dropout,
            max_seq_len=config.gs_max_seq_len,
        )

        # Team interaction GAT (Phase 3 Exp 6+)
        self.team_gat_enabled = config.enable_team_gat
        self.team_gat = None
        if config.enable_team_gat:
            self.team_gat = TeamInteractionGAT(
                n_teams=config.n_teams,
                hidden_dim=config.team_gat_hidden,
                n_layers=config.team_gat_layers,
                n_heads=config.team_gat_heads,
                dropout=config.team_gat_dropout,
                edge_features=config.h2h_edge_features,
            )

        # Per-game encoder (uses shared player_embed)
        self.per_game_encoder = PerGameEncoder(
            player_embed=self.player_embed,
            hidden_dim=h,
            score_dim=config.score_dim,
            opponent_dim=config.opponent_dim,
            location_dim=config.location_dim,
            contribution_dim=config.player_contribution_dim,
            contribution_heads=config.player_contribution_heads,
            contribution_dropout=config.player_contribution_dropout,
            n_teams=config.n_teams,
            use_ple=config.use_ple,
            n_ple_bins=config.n_ple_bins,
            n_player_stats=config.n_player_stats,
            stat_hidden_dim=config.stat_hidden_dim,
            n_positions=config.n_positions,
            position_dim=config.position_dim,
            interaction_layers=config.player_interaction_layers,
            interaction_heads=config.player_interaction_heads,
            interaction_ff_dim=config.player_interaction_ff_dim,
            interaction_dropout=config.player_interaction_dropout,
            n_pool_queries=config.player_contribution_n_pool_queries,
            has_team_gat=config.enable_team_gat,
            n_efficiency_features=config.n_efficiency_features,
            efficiency_hidden_dim=config.efficiency_hidden_dim,
            gs_summary_dim=config.gs_summary_dim,
            flag_dim=config.flag_dim,
        )

        # Roster-conditioned temporal (Phase 3 Exp 5+)
        self.roster_context_enabled = config.enable_roster_context
        if config.enable_roster_context:
            self.roster_overlap_embed = nn.Embedding(16, h)
            self.traj_query_proj = nn.Sequential(
                nn.Linear(config.player_embed_dim, h),
                nn.LayerNorm(h),
                nn.GELU(),
            )
            self.trajectory_attn = nn.MultiheadAttention(
                h,
                config.roster_context_heads,
                dropout=config.roster_context_dropout,
                batch_first=True,
            )
            self.traj_no_history = nn.Parameter(torch.zeros(h))
            self.reinjection_norm = nn.LayerNorm(h)
            self.reinjection_crossattn = nn.MultiheadAttention(
                h,
                config.roster_context_heads,
                dropout=config.roster_context_dropout,
                batch_first=True,
            )

        # Temporal module (transformer or GRU)
        if config.temporal_type == "gru":
            from src.transformer.phase2.models.temporal_gru import Phase2TemporalGRU

            self.temporal_attention = Phase2TemporalGRU(
                hidden_dim=h,
                gru_hidden=config.temporal_gru_hidden,
                num_layers=config.temporal_gru_layers,
                dropout=config.temporal_gru_dropout,
                max_days=config.temporal_max_days,
                time_dim=config.temporal_gru_time_dim,
                n_pool_queries=config.temporal_n_pool_queries,
                pool_heads=config.temporal_heads,
            )
        else:
            self.temporal_attention = Phase2TemporalAttention(
                hidden_dim=h,
                num_layers=config.temporal_layers,
                num_heads=config.temporal_heads,
                ff_dim=config.temporal_ff_dim,
                dropout=config.temporal_dropout,
                max_days=config.temporal_max_days,
                n_pool_queries=config.temporal_n_pool_queries,
                pos_encoding=config.temporal_pos_encoding,
            )

        # Player form encoder (optional)
        self.form_encoder = None
        form_dim = 0
        if config.enable_player_form:
            from src.transformer.phase2.models.player_form_encoder import (
                PlayerFormEncoder,
            )

            form_dim = config.player_form_dim
            self.form_encoder = PlayerFormEncoder(
                form_dim=form_dim,
                days_embed_dim=config.player_form_days_dim,
                max_days=config.temporal_max_days,
                n_heads=config.player_form_heads,
                ff_dim=config.player_form_ff_dim,
                dropout=config.player_form_dropout,
                n_player_stats=config.n_player_stats,
                stat_hidden_dim=config.stat_hidden_dim,
            )

        # Roster encoder (uses shared player_embed, optionally wider with form)
        self.roster_encoder = Phase2RosterEncoder(
            player_embed=self.player_embed,
            hidden_dim=h,
            n_heads=config.roster_heads,
            n_layers=config.roster_layers,
            ff_dim=config.roster_ff_dim,
            dropout=config.roster_dropout,
            form_dim=form_dim,
        )

        # Rest days embedding (kept at rest_embed_dim, not wastefully projected to h)
        self.rest_embed = nn.Sequential(
            nn.Embedding(config.max_rest_days, config.rest_embed_dim),
        )

        # Season-level efficiency projection (Phase 3 Exp 7+)
        self.season_efficiency_proj = None
        season_eff_dim = 0
        if config.n_efficiency_features > 0:
            season_eff_dim = config.season_efficiency_dim
            self.season_efficiency_proj = nn.Sequential(
                nn.Linear(config.n_efficiency_features, season_eff_dim),
                nn.LayerNorm(season_eff_dim),
                nn.GELU(),
            )

        # Per-team combine: concat([season(h), roster(h), rest(rest_embed_dim), season_eff]) -> h
        team_combine_input = h * 2 + config.rest_embed_dim + season_eff_dim
        self.team_combine = nn.Sequential(
            nn.Linear(team_combine_input, h),
            nn.LayerNorm(h),
        )

        # Fusion
        self.fusion = Phase2Fusion(
            hidden_dim=h,
            dropout=config.fusion_dropout,
            use_cross_attention=config.use_cross_attention_fusion,
            n_heads=config.fusion_heads,
        )

        # Auxiliary matchup features encoder (Phase 6 Exp 1+)
        # Output layer is zero-initialized so the residual starts at zero,
        # preserving pre-trained model behavior at initialization.
        self.aux_encoder = None
        if config.n_aux_features > 0:
            aux_output = nn.Linear(config.aux_hidden_dim, h)
            nn.init.zeros_(aux_output.weight)
            nn.init.zeros_(aux_output.bias)
            self.aux_encoder = nn.Sequential(
                nn.Linear(config.n_aux_features, config.aux_hidden_dim),
                nn.LayerNorm(config.aux_hidden_dim),
                nn.GELU(),
                aux_output,
            )

        # Prediction heads (reuse Phase 1)
        self.prediction_heads = PredictionHeads(
            input_dim=h,
            hidden_dim=config.prediction_hidden_dim,
            spread_min_std=config.spread_min_std,
            spread_max_std=config.spread_max_std,
            score_min_std=config.score_min_std,
            dropout=config.prediction_dropout,
            derive_spread=config.derive_spread_from_scores,
        )

    def _encode_team(self, batch: dict, prefix: str) -> torch.Tensor:
        """Encode one team's full context into a single representation."""
        p = prefix  # "home_" or "away_"

        scores = batch[p + "scores"]
        opponent_ids = batch[p + "opponent_ids"]
        location = batch[p + "location"]
        player_ids = batch[p + "player_ids"]
        player_points = batch[p + "player_points"]
        player_mask = batch[p + "player_mask"]
        days_before = batch[p + "days_before"]
        is_recent = batch[p + "is_recent"]
        game_mask = batch[p + "game_mask"]

        B, G = scores.shape[:2]
        h = self.config.hidden_dim

        # 1. Encode GameStates dynamics for recent games
        gs_data = {
            "periods": batch[p + "gs_periods"],
            "clock_buckets": batch[p + "gs_clock_buckets"],
            "home_score_buckets": batch[p + "gs_home_score_buckets"],
            "away_score_buckets": batch[p + "gs_away_score_buckets"],
            "margin_buckets": batch[p + "gs_margin_buckets"],
        }
        gs_lengths = batch[p + "gs_lengths"]

        dynamics_recent = self.gs_encoder(gs_data, gs_lengths)  # (B, N_recent, h)

        # 2. Scatter dynamics into full game sequence at is_recent positions
        dynamics = torch.zeros(
            B, G, h, device=scores.device, dtype=dynamics_recent.dtype
        )
        for b in range(B):
            recent_positions = is_recent[b].nonzero(as_tuple=True)[0]
            n_avail = min(len(recent_positions), dynamics_recent.shape[1])
            if n_avail > 0:
                dynamics[b, recent_positions[:n_avail]] = dynamics_recent[b, :n_avail]

        # 3. Per-game encoder
        player_stats = batch.get(p + "player_stats")
        player_positions = batch.get(p + "player_positions")
        player_pm_available = batch.get(p + "player_pm_available")

        # Compute H2H team representations if GAT is enabled
        h2h_team_repr = None
        if self.team_gat_enabled and self.team_gat is not None:
            h2h_features = batch.get("h2h_features")
            if h2h_features is not None:
                h2h_team_repr = self.team_gat(h2h_features)  # (B, 30, gat_hidden)

        # Extract efficiency features if present
        efficiency_features = batch.get(p + "efficiency_features")
        gs_summary_features = batch.get(p + "gs_summary_features")
        context_flags = batch.get(p + "context_flags")

        game_reprs = self.per_game_encoder(
            scores=scores,
            opponent_ids=opponent_ids,
            location=location,
            player_ids=player_ids,
            player_points=player_points,
            player_mask=player_mask,
            dynamics=dynamics,
            is_recent=is_recent,
            player_stats=player_stats,
            player_positions=player_positions,
            player_pm_available=player_pm_available,
            h2h_team_repr=h2h_team_repr,
            efficiency_features=efficiency_features,
            gs_summary_features=gs_summary_features,
            context_flags=context_flags,
        )  # (B, G, h)

        # 3b. Roster-conditioned temporal: two-pass heterogeneous message passing
        if self.roster_context_enabled:
            roster_ids = batch[p[:-1] + "_roster"]  # (B, R=15)
            P = roster_ids.shape[1]

            # Build per-player game presence mask: which games did each roster player appear in?
            # player_ids: (B, G, 15) — players per historical game
            # roster_ids: (B, 15) — tonight's roster
            roster_exp = roster_ids.unsqueeze(2).unsqueeze(3)  # (B, R, 1, 1)
            game_exp = player_ids.unsqueeze(1)  # (B, 1, G, 15)
            presence = (roster_exp == game_exp).any(dim=-1)  # (B, R, G)
            # A padded roster slot (id 0) matches padded game slots; it's not a player
            presence = presence & (roster_ids != 0).unsqueeze(-1)

            # Absence mask for attention: absent games + padded games + padded roster slots
            absence = ~presence | game_mask.unsqueeze(1)  # (B, R, G)
            absence = absence | (roster_ids == 0).unsqueeze(-1)  # (B, R, G)

            # Roster overlap embedding: how many roster players appeared in each game
            overlap_count = presence.sum(dim=1).clamp(max=15)  # (B, G)
            game_reprs = game_reprs + self.roster_overlap_embed(overlap_count.long())

            # Pass 1: Game→Player trajectory extraction
            # Player queries: project player embeddings to hidden_dim
            roster_emb = self.traj_query_proj(
                self.player_embed(roster_ids)
            )  # (B, R, h)

            # Expand game_reprs for batched per-player attention
            game_reprs_exp = (
                game_reprs.unsqueeze(1).expand(-1, P, -1, -1).contiguous()
            )  # (B, R, G, h)

            # Batched cross-attention: (B*R, 1, h) queries, (B*R, G, h) keys/values
            BxP = B * P
            traj_out, _ = self.trajectory_attn(
                roster_emb.reshape(BxP, 1, h),
                game_reprs_exp.reshape(BxP, G, h),
                game_reprs_exp.reshape(BxP, G, h),
                key_padding_mask=absence.reshape(BxP, G),
                need_weights=False,
            )
            player_trajectories = traj_out.squeeze(1).reshape(B, P, h)  # (B, R, h)

            # Players with zero historical appearances → learned fallback
            no_games = absence.all(dim=-1)  # (B, R)
            player_trajectories = player_trajectories.masked_fill(
                no_games.unsqueeze(-1), 0
            )
            player_trajectories = (
                player_trajectories
                + no_games.unsqueeze(-1).float() * self.traj_no_history
            )

            # Pass 2: Player→Game re-injection
            game_reprs_norm = self.reinjection_norm(game_reprs)
            reinjected, _ = self.reinjection_crossattn(
                game_reprs_norm,
                player_trajectories,
                player_trajectories,
                key_padding_mask=(roster_ids == 0),  # mask padding roster slots
                need_weights=False,
            )
            game_reprs = game_reprs + reinjected  # residual

        # 4. Temporal attention
        season_repr = self.temporal_attention(
            game_reprs, days_before, game_mask
        )  # (B, h)

        # 5. Roster encoder (with optional form vectors)
        roster_ids = batch[p[:-1] + "_roster"]
        form_vectors = None

        if self.form_encoder is not None:
            form_key = p + "roster_form_points"
            if form_key in batch:
                form_stats = batch.get(p + "roster_form_stats")
                form_pm_avail = batch.get(p + "roster_form_pm_available")
                form_vectors = self.form_encoder(
                    points=batch[p + "roster_form_points"],
                    days=batch[p + "roster_form_days"],
                    mask=batch[p + "roster_form_mask"],
                    stats=form_stats,
                    pm_available=form_pm_avail,
                )  # (B, R, form_dim)

        roster_repr = self.roster_encoder(roster_ids, form_vectors)  # (B, h)

        # 6. Rest days (kept at rest_embed_dim, no wasteful projection to h)
        rest_days = batch[p[:-1] + "_rest_days"]
        rest_repr = self.rest_embed[0](rest_days)  # (B, rest_embed_dim)

        # 7. Per-team combine
        combine_parts = [season_repr, roster_repr, rest_repr]

        # Season-average efficiency features
        if self.season_efficiency_proj is not None and efficiency_features is not None:
            # Compute masked mean across context games
            eff_mask = ~game_mask  # True = valid game
            eff_masked = efficiency_features * eff_mask.unsqueeze(-1).float()
            eff_sum = eff_masked.sum(dim=1)  # (B, n_eff)
            eff_count = eff_mask.sum(dim=1, keepdim=True).float().clamp(min=1)  # (B, 1)
            eff_mean = eff_sum / eff_count  # (B, n_eff)
            season_eff_repr = self.season_efficiency_proj(
                eff_mean
            )  # (B, season_eff_dim)
            combine_parts.append(season_eff_repr)

        team_repr = self.team_combine(torch.cat(combine_parts, dim=-1))  # (B, h)

        return team_repr

    def forward(self, batch: dict) -> GamePrediction:
        """
        Forward pass through the complete Phase 2 model.

        Args:
            batch: Dict with home_ and away_ prefixed tensors

        Returns:
            GamePrediction with all prediction outputs
        """
        home_repr = self._encode_team(batch, "home_")
        away_repr = self._encode_team(batch, "away_")

        matchup_repr = self.fusion(home_repr, away_repr)

        # Inject auxiliary matchup features (Phase 6 Exp 1+)
        if self.aux_encoder is not None and "aux_features" in batch:
            aux_proj = self.aux_encoder(batch["aux_features"])  # (B, h)
            matchup_repr = matchup_repr + aux_proj

        return self.prediction_heads(matchup_repr)

    def get_num_parameters(self) -> dict[str, int]:
        """Count parameters per component."""
        components = {
            "player_embed": self.player_embed,
            "gs_encoder": self.gs_encoder,
            "per_game_encoder": self.per_game_encoder,
            "temporal_attention": self.temporal_attention,
            "roster_encoder": self.roster_encoder,
            "rest_embed": self.rest_embed[0],
            "team_combine": self.team_combine,
            "fusion": self.fusion,
            "prediction_heads": self.prediction_heads,
        }

        if self.form_encoder is not None:
            components["form_encoder"] = self.form_encoder

        if self.team_gat is not None:
            components["team_gat"] = self.team_gat

        if self.season_efficiency_proj is not None:
            components["season_efficiency_proj"] = self.season_efficiency_proj

        if self.aux_encoder is not None:
            components["aux_encoder"] = self.aux_encoder

        counts = {}
        total = 0
        for name, module in components.items():
            n = sum(p.numel() for p in module.parameters())
            counts[name] = n
            total += n

        if self.roster_context_enabled:
            rc_modules = nn.ModuleList(
                [
                    self.roster_overlap_embed,
                    self.traj_query_proj,
                    self.trajectory_attn,
                    self.reinjection_norm,
                    self.reinjection_crossattn,
                ]
            )
            rc_params = sum(p.numel() for p in rc_modules.parameters())
            rc_params += self.traj_no_history.numel()
            counts["roster_context"] = rc_params

        # Shared params are counted under player_embed and not double-counted
        # in per_game_encoder and roster_encoder. Compute actual total.
        counts["total"] = sum(p.numel() for p in self.parameters())
        return counts
