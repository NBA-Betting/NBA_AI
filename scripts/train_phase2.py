#!/usr/bin/env python
"""
Training Script for Phase 2 NBA Transformer Model.

Usage:
    # Build cache first (run once):
    python -m src.transformer.phase2.cache_builder

    # Train with config:
    python scripts/train_phase2.py --config configs/transformer/phase2_baseline.yaml

    # Quick smoke test:
    python scripts/train_phase2.py --quick-test

    # Resume from checkpoint:
    python scripts/train_phase2.py --config configs/transformer/phase2_baseline.yaml --resume best.pt
"""

import argparse
import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from torch.utils.data import DataLoader

from src.transformer.phase2.config import Phase2ExperimentConfig
from src.transformer.phase2.cache_builder import build_cache, load_cache
from src.transformer.phase2.dataset import Phase2Dataset, collate_phase2
from src.transformer.phase2.models.phase2_model import Phase2Model
from src.transformer.phase2.training.trainer import Phase2Trainer


def setup_logging(log_level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def ensure_cache(config: Phase2ExperimentConfig) -> dict:
    """Load cache if it exists, otherwise build it."""
    cache_dir = config.data.cache_dir
    cache_path = Path(cache_dir)

    if (cache_path / "game_features.pt").exists():
        logging.info(f"Loading existing cache from {cache_dir}")
        return load_cache(cache_dir)
    else:
        all_seasons = sorted(
            set(
                config.data.train_seasons
                + config.data.val_seasons
                + config.data.test_seasons
            )
        )
        logging.info(f"Building cache for seasons: {all_seasons}")
        build_cache(all_seasons, cache_dir)
        return load_cache(cache_dir)


def load_aux_features(path: str) -> dict | None:
    """Load auxiliary features from JSON file if specified."""
    if not path:
        return None
    import json

    path = Path(path)
    if not path.exists():
        logging.warning(f"Aux features file not found: {path}")
        return None
    with open(path) as f:
        data = json.load(f)
    logging.info(f"Loaded {len(data)} aux feature vectors from {path}")
    return data


def load_projected_rosters(cache: dict, seasons: list[str], cache_dir: str) -> dict:
    """Pre-game rosters for every game in the seasons, built as the predictor
    builds them (RosterAssembler as of tip-off, capped at ROSTER_CAP), so the
    model trains on what it sees live. Saved in the cache dir, which only
    gains games it lacks: delete it after changing the roster code."""
    import json

    from src.database import DB_PATH
    from src.pipeline.phase3_predictor import ROSTER_CAP
    from src.pipeline.roster_assembler import RosterAssembler

    path = Path(cache_dir) / "projected_rosters.json"
    rosters = (
        {g: tuple(r) for g, r in json.loads(path.read_text()).items()}
        if path.exists()
        else {}
    )
    game_ids = [
        g
        for g, gf in cache["game_features"].items()
        if gf.season in seasons and g not in rosters
    ]
    if not game_ids:
        return rosters
    logging.info(f"Projecting pre-game rosters for {len(game_ids)} games...")
    assembler = RosterAssembler(DB_PATH, historical=True)
    for i in range(0, len(game_ids), 500):
        for g, r in assembler.get_projected_rosters(game_ids[i : i + 500]).items():
            rosters[g] = (r["home_players"][:ROSTER_CAP], r["away_players"][:ROSTER_CAP])
    path.write_text(json.dumps(rosters))
    return rosters


def create_dataloaders(
    config: Phase2ExperimentConfig,
    cache: dict,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test data loaders."""

    aux_features = load_aux_features(config.data.aux_features_file)
    projected_rosters = load_projected_rosters(
        cache,
        config.data.train_seasons + config.data.val_seasons + config.data.test_seasons,
        config.data.cache_dir,
    )

    common_kwargs = dict(
        game_features=cache["game_features"],
        season_index=cache["season_index"],
        gs_cache=cache["gs_cache"],
        player_id_map=cache.get("player_id_map", {}),
        n_recent_games=config.data.n_recent_games,
        max_context_games=config.data.max_context_games,
        min_history_games=config.data.min_history_games,
        max_players_per_game=config.data.max_players_per_game,
        max_roster_size=config.data.max_roster_size,
        enable_player_form=config.model.enable_player_form,
        max_player_appearances=config.model.max_player_appearances,
        n_player_stats=config.data.n_player_stats,
        enable_team_gat=config.model.enable_team_gat,
        n_efficiency_features=config.data.n_efficiency_features,
        player_experience=cache.get("player_experience", {}),
        aux_features=aux_features,
        projected_rosters=projected_rosters,
    )

    logging.info("Creating training dataset...")
    train_dataset = Phase2Dataset(
        seasons=config.data.train_seasons,
        enable_augmentation=config.data.enable_augmentation,
        **common_kwargs,
    )

    logging.info("Creating validation dataset...")
    val_dataset = Phase2Dataset(
        seasons=config.data.val_seasons,
        enable_augmentation=False,
        **common_kwargs,
    )

    logging.info("Creating test dataset...")
    test_dataset = Phase2Dataset(
        seasons=config.data.test_seasons,
        enable_augmentation=False,
        **common_kwargs,
    )

    pin_memory = config.data.pin_memory and torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        collate_fn=collate_phase2,
        num_workers=config.data.num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        collate_fn=collate_phase2,
        num_workers=config.data.num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        collate_fn=collate_phase2,
        num_workers=config.data.num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader


def create_model(config: Phase2ExperimentConfig) -> Phase2Model:
    """Create Phase 2 model from configuration."""
    model = Phase2Model(config.model)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(
        f"Model parameters: {total_params:,} total, {trainable_params:,} trainable"
    )

    param_counts = model.get_num_parameters()
    for name, count in param_counts.items():
        logging.info(f"  {name}: {count:,}")

    return model


def load_pretrained_weights(model: Phase2Model, weights_path: str) -> set[str]:
    """Load pre-trained weights into Phase2Model. Returns set of loaded parameter names."""
    logger = logging.getLogger(__name__)

    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)

    # Handle both raw state dicts and full training checkpoints
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        logger.info(f"Extracted weights from checkpoint ({len(state_dict)} tensors)")
    else:
        state_dict = checkpoint
    logger.info(
        f"Loading pre-trained weights from {weights_path} ({len(state_dict)} tensors)"
    )

    # Load matching parameters
    model_state = model.state_dict()
    loaded = set()
    skipped = []

    for name, tensor in state_dict.items():
        if name in model_state:
            if model_state[name].shape == tensor.shape:
                model_state[name] = tensor
                loaded.add(name)
            else:
                skipped.append(
                    f"{name}: shape mismatch {tensor.shape} vs {model_state[name].shape}"
                )
        else:
            skipped.append(f"{name}: not in model")

    model.load_state_dict(model_state)

    logger.info(f"Loaded {len(loaded)} pre-trained parameters")
    if skipped:
        for s in skipped:
            logger.warning(f"  Skipped: {s}")

    # Log parameter norms for verification
    for name in [
        "player_embed.weight",
        "per_game_encoder.context_combine.0.weight",
        "temporal_attention.layers.0.self_attn.in_proj_weight",
    ]:
        if name in loaded:
            norm = model_state[name].norm().item()
            logger.info(f"  {name}: norm={norm:.4f}")

    return loaded


def main():
    parser = argparse.ArgumentParser(description="Train Phase 2 NBA Transformer")

    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    parser.add_argument("--quick-test", action="store_true", help="Quick smoke test")
    parser.add_argument(
        "--resume", type=str, default=None, help="Checkpoint filename to resume from"
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default=None,
        help="Path to pre-trained transferable_weights.pt",
    )
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    parser.add_argument("--seed", type=int, default=None)

    args = parser.parse_args()
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    # Load config
    if args.config:
        logger.info(f"Loading config from {args.config}")
        config = Phase2ExperimentConfig.from_yaml(args.config)
    elif args.quick_test:
        logger.info("Using quick test configuration")
        config = Phase2ExperimentConfig()
        config.data.train_seasons = ["2023-2024"]
        config.data.val_seasons = ["2023-2024"]
        config.data.test_seasons = ["2023-2024"]
        config.data.batch_size = 2
        config.data.num_workers = 0
        config.data.enable_augmentation = False
        config.training.max_epochs = 2
        config.training.patience = 5
        config.training.gradient_accumulation_steps = 2
        config.training.log_every_n_steps = 5
        config.training.experiment_name = "phase2_quick_test"
        config.training.use_ema = False
    else:
        parser.error("--config is required. Use --quick-test for debugging.")

    # CLI overrides
    if args.experiment_name:
        config.training.experiment_name = args.experiment_name
    if args.device:
        config.training.device = args.device
    if args.seed is not None:
        config.training.seed = args.seed

    # Seed
    torch.manual_seed(config.training.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.training.seed)

    logger.info(f"Experiment: {config.training.experiment_name}")
    logger.info(f"Device: {config.training.device}")

    # Load or build cache
    cache = ensure_cache(config)

    # Create data loaders
    train_loader, val_loader, test_loader = create_dataloaders(config, cache)

    # Create model
    model = create_model(config)

    # Load pre-trained weights if specified
    pretrained_path = args.pretrained or config.training.pretrained_checkpoint
    pretrained_params = set()
    if pretrained_path:
        pretrained_params = load_pretrained_weights(model, pretrained_path)

    # Create trainer
    trainer = Phase2Trainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        pretrained_params=pretrained_params if pretrained_path else None,
    )

    # Resume if specified
    if args.resume:
        trainer.resume_from_checkpoint(args.resume)

    # Train
    logger.info("Starting training...")
    final_metrics = trainer.train()

    # Print results
    print("\n" + "=" * 60)
    print("PHASE 2 TRAINING COMPLETE")
    print("=" * 60)
    print(f"Spread MAE:  {final_metrics.spread_mae:.2f}")
    print(f"Spread RMSE: {final_metrics.spread_rmse:.2f}")
    print(f"Home MAE:    {final_metrics.home_mae:.2f}")
    print(f"Away MAE:    {final_metrics.away_mae:.2f}")
    print(f"Win Accuracy: {final_metrics.win_accuracy:.3f}")
    print(f"Win AUC:     {final_metrics.win_auc:.3f}")
    print(f"Brier Score: {final_metrics.brier_score:.4f}")
    print(f"ECE:         {final_metrics.ece:.4f}")
    print(f"90% Coverage: {final_metrics.spread_coverage_90:.3f}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
