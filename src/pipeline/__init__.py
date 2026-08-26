"""Production pipeline for daily NBA prediction generation."""

from importlib import import_module

__all__ = ["Phase3CacheUpdater", "Phase3Predictor"]


def __getattr__(name):
    """Load optional PyTorch pipeline components only when they are requested."""
    if name == "Phase3CacheUpdater":
        return import_module("src.pipeline.phase3_cache_updater").Phase3CacheUpdater
    if name == "Phase3Predictor":
        return import_module("src.pipeline.phase3_predictor").Phase3Predictor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
