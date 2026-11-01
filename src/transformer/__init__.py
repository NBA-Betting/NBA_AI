"""
NBA Transformer Architecture - Custom sequence-based prediction model.

This module contains the Phase 1 implementation of the transformer-based
NBA game prediction system using play-by-play sequences.

Submodules:
    - tokenizer: Event tokenization from PBP + GameStates
    - sequence_builder: Historical game sequence construction
    - dataset: PyTorch Dataset for training
    - models: Neural network architectures
    - training: Training loop, loss functions, metrics
    - evaluation: Evaluation and ablation study tools
"""

__version__ = "0.1.0"
