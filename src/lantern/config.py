"""
config.py

This module contains configuration dataclasses useful for AI and ML projects.

Course: CSCI 357 - AI and Neural Networks
Author: Chang Min Bark
Date: 2/16/2026

"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

import torch


@dataclass
class TrainerConfig:
    """Hyperparameters and settings for the training loop.

    Attributes:
        trainer_batch_size: Batch size used during training.
        evaluator_batch_size: Batch size used during evaluation/validation.
        learning_rate: Optimizer learning rate.
        device: Device to run training on (cpu or cuda).
        num_epochs: Number of full passes over the training data.
        weight_decay: L2 regularization coefficient for the optimizer.
        early_stopping_patience: Epochs to wait for improvement before stopping (None to disable).
        early_stopping_min_delta: Minimum change in loss to qualify as an improvement.
        optimizer_name: Name of the optimizer to use to train the model (e.g. "adam", "sgd", "momentum").
        momentum: Momentum used in gradient optimization (only applies to momentum optimizer).
        checkpoint_dir: Directory where checkpoints will be saved.
        checkpoint_last_filename: Filename for the most recent checkpoint.
        checkpoint_save_interval: Save a checkpoint every N epochs.
        checkpoint_best_filename: Filename for the best model checkpoint.
    """

    trainer_batch_size: int = 64
    evaluator_batch_size: int = 256
    learning_rate: float = 0.001
    device: torch.device = torch.device("cpu")
    num_epochs: int = 10
    weight_decay: float = 0.0
    early_stopping_patience: Optional[int] = 5  # Set to None to disable early stopping
    early_stopping_min_delta: float = 0.001
    optimizer_name: str = "adam"
    momentum: float = 0.9

    # Checkpointing Settings
    checkpoint_dir: str = "./checkpoints"  # Directory where checkpoints will be saved
    checkpoint_last_filename: str = "last.pt"  # Filename for most recent checkpoint
    checkpoint_save_interval: int = 5  # Save checkpoint every N epochs
    checkpoint_best_filename: str = "best.pt"  # Filename for the best model checkpoint

    # Learning Rate Scheduler Settings
    use_scheduler: bool = False  # Enable/disable scheduling
    scheduler_type: str = "reduce_on_plateau"  # Options: "step", "exponential", "cosine", "reduce_on_plateau"
    scheduler_step_size: int = 10  # For StepLR: epochs between LR drops
    scheduler_gamma: float = 0.1  # Factor to reduce LR
    scheduler_patience: int = 3  # For ReduceLROnPlateau: epochs to wait before reducing
    scheduler_min_lr: float = (
        1e-6  # Minimum learning rate (prevents it from going too low)
    )

    # Metrics Settings
    num_classes: int = 10  # Number of classes for classification metrics (e.g. F1)

    # DataLoader Settings
    num_workers: int = 2
    pin_memory: bool = True

class ModelType(Enum):
    """Supported model architecture types."""

    MLP = "mlp"
    CNN = "cnn"

    def __str__(self) -> str:
        return self.value
    
@dataclass
class ConvBlockConfig:
    """Configuration for a single [Conv2d -> ReLU -> MaxPool2d] block.

    Attributes:
        out_channels: Number of filters (output feature maps) for this block.
        kernel_size: Spatial size of the convolution kernel.
        stride: Stride of the convolution.
        padding: Zero-padding added to both sides of the input.
        pool_size: Kernel size for MaxPool2d. Set to 0 to skip pooling.
    """
    out_channels: int
    kernel_size: int = 3
    stride: int = 1
    padding: int = 1
    pool_size: int = 2


@dataclass
class ModelConfig:
    """Architecture configuration for model construction.

    Attributes:
        model_type: Model architecture identifier (uses ModelType enum).
        hidden_units: Number of neurons in each hidden layer.
        dropout: Dropout rate after each hidden layer (aligned with hidden_units).
    """

    model_type: ModelType = ModelType.MLP
    hidden_units: List[int] = field(default_factory=lambda: [128, 64])
    dropout: List[float] = field(default_factory=lambda: [0.1, 0.2])
    # CNN Fields
    conv_blocks: List[ConvBlockConfig] = field(default_factory=list)
    in_channels: int = 1        # 1 for grayscale, 3 for RGB

    def __post_init__(self) -> None:
        """Convert model_type from string to ModelType enum if needed."""
        if isinstance(self.model_type, str):
            try:
                self.model_type = ModelType(self.model_type)
            except ValueError:
                valid = [e.value for e in ModelType]
                raise ValueError(
                    f"Unknown model_type '{self.model_type}'. Must be one of {valid}"
                )

