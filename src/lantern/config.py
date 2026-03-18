"""
config.py

This module contains configuration dataclasses useful for AI and ML projects.

Course: CSCI 357 - AI and Neural Networks
Author: Chang Min Bark
Date: 2/16/2026

"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Union

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
        use_scheduler: Whether to enable a learning rate scheduler.
        scheduler_type: Scheduler variant. One of ``"step"``, ``"exponential"``,
            ``"cosine"``, ``"reduce_on_plateau"``.
        scheduler_step_size: Epochs between LR drops (StepLR only).
        scheduler_gamma: Multiplicative factor to reduce the learning rate.
        scheduler_patience: Epochs without improvement before reducing LR
            (ReduceLROnPlateau only).
        scheduler_min_lr: Lower bound on the learning rate.
        num_workers: Number of subprocesses for DataLoader data loading.
        pin_memory: If True, DataLoader copies tensors into pinned memory before returning.
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

    # DataLoader Settings
    num_workers: int = 2
    pin_memory: bool = True


@dataclass
class MetricsConfig:
    """Configuration for training metrics tracking and reporting.

    Attributes:
        task: torchmetrics task type. One of ``"binary"``, ``"multiclass"``.
        names: Metric names to compute and log. Supported values: ``"loss"``,
            ``"accuracy"``, ``"f1"``, ``"precision"``, ``"recall"``.
            Defaults to ``["loss", "accuracy", "f1"]``.
    """

    task: str = "multiclass"
    names: List[str] = field(default_factory=lambda: ["loss", "accuracy", "f1"])


class ModelType(Enum):
    """Supported model architecture types."""

    MLP = "mlp"
    CNN = "cnn"
    BOW = "bow"
    TEXTCNN = "textcnn"
    SKIPGRAM = "skipgram"

    def __str__(self) -> str:
        """Return the string value of the enum member (e.g. ``"mlp"``, ``"cnn"``, ``"bow"``, ``"textcnn"``, ``"skipgram"``)."""
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
        batch_norm: If True, applies Batch Normalization after the convolution.
    """

    out_channels: int
    kernel_size: int = 3
    stride: int = 1
    padding: int = 1
    pool_size: int = 2
    batch_norm: bool = False


@dataclass
class ResidualBlockConfig:
    """
    Configuration for a single residual block.

    This configuration specifies the parameters used to construct a residual block within a convolutional neural network.
    It determines the number of output channels (filters) and the stride for the convolution, enabling the creation
    of blocks that support skip connections and flexible downsampling, as used in deep residual networks.
    NOTE: Remember - pool_size is intentionally omitted - stride handles downsampling

    Attributes:
        out_channels: Number of filters (output feature maps) for this block
        stride: Stride of the convolution.
    """

    out_channels: int
    stride: int = 1


@dataclass
class ModelConfig:
    """Architecture configuration for model construction.

    Attributes:
        model_type: Model architecture identifier (uses ModelType enum).
            One of ``ModelType.MLP``, ``ModelType.CNN``, ``ModelType.BOW``,
            ``ModelType.TEXTCNN``, or ``ModelType.SKIPGRAM``.
        hidden_units: Number of neurons in each hidden layer (MLP only).
        dropout: Dropout rate after each hidden layer, aligned with hidden_units (MLP only).
        conv_blocks: List of ConvBlockConfig or ResidualBlockConfig instances
            defining the convolutional layers (2D-CNN only).
        in_channels: Number of input channels. 1 for grayscale, 3 for RGB (2D-CNN only).
        use_GAP: If True, applies Global Average Pooling before the classifier
            head instead of flattening (2D-CNN only).
        filter_sizes: List of kernel sizes for parallel 1D convolutions
            (TextCNN1D / 1D-CNN only).
        num_filters: Number of filters per kernel size (TextCNN1D / 1D-CNN only).
        vocab_size: Total number of tokens in the vocabulary, including special
            tokens ``<PAD>`` and ``<UNK>`` (NLP only).
        embedding_dim: Dimensionality of each token embedding vector (NLP only).
        padding_idx: Vocabulary index of the ``<PAD>`` token; the corresponding
            embedding row is kept at zero and receives no gradient (NLP only).
        freeze_embeddings: If True, the embedding layer weights are frozen and
            will not be updated during training (NLP only).
    """

    model_type: ModelType = ModelType.MLP
    hidden_units: List[int] = field(default_factory=lambda: [128, 64])
    dropout: List[float] = field(default_factory=lambda: [0.1, 0.2])
    # 2D-CNN Fields
    conv_blocks: List[Union[ConvBlockConfig, ResidualBlockConfig]] = field(
        default_factory=list
    )
    in_channels: int = 1  # 1 for grayscale, 3 for RGB
    use_GAP: bool = False  # Global Average Pooling
    # 1D-CNN (Text) Fields
    filter_sizes: List[int] = field(
        default_factory=lambda: [3, 4, 5]
    )  # sizes of filters to use in TextCNN1D model.
    num_filters: int = 100  # the number of filters to use in TextCNN1D model

    # NLP / Embedding Fields
    vocab_size: int = 0
    embedding_dim: int = 100
    padding_idx: int = 0
    freeze_embeddings: bool = False

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
        self.conv_blocks = [_parse_conv_block(b) for b in self.conv_blocks]


def _parse_conv_block(raw_block: Union[ConvBlockConfig, ResidualBlockConfig, dict]):
    """Parse a raw conv block dict or config object into a typed block config.

    Args:
        raw_block: A ConvBlockConfig/ResidualBlockConfig instance, or a dict
            with a ``block_type`` key (``"conv"`` or ``"residual"``) and the
            remaining fields matching the corresponding config dataclass.

    Returns:
        A ConvBlockConfig or ResidualBlockConfig instance.

    Raises:
        TypeError: If raw_block is neither a dict nor a block config object.
        ValueError: If ``block_type`` is not ``"conv"`` or ``"residual"``.
    """
    # Already-parsed dataclass objects are allowed
    if isinstance(raw_block, (ConvBlockConfig, ResidualBlockConfig)):
        return raw_block

    if not isinstance(raw_block, dict):
        raise TypeError(
            f"Each conv block must be a dict or block config object, got {type(raw_block)}"
        )

    block_data = dict(raw_block)  # copy so we can safely pop
    block_type = block_data.pop("block_type", "conv")  # backward-compatible default

    if block_type == "conv":
        return ConvBlockConfig(**block_data)
    if block_type == "residual":
        return ResidualBlockConfig(**block_data)

    raise ValueError(f"Unknown block_type '{block_type}' in conv_blocks")
