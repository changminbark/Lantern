
"""
utils.py

This module contains a collection of helper utility functions that will be used throughout the course.
You will find reusable functions for metrics, data handling, and other tools to support labs,
assignments, and projects.

Course: CSCI 357 - AI and Neural Networks
Author: Chang Min Bark
Date: 2/16/2026

"""

import os
from pathlib import Path
from typing import Iterable, Optional, Union
import torch
import torch.nn as nn
from lantern.config import ModelConfig, ModelType, TrainerConfig
from lantern.model import MLP_Model

def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute classification accuracy from raw logits.

    Args:
        logits: Model output of shape (batch_size, num_classes).
        labels: Ground-truth class indices of shape (batch_size,).

    Returns:
        Accuracy as a float in [0, 1].
    """
    predictions = torch.argmax(logits, dim=1)
    correct = (predictions == labels).sum().item()
    totals = labels.size(0)
    return correct / totals


def build_model(num_inputs: int, num_outputs: int, config: ModelConfig) -> nn.Module:
    """Factory function that constructs a model based on config.model_type.

    Args:
        num_inputs: Dimensionality of input features.
        num_outputs: Number of output classes.
        config: Model architecture configuration.

    Returns:
        An instantiated nn.Module ready for training.

    Raises:
        ValueError: If config.model_type is not recognized.
    """
    if config.model_type == ModelType.MLP:
        return MLP_Model(num_inputs, num_outputs, config)
    else:
        raise ValueError(f"Unknown model type: {config.model_type}")

def make_optimizer(params: Iterable[torch.nn.Parameter], config: TrainerConfig) -> torch.optim.Optimizer:
    """
    Factory for optimizers.

    Args:
        params (Iterable[torch.nn.Parameter]): Parameters to optimize.
        config (TrainerConfig): Configuration for the optimizer.

    Returns:
        torch.optim.Optimizer: Configured optimizer instance.
    """
    if config.optimizer_name == "sgd":
        return torch.optim.SGD(params=params, lr=config.learning_rate, weight_decay=config.weight_decay)
    elif config.optimizer_name == "momentum":
        return torch.optim.SGD(params=params, lr=config.learning_rate, weight_decay=config.weight_decay, momentum=config.momentum)
    elif config.optimizer_name == "adam":
        return torch.optim.Adam(params=params, lr=config.learning_rate, weight_decay=config.weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer_name}")
    
def load_model_from_checkpoint(
    checkpoint_path: Union[str, Path],
    device: torch.device = torch.device('cpu')
) -> nn.Module:
    """Reconstructs any model from a checkpoint file.

    This factory function inspects the checkpoint's model_architecture to
    determine the model type, then dispatches to the appropriate constructor.

    NOTE: This ONLY restores the model architecture and weights, not optimizer state or other metadata.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load onto (default: CPU)

    Returns:
        Reconstructed model with loaded weights

    Raises:
        ValueError: If model_type in checkpoint is unrecognized
        FileNotFoundError: if the checkpoint file does not exist
        KeyError: if the checkpoint is missing the model_architecture metadata
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found with path: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if "model_architecture" not in checkpoint:
        raise KeyError("Checkpoint dictionary does not have model_architecture metadata")
    architecture = checkpoint["model_architecture"]

    if "model_type" not in architecture:
        raise KeyError("model_architecture metadata does not have model_type")
    model_type = architecture["model_type"]

    # Reconstruct model from saved architecture config
    if model_type == ModelType.MLP:
        config = ModelConfig(**architecture["config"])
        model = MLP_Model(architecture["num_inputs"], architecture["num_outputs"], config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Load saved weights and move to target device
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    return model

def make_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    config: TrainerConfig
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """
    Factory for learning rate schedulers.
    
    Args:
        optimizer: The optimizer to schedule
        config: Configuration containing scheduler settings
        
    Returns:
        Scheduler instance, or None if use_scheduler is False
        
    Raises:
        ValueError: If scheduler_type is unrecognized
    """
    if not config.use_scheduler:
        return None
    
    if config.scheduler_type == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.scheduler_step_size,
            gamma=config.scheduler_gamma
        )
    elif config.scheduler_type == "reduce_on_plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config.scheduler_gamma,
            patience=config.scheduler_patience,
            min_lr=config.scheduler_min_lr
        )
    # Add more schedulers as needed...
    else:
        raise ValueError(f"Unknown scheduler type: {config.scheduler_type}")