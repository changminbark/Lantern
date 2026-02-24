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
from typing import Iterable, List, Optional, Union

import torch
import torch.nn as nn

from lantern.config import ModelConfig, ModelType, TrainerConfig
from lantern.model import CNN_Model, MLP_Model


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


def build_model(input_spec: Union[int, tuple, list], num_outputs: int, config: ModelConfig) -> nn.Module:
    """Factory function that constructs a model based on config.model_type.

    Args:
        input_spec: For MLP models, an int giving the flattened input size.
            For CNN models, a list/tuple of (height, width).
        num_outputs: Number of output classes.
        config: Model architecture configuration.

    Returns:
        An instantiated nn.Module ready for training.

    Raises:
        ValueError: If config.model_type is not recognized, or if input_spec
            does not match the expected type for the chosen model.
    """
    if config.model_type == ModelType.MLP:
        if not isinstance(input_spec, int):
            raise ValueError("MLP requires input_spec as int (flattened input size).")
        return MLP_Model(num_inputs=input_spec, num_outputs=num_outputs, config=config)
    elif config.model_type == ModelType.CNN:
        if not isinstance(input_spec, (tuple, list)) or len(input_spec) != 2:
            raise ValueError("CNN requires input_spec as (height, width).")
        h, w = input_spec[0], input_spec[1]
        return CNN_Model(input_height=h, input_width=w, num_outputs=num_outputs, config=config)
    else:
        raise ValueError(f"Unknown model type: {config.model_type}. Supported types: 'ModelType.MLP', 'ModelType.CNN'")


def make_optimizer(
    params: Iterable[torch.nn.Parameter], config: TrainerConfig
) -> torch.optim.Optimizer:
    """
    Factory for optimizers.

    Args:
        params (Iterable[torch.nn.Parameter]): Parameters to optimize.
        config (TrainerConfig): Configuration for the optimizer.

    Returns:
        torch.optim.Optimizer: Configured optimizer instance.
    """
    if config.optimizer_name == "sgd":
        return torch.optim.SGD(
            params=params, lr=config.learning_rate, weight_decay=config.weight_decay
        )
    elif config.optimizer_name == "momentum":
        return torch.optim.SGD(
            params=params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            momentum=config.momentum,
        )
    elif config.optimizer_name == "adam":
        return torch.optim.Adam(
            params=params, lr=config.learning_rate, weight_decay=config.weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer_name}")


def load_model_from_checkpoint(
    checkpoint_path: Union[str, Path], device: torch.device = torch.device("cpu")
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
        raise FileNotFoundError(
            f"Checkpoint file not found with path: {checkpoint_path}"
        )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if "model_architecture" not in checkpoint:
        raise KeyError(
            "Checkpoint dictionary does not have model_architecture metadata"
        )
    architecture = checkpoint["model_architecture"]

    if "model_type" not in architecture:
        raise KeyError("model_architecture metadata does not have model_type")
    model_type = architecture["model_type"]

    # Reconstruct model from saved architecture config
    if model_type == ModelType.MLP:
        config = ModelConfig(**architecture["config"])
        model = MLP_Model(
            num_inputs=architecture["num_inputs"], 
            num_outputs=architecture["num_outputs"], 
            config=config
        )
    elif model_type == ModelType.CNN:
        config = ModelConfig(**architecture["config"])
        model = CNN_Model(
            input_height=architecture["input_height"],
            input_width=architecture["input_width"],
            num_outputs=architecture["num_outputs"],
            config=config
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Load saved weights and move to target device
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    return model


def make_lr_scheduler(
    optimizer: torch.optim.Optimizer, config: TrainerConfig
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
            gamma=config.scheduler_gamma,
        )
    elif config.scheduler_type == "reduce_on_plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=config.scheduler_gamma,
            patience=config.scheduler_patience,
            min_lr=config.scheduler_min_lr,
        )
    # Add more schedulers as needed...
    else:
        raise ValueError(f"Unknown scheduler type: {config.scheduler_type}")

def lr_range_test(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device = torch.device("cpu"),
    start_lr: float = 1e-6,
    end_lr: float = 1.0,
    num_iterations: int = 100,
) -> tuple[list[float], list[float]]:
    """Performs Leslie Smith's LR Range Test.

    Trains the model for num_iterations mini-batches, exponentially increasing the
    learning rate from start_lr to end_lr. Records the loss at each step.

    WARNING: This function modifies the model weights and optimizer state.
    You should create a fresh model/optimizer before calling this, or save and
    restore a checkpoint afterward.

    Args:
        model: The model to test.
        train_loader: DataLoader for training data.
        criterion: Loss function.
        optimizer: Optimizer (will have its LR modified).
        device: Device to train on.
        start_lr: Starting learning rate.
        end_lr: Ending learning rate.
        num_iterations: Number of mini-batches to train.

    Returns:
        Tuple of (lrs, losses) -- lists of learning rates and corresponding losses.
    """
    # Implement the LR range test.
    # 1. Set the optimizer's LR to start_lr
    # 2. Compute the multiplicative factor: gamma = (end_lr / start_lr) ** (1 / num_iterations)
    # 3. Create a LambdaLR scheduler that multiplies LR by gamma each step:
    #    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: gamma ** step)
    # 4. Loop for num_iterations batches (cycle through train_loader if needed):
    #    a. Get the next batch, move to device
    #    b. Forward pass, compute loss
    #    c. Record current LR and loss
    #    d. Backward pass, optimizer step, scheduler step
    #    e. If loss > 4 * best_loss_so_far, stop early (diverging)
    # 5. Return (lrs, losses)

    # Move the model to the specified device and set it to training mode
    model = model.to(device)
    model.train()

    # Set the optimizer's learning rate to the starting value
    for pg in optimizer.param_groups:
        pg["lr"] = start_lr

    # Calculate the multiplicative factor for exponentially increasing the LR
    gamma = (end_lr / start_lr) ** (1 / num_iterations)

    # Create a LambdaLR scheduler that multiplies the LR by gamma each step
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: gamma ** step
    )

    lrs: list[float] = []     # Store the learning rates for each batch
    losses: list[float] = []  # Store the losses for each batch
    best_loss = float("inf")  # Track the smallest loss encountered
    loader_iter = iter(train_loader)  # Iterator for cycling through the train loader

    for _ in range(num_iterations):
        try:
            # Get the next batch from the data loader
            inputs, targets = next(loader_iter)
        except StopIteration:
            # Restart the iterator if we run out of data
            loader_iter = iter(train_loader)
            inputs, targets = next(loader_iter)

        # Move the data to the correct device
        inputs, targets = inputs.to(device), targets.to(device)

        # Zero out the previous gradients
        optimizer.zero_grad()

        # Forward pass: compute outputs and loss
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss_val = loss.item()
        curr_lr = optimizer.param_groups[0]["lr"]

        # Record the current learning rate and loss
        lrs.append(curr_lr)
        losses.append(loss_val)

        # Update the best loss seen so far, and early stop if loss diverges
        if loss_val < best_loss:
            best_loss = loss_val
        elif loss_val > 4 * best_loss:
            break

        # Backward pass and optimizer/scheduler step
        loss.backward()
        optimizer.step()
        scheduler.step()

    # Return recorded learning rates and corresponding losses
    return lrs, losses