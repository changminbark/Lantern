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

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix

from lantern.config import ModelConfig, ModelType, TrainerConfig
from lantern.model import BagOfEmbeddings, CNN_Model, MLP_Model, SkipGram, TextCNN1D


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


def build_model(
    input_spec: Union[int, tuple, list], num_outputs: int, config: ModelConfig
) -> nn.Module:
    """Factory function that constructs a model based on config.model_type.

    Args:
        input_spec: For MLP models, an int giving the flattened input size.
            For CNN models, a list/tuple of (height, width).
            For BOW models, this argument is unused (vocab size and embedding
            dim are read from ``config``); pass any value or ``None``.
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
        return CNN_Model(
            input_height=h, input_width=w, num_outputs=num_outputs, config=config
        )
    elif config.model_type == ModelType.TEXTCNN:
        return TextCNN1D(num_outputs=num_outputs, config=config)
    elif config.model_type == ModelType.BOW:
        return BagOfEmbeddings(num_outputs=num_outputs, config=config)
    elif config.model_type == ModelType.SKIPGRAM:
        return SkipGram(config=config)
    else:
        raise ValueError(
            f"Unknown model type: {config.model_type}. Supported types: 'ModelType.MLP', 'ModelType.CNN', 'ModelType.TEXTCNN', 'ModelType.BOW', 'ModelType.SKIPGRAM'"
        )


def make_optimizer(
    params: Iterable[torch.nn.Parameter], config: TrainerConfig
) -> torch.optim.Optimizer:
    """Factory for optimizers.

    Args:
        params: Model parameters to optimize.
        config: Trainer configuration specifying optimizer name, learning rate,
            weight decay, and momentum.

    Returns:
        Configured optimizer instance.

    Raises:
        ValueError: If config.optimizer_name is not 'sgd', 'momentum', or 'adam'.
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
    NOTE: Only ``ModelType.MLP`` and ``ModelType.CNN`` are currently supported.
    TextCNN1D, BagOfEmbeddings, and SkipGram models cannot be restored with this function.

    Args:
        checkpoint_path: Path to checkpoint file.
        device: Device to load onto (default: CPU).

    Returns:
        Reconstructed model with loaded weights.

    Raises:
        FileNotFoundError: If the checkpoint file does not exist.
        KeyError: If the checkpoint is missing the model_architecture metadata.
        ValueError: If model_type in checkpoint is not ``ModelType.MLP`` or ``ModelType.CNN``.
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
            config=config,
        )
    elif model_type == ModelType.CNN:
        config = ModelConfig(**architecture["config"])
        model = CNN_Model(
            input_height=architecture["input_height"],
            input_width=architecture["input_width"],
            num_outputs=architecture["num_outputs"],
            config=config,
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
    """Factory for learning rate schedulers.

    Args:
        optimizer: The optimizer whose learning rate will be scheduled.
        config: Trainer configuration containing scheduler settings (use_scheduler,
            scheduler_type, scheduler_step_size, scheduler_gamma, scheduler_patience,
            scheduler_min_lr).

    Returns:
        Scheduler instance, or None if config.use_scheduler is False.

    Raises:
        ValueError: If config.scheduler_type is not ``'step'`` or
            ``'reduce_on_plateau'``. (``'exponential'`` and ``'cosine'`` are
            listed in TrainerConfig but are not yet implemented here.)
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
        optimizer, lr_lambda=lambda step: gamma**step
    )

    lrs: list[float] = []  # Store the learning rates for each batch
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


def compute_confusion_matrix(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, list[int], list[int]]:
    """Compute a confusion matrix by running the model over the entire dataset.

    Args:
        model: Trained model.
        data_loader: DataLoader for the evaluation set.
        device: Device to run inference on.

    Returns:
        Tuple of (confusion_matrix as numpy array, all_preds list, all_labels list)
    """

    was_training = model.training
    model.eval()
    model.to(device)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            # Binary: threshold logit at 0 (equivalent to sigmoid >= 0.5)
            # Multiclass: argmax over class dimension
            if outputs.shape[1] == 1:
                preds = (outputs.squeeze(-1) >= 0.0).long().cpu().tolist()
            else:
                preds = torch.argmax(outputs, dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().tolist())

    if was_training:
        model.train()

    cm = confusion_matrix(all_labels, all_preds)
    return cm, all_preds, all_labels


def compute_saliency_map(
    model: nn.Module,
    image_tensor: torch.Tensor,
    target_class: int = None,
    device: torch.device = None,
) -> tuple[np.ndarray, int]:
    """Compute a saliency map for a single image.

    The saliency map highlights which pixels most influence the model's
    prediction by computing |d(score_c) / d(input)|.

    Args:
        model: Trained model (will be set to eval mode).
        image_tensor: Single image tensor of shape (C, H, W). Do NOT include batch dim.
        target_class: Class index to compute saliency for.
                      If None, uses the model's predicted class (argmax).
        device: Device for computation. If None, uses the model's device.

    Returns:
        Tuple of (saliency_map as 2D numpy array normalized to [0, 1],
                  target_class index used).
    """
    model.eval()

    # If device is not specified, infer from model parameters
    if device is None:
        device = next(model.parameters()).device

    # Add batch dimension, move to device, and enable gradient tracking
    image_tensor = image_tensor.unsqueeze(0).to(device=device)
    image_tensor.requires_grad_()

    # Forward pass
    output_logits = model(image_tensor)

    # Use predicted class if target_class not specified
    if target_class is None:
        target_class = torch.argmax(output_logits, dim=1).item()

    # Backpropagate from the target class score (all the way to image_tensor input)
    model.zero_grad()
    score = output_logits[0, target_class]
    score.backward()

    # Extract gradients for each input pixel: shape (1, C, H, W) -> (C, H, W)
    saliency = image_tensor.grad.data.abs().squeeze(0)

    # For multi-channel images take max across channels -> (H, W)
    saliency, _ = torch.max(saliency, dim=0)

    # Normalize to [0, 1]
    saliency = saliency - saliency.min()
    saliency = saliency / (saliency.max() + 1e-8)

    return saliency.cpu().numpy(), target_class


def denormalize_image(
    tensor: torch.Tensor,
    mean: Optional[tuple] = (0.5, 0.5, 0.5),
    std: Optional[tuple] = (0.5, 0.5, 0.5),
) -> torch.Tensor:
    """Denormalize a tensor image using the given mean and standard deviation.

    Reverses a channel-wise normalization transform: output = tensor * std + mean,
    then clamps to [0, 1].

    Args:
        tensor: Image tensor of shape (C, H, W).
        mean: Per-channel mean used during the original normalization. Default: (0.5, 0.5, 0.5).
        std: Per-channel std used during the original normalization. Default: (0.5, 0.5, 0.5).

    Returns:
        Denormalized image tensor of the same shape, with values clamped to [0, 1].
    """
    # Goes from (C,) to (C, H, W)  which broadcasts over (C, H, W) in tensor
    # Each channel's pixels are denormalized according to their respective means and stds
    mean_t = torch.tensor(mean, dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)
    std_t = torch.tensor(std, dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)
    return (tensor * std_t + mean_t).clamp(0, 1)


def get_text_saliency(
    model: nn.Module,
    token_ids: torch.Tensor,
    target_class: int,
    device: Optional[torch.device],
) -> tuple[np.ndarray, torch.Tensor]:
    """Compute per-token saliency as gradient magnitude w.r.t. the embedding output.

    Backpropagates the score of target_class through the model and measures how
    much each token's embedding contributes via its gradient L2 norm. Temporarily
    enables grad on a frozen embedding weight so that gradients still flow for
    visualization purposes.

    NOTE: This function is specific to TextCNN1D models, as it directly accesses
    model.embedding, model.convs, and model.classifier_head.

    Args:
        model: Trained TextCNN1D model (will be set to eval mode).
        token_ids: 1-D token index tensor of shape (seq_len,). Do NOT include batch dim.
        target_class: Class index to compute saliency for.
        device: Device for computation. If None, infers from model parameters.

    Returns:
        Tuple of (saliency as a 1-D numpy array of shape (seq_len,) with L2 gradient
        norms per token, logits tensor of shape (1, num_classes)).
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    x = token_ids.unsqueeze(0).to(device)  # (1, seq_len)

    # If embeddings are frozen, briefly unfreeze so gradients can flow
    was_frozen = not model.embedding.weight.requires_grad
    if was_frozen:
        model.embedding.weight.requires_grad_(True)

    try:
        embedded = model.embedding(x)  # (1, seq_len, embed_dim)
        embedded.retain_grad()

        # Manually replay the TextCNN1D forward from the embedding output
        emb_t = embedded.permute(0, 2, 1)  # (1, embed_dim, seq_len)
        pooled = [torch.relu(conv(emb_t)).max(dim=2).values for conv in model.convs]
        features = torch.cat(pooled, dim=1)
        logits = model.classifier_head(features)

        logits[0, target_class].backward()

        grad = embedded.grad[0]  # (seq_len, embed_dim)
        saliency = grad.norm(dim=1)  # (seq_len,)  — L2 norm over embed dim
        return saliency.detach().cpu().numpy(), logits.detach().cpu()
    finally:
        if was_frozen:
            model.embedding.weight.requires_grad_(False)


def render_text_saliency_html(
    words: list[str], saliency: np.ndarray, title: str = ""
) -> str:
    """Return an HTML string with words highlighted by saliency intensity.

    Each word is wrapped in a <span> whose background opacity scales linearly
    with its normalized saliency score (red channel, [0, 1]).

    Args:
        words: List of word strings corresponding to each token.
        saliency: 1-D numpy array of raw saliency scores, one per token.
            Will be normalized to [0, 1] internally.
        title: Optional title displayed in bold above the highlighted text.

    Returns:
        HTML string containing a <p> block with inline-styled word spans.
    """
    s_min, s_max = saliency.min(), saliency.max()
    norm = (saliency - s_min) / (s_max - s_min + 1e-8)  # normalise to [0, 1]

    parts = [f'<p style="font-family:monospace; line-height:2;"><b>{title}</b><br>']
    for word, alpha in zip(words, norm):
        color = f"rgba(220, 50, 50, {float(alpha):.2f})"
        parts.append(
            f'<span style="background-color:{color}; padding:2px 4px; '
            f'margin:1px; border-radius:3px;">{word}</span> '
        )
    parts.append("</p>")
    return "".join(parts)
