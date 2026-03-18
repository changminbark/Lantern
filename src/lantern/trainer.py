"""
trainer.py

Training loop, checkpointing, and validation utilities for PyTorch models.

Course: CSCI 357 - AI and Neural Networks
Author: Chang Min Bark
"""

import os
from dataclasses import asdict
from typing import Callable, Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
import wandb

from lantern.config import MetricsConfig, TrainerConfig
from lantern.utils import make_lr_scheduler


class Trainer:
    """A training loop wrapper that handles training, validation, and logging for PyTorch models."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        config: TrainerConfig = TrainerConfig(),
        metrics_config: Optional[MetricsConfig] = None,
        run: Optional[wandb.Run] = None,
        wandb_watch: bool = True,
    ) -> None:
        """Initialize the trainer with a model, optimizer, loss function, and optional W&B run.

        Args:
            model: The neural network to train (moved to config.device).
            optimizer: The optimizer for updating model parameters.
            criterion: Loss function mapping (predictions, targets) -> scalar loss.
            config: Training hyperparameters and device settings.
            metrics_config: Which metrics to compute and log. Defaults to
                ``MetricsConfig()`` (loss, accuracy, and macro F1 for multiclass).
            run: Optional Weights & Biases run for experiment tracking.
            wandb_watch: If True and a run is provided, calls ``wandb.watch``
                on the model to log gradients and parameters.
        """
        self.model = model.to(config.device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.config = config
        self.metrics_config = metrics_config or MetricsConfig()
        self.run = run
        if self.run is not None and wandb_watch:
            wandb.watch(self.model, log="all", log_freq=100)
        # Initialize learning rate scheduler if enabled
        self.scheduler = None
        if config.use_scheduler:
            self.scheduler = make_lr_scheduler(
                optimizer=self.optimizer, config=self.config
            )

        # Initialize state variables for checkpointing and early stopping
        self.start_epoch = 0  # Use for resuming training from checkpoint
        self.current_epoch = 0  # Tracks current epoch during training
        self.best_val_loss = float("inf")  # Default best validation loss
        self.patience_counter = 0  # How many epochs without improvement before stopping

    def _build_torchmetrics(self) -> Dict[str, torchmetrics.Metric]:
        """Build fresh torchmetrics instances for the names in metrics_config."""
        cfg = self.metrics_config
        base_kwargs: Dict = {"task": cfg.task}

        # If metrics task is multi class
        if cfg.task == "multiclass":
            if self.model.num_outputs is None:
                raise ValueError(
                    "self.model.num_outputs is required when task='multiclass'"
                )
            base_kwargs["num_classes"] = self.model.num_outputs

        # Keyword arguments for average metrics
        avg_kwargs = {} if cfg.task == "binary" else {"average": "macro"}

        # List of available metrics
        name_to_cls = {
            "accuracy": (torchmetrics.Accuracy, avg_kwargs),
            "f1": (torchmetrics.F1Score, avg_kwargs),
            "precision": (torchmetrics.Precision, avg_kwargs),
            "recall": (torchmetrics.Recall, avg_kwargs),
        }

        result: Dict[str, torchmetrics.Metric] = {}
        for name in cfg.names:
            if name in name_to_cls:
                cls, extra = name_to_cls[name]
                result[name] = cls(**base_kwargs, **extra).to(self.config.device)
        return result

    def __enter__(self) -> "Trainer":
        """Return self to support use as a context manager."""
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        """Finish the W&B run on context exit, even if an exception occurred."""
        self.finish_run()

    def train_one_epoch(
        self, train_loader: torch.utils.data.DataLoader
    ) -> Dict[str, float]:
        """Run one training epoch over the entire train_loader.

        Args:
            train_loader: DataLoader yielding (inputs, targets) batches.

        Returns:
            A dict mapping metric name to its epoch-average value. Always
            includes ``"loss"``; additional keys match ``metrics_config.names``.
        """
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        tm_metrics = self._build_torchmetrics()

        for inputs, targets in train_loader:
            inputs, targets = (
                inputs.to(self.config.device),
                targets.to(self.config.device),
            )

            # Forward pass, backprop, and parameter update
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            if self.metrics_config.task == "binary":
                loss = self.criterion(outputs.squeeze(-1), targets.float())
            else:
                loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            batch_size = inputs.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            preds = (
                outputs.squeeze(-1) if self.metrics_config.task == "binary" else outputs
            )
            for m in tm_metrics.values():
                m.update(preds, targets)

        if total_samples == 0:
            return {"loss": 0.0}

        result: Dict[str, float] = {"loss": total_loss / total_samples}
        for name, m in tm_metrics.items():
            result[name] = m.compute().item()
        return result

    def validate(self, val_loader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """Evaluate the model on a validation set without updating parameters.

        Args:
            val_loader: DataLoader yielding (inputs, targets) batches.

        Returns:
            A dict mapping metric name to its average value over the validation
            set. Always includes ``"loss"``; additional keys match
            ``metrics_config.names``.
        """
        self.model.eval()
        running_loss = 0.0
        total_samples = 0
        tm_metrics = self._build_torchmetrics()

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(self.config.device)
                y_batch = y_batch.to(self.config.device)

                logits = self.model(X_batch)
                if self.metrics_config.task == "binary":
                    loss = self.criterion(logits.squeeze(-1), y_batch.float())
                else:
                    loss = self.criterion(logits, y_batch)

                batch_size = X_batch.size(0)
                running_loss += loss.item() * batch_size
                total_samples += batch_size

                preds = (
                    logits.squeeze(-1)
                    if self.metrics_config.task == "binary"
                    else logits
                )
                for m in tm_metrics.values():
                    m.update(preds, y_batch)

        if total_samples == 0:
            return {"loss": 0.0}

        result: Dict[str, float] = {"loss": running_loss / total_samples}
        for name, m in tm_metrics.items():
            result[name] = m.compute().item()
        return result

    def fit(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        resume_from_last_checkpoint: bool = False,
        override_num_epochs: Optional[int] = None,
    ) -> Dict[str, float]:
        """Run the full training loop with checkpointing and early stopping.

        Trains for config.num_epochs (starting from self.start_epoch if resuming),
        validates after each epoch, saves periodic and best checkpoints, and stops
        early if validation loss doesn't improve for early_stopping_patience epochs.

        Args:
            train_loader: DataLoader for training data.
            val_loader: DataLoader for validation data.
            resume_from_last_checkpoint: If True, load the last checkpoint before
                training so the loop resumes from where it left off.
            override_num_epochs: If provided, train for this many epochs instead
                of config.num_epochs.

        Returns:
            A dict with ``"train_<name>"`` and ``"val_<name>"`` keys for every
            name in ``metrics_config.names``, plus ``"num_epochs"``.

        Raises:
            ValueError: If train_loader.batch_size doesn't match config.trainer_batch_size.
        """
        active_names = self.metrics_config.names

        # Sanity check: Verify the batch sizes match the config supplied
        if hasattr(train_loader, "batch_size") and train_loader.batch_size is not None:
            if train_loader.batch_size != self.config.trainer_batch_size:
                raise ValueError(
                    f"Train loader batch size ({train_loader.batch_size}) does not match config ({self.config.trainer_batch_size})"
                )

        # Resume from checkpoint if specified
        if resume_from_last_checkpoint:
            try:
                self.load_checkpoint(retrieve_best=False)
            except (RuntimeError, Exception) as e:
                print(f"Could not load checkpoint ({e}), \nSTARTING FROM SCRATCH.")

        # Update wandb's run config with ModelConfig and TrainerConfig
        if self.run is not None:
            num_params, num_trainable_params = self.model.num_parameters()
            self.run.config.update(
                {
                    "model_config": asdict(self.model.config),
                    "trainer_config": asdict(self.config),
                    "num_parameters": num_params,
                    "num_train_parameters": num_trainable_params,
                    "pin_memory": self.config.pin_memory,
                }
            )

        num_epochs = (
            override_num_epochs
            if override_num_epochs is not None
            else self.config.num_epochs
        )

        train_metrics: Dict[str, float] = {}
        val_metrics: Dict[str, float] = {}

        for epoch in range(self.start_epoch, num_epochs):
            self.current_epoch = epoch

            train_metrics = self.train_one_epoch(train_loader)
            val_metrics = self.validate(val_loader)

            # Build per-epoch print output and wandb log dict
            parts = [f"Epoch {epoch}:"]
            wandb_log: Dict[str, float] = {"epoch": epoch}
            for name in active_names:
                t_val = train_metrics.get(name, 0.0)
                v_val = val_metrics.get(name, 0.0)
                if name == "accuracy":
                    parts.append(
                        f"Train Accuracy={t_val * 100:.2f}%  Val Accuracy={v_val * 100:.2f}%"
                    )
                else:
                    label = name.capitalize() if name != "loss" else "Loss"
                    parts.append(f"Train {label}={t_val:.4f}  Val {label}={v_val:.4f}")
                wandb_log[f"train_{name}"] = t_val
                wandb_log[f"val_{name}"] = v_val
            print("\n".join(parts))

            # Early stopping uses val_loss (always computed even if not in names)
            val_loss = val_metrics["loss"]
            if val_loss < self.best_val_loss - self.config.early_stopping_min_delta:
                self.best_val_loss = val_loss
                self.save_checkpoint(is_best=True)
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            # Step the learning rate scheduler
            if self.scheduler is not None:
                if self.config.scheduler_type == "reduce_on_plateau":
                    # ReduceLROnPlateau needs the validation loss
                    self.scheduler.step(val_loss)
                else:
                    # Other schedulers just need to know an epoch completed
                    self.scheduler.step()

                current_lr = self.optimizer.param_groups[0]["lr"]
                wandb_log["learning_rate"] = current_lr

            if self.run is not None:
                self.run.log(wandb_log, step=epoch)

            # Early stop if patience counter has reached threshold
            if self.patience_counter == self.config.early_stopping_patience:
                break

            # Periodic checkpoint saving (every N epochs)
            if (self.current_epoch + 1) % self.config.checkpoint_save_interval == 0:
                self.save_checkpoint(is_best=False)

        result: Dict[str, float] = {"num_epochs": epoch + 1}
        for name in active_names:
            result[f"train_{name}"] = train_metrics.get(name, 0.0)
            result[f"val_{name}"] = val_metrics.get(name, 0.0)
        return result

    def finish_run(self) -> None:
        """Unwatch the model and finish the W&B run. No-op if no run is active."""
        if self.run is not None:
            try:
                wandb.unwatch(self.model)
            except Exception:
                pass
            self.run.finish()

    def save_checkpoint(self, is_best: bool = False) -> None:
        """Save current model, optimizer, and training state to a checkpoint file.

        Saves to the 'last' checkpoint path. If is_best is True, also saves
        a copy to the 'best' checkpoint path.

        Args:
            is_best: If True, save an additional copy as the best checkpoint.
        """
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        filepath = os.path.join(
            self.config.checkpoint_dir, self.config.checkpoint_last_filename
        )

        checkpoint = {
            # Model weights (the numbers)
            "model_state_dict": self.model.state_dict(),
            # Architecture specification (the blueprint)
            "model_architecture": self.model.get_architecture_config()
            if hasattr(self.model, "get_architecture_config")
            else None,
            # Training state
            "trainer_config": asdict(self.config),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "epoch": self.current_epoch,
            "patience_counter": self.patience_counter,
            "scheduler_state_dict": self.scheduler.state_dict()
            if self.scheduler
            else None,
        }

        if is_best:
            best_path = os.path.join(
                self.config.checkpoint_dir, self.config.checkpoint_best_filename
            )
            torch.save(checkpoint, best_path)
            print(f"--> New best checkpoint saved: {best_path}")
            print(f"--> Also saving as last checkpoint: {filepath}")
        else:
            print(f"--> Saving checkpoint: {filepath}")
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, retrieve_best: bool = False) -> None:
        """Load a saved checkpoint and restore model, optimizer, and training state.

        Restores weights, optimizer state, and sets self.start_epoch so that
        fit() resumes from the next epoch after the one that was checkpointed.

        Args:
            retrieve_best: If True, load the best checkpoint; otherwise load the last checkpoint.

        Raises:
            RuntimeError: If the requested checkpoint file does not exist.
        """
        filepath = os.path.join(
            self.config.checkpoint_dir, self.config.checkpoint_last_filename
        )
        best_path = os.path.join(
            self.config.checkpoint_dir, self.config.checkpoint_best_filename
        )

        # Verify the requested checkpoint file exists
        if retrieve_best and not os.path.exists(best_path):
            raise RuntimeError("Best checkpoint file does not exist")
        if not retrieve_best and not os.path.exists(filepath):
            raise RuntimeError("Last checkpoint file does not exist")

        # Load checkpoint
        checkpoint = (
            torch.load(best_path, weights_only=False)
            if retrieve_best
            else torch.load(filepath, weights_only=False)
        )

        # Restore model, optimizer, and lr scheduler state
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler and checkpoint.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        # Resume from the epoch after the one that was saved
        self.start_epoch = checkpoint["epoch"] + 1
        self.best_val_loss = checkpoint["best_val_loss"]
        self.patience_counter = checkpoint["patience_counter"]
