
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
import wandb
from typing import Callable, Iterable, Tuple, Optional, Dict
from dataclasses import asdict, dataclass

from lantern.config import TrainerConfig, ModelConfig
from lantern.metrics import Metrics, ALL_METRICS
from lantern.utils import accuracy_from_logits, make_lr_scheduler

class Trainer:
    """A training loop wrapper that handles training, validation, and logging for PyTorch models."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        config: TrainerConfig = TrainerConfig(),
        run: Optional[wandb.Run] = None,
    ) -> None:
        """Initialize the trainer with a model, optimizer, loss function, and optional W&B run.

        Args:
            model: The neural network to train (moved to config.device).
            optimizer: The optimizer for updating model parameters.
            criterion: Loss function mapping (predictions, targets) -> scalar loss.
            config: Training hyperparameters and device settings.
            run: Optional Weights & Biases run for experiment tracking.
        """
        self.model = model.to(config.device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.config = config
        self.run = run
        if self.run is not None:
            wandb.watch(self.model, log="all", log_freq=1)
        # Initialize learning rate scheduler if enabled
        self.scheduler = None
        if config.use_scheduler:
            self.scheduler = make_lr_scheduler(optimizer=self.optimizer, config=self.config)
            
        # Initialize state variables for checkpointing and early stopping
        self.start_epoch = 0                # Use for resuming training from checkpoint
        self.current_epoch = 0              # Tracks current epoch during training
        self.best_val_loss = float('inf')   # Default best validation loss
        self.patience_counter = 0           # How many epochs without improvement before stopping

    def train_one_epoch(self, train_loader: torch.utils.data.DataLoader) -> Tuple[float, float, float]:
        """Run one training epoch over the entire train_loader.

        Args:
            train_loader: DataLoader yielding (inputs, targets) batches.

        Returns:
            A tuple of (average_loss, average_accuracy, f1_macro) for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        total_acc = 0.0
        total_samples = 0
        f1_macro_metric = torchmetrics.F1Score(task="multiclass", num_classes=self.config.num_classes, average="macro").to(self.config.device)

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(self.config.device), targets.to(self.config.device)

            # Need to flatten the input into batch_size x flattened_dims
            inputs = inputs.view(inputs.size(0), -1)

            # Forward pass, backprop, and parameter update
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            # Accumulate weighted loss and accuracy for averaging
            batch_size = inputs.size(0)
            total_loss += loss.item() * batch_size
            total_acc += accuracy_from_logits(outputs, targets) * batch_size
            total_samples += batch_size

            # Accumulate predictions for F1 computation
            f1_macro_metric.update(outputs, targets)

        if total_samples == 0:
            return 0.0, 0.0, 0.0

        avg_loss = total_loss / total_samples
        avg_acc = total_acc / total_samples
        avg_f1_macro = f1_macro_metric.compute().item()
        return avg_loss, avg_acc, avg_f1_macro

    def validate(self, val_loader: torch.utils.data.DataLoader) -> Tuple[float, float, float]:
        """Evaluate the model on a validation set without updating parameters.

        Args:
            val_loader: DataLoader yielding (inputs, targets) batches.

        Returns:
            A tuple of (average_loss, average_accuracy, f1_macro) over the validation set.
        """
        self.model.eval()
        running_loss = 0.0
        running_acc = 0.0
        total_samples = 0
        f1_macro_metric = torchmetrics.F1Score(task="multiclass", num_classes=self.config.num_classes, average="macro").to(self.config.device)

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(self.config.device)
                y_batch = y_batch.to(self.config.device)

                # Need to flatten the X_batch into batch_size x flattened_dims
                X_batch = X_batch.view(X_batch.size(0), -1)

                logits = self.model(X_batch)
                loss = self.criterion(logits, y_batch)
                batch_acc = accuracy_from_logits(logits, y_batch)

                # Weight metrics by batch size for correct averaging
                batch_size = X_batch.size(0)
                running_loss += loss.item() * batch_size
                running_acc += batch_acc * batch_size
                total_samples += batch_size

                # Accumulate predictions for F1 computation
                f1_macro_metric.update(logits, y_batch)

        if total_samples == 0:
            return 0.0, 0.0, 0.0

        avg_loss = running_loss / total_samples
        avg_acc = running_acc / total_samples
        avg_f1_macro = f1_macro_metric.compute().item()
        return avg_loss, avg_acc, avg_f1_macro

    def fit(self, train_loader: torch.utils.data.DataLoader, val_loader: torch.utils.data.DataLoader, resume_from_last_checkpoint: bool = False,
            override_num_epochs: Optional[int] = None, *metrics: Metrics) -> Dict[str, float]:
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
            *metrics: Variable number of Metrics enum values specifying which
                metrics to log and return. Defaults to all metrics if none provided.

        Returns:
            A dict with final metrics filtered by the requested metrics.

        Raises:
            ValueError: If train_loader.batch_size doesn't match config.trainer_batch_size.
        """
        # Default to all metrics if none specified
        active_metrics = set(metrics) if metrics else set(ALL_METRICS)
        log_loss = Metrics.LOSS in active_metrics
        log_acc = Metrics.ACC in active_metrics
        log_f1_macro = Metrics.F1_MACRO in active_metrics
        
        # Sanity check: Verify the batch sizes match the config supplied
        if hasattr(train_loader, 'batch_size') and train_loader.batch_size is not None:
            if train_loader.batch_size != self.config.trainer_batch_size:
                raise ValueError(f"Train loader batch size ({train_loader.batch_size}) does not match config ({self.config.trainer_batch_size})")
            
        # Resume from checkpoint if specified
        if resume_from_last_checkpoint:
            try:
                self.load_checkpoint(retrieve_best=False)
            except (RuntimeError, Exception) as e:
                print(f"Could not load checkpoint ({e}), \nSTARTING FROM SCRATCH.")
            
        # Update wandb's run config with ModelConfig and TrainerConfig
        if self.run is not None:
            self.run.config.update({"model_config": asdict(self.model.config), "trainer_config": asdict(self.config)})

        num_epochs = override_num_epochs if override_num_epochs is not None else self.config.num_epochs

        for epoch in range(self.start_epoch, num_epochs):
            self.current_epoch = epoch
            
            # Train and validate
            train_loss, train_acc, train_f1_macro = self.train_one_epoch(train_loader)
            val_loss, val_acc, val_f1_macro = self.validate(val_loader)

            # Build print and wandb log entries based on active metrics
            parts = [f"Epoch {epoch}:"]
            wandb_log = {"epoch": epoch}
            if log_loss:
                parts.append(f"Train Loss={train_loss:.4f}")
                parts.append(f"Val Loss={val_loss:.4f}")
                wandb_log["train_loss"] = train_loss
                wandb_log["val_loss"] = val_loss
            if log_acc:
                parts.append(f"Train Acc={train_acc:.4f}")
                parts.append(f"Val Acc={val_acc * 100:.2f}%")
                wandb_log["train_acc"] = train_acc
                wandb_log["val_acc"] = val_acc
            if log_f1_macro:
                parts.append(f"Train F1 Macro={train_f1_macro:.4f}")
                parts.append(f"Val F1 Macro={val_f1_macro:.4f}")
                wandb_log["train_f1_macro"] = train_f1_macro
                wandb_log["val_f1_macro"] = val_f1_macro
            print("\n".join(parts))

            # Log to wandb
            if self.run is not None:
                self.run.log(wandb_log)
            
            # Check and update best val
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
                
                # Log current learning rate to W&B
                current_lr = self.optimizer.param_groups[0]['lr']
                if self.run is not None:
                    self.run.log({"learning_rate": current_lr}, step=self.current_epoch)
                
            # Early stop if patience counter has reached threshold
            if self.patience_counter == self.config.early_stopping_patience:
                break
                
            # Periodic checkpoint saving (every N epochs)
            if (self.current_epoch + 1) % self.config.checkpoint_save_interval == 0:
                self.save_checkpoint(is_best=False)

        result = {"num_epochs": epoch + 1}
        if log_loss:
            result["train_loss"] = train_loss
            result["val_loss"] = val_loss
        if log_acc:
            result["train_acc"] = train_acc
            result["val_acc"] = val_acc
        if log_f1_macro:
            result["train_f1_macro"] = train_f1_macro
            result["val_f1_macro"] = val_f1_macro
        return result
        
    def finish_run(self) -> None:
        """Unwatch the model and finish the W&B run. No-op if no run is active."""
        if self.run is not None:
            wandb.unwatch(self.model)
            self.run.finish()
            
    def save_checkpoint(self, is_best: bool = False) -> None:
        """Save current model, optimizer, and training state to a checkpoint file.

        Saves to the 'last' checkpoint path. If is_best is True, also saves
        a copy to the 'best' checkpoint path.

        Args:
            is_best: If True, save an additional copy as the best checkpoint.
        """
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        filepath = os.path.join(self.config.checkpoint_dir, self.config.checkpoint_last_filename)

        checkpoint = {
            # Model weights (the numbers)
            'model_state_dict': self.model.state_dict(),

            # Architecture specification (the blueprint)
            'model_architecture': self.model.get_architecture_config() if hasattr(self.model, 'get_architecture_config') else None,

            # Training state
            'trainer_config': asdict(self.config),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'epoch': self.current_epoch,
            'patience_counter': self.patience_counter,
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
        }

        if is_best:
            best_path = os.path.join(self.config.checkpoint_dir, self.config.checkpoint_best_filename)
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
        filepath = os.path.join(self.config.checkpoint_dir, self.config.checkpoint_last_filename)
        best_path = os.path.join(self.config.checkpoint_dir, self.config.checkpoint_best_filename)

        # Verify the requested checkpoint file exists
        if retrieve_best and not os.path.exists(best_path):
            raise RuntimeError("Best checkpoint file does not exist")
        if not retrieve_best and not os.path.exists(filepath):
            raise RuntimeError("Last checkpoint file does not exist")

        # Load checkpoint
        checkpoint = torch.load(best_path, weights_only=False) if retrieve_best else torch.load(filepath, weights_only=False)

        # Restore model, optimizer, and lr scheduler state
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # Resume from the epoch after the one that was saved
        self.start_epoch = checkpoint["epoch"] + 1
        self.best_val_loss = checkpoint["best_val_loss"]
        self.patience_counter = checkpoint["patience_counter"]