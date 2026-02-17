
import os
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from typing import Callable, Iterable, Tuple, Optional, Dict
from dataclasses import asdict, dataclass

from my_engine.config import TrainerConfig, ModelConfig
from my_engine.utils import accuracy_from_logits

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
            
        # Initialize state variables for checkpointing and early stopping
        self.start_epoch = 0                # Use for resuming training from checkpoint
        self.current_epoch = 0              # Tracks current epoch during training
        self.best_val_loss = float('inf')   # Default best validation loss
        self.patience_counter = 0           # How many epochs without improvement before stopping

    def train_one_epoch(self, train_loader: torch.utils.data.DataLoader) -> Tuple[float, float]:
        """Run one training epoch over the entire train_loader.

        Args:
            train_loader: DataLoader yielding (inputs, targets) batches.

        Returns:
            A tuple of (average_loss, average_accuracy) for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        total_acc = 0.0
        total_samples = 0

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

        if total_samples == 0:
            return 0.0, 0.0

        avg_loss = total_loss / total_samples
        avg_acc = total_acc / total_samples
        return avg_loss, avg_acc

    def validate(self, val_loader: torch.utils.data.DataLoader) -> Tuple[float, float]:
        """Evaluate the model on a validation set without updating parameters.

        Args:
            val_loader: DataLoader yielding (inputs, targets) batches.

        Returns:
            A tuple of (average_loss, average_accuracy) over the validation set.
        """
        self.model.eval()
        running_loss = 0.0
        running_acc = 0.0
        total_samples = 0

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

        if total_samples == 0:
            return 0.0, 0.0

        avg_loss = running_loss / total_samples
        avg_acc = running_acc / total_samples
        return avg_loss, avg_acc

    def fit(self, train_loader: torch.utils.data.DataLoader, val_loader: torch.utils.data.DataLoader, resume_from_last_checkpoint: bool = False,
            override_num_epochs: Optional[int] = None) -> Dict[str, float]:
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
            A dict with final metrics: num_epochs, train_loss, val_loss, val_acc.

        Raises:
            ValueError: If train_loader.batch_size doesn't match config.trainer_batch_size.
        """
        # Sanity check: Verify the batch sizes match the config supplied
        if hasattr(train_loader, 'batch_size') and train_loader.batch_size is not None:
            if train_loader.batch_size != self.config.trainer_batch_size:
                raise ValueError(f"Train loader batch size ({train_loader.batch_size}) does not match config ({self.config.trainer_batch_size})")
            
        # Resume from checkpoint if specified
        if resume_from_last_checkpoint:
            self.load_checkpoint(retrieve_best=False)
            
        # Update wandb's run config with ModelConfig and TrainerConfig
        if self.run is not None:
            self.run.config.update({"model_config": asdict(self.model.config), "trainer_config": asdict(self.config)})

        num_epochs = override_num_epochs if override_num_epochs is not None else self.config.num_epochs

        for epoch in range(self.start_epoch, num_epochs):
            self.current_epoch = epoch
            
            # Train and validate
            train_loss, train_acc = self.train_one_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)
            print(
                f"Epoch {epoch}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}"
                f"Val Loss={val_loss:.4f}, Val Acc={val_acc * 100:.2f}%"
            )
            
            # Log to wandb
            if self.run is not None:
                self.run.log({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc
                })
            
            # Check and update best val
            if val_loss < self.best_val_loss - self.config.early_stopping_min_delta:
                self.best_val_loss = val_loss
                self.save_checkpoint(is_best=True)
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                
            # Early stop if patience counter has reached threshold
            if self.patience_counter == self.config.early_stopping_patience:
                break
                
            # Periodic checkpoint saving (every N epochs)
            if (self.current_epoch + 1) % self.config.checkpoint_save_interval == 0:
                self.save_checkpoint(is_best=False)

        return { "num_epochs": epoch+1,
                 "train_loss": train_loss,
                 "val_loss": val_loss,
                 "val_acc": val_acc }
        
    def finish_run(self) -> None:
        """Unwatch the model and finish the W&B run. No-op if no run is active."""
        if self.run is not None:
            wandb.unwatch(self.model)
            self.run.finish()
            
    def save_checkpoint(self, is_best: bool = False) -> None:
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

        # Restore model and optimizer state
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Resume from the epoch after the one that was saved
        self.start_epoch = checkpoint["epoch"] + 1
        self.best_val_loss = checkpoint["best_val_loss"]
        self.patience_counter = checkpoint["patience_counter"]