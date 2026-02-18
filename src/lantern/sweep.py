"""Utilities for managing Weights & Biases hyperparameter sweeps."""

import os
from typing import Optional
import torch
from lantern.trainer import Trainer
from lantern.utils import build_model, make_optimizer
import wandb
from lantern.config import ModelConfig, ModelType, TrainerConfig
from torch.utils.data import DataLoader
from torch import nn

def print_sweep_info(sweep_id: str) -> None:
    """Print run count, expected runs, and current state of a W&B sweep."""
    api = wandb.Api()
    sweep = api.sweep(sweep_id)
    print(f"Sweep {sweep_id} has {len(sweep.runs)} runs")
    print(f"Sweep {sweep_id} expected {sweep.expected_run_count} runs")
    print(f"Sweep {sweep_id} current state is: {sweep.state}")


def terminate_sweep(sweep_id: str) -> bool:
    """Stop a running W&B sweep. Returns True if stopped or already finished."""
    api = wandb.Api()
    sweep = api.sweep(sweep_id)
    if sweep.state != "FINISHED":
        # Build fully-qualified sweep path: entity/project/sweep_name
        s = sweep.entity + '/' + sweep.project + '/' + sweep.id
        print(f"Stopping sweep {s}")
        exit_code = os.system('wandb sweep --stop ' + s)
        if exit_code != 0:
            print(f"Failed to stop sweep {s}")
            print(f"Exit code: {exit_code}")
            return False
        else:
            print(f"Sweep {s} stopped successfully")
            return True
    else:
        print(f"Sweep {sweep_id} is already finished")
        return True

def make_train_sweep(
    wandb_project_name: str,
    datasets: tuple,
    device: torch.device,
    num_inputs: int,
    num_outputs: int,
    wandb_entity_name: Optional[str] = None,
    checkpoint_resume: bool = False,
) -> callable:
    """Create a training function for use with a W&B sweep agent.

    Returns a closure that reads hyperparameters from ``wandb.config``,
    builds a model, trains it, and logs results to W&B.

    Args:
        wandb_project_name: W&B project name passed to ``wandb.init``.
        datasets: ``(train_dataset, val_dataset)`` tuple.
        device: Torch device for training.
        num_inputs: Number of input features.
        num_outputs: Number of output classes.
        wandb_entity_name: Optional W&B entity (user or team) name.
        checkpoint_resume: Bool of whether checkpointing is activated

    Returns:
        A no-arg function suitable for ``wandb.agent(sweep_id, function=...)``.
    """
    train_dataset, val_dataset = datasets

    def train_sweep():
        # Initialize a W&B run (sweep controller populates wandb.config) 
        # The values passed by the factory function are essentially captured/hardcoded for duration of closure's lifetime
        run = wandb.init(
            project=wandb_project_name,
            entity=wandb_entity_name,
            reinit=True,
            settings=wandb.Settings(x_stats_sampling_interval=2.0),
        )
        
        # Load fallback default configs
        default_trainer_config = TrainerConfig()
        default_model_config = ModelConfig()

        # Read hyperparameters from wandb.config (reads every time this closure is called, kind of like a global object)
        config = wandb.config
        print(f"wandb.config: {config}")
        hidden_units = getattr(config, "hidden_units", default_model_config.hidden_units)
        trainer_batch_size = getattr(config, "trainer_batch_size", default_trainer_config.trainer_batch_size)
        evaluator_batch_size = getattr(config, "evaluator_batch_size", default_trainer_config.evaluator_batch_size)
        learning_rate = getattr(config, "learning_rate", default_trainer_config.learning_rate)
        num_epochs = getattr(config, "num_epochs", default_trainer_config.num_epochs)
        dropout = getattr(config, "dropout", default_model_config.dropout)
        optimizer_name = getattr(config, "optimizer_name", default_trainer_config.optimizer_name)
        weight_decay = getattr(config, "weight_decay", default_trainer_config.weight_decay)
        momentum = getattr(config, "momentum", default_trainer_config.momentum)
        early_stopping_patience = getattr(config, "early_stopping_patience", default_trainer_config.early_stopping_patience)

        # Descriptive run name for the W&B dashboard
        hidden_str = "x".join(map(str, hidden_units))
        run.name = f"bs{trainer_batch_size}_lr{learning_rate}_h{hidden_str}"
        print(f"Run name set to: {run.name}")

        # New DataLoaders per run (batch size varies across runs)
        train_loader = DataLoader(train_dataset, batch_size=trainer_batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=evaluator_batch_size, shuffle=False)

        # Build configs from sweep hyperparameters
        trainer_config = TrainerConfig(
            trainer_batch_size=trainer_batch_size,
            evaluator_batch_size=evaluator_batch_size,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            device=device,
            optimizer_name=optimizer_name,
            weight_decay=weight_decay,
            momentum=momentum,
            early_stopping_patience=early_stopping_patience,
            checkpoint_last_filename= wandb_project_name + "-last.pt",
            checkpoint_best_filename= wandb_project_name + "-best.pt",
            num_classes=num_outputs,
        )
        model_config = ModelConfig(
            model_type=ModelType.MLP,
            hidden_units=hidden_units,
            dropout=dropout,
        )

        # Build model, optimizer, and criterion
        model = build_model(num_inputs=num_inputs, num_outputs=num_outputs, config=model_config)
        optimizer = make_optimizer(model.parameters(), trainer_config)
        criterion = nn.CrossEntropyLoss()

        # Train (pass run so Trainer logs to this W&B run)
        trainer = Trainer(
            model=model, optimizer=optimizer, criterion=criterion, config=trainer_config, run=run,
        )
        results = trainer.fit(train_loader, val_loader, resume_from_last_checkpoint=checkpoint_resume)

        # Clean up
        trainer.finish_run()
        print(f"Run complete! Final val_loss: {results['val_loss']:.4f}, val_acc: {results['val_acc']*100:.2f}%")

    return train_sweep