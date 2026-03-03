"""Utilities for managing Weights & Biases hyperparameter sweeps."""

import os
from typing import Optional, Union

import torch
import wandb
from torch import nn
from torch.utils.data import DataLoader

from lantern.config import ConvBlockConfig, MetricsConfig, ModelConfig, ModelType, ResidualBlockConfig, TrainerConfig
from lantern.trainer import Trainer
from lantern.utils import build_model, make_optimizer


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
        s = sweep.entity + "/" + sweep.project + "/" + sweep.id
        print(f"Stopping sweep {s}")
        exit_code = os.system("wandb sweep --stop " + s)
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
    input_spec: Union[int, tuple, list],
    num_outputs: int,
    wandb_entity_name: Optional[str] = None,
    checkpoint_resume: bool = False,
    wandb_name_prefix: Optional[str] = None,
) -> callable:
    """Create a training function for use with a W&B sweep agent.

    Returns a closure that reads hyperparameters from ``wandb.config``,
    builds a model, trains it, and logs results to W&B.

    Args:
        wandb_project_name: W&B project name passed to ``wandb.init``.
        datasets: ``(train_dataset, val_dataset)`` tuple.
        device: Torch device for training.
        input_spec: For MLP models, an int giving the flattened input size.
            For CNN models, a list/tuple of (height, width).
        num_outputs: Number of output classes.
        wandb_entity_name: Optional W&B entity (user or team) name.
        checkpoint_resume: Bool of whether checkpointing is activated.
        wandb_name_prefix: Optional prefix (user's initial) to be appended to run name.

    Returns:
        A no-arg function suitable for ``wandb.agent(sweep_id, function=...)``.
    """
    train_dataset, val_dataset = datasets

    def train_sweep():
        # Initialize a W&B run (sweep controller populates wandb.config)
        # The values passed by the factory function are essentially captured/hardcoded for duration of closure's lifetime
        # When the agent calls train_sweep, wandb.init connects to the sweep controller and poplate wandb.config
        run = wandb.init(
            project=wandb_project_name,
            entity=wandb_entity_name,
            reinit=True,
            settings=wandb.Settings(x_stats_sampling_interval=2.0),
        )

        # Load fallback default configs
        default_trainer_config = TrainerConfig()
        default_model_config = ModelConfig()
        default_metrics_config = MetricsConfig()

        # Read hyperparameters from wandb.config (reads every time this closure is called, kind of like a global object)
        config = wandb.config
        print(f"wandb.config: {config}")
        
        # Model related config hyperparameters
        model_type = getattr(config, "model_type", default_model_config.model_type)
        hidden_units = getattr(
            config, "hidden_units", default_model_config.hidden_units
        )
        dropout = getattr(config, "dropout", default_model_config.dropout)
        use_GAP = getattr(config, "use_GAP", default_model_config.use_GAP)
        
        def _parse_conv_block(raw_block):
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
        
        raw_blocks = getattr(config, "conv_blocks", default_model_config.conv_blocks)
        conv_blocks = [_parse_conv_block(b) for b in raw_blocks]
        in_channels = getattr(config, "in_channels", default_model_config.in_channels)
        if model_type == ModelType.CNN and not conv_blocks:
            raise ValueError("CNN Config is missing conv_blocks")
        
        # Trainer related config hyperparameters
        trainer_batch_size = getattr(
            config, "trainer_batch_size", default_trainer_config.trainer_batch_size
        )
        evaluator_batch_size = getattr(
            config, "evaluator_batch_size", default_trainer_config.evaluator_batch_size
        )
        num_workers = getattr(config, "num_workers", default_trainer_config.num_workers)
        pin_memory = getattr(config, "pin_memory", default_trainer_config.pin_memory)
        learning_rate = getattr(
            config, "learning_rate", default_trainer_config.learning_rate
        )
        num_epochs = getattr(config, "num_epochs", default_trainer_config.num_epochs)
        optimizer_name = getattr(
            config, "optimizer_name", default_trainer_config.optimizer_name
        )
        weight_decay = getattr(
            config, "weight_decay", default_trainer_config.weight_decay
        )
        momentum = getattr(config, "momentum", default_trainer_config.momentum)
        early_stopping_patience = getattr(
            config,
            "early_stopping_patience",
            default_trainer_config.early_stopping_patience,
        )
        scheduler_gamma = getattr(
            config, "scheduler_gamma", default_trainer_config.scheduler_gamma
        )
        use_scheduler = getattr(config, "use_scheduler", "scheduler_gamma" in config.keys())

        # Descriptive run name for the W&B dashboard
        hidden_str = "x".join(map(str, hidden_units))
        if wandb_name_prefix:
            run.name = f"{wandb_name_prefix}_{model_type}_bs{trainer_batch_size}_lr{learning_rate}_h{hidden_str}"
        else:
            run.name = f"{model_type}_bs{trainer_batch_size}_lr{learning_rate}_h{hidden_str}"
        print(f"Run name set to: {run.name}")

        # New DataLoaders per run (batch size varies across runs)
        train_loader = DataLoader(
            train_dataset, batch_size=trainer_batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory
        )
        val_loader = DataLoader(
            val_dataset, batch_size=evaluator_batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
        )

        # Build configs from sweep hyperparameters
        trainer_config = TrainerConfig(
            trainer_batch_size=trainer_batch_size,
            evaluator_batch_size=evaluator_batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            device=device,
            optimizer_name=optimizer_name,
            weight_decay=weight_decay,
            momentum=momentum,
            early_stopping_patience=early_stopping_patience,
            use_scheduler=use_scheduler,
            scheduler_gamma=scheduler_gamma,
            checkpoint_last_filename=wandb_project_name + "-last.pt",
            checkpoint_best_filename=wandb_project_name + "-best.pt",
        )
        # Metrics / output related hyperparameters
        task = getattr(config, "task", default_metrics_config.task)
        # num_classes is derived from num_outputs for multiclass; binary doesn't need it
        num_classes = num_outputs if task != "binary" else None
        metrics_config = MetricsConfig(
            task=task,
            num_classes=num_classes,
            names=default_metrics_config.names,
        )
        model_config = ModelConfig(
            model_type=model_type,
            hidden_units=hidden_units,
            dropout=dropout,
            conv_blocks=conv_blocks,
            in_channels=in_channels,
            use_GAP=use_GAP,
        )

        # Build model, optimizer, and criterion
        model = build_model(
            input_spec=input_spec, num_outputs=num_outputs, config=model_config
        )
        optimizer = make_optimizer(model.parameters(), trainer_config)
        criterion = nn.CrossEntropyLoss()

        # Train (pass run so Trainer logs to this W&B run)
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            config=trainer_config,
            metrics_config=metrics_config,
            run=run,
        )
        results = trainer.fit(
            train_loader, val_loader, resume_from_last_checkpoint=checkpoint_resume
        )

        # Clean up
        trainer.finish_run()
        parts = ["Run complete!"]
        if "val_loss" in results:
            parts.append(f"val_loss: {results['val_loss']:.4f}")
        if "val_accuracy" in results:
            parts.append(f"val_accuracy: {results['val_accuracy'] * 100:.2f}%")
        if "val_f1" in results:
            parts.append(f"val_f1: {results['val_f1']:.4f}")
        print(" ".join(parts))

    return train_sweep
