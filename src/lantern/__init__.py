"""
lantern

Public API for the lantern training library used in CSCI 357 - AI and Neural Networks.

Re-exports the most commonly used classes and functions from the sub-modules so
that notebooks and scripts can do ``from lantern import Trainer, ModelConfig``, etc.

Course: CSCI 357 - AI and Neural Networks
Author: Chang Min Bark
"""

from lantern.config import (
    ConvBlockConfig,
    MetricsConfig,
    ModelConfig,
    ModelType,
    TrainerConfig,
)
from lantern.data import get_dataloaders, get_torchvision_datasets
from lantern.model import CNN_Model, MLP_Model
from lantern.sweep import make_train_sweep, print_sweep_info, terminate_sweep
from lantern.trainer import Trainer
from lantern.utils import (
    accuracy_from_logits,
    build_model,
    load_model_from_checkpoint,
    make_lr_scheduler,
    make_optimizer,
)

__all__ = [
    # config
    "ConvBlockConfig",
    "MetricsConfig",
    "ModelConfig",
    "ModelType",
    "TrainerConfig",
    # data
    "get_dataloaders",
    "get_torchvision_datasets",
    # model
    "CNN_Model",
    "MLP_Model",
    # sweep
    "make_train_sweep",
    "print_sweep_info",
    "terminate_sweep",
    # trainer
    "Trainer",
    # utils
    "accuracy_from_logits",
    "build_model",
    "load_model_from_checkpoint",
    "make_lr_scheduler",
    "make_optimizer",
]
