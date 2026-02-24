"""
data.py

Helper functions for loading torchvision datasets and constructing DataLoaders.

Course: CSCI 357 - AI and Neural Networks
Author: Chang Min Bark
"""

from typing import Optional, Tuple

from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


_DATASET_REGISTRY = {
    "mnist": datasets.MNIST,
    "fashion_mnist": datasets.FashionMNIST,
    "cifar10": datasets.CIFAR10,
    "cifar100": datasets.CIFAR100,
}


def get_torchvision_datasets(
    name: str,
    root: str = "data",
    transform: Optional[transforms.Compose] = None,
) -> Tuple[Dataset, Dataset]:
    """Load a torchvision dataset by name, returning train and test splits.

    Args:
        name: Dataset name. Supported values: 'mnist', 'fashion_mnist',
            'cifar10', 'cifar100'.
        root: Root directory where the dataset will be downloaded/cached.
        transform: Optional transform applied to every sample. Defaults to
            ``transforms.ToTensor()`` and ``transforms.Normalize((0.5,), (0.5,))`` when not provided.

    Returns:
        A (train_dataset, test_dataset) tuple.

    Raises:
        ValueError: If ``name`` is not in the supported dataset registry.
    """
    if name not in _DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset '{name}'. Supported: {list(_DATASET_REGISTRY.keys())}"
        )

    if transform is None:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    cls = _DATASET_REGISTRY[name]
    train_dataset = cls(root=root, train=True, download=True, transform=transform)
    test_dataset = cls(root=root, train=False, download=True, transform=transform)
    return train_dataset, test_dataset


def get_dataloaders(
    train_dataset: Dataset,
    test_dataset: Dataset,
    train_batch_size: int = 64,
    eval_batch_size: int = 256,
    num_workers: int = 2,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """Wrap datasets in DataLoaders.

    Args:
        train_dataset: Dataset for training (shuffled).
        test_dataset: Dataset for evaluation (not shuffled).
        train_batch_size: Batch size for the training loader.
        eval_batch_size: Batch size for the evaluation loader.
        num_workers: Number of subprocesses for data loading.
        pin_memory: If True, pin tensors to page-locked memory for faster GPU transfer.

    Returns:
        A (train_loader, test_loader) tuple.
    """
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, test_loader
