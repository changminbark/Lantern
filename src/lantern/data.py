"""
data.py

Helper functions for loading torchvision datasets and constructing DataLoaders.

Course: CSCI 357 - AI and Neural Networks
Author: Chang Min Bark
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


class TabularDataset(Dataset):
    """A PyTorch Dataset wrapping tabular (X, y) numpy arrays.

    Attributes:
        feature_names: List of input feature column names.
        target_names: List of unique class label strings, ordered by label index.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        target_names: List[str],
    ) -> None:
        """Store features and labels as tensors alongside metadata.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Integer class labels of shape (n_samples,).
            feature_names: Column names for the input features.
            target_names: Class label strings ordered by label index.
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.feature_names = feature_names
        self.target_names = target_names

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the feature tensor and label for the sample at index ``idx``."""
        return self.X[idx], self.y[idx]
    
    def print_class_distribution(self) -> None:
        """Print the sample count and percentage for each class."""
        counts = torch.bincount(self.y, minlength=len(self.target_names))
        total = len(self.y)
        for i, name in enumerate(self.target_names):
            pct = 100 * counts[i].item() / total
            print(f"  {i} ({name}):  {counts[i].item()} samples ({pct:.1f}%)")

    def get_class_weights(self) -> Tuple[Dict[str, float], torch.Tensor]:
        """Return the class weights using the formula N_total / (N_classes * N_c).

        Returns:
            A (weights_dict, weights_tensor) tuple where weights_dict maps class
            name to weight and weights_tensor is ordered by class index for use
            with torch.nn.CrossEntropyLoss(weight=...).
        """
        counts = torch.bincount(self.y, minlength=len(self.target_names))
        N_total = len(self.y)
        N_classes = len(self.target_names)
        weights = {}
        for i, name in enumerate(self.target_names):
            weights[name] = N_total / (N_classes * counts[i].item())
        weight_tensor = torch.tensor([weights[name] for name in self.target_names])
        return weights, weight_tensor

def get_ucimlrepo_datasets(
    id: int,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[TabularDataset, TabularDataset]:
    """Fetch a dataset from the UCI ML Repository and return stratified train/val splits.

    Features are standardized using StandardScaler fit on the training split only.

    Args:
        id: UCI ML Repository dataset ID.
        test_size: Fraction of data reserved for validation.
        random_state: Random seed for reproducibility.

    Returns:
        A (train_dataset, val_dataset) tuple of TabularDataset instances.
    """
    from ucimlrepo import fetch_ucirepo

    repo = fetch_ucirepo(id=id)
    X: pd.DataFrame = repo.data.features
    y: pd.DataFrame = repo.data.targets

    # Drop rows with missing values
    n_features = len(X.columns)
    combined = pd.concat([X, y], axis=1).dropna()
    X = combined.iloc[:, :n_features]
    y = combined.iloc[:, n_features:]

    feature_names: List[str] = X.columns.tolist()
    target_col = y.columns[0]
    y_vals = y[target_col].values.astype(int)

    unique_labels = sorted(np.unique(y_vals))
    target_names: List[str] = [str(label) for label in unique_labels]

    X_train, X_val, y_train, y_val = train_test_split(
        X.values,
        y_vals,
        test_size=test_size,
        random_state=random_state,
        stratify=y_vals,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    return (
        TabularDataset(X_train, y_train, feature_names, target_names),
        TabularDataset(X_val, y_val, feature_names, target_names),
    )


def get_kagglehub_datasets(
    handle: str,
    target_col: str,
    test_size: float = 0.2,
    random_state: int = 42,
    drop_cols: Optional[List[str]] = None,
    csv_filename: Optional[str] = None,
) -> Tuple[TabularDataset, TabularDataset]:
    """Download a Kaggle dataset via kagglehub and return stratified train/val splits.

    Features are standardized using StandardScaler fit on the training split only.

    Args:
        handle: Kaggle dataset slug in ``"owner/dataset-name"`` format.
        target_col: Name of the target column in the CSV.
        test_size: Fraction of data reserved for validation.
        random_state: Random seed for reproducibility.
        drop_cols: Optional list of columns to drop before splitting X/y.
        csv_filename: Specific CSV filename to load. If None, the first CSV found
            in the downloaded directory is used.

    Returns:
        A (train_dataset, val_dataset) tuple of TabularDataset instances.
    """
    import kagglehub

    path = kagglehub.dataset_download(handle)

    if csv_filename is not None:
        csv_path = os.path.join(path, csv_filename)
    else:
        csv_files = [f for f in os.listdir(path) if f.endswith(".csv")]
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in downloaded dataset at {path}")
        csv_path = os.path.join(path, csv_files[0])

    df = pd.read_csv(csv_path)

    if drop_cols:
        df = df.drop(columns=drop_cols, errors="ignore")

    df = df.dropna()

    X = df.drop(columns=[target_col])
    y = df[target_col]

    feature_names: List[str] = X.columns.tolist()
    unique_labels = sorted(y.unique())
    target_names: List[str] = [str(label) for label in unique_labels]

    y_vals = y.values.astype(int)

    X_train, X_val, y_train, y_val = train_test_split(
        X.values.astype(np.float32),
        y_vals,
        test_size=test_size,
        random_state=random_state,
        stratify=y_vals,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    return (
        TabularDataset(X_train, y_train, feature_names, target_names),
        TabularDataset(X_val, y_val, feature_names, target_names),
    )


_DATASET_REGISTRY = {
    "mnist": datasets.MNIST,
    "fashion_mnist": datasets.FashionMNIST,
    "cifar10": datasets.CIFAR10,
    "cifar100": datasets.CIFAR100,
}


def get_torchvision_datasets(
    name: str,
    root: str = "data",
    train_transform: Optional[transforms.Compose] = None,
    test_transform: Optional[transforms.Compose] = None,
) -> Tuple[Dataset, Dataset]:
    """Load a torchvision dataset by name, returning train and test splits.

    Args:
        name: Dataset name. Supported values: 'mnist', 'fashion_mnist',
            'cifar10', 'cifar100'.
        root: Root directory where the dataset will be downloaded/cached.
        train_transform: Optional transform applied to training samples. Defaults to
            ``transforms.ToTensor()`` and ``transforms.Normalize((0.5,), (0.5,))``
            when not provided.
        test_transform: Optional transform applied to test samples. Defaults to
            the same as ``train_transform`` when not provided.

    Returns:
        A (train_dataset, test_dataset) tuple.

    Raises:
        ValueError: If ``name`` is not in the supported dataset registry.
    """
    if name not in _DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset '{name}'. Supported: {list(_DATASET_REGISTRY.keys())}"
        )

    default_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    if train_transform is None:
        train_transform = default_transform
    if test_transform is None:
        test_transform = train_transform

    cls = _DATASET_REGISTRY[name]
    train_dataset = cls(root=root, train=True, download=True, transform=train_transform)
    test_dataset = cls(root=root, train=False, download=True, transform=test_transform)
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
