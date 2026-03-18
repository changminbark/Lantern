"""
data.py

Helper functions for loading torchvision datasets and constructing DataLoaders.

Course: CSCI 357 - AI and Neural Networks
Author: Chang Min Bark
"""

import os
import re
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from lantern.text import build_vocab


class TextDataset(Dataset):
    """Dataset that stores tokenized text as integer sequences with labels.

    Each sample is a ``(token_ids_tensor, label_tensor)`` pair where
    ``token_ids_tensor`` is a 1D ``LongTensor`` of variable length and
    ``label_tensor`` is a scalar ``LongTensor``.

    Because sequences can have different lengths, collating a batch requires
    padding. Use ``torch.nn.utils.rnn.pad_sequence`` (or a custom collate
    function) when building a DataLoader from this dataset.

    Attributes:
        samples: List of ``(token_ids_tensor, label_tensor)`` tuples.
    """

    def __init__(self, token_id_sequences: list, labels: list):
        """Store token-id sequences and labels as paired tensors.

        Args:
            token_id_sequences: Iterable of integer lists, one per sample,
                where each integer is a vocabulary index.
            labels: Iterable of integer class labels, one per sample.
        """
        self.samples = [
            (torch.tensor(ids, dtype=torch.long), torch.tensor(lbl, dtype=torch.long))
            for ids, lbl in zip(token_id_sequences, labels)
        ]

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, index: int):
        """Return the (token_ids_tensor, label_tensor) pair at the given index."""
        return self.samples[index]


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


def get_dataloaders(
    train_dataset: Dataset,
    test_dataset: Dataset,
    train_batch_size: int = 64,
    eval_batch_size: int = 256,
    num_workers: int = 2,
    pin_memory: bool = True,
    collate_fn: Callable = None,
) -> Tuple[DataLoader, DataLoader]:
    """Wrap datasets in DataLoaders.

    Args:
        train_dataset: Dataset for training (shuffled).
        test_dataset: Dataset for evaluation (not shuffled).
        train_batch_size: Batch size for the training loader.
        eval_batch_size: Batch size for the evaluation loader.
        num_workers: Number of subprocesses for data loading.
        pin_memory: If True, pin tensors to page-locked memory for faster GPU transfer.
        collate_fn: Optional callable to merge a list of samples into a batch. Useful
            for datasets with variable-length sequences (e.g., padding with
            ``torch.nn.utils.rnn.pad_sequence``). If None, the default collate is used.

    Returns:
        A (train_loader, test_loader) tuple.
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )
    return train_loader, test_loader


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
            raise FileNotFoundError(
                f"No CSV files found in downloaded dataset at {path}"
            )
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

    default_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    if train_transform is None:
        train_transform = default_transform
    if test_transform is None:
        test_transform = train_transform

    cls = _DATASET_REGISTRY[name]
    train_dataset = cls(root=root, train=True, download=True, transform=train_transform)
    test_dataset = cls(root=root, train=False, download=True, transform=test_transform)
    return train_dataset, test_dataset


def get_hf_text_dataset(
    dataset_name: str,
    max_vocab_size: int = 25000,
    min_freq: int = 2,
    train_subset_fn: Optional[Callable] = None,
    test_subset_fn: Optional[Callable] = None,
) -> Tuple[TextDataset, TextDataset, Dict[str, int]]:
    """Load a HuggingFace text dataset and return TextDatasets with a shared vocabulary.

    Downloads the requested dataset via the HuggingFace ``datasets`` library,
    builds a vocabulary from the training split using whitespace and punctuation removal tokenization
    (lowercased and punctuation removed), and encodes both splits into integer-index sequences.

    Supported Datasets
    ------------------
    "imdb"
        The Large Movie Review Dataset for binary sentiment classification
        (positive / negative). Contains 25,000 training reviews and 25,000
        test reviews drawn from IMDB. Labels are 0 (negative) and 1 (positive).

    "ag_news"
        AG News Corpus for 4-class news topic classification. Contains 120,000
        training articles and 7,600 test articles. Labels are 0 (World),
        1 (Sports), 2 (Business), and 3 (Sci/Tech).

    Args:
        dataset_name: Identifier for which dataset to load.
            Supported values: "imdb", "ag_news".
        max_vocab_size: Maximum vocabulary size (excluding special tokens
            ``<PAD>`` and ``<UNK>``). Defaults to 25000.
        min_freq: Minimum word frequency required for a token to be included
            in the vocabulary. Defaults to 2.
        train_subset_fn: Optional callable applied to the raw HuggingFace train
            split before tokenization (e.g., to subsample for faster experiments).
        test_subset_fn: Optional callable applied to the raw HuggingFace test
            split before tokenization.

    Returns:
        Tuple of (train_dataset, test_dataset, vocab):
            train_dataset: TextDataset for the training split.
            test_dataset: TextDataset for the test/evaluation split.
            vocab: Word-to-index dictionary (includes ``<PAD>`` at 0 and
                ``<UNK>`` at 1).

    Raises:
        ValueError: If dataset_name is unsupported.
    """
    if dataset_name == "imdb":
        hf_name = "imdb"
        train_split = "train"
        test_split = "test"
        text_key = "text"
        label_key = "label"
    elif dataset_name == "ag_news":
        hf_name = "ag_news"
        train_split = "train"
        test_split = "test"
        text_key = "text"
        label_key = "label"
    else:
        raise ValueError(
            f"Unsupported dataset_name='{dataset_name}'. "
            "Supported values: 'imdb', 'ag_news'."
        )

    ds = load_dataset(hf_name)
    train_data = ds[train_split]
    test_data = ds[test_split]

    if train_subset_fn is not None:
        train_data = train_subset_fn(train_data)
    if test_subset_fn is not None:
        test_data = test_subset_fn(test_data)

    def tokenize(text):
        text = text.lower()
        # remove punctuation but keep letters, numbers, spaces and apostrophes
        text = re.sub(r"[^\w\s\']", "", text)
        return text.split()

    train_tokens = [tokenize(sample[text_key]) for sample in train_data]
    test_tokens = [tokenize(sample[text_key]) for sample in test_data]

    vocab = build_vocab(train_tokens, max_vocab_size=max_vocab_size, min_freq=min_freq)

    def encode(tokens, vocab):
        return [vocab.get(t, vocab["<UNK>"]) for t in tokens]

    train_ids = [encode(tokens, vocab) for tokens in train_tokens]
    test_ids = [encode(tokens, vocab) for tokens in test_tokens]

    train_labels = [sample[label_key] for sample in train_data]
    test_labels = [sample[label_key] for sample in test_data]

    train_dataset = TextDataset(train_ids, train_labels)
    test_dataset = TextDataset(test_ids, test_labels)

    return train_dataset, test_dataset, vocab
