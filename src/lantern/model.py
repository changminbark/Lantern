"""
model.py

PyTorch model definitions for the lantern library.

Includes MLP, CNN (with optional residual blocks), Bag-of-Embeddings, TextCNN1D,
and SkipGram architectures, all configured via ModelConfig.

Course: CSCI 357 - AI and Neural Networks
Author: Chang Min Bark
"""

import math
from dataclasses import asdict
from typing import Any, Dict

import torch
import torch.nn as nn

from lantern.config import ModelConfig, ModelType, ResidualBlockConfig


class MLP_Model(nn.Module):
    """A configurable multi-layer perceptron with ReLU activations and optional dropout."""

    def __init__(self, num_inputs: int, num_outputs: int, config: ModelConfig):
        """Build the MLP layers from the given ModelConfig.

        Args:
            num_inputs: Dimensionality of the input features.
            num_outputs: Number of output classes/logits.
            config: Specifies hidden layer sizes and dropout rates.
        """
        super().__init__()

        if config.model_type != ModelType.MLP:
            raise ValueError(
                f"Invalid model_type: {config.model_type}. Expected 'ModelType.MLP'."
            )

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.config = config

        layers = [nn.Flatten()]

        # Dynamically construct hidden layers with ReLU + optional dropout
        prev_dim = num_inputs
        for hidden_layer, dropout_rate in zip(
            self.config.hidden_units, self.config.dropout
        ):
            layers.append(nn.Linear(prev_dim, hidden_layer))
            layers.append(nn.ReLU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_layer
        # Final projection to output logits (no activation)
        layers.append(nn.Linear(prev_dim, num_outputs))

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pass input through all layers and return raw logits.

        Args:
            x: Input tensor of shape (batch_size, num_inputs) or any shape that flattens to num_inputs.

        Returns:
            Logits tensor of shape (batch_size, num_outputs).
        """
        return self.layers(x)

    def num_parameters(self) -> tuple[int, int]:
        """Count the total and trainable parameters in the model.

        Returns:
            A tuple of (total_parameters, trainable_parameters).
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

    def get_architecture_config(self) -> Dict[str, Any]:
        """Return a serializable dict describing the model architecture for checkpointing.

        Returns:
            A dict with keys ``model_type``, ``num_inputs``, ``num_outputs``, and ``config``.
        """
        return {
            "model_type": self.config.model_type,
            "num_inputs": self.num_inputs,
            "num_outputs": self.num_outputs,
            "config": asdict(self.config),
        }

    def __str__(self) -> str:
        """Return a human-readable summary of the model configuration."""
        return f"MLP_Model(num_inputs={self.num_inputs}, num_outputs={self.num_outputs}, config={self.config!r})"

    def __repr__(self) -> str:
        """Return the same string as __str__ for consistent display."""
        return str(self)


class ResidualBlock(nn.Module):
    """
    Implements a single residual block inspired by the architecture introduced in ResNet (He et al., 2015).

    This block consists of two convolutional layers each followed by batch normalization,
    with a skip connection (shortcut path) from the input to the output. The main idea
    is to allow the network to learn residual mappings, which helps in training deeper
    neural networks by alleviating the vanishing gradient problem and making optimization easier.

    Architecture:
        x --> Conv -> BN -> ReLU -> Conv -> BN --> (+) -> ReLU -> out
        |                                          ^
        └──────────── shortcut (identity or 1x1) ──┘

    If in_channels != out_channels or stride != 1, the shortcut uses a 1x1 convolution (with batch norm)
    to match the shape of the main path; otherwise, it is the identity.

    References:
        Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.
        "Deep Residual Learning for Image Recognition." arXiv preprint arXiv:1512.03385 (2015).
    """

    def __init__(self, in_channels: int, res_config: ResidualBlockConfig):
        """Build the residual and shortcut paths from the given config.

        Args:
            in_channels: Number of input feature maps.
            res_config: Specifies out_channels and stride for the block.
        """
        super().__init__()
        # Build the two-conv "residual path", and don't forget that no bias is
        # necessary for the convolutions because batch norm has its own bias.
        res_path_layers = []
        # Downsizing (via stride) and filter/feature extraction (in_channel -> out_channel) happens in the first conv layer
        res_path_layers.append(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=res_config.out_channels,
                kernel_size=3,
                stride=res_config.stride,
                padding=1,
                bias=False,
            )
        )
        res_path_layers.append(nn.BatchNorm2d(res_config.out_channels))
        res_path_layers.append(nn.ReLU())
        res_path_layers.append(
            nn.Conv2d(
                in_channels=res_config.out_channels,
                out_channels=res_config.out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            )
        )
        res_path_layers.append(nn.BatchNorm2d(res_config.out_channels))
        self.res_path = nn.Sequential(*res_path_layers)

        # If dimensions change, use a 1x1 conv to project x to the right shape.
        # Otherwise, the shortcut is just the identity (nn.Sequential() with no layers).
        self.shortcut = nn.Sequential()
        if res_config.stride != 1 or in_channels != res_config.out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=res_config.out_channels,
                    kernel_size=1,
                    stride=res_config.stride,
                    bias=False,
                ),
                nn.BatchNorm2d(res_config.out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the residual block: res_path(x) + shortcut(x), then ReLU.

        Args:
            x: Input tensor of shape (batch, in_channels, H, W).

        Returns:
            Output tensor of shape (batch, out_channels, H', W').
        """
        # 1. Pass x through conv1 -> bn1 -> relu
        # 2. Pass through conv2 -> bn2
        # 3. Add the shortcut(x)
        # 4. Apply final ReLU
        res_path_output = self.res_path(x)
        shortcut_output = self.shortcut(x)
        x = res_path_output + shortcut_output
        return nn.functional.relu(x)


class CNN_Model(nn.Module):
    """Convolutional Neural Network following the [Conv2d -> ReLU -> MaxPool2d] x N motif.

    The model consists of:
      - A feature extractor: sequential conv blocks built from config.conv_blocks
      - A classifier head: Flatten -> Linear layers with dropout

    The flattened feature dimension is computed automatically via a dummy forward pass,
    so the model adapts to any input spatial size without manual calculation.
    """

    def __init__(
        self, input_height: int, input_width: int, num_outputs: int, config: ModelConfig
    ) -> None:
        """Build the CNN feature extractor and classifier head from the given config.

        Args:
            input_height: Height of the input images in pixels.
            input_width: Width of the input images in pixels.
            num_outputs: Number of output classes/logits.
            config: Specifies conv blocks, hidden units, dropout, and other settings.
        """
        super().__init__()

        if config.model_type != ModelType.CNN:
            raise ValueError(
                f"Invalid model_type: {config.model_type}. Expected 'ModelType.CNN'."
            )

        self.input_height = input_height
        self.input_width = input_width
        self.num_outputs = num_outputs
        self.config = config

        # --- Build the feature extractor ---
        # Your list of layers in self.feature_extractor is determined by your config.conv_blocks.
        # Loop through config.conv_blocks. For each ConvBlockConfig, append:
        #   nn.Conv2d(parameters from block configg)
        #   nn.BatchNorm2d(conv layer outputs) to normalize the output of conv layer by normalizing each channel across batch and spatial dimensions
        #   nn.ReLU()
        #   nn.MaxPool2d(pool_size from block config) <-- only if block.pool_size > 0
        # Track current_in_channels: starts at config.in_channels, then becomes block.out_channels.
        conv_layers = []
        current_in_channels = self.config.in_channels
        for conv_block in self.config.conv_blocks:
            if isinstance(conv_block, ResidualBlockConfig):
                conv_layers.append(ResidualBlock(current_in_channels, conv_block))
            else:
                conv_layers.append(
                    nn.Conv2d(
                        current_in_channels,
                        conv_block.out_channels,
                        conv_block.kernel_size,
                        conv_block.stride,
                        conv_block.padding,
                    )
                )
                if conv_block.batch_norm:
                    conv_layers.append(nn.BatchNorm2d(conv_block.out_channels))
                conv_layers.append(nn.ReLU())
                if conv_block.pool_size > 0:
                    conv_layers.append(nn.MaxPool2d(conv_block.pool_size))
            current_in_channels = conv_block.out_channels

        self.feature_extractor = nn.Sequential(*conv_layers)

        if self.config.use_GAP:
            self.gap = nn.AdaptiveAvgPool2d((1, 1))
            self._flat_features = current_in_channels
        else:
            self.gap = None
            # --- Compute flattened feature dimension via dummy forward pass ---
            with torch.no_grad():
                self.feature_extractor.eval()
                sample_tensor = torch.ones(
                    (1, self.config.in_channels, self.input_height, self.input_width)
                )
                self._flat_features = self.feature_extractor(sample_tensor).numel()
                self.feature_extractor.train()

        # --- Build the classifier head ---
        self.classifier_head = _construct_fc_layers(
            start_layer_size=self._flat_features,
            config=self.config,
            num_outputs=num_outputs,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pass input through feature extractor, flatten or GAP, then classifier head.

        Args:
            x: Input tensor of shape (batch, in_channels, H, W).

        Returns:
            Logits tensor of shape (batch, num_outputs).
        """
        # Pass x through self.feature_extractor, flatten/GAP, then self.classifier_head
        x = self.feature_extractor(x)
        # If GAP is enabled, apply global average pooling and squeeze spatial dimensions
        if self.gap is not None:
            x = self.gap(x)  # (batch, channels, 1, 1)
            x = x.squeeze(-1).squeeze(-1)  # (batch, channels)
        # Otherwise, flatten the feature maps into a 1D vector for the classifier head.
        else:
            x = torch.flatten(x, start_dim=1)
        x = self.classifier_head(x)
        return x

    def num_parameters(self) -> tuple[int, int]:
        """Count total and trainable parameters in the model.

        Returns:
            A tuple of (total_parameters, trainable_parameters).
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params, trainable_params

    def get_architecture_config(self) -> dict:
        """Return a serializable dict describing the model architecture for checkpointing.

        Returns:
            A dict with keys ``model_type``, ``input_height``, ``input_width``,
            ``num_outputs``, and ``config`` (with conv blocks tagged by ``block_type``).
        """

        def _serialize_config() -> dict:
            config_dict = asdict(self.config)  # still use asdict for everything else
            config_dict["conv_blocks"] = [
                {"block_type": "residual", **asdict(b)}
                if isinstance(b, ResidualBlockConfig)
                else {"block_type": "conv", **asdict(b)}
                for b in self.config.conv_blocks
            ]
            return config_dict

        return {
            "model_type": self.config.model_type,
            "input_height": self.input_height,
            "input_width": self.input_width,
            "num_outputs": self.num_outputs,
            "config": _serialize_config(),
        }

    def __str__(self) -> str:
        """Returns a string representation of the model.

        This provides a concise summary including input shape, number of input channels,
        the convolutional blocks specification, classifier head architecture, and total parameters.

        Returns:
            str: Human-readable summary of model architecture and size.
        """

        def block_str(b):
            if isinstance(b, ResidualBlockConfig):
                return f"Residual(out={b.out_channels}, s={b.stride})"
            return f"Conv(out={b.out_channels}, k={b.kernel_size}, s={b.stride}, p={b.padding}, bn={b.batch_norm})"

        blocks_summary = (
            "[" + ", ".join(block_str(b) for b in self.config.conv_blocks) + "]"
        )
        hidden_str = ", ".join(str(u) for u in self.config.hidden_units)
        head_summary = (
            f"[{self._flat_features} -> [{hidden_str}] -> {self.num_outputs}]"
        )
        dropout_summary = "[" + ", ".join(str(d) for d in self.config.dropout) + "]"
        return (
            f"CNN_Model(input={self.input_height}x{self.input_width}, in_channels={self.config.in_channels})\n"
            f"- blocks={blocks_summary}\n"
            f"- head={head_summary}\n"
            f"- dropout={dropout_summary}\n"
        )

    def __repr__(self) -> str:
        return self.__str__()


class TextCNN1D(nn.Module):
    """1D CNN text classifier with multiple filter sizes.

    Implements the Kim (2014) sentence classification architecture: parallel
    Conv1d branches with different filter widths each perform max-over-time
    pooling, and the resulting feature vectors are concatenated before a
    single linear output layer.

    Architecture:
        Embedding -> [Conv1d(fs) -> ReLU -> MaxPool] for each fs
                  -> Concatenate -> Classifier Head (Linear -> ReLU -> Dropout) x N -> Linear

    Args:
        num_outputs (int): Number of output classes (e.g., 2 for binary sentiment).
        config (ModelConfig): Hyperparameter configuration. Relevant fields:
            vocab_size        -- number of rows in the embedding matrix
            embedding_dim     -- width (D) of each embedding vector
            padding_idx       -- token ID treated as padding (zero vector, not updated)
            freeze_embeddings -- if True, embedding weights are not updated during training
            hidden_units      -- list of hidden layer sizes for the classifier head
            dropout           -- list of dropout rates, one per hidden layer
            TWO PARAMETERS SPECIFIC TO TextCNN1D
            num_filters       -- Number of output channels for each Conv1d branch.
            filter_sizes      -- (tuple[int, ...]): Kernel widths for each parallel branch.

    Raises:
        ValueError: If ``config.model_type`` is not ``'textcnn'``.

    References:
        Yoon Kim. "Convolutional Neural Networks for Sentence Classification."
        EMNLP 2014. https://arxiv.org/abs/1408.5882
    """

    def __init__(self, num_outputs: int, config: ModelConfig):
        """Initializes the TextCNN1D model, building embedding, conv, and output layers.

        Args:
            num_outputs (int): Number of output classes or regression targets.
            config (ModelConfig): Model configuration; must have ``model_type == 'textcnn'``.

        Raises:
            ValueError: If ``config.model_type`` is not ``'textcnn'``.
        """
        super().__init__()

        if config.model_type != ModelType.TEXTCNN:
            raise ValueError(
                f"Invalid model_type: {config.model_type}. Expected 'textcnn'."
            )

        # Store hyperparameters so they can be retrieved later (e.g., get_architecture_config).
        self.num_outputs = num_outputs
        self.config = config

        # Embedding table: maps each integer token ID to a dense vector of size embedding_dim.
        # padding_idx ensures the <PAD> row stays at zero and receives no gradient updates.
        self.embedding = nn.Embedding(
            num_embeddings=self.config.vocab_size,
            embedding_dim=self.config.embedding_dim,
            padding_idx=self.config.padding_idx,
        )

        # Provide the option to freeze pretrained embeddings (e.g., GloVe)
        if self.config.freeze_embeddings:
            self.embedding.requires_grad_(False)

        # One Conv1d branch per filter size; each branch independently scans the sequence
        # with a different n-gram width, capturing features at different granularities.
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=self.config.embedding_dim,
                    out_channels=self.config.num_filters,
                    kernel_size=fs,
                )
                for fs in self.config.filter_sizes
            ]
        )

        # Concatenated output of all branches feeds into the classifier head.
        self.classifier_head = _construct_fc_layers(
            start_layer_size=self.config.num_filters * len(self.config.filter_sizes),
            config=self.config,
            num_outputs=num_outputs,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Runs the forward pass.

        Args:
            x (torch.Tensor): LongTensor of token IDs, shape ``(batch_size, seq_len)``.

        Returns:
            torch.Tensor: Logit tensor of shape ``(batch_size, num_outputs)``.
        """
        # Look up a dense vector for every token ID in the sequence.
        embedded = self.embedding(x)  # (batch, seq_len, embed_dim)

        # Conv1d expects (batch, channels, length), so permute embed_dim to the channel axis.
        embedded = embedded.permute(0, 2, 1)  # (batch, embed_dim, seq_len)

        # Run each branch: convolve -> activate -> max-pool over the entire sequence length.
        # Max-over-time pooling reduces variable-length feature maps to a single scalar per
        # filter, making the representation independent of sequence length.
        pooled = [
            torch.relu(conv(embedded)).max(dim=2).values  # (batch, num_filters)
            for conv in self.convs
        ]

        # Concatenate the pooled vectors from all branches into one feature vector,
        # then pass through the classifier head (dropout + linear layers).
        features = torch.cat(pooled, dim=1)  # (batch, num_filters * len(filter_sizes))

        return self.classifier_head(features)  # (batch, num_outputs)

    def num_parameters(self) -> tuple[int, int]:
        """Returns the total and trainable parameter counts.

        Returns:
            tuple[int, int]: ``(total_params, trainable_params)``.
        """
        # Count every parameter (frozen embeddings are included in total but not trainable).
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

    def get_architecture_config(self) -> dict:
        """Returns a serializable dictionary describing the full model architecture.

        The returned dictionary contains all information needed to reconstruct
        this model instance and is suitable for logging or checkpointing.

        Returns:
            dict: A dictionary with keys:
                - ``'model_type'`` (str): Always ``'textcnn'``.
                - ``'num_outputs'`` (int): Number of output classes.
                - ``'config'`` (dict): Dataclass-serialized ``ModelConfig``.
        """

        return {
            "model_type": "textcnn",
            "num_outputs": self.num_outputs,
            "config": asdict(self.config),
        }

    def __str__(self) -> str:
        """Returns a concise human-readable summary of the model architecture.

        Returns:
            str: Single-line description including vocab size, embedding dimension,
                filter configuration, and number of outputs.
        """
        # Reflect the freeze status so it's visible at a glance when printing the model.
        frozen = "frozen" if self.config.freeze_embeddings else "trainable"
        return (
            f"TextCNN1D(vocab={self.config.vocab_size}, embed_dim={self.config.embedding_dim} ({frozen}), "
            f"num_filters={self.config.num_filters}, filter_sizes={self.config.filter_sizes}, "
            f"num_outputs={self.num_outputs})"
        )

    def __repr__(self) -> str:
        return self.__str__()


class BagOfEmbeddings(nn.Module):
    """Bag-of-Embeddings text classifier.

    Architecture:
        Embedding -> Masked Mean Pool -> Classifier Head

    The embedding layer converts token indices to dense vectors. Mean pooling
    averages the token embeddings over the sequence dimension, *excluding* padding
    tokens, producing a fixed-size document vector regardless of sequence length.
    The classifier head is a stack of Linear + ReLU + Dropout layers, reusing
    the same motif as MLP_Model.

    This is the NLP analog of Global Average Pooling (GAP) from CNN architectures:
    GAP averages over spatial dimensions (H, W); mean pooling averages over the
    sequence dimension L.

    Args:
        num_outputs (int): Number of output classes (e.g., 2 for binary sentiment).
        config (ModelConfig): Hyperparameter configuration. Relevant fields:
            vocab_size      -- number of rows in the embedding matrix
            embedding_dim   -- width (D) of each embedding vector
            padding_idx     -- token ID treated as padding (kept at zero vector)
            freeze_embeddings -- if True, embedding weights are not updated
            hidden_units    -- list of hidden layer sizes for the classifier head
            dropout         -- list of dropout rates (one per hidden layer)
    """

    def __init__(self, num_outputs: int, config: ModelConfig):
        """Build embedding layer and classifier head from config.

        Args:
            num_outputs: Number of output classes.
            config: Model configuration; must have ``model_type == 'bow'``.

        Raises:
            ValueError: If ``config.model_type`` is not ``ModelType.BOW``.
        """
        super().__init__()
        if config.model_type != ModelType.BOW:
            raise ValueError(
                f"Invalid model_type: {config.model_type}. Expected 'bow'."
            )

        self.num_outputs = num_outputs
        self.config = config

        # --- Embedding layer ---
        # Create self.embedding using nn.Embedding.
        #       Pass config.vocab_size, config.embedding_dim, and config.padding_idx.
        #       PyTorch will automatically keep the padding_idx row as a zero vector
        #       and will not update it during backprop.
        self.embedding = nn.Embedding(
            num_embeddings=self.config.vocab_size,
            embedding_dim=self.config.embedding_dim,
            padding_idx=self.config.padding_idx,
        )

        # If config.freeze_embeddings is True, prevent the embedding weights
        #       from receiving gradient updates by setting requires_grad = False.
        #       This is useful when you want to preserve pretrained GloVe knowledge
        #       and only train the classifier head (see Challenge 1).
        if self.config.freeze_embeddings:
            self.embedding.requires_grad_(False)

        # --- Classifier head: embedding_dim -> hidden_units -> num_outputs ---
        self.classifier_head = _construct_fc_layers(
            start_layer_size=self.config.embedding_dim,
            config=self.config,
            num_outputs=num_outputs,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: LongTensor of token IDs, shape (batch_size, seq_len).

        Returns:
            Logit tensor of shape (batch_size, num_outputs).
        """
        # Step 1 — Look up embeddings for each token ID.
        #       self.embedding(x) maps each integer ID to its D-dimensional vector.
        #       Result shape: (batch_size, seq_len, embedding_dim)
        embedded = self.embedding(x)

        # Step 2 — Build the padding mask.
        #       Create a float tensor that is 1.0 where x != padding_idx, 0.0 elsewhere.
        #       Unsqueeze the last dimension so it broadcasts over embed_dim.
        #       Shape after unsqueeze: (batch_size, seq_len, 1)
        mask = (x != self.config.padding_idx).unsqueeze(-1).float()

        # Step 3 — Masked mean pooling.
        #       Multiply embeddings by the mask to zero out padding positions,
        #       sum over the sequence dimension (dim=1), then divide by the count
        #       of real tokens. Clamp the denominator to at least 1 to avoid
        #       division-by-zero on hypothetical all-padding sequences.
        summed = (embedded * mask).sum(dim=1)  # (batch, embed_dim)
        lengths = mask.sum(dim=1).clamp(min=1)  # (batch, 1)
        pooled = summed / lengths  # (batch, embed_dim)

        # Step 4 — Pass the document vector through the classifier head.
        return self.classifier_head(pooled)

    def num_parameters(self) -> tuple[int, int]:
        """Returns (total_params, trainable_params)."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

    def get_architecture_config(self) -> dict:
        """Returns a serializable dictionary describing the full model architecture.

        Returns:
            dict: A dictionary with keys:
                - ``'model_class'`` (str): Always ``'BagOfEmbeddings'``.
                - ``'num_outputs'`` (int): Number of output classes.
                - ``'config'`` (dict): Dataclass-serialized ``ModelConfig``.
        """
        from dataclasses import asdict

        return {
            "model_class": "BagOfEmbeddings",
            "num_outputs": self.num_outputs,
            "config": asdict(self.config),
        }

    def __str__(self) -> str:
        """Return a human-readable summary of the model configuration."""
        frozen = "frozen" if self.config.freeze_embeddings else "trainable"
        return (
            f"BagOfEmbeddings(vocab={self.config.vocab_size}, "
            f"embed_dim={self.config.embedding_dim}, {frozen})\n"
            f"  head=[{self.config.embedding_dim} -> "
            f"{self.config.hidden_units} -> {self.num_outputs}]\n"
            f"  dropout={self.config.dropout}"
        )

    def __repr__(self) -> str:
        """Return the same string as __str__ for consistent display."""
        return self.__str__()


class SkipGram(nn.Module):
    """Skip-gram word2vec model.

    Maps a center token ID through an embedding table and a linear projection
    to produce logits over the full vocabulary. Trained with cross-entropy loss
    against context token IDs generated by ``generate_skip_gram_pairs``.

    Architecture:
        Embedding(vocab_size, embedding_dim) -> Linear(embedding_dim, vocab_size)
    """

    def __init__(self, config: ModelConfig):
        """Build the embedding and output projection layers from config.

        Args:
            config: Model configuration; must have ``model_type == 'skipgram'``.
                Uses ``config.vocab_size`` and ``config.embedding_dim``.

        Raises:
            ValueError: If ``config.model_type`` is not ``ModelType.SKIPGRAM``.
        """
        super().__init__()
        if config.model_type != ModelType.SKIPGRAM:
            raise ValueError(
                f"Invalid model_type: {config.model_type}. Expected 'ModelType.SKIPGRAM'."
            )
        self.config = config
        self.num_outputs = self.config.vocab_size
        self.embedding = nn.Embedding(self.config.vocab_size, self.config.embedding_dim)
        self.linear = nn.Linear(self.config.embedding_dim, self.config.vocab_size)

    def forward(self, center_ids: torch.Tensor) -> torch.Tensor:
        """Embed center token IDs and project to vocabulary logits.

        Args:
            center_ids: LongTensor of center token IDs, shape (batch_size,).

        Returns:
            Logit tensor of shape (batch_size, vocab_size).
        """
        emb = self.embedding(center_ids)
        return self.linear(emb)

    def num_parameters(self) -> tuple[int, int]:
        """Returns (total_params, trainable_params)."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

    def get_architecture_config(self) -> dict:
        """Returns a serializable dictionary describing the model architecture."""
        return {
            "model_class": "SkipGram",
            "config": asdict(self.config),
        }

    def __str__(self) -> str:
        return f"SkipGram(vocab_size={self.config.vocab_size}, embedding_dim={self.config.embedding_dim})"

    def __repr__(self) -> str:
        return self.__str__()


class RNNModel(nn.Module):
    """Vanilla RNN (or LSTM or GRU) for regression or classification on continuous sequences.

    This model processes sequence data (such as time series) using a recurrent neural network backbone (RNN, LSTM, or GRU),
    followed by a fully connected classifier or regressor head. It does not perform embedding or tokenization and is assumed
    to receive raw float inputs.

    Architecture:
        Input shape: (batch_size, seq_len, input_size)
            └──> nn.RNN / nn.LSTM / nn.GRU → (batch_size, seq_len, num_directions * hidden_size)
            └──> Final hidden state extraction → (batch_size, hidden_size * num_directions)
            └──> Fully connected head → (batch_size, num_outputs)

        Here ``num_directions`` is 1 for unidirectional and 2 for bidirectional RNNs (matching PyTorch).

    Example:
        >>> from lantern.config import ModelConfig
        >>> config = ModelConfig(
        ...     model_type="rnn",
        ...     rnn_hidden_size=32,
        ...     rnn_num_layers=2,
        ...     bidirectional=True,
        ...     rnn_type="lstm",
        ...     hidden_units=[16],
        ...     dropout=[0.1]
        ... )
        >>> model = RNNModel(input_size=8, num_outputs=3, config=config)
        >>> x = torch.randn(4, 22, 8)  # batch of 4, sequence length 22, 8 features
        >>> y = model(x)  # (4, 3)

    Attributes:
        input_size (int): Number of input features per time step (e.g., 1 for univariate).
        num_outputs (int): Number of output values (1 for regression, C for classification).
        config (ModelConfig): Configuration object.
        rnn (nn.Module): The RNN/LSTM/GRU module.
        classifier_head (nn.Sequential): Sequential fully connected head after the recurrent backbone.
    """

    def __init__(self, input_size: int, num_outputs: int, config: ModelConfig) -> None:
        """Build the recurrent backbone and output head.

        Args:
            input_size: Number of input features per time step (e.g., ``1`` for a
                univariate series).
            num_outputs: Number of outputs (``1`` for regression, ``C`` for
                ``C``-way classification).
            config: Model configuration. Uses ``rnn_hidden_size``,
                ``rnn_num_layers``, ``bidirectional``, ``rnn_type`` (``"rnn"``,
                ``"lstm"``, or ``"gru"``), and ``hidden_units`` / ``dropout`` for
                the fully connected head built by ``_construct_fc_layers``.

        Raises:
            ValueError: If ``config.model_type`` is not ``"rnn"``.
            ValueError: If ``config.rnn_type`` is not one of ``"rnn"``, ``"lstm"``, or ``"gru"``.

        """
        super().__init__()

        # Validate config.model_type is ModelType.RNN. If not, throw a ValueError
        if config.model_type != ModelType.RNN:
            raise ValueError("config.model_type is not set to `rnn`")

        # Store self copies of parameters
        self.config = config
        self.input_size = input_size
        self.num_outputs = num_outputs
        self.rnn_hidden_size = self.config.rnn_hidden_size
        self.rnn_num_layers = self.config.rnn_num_layers
        self.bidirectional = self.config.bidirectional
        self.rnn_type = self.config.rnn_type
        self.clip_grad_norm = self.config.clip_grad_norm

        # Determine if bidirectional is set, so you know how many directions your RNN will have.
        num_directions = 2 if self.bidirectional else 1

        # Build the recurrent backbone. Choose the correct nn module depending on rnn_type.
        # Throws a ValueError if not "rnn", "lstm" or "gru"
        if self.config.rnn_type == "lstm":
            rnn_module = nn.LSTM
        elif self.config.rnn_type == "gru":
            rnn_module = nn.GRU
        elif self.config.rnn_type == "rnn":
            rnn_module = nn.RNN
        else:
            raise ValueError(
                f"Invalid rnn_type: {self.config.rnn_type}. Supported: 'rnn', 'lstm', 'gru'."
            )

        # Instantiate your recurrent module ("self.rnn") using the chosen rnn_module. Make sure
        #       to use input_size, hidden_size, num_layers, batch_first, and bidirectional from config.
        self.rnn = rnn_module(
            input_size=input_size,
            hidden_size=self.rnn_hidden_size,
            num_layers=self.rnn_num_layers,
            bidirectional=self.bidirectional,
            batch_first=True,
        )

        # The output head receives the concatenated final hidden state(s).
        #       Its input dimension is hidden_size * num_directions.
        head_input_size = self.config.rnn_hidden_size * num_directions

        # Call _construct_fc_layers to create a classifier/regressor head after the RNN
        self.classifier_head = _construct_fc_layers(
            head_input_size, self.config, num_outputs
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode each sequence and apply the output head.

        For LSTM, only the last hidden state ``h_n`` is used (not the cell state).
        For bidirectional models, the last forward and backward states of the top
        layer are concatenated before the head.

        Args:
            x: Float tensor of shape ``(batch, seq_len, input_size)``.

        Returns:
            Float tensor of shape ``(batch, num_outputs)``.
        """
        # Run through the recurrent backbone (RNN, LSTM, or GRU)
        # All three return (output, hidden_state); we only need the hidden state.
        _, hidden = self.rnn(x)

        # Extract the last hidden state(s) depending on RNN type
        if self.rnn_type == "lstm":
            # LSTM hidden_state is a tuple (h_n, c_n); discard cell state
            h_n, _ = hidden
        else:
            # GRU and vanilla RNN return h_n directly
            h_n = hidden

        # h_n shape: (num_layers * num_directions, batch, hidden_size)
        if self.bidirectional:
            # Forward final state: h_n[-2], Backward final state: h_n[-1]
            # Both have shape (batch, hidden_size), so we cat along dim=1
            final_state = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            # Use the last layer's final hidden state (shape: batch, hidden_size)
            final_state = h_n[-1]

        # Pass through classifier head
        return self.classifier_head(final_state)

    def num_parameters(self) -> tuple[int, int]:
        """Count total and trainable parameters in this module and submodules.

        Returns:
            A tuple ``(total_params, trainable_params)`` where both counts are
            non-negative integers.
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

    def get_architecture_config(self) -> dict:
        """Return a JSON-serializable description of the architecture.

        Returns:
            A dictionary with keys:

            * ``model_class``: The string ``"RNNModel"``.
            * ``input_size``: Features per time step.
            * ``num_outputs``: Size of the head output.
            * ``config``: The full ``ModelConfig`` as a plain dict (via
              ``dataclasses.asdict``).
        """
        return {
            "model_class": "RNNModel",
            "input_size": self.input_size,
            "num_outputs": self.num_outputs,
            "config": asdict(self.config),
        }

    def __str__(self) -> str:
        """Return a short, human-readable summary of hyperparameters.

        Returns:
            A string listing ``rnn_type``, directionality (uni/bi), ``input_size``,
            ``rnn_hidden_size``, ``rnn_num_layers``, and ``num_outputs``.
        """
        direction = "bi" if self.config.bidirectional else "uni"
        return (
            f"RNNModel(type={self.config.rnn_type}, {direction}, "
            f"input={self.input_size}, hidden={self.config.rnn_hidden_size}, "
            f"layers={self.config.rnn_num_layers}, out={self.num_outputs})"
        )

    def __repr__(self) -> str:
        """Return the same string as :meth:`__str__` for notebook and debugger display.

        Returns:
            Identical to :meth:`__str__`.
        """
        return self.__str__()


class TextRNNModel(nn.Module):
    """RNN text classifier using an embedding layer and RNN backbone.

    This model maps input token ID sequences to embeddings, processes them with an RNN or LSTM,
    and classifies the resulting sequence representations. Handles variable-length padded sequences
    by extracting the hidden state at the last non-padded token for each example in the batch.

    Architecture:
        Token IDs -> Embedding -> RNN -> Final Hidden State -> Classifier Head

    Handles padded sequences by extracting the hidden state at the actual
    (non-padded) last time step for each sequence in the batch.

    Args:
        num_outputs (int): Number of output classes.
        config (ModelConfig): Configuration object containing model hyperparameters, including:
            - vocab_size (int): Vocabulary size for the embedding layer.
            - embedding_dim (int): Dimensionality of the embedding vectors.
            - padding_idx (int): Token index used for padding.
            - freeze_embeddings (bool): If True, the embedding weights are not updated during training.
            - rnn_hidden_size (int): Hidden size of the RNN/LSTM.
            - rnn_num_layers (int): Number of stacked RNN/LSTM layers.
            - bidirectional (bool): If True, use a bidirectional RNN/LSTM.
            - rnn_type (str): Type of RNN backbone ("rnn", "lstm", or "gru").
            - hidden_units (list[int] or None): MLP head hidden layer sizes.
            - dropout (list[float]): Dropout rates for the classifier head layers.
    """

    def __init__(self, num_outputs: int, config: ModelConfig):
        super().__init__()

        if config.model_type != ModelType.TEXTRNN:
            raise ValueError(
                f"Invalid model_type: {config.model_type}. Expected 'text_rnn'."
            )

        self.num_outputs = num_outputs
        self.config = config

        # Set up embedding layer (same as BoE), and freeze the embeddings if requested.
        self.embedding = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.embedding_dim,
            padding_idx=config.padding_idx,
        )
        if self.config.freeze_embeddings:
            self.embedding.requires_grad_(False)

        # Determine if bidirectional is set, so you know how many directions your RNN will have.
        num_directions = 2 if self.config.bidirectional else 1

        # Build the recurrent backbone. Choose the correct nn module depending on rnn_type.
        # Throws a ValueError if not "rnn", "lstm" or "gru"
        if self.config.rnn_type == "lstm":
            rnn_module = nn.LSTM
        elif self.config.rnn_type == "gru":
            rnn_module = nn.GRU
        elif self.config.rnn_type == "rnn":
            rnn_module = nn.RNN
        else:
            raise ValueError(
                f"Invalid rnn_type: {self.config.rnn_type}. Supported: 'rnn', 'lstm', 'gru'."
            )

        # Instantiate your recurrent module ("self.rnn") using embedding_dim as input_size
        self.rnn = rnn_module(
            input_size=self.config.embedding_dim,
            hidden_size=self.config.rnn_hidden_size,
            num_layers=self.config.rnn_num_layers,
            bidirectional=self.config.bidirectional,
            batch_first=True,
        )

        # The output head receives the concatenated final hidden state(s).
        # Its input dimension is hidden_size * num_directions.
        head_input_size = self.config.rnn_hidden_size * num_directions

        # Call _construct_fc_layers to create a classifier/regressor head after the RNN
        self.classifier_head = _construct_fc_layers(
            head_input_size, self.config, num_outputs
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: LongTensor of token IDs, shape (batch, seq_len).

        Returns:
            Logit tensor of shape (batch, num_outputs).
        """
        # Compute the actual (non-padding) length of each sequence.
        # Result shape: (batch,)
        lengths = (x != self.config.padding_idx).sum(dim=1)

        # Look up embeddings for each token ID.
        # Result shape: (batch, seq_len, embedding_dim)
        embedded = self.embedding(x)

        # Run the embedded sequence through the RNN backbone.
        # rnn_out shape: (batch, seq_len, hidden_size * num_directions)
        rnn_out, _ = self.rnn(embedded)

        # Build index tensors to select the last real token's hidden state.
        # last_idx: subtract 1 from each length; clamp to 0 to guard against empty sequences.
        last_idx = (lengths - 1).clamp(min=0)  # (batch,)
        batch_idx = torch.arange(x.size(0), device=x.device)  # (batch,)

        # Extract the final hidden state, handling bidirectionality.
        # For bidirectional models, the two directions finish at DIFFERENT positions:
        #   - Forward direction  → final state is at last_idx (last real token)
        #   - Backward direction → final state is at position 0
        if self.config.bidirectional:
            hidden_size = self.config.rnn_hidden_size
            fwd = rnn_out[batch_idx, last_idx, :hidden_size]  # (batch, hidden_size)
            bwd = rnn_out[batch_idx, 0, hidden_size:]  # (batch, hidden_size)
            final_hidden = torch.cat([fwd, bwd], dim=1)  # (batch, hidden_size*2)
        else:
            final_hidden = rnn_out[batch_idx, last_idx]  # (batch, hidden_size)

        # Pass the final hidden state through the classifier head.
        return self.classifier_head(final_hidden)

    def num_parameters(self) -> tuple[int, int]:
        """Count total and trainable parameters in this module and submodules.

        Returns:
            A tuple ``(total_params, trainable_params)`` where both counts are
            non-negative integers.
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

    def get_architecture_config(self) -> dict:
        """Return a JSON-serializable description of the architecture.

        Returns:
            A dictionary with keys:

            * ``model_class``: The string ``"TextRNNModel"``.
            * ``num_outputs``: Size of the head output.
            * ``config``: The full ``ModelConfig`` as a plain dict (via
              ``dataclasses.asdict``).
        """
        from dataclasses import asdict

        return {
            "model_class": "TextRNNModel",
            "num_outputs": self.num_outputs,
            "config": asdict(self.config),
        }

    def __str__(self) -> str:
        """Return a short, human-readable summary of hyperparameters.

        Returns:
            A string listing ``rnn_type``, directionality (uni/bi), ``vocab_size``,
            ``embedding_dim``, ``rnn_hidden_size``, ``rnn_num_layers``, and ``num_outputs``.
        """
        frozen = "frozen" if self.config.freeze_embeddings else "trainable"
        direction = "bi" if self.config.bidirectional else "uni"
        return (
            f"TextRNNModel(type={self.config.rnn_type}, {direction}, "
            f"vocab={self.config.vocab_size}, embed={self.config.embedding_dim} ({frozen}), "
            f"hidden={self.config.rnn_hidden_size}, layers={self.config.rnn_num_layers}, "
            f"out={self.num_outputs})"
        )

    def __repr__(self) -> str:
        """Return the same string as :meth:`__str__` for notebook and debugger display.

        Returns:
            Identical to :meth:`__str__`.
        """
        return self.__str__()


class Seq2SeqForecaster(nn.Module):
    """Encoder-decoder LSTM for multi-step time series forecasting.

    The encoder reads a sequence of historical values and produces a final
    hidden state. The decoder autoregressively generates future values,
    feeding each prediction as the next input.

    Args:
        input_size: Number of features per time step (1 for univariate).
        forecast_horizon: Number of future steps to predict.
        config: ModelConfig with model_type == ModelType.SEQ2SEQ.

    Raises:
        ValueError: If config.model_type is not ModelType.SEQ2SEQ.
    """

    def __init__(self, input_size: int, forecast_horizon: int, config: ModelConfig):
        super().__init__()
        if config.model_type != ModelType.SEQ2SEQ:
            raise ValueError(
                f"Invalid model_type: {config.model_type}. Expected 'ModelType.SEQ2SEQ'."
            )
        self.forecast_horizon = forecast_horizon
        self.hidden_size = config.rnn_hidden_size
        self.config = config

        # Create encoder LSTM (reads the input window)
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=self.config.rnn_hidden_size,
            num_layers=self.config.rnn_num_layers,
            bidirectional=self.config.bidirectional,
            batch_first=True,
        )

        # Create decoder LSTM (generates predictions step by step)
        self.decoder = nn.LSTM(
            input_size=input_size,
            hidden_size=self.config.rnn_hidden_size,
            num_layers=self.config.rnn_num_layers,
            bidirectional=self.config.bidirectional,
            batch_first=True,
        )

        # Linear layer to map decoder hidden state to a single value
        self.output_layer = _construct_fc_layers(
            start_layer_size=self.hidden_size, config=self.config, num_outputs=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, input_size).

        Returns:
            Predictions of shape (batch, forecast_horizon).
        """
        # Encode the input sequence
        _, (h, c) = self.encoder(x)

        # Autoregressive decoding
        # Start with the last known value as the first decoder input
        decoder_input = x[:, -1:, :]  # (batch, 1, input_size)

        predictions = []
        for _ in range(self.forecast_horizon):
            # One decoder step; initialize hidden state from encoder
            output, (h, c) = self.decoder(
                decoder_input, (h, c)
            )  # output has size (batch, 1, hidden_size)

            # Map to predicted value
            pred = self.output_layer(output.squeeze(1))  # (batch, 1)
            predictions.append(pred)

            # Feed prediction back as next input
            decoder_input = pred.unsqueeze(1)  # (batch, 1, input_size)

        return torch.cat(predictions, dim=1)  # (batch, forecast_horizon)


class AttentionClassifier(nn.Module):
    """Self-attention text classifier using nn.MultiheadAttention.

    Architecture:
        Token IDs -> Embedding -> MultiheadAttention (self-attention) ->
        Masked Mean Pool -> Classifier Head

    The model applies self-attention over the embedded token sequence, then
    performs masked mean pooling (excluding padding tokens) to produce a
    fixed-size document vector, which is passed through a fully connected
    classifier head.

    Args:
        num_outputs (int): Number of output classes.
        config (ModelConfig): Configuration containing:
            vocab_size (int): Vocabulary size for the embedding layer.
            embedding_dim (int): Dimensionality of each embedding vector.
                Must satisfy embedding_dim % num_heads == 0.
            padding_idx (int): Token index treated as padding.
            freeze_embeddings (bool): If True, embedding weights are frozen.
            num_heads (int): Number of attention heads.
            dropout (list[float]): Dropout rates for the classifier head.
            hidden_units (list[int]): Hidden layer sizes for the classifier head.
    """

    def __init__(self, num_outputs: int, config: ModelConfig):
        super().__init__()

        # Validate that config.model_type is "text_attn"
        # Raise ValueError if it's not
        if config.model_type != ModelType.TEXTATTN:
            raise ValueError(
                f"Invalid model_type: {config.model_type}. Expected 'text_attn'."
            )

        # Validate that embedding_dim is divisible by num_heads
        # This is required by nn.MultiheadAttention
        # Raise ValueError if the constraint is violated
        if config.embedding_dim % config.num_heads != 0:
            raise ValueError(
                "Invalid embedding dimension and number of heads. Expected config.embedding_dim % config.num_heads == 0"
            )

        # Store num_outputs and config as instance attributes
        self.num_outputs = num_outputs
        self.config = config

        # ── Embedding layer ──
        # Create the embedding layer
        # Use nn.Embedding with vocab_size, embedding_dim, and padding_idx from config
        self.embedding = nn.Embedding(
            num_embeddings=self.config.vocab_size,
            embedding_dim=self.config.embedding_dim,
            padding_idx=self.config.padding_idx,
        )

        # If config.freeze_embeddings is True, freeze the embedding weights
        if self.config.freeze_embeddings:
            self.embedding.weight.requires_grad = False

        # ── Self-attention layer ──
        # Create the self-attention layer using nn.MultiheadAttention
        # - embed_dim should be config.embedding_dim
        # - num_heads should be config.num_heads
        # - batch_first should be True (so input/output shape is (batch, seq_len, embed_dim))
        # - dropout should be config.dropout[0] if available, else 0.0
        self.attention = nn.MultiheadAttention(
            embed_dim=self.config.embedding_dim,
            num_heads=self.config.num_heads,
            dropout=self.config.dropout[0] if self.config.dropout else 0.0,
            batch_first=True,
        )

        # ── Classifier head: embedding_dim -> hidden_units -> num_outputs ──
        # =Create the classifier head using _construct_fc_layers
        # - start_layer_size should be config.embedding_dim (output of pooling)
        # - Pass config and num_outputs
        self.classifier_head = _construct_fc_layers(
            self.config.embedding_dim, self.config, num_outputs
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: LongTensor of token IDs, shape (batch, seq_len).

        Returns:
            Logit tensor of shape (batch, num_outputs).
        """
        # ── Step 1: Build the padding mask ──
        # True where x IS padding (PyTorch convention: True = ignore)
        padding_mask = x == self.config.padding_idx  # (batch, seq_len)

        # ── Step 2: Embed tokens ──
        embedded = self.embedding(x)

        # ── Step 3: Self-attention ──
        # For self-attention, query = key = value = embedded
        # key_padding_mask tells attention to ignore padding positions
        attn_out, _ = self.attention(
            embedded, embedded, embedded, key_padding_mask=padding_mask
        )  # attn_out: (batch, seq_len, embed_dim)

        # ── Step 4: Masked mean pooling (exclude padding) ──
        # Zero out padding positions so they don't contribute to the sum
        mask = (~padding_mask).unsqueeze(-1).float()  # (batch, seq_len, 1)
        attn_out = attn_out * mask

        # Count real (non-padding) tokens per sequence
        lengths = mask.sum(dim=1)  # (batch, 1)
        lengths = lengths.clamp(min=1)  # avoid division by zero

        # Sum over sequence dim and divide by real token count
        pooled = attn_out.sum(dim=1) / lengths  # (batch, embed_dim)

        # ── Step 5: Classifier head ──
        return self.classifier_head(pooled)

    def num_parameters(self) -> tuple:
        """Return (total_params, trainable_params)."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer models.

    Adds position-dependent sinusoidal signals to the input embeddings so that
    the transformer can distinguish token positions despite self-attention being
    permutation-invariant. Uses the formulation from Vaswani et al. (2017):
        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    The PE matrix is precomputed for positions 0..max_len-1 and registered as a
    buffer (not a parameter) so it moves with the model to GPU but is not updated
    by the optimizer.

    Args:
        d_model (int): Embedding / model dimension.
        max_len (int): Maximum sequence length to precompute encodings for.
        dropout (float): Dropout probability applied after adding PE.
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # ── Precompute the sinusoidal PE matrix ──
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()  # (max_len, 1)

        # Compute the division term: 10000^(2i/d_model) via log-space for numerical stability
        i = torch.arange(0, d_model, 2).float()
        # a^b = exp(b * log(a))
        div_term = torch.exp((-i / d_model) * math.log(10000.0))  # (d_model/2,)

        # Even indices: sin; odd indices: cos; (1, max_len, d_model) for broadcasting
        pe[:, 0::2] = torch.sin(position * div_term)  # even indices: sin
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices: cos
        pe = pe.unsqueeze(0)  # (1, max_len, d_model) for broadcasting

        # Register as buffer — moves with .to(device) but is not a trainable parameter
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to the input embeddings.

        Args:
            x: Tensor of shape (batch, seq_len, d_model).

        Returns:
            Tensor of same shape with positional encoding added and dropout applied.
        """
        x = x + self.pe[:, : x.size(1), :]  # slice PE to match actual seq_len
        return self.dropout(x)


class TransformerClassifier(nn.Module):
    """Transformer encoder text classifier using PyTorch's built-in modules.

    Architecture:
        Token IDs -> Embedding (* sqrt(d_model)) -> PositionalEncoding ->
        TransformerEncoder (N layers) -> Masked Mean Pool -> Classifier Head

    Each TransformerEncoderLayer contains self-attention + FFN + residual + LayerNorm.
    The model uses Pre-LN ordering (norm_first=True) for training stability.

    Args:
        num_outputs (int): Number of output classes.
        config (ModelConfig): Configuration containing:
            model_type (str): Must be "ModelType.TEXTTRANSFORMER".
            vocab_size (int): Vocabulary size for the embedding layer.
            embedding_dim (int): Model dimension (d_model). Must be divisible by num_heads.
            padding_idx (int): Token index treated as padding.
            freeze_embeddings (bool): If True, embedding weights are frozen.
            num_heads (int): Number of attention heads per encoder layer.
            num_encoder_layers (int): Number of stacked encoder layers (N).
            dim_feedforward (int): Hidden dimension of the FFN sublayer.
            dropout (list[float]): Dropout rates for the classifier head.
            hidden_units (list[int]): Hidden layer sizes for the classifier head.
    """

    def __init__(self, num_outputs: int, config: ModelConfig):
        super().__init__()

        if config.model_type != ModelType.TEXTTRANSFORMER:
            raise ValueError(
                f"Invalid model_type: {config.model_type}. Expected 'ModelType.TEXTTRANSFORMER'."
            )

        # Confirm compatability of embedding dimension and number of heads
        if config.embedding_dim % config.num_heads != 0:
            raise ValueError(
                f"embedding_dim ({config.embedding_dim}) must be divisible by "
                f"num_heads ({config.num_heads})."
            )

        self.num_outputs = num_outputs
        self.config = config
        self.d_model = config.embedding_dim

        # ── Embedding layer ──
        self.embedding = nn.Embedding(
            num_embeddings=self.config.vocab_size,
            embedding_dim=self.config.embedding_dim,
            padding_idx=self.config.padding_idx,
        )
        if self.config.freeze_embeddings:
            self.embedding.weight.requires_grad = False

        # ── Positional encoding ──
        self.pos_encoder = PositionalEncoding(
            d_model=self.config.embedding_dim,
            max_len=self.config.max_seq_len,
            dropout=self.config.dropout[0] if self.config.dropout else 0.1,
        )

        # ── Transformer encoder stack ──
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.config.embedding_dim,
            nhead=self.config.num_heads,
            dim_feedforward=self.config.dim_feedforward,
            dropout=self.config.dropout[0] if self.config.dropout else 0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        # -- TransformerEncoder --
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=self.config.num_encoder_layers,
            norm=nn.LayerNorm(self.config.embedding_dim),
        )

        # ── Classifier head ──
        self.classifier_head = _construct_fc_layers(
            start_layer_size=self.config.embedding_dim,
            config=self.config,
            num_outputs=self.num_outputs,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: LongTensor of token IDs, shape (batch, seq_len).

        Returns:
            Logit tensor of shape (batch, num_outputs).
        """
        # ── Build padding mask ──
        # PyTorch convention: True = ignore this position
        padding_mask = x == self.config.padding_idx  # (batch, seq_len)

        # ── Embed + scale + positional encoding ──
        embedded = self.embedding(x) * math.sqrt(self.d_model)
        embedded = self.pos_encoder(embedded)

        # ── Transformer encoder ──
        # src_key_padding_mask has identical semantics to key_padding_mask from Week 11
        encoder_out = self.transformer_encoder(
            embedded, src_key_padding_mask=padding_mask
        )

        # ── Masked mean pooling ──
        encoder_out = encoder_out.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        lengths = (~padding_mask).sum(dim=1, keepdim=True).float().clamp(min=1)
        pooled = encoder_out.sum(dim=1) / lengths  # (batch, d_model)

        # ── Classifier head ──
        return self.classifier_head(pooled)

    def num_parameters(self) -> tuple:
        """Return (total_params, trainable_params)."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

    def get_architecture_config(self) -> dict:
        """Return a JSON-serializable description of the architecture."""
        from dataclasses import asdict

        return {
            "model_class": "TransformerClassifier",
            "num_outputs": self.num_outputs,
            "config": asdict(self.config),
        }

    def __str__(self) -> str:
        frozen = "frozen" if self.config.freeze_embeddings else "trainable"
        return (
            f"TransformerClassifier(vocab={self.config.vocab_size}, "
            f"d_model={self.config.embedding_dim} ({frozen}), "
            f"heads={self.config.num_heads}, layers={self.config.num_encoder_layers}, "
            f"d_ff={self.config.dim_feedforward}, out={self.num_outputs})"
        )

    def __repr__(self) -> str:
        return self.__str__()


def _construct_fc_layers(
    start_layer_size: int, config: ModelConfig, num_outputs: int
) -> nn.Sequential:
    """Build a fully-connected classifier head.

    Constructs a stack of Linear -> ReLU -> Dropout layers from config.hidden_units
    and config.dropout, followed by a final Linear(last_hidden, num_outputs) layer.

    Args:
        start_layer_size: Input feature dimension to the first linear layer.
        config: ModelConfig supplying hidden_units and dropout lists.
        num_outputs: Number of output logits for the final linear layer.

    Returns:
        nn.Sequential containing the complete classifier head.
    """
    layers = []
    prev_size = start_layer_size
    for i, hidden_size in enumerate(config.hidden_units):
        layers.append(nn.Linear(prev_size, hidden_size))
        layers.append(nn.ReLU())
        if i < len(config.dropout):
            layers.append(nn.Dropout(config.dropout[i]))
        prev_size = hidden_size
    layers.append(nn.Linear(prev_size, num_outputs))
    return nn.Sequential(*layers)


# ============================ EXPERIMENTAL ============================


class AttentionClassifierWithPE(nn.Module):
    """AttentionClassifier with sinusoidal positional encoding.

    Architecture:
        Token IDs -> Embedding -> PositionalEncoding -> MultiheadAttention ->
        Masked Mean Pool -> Classifier Head

    This is identical to the engine's AttentionClassifier except for the PE layer.

    Args:
        num_outputs (int): Number of output classes.
        config (ModelConfig): Must have model_type="text_attn".
    """

    def __init__(self, num_outputs, config):
        super().__init__()

        # Validate that config.model_type is "text_attn"
        # Raise ValueError if it's not
        if config.model_type != ModelType.TEXTATTN:
            raise ValueError(
                f"Invalid model_type: {config.model_type}. Expected 'text_attn'."
            )

        # Validate that embedding_dim is divisible by num_heads
        # This is required by nn.MultiheadAttention
        # Raise ValueError if the constraint is violated
        if config.embedding_dim % config.num_heads != 0:
            raise ValueError(
                "Invalid embedding dimension and number of heads. Expected config.embedding_dim % config.num_heads == 0"
            )

        # Store num_outputs and config as instance attributes
        self.num_outputs = num_outputs
        self.config = config

        # ── Embedding layer ──
        # Create the embedding layer
        # Use nn.Embedding with vocab_size, embedding_dim, and padding_idx from config
        self.embedding = nn.Embedding(
            num_embeddings=self.config.vocab_size,
            embedding_dim=self.config.embedding_dim,
            padding_idx=self.config.padding_idx,
        )

        # If config.freeze_embeddings is True, freeze the embedding weights
        if self.config.freeze_embeddings:
            self.embedding.weight.requires_grad = False

        # Positional Encoding layer
        self.pos_enc = PositionalEncoding(
            self.config.embedding_dim,
            self.config.max_seq_len,
            dropout=self.config.dropout[0] if self.config.dropout else 0.0,
        )

        # ── Self-attention layer ──
        # Create the self-attention layer using nn.MultiheadAttention
        # - embed_dim should be config.embedding_dim
        # - num_heads should be config.num_heads
        # - batch_first should be True (so input/output shape is (batch, seq_len, embed_dim))
        # - dropout should be config.dropout[0] if available, else 0.0
        self.attention = nn.MultiheadAttention(
            embed_dim=self.config.embedding_dim,
            num_heads=self.config.num_heads,
            dropout=self.config.dropout[0] if self.config.dropout else 0.0,
            batch_first=True,
        )

        # ── Classifier head: embedding_dim -> hidden_units -> num_outputs ──
        # =Create the classifier head using _construct_fc_layers
        # - start_layer_size should be config.embedding_dim (output of pooling)
        # - Pass config and num_outputs
        self.classifier_head = _construct_fc_layers(
            self.config.embedding_dim, self.config, num_outputs
        )

    def forward(self, x):
        """Forward pass with positional encoding."""
        # ── Step 1: Build the padding mask ──
        # True where x IS padding (PyTorch convention: True = ignore)
        padding_mask = x == self.config.padding_idx  # (batch, seq_len)

        # ── Step 2: Embed tokens ──
        embedded = self.embedding(x)

        # ── Step 3: Positional Encoding ──
        embedded_pos_enc = self.pos_enc(embedded)

        # ── Step 4: Self-attention ──
        # For self-attention, query = key = value = embedded
        # key_padding_mask tells attention to ignore padding positions
        attn_out, _ = self.attention(
            embedded_pos_enc,
            embedded_pos_enc,
            embedded_pos_enc,
            key_padding_mask=padding_mask,
        )  # attn_out: (batch, seq_len, embed_dim)

        # ── Step 5: Masked mean pooling (exclude padding) ──
        # Zero out padding positions so they don't contribute to the sum
        mask = (~padding_mask).unsqueeze(-1).float()  # (batch, seq_len, 1)
        attn_out = attn_out * mask

        # Count real (non-padding) tokens per sequence
        lengths = mask.sum(dim=1)  # (batch, 1)
        lengths = lengths.clamp(min=1)  # avoid division by zero

        # Sum over sequence dim and divide by real token count
        pooled = attn_out.sum(dim=1) / lengths  # (batch, embed_dim)

        # ── Step 5: Classifier head ──
        return self.classifier_head(pooled)

    def num_parameters(self):
        """Return (total, trainable) parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


class TimeSeriesAttentionClassifier(nn.Module):
    """Self-attention classifier for multivariate time series.

    Architecture:
        Linear(num_features -> embed_dim) -> MultiheadAttention -> Mean Pool -> FC Head

    Args:
        num_features (int): Number of input features per timestep.
        embed_dim (int): Dimension to project features into before attention.
        num_heads (int): Number of attention heads.
        num_outputs (int): Number of output classes.
        hidden_dim (int): Hidden layer size in the FC head.
        dropout (float): Dropout rate.
    """

    def __init__(
        self,
        num_features,
        embed_dim=64,
        num_heads=4,
        num_outputs=2,
        hidden_dim=32,
        dropout=0.2,
    ):
        super().__init__()

        self.num_outputs = num_outputs

        self.linear_proj = nn.Linear(
            in_features=num_features,
            out_features=embed_dim,
        )
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.classifier_head = nn.Sequential(
            nn.Linear(
                in_features=embed_dim,
                out_features=hidden_dim,
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_outputs),
        )

    def forward(self, x):
        """Forward pass.

        Args:
            x: FloatTensor of shape (batch, seq_len, num_features).

        Returns:
            Logit tensor of shape (batch, num_outputs).
        """

        x = self.linear_proj(x)
        x, _ = self.mha(x, x, x)
        x = x.mean(dim=1)

        return self.classifier_head(x)
