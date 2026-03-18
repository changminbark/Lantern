"""
model.py

PyTorch model definitions for the lantern library.

Includes MLP, CNN (with optional residual blocks), Bag-of-Embeddings, TextCNN1D,
and SkipGram architectures, all configured via ModelConfig.

Course: CSCI 357 - AI and Neural Networks
Author: Chang Min Bark
"""

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

        if config.use_GAP:
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
            start_layer_size=self._flat_features, config=config, num_outputs=num_outputs
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
            num_embeddings=config.vocab_size,
            embedding_dim=config.embedding_dim,
            padding_idx=config.padding_idx,
        )

        # Provide the option to freeze pretrained embeddings (e.g., GloVe)
        if config.freeze_embeddings:
            self.embedding.requires_grad_(False)

        # One Conv1d branch per filter size; each branch independently scans the sequence
        # with a different n-gram width, capturing features at different granularities.
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=config.embedding_dim,
                    out_channels=config.num_filters,
                    kernel_size=fs,
                )
                for fs in config.filter_sizes
            ]
        )

        # Concatenated output of all branches feeds into the classifier head.
        self.classifier_head = _construct_fc_layers(
            start_layer_size=config.num_filters * len(config.filter_sizes),
            config=config,
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
            num_embeddings=config.vocab_size,
            embedding_dim=config.embedding_dim,
            padding_idx=config.padding_idx,
        )

        # If config.freeze_embeddings is True, prevent the embedding weights
        #       from receiving gradient updates by setting requires_grad = False.
        #       This is useful when you want to preserve pretrained GloVe knowledge
        #       and only train the classifier head (see Challenge 1).
        if config.freeze_embeddings:
            self.embedding.requires_grad_(False)

        # --- Classifier head: embedding_dim -> hidden_units -> num_outputs ---
        self.classifier_head = _construct_fc_layers(
            start_layer_size=config.embedding_dim,
            config=config,
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
        self.num_outputs = config.vocab_size
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.linear = nn.Linear(config.embedding_dim, config.vocab_size)

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
