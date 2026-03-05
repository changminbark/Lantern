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
            raise ValueError(f"Invalid model_type: {config.model_type}. Expected 'ModelType.MLP'.")

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.config = config

        layers = [nn.Flatten()]

        # Dynamically construct hidden layers with ReLU + optional dropout
        prev_dim = num_inputs
        for hidden_layer, dropout_rate in zip(self.config.hidden_units, self.config.dropout):
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

    def __init__(self,
            input_height: int,
            input_width: int,
            num_outputs: int,
            config: ModelConfig
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
            raise ValueError(f"Invalid model_type: {config.model_type}. Expected 'ModelType.CNN'.")

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
                        conv_block.padding
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
                sample_tensor = torch.ones((1, self.config.in_channels, self.input_height, self.input_width))
                self._flat_features = self.feature_extractor(sample_tensor).numel()
                self.feature_extractor.train()

        # --- Build the classifier head ---
        # Build self.classifier_head as an nn.Sequential.
        # First layer: nn.Linear(self._flat_features, config.hidden_units[0])
        # Then for each subsequent hidden unit / dropout pair: Linear -> ReLU -> Dropout
        # Final layer: nn.Linear(last_hidden, num_outputs)
        # (Follow the same pattern as MLP_Model for the linear layers.)
        classifier_layers = []
        prev_dim = self._flat_features
        for hidden_layer, dropout_rate in zip(self.config.hidden_units, self.config.dropout):
            classifier_layers.append(nn.Linear(prev_dim, hidden_layer))
            classifier_layers.append(nn.ReLU())
            if dropout_rate > 0:
                classifier_layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_layer
        # Final projection to output logits (no activation)
        classifier_layers.append(nn.Linear(prev_dim, num_outputs))

        self.classifier_head = nn.Sequential(*classifier_layers)

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
            x = self.gap(x)                 # (batch, channels, 1, 1)
            x = x.squeeze(-1).squeeze(-1)   # (batch, channels)
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
            config_dict['conv_blocks'] = [
                {'block_type': 'residual', **asdict(b)} if isinstance(b, ResidualBlockConfig)
                else {'block_type': 'conv', **asdict(b)}
                for b in self.config.conv_blocks
            ]
            return config_dict
        
        return {
            'model_type': self.config.model_type,
            'input_height': self.input_height,
            'input_width': self.input_width,
            'num_outputs': self.num_outputs,
            'config': _serialize_config(),
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
        blocks_summary = "[" + ", ".join(block_str(b) for b in self.config.conv_blocks) + "]"
        hidden_str = ", ".join(str(u) for u in self.config.hidden_units)
        head_summary = f"[{self._flat_features} -> [{hidden_str}] -> {self.num_outputs}]"
        dropout_summary = "[" + ", ".join(str(d) for d in self.config.dropout) + "]"
        return (
            f"CNN_Model(input={self.input_height}x{self.input_width}, in_channels={self.config.in_channels})\n"
            f"- blocks={blocks_summary}\n"
            f"- head={head_summary}\n"
            f"- dropout={dropout_summary}\n"
        )

    def __repr__(self) -> str:
        return self.__str__()