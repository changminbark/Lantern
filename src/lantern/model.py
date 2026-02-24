from dataclasses import asdict
from typing import Any, Dict

import torch
import torch.nn as nn

from lantern.config import ModelConfig, ModelType


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
        config: ModelConfig) -> None:
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
            conv_layers.append(nn.Conv2d(current_in_channels, conv_block.out_channels, conv_block.kernel_size, conv_block.stride, conv_block.padding))
            conv_layers.append(nn.BatchNorm2d(conv_block.out_channels))
            conv_layers.append(nn.ReLU())
            if conv_block.pool_size > 0:
                conv_layers.append(nn.MaxPool2d(conv_block.pool_size))
            current_in_channels = conv_block.out_channels
            
        self.feature_extractor = nn.Sequential(*conv_layers)

        # --- Compute flattened feature dimension via dummy forward pass ---
        # Create a dummy tensor of shape (1, in_channels, input_height, input_width),
        # pass it through self.feature_extractor, and store the total number of elements
        # as self._flat_features.
        with torch.no_grad():
            sample_tensor = torch.ones((1, self.config.in_channels, self.input_height, self.input_width))
            self._flat_features = self.feature_extractor(sample_tensor).numel()

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
        # Pass x through self.feature_extractor, flatten, then self.classifier_head
        x = self.feature_extractor(x)
        x = x.view(x.shape[0], -1)
        x = self.classifier_head(x)
        return x

    def num_parameters(self) -> tuple[int, int]:
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params, trainable_params

    def get_architecture_config(self) -> dict:
        return {
            'model_type': self.config.model_type,
            'input_height': self.input_height,
            'input_width': self.input_width,
            'num_outputs': self.num_outputs,
            'config': asdict(self.config),
        }

    def __str__(self) -> str:
        """Returns a string representation of the model.

        This provides a concise summary including input shape, number of input channels,
        the convolutional blocks specification, classifier head architecture, and total parameters.

        Returns:
            str: Human-readable summary of model architecture and size.
        """
        blocks_summary = "[" + ", ".join(
            f"({b.out_channels}, k={b.kernel_size}, s={b.stride}, p={b.padding})"
            for b in self.config.conv_blocks
        ) + "]"
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