
from dataclasses import asdict
from typing import Any, Dict
import torch
import torch.nn as nn
from lantern.config import ModelConfig

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
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.config = config

        layers = []

        # Dynamically construct hidden layers with ReLU + optional dropout
        prev_dim = num_inputs
        for hidden_layer, dropout_rate in zip(config.hidden_units, config.dropout):
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
            x: Input tensor of shape (batch_size, num_inputs).

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
            "config": asdict(self.config)
        }

    def __str__(self) -> str:
        """Return a human-readable summary of the model configuration."""
        return f"MLP_Model(num_inputs={self.num_inputs}, num_outputs={self.num_outputs}, config={self.config!r})"

    def __repr__(self) -> str:
        """Return the same string as __str__ for consistent display."""
        return str(self)
