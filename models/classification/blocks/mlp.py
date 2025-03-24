"""
MLP module with dropout and configurable activation layer.
Commonly used in Vision Transformer, MLP-Mixer and related networks.
"""
import collections.abc
from functools import partial
from itertools import repeat

from torch import nn


def _ntuple(n):
    """
    Convert a single value to an n-tuple, or keep iterable inputs as-is.
    Borrowed from PyTorch internals.
    """
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse


# Helper function to convert values to pairs (2-tuples)
to_2tuple = _ntuple(2)


class Mlp(nn.Module):
    """
    Multi-Layer Perceptron (MLP) module with configurable features.
    
    Supports:
    - Linear or Conv1x1 layers
    - Customizable activation
    - Optional normalization
    - Dropout on both layers
    """
    
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.,
            use_conv=False,
    ):
        """
        Args:
            in_features: Input dimension size
            hidden_features: Hidden dimension size (defaults to in_features)
            out_features: Output dimension size (defaults to in_features)
            act_layer: Activation function to use
            norm_layer: Optional normalization layer
            bias: Whether to use bias in linear/conv layers (can be a tuple for each layer)
            drop: Dropout probability (can be a tuple for each layer)
            use_conv: If True, use 1x1 convs instead of linear layers
        """
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        # Convert bias and dropout to pairs for first and second layers
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        
        # Select layer type based on use_conv flag
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        
        # Optional normalization
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        """Forward pass through MLP."""
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        
        x = self.fc2(x)
        x = self.drop2(x)
        
        return x