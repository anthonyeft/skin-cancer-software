"""
Common modules for segmentation models.
Provides configurable building blocks for creating segmentation architectures.
"""
from typing import Union, Callable, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from inplace_abn import InPlaceABN
except ImportError:
    InPlaceABN = None


class Conv2dReLU(nn.Sequential):
    """
    Conv2d -> BatchNorm -> Activation block with configurable components.
    Supports regular BatchNorm, InPlaceABN, or no normalization.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, tuple],
        padding: Union[int, tuple, str] = 0,
        stride: Union[int, tuple] = 1,
        use_batchnorm: Union[bool, str] = True,
        activation: Union[str, Callable] = "relu",
        **activation_params
    ):
        # Create convolution layer (without bias if using batchnorm)
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not use_batchnorm,
        )

        # Create a list of layers
        layers = [conv]

        # Add normalization layer based on the specified option
        if use_batchnorm == "inplace":
            # InPlaceABN includes activation, so we'll configure it based on the requested activation
            activation_type = "leaky_relu" if activation == "relu" else "identity"
            layers.append(
                InPlaceABN(
                    out_channels, 
                    activation=activation_type,
                    activation_param=0.0,
                    **activation_params
                )
            )
        elif use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
            
            # Add activation as a separate layer
            if activation:
                layers.append(Activation(activation, **activation_params))
        else:
            # No batchnorm, just add activation
            if activation:
                layers.append(Activation(activation, **activation_params))

        super().__init__(*layers)


class Activation(nn.Module):
    """
    Activation layer wrapper supporting various activation functions.
    Creates the appropriate activation based on the name or callable provided.
    """
    def __init__(self, name: Union[str, Callable, None], **params):
        super().__init__()
        
        # Dictionary of activation functions
        activations: Dict[str, Callable] = {
            "identity": nn.Identity,
            "sigmoid": nn.Sigmoid,
            "softmax2d": lambda **p: nn.Softmax(dim=1, **p),
            "softmax": nn.Softmax,
            "logsoftmax": nn.LogSoftmax,
            "tanh": nn.Tanh,
            "relu": lambda **p: nn.ReLU(inplace=params.pop("inplace", True), **p),
            "relu6": nn.ReLU6,
            "leakyrelu": lambda **p: nn.LeakyReLU(inplace=params.pop("inplace", True), **p),
            "gelu": nn.GELU,
            "silu": nn.SiLU,  # AKA Swish
            "mish": nn.Mish,
            "elu": nn.ELU,
        }
        
        if name is None:
            self.activation = nn.Identity(**params)
        elif isinstance(name, str):
            name = name.lower()
            if name not in activations:
                raise ValueError(
                    f"Activation '{name}' not supported. Available options: "
                    f"{', '.join(activations.keys())}"
                )
            self.activation = activations[name](**params)
        elif callable(name):
            # Allow passing a custom activation function
            self.activation = name(**params)
        else:
            raise TypeError(
                f"Activation should be str, callable or None; got {type(name).__name__}"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x)