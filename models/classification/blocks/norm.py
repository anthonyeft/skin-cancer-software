import torch
import torch.nn as nn
import torch.nn.functional as F

from .fast_norm import is_fast_norm, fast_layer_norm


class LayerNorm(nn.LayerNorm):
    """LayerNorm with optimized implementation option.
    
    Layer normalization module with support for an optimized implementation
    through the fast_norm option.
    
    Args:
        num_channels (int): Number of features/channels to normalize over
        eps (float): Small constant for numerical stability. Default: 1e-6
        affine (bool): If True, applies learnable affine parameters. Default: True
        
    Shape:
        - Input: (batch_size, num_channels, ...)
        - Output: Same shape as input
    """
    def __init__(self, num_channels, eps=1e-6, affine=True):
        super().__init__(num_channels, eps=eps, elementwise_affine=affine)
        self._fast_norm = is_fast_norm()  # Required flag for scripting (can't use globals)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._fast_norm:
            x = fast_layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        else:
            x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x


class LayerNorm2d(nn.LayerNorm):
    """LayerNorm for channels of 2D spatial tensors.
    
    This implementation handles NCHW format tensors by internally permuting dimensions
    to normalize over the channel dimension.
    
    Args:
        num_channels (int): Number of channels (C) in the input tensor
        eps (float): Small constant for numerical stability. Default: 1e-6
        affine (bool): If True, applies learnable affine parameters. Default: True
        
    Shape:
        - Input: (batch_size, channels, height, width)
        - Output: Same shape as input
    """
    def __init__(self, num_channels, eps=1e-6, affine=True):
        super().__init__(num_channels, eps=eps, elementwise_affine=affine)
        self._fast_norm = is_fast_norm()  # Required flag for scripting (can't use globals)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Permute to (batch_size, height, width, channels)
        x = x.permute(0, 2, 3, 1)
        if self._fast_norm:
            x = fast_layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        else:
            x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        # Permute back to (batch_size, channels, height, width)
        x = x.permute(0, 3, 1, 2)
        return x