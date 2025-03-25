from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .format import get_spatial_dim, get_channel_dim

_int_tuple_2_t = Union[int, Tuple[int, int]]


def adaptive_pool_feat_mult(pool_type='avg'):
    """Calculate feature multiplier for a given pooling type.
    
    Args:
        pool_type (str): Type of pooling operation ('avg', 'max', 'avgmax', 'catavgmax')
        
    Returns:
        int: Multiplier for the number of output features (2 for concatenated pools, 1 otherwise)
    """
    if pool_type.endswith('catavgmax'):
        return 2
    else:
        return 1


def adaptive_avgmax_pool2d(x, output_size: _int_tuple_2_t = 1):
    """Average-max pooling operation.
    
    Performs both average and max pooling and returns the average of the results.
    
    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)
        output_size (int or tuple): Size of the output tensor's spatial dimensions
        
    Returns:
        torch.Tensor: Pooled tensor
    """
    x_avg = F.adaptive_avg_pool2d(x, output_size)
    x_max = F.adaptive_max_pool2d(x, output_size)
    return 0.5 * (x_avg + x_max)


def adaptive_catavgmax_pool2d(x, output_size: _int_tuple_2_t = 1):
    """Concatenated average-max pooling operation.
    
    Performs both average and max pooling and concatenates the results along the channel dimension.
    
    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)
        output_size (int or tuple): Size of the output tensor's spatial dimensions
        
    Returns:
        torch.Tensor: Pooled tensor with doubled channel dimension
    """
    x_avg = F.adaptive_avg_pool2d(x, output_size)
    x_max = F.adaptive_max_pool2d(x, output_size)
    return torch.cat((x_avg, x_max), 1)


def select_adaptive_pool2d(x, pool_type='avg', output_size: _int_tuple_2_t = 1):
    """Selectable global pooling function with dynamic input kernel size.
    
    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)
        pool_type (str): Pooling operation type ('avg', 'max', 'avgmax', 'catavgmax')
        output_size (int or tuple): Size of the output tensor's spatial dimensions
        
    Returns:
        torch.Tensor: Pooled tensor
        
    Raises:
        AssertionError: If an invalid pool_type is provided
    """
    if pool_type == 'avg':
        x = F.adaptive_avg_pool2d(x, output_size)
    elif pool_type == 'avgmax':
        x = adaptive_avgmax_pool2d(x, output_size)
    elif pool_type == 'catavgmax':
        x = adaptive_catavgmax_pool2d(x, output_size)
    elif pool_type == 'max':
        x = F.adaptive_max_pool2d(x, output_size)
    else:
        assert False, f'Invalid pool type: {pool_type}'
    return x


class FastAdaptiveAvgPool(nn.Module):
    """Fast implementation of adaptive average pooling.
    
    Uses torch.mean() directly on the spatial dimensions for efficient pooling
    when output size is 1x1.
    
    Args:
        flatten (bool): If True, flatten the output tensor. Default: False
        input_fmt (str): Input tensor format ('NCHW', 'NHWC'). Default: 'NCHW'
        
    Shape:
        - Input: (batch_size, channels, height, width) or (batch_size, height, width, channels)
        - Output: (batch_size, channels) if flatten=True, else (batch_size, channels, 1, 1) or equivalent
    """
    def __init__(self, flatten: bool = False, input_fmt: str = 'NCHW'):
        super(FastAdaptiveAvgPool, self).__init__()
        self.flatten = flatten
        self.dim = get_spatial_dim(input_fmt)

    def forward(self, x):
        return x.mean(self.dim, keepdim=not self.flatten)


class FastAdaptiveMaxPool(nn.Module):
    """Fast implementation of adaptive max pooling.
    
    Uses torch.amax() directly on the spatial dimensions for efficient pooling
    when output size is 1x1.
    
    Args:
        flatten (bool): If True, flatten the output tensor. Default: False
        input_fmt (str): Input tensor format ('NCHW', 'NHWC'). Default: 'NCHW'
        
    Shape:
        - Input: (batch_size, channels, height, width) or (batch_size, height, width, channels)
        - Output: (batch_size, channels) if flatten=True, else (batch_size, channels, 1, 1) or equivalent
    """
    def __init__(self, flatten: bool = False, input_fmt: str = 'NCHW'):
        super(FastAdaptiveMaxPool, self).__init__()
        self.flatten = flatten
        self.dim = get_spatial_dim(input_fmt)

    def forward(self, x):
        return x.amax(self.dim, keepdim=not self.flatten)


class FastAdaptiveAvgMaxPool(nn.Module):
    """Fast implementation of combined average-max pooling.
    
    Performs both average and max pooling and returns the average of the results.
    Optimized for 1x1 output size.
    
    Args:
        flatten (bool): If True, flatten the output tensor. Default: False
        input_fmt (str): Input tensor format ('NCHW', 'NHWC'). Default: 'NCHW'
        
    Shape:
        - Input: (batch_size, channels, height, width) or (batch_size, height, width, channels)
        - Output: (batch_size, channels) if flatten=True, else (batch_size, channels, 1, 1) or equivalent
    """
    def __init__(self, flatten: bool = False, input_fmt: str = 'NCHW'):
        super(FastAdaptiveAvgMaxPool, self).__init__()
        self.flatten = flatten
        self.dim = get_spatial_dim(input_fmt)

    def forward(self, x):
        x_avg = x.mean(self.dim, keepdim=not self.flatten)
        x_max = x.amax(self.dim, keepdim=not self.flatten)
        return 0.5 * x_avg + 0.5 * x_max


class FastAdaptiveCatAvgMaxPool(nn.Module):
    """Fast implementation of concatenated average-max pooling.
    
    Performs both average and max pooling and concatenates the results
    along the channel dimension. Optimized for 1x1 output size.
    
    Args:
        flatten (bool): If True, flatten the output tensor. Default: False
        input_fmt (str): Input tensor format ('NCHW', 'NHWC'). Default: 'NCHW'
        
    Shape:
        - Input: (batch_size, channels, height, width) or (batch_size, height, width, channels)
        - Output: (batch_size, channels*2) if flatten=True, 
                 else (batch_size, channels*2, 1, 1) or equivalent
    """
    def __init__(self, flatten: bool = False, input_fmt: str = 'NCHW'):
        super(FastAdaptiveCatAvgMaxPool, self).__init__()
        self.flatten = flatten
        self.dim_reduce = get_spatial_dim(input_fmt)
        if flatten:
            self.dim_cat = 1
        else:
            self.dim_cat = get_channel_dim(input_fmt)

    def forward(self, x):
        x_avg = x.mean(self.dim_reduce, keepdim=not self.flatten)
        x_max = x.amax(self.dim_reduce, keepdim=not self.flatten)
        return torch.cat((x_avg, x_max), self.dim_cat)


class AdaptiveAvgMaxPool2d(nn.Module):
    """Combined average-max pooling layer.
    
    Performs both average and max pooling and returns the average of the results.
    
    Args:
        output_size (int or tuple): Size of the output tensor's spatial dimensions
        
    Shape:
        - Input: (batch_size, channels, height, width)
        - Output: (batch_size, channels, output_size, output_size)
    """
    def __init__(self, output_size: _int_tuple_2_t = 1):
        super(AdaptiveAvgMaxPool2d, self).__init__()
        self.output_size = output_size

    def forward(self, x):
        return adaptive_avgmax_pool2d(x, self.output_size)


class AdaptiveCatAvgMaxPool2d(nn.Module):
    """Concatenated average-max pooling layer.
    
    Performs both average and max pooling and concatenates the results
    along the channel dimension.
    
    Args:
        output_size (int or tuple): Size of the output tensor's spatial dimensions
        
    Shape:
        - Input: (batch_size, channels, height, width)
        - Output: (batch_size, channels*2, output_size, output_size)
    """
    def __init__(self, output_size: _int_tuple_2_t = 1):
        super(AdaptiveCatAvgMaxPool2d, self).__init__()
        self.output_size = output_size

    def forward(self, x):
        return adaptive_catavgmax_pool2d(x, self.output_size)


class SelectAdaptivePool2d(nn.Module):
    """Selectable global pooling layer with dynamic input kernel size.
    
    This module provides a unified interface for different pooling strategies,
    supporting both standard PyTorch pooling and optimized pooling implementations.
    
    Args:
        output_size (int or tuple): Size of the output tensor's spatial dimensions. Default: 1
        pool_type (str): Pooling operation type. Options include:
            - '' or None: Identity mapping (no pooling)
            - 'avg', 'max', 'avgmax', 'catavgmax': Standard pooling operations
            - 'fast*': Optimized versions for 1x1 output size
        flatten (bool): If True, flatten the output tensor. Default: False
        input_fmt (str): Input tensor format ('NCHW', 'NHWC'). Default: 'NCHW'
        
    Shape:
        - Input: (batch_size, channels, height, width) or format specified by input_fmt
        - Output: Depends on pool_type and flatten parameters
    """
    def __init__(
            self,
            output_size: _int_tuple_2_t = 1,
            pool_type: str = 'fast',
            flatten: bool = False,
            input_fmt: str = 'NCHW',
    ):
        super(SelectAdaptivePool2d, self).__init__()
        assert input_fmt in ('NCHW', 'NHWC'), f"Invalid input format: {input_fmt}"
        self.pool_type = pool_type or ''  # convert other falsy values to empty string for consistent TS typing
        
        if not pool_type:
            # Identity case (no pooling)
            self.pool = nn.Identity()
            self.flatten = nn.Flatten(1) if flatten else nn.Identity()
        elif pool_type.startswith('fast') or input_fmt != 'NCHW':
            assert output_size == 1, 'Fast pooling and non NCHW input formats require output_size == 1.'
            
            # Fast pooling implementations
            if pool_type.endswith('catavgmax'):
                self.pool = FastAdaptiveCatAvgMaxPool(flatten, input_fmt=input_fmt)
            elif pool_type.endswith('avgmax'):
                self.pool = FastAdaptiveAvgMaxPool(flatten, input_fmt=input_fmt)
            elif pool_type.endswith('max'):
                self.pool = FastAdaptiveMaxPool(flatten, input_fmt=input_fmt)
            else:
                self.pool = FastAdaptiveAvgPool(flatten, input_fmt=input_fmt)
            self.flatten = nn.Identity()
        else:
            # Standard PyTorch pooling implementations
            assert input_fmt == 'NCHW', "Standard pooling only supports NCHW format"
            
            if pool_type == 'avgmax':
                self.pool = AdaptiveAvgMaxPool2d(output_size)
            elif pool_type == 'catavgmax':
                self.pool = AdaptiveCatAvgMaxPool2d(output_size)
            elif pool_type == 'max':
                self.pool = nn.AdaptiveMaxPool2d(output_size)
            else:
                self.pool = nn.AdaptiveAvgPool2d(output_size)
            self.flatten = nn.Flatten(1) if flatten else nn.Identity()

    def is_identity(self):
        """Check if this module is an identity mapping (no pooling).
        
        Returns:
            bool: True if no pooling is applied, False otherwise
        """
        return not self.pool_type

    def forward(self, x):
        x = self.pool(x)
        x = self.flatten(x)
        return x

    def feat_mult(self):
        """Get the feature multiplier for this pooling type.
        
        Returns:
            int: Multiplier for the number of output features (2 for concatenated pools, 1 otherwise)
        """
        return adaptive_pool_feat_mult(self.pool_type)

    def __repr__(self):
        return f"{self.__class__.__name__}(pool_type='{self.pool_type}', flatten={self.flatten})"