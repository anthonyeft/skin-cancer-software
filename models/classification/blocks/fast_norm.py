"""
Optimized normalization implementations that work with mixed precision training.

These normalization functions maintain lower precision during normalization operations,
rather than defaulting to float32 as PyTorch's native implementation does.
This can improve performance and memory usage with minimal accuracy impact.
"""
from typing import List, Optional, Union

import torch
from torch.nn import functional as F


# Global flag to control usage of fast norm implementations
# Can be disabled if numerical stability issues are encountered
_USE_FAST_NORM = False


def is_fast_norm() -> bool:
    return _USE_FAST_NORM


def fast_group_norm(
    x: torch.Tensor,
    num_groups: int,
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-5
) -> torch.Tensor:
    """
    Group normalization that preserves autocast dtype instead of using float32.
    
    Args:
        x: Input tensor
        num_groups: Number of groups to separate the channels into
        weight: Scale parameter (gamma)
        bias: Shift parameter (beta)
        eps: Small constant for numerical stability
        
    Returns:
        Normalized tensor with same dtype as input when autocast is enabled
    """
    # Handle TorchScript mode - use standard implementation
    if torch.jit.is_scripting():
        # TorchScript can't use is_autocast_enabled, so fall back to standard GN
        return F.group_norm(x, num_groups, weight, bias, eps)

    # If autocast is active, maintain the current precision instead of using float32
    if torch.is_autocast_enabled():
        # Get current autocast dtype (could be float16 or bfloat16)
        dt = torch.get_autocast_gpu_dtype()
        
        # Convert inputs to the current autocast dtype
        x = x.to(dt)
        if weight is not None:
            weight = weight.to(dt)
        if bias is not None:
            bias = bias.to(dt)

    # Temporarily disable autocast to prevent nested autocast context
    # This allows us to control the precision directly rather than using PyTorch's defaults
    with torch.cuda.amp.autocast(enabled=False):
        return F.group_norm(x, num_groups, weight, bias, eps)


def fast_layer_norm(
    x: torch.Tensor,
    normalized_shape: Union[int, List[int], torch.Size],
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-5
) -> torch.Tensor:
    """
    Layer normalization that preserves autocast dtype instead of using float32.
    
    Args:
        x: Input tensor
        normalized_shape: Shape of the normalized dimensions (excluding batch)
        weight: Scale parameter (gamma)
        bias: Shift parameter (beta)
        eps: Small constant for numerical stability
        
    Returns:
        Normalized tensor with same dtype as input when autocast is enabled
    """
    # Handle TorchScript mode - use standard implementation
    if torch.jit.is_scripting():
        # TorchScript can't use is_autocast_enabled, so fall back to standard LN
        return F.layer_norm(x, normalized_shape, weight, bias, eps)

    # When autocast is active, maintain the current precision instead of using float32
    if torch.is_autocast_enabled():
        # Get current autocast dtype
        dt = torch.get_autocast_gpu_dtype()
        
        # Convert inputs to the current autocast dtype
        x = x.to(dt)
        if weight is not None:
            weight = weight.to(dt)
        if bias is not None:
            bias = bias.to(dt)

    # Temporarily disable autocast to prevent nested autocast context
    # This allows us to control the precision directly
    with torch.cuda.amp.autocast(enabled=False):
        return F.layer_norm(x, normalized_shape, weight, bias, eps)