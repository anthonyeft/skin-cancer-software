""" 'Fast' Normalization Functions

For GroupNorm and LayerNorm these functions bypass typical AMP upcast to float32.

Additionally, for LayerNorm, the APEX fused LN is used if available (which also does not upcast)

Hacked together by / Copyright 2022 Ross Wightman
"""
from typing import List, Optional

import torch
from torch.nn import functional as F


# fast (ie lower precision LN) can be disabled with this flag if issues crop up
_USE_FAST_NORM = False  # defaulting to False for now


def is_fast_norm():
    return _USE_FAST_NORM


def fast_group_norm(
    x: torch.Tensor,
    num_groups: int,
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-5
) -> torch.Tensor:
    if torch.jit.is_scripting():
        # currently cannot use is_autocast_enabled within torchscript
        return F.group_norm(x, num_groups, weight, bias, eps)

    if torch.is_autocast_enabled():
        # normally native AMP casts GN inputs to float32
        # here we use the low precision autocast dtype
        # FIXME what to do re CPU autocast?
        dt = torch.get_autocast_gpu_dtype()
        x, weight, bias = x.to(dt), weight.to(dt), bias.to(dt) if bias is not None else None

    with torch.cuda.amp.autocast(enabled=False):
        return F.group_norm(x, num_groups, weight, bias, eps)


def fast_layer_norm(
    x: torch.Tensor,
    normalized_shape: List[int],
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-5
) -> torch.Tensor:
    if torch.jit.is_scripting():
        # currently cannot use is_autocast_enabled within torchscript
        return F.layer_norm(x, normalized_shape, weight, bias, eps)

    if torch.is_autocast_enabled():
        # normally native AMP casts LN inputs to float32
        # apex LN does not, this is behaving like Apex
        dt = torch.get_autocast_gpu_dtype()
        # FIXME what to do re CPU autocast?
        x, weight, bias = x.to(dt), weight.to(dt), bias.to(dt) if bias is not None else None

    with torch.cuda.amp.autocast(enabled=False):
        return F.layer_norm(x, normalized_shape, weight, bias, eps)