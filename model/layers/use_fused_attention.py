import torch

# use torch.scaled_dot_product_attention where possible
_HAS_FUSED_ATTN = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

_USE_FUSED_ATTN = 1  # 0 == off, 1 == on (for tested use)

def use_fused_attn(experimental: bool = False) -> bool:
    if not _HAS_FUSED_ATTN:
        return False
    return _USE_FUSED_ATTN > 0