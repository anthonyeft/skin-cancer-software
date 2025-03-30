"""
MixVisionTransformer (MiT) implementation for segmentation tasks.

This module implements the MixVisionTransformer architecture which combines 
hierarchical Transformers with overlapping patch embeddings for efficient 
semantic segmentation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from typing import List, Tuple, Optional, Union, Dict, Any

from models.classification.blocks.drop import DropPath


class DWConv(nn.Module):
    """Depthwise Convolution module used in MLP blocks."""
    
    def __init__(self, dim: int = 768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=True)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, H, W)
        x = self.dwconv(x)
        return x.flatten(2).transpose(1, 2)


class Mlp(nn.Module):
    """
    MLP with depthwise convolution for spatial mixing.
    """
    def __init__(
        self, 
        in_features: int, 
        hidden_features: Optional[int] = None, 
        out_features: Optional[int] = None, 
        act_layer: nn.Module = nn.GELU, 
        drop: float = 0.0
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    """
    Multi-head self-attention with spatial reduction option.
    """
    def __init__(
        self, 
        dim: int, 
        num_heads: int = 8, 
        qkv_bias: bool = False, 
        attn_drop: float = 0.0, 
        proj_drop: float = 0.0, 
        sr_ratio: int = 1
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        # Use PyTorch's optimized attention implementation
        x = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=None,
            dropout_p=self.attn_drop.p if self.training else 0.0,
            is_causal=False
        )
        
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):
    """
    Transformer block with self-attention and MLP.
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
        sr_ratio: int = 1,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim, 
            hidden_features=mlp_hidden_dim, 
            act_layer=act_layer, 
            drop=drop
        )

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


class OverlapPatchEmbed(nn.Module):
    """
    Image to Patch Embedding with overlapping patches.
    """
    def __init__(
        self, 
        img_size: int = 224, 
        patch_size: int = 7, 
        stride: int = 4, 
        in_chans: int = 3, 
        embed_dim: int = 768
    ):
        super().__init__()
        img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // stride, img_size[1] // stride
        self.num_patches = self.H * self.W
        
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=(patch_size[0] // 2, patch_size[1] // 2),
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


class MixVisionTransformer(nn.Module):
    """
    MixVisionTransformer (MiT) backbone for segmentation.
    
    Hierarchical Transformer with four stages of different resolution features.
    """
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        num_classes: int = 1000,
        embed_dims: List[int] = [64, 128, 256, 512],
        num_heads: List[int] = [1, 2, 4, 8],
        mlp_ratios: List[float] = [4, 4, 4, 4],
        qkv_bias: bool = False,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        depths: List[int] = [3, 4, 6, 3],
        sr_ratios: List[int] = [8, 4, 2, 1],
    ):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths

        # Patch embeddings
        self.patch_embed1 = OverlapPatchEmbed(
            img_size=img_size, patch_size=7, stride=4, in_chans=in_chans, embed_dim=embed_dims[0]
        )
        self.patch_embed2 = OverlapPatchEmbed(
            img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0], embed_dim=embed_dims[1]
        )
        self.patch_embed3 = OverlapPatchEmbed(
            img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1], embed_dim=embed_dims[2]
        )
        self.patch_embed4 = OverlapPatchEmbed(
            img_size=img_size // 16, patch_size=3, stride=2, in_chans=embed_dims[2], embed_dim=embed_dims[3]
        )

        # Create drop path rate schedule
        self.dpr = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        
        # Create transformer blocks for each stage
        self._create_stage_blocks(
            embed_dims, num_heads, mlp_ratios, qkv_bias, drop_rate, attn_drop_rate, norm_layer, sr_ratios
        )

    def _create_stage_blocks(
        self, 
        embed_dims: List[int], 
        num_heads: List[int], 
        mlp_ratios: List[float], 
        qkv_bias: bool, 
        drop_rate: float, 
        attn_drop_rate: float, 
        norm_layer: nn.Module, 
        sr_ratios: List[int]
    ):
        """Create transformer blocks for all stages."""
        # Stage 1
        cur = 0
        self.block1 = nn.ModuleList([
            Block(
                dim=embed_dims[0],
                num_heads=num_heads[0],
                mlp_ratio=mlp_ratios[0],
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=self.dpr[cur + i],
                norm_layer=norm_layer,
                sr_ratio=sr_ratios[0],
            )
            for i in range(self.depths[0])
        ])
        self.norm1 = norm_layer(embed_dims[0])

        # Stage 2
        cur += self.depths[0]
        self.block2 = nn.ModuleList([
            Block(
                dim=embed_dims[1],
                num_heads=num_heads[1],
                mlp_ratio=mlp_ratios[1],
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=self.dpr[cur + i],
                norm_layer=norm_layer,
                sr_ratio=sr_ratios[1],
            )
            for i in range(self.depths[1])
        ])
        self.norm2 = norm_layer(embed_dims[1])

        # Stage 3
        cur += self.depths[1]
        self.block3 = nn.ModuleList([
            Block(
                dim=embed_dims[2],
                num_heads=num_heads[2],
                mlp_ratio=mlp_ratios[2],
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=self.dpr[cur + i],
                norm_layer=norm_layer,
                sr_ratio=sr_ratios[2],
            )
            for i in range(self.depths[2])
        ])
        self.norm3 = norm_layer(embed_dims[2])

        # Stage 4
        cur += self.depths[2]
        self.block4 = nn.ModuleList([
            Block(
                dim=embed_dims[3],
                num_heads=num_heads[3],
                mlp_ratio=mlp_ratios[3],
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=self.dpr[cur + i],
                norm_layer=norm_layer,
                sr_ratio=sr_ratios[3],
            )
            for i in range(self.depths[3])
        ])
        self.norm4 = norm_layer(embed_dims[3])

    def reset_drop_path(self, drop_path_rate: float):
        """Reset drop path probability for all blocks."""
        dpr = torch.linspace(0, drop_path_rate, sum(self.depths)).tolist()
        
        cur = 0
        for i in range(self.depths[0]):
            self.block1[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[0]
        for i in range(self.depths[1]):
            self.block2[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[1]
        for i in range(self.depths[2]):
            self.block3[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[2]
        for i in range(self.depths[3]):
            self.block4[i].drop_path.drop_prob = dpr[cur + i]

    def _process_stage(self, x: torch.Tensor, patch_embed: nn.Module, blocks: nn.ModuleList, norm: nn.Module) -> torch.Tensor:
        """Process one stage of the network."""
        B = x.shape[0]
        x, H, W = patch_embed(x)
        
        for blk in blocks:
            x = blk(x, H, W)
            
        x = norm(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        return x

    def forward_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract hierarchical features from input image."""
        outs = []

        # Stage 1
        x = self._process_stage(x, self.patch_embed1, self.block1, self.norm1)
        outs.append(x)

        # Stage 2
        x = self._process_stage(x, self.patch_embed2, self.block2, self.norm2)
        outs.append(x)

        # Stage 3
        x = self._process_stage(x, self.patch_embed3, self.block3, self.norm3)
        outs.append(x)

        # Stage 4
        x = self._process_stage(x, self.patch_embed4, self.block4, self.norm4)
        outs.append(x)

        return outs

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass returning hierarchical features."""
        return self.forward_features(x)


class MixVisionTransformerEncoder(MixVisionTransformer):
    """
    MixVisionTransformer encoder for segmentation tasks.
    
    Adapts MixVisionTransformer to return features in the format expected by 
    segmentation decoders.
    """
    def __init__(self, out_channels: Tuple[int, ...], depth: int = 5, **kwargs):
        super().__init__(**kwargs)
        self.out_channels = out_channels
        self._depth = depth
        self._in_channels = 3

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass returning a list of tensors with the input and feature maps.
        
        Returns:
            List with [input, dummy tensor for stride-2 features, stride-4 features, ...].
        """
        # Create placeholder for stride-2 features (not computed by this encoder)
        B, _, H, W = x.shape
        dummy = torch.empty([B, 0, H // 2, W // 2], dtype=x.dtype, device=x.device)

        # Return input, dummy placeholder, and computed feature maps
        return [x, dummy] + self.forward_features(x)[: self._depth - 1]

    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True) -> nn.Module:
        """Load state dict, removing classification head if present."""
        # Remove classification head parameters if present
        state_dict.pop("head.weight", None)
        state_dict.pop("head.bias", None)
        return super().load_state_dict(state_dict, strict)


def get_encoder(
    name: str, 
    in_channels: int = 3, 
    depth: int = 5, 
    weights: Optional[Dict[str, torch.Tensor]] = None, 
    output_stride: int = 32, 
    **kwargs
) -> MixVisionTransformerEncoder:
    """
    Create a MixVisionTransformer encoder with the specified parameters.
    
    Args:
        name: Model name identifier
        in_channels: Number of input channels
        depth: Depth of encoder (number of returned feature maps)
        weights: Optional pre-trained weights
        output_stride: Output stride of the encoder (not used, included for API compatibility)
        
    Returns:
        Configured MixVisionTransformer encoder
    """
    # Default parameters for the encoder
    params = {
        "out_channels": (3, 0, 64, 128, 320, 512),
        "patch_size": 4,
        "embed_dims": [64, 128, 320, 512],
        "num_heads": [1, 2, 5, 8],
        "mlp_ratios": [4, 4, 4, 4],
        "qkv_bias": True,
        "norm_layer": partial(nn.LayerNorm, eps=1e-6),
        "depths": [3, 8, 27, 3],
        "sr_ratios": [8, 4, 2, 1],
        "drop_rate": 0.0,
        "drop_path_rate": 0.1,
        "depth": depth,
    }
    
    # Update with any custom parameters
    params.update(kwargs)
    
    # Create encoder
    encoder = MixVisionTransformerEncoder(**params)
    
    # Load weights if provided
    if weights is not None:
        encoder.load_state_dict(weights)
        
    return encoder