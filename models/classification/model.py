import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from collections import OrderedDict

from torch.nn.init import trunc_normal_

from .blocks.drop import DropPath
from .blocks.selectpool import SelectAdaptivePool2d
from .blocks.norm import LayerNorm, LayerNorm2d
from .blocks.mlp import Mlp


class StarReLU(nn.Module):
    """StarReLU: s * relu(x) ** 2 + b"""
    def __init__(self, scale_value=1.0, bias_value=0.0, scale_learnable=True, bias_learnable=True, inplace=False):
        super().__init__()
        self.relu = nn.ReLU(inplace=inplace)
        self.scale = nn.Parameter(scale_value * torch.ones(1), requires_grad=scale_learnable)
        self.bias = nn.Parameter(bias_value * torch.ones(1), requires_grad=bias_learnable)

    def forward(self, x):
        return self.scale * self.relu(x) ** 2 + self.bias


class SquaredReLU(nn.Module):
    """Squared ReLU: https://arxiv.org/abs/2109.08668"""
    def __init__(self, inplace=False):
        super().__init__()
        self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        return torch.square(self.relu(x))


class Scale(nn.Module):
    """Scale vector by element multiplications."""
    def __init__(self, dim, init_value=1.0, trainable=True, use_nchw=True):
        super().__init__()
        self.shape = (dim, 1, 1) if use_nchw else (dim,)
        self.scale = nn.Parameter(init_value * torch.ones(dim), requires_grad=trainable)

    def forward(self, x):
        return x * self.scale.view(self.shape)


class Attention(nn.Module):
    """Vanilla self-attention module."""
    def __init__(self, dim, head_dim=32, num_heads=None, qkv_bias=False, 
                 attn_drop=0.0, proj_drop=0.0, proj_bias=False, **kwargs):
        super().__init__()
        
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        
        self.num_heads = num_heads if num_heads else dim // head_dim
        if self.num_heads == 0:
            self.num_heads = 1
            
        self.attention_dim = self.num_heads * self.head_dim
        
        self.qkv = nn.Linear(dim, self.attention_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.attention_dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        
        qkv = (self.qkv(x)
               .reshape(B, N, 3, self.num_heads, self.head_dim)
               .permute(2, 0, 3, 1, 4))
        q, k, v = qkv.unbind(0)
        
        x = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0)
        
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class SepConv(nn.Module):
    """Inverted separable convolution from MobileNetV2."""
    def __init__(self, dim, expansion_ratio=2, act1_layer=StarReLU, act2_layer=nn.Identity,
                 bias=False, kernel_size=7, padding=3, **kwargs):
        super().__init__()
        mid_channels = int(expansion_ratio * dim)
        self.pwconv1 = nn.Conv2d(dim, mid_channels, kernel_size=1, bias=bias)
        self.act1 = act1_layer()
        self.dwconv = nn.Conv2d(mid_channels, mid_channels, kernel_size=kernel_size,
                               padding=padding, groups=mid_channels, bias=bias)
        self.act2 = act2_layer()
        self.pwconv2 = nn.Conv2d(mid_channels, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.pwconv1(x)
        x = self.act1(x)
        x = self.dwconv(x)
        x = self.act2(x)
        x = self.pwconv2(x)
        return x


class GroupNorm1NoBias(nn.GroupNorm):
    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)
        self.eps = kwargs.get('eps', 1e-6)
        self.bias = None


class LayerNorm2dNoBias(LayerNorm2d):
    def __init__(self, num_channels, **kwargs):
        super().__init__(num_channels, **kwargs)
        self.eps = kwargs.get('eps', 1e-6)
        self.bias = None


class LayerNormNoBias(nn.LayerNorm):
    def __init__(self, num_channels, **kwargs):
        super().__init__(num_channels, **kwargs)
        self.eps = kwargs.get('eps', 1e-6)
        self.bias = None


class Stem(nn.Module):
    """Stem implemented by a layer of convolution."""
    def __init__(self, in_channels, out_channels, norm_layer=None):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=4, padding=2)
        self.norm = norm_layer(out_channels) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return x


class Downsampling(nn.Module):
    """Downsampling implemented by a layer of convolution."""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, norm_layer=None):
        super().__init__()
        self.norm = norm_layer(in_channels) if norm_layer else nn.Identity()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        x = self.norm(x)
        x = self.conv(x)
        return x


class MlpHead(nn.Module):
    """MLP classification head."""
    def __init__(self, dim, num_classes=7, mlp_ratio=4, act_layer=SquaredReLU, 
                 norm_layer=LayerNorm, drop_rate=0., bias=True):
        super().__init__()
        hidden_features = int(mlp_ratio * dim)
        self.fc1 = nn.Linear(dim, hidden_features, bias=bias)
        self.act = act_layer()
        self.norm = norm_layer(hidden_features)
        self.fc2 = nn.Linear(hidden_features, num_classes, bias=bias)
        self.head_drop = nn.Dropout(drop_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.norm(x)
        x = self.head_drop(x)
        x = self.fc2(x)
        return x


class MetaFormerBlock(nn.Module):
    """Implementation of one MetaFormer block."""
    def __init__(self, dim, token_mixer=Attention, mlp_act=StarReLU, mlp_bias=False,
                 norm_layer=LayerNorm2d, proj_drop=0., drop_path=0., use_nchw=True,
                 layer_scale_init_value=None, res_scale_init_value=None, **kwargs):
        super().__init__()
        
        # Create Scale layers if init values are provided
        if layer_scale_init_value is not None:
            self.layer_scale1 = Scale(dim=dim, init_value=layer_scale_init_value, use_nchw=use_nchw)
            self.layer_scale2 = Scale(dim=dim, init_value=layer_scale_init_value, use_nchw=use_nchw)
        else:
            self.layer_scale1 = nn.Identity()
            self.layer_scale2 = nn.Identity()
            
        if res_scale_init_value is not None:
            self.res_scale1 = Scale(dim=dim, init_value=res_scale_init_value, use_nchw=use_nchw)
            self.res_scale2 = Scale(dim=dim, init_value=res_scale_init_value, use_nchw=use_nchw)
        else:
            self.res_scale1 = nn.Identity()
            self.res_scale2 = nn.Identity()

        self.norm1 = norm_layer(dim)
        self.token_mixer = token_mixer(dim=dim, proj_drop=proj_drop, **kwargs)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(dim, int(4 * dim), act_layer=mlp_act, bias=mlp_bias, 
                        drop=proj_drop, use_conv=use_nchw)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = self.res_scale1(x) + self.layer_scale1(
            self.drop_path1(self.token_mixer(self.norm1(x))))
        x = self.res_scale2(x) + self.layer_scale2(
            self.drop_path2(self.mlp(self.norm2(x))))
        return x


class MetaFormerStage(nn.Module):
    def __init__(self, in_chs, out_chs, depth=2, token_mixer=nn.Identity, mlp_act=StarReLU,
                 mlp_bias=False, downsample_norm=LayerNorm2d, norm_layer=LayerNorm2d,
                 proj_drop=0., dp_rates=[0.] * 2, layer_scale_init_value=None,
                 res_scale_init_value=None, **kwargs):
        super().__init__()
        
        self.use_nchw = not issubclass(token_mixer, Attention)
        
        # Downsample if input and output channels differ
        self.downsample = nn.Identity() if in_chs == out_chs else Downsampling(
            in_chs, out_chs, kernel_size=3, stride=2, padding=1, norm_layer=downsample_norm)
        
        # Create blocks
        blocks = []
        for i in range(depth):
            blocks.append(MetaFormerBlock(
                dim=out_chs, token_mixer=token_mixer, mlp_act=mlp_act, mlp_bias=mlp_bias,
                norm_layer=norm_layer, proj_drop=proj_drop, drop_path=dp_rates[i],
                layer_scale_init_value=layer_scale_init_value, res_scale_init_value=res_scale_init_value,
                use_nchw=self.use_nchw, **kwargs))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: Tensor):
        x = self.downsample(x)
        B, C, H, W = x.shape
        
        if not self.use_nchw:
            x = x.reshape(B, C, -1).transpose(1, 2)
            
        x = self.blocks(x)
        
        if not self.use_nchw:
            x = x.transpose(1, 2).reshape(B, C, H, W)
            
        return x


class CAFormerB36(nn.Module):
    """CAFormer-B36 model implementation."""
    def __init__(self, in_chans=3, num_classes=7, global_pool='avg', drop_rate=0.0, use_metadata=False, metadata_dim=0, metadata_hidden=64):
        super().__init__()
        
        # Model configuration
        depths = [3, 12, 18, 3]
        dims = [128, 256, 512, 768]
        token_mixers = [SepConv, SepConv, Attention, Attention]
        norm_layers = [LayerNorm2dNoBias] * 2 + [LayerNormNoBias] * 2
        res_scale_init_values = (None, None, 1.0, 1.0)
        
        self.num_classes = num_classes
        self.num_features = dims[-1]
        self.drop_rate = drop_rate
        
        # Stem layer
        self.stem = Stem(in_chans, dims[0], norm_layer=LayerNorm2dNoBias)
        
        # Build stages
        stages = []
        prev_dim = dims[0]
        dp_rates = [x.tolist() for x in torch.linspace(0, 0.1, sum(depths)).split(depths)]
        
        for i in range(len(depths)):
            stages.append(MetaFormerStage(
                prev_dim, dims[i], depth=depths[i], token_mixer=token_mixers[i],
                mlp_act=StarReLU, mlp_bias=False, proj_drop=0.0, dp_rates=dp_rates[i],
                layer_scale_init_value=None, res_scale_init_value=res_scale_init_values[i],
                downsample_norm=LayerNorm2dNoBias, norm_layer=norm_layers[i]))
            prev_dim = dims[i]
            
        self.stages = nn.Sequential(*stages)
        
        # Head
        self.use_metadata = use_metadata
        self.num_classes = num_classes
        self.num_features = dims[-1]
        self.drop_rate = drop_rate

        if self.use_metadata:
            self.metadata_mlp = nn.Sequential(
                nn.Linear(metadata_dim, metadata_hidden),
                nn.ReLU(),
                nn.Linear(metadata_hidden, metadata_hidden),
                nn.ReLU()
            )
            self.head = nn.Sequential(OrderedDict([
                ('global_pool', SelectAdaptivePool2d(pool_type=global_pool)),
                ('norm', LayerNorm2d(self.num_features)),
                ('flatten', nn.Flatten(1)),
                ('drop', nn.Dropout(drop_rate)),
                ('fc', MlpHead(self.num_features + metadata_hidden, num_classes, drop_rate=drop_rate))
            ]))
        else:
            self.head = nn.Sequential(OrderedDict([
                ('global_pool', SelectAdaptivePool2d(pool_type=global_pool)),
                ('norm', LayerNorm2d(self.num_features)),
                ('flatten', nn.Flatten(1)),
                ('drop', nn.Dropout(drop_rate)),
                ('fc', MlpHead(self.num_features, num_classes, drop_rate=drop_rate))
            ]))

    def forward_features(self, x: Tensor):
        x = self.stem(x)
        x = self.stages(x)
        return x

    def forward(self, x: Tensor, meta: Tensor = None):
        x = self.forward_features(x)
        x = self.head.global_pool(x)
        x = self.head.norm(x)
        x = self.head.flatten(x)
        x = self.head.drop(x)

        if self.use_metadata and meta is not None:
            meta_embed = self.metadata_mlp(meta)
            x = torch.cat([x, meta_embed], dim=1)

        x = self.head.fc(x)
        return x

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.LayerNorm, LayerNorm2d, LayerNorm2dNoBias, LayerNormNoBias)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)


def caformer_b36(num_classes=7, **kwargs) -> CAFormerB36:
    """Create a CAFormer-B36 model instance."""
    return CAFormerB36(num_classes=num_classes, **kwargs)