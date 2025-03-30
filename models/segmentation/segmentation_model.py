from typing import Optional, Union, List, Callable

import torch
import torch.nn as nn

from .decoder import UnetDecoder
from .modules import Activation
from .mix_transformer import get_encoder


class SegmentationHead(nn.Sequential):
    """Simple segmentation head: Conv2d -> Upsampling -> Activation"""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, 
                 activation: Optional[str] = None, upsampling: int = 1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                           padding=kernel_size // 2)
        upsampling = (nn.UpsamplingBilinear2d(scale_factor=upsampling) 
                      if upsampling > 1 else nn.Identity())
        activation = Activation(activation)
        super().__init__(conv2d, upsampling, activation)


class SegmentationModel(nn.Module):
    """Base segmentation model with encoder-decoder structure"""
    def forward(self, x):
        features = self.encoder(x)
        decoder_output = self.decoder(*features)
        masks = self.segmentation_head(decoder_output)
        return masks


class Unet(SegmentationModel):
    """U-Net implementation with MixTransformer backbone"""
    def __init__(
        self,
        encoder_name: str = "mit_b4",
        encoder_depth: int = 5,
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[Union[str, Callable]] = None,
    ):
        super().__init__()

        # Initialize encoder (backbone)
        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
        )

        # Create decoder
        self.decoder = UnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            use_batchnorm=decoder_use_batchnorm,
        )

        # Create segmentation head for final prediction
        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )

        self.name = f"u-{encoder_name}"


def mit_unet():
    """Create a U-Net model with MiT-B4 backbone and sigmoid activation"""
    return Unet(encoder_name="mit_b4", activation="sigmoid")