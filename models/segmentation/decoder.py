import torch
import torch.nn as nn
import torch.nn.functional as F

from . import modules as md

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, use_batchnorm=True):
        super().__init__()
        self.conv1 = md.Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        print("decoder block input", x.shape)
        x = self.conv1(x)
        x = self.conv2(x)
        print("decoder block output", x.shape)
        return x

class UnetDecoder(nn.Module):
    def __init__(self, encoder_channels, decoder_channels, use_batchnorm=True):
        super().__init__()

        n_blocks = 5
        if n_blocks != len(decoder_channels):
            raise ValueError(
                f"Model depth is {n_blocks}, but you provide `decoder_channels` for {len(decoder_channels)} blocks."
            )

        encoder_channels = encoder_channels[1:]

        encoder_channels = encoder_channels[::-1]

        # Computing blocks input and output channels
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = decoder_channels

        self.center = nn.Identity()  # CenterBlock is not needed

        # Combining decoder blocks
        self.blocks = nn.ModuleList([
            DecoderBlock(in_ch, skip_ch, out_ch, use_batchnorm=use_batchnorm)
            for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
        ])

    def forward(self, *features):
        features = features[1:]  # Remove first skip with same spatial resolution
        features = features[::-1]  # Reverse channels to start from head of encoder

        head = features[0]
        skips = features[1:]

        x = self.center(head)
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)

        return x