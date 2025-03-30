import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union

from . import modules as md


class DecoderBlock(nn.Module):
    """
    U-Net decoder block: upsampling followed by conv layers with skip connection.
    
    Args:
        in_channels: Number of input channels
        skip_channels: Number of skip connection channels
        out_channels: Number of output channels
        use_batchnorm: Whether to use batch normalization
        interpolation_mode: Upsampling interpolation mode
    """
    def __init__(
        self, 
        in_channels: int, 
        skip_channels: int, 
        out_channels: int, 
        use_batchnorm: bool = True,
        interpolation_mode: str = "nearest",
    ):
        super().__init__()
        self.interpolation_mode = interpolation_mode
        
        # First conv after concatenating upsampled features with skip connection
        self.conv1 = md.Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        
        # Second conv for further feature refinement
        self.conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )

    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with optional skip connection."""
        # Upsample input features
        x = F.interpolate(x, scale_factor=2, mode=self.interpolation_mode)
        
        # Concatenate with skip connection if provided
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            
        # Apply convolutions
        x = self.conv1(x)
        x = self.conv2(x)
        
        return x


class UnetDecoder(nn.Module):
    """
    U-Net decoder consisting of center block and a sequence of decoder blocks.
    
    Args:
        encoder_channels: List of encoder channels (in reverse order for skip connections)
        decoder_channels: List of decoder channels
        use_batchnorm: Whether to use batch normalization
        interpolation_mode: Upsampling interpolation mode
    """
    def __init__(
        self, 
        encoder_channels: List[int], 
        decoder_channels: List[int], 
        use_batchnorm: bool = True,
        interpolation_mode: str = "nearest",
    ):
        super().__init__()
        
        # Validate decoder configuration
        if len(decoder_channels) == 0:
            raise ValueError("Decoder channels list cannot be empty")
            
        # Extract relevant encoder channels for skip connections
        encoder_channels = encoder_channels[1:]  # First channel doesn't need a skip
        encoder_channels = encoder_channels[::-1]  # Reverse to match decoder blocks
        
        # Compute input and skip channels for each decoder block
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]  # Last block has no skip
        out_channels = decoder_channels
        
        # Center block (identity in this implementation)
        self.center = nn.Identity()
        
        # Create decoder blocks
        self.blocks = nn.ModuleList([
            DecoderBlock(
                in_ch, skip_ch, out_ch, 
                use_batchnorm=use_batchnorm,
                interpolation_mode=interpolation_mode
            )
            for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
        ])

    def forward(self, *features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through decoder blocks with skip connections.
        
        Args:
            *features: List of features from encoder (from deepest to shallowest)
            
        Returns:
            Decoder output tensor
        """
        # Process encoder features (exclude first, reverse order)
        features = features[1:]
        features = features[::-1]  # Reverse to process from deepest
        
        # Extract head feature and skip connections
        head = features[0]
        skips = features[1:] if len(features) > 1 else []
        
        # Process through center and decoder blocks
        x = self.center(head)
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)
            
        return x