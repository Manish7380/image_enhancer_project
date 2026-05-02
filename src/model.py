"""
Low-Light Image Enhancement Model

Original simple U-Net architecture that was trained and saved in working/model.pth
This model matches the checkpoint and is compatible with existing weights.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Simple convolutional block with two conv layers + batch norm + ReLU"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    """Up-sampling block with transposed convolution + concatenation + ConvBlock"""
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNetEnhancer(nn.Module):
    """
    Simple U-Net for Low-Light Image Enhancement
    
    Architecture:
    - Encoder: 3 ConvBlocks with MaxPool downsampling (3 -> 32 -> 64 -> 128)
    - Bottleneck: 1 ConvBlock at lowest resolution (128 -> 256)
    - Decoder: 3 UpBlocks with skip connections from encoder (256 -> 128 -> 64 -> 32)
    - Output: Single conv + Sigmoid activation for [0,1] range
    
    This matches the architecture that was trained and saved in working/model.pth
    """
    def __init__(self, in_channels: int = 3, out_channels: int = 3):
        super().__init__()
        # Encoder
        self.encoder1 = ConvBlock(in_channels, 32)
        self.encoder2 = ConvBlock(32, 64)
        self.encoder3 = ConvBlock(64, 128)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Bottleneck
        self.bottleneck = ConvBlock(128, 256)
        
        # Decoder
        self.decoder3 = UpBlock(256, 128, 128)
        self.decoder2 = UpBlock(128, 64, 64)
        self.decoder1 = UpBlock(64, 32, 32)
        
        # Output layer
        self.output_layer = nn.Sequential(
            nn.Conv2d(32, out_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, apply_postprocessing: bool = True) -> torch.Tensor:
        # Encoder with skip connections
        skip1 = self.encoder1(x)
        skip2 = self.encoder2(self.pool(skip1))
        skip3 = self.encoder3(self.pool(skip2))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool(skip3))
        
        # Decoder with skip connections
        dec = self.decoder3(bottleneck, skip3)
        dec = self.decoder2(dec, skip2)
        dec = self.decoder1(dec, skip1)
        
        # Output
        raw_output = self.output_layer(dec)
        
        if not apply_postprocessing:
            return raw_output
            
        # --- BAKED-IN POST-PROCESSING VIA TENSOR OPERATIONS ---
        
        # 1. Dark detection (calculate per-image mean intensity in the batch)
        mean_intensity = x.mean(dim=(1, 2, 3), keepdim=True)
        is_extreme_dark = (mean_intensity < 0.05).float()
        
        # 2. Gamma correction (balance midtones for much brighter overall image)
        gamma = 0.75 * is_extreme_dark + 0.65 * (1.0 - is_extreme_dark)
        output = raw_output ** gamma
        
        # 3. Saturation boost (RGB space approximation)
        # Luminance weights for RGB: 0.299, 0.587, 0.114
        out_lum = 0.299 * output[:, 0:1, :, :] + 0.587 * output[:, 1:2, :, :] + 0.114 * output[:, 2:3, :, :]
        sat_boost = 1.1 * is_extreme_dark + 1.25 * (1.0 - is_extreme_dark)
        output = out_lum + sat_boost * (output - out_lum)
        output = torch.clamp(output, 0.0, 1.0)
        
        # 4. Highlight blending (Prevent blown highlights / glowing edges)
        orig_lum = 0.299 * x[:, 0:1, :, :] + 0.587 * x[:, 1:2, :, :] + 0.114 * x[:, 2:3, :, :]
        highlight_mask = torch.clamp((orig_lum - 0.5) * 2.5, 0.0, 1.0)
        output = output * (1.0 - highlight_mask) + x * highlight_mask
        
        # 5. Extreme dark handling (Prevent unrealistic camera noise)
        dark_clip = 0.85 * is_extreme_dark + 1.0 * (1.0 - is_extreme_dark)
        output = torch.clamp(output * dark_clip, 0.0, 1.0)
        
        # 6. Deep Shadow / Noise Masking (Fixes gray/grainy blacks)
        # Softened to let more light into shadows without fully exposing noise
        shadow_mask = torch.clamp(orig_lum * 35.0, 0.0, 1.0) # Ramps up much faster, leaving only the darkest darks black
        output = output * shadow_mask + x * (1.0 - shadow_mask)
        
        return output
