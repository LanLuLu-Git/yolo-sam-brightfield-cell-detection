import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import os



# https://arxiv.org/pdf/2411.18409


# Fix: remove the normalized parameter of torch.fft.rfft and use manual normalization instead (compatible with all PyTorch versions)
def channel_fft(x):
    """Perform a Fourier transform over the channel dimension (manual normalization to avoid version compatibility issues)."""
    x_permuted = x.permute(0, 2, 3, 1)  # [B, H, W, C]
    # perform RFFT, manually computing the normalization factor (equivalent to the original normalized=True behavior)
    fft_complex = torch.fft.rfft(x_permuted, dim=-1)
    c = x_permuted.shape[-1]  # original number of channels
    fft_complex = fft_complex / torch.sqrt(torch.tensor(c, dtype=fft_complex.dtype, device=fft_complex.device))
    return fft_complex.permute(0, 3, 1, 2)  # [B, (C//2)+1, H, W]


# Fix: remove the normalized parameter of torch.fft.irfft and use manual inverse normalization
def channel_ifft(fft_x, original_channels):
    """Perform an inverse Fourier transform over the channel dimension (manual inverse normalization)."""
    fft_permuted = fft_x.permute(0, 2, 3, 1)  # [B, H, W, (C//2)+1]
    # perform inverse RFFT, manually restoring the normalization
    # irfft returns a real tensor, so .real is not needed
    ifft_real = torch.fft.irfft(fft_permuted, dim=-1, n=original_channels)
    ifft_real = ifft_real * torch.sqrt(torch.tensor(original_channels, dtype=ifft_real.dtype, device=ifft_real.device))
    return ifft_real.permute(0, 3, 1, 2)  # [B, C, H, W]


# amplitude/phase enhancement module (CLC)
class CLCBlock(nn.Module):
    """Conv + LeakyReLU + Conv module."""
    def __init__(self, in_channels, kernel_size=1, leaky_slope=0.2):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.clc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, bias=True),
            nn.LeakyReLU(negative_slope=leaky_slope, inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, bias=True)
        )

    def forward(self, x):
        return self.clc(x)


# DNRU module
class DNRU(nn.Module):
    """Depth-wise Conv3×3 + Normalization + ReLU + Upsampling"""
    def __init__(self, in_channels, out_channels, upscale_factor=2, leaky_slope=0.2):
        super().__init__()
        self.upscale_factor = upscale_factor
        self.dnru = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=leaky_slope, inplace=True)
        )

    def forward(self, x):
        x_refined = self.dnru(x)
        x_upsampled = F.interpolate(
            x_refined, scale_factor=self.upscale_factor, mode='bilinear', align_corners=True
        )
        return x_upsampled


# FRCAB core module
class FRCAB(nn.Module):
    """Fourier Residual Channel Attention Block

    Args:
        c1: number of input channels
        c2: number of output channels (defaults to the same as c1)
        e: expansion ratio, used to control the number of internal channels (default 1.0)
        leaky_slope: negative slope of LeakyReLU (default 0.2)
    """
    def __init__(self, c1, c2=None, e=1.0, leaky_slope=0.2):
        super().__init__()
        c2 = c2 if c2 is not None else c1
        self.in_channels = c1
        self.out_channels = c2

        # module initialization
        self.clc3 = CLCBlock(c1, kernel_size=3, leaky_slope=leaky_slope)
        self.clc1 = CLCBlock(c1, kernel_size=1, leaky_slope=leaky_slope)
        self.conv_adjust = nn.Conv2d(c1, (c1 // 2) + 1, kernel_size=1, padding=0)
        self.dnru = DNRU(c1, c2, upscale_factor=2)

    def forward(self, x):
        # CLC3 processing and global average pooling
        x_clc3 = self.clc3(x)
        x_gap = F.adaptive_avg_pool2d(x_clc3, output_size=(1, 1))

        # channel Fourier transform and amplitude/phase extraction (using the fixed fft function)
        x_cfft = channel_fft(x_gap)
        x_amp = torch.abs(x_cfft)
        x_pha = torch.angle(x_cfft)

        # CLC1 processing and channel adjustment
        x_clc1 = self.clc1(x)
        x_clc1_gap = F.adaptive_avg_pool2d(x_clc1, output_size=(1, 1))
        x_clc1_gap_adjust = self.conv_adjust(x_clc1_gap)

        # amplitude and phase enhancement
        x_amp_enhanced = x_amp * x_clc1_gap_adjust
        x_pha_enhanced = x_pha * x_clc1_gap_adjust

        # reconstruct the frequency-domain features (using Euler's formula, fully supports automatic differentiation)
        # polar(r, θ) = r * (cos(θ) + i*sin(θ))
        x_cfft_enhanced_real = x_amp_enhanced * torch.cos(x_pha_enhanced)
        x_cfft_enhanced_imag = x_amp_enhanced * torch.sin(x_pha_enhanced)
        x_cfft_enhanced = torch.complex(x_cfft_enhanced_real, x_cfft_enhanced_imag)
        x_icfft = channel_ifft(x_cfft_enhanced, original_channels=self.in_channels)
        x_icfft_expanded = F.interpolate(x_icfft, size=x.shape[2:], mode='bilinear', align_corners=True)

        # residual fusion and DNRU output
        x_residual = x_icfft_expanded * x_clc3 + x
        return self.dnru(x_residual)




def main():
    # set the random seed to ensure reproducibility
    torch.manual_seed(42)
    print("=" * 50)
    print("Start testing the FRCAB module...")

    # 1. Configure test parameters (simulating actual input dimensions)
    batch_size = 2       # batch size
    in_channels = 32     # number of input channels
    out_channels = 64    # number of output channels
    height, width = 27, 31  # input feature map size
    upscale_factor = 2   # upsampling factor

    # 2. Create a random test input (shape: [B, C, H, W])
    input_tensor = torch.randn(batch_size, in_channels, height, width)
    print(f"Input tensor shape: {input_tensor.shape}")

    # 3. Initialize the FRCAB module
    frcab = FRCAB(
        in_channels=in_channels,
        out_channels=out_channels,
        upscale_factor=upscale_factor
    )

    # 4. Run the forward pass
    output_tensor = frcab(input_tensor)

    # 5. Verify that the output dimensions meet expectations
    expected_height = height * upscale_factor
    expected_width = width * upscale_factor
    print(f"Output tensor shape: {output_tensor.shape}")
    print(f"Expected output shape: ({batch_size}, {out_channels}, {expected_height}, {expected_width})")

    # 6. Assertion check (raises an error if dimensions do not match)
    assert output_tensor.shape == (batch_size, out_channels, expected_height, expected_width), \
        f"Dimension check failed! Actual output {output_tensor.shape}, expected ({batch_size}, {out_channels}, {expected_height}, {expected_width})"

    # 7. Print the test result
    print("\n" + "=" * 50)
    print("FRCAB module test passed!")
    print(f"✅ Forward pass ran without errors")
    print(f"✅ Dimension match: input {height}x{width} -> output {expected_height}x{expected_width}")
    print(f"✅ Channel match: input {in_channels} -> output {out_channels}")
    print("=" * 50)


if __name__ == "__main__":
    main()