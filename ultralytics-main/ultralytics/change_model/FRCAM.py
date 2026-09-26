import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import os



# https://arxiv.org/pdf/2411.18409


# 修复：移除torch.fft.rfft的normalized参数，改用手动归一化（适配所有PyTorch版本）
def channel_fft(x):
    """对通道维度执行傅里叶变换（手动归一化，避免版本兼容问题）"""
    x_permuted = x.permute(0, 2, 3, 1)  # [B, H, W, C]
    # 执行RFFT，手动计算归一化因子（对应原normalized=True的效果）
    fft_complex = torch.fft.rfft(x_permuted, dim=-1)
    c = x_permuted.shape[-1]  # 原始通道数
    fft_complex = fft_complex / torch.sqrt(torch.tensor(c, dtype=fft_complex.dtype, device=fft_complex.device))
    return fft_complex.permute(0, 3, 1, 2)  # [B, (C//2)+1, H, W]


# 修复：移除torch.fft.irfft的normalized参数，手动逆归一化
def channel_ifft(fft_x, original_channels):
    """对通道维度执行逆傅里叶变换（手动逆归一化）"""
    fft_permuted = fft_x.permute(0, 2, 3, 1)  # [B, H, W, (C//2)+1]
    # 执行逆RFFT，手动恢复归一化
    # irfft返回的是实数张量，不需要.real
    ifft_real = torch.fft.irfft(fft_permuted, dim=-1, n=original_channels)
    ifft_real = ifft_real * torch.sqrt(torch.tensor(original_channels, dtype=ifft_real.dtype, device=ifft_real.device))
    return ifft_real.permute(0, 3, 1, 2)  # [B, C, H, W]


# 幅度/相位增强模块（CLC）
class CLCBlock(nn.Module):
    """Conv + LeakyReLU + Conv 模块"""
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


# DNRU模块
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


# FRCAB核心模块
class FRCAB(nn.Module):
    """Fourier Residual Channel Attention Block
    
    参数说明:
        c1: 输入通道数
        c2: 输出通道数（默认与c1相同）
        e: 扩展比例，用于控制内部通道数（默认1.0）
        leaky_slope: LeakyReLU的负斜率（默认0.2）
    """
    def __init__(self, c1, c2=None, e=1.0, leaky_slope=0.2):
        super().__init__()
        c2 = c2 if c2 is not None else c1
        self.in_channels = c1
        self.out_channels = c2

        # 模块初始化
        self.clc3 = CLCBlock(c1, kernel_size=3, leaky_slope=leaky_slope)
        self.clc1 = CLCBlock(c1, kernel_size=1, leaky_slope=leaky_slope)
        self.conv_adjust = nn.Conv2d(c1, (c1 // 2) + 1, kernel_size=1, padding=0)
        self.dnru = DNRU(c1, c2, upscale_factor=2)

    def forward(self, x):
        # CLC3处理与全局平均池化
        x_clc3 = self.clc3(x)
        x_gap = F.adaptive_avg_pool2d(x_clc3, output_size=(1, 1))

        # 通道傅里叶变换与幅度/相位提取（使用修复后的fft函数）
        x_cfft = channel_fft(x_gap)
        x_amp = torch.abs(x_cfft)
        x_pha = torch.angle(x_cfft)

        # CLC1处理与通道调整
        x_clc1 = self.clc1(x)
        x_clc1_gap = F.adaptive_avg_pool2d(x_clc1, output_size=(1, 1))
        x_clc1_gap_adjust = self.conv_adjust(x_clc1_gap)

        # 幅度和相位增强
        x_amp_enhanced = x_amp * x_clc1_gap_adjust
        x_pha_enhanced = x_pha * x_clc1_gap_adjust

        # 重构频域特征（使用欧拉公式，完全支持自动微分）
        # polar(r, θ) = r * (cos(θ) + i*sin(θ))
        x_cfft_enhanced_real = x_amp_enhanced * torch.cos(x_pha_enhanced)
        x_cfft_enhanced_imag = x_amp_enhanced * torch.sin(x_pha_enhanced)
        x_cfft_enhanced = torch.complex(x_cfft_enhanced_real, x_cfft_enhanced_imag)
        x_icfft = channel_ifft(x_cfft_enhanced, original_channels=self.in_channels)
        x_icfft_expanded = F.interpolate(x_icfft, size=x.shape[2:], mode='bilinear', align_corners=True)

        # 残差融合与DNRU输出
        x_residual = x_icfft_expanded * x_clc3 + x
        return self.dnru(x_residual)




def main():
    # 设置随机种子，保证结果可复现
    torch.manual_seed(42)
    print("=" * 50)
    print("开始测试FRCAB模块...")

    # 1. 配置测试参数（模拟实际输入维度）
    batch_size = 2       # 批量大小
    in_channels = 32     # 输入通道数
    out_channels = 64    # 输出通道数
    height, width = 27, 31  # 输入特征图尺寸
    upscale_factor = 2   # 上采样倍数

    # 2. 创建随机测试输入（形状：[B, C, H, W]）
    input_tensor = torch.randn(batch_size, in_channels, height, width)
    print(f"输入张量形状: {input_tensor.shape}")

    # 3. 初始化FRCAB模块
    frcab = FRCAB(
        in_channels=in_channels,
        out_channels=out_channels,
        upscale_factor=upscale_factor
    )

    # 4. 执行前向传播
    output_tensor = frcab(input_tensor)

    # 5. 验证输出维度是否符合预期
    expected_height = height * upscale_factor
    expected_width = width * upscale_factor
    print(f"输出张量形状: {output_tensor.shape}")
    print(f"预期输出形状: ({batch_size}, {out_channels}, {expected_height}, {expected_width})")

    # 6. 断言验证（维度不匹配则报错）
    assert output_tensor.shape == (batch_size, out_channels, expected_height, expected_width), \
        f"维度验证失败！实际输出 {output_tensor.shape}，预期 ({batch_size}, {out_channels}, {expected_height}, {expected_width})"

    # 7. 输出测试结果
    print("\n" + "=" * 50)
    print("FRCAB模块测试通过！")
    print(f"✅ 前向传播无报错")
    print(f"✅ 维度匹配：输入 {height}x{width} → 输出 {expected_height}x{expected_width}")
    print(f"✅ 通道数匹配：输入 {in_channels} → 输出 {out_channels}")
    print("=" * 50)


if __name__ == "__main__":
    main()