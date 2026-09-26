import torch
import torch.nn as nn
from pytorch_wavelets import DWTForward  # used to implement the discrete wavelet transform (DWT)
# pip install pywavelets==1.7.0 -i https://pypi.tuna.tsinghua.edu.cn/simple/
"""
    Paper address: https://arxiv.org/pdf/2405.01992
    Paper title: SFFNet: A Wavelet-Based Spatial and Frequency Domain Fusion Network for Remote Sensing Segmentation (TOP journal)
    Chinese title: SFFNet: A wavelet-based spatial and frequency domain fusion network for remote sensing image segmentation
    Tutorial video: https://www.bilibili.com/video/BV1GkzZB7EXC/
    Wavelet Transform Feature Decomposer (WTFD):
        Practical significance: ① Grayscale-sensitive regions are hard to segment: in shadow, edge, and texture-transition areas, spatial-domain features are easily affected by illumination and background changes, leading to blurred boundaries and confused categories.
                ② Using frequency-domain features alone loses spatial structure information: although wavelet transform / Fourier methods can capture grayscale-mutation features, they destroy spatial position information, resulting in incomplete object shapes or semantics.
        Implementation: first apply the Haar wavelet transform to the input spatial-domain features, decomposing them into 1 low-frequency component (A, corresponding to the global smooth information of the image) and 3 high-frequency components (horizontal, vertical,
                and diagonal edge/detail information), providing frequency-domain information support for spatial-domain feature fusion.
"""

class WTFD(nn.Module):
    """
    WTFD: Wavelet Transform Feature Decomposition (wavelet transform feature decomposition module)
    Decomposes the input features into low-frequency (smooth) and high-frequency (edge/detail) components and processes them with separate convolutions.
    Supports upsampling to keep the output size consistent with the input.
    """
    def __init__(self, in_ch, out_ch, upsample=True, fusion=True):
        super(WTFD, self).__init__()
        self.upsample = upsample
        self.fusion = fusion
        # 1-level Haar wavelet decomposition
        self.wt = DWTForward(J=1, mode='zero', wave='haar')

        # high-frequency branch: HL/LH/HH concatenated into 3*in_ch channels -> 1x1 to reduce back to in_ch
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(in_ch * 3, in_ch, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
        )

        # low-frequency output channel alignment
        self.outconv_bn_relu_L = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

        # high-frequency output channel alignment
        self.outconv_bn_relu_H = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

        # fused output channel alignment (used when fusion=True)
        self.outconv_bn_relu_F = nn.Sequential(
            nn.Conv2d(out_ch * 2, out_ch, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """ Forward propagation logic:
            Input x -> wavelet decomposition -> high-frequency concatenation -> convolution fusion -> output low- and high-frequency features
            If upsample=True, the outputs are upsampled back to the original size
            If fusion=True, the low- and high-frequency features are fused and a single feature map is output
         """
        # use the wavelet transform for feature decomposition # yL: low-frequency part (smooth information)
        # yH: high-frequency part (contains detail information in 3 directions)
        yL, yH = self.wt(x)

        # extract the 3 directional components from the high-frequency coefficients: HL (horizontal), LH (vertical), HH (diagonal)
        yH0 = yH[0]                 # [B, C, 3, H2, W2]
        y_HL = yH0[:, :, 0, :, :]   # [B, C, H2, W2]
        y_LH = yH0[:, :, 1, :, :]
        y_HH = yH0[:, :, 2, :, :]
        # # concatenate the high-frequency information of the three directions along the channel dimension
        yH_cat = torch.cat([y_HL, y_LH, y_HH], dim=1)  # [B, 3*C, H2, W2]
        # apply convolution + BN + ReLU to the concatenated high-frequency features
        yH_feat = self.conv_bn_relu(yH_cat)            # [B, C, H2, W2]

        # apply output convolutions to the low- and high-frequency features separately
        yL_out = self.outconv_bn_relu_L(yL)           # [B, out_ch, H2, W2]
        yH_out = self.outconv_bn_relu_H(yH_feat)      # [B, out_ch, H2, W2]

        # if upsampling is needed, restore the output to the original size
        if self.upsample:
            original_size = x.shape[2:]
            yL_out = nn.functional.interpolate(yL_out, size=original_size, mode='bilinear', align_corners=False)
            yH_out = nn.functional.interpolate(yH_out, size=original_size, mode='bilinear', align_corners=False)

        # if fusion is needed, fuse the low- and high-frequency features and output
        if self.fusion:
            yF_out = torch.cat([yL_out, yH_out], dim=1)  # [B, 2*out_ch, H, W]
            yF_out = self.outconv_bn_relu_F(yF_out)     # [B, out_ch, H, W]
            return yF_out

        return yL_out, yH_out

if __name__ == "__main__":
    # It is recommended that the input H and W be even to accommodate the 1-level wavelet downsampling
    x = torch.randn(1, 32, 50, 50)

    # test the fusion version (for YOLOv8 integration)
    print("=== Test the fusion version (fusion=True, upsample=True) ===")
    model = WTFD(32, 32, upsample=True, fusion=True)
    out = model(x)
    print(f"Input tensor shape: {x.shape}")
    print(f"Fused output tensor shape: {out.shape}")

    # test the non-fusion version
    print("\n=== Test the non-fusion version (fusion=False, upsample=True) ===")
    model2 = WTFD(32, 32, upsample=True, fusion=False)
    out_L, out_H = model2(x)
    print(f"Input tensor shape: {x.shape}")
    print(f"Low-frequency output tensor shape: {out_L.shape}")
    print(f"High-frequency output tensor shape: {out_H.shape}")

    # test the no-upsampling version
    print("\n=== Test the no-upsampling version (fusion=False, upsample=False) ===")
    model3 = WTFD(32, 32, upsample=False, fusion=False)
    out_L2, out_H2 = model3(x)
    print(f"Input tensor shape: {x.shape}")
    print(f"Low-frequency output tensor shape: {out_L2.shape}")
    print(f"High-frequency output tensor shape: {out_H2.shape}")

    print("\nSame WeChat official account, Bilibili, and CSDN account")
    print("布尔大学士 reminds you: the code is error-free~~~~")
