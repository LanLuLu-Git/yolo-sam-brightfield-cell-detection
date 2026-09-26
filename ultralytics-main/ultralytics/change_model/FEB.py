import torch
import torch.nn as nn


# Paper title: Spatial-Frequency Enhanced Mamba for Multi-Modal Image Fusion
# Paper link: https://arxiv.org/pdf/2511.06593
# Official GitHub: https://github.com/SunHui1216/SFMFusion/blob/main/models/model.py


class FB(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.select1 = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, stride=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, stride=1),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.select2 = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):  # input BCHW output BCHW
        # Save the original input (for the residual connection); at this point x is in BCHW format
        skip = x  # B C H W
        B, C, H, W = x.shape

        # Step 1: BCHW -> BHWC (to match the dimension format of the original logic)
        x = x.permute(0, 2, 3, 1).contiguous()  # B H W C

        # Step 2: convert back to BCHW for convolution and Fourier transform (original core logic)
        x = x.permute(0, 3, 1, 2).contiguous()  # B C H W
        y = torch.fft.rfft2(x) + 1e-8  # Fourier transform (B C H W/2+1, complex)
        a = torch.abs(y)  # amplitude spectrum (B C H W/2+1)
        p = torch.angle(y)  # phase spectrum (B C H W/2+1)

        # feature extraction on amplitude and phase spectra
        a = self.select1(a)
        p = self.select2(p)

        # reconstruct complex domain
        real = a * torch.cos(p)
        imag = a * torch.sin(p)
        out = torch.complex(real, imag) + 1e-8

        # inverse Fourier transform to restore spatial domain
        out = torch.fft.irfft2(out, s=(H, W), norm='backward') + 1e-8
        out = torch.abs(out) + 1e-8  # ensure real output

        # residual connection (added directly in BCHW format)
        out = out + skip  # B C H W

        # final output remains in BCHW format (no extra dimension conversion needed)
        return out



def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))

class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))

class Bottleneck_FEB(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = FB(c_)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C3k_FEB(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck_FEB(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))

# when c3k=True, use Bottleneck_FEB; when false, use the regular Bottleneck for feature extraction
class C3k2_FEB(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k_FEB(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g) for _ in range(n)
        )



def main():
    # 1. Set the random seed to ensure reproducibility
    torch.manual_seed(42)
    # automatically detect GPU/CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Define test hyperparameters (BCHW format)
    batch_size = 2  # batch size
    dim = 32  # number of channels
    height = 64  # feature map height
    width = 64  # feature map width

    # 3. Create the test input (BCHW format, matching PyTorch's default convolution input format)
    test_input = torch.randn(batch_size, dim, height, width).to(device)
    print(f"\nInput shape (BCHW): {test_input.shape}")

    # 4. Initialize the FB module and move it to the specified device
    fb_module = FB(dim=dim).to(device)

    # 5. Forward pass (disable gradient computation to save memory)
    with torch.no_grad():
        output = fb_module(test_input)

    # 6. Verify the output shape
    print(f"Output shape (BCHW): {output.shape}")
    # assert that input and output shapes are consistent
    assert test_input.shape == output.shape, "Input and output shapes are inconsistent!"
    print("✅ Forward pass test passed; input and output shapes are consistent!")

    # 7. Print key statistics (for debugging)
    print(f"\nOutput tensor statistics:")
    print(f"  Mean: {output.mean().item():.4f}")
    print(f"  Max: {output.max().item():.4f}")
    print(f"  Min: {output.min().item():.4f}")
    print(f"  Std: {output.std().item():.4f}")


if __name__ == "__main__":
    main()