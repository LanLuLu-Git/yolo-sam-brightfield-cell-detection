import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Softmax
from einops import rearrange, repeat

# Reference paper: https://arxiv.org/pdf/2503.11030

def custom_complex_normalization(input_tensor, dim=-1):
    """
    Custom complex normalization function.

    Applies softmax normalization to the real and imaginary parts of a complex tensor separately.
    This is a key step in the frequency-domain attention mechanism, used to ensure the numerical
    stability of the attention weights.

    Args:
        input_tensor: complex tensor, containing real and imaginary parts
        dim: dimension to normalize over, defaults to -1 (last dimension)

    Returns:
        normalized complex tensor
    """
    real_part = input_tensor.real  # extract the real part
    imag_part = input_tensor.imag  # extract the imaginary part
    norm_real = F.softmax(real_part, dim=dim)  # softmax normalization of the real part
    norm_imag = F.softmax(imag_part, dim=dim)  # softmax normalization of the imaginary part
    return torch.complex(norm_real, norm_imag)  # recombine into a complex tensor


class PFAE(nn.Module):
    """
    PFAE (Parallel Frequency Attention Enhancement) module.

    A frequency-enhanced attention network module designed specifically for camouflaged object detection.
    It enhances the feature representation capability through parallel processing, multi-scale feature
    extraction, and a frequency-domain attention mechanism.

    Key features:
    1. Parallel processing: six different types of convolution branches process the input features in parallel
    2. Multi-scale features: convolution kernels with different dilation rates capture multi-scale information
    3. Frequency-domain attention: attention is computed in the frequency domain, enhancing the spatial-frequency representation of features
    4. Residual connection: maintains gradient flow and avoids vanishing gradients

    Overall architecture:
    Input -> downsampling -> 6 parallel branches -> feature fusion -> output
    """

    def __init__(self, dim):
        """
        Initialize the PFAE module.

        Args:
            dim: number of channels of the input feature map
        """
        super().__init__()

        # =================================================================
        # Part 1: downsampling layer - reduces computational complexity and extracts main features
        # =================================================================
        in_dim = dim // 2  # number of channels after downsampling, reduced to half
        self.down_conv = nn.Sequential(
            nn.Conv2d(dim, in_dim, 3, padding=1),    # 3x3 convolution for feature extraction
            nn.BatchNorm2d(in_dim),                  # batch normalization to stabilize training
            nn.ReLU(True)                            # ReLU activation function to introduce nonlinearity
        )
        down_dim = in_dim // 2  # output channels of each branch

        # =================================================================
        # Part 2: six parallel convolution branches - extract multi-scale features
        # =================================================================

        # Branch 1: 1x1 convolution - captures global information and reduces the number of parameters
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=1),  # 1x1 convolution for channel transformation
            nn.BatchNorm2d(down_dim),                    # batch normalization
            nn.ReLU(True)                                # ReLU activation
        )

        # Branch 2: dilated convolution (dilation=3) - captures mid-scale features
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=3, padding=3),  # dilated convolution to enlarge the receptive field
            nn.BatchNorm2d(down_dim),                  # batch normalization
            nn.ReLU(True)                              # ReLU activation
        )

        # Branch 3: dilated convolution (dilation=5) - captures larger-scale features
        self.conv3 = nn.Sequential(
            nn.Conv2d(down_dim, down_dim, kernel_size=3, dilation=5, padding=5),  # input channels are down_dim
            nn.BatchNorm2d(down_dim),                  # batch normalization
            nn.ReLU(True)                              # ReLU activation
        )

        # Branch 4: dilated convolution (dilation=7) - captures even larger-scale features
        self.conv4 = nn.Sequential(
            nn.Conv2d(down_dim, down_dim, kernel_size=3, dilation=7, padding=7),  # input channels are down_dim
            nn.BatchNorm2d(down_dim),                  # batch normalization
            nn.ReLU(True)                              # ReLU activation
        )

        # Branch 5: dilated convolution (dilation=9) - captures the largest-scale features
        self.conv5 = nn.Sequential(
            nn.Conv2d(down_dim, down_dim, kernel_size=3, dilation=9, padding=9),  # input channels are down_dim
            nn.BatchNorm2d(down_dim),                  # batch normalization
            nn.ReLU(True)                              # ReLU activation
        )

        # Branch 6: global average pooling branch - captures global context information
        # Note: BatchNorm is avoided here during global pooling because a single pixel cannot be batch-normalized
        self.conv6_no_bn = nn.Conv2d(in_dim, down_dim, kernel_size=1)  # 1x1 convolution for channel transformation
        self.conv6_act = nn.ReLU(True)                           # ReLU activation function

        # =================================================================
        # Part 3: feature fusion layer - fuses features from all branches
        # =================================================================
        self.fuse = nn.Sequential(
            nn.Conv2d(6 * down_dim, down_dim, kernel_size=1),  # 1x1 convolution fusing features from the 6 branches
            nn.BatchNorm2d(down_dim),                          # batch normalization
            nn.ReLU(True)                                      # ReLU activation
        )

        # =================================================================
        # Part 4: output layer - restores the original number of channels and outputs the final features
        # =================================================================
        self.out = nn.Sequential(
            nn.Conv2d(down_dim, down_dim // 2, kernel_size=3, padding=1),  # 3x3 convolution for further processing
            nn.BatchNorm2d(down_dim // 2),                                # batch normalization
            nn.ReLU(True),                                                # ReLU activation
            nn.Conv2d(down_dim // 2, dim, kernel_size=1)                  # 1x1 convolution to restore the original channels
        )

        # =================================================================
        # Part 5: frequency-domain attention parameters
        # =================================================================

        # attention temperature parameter, used to adjust the distribution of attention weights
        self.temperature = nn.Parameter(torch.ones(8, 1, 1))

        # feature projection layer, projecting the multi-head attention output back to the original dimension
        self.project_out = nn.Conv2d(down_dim * 2, down_dim, kernel_size=1, bias=False)

        # attention weight generation network, used to generate local attention weights
        self.weight = nn.Sequential(
            nn.Conv2d(down_dim, down_dim // 8, 1, bias=True),  # 1x1 convolution for dimension reduction
            nn.BatchNorm2d(down_dim // 8),                    # batch normalization
            nn.ReLU(True),                                    # ReLU activation
            nn.Conv2d(down_dim // 8, down_dim, 1, bias=True), # 1x1 convolution for dimension increase
            nn.Sigmoid()                                      # Sigmoid activation to generate attention weights
        )

        # number of heads for multi-head attention, used for parallel attention computation
        self.num_heads = 8

    def frequency_attention(self, x):
        """
        Frequency-domain attention computation function.

        This is the core innovation of the PFAE module. Computing attention in the frequency domain
        better captures the frequency characteristics of the image, which is especially effective for
        camouflaged object detection because camouflaged objects usually have a special representation
        in the frequency domain.

        Processing flow:
        1. Spatial domain -> frequency domain (FFT transform)
        2. Multi-head attention computation
        3. Complex normalization
        4. Frequency domain -> spatial domain (IFFT transform)
        5. Local attention enhancement
        6. Residual connection

        Args:
            x: input feature map [B, C, H, W]

        Returns:
            attention-enhanced feature map [B, C, H, W]
        """
        b, c, h, w = x.shape  # get the feature map size information

        # =================================================================
        # Step 1: spatial-domain to frequency-domain transform
        # =================================================================
        q_f = torch.fft.fft2(x.float())  # FFT transform of the Query features
        k_f = torch.fft.fft2(x.float())  # FFT transform of the Key features
        v_f = torch.fft.fft2(x.float())  # FFT transform of the Value features
        tepqkv = torch.fft.fft2(x.float())  # temporary variable for the subsequent local attention computation

        # =================================================================
        # Step 2: multi-head processing - split features by the number of attention heads
        # =================================================================
        q_f = rearrange(q_f, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k_f = rearrange(k_f, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v_f = rearrange(v_f, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        # =================================================================
        # Step 3: feature normalization - ensures numerical stability
        # =================================================================
        q_f = F.normalize(q_f, dim=-1)  # L2 normalization of Query features
        k_f = F.normalize(k_f, dim=-1)  # L2 normalization of Key features

        # =================================================================
        # Step 4: attention weight computation - similarity computation in the frequency domain
        # =================================================================
        attn_f = (q_f @ k_f.transpose(-2, -1)) * self.temperature  # matrix multiplication to compute attention weights
        attn_f = custom_complex_normalization(attn_f, dim=-1)     # complex normalization

        # =================================================================
        # Step 5: frequency-to-spatial inverse transform - obtain the attention output in the spatial domain
        # =================================================================
        out_f = torch.abs(torch.fft.ifft2(attn_f @ v_f))  # IFFT transform and take the absolute value
        out_f = rearrange(out_f, 'b head c (h w) -> b (head c) h w',
                         head=self.num_heads, h=h, w=w)   # reshape back to spatial-domain features

        # =================================================================
        # Step 6: local attention enhancement - combined with the weight generation network
        # =================================================================
        out_f_l = torch.abs(torch.fft.ifft2(self.weight(tepqkv.real) * tepqkv))

        # =================================================================
        # Step 7: feature fusion and residual connection
        # =================================================================
        out = self.project_out(torch.cat((out_f, out_f_l), 1))  # fuse global and local attention
        return out + x  # residual connection to maintain gradient flow

    def forward(self, x):
        """
        Forward propagation function of the PFAE module.

        This is the entry function of the module, defining the data flow path:
        Input -> downsampling -> parallel branch processing -> feature fusion -> output

        It specially handles the numerical instability of BatchNorm when batch_size=1.

        Args:
            x: input feature map [B, C, H, W]

        Returns:
            output feature map [B, C, H, W] - with the same spatial size and number of channels as the input
        """
        # =================================================================
        # BatchNorm fix: when batch_size=1, temporarily switch to evaluation mode to avoid numerical instability
        # =================================================================
        training_backup = self.training  # save the original training state
        if x.size(0) == 1:
            self.eval()  # set evaluation mode to avoid the problem where BatchNorm needs multiple samples

        try:
            # =================================================================
            # Step 1: initial downsampling - reduces computational complexity
            # =================================================================
            x = self.down_conv(x)  # [B, in_dim, H, W]

            # =================================================================
            # Step 2: Branch 1 processing - 1x1 convolution captures global information
            # =================================================================
            conv1 = self.conv1(x)  # [B, down_dim, H, W]

            # =================================================================
            # Step 3: frequency attention processing for the other branches
            # Note: residual connections are used between branches; the output of the previous branch is the input of the next
            # =================================================================
            F_2 = self.frequency_attention(self.conv2(x))      # Branch 2: dilation=3
            F_3 = self.frequency_attention(self.conv3(F_2))    # Branch 3: dilation=5, uses F_2 as input
            F_4 = self.frequency_attention(self.conv4(F_3))    # Branch 4: dilation=7, uses F_3 as input
            F_5 = self.frequency_attention(self.conv5(F_4))    # Branch 5: dilation=9, uses F_4 as input

            # =================================================================
            # Step 4: global branch processing - captures global context
            # =================================================================
            global_feat = F.adaptive_avg_pool2d(x, 1)  # global average pooling to 1x1
            conv5 = self.conv6_act(self.conv6_no_bn(global_feat))  # 1x1 convolution + ReLU
            conv5 = F.interpolate(conv5, size=x.size()[2:], mode='bilinear', align_corners=False)  # upsample back to the original size, specify align_corners to avoid warnings

            # =================================================================
            # Step 5: feature fusion - fuse the features of all 6 branches
            # =================================================================
            F_out = self.out(self.fuse(torch.cat((conv1, F_2, F_3, F_4, F_5, conv5), 1)))

        finally:
            # restore the original training mode
            self.training = training_backup

        return F_out


# =============================================================================
# The following are common base modules used in the YOLO series of models, referencing the YOLOv8 implementation
# =============================================================================

def autopad(k, p=None, d=1):
    """
    Automatically compute the padding size to achieve 'same' output size.

    Args:
        k: convolution kernel size
        p: padding size (auto-computed if None)
        d: dilation factor

    Returns:
        computed padding size
    """
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    """
    Standard convolution module.

    A combination of convolution, batch normalization, and activation function; a fundamental building block in deep learning.
    """

    default_act = nn.SiLU()  # default activation function

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """
        Initialize the convolution module.

        Args:
            c1: number of input channels
            c2: number of output channels
            k: convolution kernel size
            s: stride
            p: padding size
            g: number of groups
            d: dilation factor
            act: whether to use an activation function
        """
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation function."""
        x = x.to(self.conv.weight.device)  # ensure the input tensor is on the device of the convolution layer
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform the fused convolution operation (without BatchNorm)."""
        x = x.to(self.conv.weight.device)  # ensure the input tensor is on the device of the convolution layer
        return self.act(self.conv(x))


class Bottleneck(nn.Module):
    """
    Standard bottleneck module.

    With a structure of 1x1 convolution for dimension reduction, 3x3 convolution for feature extraction,
    and 1x1 convolution for dimension increase, it reduces the number of parameters while maintaining
    feature representation capability.
    """

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """
        Initialize the bottleneck module.

        Args:
            c1: number of input channels
            c2: number of output channels
            shortcut: whether to use a residual connection
            g: number of groups
            k: convolution kernel size tuple
            e: expansion ratio (the ratio of hidden-layer channels)
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden-layer channels
        self.cv1 = Conv(c1, c_, k[0], 1)      # 1x1 dimension-reduction convolution
        self.cv2 = Conv(c_, c2, k[1], 1, g=g) # 3x3 feature-extraction convolution
        self.add = shortcut and c1 == c2      # condition for using a residual connection

    def forward(self, x):
        """Apply the forward propagation of the bottleneck module."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f(nn.Module):
    """
    C2f module: a faster CSP bottleneck implementation.

    Through an improved branch structure and residual connections, it improves computational efficiency while maintaining performance.
    """

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """
        Initialize the C2f module.

        Args:
            c1: number of input channels
            c2: number of output channels
            n: number of Bottleneck modules
            shortcut: whether to use a residual connection
            g: number of groups
            e: expansion ratio
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden-layer channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)        # initial 1x1 convolution
        self.cv2 = Conv((2 + n) * self.c, c2, 1)     # final 1x1 convolution
        # create n Bottleneck modules
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward propagation of the C2f module."""
        y = list(self.cv1(x).chunk(2, 1))  # split features
        y.extend(m(y[-1]) for m in self.m)  # pass through Bottleneck modules in sequence
        return self.cv2(torch.cat(y, 1))    # concatenate and pass through the final convolution

    def forward_split(self, x):
        """Forward propagation using split() instead of chunk()."""
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions"""

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


class Bottleneck_PFAE(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = PFAE(c_)  # embed the PFAE module in the bottleneck
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # Use Bottleneck_PFAE instead of the regular Bottleneck, integrating the PFAE attention mechanism into feature extraction
        self.m = nn.Sequential(*(Bottleneck_PFAE(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


# when c3k=True, use Bottleneck_PFAE feature fusion; when false, use the regular Bottleneck for feature extraction
class C3k2_PFAE(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g) for _ in range(n)
        )


def main():
    """Test function - verifies the correctness of the PFAE module."""
    # test parameters
    batch_size = 2
    channels = 64
    height, width = 8, 8

    # create the input tensor
    x = torch.randn(batch_size, channels, height, width)
    print(f"Input shape: {x.shape}")

    # initialize the PFAE module
    pfae = PFAE(dim=channels)
    print(f"\nModule parameters: {sum(p.numel() for p in pfae.parameters()):,}")

    # forward pass test
    output = pfae(x)
    print(f"Output shape: {output.shape}")

    # verify whether the output shape is correct
    assert output.shape == x.shape, f"Output shape mismatch: expected {x.shape}, got {output.shape}"
    print("✅ Test passed! The PFAE module works correctly.")


if __name__ == "__main__":
    main()