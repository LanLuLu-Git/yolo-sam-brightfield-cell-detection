import torch
import torch.nn as nn
from einops import rearrange

"""
    Paper address: https://arxiv.org/pdf/2403.10067
    Paper title: Hybrid Convolutional and Attention Network for Hyperspectral Image Denoising (2024 TOP)
    Chinese title: Hybrid Convolutional and Attention Network for Hyperspectral Image Denoising (2024 TOP)
    Tutorial video: https://www.bilibili.com/video/BV1qh4CzNEce/
    Convolution and Attention Fusion Module (CAFM):
        Practical significance: ① Limitations of convolution: convolutional neural networks (CNNs) excel at extracting local neighborhood features (such as spectral correlation and spatial details), but due to their limited receptive field, they struggle to capture long-range dependencies.
                ② Shortcomings of the attention mechanism: it lacks fine-grained local information. The Transformer attention mechanism has advantages in global modeling and long-range dependency capture, but it often ignores the detailed features of local neighborhoods.
        Implementation: the local branch uses convolution to extract details, the global branch uses attention to capture dependencies, and finally the two are added and fused, using both local and global features for denoising.
"""

class CAFM(nn.Module):
    def __init__(self, dim, num_heads, bias=False):
        super(CAFM, self).__init__()
        # number of attention heads
        self.num_heads = num_heads
        # temperature parameter used to scale the attention scores (learnable)
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # generate Q, K, V features at once (channels x3), kernel size 1x1x1
        self.qkv = nn.Conv3d(dim, dim * 3, kernel_size=(1, 1, 1), bias=bias)
        # apply depthwise separable convolution to Q, K, V to enhance local modeling capability
        self.qkv_dwconv = nn.Conv3d(
            dim * 3, dim * 3, kernel_size=(3, 3, 3),
            stride=1, padding=1, groups=dim * 3, bias=bias
        )
        # output projection, mapping the fused features back to the original dimension
        self.project_out = nn.Conv3d(dim, dim, kernel_size=(1, 1, 1), bias=bias)
        # an additional 1x1x1 convolution layer that linearly combines the multi-head features
        self.fc = nn.Conv3d(3 * self.num_heads, 9, kernel_size=(1, 1, 1), bias=True)

        # depthwise convolution for local convolution enhancement; groups are kept aligned with the multi-head dimension
        self.dep_conv = nn.Conv3d(
            9 * dim // self.num_heads, dim, kernel_size=(3, 3, 3),
            bias=True, groups=dim // self.num_heads, padding=1
        )

    def forward(self, x):
        # input shape BxCxHxW
        b, c, h, w = x.shape
        # expand a dimension to become BxCx1xHxW to fit 3D convolution
        x = x.unsqueeze(2)

        # first apply a 1x1x1 convolution to obtain QKV, then use depthwise convolution to enhance local information
        qkv = self.qkv_dwconv(self.qkv(x))
        # remove the expanded dimension and return to Bx3CxHxW
        qkv = qkv.squeeze(2)

        # ========== Local convolution enhancement ==========
        # reorder the dimensions for subsequent operations
        f_conv = qkv.permute(0, 2, 3, 1)
        # reshape into (B, HW, 3*num_heads, channel blocks)
        f_all = qkv.reshape(f_conv.shape[0], h * w, 3 * self.num_heads, -1).permute(0, 2, 1, 3)
        # apply a 1x1x1 convolution to f_all to obtain 9-channel local features
        f_all = self.fc(f_all.unsqueeze(2))
        f_all = f_all.squeeze(2)

        # reshape the local convolution features back to the convolution input format
        f_conv = f_all.permute(0, 3, 1, 2).reshape(x.shape[0], 9 * x.shape[1] // self.num_heads, h, w)
        f_conv = f_conv.unsqueeze(2)
        # extract local features via depthwise convolution
        out_conv = self.dep_conv(f_conv)
        out_conv = out_conv.squeeze(2)

        # ========== Global self-attention ==========
        # split qkv into Q, K, V along the channel dimension
        q, k, v = qkv.chunk(3, dim=1)
        # rearrange into (B, num_heads, C_per_head, HW)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        # normalize Q and K to ensure computational stability
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        # compute the attention weights: QxK^T, multiplied by the temperature parameter
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        # weight V with the attention weights
        out = (attn @ v)
        # restore to the (B, C, H, W) format
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = out.unsqueeze(2)
        out = self.project_out(out)
        out = out.squeeze(2)

        # ========== Fused output ==========
        # global self-attention result + local convolution enhancement result
        output = out + out_conv
        return output

if __name__ == '__main__':
    # define the model with 32 input channels and 8 attention heads
    model = CAFM(dim=32, num_heads=8)
    # input tensor, shape BxCxHxW
    input = torch.rand(2, 32, 50, 50)
    output = model(input)
    print(f"Input tensor shape: {input.shape}")
    print(f"Output tensor shape: {output.shape}")
    print("Same WeChat official account, Bilibili, and CSDN account")
    print("The code is error-free~~~~")
