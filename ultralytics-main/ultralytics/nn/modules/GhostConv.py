import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import autopad

class GhostConv(nn.Module):
    """Ghost Convolution v1"""
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True, ratio=2):
        super().__init__()
        c_ = c2 // ratio
        self.primary = nn.Conv2d(c1, c_, k, s, autopad(k), groups=g, bias=False)
        self.cheap   = nn.Conv2d(c_, c_, 5, 1, 2, groups=c_, bias=False)
        self.bn      = nn.BatchNorm2d(c2)
        self.act     = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        y = self.primary(x)
        ghost = self.cheap(y)
        out = torch.cat([y, ghost], dim=1)[:, :self.primary.out_channels*2, :, :]
        return self.act(self.bn(out))

# Below is C3_Ghost, which can be used to directly replace C3
class GhostBottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = GhostConv(c1, c_, 1, 1)
        self.cv2 = GhostConv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C3_Ghost(nn.Module):
    # CSP Bottleneck with 3 convolutions + Ghost
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(2 * c_, c2, 1)
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_, shortcut, g, 1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))
