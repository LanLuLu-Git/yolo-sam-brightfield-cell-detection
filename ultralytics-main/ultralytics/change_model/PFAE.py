import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Softmax
from einops import rearrange, repeat

# 参考论文：https://arxiv.org/pdf/2503.11030

def custom_complex_normalization(input_tensor, dim=-1):
    """
    自定义复数归一化函数
    
    对复数张量的实部和虚部分别进行softmax归一化处理。
    这是频率域注意力机制中的关键步骤，用于确保注意力权重的数值稳定性。
    
    参数:
        input_tensor: 复数张量，包含实部和虚部
        dim: 归一化操作的维度，默认为-1（最后一个维度）
    
    返回:
        归一化后的复数张量
    """
    real_part = input_tensor.real  # 提取实部
    imag_part = input_tensor.imag  # 提取虚部
    norm_real = F.softmax(real_part, dim=dim)  # 实部softmax归一化
    norm_imag = F.softmax(imag_part, dim=dim)  # 虚部softmax归一化
    return torch.complex(norm_real, norm_imag)  # 重新组合成复数张量


class PFAE(nn.Module):
    """
    PFAE (Parallel Frequency Attention Enhancement) 模块
    
    这是一个专门用于伪装目标检测的频率增强注意力网络模块。
    它通过并行处理和多尺度特征提取，结合频率域注意力机制来增强特征表示能力。
    
    主要特点：
    1. 并行处理：使用6个不同类型的卷积分支并行处理输入特征
    2. 多尺度特征：通过不同膨胀率的卷积核捕获多尺度信息
    3. 频率域注意力：在频域进行注意力计算，增强特征的空间频率表示
    4. 残差连接：保持梯度流动，避免梯度消失
    
    整体架构：
    输入 -> 下采样 -> 6个并行分支 -> 特征融合 -> 输出
    """

    def __init__(self, dim):
        """
        初始化PFAE模块
        
        参数:
            dim: 输入特征图的通道数
        """
        super().__init__()
        
        # =================================================================
        # 第一部分：下采样层 - 减少计算复杂度并提取主要特征
        # =================================================================
        in_dim = dim // 2  # 下采样后的通道数，缩减为原来的一半
        self.down_conv = nn.Sequential(
            nn.Conv2d(dim, in_dim, 3, padding=1),    # 3x3卷积进行特征提取
            nn.BatchNorm2d(in_dim),                  # 批归一化，稳定训练过程
            nn.ReLU(True)                            # ReLU激活函数，引入非线性
        )
        down_dim = in_dim // 2  # 各分支的输出通道数

        # =================================================================
        # 第二部分：六个并行卷积分支 - 提取多尺度特征
        # =================================================================
        
        # 分支1：1x1卷积 - 捕获全局信息，减少参数数量
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=1),  # 1x1卷积，用于通道变换
            nn.BatchNorm2d(down_dim),                    # 批归一化
            nn.ReLU(True)                                # ReLU激活
        )

        # 分支2：膨胀卷积(dilation=3) - 捕获中等尺度特征
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=3, padding=3),  # 膨胀卷积，扩大感受野
            nn.BatchNorm2d(down_dim),                  # 批归一化
            nn.ReLU(True)                              # ReLU激活
        )

        # 分支3：膨胀卷积(dilation=5) - 捕获较大尺度特征  
        self.conv3 = nn.Sequential(
            nn.Conv2d(down_dim, down_dim, kernel_size=3, dilation=5, padding=5),  # 输入通道为down_dim
            nn.BatchNorm2d(down_dim),                  # 批归一化
            nn.ReLU(True)                              # ReLU激活
        )

        # 分支4：膨胀卷积(dilation=7) - 捕获更大尺度特征
        self.conv4 = nn.Sequential(
            nn.Conv2d(down_dim, down_dim, kernel_size=3, dilation=7, padding=7),  # 输入通道为down_dim
            nn.BatchNorm2d(down_dim),                  # 批归一化
            nn.ReLU(True)                              # ReLU激活
        )

        # 分支5：膨胀卷积(dilation=9) - 捕获最大尺度特征
        self.conv5 = nn.Sequential(
            nn.Conv2d(down_dim, down_dim, kernel_size=3, dilation=9, padding=9),  # 输入通道为down_dim
            nn.BatchNorm2d(down_dim),                  # 批归一化
            nn.ReLU(True)                              # ReLU激活
        )

        # 分支6：全局平均池化分支 - 捕获全局上下文信息
        # 注意：这里避免在全局池化时使用BatchNorm，因为单个像素无法进行批归一化
        self.conv6_no_bn = nn.Conv2d(in_dim, down_dim, kernel_size=1)  # 1x1卷积进行通道变换
        self.conv6_act = nn.ReLU(True)                           # ReLU激活函数

        # =================================================================
        # 第三部分：特征融合层 - 融合所有分支的特征
        # =================================================================
        self.fuse = nn.Sequential(
            nn.Conv2d(6 * down_dim, down_dim, kernel_size=1),  # 1x1卷积融合6个分支的特征
            nn.BatchNorm2d(down_dim),                          # 批归一化
            nn.ReLU(True)                                      # ReLU激活
        )

        # =================================================================
        # 第四部分：输出层 - 恢复原始通道数并输出最终特征
        # =================================================================
        self.out = nn.Sequential(
            nn.Conv2d(down_dim, down_dim // 2, kernel_size=3, padding=1),  # 3x3卷积进一步处理
            nn.BatchNorm2d(down_dim // 2),                                # 批归一化
            nn.ReLU(True),                                                # ReLU激活
            nn.Conv2d(down_dim // 2, dim, kernel_size=1)                  # 1x1卷积恢复原始通道数
        )

        # =================================================================
        # 第五部分：频率域注意力相关参数
        # =================================================================
        
        # 注意力温度参数，用于调节注意力权重的分布
        self.temperature = nn.Parameter(torch.ones(8, 1, 1))
        
        # 特征投影层，将多头注意力的输出特征投影回原始维度
        self.project_out = nn.Conv2d(down_dim * 2, down_dim, kernel_size=1, bias=False)

        # 注意力权重生成网络，用于生成局部注意力权重
        self.weight = nn.Sequential(
            nn.Conv2d(down_dim, down_dim // 8, 1, bias=True),  # 1x1卷积降维
            nn.BatchNorm2d(down_dim // 8),                    # 批归一化
            nn.ReLU(True),                                    # ReLU激活
            nn.Conv2d(down_dim // 8, down_dim, 1, bias=True), # 1x1卷积升维
            nn.Sigmoid()                                      # Sigmoid激活，生成注意力权重
        )

        # 多头注意力的头数，用于并行的注意力计算
        self.num_heads = 8

    def frequency_attention(self, x):
        """
        频率域注意力计算函数
        
        这是PFAE模块的核心创新点。在频域进行注意力计算，能够更好地捕捉
        图像的频率特征，对于伪装目标检测特别有效，因为伪装物体通常在
        频率域有特殊的表示。
        
        处理流程：
        1. 空间域 -> 频域 (FFT变换)
        2. 多头注意力计算
        3. 复数归一化
        4. 频域 -> 空间域 (IFFT变换)
        5. 局部注意力增强
        6. 残差连接
        
        参数:
            x: 输入特征图 [B, C, H, W]
            
        返回:
            注意力增强后的特征图 [B, C, H, W]
        """
        b, c, h, w = x.shape  # 获取特征图尺寸信息
        
        # =================================================================
        # 步骤1：空间域到频域的变换
        # =================================================================
        q_f = torch.fft.fft2(x.float())  # Query特征的FFT变换
        k_f = torch.fft.fft2(x.float())  # Key特征的FFT变换  
        v_f = torch.fft.fft2(x.float())  # Value特征的FFT变换
        tepqkv = torch.fft.fft2(x.float())  # 临时变量，用于后续局部注意力计算

        # =================================================================
        # 步骤2：多头处理 - 将特征按注意力头数分割
        # =================================================================
        q_f = rearrange(q_f, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k_f = rearrange(k_f, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v_f = rearrange(v_f, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        # =================================================================
        # 步骤3：特征归一化 - 确保数值稳定性
        # =================================================================
        q_f = F.normalize(q_f, dim=-1)  # L2归一化Query特征
        k_f = F.normalize(k_f, dim=-1)  # L2归一化Key特征

        # =================================================================
        # 步骤4：注意力权重计算 - 频域中的相似性计算
        # =================================================================
        attn_f = (q_f @ k_f.transpose(-2, -1)) * self.temperature  # 矩阵乘法计算注意力权重
        attn_f = custom_complex_normalization(attn_f, dim=-1)     # 复数归一化处理

        # =================================================================
        # 步骤5：频域到空间域的反变换 - 获得空间域的注意力输出
        # =================================================================
        out_f = torch.abs(torch.fft.ifft2(attn_f @ v_f))  # IFFT变换并取绝对值
        out_f = rearrange(out_f, 'b head c (h w) -> b (head c) h w', 
                         head=self.num_heads, h=h, w=w)   # 重新整形为空间域特征

        # =================================================================
        # 步骤6：局部注意力增强 - 结合权重生成网络
        # =================================================================
        out_f_l = torch.abs(torch.fft.ifft2(self.weight(tepqkv.real) * tepqkv))

        # =================================================================
        # 步骤7：特征融合和残差连接
        # =================================================================
        out = self.project_out(torch.cat((out_f, out_f_l), 1))  # 融合全局和局部注意力
        return out + x  # 残差连接，保持梯度流动

    def forward(self, x):
        """
        PFAE模块的前向传播函数
        
        这是整个模块的入口函数，定义了数据的流动路径：
        输入 -> 下采样 -> 并行分支处理 -> 特征融合 -> 输出
        
        特别处理了BatchNorm在batch_size=1时的数值不稳定问题。
        
        参数:
            x: 输入特征图 [B, C, H, W]
            
        返回:
            输出特征图 [B, C, H, W] - 与输入具有相同的空间尺寸和通道数
        """
        # =================================================================
        # BatchNorm修复：当batch_size=1时，临时切换到评估模式避免数值不稳定
        # =================================================================
        training_backup = self.training  # 保存原始训练状态
        if x.size(0) == 1:
            self.eval()  # 设置为评估模式，避免BatchNorm需要多个样本的问题
        
        try:
            # =================================================================
            # 步骤1：初始下采样 - 减少计算复杂度
            # =================================================================
            x = self.down_conv(x)  # [B, in_dim, H, W]
            
            # =================================================================
            # 步骤2：分支1处理 - 1x1卷积捕获全局信息
            # =================================================================
            conv1 = self.conv1(x)  # [B, down_dim, H, W]

            # =================================================================
            # 步骤3：其他分支的频率注意力处理
            # 注意：各分支之间采用残差连接，前一个分支的输出作为下一个分支的输入
            # =================================================================
            F_2 = self.frequency_attention(self.conv2(x))      # 分支2：dilation=3
            F_3 = self.frequency_attention(self.conv3(F_2))    # 分支3：dilation=5，使用F_2作为输入
            F_4 = self.frequency_attention(self.conv4(F_3))    # 分支4：dilation=7，使用F_3作为输入  
            F_5 = self.frequency_attention(self.conv5(F_4))    # 分支5：dilation=9，使用F_4作为输入

            # =================================================================
            # 步骤4：全局分支处理 - 捕获全局上下文
            # =================================================================
            global_feat = F.adaptive_avg_pool2d(x, 1)  # 全局平均池化到1x1
            conv5 = self.conv6_act(self.conv6_no_bn(global_feat))  # 1x1卷积+ReLU
            conv5 = F.interpolate(conv5, size=x.size()[2:], mode='bilinear', align_corners=False)  # 上采样回原尺寸，指定align_corners避免警告

            # =================================================================
            # 步骤5：特征融合 - 融合所有6个分支的特征
            # =================================================================
            F_out = self.out(self.fuse(torch.cat((conv1, F_2, F_3, F_4, F_5, conv5), 1)))
            
        finally:
            # 恢复原始的训练模式
            self.training = training_backup
            
        return F_out


# =============================================================================
# 以下是YOLO系列模型中常用的基础模块，参考YOLOv8的实现
# =============================================================================

def autopad(k, p=None, d=1):
    """
    自动计算填充大小以实现'same'输出尺寸
    
    参数:
        k: 卷积核大小
        p: 填充大小（如果为None则自动计算）
        d: 膨胀系数
        
    返回:
        计算后的填充大小
    """
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    """
    标准卷积模块
    
    包含卷积、批归一化和激活函数的组合，是深度学习中的基础构建块。
    """

    default_act = nn.SiLU()  # 默认激活函数

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """
        初始化卷积模块
        
        参数:
            c1: 输入通道数
            c2: 输出通道数  
            k: 卷积核大小
            s: 步长
            p: 填充大小
            g: 分组数
            d: 膨胀系数
            act: 是否使用激活函数
        """
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """应用卷积、批归一化和激活函数"""
        x = x.to(self.conv.weight.device)  # 确保输入张量在卷积层所在设备
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """执行融合的卷积操作（不包含BatchNorm）"""
        x = x.to(self.conv.weight.device)  # 确保输入张量在卷积层所在设备
        return self.act(self.conv(x))


class Bottleneck(nn.Module):
    """
    标准瓶颈模块
    
    通过1x1卷积降维、3x3卷积提取特征、1x1卷积升维的结构，
    减少参数量的同时保持特征表达能力。
    """

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """
        初始化瓶颈模块
        
        参数:
            c1: 输入通道数
            c2: 输出通道数
            shortcut: 是否使用残差连接
            g: 分组数
            k: 卷积核大小元组
            e: 膨胀系数（隐藏层通道数的比例）
        """
        super().__init__()
        c_ = int(c2 * e)  # 隐藏层通道数
        self.cv1 = Conv(c1, c_, k[0], 1)      # 1x1降维卷积
        self.cv2 = Conv(c_, c2, k[1], 1, g=g) # 3x3特征提取卷积
        self.add = shortcut and c1 == c2      # 是否使用残差连接的条件

    def forward(self, x):
        """应用瓶颈模块的前向传播"""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f(nn.Module):
    """
    C2f模块：更快的CSP瓶颈实现
    
    通过改进的分支结构和残差连接，在保持性能的同时提高计算效率。
    """

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """
        初始化C2f模块
        
        参数:
            c1: 输入通道数
            c2: 输出通道数
            n: Bottleneck模块的数量
            shortcut: 是否使用残差连接
            g: 分组数
            e: 膨胀系数
        """
        super().__init__()
        self.c = int(c2 * e)  # 隐藏层通道数
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)        # 初始1x1卷积
        self.cv2 = Conv((2 + n) * self.c, c2, 1)     # 最后的1x1卷积
        # 创建n个Bottleneck模块
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """C2f模块的前向传播"""
        y = list(self.cv1(x).chunk(2, 1))  # 分割特征
        y.extend(m(y[-1]) for m in self.m)  # 依次通过Bottleneck模块
        return self.cv2(torch.cat(y, 1))    # 拼接并通过最终卷积

    def forward_split(self, x):
        """使用split()而非chunk()的前向传播"""
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
        self.cv2 = PFAE(c_)  # 在瓶颈模块中嵌入PFAE模块
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
        # 使用Bottleneck_PFAE替代普通的Bottleneck，在特征提取中集成PFAE注意力机制
        self.m = nn.Sequential(*(Bottleneck_PFAE(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


# 在c3k=True时，使用Bottleneck_PFAE特征融合，为false的时候我们使用普通的Bottleneck提取特征
class C3k2_PFAE(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g) for _ in range(n)
        )


def main():
    """测试函数 - 验证PFAE模块的正确性"""
    # 测试参数
    batch_size = 2
    channels = 64
    height, width = 8, 8
    
    # 创建输入张量
    x = torch.randn(batch_size, channels, height, width)
    print(f"输入形状: {x.shape}")

    # 初始化PFAE模块
    pfae = PFAE(dim=channels)
    print(f"\n模块参数量: {sum(p.numel() for p in pfae.parameters()):,}")

    # 前向传播测试
    output = pfae(x)
    print(f"输出形状: {output.shape}")
    
    # 验证输出形状是否正确
    assert output.shape == x.shape, f"输出形状不匹配: 期望{x.shape}, 得到{output.shape}"
    print("✅ 测试通过！PFAE模块工作正常")


if __name__ == "__main__":
    main()