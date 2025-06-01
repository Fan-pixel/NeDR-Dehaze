from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            c = x.size(1)
            if c != self.normalized_shape[0]:
                raise ValueError(f"Input channel size {c} does not match normalized_shape {self.normalized_shape[0]}")
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

    def update_normalized_shape(self, new_shape, device):
        self.normalized_shape = (new_shape,)
        self.weight = nn.Parameter(torch.ones(new_shape).to(device))
        self.bias = nn.Parameter(torch.zeros(new_shape).to(device))
device = "cuda" if torch.cuda.is_available() else "cpu"
class MCC(nn.Module):
    def __init__(self, f_number, num_heads, padding_mode, bias=False):
        super(MCC, self).__init__()
        self.norm = LayerNorm(f_number, eps=1e-6, data_format='channels_first')
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.pwconv = nn.Conv2d(f_number, f_number * 3, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(f_number * 3, f_number * 3, 3, 1, 1, bias=bias, padding_mode=padding_mode, groups=f_number * 3)
        self.project_out = nn.Conv2d(f_number, f_number, kernel_size=1, bias=bias)
        self.feedforward = nn.Sequential(
            nn.Conv2d(f_number, f_number, 1, 1, 0, bias=bias),
            nn.GELU(),
            nn.Conv2d(f_number, f_number, 3, 1, 1, bias=bias, groups=f_number, padding_mode=padding_mode),
            nn.GELU()
        )

        # 添加去雾层
        self.dehaze_layers = nn.Sequential(
            nn.Conv2d(f_number, f_number, kernel_size=3, padding=1, bias=bias),
            nn.GELU(),
            nn.Conv2d(f_number, f_number, kernel_size=3, padding=1, bias=bias),
            nn.GELU()
        )

        # 色彩域检测器
        self.color_space_detector = EnhancedColorSpaceDetector()

        # 预训练的色彩域转换模型
        self.hvi_to_rgb_model = EnhancedHVItoRGBModel()
        self.ycbcr_to_rgb_model = EnhancedYCbCrtoRGBModel()

        # 通道注意力模块
        self.channel_attention = ChannelAttention(f_number)

        # 添加一个卷积层来调整 x 的通道数
        self.channel_adjust = nn.Conv2d(f_number * 3, f_number, kernel_size=1, bias=bias)

    def forward(self, x):
        # 自动检测色彩域
        color_space = self.auto_detect_color_space(x)
        if color_space == 'hvi':
            x = self.hvi_to_rgb(x)
        elif color_space == 'ycbcr':
            x = self.ycbcr_to_rgb(x)

        # 确保输入张量的通道数是 32
        if x.shape[1] != 32:
            raise ValueError(f"Input tensor channel size {x.shape[1]} does not match expected 32")

        attn = self.norm(x)
        _, _, h, w = attn.shape

        qkv = self.dwconv(self.pwconv(attn))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        out = self.feedforward(out + x)
        return out

    def auto_detect_color_space(self, x):
        """
        自动检测输入图像的色彩域。
        使用深度学习特征来判断图像更适合在哪个色彩域进行处理。
        """
        features = self.color_space_detector(x)  # 输出形状: (batch_size, 3)
        _, predicted = torch.max(features, dim=1)  # predicted 形状: (batch_size,)
        color_spaces = ['rgb', 'hvi', 'ycbcr']
        return color_spaces[predicted[0].item()]  # 取第一个样本的预测结果

    def dynamic_adjust_parameters(self, color_space):
        """
        根据检测到的色彩域动态调整参数。
        """
        if color_space == 'rgb':
            self.temperature.data = torch.ones(self.num_heads, 1, 1) * 0.5
        elif color_space == 'hvi':
            self.temperature.data = torch.ones(self.num_heads, 1, 1) * 0.7
        elif color_space == 'ycbcr':
            self.temperature.data = torch.ones(self.num_heads, 1, 1) * 0.9

    def multi_scale_process(self, x):
        scales = [1, 0.5, 0.25]
        features = []
        original_size = x.shape[2:]  # 获取原始输入张量的空间尺寸 (H, W)
        for scale in scales:
            scaled_x = F.interpolate(x, scale_factor=scale, mode='bilinear', align_corners=False)
            scaled_feature = self.dehaze_layers(scaled_x)
            # 将特征张量上采样到原始输入张量的空间尺寸
            scaled_feature = F.interpolate(scaled_feature, size=original_size, mode='bilinear', align_corners=False)
            features.append(scaled_feature)
        return torch.cat(features, dim=1)

    def hvi_to_rgb(self, x):
        """
        使用预训练的模型进行 HVI 到 RGB 的转换。
        """
        return self.hvi_to_rgb_model(x)

    def ycbcr_to_rgb(self, x):
        """
        使用预训练的模型进行 YCbCr 到 RGB 的转换。
        """
        return self.ycbcr_to_rgb_model(x)

class EnhancedColorSpaceDetector(nn.Module):
    def __init__(self):
        super(EnhancedColorSpaceDetector, self).__init__()
        self.conv1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # 全局平均池化，将空间维度降为 1x1
        self.fc = nn.Linear(256, 3)  # 输出 3 个类别（RGB、HVI、YCbCr）

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x)  # 输出形状: (batch_size, 256, 1, 1)
        x = x.view(x.size(0), -1)  # 展平为 (batch_size, 256)
        x = self.fc(x)  # 输出形状: (batch_size, 3)
        return x

class EnhancedHVItoRGBModel(nn.Module):
    def __init__(self):
        super(EnhancedHVItoRGBModel, self).__init__()
        self.conv1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x

class EnhancedYCbCrtoRGBModel(nn.Module):
    def __init__(self):
        super(EnhancedYCbCrtoRGBModel, self).__init__()
        self.conv1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return x * self.sigmoid(out)

