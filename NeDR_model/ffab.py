
import os
import torch
import torch.nn as nn
from functools import partial
import math
from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_, DropPath, to_2tuple
# 替换旧的导入路径
# from timm.models import create_model  # 替换旧的 timm.models.registry
# from timm.layers import *  # 替换旧的 timm.models.layers

class FEB(nn.Module):
    def __init__(self, nc):
        super(FEB, self).__init__()
        self.fpre = nn.Conv2d(nc, nc, 1, 1, 0)
        self.process1 = nn.Sequential(
            nn.Conv2d(nc, nc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nc, nc, 1, 1, 0))
        self.process2 = nn.Sequential(
            nn.Conv2d(nc, nc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nc, nc, 1, 1, 0))

    def forward(self, x):
        _, _, H, W = x.shape
        x_freq = torch.fft.rfft2(self.fpre(x), norm='backward')
        mag = torch.abs(x_freq)
        pha = torch.angle(x_freq)
        mag = self.process1(mag)
        pha = self.process2(pha)
        real = mag * torch.cos(pha)
        imag = mag * torch.sin(pha)
        x_out = torch.complex(real, imag)
        x_out = torch.fft.irfft2(x_out, s=(H, W), norm='backward')

        return x_out + x


class ProcessBlock(nn.Module):
    def __init__(self, in_nc):
        super(ProcessBlock, self).__init__()
        self.spatial_process = nn.Identity()
        self.frequency_process = FEB(in_nc)
        self.cat = nn.Conv2d(in_nc, in_nc, 1, 1, 0)

    def forward(self, x):
        xori = x
        x_freq = self.frequency_process(x)
        # x_spatial = self.spatial_process(x)
        # xcat = torch.cat([x_spatial,x_freq],1)
        x_out = self.cat(x_freq)

        return x_out + xori



class FFAB(nn.Module):
    def __init__(self, nc):
        super(FFAB, self).__init__()

        self.conv0 = nn.Sequential(
            nn.Conv2d(nc, nc, 1, 1, 0),
            ProcessBlock(nc),
        )
        self.conv1 = ProcessBlock(nc)
        self.conv2 = ProcessBlock(nc)
        self.conv3 = ProcessBlock(nc)
        self.conv4 = nn.Sequential(
            ProcessBlock(nc * 2),
            nn.Conv2d(nc * 2, nc, 1, 1, 0),
        )
        self.conv5 = nn.Sequential(
            ProcessBlock(nc * 2),
            nn.Conv2d(nc * 2, nc, 1, 1, 0),
        )
        self.convout = nn.Sequential(
            ProcessBlock(nc * 2),
            nn.Conv2d(nc * 2, nc, 1, 1, 0),
        )

    def forward(self, x):
        x = self.conv0(x)
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(torch.cat((x2, x3), dim=1))
        x5 = self.conv5(torch.cat((x1, x4), dim=1))
        xout = self.convout(torch.cat((x, x5), dim=1))

        return xout