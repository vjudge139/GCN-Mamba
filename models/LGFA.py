import math
import torch
import torch.nn as nn
from monai.networks.blocks.segresnet_block import get_conv_layer
from monai.networks.layers.utils import get_act_layer, get_norm_layer
from models.utils import ConvMamba


class LGFA(nn.Module):
    def __init__(self, spatial_dims, in_channels, norm,
                 act=("RELU", {"inplace": True}), conv_mode="deepwise",
                 stride=1, out_channels=None):
        super().__init__()
        out_channels = out_channels or in_channels
        self.act   = get_act_layer(act)
        self.norm1 = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels)
        self.norm2 = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels)
        self.conv1 = ARMB(in_channels, in_channels, conv_mode=conv_mode)
        self.conv2 = ARMB(in_channels, out_channels, conv_mode=conv_mode)
        self.shortcut = (get_conv_layer(spatial_dims, in_channels, out_channels, kernel_size=1)
                         if out_channels != in_channels else nn.Identity())
        self.pool = (nn.MaxPool2d(kernel_size=stride, stride=stride)
                     if stride > 1 and spatial_dims == 2 else nn.Identity())

    def forward(self, x):
        out = self.conv2(self.act(self.norm2(self.conv1(self.act(self.norm1(x))))))
        return self.pool(out + self.shortcut(x))
    
class ARMB(nn.Module):
    def __init__(self, input_dim, output_dim, d_state=16, d_conv=4, expand=2, conv_mode="SevenBySeven"):
        super().__init__()
        self.output_dim = output_dim
        self.norm       = nn.LayerNorm(input_dim)
        self.mamba      = ConvMamba(d_model=input_dim, d_state=d_state, d_conv=d_conv,
                                     expand=expand, conv_mode=conv_mode)
        self.proj       = nn.Linear(input_dim, output_dim)
        self.skip_scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.float()
        B, C, *spatial = x.shape
        n = math.prod(spatial)
        s = x.reshape(B, C, n).transpose(-1, -2)
        out = self.norm(self.mamba(self.norm(s)) + self.skip_scale * s)
        return self.proj(out).transpose(-1, -2).reshape(B, self.output_dim, *spatial)