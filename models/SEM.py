from __future__ import annotations
import math
import torch
import torch.nn as nn
from einops import rearrange
from models.utils import ssm, BasicConv

class SEM(nn.Module):
    def __init__(self, in_planes, out_planes, scale=0.1, map_reduce=8):
        super().__init__()
        self.scale = scale
        inter = in_planes // map_reduce
        self.branch0 = nn.Sequential(
            BasicConv(in_planes, 2*inter, kernel_size=1),
            BasicConv(2*inter, 2*inter, kernel_size=3, padding=1, relu=False))
        self.branch1 = nn.Sequential(
            BasicConv(in_planes, inter, kernel_size=1),
            BasicConv(inter, (inter//2)*3, kernel_size=(1,3), padding=(0,1)),
            BasicConv((inter//2)*3, 2*inter, kernel_size=(3,1), padding=(1,0)),
            BasicConv(2*inter, 2*inter, kernel_size=3, padding=5, dilation=5, relu=False))
        self.branch2 = nn.Sequential(
            BasicConv(in_planes, inter, kernel_size=1),
            BasicConv(inter, (inter//2)*3, kernel_size=(3,1), padding=(1,0)),
            BasicConv((inter//2)*3, 2*inter, kernel_size=(1,3), padding=(0,1)),
            BasicConv(2*inter, 2*inter, kernel_size=3, padding=5, dilation=5, relu=False))
        self.scale_factor = nn.Parameter(torch.ones(1, in_planes, 1, 1))
        self.ConvLinear = BasicConv(6*inter + in_planes, out_planes, kernel_size=1, relu=False)
        self.shortcut   = BasicConv(in_planes, out_planes, kernel_size=1, relu=False)
        self.relu = nn.ReLU(inplace=False)
        self.norm = nn.LayerNorm(in_planes)
        self.ssm  = ssm(d_model=in_planes, bimamba_type="v2")
        self.ffn = nn.Sequential(
            nn.Conv1d(in_planes, in_planes * 4, kernel_size=1),
            nn.Conv1d(in_planes * 4, in_planes * 4, kernel_size=3, padding=1, groups=in_planes * 4),
            nn.SiLU(inplace=True),
            nn.Conv1d(in_planes * 4, in_planes, kernel_size=1),
            nn.Dropout(p=0.1)
        )

    def forward(self, x, y=None):
        if y is None:
            branch3 = x * self.scale_factor
            out = torch.cat([self.branch0(x), self.branch1(x), self.branch2(x), branch3], dim=1)
            out = self.relu(self.ConvLinear(out) * self.scale + self.shortcut(x))
            out = self.norm(rearrange(out, "b d h w -> b (h w) d"))
            out = self.ssm(out)
            out = rearrange(out, "b l d -> b d l")
            out = self.ffn(out)
            out = rearrange(out, "b d l -> b l d")
            b, l, d = out.shape
            h = int(math.sqrt(l))
            return rearrange(out, "b (h w) d -> b d h w", h=h)
        branch3 = x * self.scale_factor
        out = torch.cat([self.branch0(x), self.branch1(x), self.branch2(x), branch3], dim=1)
        out = self.relu(self.ConvLinear(out) * self.scale + self.shortcut(x))
        out = self.norm(rearrange(out, "b d h w -> b (h w) d"))
        x_ssm = self.ssm(out)
        y_reshaped = rearrange(y, "b d h w -> b (h w) d")
        if x_ssm.size(-1) != y_reshaped.size(-1):
            adjust_conv = nn.Linear(y_reshaped.size(-1), x_ssm.size(-1)).to(y_reshaped.device)
            y_reshaped = adjust_conv(y_reshaped)
        
        combined = x_ssm + y_reshaped
        combined = rearrange(combined, "b l d -> b d l")
        combined = self.ffn(combined)
        combined = rearrange(combined, "b d l -> b l d")
        b, l, d = combined.shape
        h = int(math.sqrt(l))
        return rearrange(combined, "b (h w) d -> b d h w", h=h)