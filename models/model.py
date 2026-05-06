from __future__ import annotations
import torch
import torch.nn as nn
from monai.networks.blocks.segresnet_block import get_conv_layer
from monai.networks.layers.utils import get_act_layer
from monai.networks.layers.factories import Dropout
from monai.utils import UpsampleMode

from models.LGFA import LGFA
from models.SEM import SEM
from models.DPAGMF import DPAGMF
from models.utils import build_final_conv, build_upsample_block


class GCNMamba(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 2,
        init_filters: int = 16,
        conv_mode: str = "deepwise",
        stage: int = 4,
        mamba_act: str = "silu",
    ):
        super().__init__()
        spatial_dims = 2
        dropout_prob = None
        act = ("RELU", {"inplace": True})
        norm = ("GROUP", {"num_groups": 8})
        use_conv_final = True
        blocks_down = (1, 2, 2, 4)
        blocks_up = (1, 1, 1)
        up_conv_mode = "deepwise"
        residual = False
        diff_abs = "later"
        upsample_mode = UpsampleMode.NONTRAINABLE

        self.spatial_dims    = spatial_dims
        self.init_filters    = init_filters
        self.norm            = norm
        self.act             = act
        self.stage           = stage
        self.diff_abs        = diff_abs
        self.conv_mode       = conv_mode
        self.up_conv_mode    = up_conv_mode
        self.residual        = residual
        self.mamba_act       = mamba_act
        self.use_conv_final  = use_conv_final
        self.dropout_prob    = dropout_prob
        self.upsample_mode   = UpsampleMode(upsample_mode)
        self.blocks_down     = blocks_down
        self.blocks_up       = blocks_up

        ch = [init_filters * (2 ** i) for i in range(4)]
        self.act_mod = get_act_layer(act)
        self.stem    = get_conv_layer(spatial_dims, in_channels, init_filters)

        self.encoder_layers = nn.ModuleList()
        for i, n in enumerate(blocks_down):
            c    = init_filters * (2 ** i)
            down = (LGFA(spatial_dims, ch[i-1], norm, act, conv_mode,
                         stride=2, out_channels=c)
                    if i > 0 else nn.Identity())
            self.encoder_layers.append(nn.Sequential(
                down,
                *[LGFA(spatial_dims, c, norm, act, conv_mode) for _ in range(n)],
            ))

        self.sem_modules = nn.ModuleList([SEM(c, c) for c in ch])

        self.dpagmf = DPAGMF(
            channels_list=ch,
            residual=residual,
            act=mamba_act,
        )

        self.decoder_layers  = nn.ModuleList()
        self.upsample_layers = nn.ModuleList()
        for i in range(len(blocks_up)):
            c = init_filters * (2 ** (len(blocks_up) - i))
            self.upsample_layers.append(
                build_upsample_block(spatial_dims, c, self.upsample_mode))
            self.decoder_layers.append(nn.Sequential(*[
                LGFA(spatial_dims, c // 2, norm, act, up_conv_mode)
                for _ in range(blocks_up[i])
            ]))

        self.head = build_final_conv(norm, self.act_mod, spatial_dims, init_filters, out_channels)

        if dropout_prob is not None:
            self.dropout = Dropout[Dropout.DROPOUT, spatial_dims](dropout_prob)

    def encode(self, x: torch.Tensor):
        x = self.stem(x)
        if self.dropout_prob is not None:
            x = self.dropout(x)
        feats = []
        for stage in self.encoder_layers:
            x = stage(x)
            feats.append(x)
        return x, feats

    def decode(self, x: torch.Tensor, skips: list[torch.Tensor]) -> torch.Tensor:
        for up, dec, skip in zip(self.upsample_layers, self.decoder_layers, skips[1:]):
            x = dec(up(x) + skip)
        return self.head(x) if self.use_conv_final else x

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        _, feats1 = self.encode(x1)
        _, feats2 = self.encode(x2)

        diffs = []
        s1, s2 = None, None
        
        for i, (f1, f2) in enumerate(zip(feats1, feats2)):
            if self.diff_abs == "later" and i < self.stage:
                if i == 0:
                    s1 = self.sem_modules[i](f1)
                    s2 = self.sem_modules[i](f2)
                else:
                    if s1.shape[-2:] != f1.shape[-2:]:
                        s1 = torch.nn.functional.interpolate(s1, size=f1.shape[-2:], mode='bilinear', align_corners=False)
                        s2 = torch.nn.functional.interpolate(s2, size=f2.shape[-2:], mode='bilinear', align_corners=False)
                    s1 = self.sem_modules[i](f1, s1)
                    s2 = self.sem_modules[i](f2, s2)
        
                f1, f2 = self.dpagmf(s1, s2, stage_idx=i)
            diffs.append(torch.abs(f1 - f2))

        diffs.reverse()
        return self.decode(diffs[0], diffs)