from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from models.utils import LightweightModel, ssm

BN2d = nn.BatchNorm2d
BN1d = nn.BatchNorm1d

class SpatialGCN(nn.Module):
    def __init__(self, plane):
        super().__init__()
        inter = plane // 2
        self.node_k         = nn.Conv2d(plane,   inter,   1)
        self.node_v         = nn.Conv2d(plane,   inter,   1)
        self.node_v_y       = nn.Conv2d(plane*2, plane,   1)
        self.node_q         = nn.Conv2d(plane,   inter,   1)
        self.node_q_y       = nn.Conv2d(plane*2, plane,   1)
        self.conv_wg        = nn.Sequential(nn.Conv1d(inter,   inter,   1, bias=False), BN1d(inter))
        self.conv_wg_decode = nn.Sequential(nn.Conv1d(plane*2, plane*2, 1, bias=False), BN1d(plane*2))
        self.softmax        = nn.Softmax(dim=2)
        self.out            = nn.Sequential(nn.Conv2d(inter,   plane,   1), BN2d(plane))
        self.outdecode      = nn.Sequential(nn.Conv2d(inter*4, plane*2, 1), BN2d(plane*2))
        self.xpre           = nn.Sequential(
            nn.Conv2d(inter*4, inter*2, 1), BN2d(inter*2),
            nn.Conv2d(inter*2, plane*2, 1), BN2d(plane*2))

    def forward(self, x, y):
        if y is None:
            k, v, q = self.node_k(x), self.node_v(x), self.node_q(x)
        else:
            k, v, q = x, self.node_v_y(y), self.node_q_y(y)
        b, c, h, w = k.size()
        k  = k.reshape(b, c, -1).permute(0, 2, 1)
        q  = q.reshape(b, c, -1)
        v  = v.reshape(b, c, -1).permute(0, 2, 1)
        AV = torch.bmm(k, self.softmax(torch.bmm(q, v))).transpose(1, 2).contiguous()
        if y is None:
            return F.relu_(self.out(self.conv_wg(AV).view(b, c, h, -1)) + x)
        return F.relu_(self.outdecode(self.conv_wg_decode(AV).view(b, c, h, -1)) + self.xpre(x))

def MSCB(planes):
    layers = []
    for _ in range(3):
        layers += [nn.Conv2d(planes, planes, 3, groups=planes, stride=2, padding=1, bias=False),
                   BN2d(planes)]
    return nn.Sequential(*layers)

def _ssm_to_spatial(norm, gcn_ssm, x):
    out = gcn_ssm(norm(rearrange(x, "b c h w -> b (h w) c")))
    b, l, d = out.shape
    h = int(math.sqrt(l))
    return out.view(b, d, h, h)

class CFM(nn.Module):
    def __init__(self, planes, in_dim, expand=2, bias=False):
        super().__init__()
        self.d_inner           = int(expand * in_dim)
        self.gcn_local         = SpatialGCN(planes // 2)
        self.local             = MSCB(planes)
        self.localdecode       = MSCB(planes)
        self.norm1             = nn.LayerNorm(planes)
        self.gcn_ssm           = ssm(d_model=planes, bimamba_type="v2")
        self.local_relation    = nn.Sequential(
            LightweightModel(in_dim, in_dim), nn.SiLU(),
            LightweightModel(in_dim, self.d_inner // 2))
        self.act               = nn.SiLU()
        self.out_proj          = nn.Linear(self.d_inner, in_dim, bias=bias)
        self.expand_local      = nn.Linear(in_dim, self.d_inner, bias=bias)
        self.expand_featdecode = nn.Linear(in_dim, self.d_inner, bias=bias)

    def forward(self, feat, feat_other):
        proj  = _ssm_to_spatial(self.norm1, self.gcn_ssm, feat)
        local = F.interpolate(
            self.gcn_local(self.local(proj), self.localdecode(feat_other)),
            size=feat_other.shape[2:], mode="bilinear", align_corners=True)
        proj2  = _ssm_to_spatial(self.norm1, self.gcn_ssm, feat_other)
        fd_ssm = self.expand_featdecode(
            rearrange(proj2, "b d h w -> b (h w) d")).transpose(1, 2)
        local  = self.expand_local(
            rearrange(local, "b c h w -> b (h w) c")).transpose(1, 2)
        return F.linear(
            rearrange(fd_ssm * self.act(local), "b d l -> b l d"),
            self.out_proj.weight, self.out_proj.bias)

class SRM(nn.Module):
    def __init__(self, planes, in_dim, expand=2, bias=False):
        super().__init__()
        self.d_inner        = int(expand * in_dim)
        self.theta          = nn.Sequential(nn.Conv2d(planes, planes, 1, bias=False), BN2d(planes))
        self.thetadecode    = nn.Sequential(nn.Conv2d(planes, planes, 1, bias=False), BN2d(planes))
        self.conv_adj       = nn.Sequential(nn.Conv1d(planes, planes,   1, bias=False), BN1d(planes))
        self.conv_wg        = nn.Sequential(nn.Conv1d(planes, planes*2, 1, bias=False), BN1d(planes*2))
        self.norm1          = nn.LayerNorm(planes)
        self.gcn_ssm        = ssm(d_model=planes, bimamba_type="v2")
        self.local_relation = nn.Sequential(
            LightweightModel(in_dim, in_dim), nn.SiLU(),
            LightweightModel(in_dim, self.d_inner))
        self.act            = nn.SiLU()
        self.out_proj       = nn.Linear(self.d_inner, in_dim, bias=bias)

    def forward(self, feat, feat_other):
        proj  = _ssm_to_spatial(self.norm1, self.gcn_ssm, feat)
        x_sqz = proj.reshape(proj.shape[0], proj.shape[1], -1)
        if feat_other is not None:
            b_mat = self.thetadecode(feat_other)
            local_src = feat_other
        else:
            b_mat = self.theta(feat)
            local_src = feat
        b_mat = b_mat.reshape(b_mat.shape[0], b_mat.shape[1], -1)
        z = x_sqz * b_mat
        z = z + self.conv_adj(z)
        z = self.conv_wg(z)
        local = rearrange(self.local_relation(local_src), "b c h w -> b c (h w)")
        return F.linear(
            rearrange(z * self.act(local), "b d l -> b l d"),
            self.out_proj.weight, self.out_proj.bias)

class DualPathHead(nn.Module):
    def __init__(self, inplanes, interplanes, in_dim):
        super().__init__()
        self.conva = nn.Sequential(
            nn.Conv2d(inplanes, interplanes, 3, padding=1, bias=False),
            BN2d(interplanes), nn.ReLU())
        self.cfm = CFM(interplanes, in_dim)
        self.srm = SRM(interplanes, in_dim)

    def forward(self, x, feat_other, use_global):
        out = self.conva(x)
        return self.cfm(out, feat_other) if use_global else self.srm(out, feat_other)

class CFMFusion(nn.Module):
    def __init__(self, dim, residual=False, act="silu"):
        super().__init__()
        self.residual   = residual
        self.skip_scale = nn.Parameter(torch.ones(1))
        self.head       = DualPathHead(dim, dim, dim)

    def forward(self, x1, x2):
        b, c, h, w = x1.shape
        o1 = rearrange(self.head(x1, x2, use_global=True), "b (h w) c -> b c h w", h=h)
        o2 = rearrange(self.head(x2, x1, use_global=True), "b (h w) c -> b c h w", h=h)
        if self.residual:
            o1 = o1 + self.skip_scale * x1
            o2 = o2 + self.skip_scale * x2
        return o1, o2

class SRMFusion(nn.Module):
    def __init__(self, dim, residual=False, act="silu"):
        super().__init__()
        self.residual   = residual
        self.skip_scale = nn.Parameter(torch.ones(1))
        self.head       = DualPathHead(dim, dim, dim)

    def forward(self, x1, x2):
        b, c, h, w = x1.shape
        o1 = rearrange(self.head(x1, x2, use_global=False), "b (h w) c -> b c h w", h=h)
        o2 = rearrange(self.head(x2, x1, use_global=False), "b (h w) c -> b c h w", h=h)
        if self.residual:
            o1 = o1 + self.skip_scale * x1
            o2 = o2 + self.skip_scale * x2
        return o1, o2

class AdaptiveGate(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.gate    = nn.Linear(in_dim * 2, 2, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x_srm, x_cfm):
        g = self.softmax(self.gate(
            torch.cat([x_srm.mean(dim=[2, 3]), x_cfm.mean(dim=[2, 3])], dim=-1)))
        return g

class DPAGMF(nn.Module):
    def __init__(self, channels_list, residual=False, act="silu"):
        super().__init__()
        self.srm_fusion = nn.ModuleList([
            SRMFusion(c, residual, act) for c in channels_list])
        self.cfm_fusion = nn.ModuleList([
            CFMFusion(c, residual, act) for c in channels_list])
        self.gates = nn.ModuleList([
            AdaptiveGate(c) for c in channels_list])

    def forward(self, x1, x2, stage_idx):
        b = x1.shape[0]

        x1_srm, x2_srm = self.srm_fusion[stage_idx](x1, x2)
        x1_cfm, x2_cfm = self.cfm_fusion[stage_idx](x1, x2)

        g1 = self.gates[stage_idx](x1_srm, x1_cfm)
        g2 = self.gates[stage_idx](x2_srm, x2_cfm)

        def _blend(f_srm, f_cfm, g):
            w0 = g[:, 0].view(b, 1, 1, 1)
            w1 = g[:, 1].view(b, 1, 1, 1)
            return w0 * f_srm + w1 * f_cfm

        return _blend(x1_srm, x1_cfm, g1), _blend(x2_srm, x2_cfm, g2)