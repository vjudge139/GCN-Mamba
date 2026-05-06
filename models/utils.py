import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from monai.networks.blocks.segresnet_block import get_conv_layer, get_upsample_layer
from monai.networks.layers.utils import get_norm_layer

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn = causal_conv1d_update = None

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn_no_out_proj
except ImportError:
    selective_scan_fn = mamba_inner_fn_no_out_proj = None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

class LightweightModel(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.dw = nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels)
        self.pw = nn.Conv2d(in_channels, out_channels, 1)
    def forward(self, x):
        return self.pw(self.dw(x))

class SevenBySevenNorm(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv7x7 = nn.Conv2d(in_channels, out_channels, 7, padding=3)
        self.norm = nn.LayerNorm(out_channels)
        
    def forward(self, x):
        x = self.conv7x7(x)
        x = rearrange(x, "b d h w -> b h w d")
        x = self.norm(x)
        x = rearrange(x, "b h w d -> b d h w")
        return x

def build_local_relation(conv_mode, d_model, d_inner):
    dw  = lambda i, o: LightweightModel(i, o)
    c3  = lambda i, o: nn.Conv2d(i, o, 3, padding=1)
    act = nn.SiLU()
    configs = {
        "orignal":            [c3(d_model, d_model),          act, c3(d_model, d_inner)],
        "deepwise":           [dw(d_model, d_model),          act, dw(d_model, d_inner)],
        "deepwise_dinner":    [dw(d_model, d_inner),          act, dw(d_inner, d_inner)],
        "SevenBySeven":           [SevenBySevenNorm(d_model, d_inner), act],
    }
    if conv_mode not in configs:
        raise ValueError(f"Unknown conv_mode: {conv_mode!r}")
    return nn.Sequential(*configs[conv_mode])

class MambaSSM(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dt_rank="auto",
                 dt_min=0.001, dt_max=0.1, dt_init="random", dt_scale=1.0,
                 dt_init_floor=1e-4, conv_bias=True, bias=False, use_fast_path=True,
                 layer_idx=None, device=None, dtype=None, bimamba_type="v2",
                 conv_mode="deepwise", use_local=True):
        assert bimamba_type == "v2"
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model, self.d_state, self.d_conv = d_model, d_state, d_conv
        self.expand   = expand
        self.d_inner  = int(expand * d_model)
        self.dt_rank  = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.use_local = use_local
        self.activation = "silu"
        self.act = nn.SiLU()

        if use_local:
            self.local_relation = build_local_relation(conv_mode, d_model, self.d_inner)

        self.in_proj  = nn.Linear(d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.out_proj = nn.Linear(self.d_inner, d_model,     bias=bias, **factory_kwargs)

        def _conv1d():
            return nn.Conv1d(self.d_inner, self.d_inner, bias=conv_bias,
                             kernel_size=d_conv, groups=self.d_inner,
                             padding=d_conv - 1, **factory_kwargs)
        self.conv1d, self.conv1d_b = _conv1d(), _conv1d()

        def _proj_pair():
            x  = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False, **factory_kwargs)
            dt = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)
            dt_std = self.dt_rank ** -0.5 * dt_scale
            nn.init.uniform_(dt.weight, -dt_std, dt_std) if dt_init != "constant" \
                else nn.init.constant_(dt.weight, dt_std)
            inv_dt = (lambda d: d + torch.log(-torch.expm1(-d)))(
                torch.exp(torch.rand(self.d_inner, **factory_kwargs)
                          * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
                          ).clamp(min=dt_init_floor))
            with torch.no_grad():
                dt.bias.copy_(inv_dt)
            dt.bias._no_reinit = True
            return x, dt
        self.x_proj,   self.dt_proj   = _proj_pair()
        self.x_proj_b, self.dt_proj_b = _proj_pair()

        def _A_log():
            A = repeat(torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
                       "n -> d n", d=self.d_inner).contiguous()
            p = nn.Parameter(torch.log(A)); p._no_weight_decay = True; return p
        self.A_log, self.A_b_log = _A_log(), _A_log()

        def _D():
            p = nn.Parameter(torch.ones(self.d_inner, device=device))
            p._no_weight_decay = True; return p
        self.D, self.D_b = _D(), _D()

    def forward(self, hidden_states, inference_params=None):
        batch, seqlen, _ = hidden_states.shape
        h = int(math.sqrt(seqlen))
        if self.use_local:
            local = rearrange(
                self.local_relation(rearrange(hidden_states, "b (h w) d -> b d h w", h=h)),
                "b d h w -> b d (h w)")
        if inference_params is not None:
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
            if inference_params.seqlen_offset > 0:
                out, _, _ = self.step(hidden_states, conv_state, ssm_state)
                return out
        xz = rearrange(self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
                       "d (b l) -> b d l", l=seqlen)
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")
        A, A_b = -torch.exp(self.A_log.float()), -torch.exp(self.A_b_log.float())
        if self.use_fast_path and inference_params is None:
            def _scan(xz_, conv_, xp_, dtp_, A_, D_):
                return mamba_inner_fn_no_out_proj(
                    xz_, conv_.weight, conv_.bias, xp_.weight, dtp_.weight,
                    A_, None, None, D_.float(),
                    delta_bias=dtp_.bias.float(), delta_softplus=True)
            combined = (_scan(xz, self.conv1d, self.x_proj, self.dt_proj, A, self.D)
                      + _scan(xz.flip([-1]), self.conv1d_b, self.x_proj_b,
                               self.dt_proj_b, A_b, self.D_b).flip([-1]))
            if self.use_local:
                combined = combined + local
            return F.linear(rearrange(combined, "b d l -> b l d"),
                            self.out_proj.weight, self.out_proj.bias)
        else:
            x, z = xz.chunk(2, dim=1)
            x = (self.act(self.conv1d(x)[..., :seqlen]) if causal_conv1d_fn is None
                 else causal_conv1d_fn(x, rearrange(self.conv1d.weight, "d 1 w -> d w"),
                                       self.conv1d.bias, self.activation))
            x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
            dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            dt = rearrange(self.dt_proj.weight @ dt.t(), "d (b l) -> b d l", l=seqlen)
            B  = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            C  = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            y  = selective_scan_fn(x, dt, A, B, C, self.D.float(), z=z,
                                   delta_bias=self.dt_proj.bias.float(),
                                   delta_softplus=True, return_last_state=False)
            return self.out_proj(rearrange(y, "b d l -> b l d"))

    def step(self, hidden_states, conv_state, ssm_state):
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1
        x, z = self.in_proj(hidden_states.squeeze(1)).chunk(2, dim=-1)
        if causal_conv1d_update is None:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))
            conv_state[:, :, -1] = x
            x = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)
            if self.conv1d.bias is not None:
                x = x + self.conv1d.bias
            x = self.act(x).to(dtype=dtype)
        else:
            x = causal_conv1d_update(x, conv_state,
                                     rearrange(self.conv1d.weight, "d 1 w -> d w"),
                                     self.conv1d.bias, self.activation)
        dt, B, C = torch.split(self.x_proj(x), [self.dt_rank, self.d_state, self.d_state], dim=-1)
        A = -torch.exp(self.A_log.float())
        if selective_state_update is None:
            dt = F.softplus(F.linear(dt, self.dt_proj.weight) + self.dt_proj.bias.to(dtype=dt.dtype))
            ssm_state.copy_(ssm_state * torch.exp(torch.einsum("bd,dn->bdn", dt, A))
                            + rearrange(x, "b d -> b d 1") * torch.einsum("bd,bn->bdn", dt, B))
            y = torch.einsum("bdn,bn->bd", ssm_state.to(dtype), C) + self.D.to(dtype) * x
            y = y * self.act(z)
        else:
            y = selective_state_update(ssm_state, x, F.linear(dt, self.dt_proj.weight),
                                       A, B, C, self.D, z=z,
                                       dt_bias=self.dt_proj.bias, dt_softplus=True)
        return self.out_proj(y).unsqueeze(1), conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen=None, dtype=None, **kwargs):
        dev = self.out_proj.weight.device
        return (torch.zeros(batch_size, self.d_inner, self.d_conv, device=dev,
                            dtype=self.conv1d.weight.dtype if dtype is None else dtype),
                torch.zeros(batch_size, self.d_inner, self.d_state, device=dev,
                            dtype=self.dt_proj.weight.dtype if dtype is None else dtype))

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            inference_params.key_value_memory_dict[self.layer_idx] = \
                self.allocate_inference_cache(batch_size)
        states = inference_params.key_value_memory_dict[self.layer_idx]
        if initialize_states:
            states[0].zero_(); states[1].zero_()
        return states

def ssm(d_model, **kw):
    return MambaSSM(d_model, use_local=False, **kw)

def ConvMamba(d_model, **kw):
    class SimpleConvMamba(nn.Module):
        def __init__(self, d_model, **kw):
            super().__init__()
            self.mamba = MambaSSM(d_model, use_local=True, **kw)
            self.linear = nn.Linear(d_model, d_model)
            self.mlp = mlp(d_model)
            
        def forward(self, x):
            x = self.mamba(x)
            x = self.linear(x)
            x = self.mlp(x)
            return x
    
    return SimpleConvMamba(d_model, **kw)

def build_final_conv(norm, act_mod, spatial_dims, in_channels, out_channels):
    return nn.Sequential(
        get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels),
        act_mod,
        get_conv_layer(spatial_dims, in_channels, out_channels, kernel_size=1, bias=True),
    )

def build_upsample_block(spatial_dims, in_channels, upsample_mode):
    return nn.Sequential(
        get_conv_layer(spatial_dims, in_channels, in_channels // 2, kernel_size=1),
        get_upsample_layer(spatial_dims, in_channels // 2, upsample_mode=upsample_mode),
    )

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, relu=True, bn=True, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation,
                              groups=groups, bias=bias)
        self.bn   = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn   is not None: x = self.bn(x)
        if self.relu is not None: x = self.relu(x)
        return x
    
def mlp(d_model, mlp_ratio=4):
    hidden_dim = int(d_model * mlp_ratio)
    return nn.Sequential(
        nn.Linear(d_model, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, d_model)
    )
