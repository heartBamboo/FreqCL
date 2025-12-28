import torch
import torch.nn as nn
from mpmath import sqrtm
from timm.models.registry import register_model
import math
from timm.models.layers import trunc_normal_, DropPath, LayerNorm2d
# from timm.models._builder import resolve_pretrained_cfg
#
# try:
#     from timm.models._builder import _update_default_kwargs as update_args
# except:
#     from timm.models._builder import _update_default_model_kwargs as update_args

import sys
import os
# 获取当前脚本所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 添加上级目录（项目根目录）到 PYTHONPATH
sys.path.append(os.path.join(current_dir, '..'))  # 或者改成实际路径

from timm.models.vision_transformer import Mlp, PatchEmbed
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model
import torch.nn.functional as F
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from einops import rearrange, repeat
# from .registry import register_pip_model
from pathlib import Path
from DWT_IDWT.DWT_IDWT_layer import *
from utils.replay import ReplayComponent




def _cfg(url='', **kwargs):
    return {'url': url,
            'num_classes': 1000,
            'input_size': (3, 224, 224),
            'pool_size': None,
            'crop_pct': 0.875,
            'interpolation': 'bicubic',
            'fixed_input_size': True,
            'mean': (0.485, 0.456, 0.406),
            'std': (0.229, 0.224, 0.225),
            **kwargs
            }


def window_partition(x, window_size):
    """
    Args:
        x: (B, C, H, W)
        window_size: window size
        h_w: Height of window
        w_w: Width of window
    Returns:
        local window features (num_windows*B, window_size*window_size, C)
    """
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    windows = x.permute(0, 2, 4, 3, 5, 1).reshape(-1, window_size * window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: local window features (num_windows*B, window_size, window_size, C)
        window_size: Window size
        H: Height of image
        W: Width of image
    Returns:
        x: (B, C, H, W)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))

    x = windows.reshape(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 5, 1, 3, 2, 4).reshape(B, windows.shape[2], H, W)
    return x


def _load_state_dict(module, state_dict, strict=False, logger=None):
    """Load state_dict to a module.

    This method is modified from :meth:`torch.nn.Module.load_state_dict`.
    Default value for ``strict`` is set to ``False`` and the message for
    param mismatch will be shown even if strict is False.

    Args:
        module (Module): Module that receives the state_dict.
        state_dict (OrderedDict): Weights.
        strict (bool): whether to strictly enforce that the keys
            in :attr:`state_dict` match the keys returned by this module's
            :meth:`~torch.nn.Module.state_dict` function. Default: ``False``.
        logger (:obj:`logging.Logger`, optional): Logger to log the error
            message. If not specified, print function will be used.
    """
    unexpected_keys = []
    all_missing_keys = []
    err_msg = []

    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(state_dict, prefix, local_metadata, True,
                                     all_missing_keys, unexpected_keys,
                                     err_msg)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(module)
    load = None
    missing_keys = [
        key for key in all_missing_keys if 'num_batches_tracked' not in key
    ]

    if unexpected_keys:
        err_msg.append('unexpected key in source '
                       f'state_dict: {", ".join(unexpected_keys)}\n')
    if missing_keys:
        err_msg.append(
            f'missing keys in source state_dict: {", ".join(missing_keys)}\n')

    if len(err_msg) > 0:
        err_msg.insert(
            0, 'The model and loaded state dict do not match exactly\n')
        err_msg = '\n'.join(err_msg)
        if strict:
            raise RuntimeError(err_msg)
        elif logger is not None:
            logger.warning(err_msg)
        else:
            print(err_msg)


def _load_checkpoint(model,
                     filename,
                     map_location='cpu',
                     strict=False,
                     logger=None):
    """Load checkpoint from a file or URI.

    Args:
        model (Module): Module to load checkpoint.
        filename (str): Accept local filepath, URL, ``torchvision://xxx``,
            ``open-mmlab://xxx``. Please refer to ``docs/model_zoo.md`` for
            details.
        map_location (str): Same as :func:`torch.load`.
        strict (bool): Whether to allow different params for the model and
            checkpoint.
        logger (:mod:`logging.Logger` or None): The logger for error message.

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """
    checkpoint = torch.load(filename, map_location=map_location)
    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            f'No state_dict found in checkpoint file {filename}')
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}

    if sorted(list(state_dict.keys()))[0].startswith('encoder'):
        state_dict = {k.replace('encoder.', ''): v for k, v in state_dict.items() if k.startswith('encoder.')}

    _load_state_dict(model, state_dict, strict, logger)
    return checkpoint


class Downsample(nn.Module):
    """
    Down-sampling block"
    """

    def __init__(self,
                 dim,
                 keep_dim=False,
                 ):
        """
        Args:
            dim: feature size dimension.
            norm_layer: normalization layer.
            keep_dim: bool argument for maintaining the resolution.
        """

        super().__init__()
        if keep_dim:
            dim_out = dim
        else:
            dim_out = 2 * dim
        self.reduction = nn.Sequential(
            nn.Conv2d(dim, dim_out, 3, 2, 1, bias=False),
        )

    def forward(self, x):
        x = self.reduction(x)
        return x


class PatchEmbed(nn.Module):
    """
    Patch embedding block"
    """

    def __init__(self, in_chans=3, in_dim=64, dim=96):
        """
        Args:
            in_chans: number of input channels.
            dim: feature size dimension.
        """
        # in_dim = 1
        super().__init__()
        self.proj = nn.Identity()
        self.conv_down = nn.Sequential(
            nn.Conv2d(in_chans, in_dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(in_dim, eps=1e-4),
            nn.ReLU(),
            nn.Conv2d(in_dim, dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(dim, eps=1e-4),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.proj(x)
        x = self.conv_down(x)
        return x


class DepthWiseConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        # 这一行千万不要忘记
        super(DepthWiseConv, self).__init__()

        # 逐通道卷积
        self.depth_conv = nn.Conv2d(in_channels=in_channel,
                                    out_channels=in_channel,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    groups=in_channel)
        #  groups是一个数，当groups=in_channel时,表示做逐通道卷积

        # 逐点卷积
        self.point_conv = nn.Conv2d(in_channels=in_channel,
                                    out_channels=out_channel,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1)
    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out


class ConvBlock(nn.Module):

    def __init__(self, dim,
                 drop_path=0.,
                 layer_scale=None,
                 kernel_size=3):
        super().__init__()

        # self.conv1 = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=1)
        self.conv1 = DepthWiseConv(dim, dim)
        self.norm1 = nn.BatchNorm2d(dim, eps=1e-5)
        self.act1 = nn.GELU(approximate='tanh')
        # self.conv2 = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=1)
        self.conv2 = DepthWiseConv(dim, dim)
        self.norm2 = nn.BatchNorm2d(dim, eps=1e-5)
        self.layer_scale = layer_scale
        if layer_scale is not None and type(layer_scale) in [int, float]:
            self.gamma = nn.Parameter(layer_scale * torch.ones(dim))
            self.layer_scale = True
        else:
            self.layer_scale = False
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        if self.layer_scale:
            x = x * self.gamma.view(1, -1, 1, 1)
        x = input + self.drop_path(x)
        return x


class MambaVisionMixer(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=4,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            conv_bias=True,
            bias=False,
            use_fast_path=True,
            layer_idx=None,
            device=None,
            dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)
        self.x_proj = nn.Linear(
            self.d_inner // 2, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner // 2, bias=True, **factory_kwargs)
        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(
            torch.rand(self.d_inner // 2, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner // 2,
        ).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.d_inner // 2, device=device))
        self.D._no_weight_decay = True
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.conv1d_x = nn.Conv1d(
            in_channels=self.d_inner // 2,
            out_channels=self.d_inner // 2,
            bias=conv_bias // 2,
            kernel_size=d_conv,
            groups=self.d_inner // 2,
            **factory_kwargs,
        )
        self.conv1d_z = nn.Conv1d(
            in_channels=self.d_inner // 2,
            out_channels=self.d_inner // 2,
            bias=conv_bias // 2,
            kernel_size=d_conv,
            groups=self.d_inner // 2,
            **factory_kwargs,
        )

    def forward(self, hidden_states):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        _, seqlen, _ = hidden_states.shape
        xz = self.in_proj(hidden_states)
        xz = rearrange(xz, "b l d -> b d l")
        x, z = xz.chunk(2, dim=1)
        A = -torch.exp(self.A_log.float())
        x = F.silu(F.conv1d(input=x, weight=self.conv1d_x.weight, bias=self.conv1d_x.bias, padding='same',
                            groups=self.d_inner // 2))
        z = F.silu(F.conv1d(input=z, weight=self.conv1d_z.weight, bias=self.conv1d_z.bias, padding='same',
                            groups=self.d_inner // 2))
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = rearrange(self.dt_proj(dt), "(b l) d -> b d l", l=seqlen)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        y = selective_scan_fn(x,
                              dt,
                              A,
                              B,
                              C,
                              self.D.float(),
                              z=None,
                              delta_bias=self.dt_proj.bias.float(),
                              delta_softplus=True,
                              return_last_state=None)

        y = torch.cat([y, z], dim=1)
        y = rearrange(y, "b d l -> b l d")
        out = self.out_proj(y)
        return out



class Attention(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = True

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
#
#
# # PolaLinearAttention
# class Attention(nn.Module):
#     def __init__(self, dim, window_divisor=4, num_heads=8, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,
#                  kernel_size=5, alpha=4):
#         super().__init__()
#         self.dim = dim
#         self.window_divisor = window_divisor
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.head_dim = head_dim
#
#         self.qkvg = nn.Linear(dim, dim * 4, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)
#
#         self.dwc = nn.Conv2d(in_channels=head_dim, out_channels=head_dim, kernel_size=kernel_size,
#                              groups=head_dim, padding=kernel_size // 2)
#
#         self.power = nn.Parameter(torch.zeros(size=(1, self.num_heads, 1, self.head_dim)))
#         self.alpha = alpha
#
#         self.scale = nn.Parameter(torch.zeros(size=(1, 1, dim)))
#
#         # 注册缓冲区以确保 positional_encoding 在正确的设备上
#         self.register_buffer('positional_encoding', torch.zeros(1, 1, dim))
#
#     def compute_positional_encoding(self, window_size, dim, device):
#         # 根据窗口大小动态生成位置编码，并确保在正确的设备上
#         Wh, Ww = window_size
#         if Wh == 0 or Ww == 0:
#             raise ValueError(f"Invalid window size: {window_size}")
#         positional_encoding = torch.zeros(size=(1, Wh * Ww, dim), device=device)
#         return positional_encoding
#
#     def forward(self, x, H=None, W=None, mask=None):
#         B, N, C = x.shape
#
#         # 获取当前设备
#         device = x.device
#
#         # 动态计算 H 和 W
#         if H is None or W is None:
#             assert int(N ** 0.5) ** 2 == N, "N must be a perfect square if H and W are not provided"
#             H = W = int(N ** 0.5)
#
#         # 验证 H 和 W 是否有效
#         if H <= 0 or W <= 0:
#             raise ValueError(f"Invalid spatial dimensions: H={H}, W={W}")
#
#         # 动态调整窗口大小
#         window_size = (max(1, H // self.window_divisor), max(1, W // self.window_divisor))
#         positional_encoding = self.compute_positional_encoding(window_size, C, device)
#
#         # 动态调整 positional_encoding 的形状
#         positional_encoding = positional_encoding[:, :N, :]
#         qkv = self.qkvg(x).reshape(B, N, 4, C).permute(2, 0, 1, 3)
#         q, k, v, g = qkv.unbind(0)
#         k = k + positional_encoding  # 确保形状匹配
#
#         # 后续逻辑保持不变
#         scale = nn.Softplus()(self.scale)
#         power = 1 + self.alpha * nn.functional.sigmoid(self.power)
#
#         q = q / scale
#         k = k / scale
#         q = q.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3).contiguous()
#         k = k.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3).contiguous()
#         v = v.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3).contiguous()
#
#         q_pos = nn.ReLU()(q) ** power
#         q_neg = nn.ReLU()(-q) ** power
#         k_pos = nn.ReLU()(k) ** power
#         k_neg = nn.ReLU()(-k) ** power
#
#         q_sim = torch.cat([q_pos, q_neg], dim=-1)
#         q_opp = torch.cat([q_neg, q_pos], dim=-1)
#         k = torch.cat([k_pos, k_neg], dim=-1)
#
#         v1, v2 = torch.chunk(v, 2, dim=-1)
#
#         z = 1 / (q_sim @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)
#         kv = (k.transpose(-2, -1) * (N ** -0.5)) @ (v1 * (N ** -0.5))
#         x_sim = q_sim @ kv * z
#         z = 1 / (q_opp @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)
#         kv = (k.transpose(-2, -1) * (N ** -0.5)) @ (v2 * (N ** -0.5))
#         x_opp = q_opp @ kv * z
#
#         x = torch.cat([x_sim, x_opp], dim=-1)
#         x = x.transpose(1, 2).reshape(B, N, C)
#
#         H = W = int(N ** 0.5)
#         v = v.reshape(B * self.num_heads, H, W, -1).permute(0, 3, 1, 2)
#         v = self.dwc(v).reshape(B, C, N).permute(0, 2, 1)
#
#         x = x + v
#         x = x * g
#
#         x = self.proj(x)
#         x = self.proj_drop(x)
#
#         return x
#
#     def eval(self):
#         super().eval()
#         print('eval')
#
#     def extra_repr(self) -> str:
#         return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'



class Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 counter,
                 transformer_blocks,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=False,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 Mlp_block=Mlp,
                 layer_scale=None,
                 ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        if counter in transformer_blocks:
            self.mixer = Attention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_norm=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
                norm_layer=norm_layer,
            )
            # self.mixer = Attention(
            #     dim,
            #     window_divisor=4,
            #     num_heads=num_heads,
            #     qkv_bias=qkv_bias,
            #     qk_scale=qk_scale,
            #     attn_drop=attn_drop,
            #     proj_drop=drop,
            #     alpha=1.9, # 目前1.9最佳 67.79% acc
            #     kernel_size=5,
            # )
        else:
            self.mixer = MambaVisionMixer(d_model=dim,
                                          d_state=8,
                                          d_conv=3,
                                          expand=1
                                          )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        use_layer_scale = layer_scale is not None and type(layer_scale) in [int, float]
        self.gamma_1 = nn.Parameter(layer_scale * torch.ones(dim)) if use_layer_scale else 1
        self.gamma_2 = nn.Parameter(layer_scale * torch.ones(dim)) if use_layer_scale else 1

    def forward(self, x):
        x = x + self.drop_path(self.gamma_1 * self.mixer(self.norm1(x)))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class MambaVisionLayer(nn.Module):
    """
    MambaVision layer"
    """

    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 window_size,
                 conv=False,
                 downsample=True,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 layer_scale=None,
                 layer_scale_conv=None,
                 transformer_blocks=[],
                 ):
        """
        Args:
            dim: feature size dimension.
            depth: number of layers in each stage.
            window_size: window size in each stage.
            conv: bool argument for conv stage flag.
            downsample: bool argument for down-sampling.
            mlp_ratio: MLP ratio.
            num_heads: number of heads in each stage.
            qkv_bias: bool argument for query, key, value learnable bias.
            qk_scale: bool argument to scaling query, key.
            drop: dropout rate.
            attn_drop: attention dropout rate.
            drop_path: drop path rate.
            norm_layer: normalization layer.
            layer_scale: layer scaling coefficient.
            layer_scale_conv: conv layer scaling coefficient.
            transformer_blocks: list of transformer blocks.
        """

        super().__init__()
        self.conv = conv
        self.transformer_block = False
        if conv:
            self.blocks = nn.ModuleList([ConvBlock(dim=dim,
                                                   drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                                   layer_scale=layer_scale_conv)
                                         for i in range(depth)])
            self.transformer_block = False
        else:
            self.blocks = nn.ModuleList([Block(dim=dim,
                                               counter=i,
                                               transformer_blocks=transformer_blocks,
                                               num_heads=num_heads,
                                               mlp_ratio=mlp_ratio,
                                               qkv_bias=qkv_bias,
                                               qk_scale=qk_scale,
                                               drop=drop,
                                               attn_drop=attn_drop,
                                               drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                               layer_scale=layer_scale)
                                         for i in range(depth)])
            self.transformer_block = True

        self.downsample = None if not downsample else Downsample(dim=dim)
        self.do_gt = False
        self.window_size = window_size

    def forward(self, x):
        _, _, H, W = x.shape

        if self.transformer_block:
            pad_r = (self.window_size - W % self.window_size) % self.window_size
            pad_b = (self.window_size - H % self.window_size) % self.window_size
            if pad_r > 0 or pad_b > 0:
                x = torch.nn.functional.pad(x, (0, pad_r, 0, pad_b))
                _, _, Hp, Wp = x.shape
            else:
                Hp, Wp = H, W
            Hp, Wp = H, W
            x = window_partition(x, self.window_size)

        for _, blk in enumerate(self.blocks):
            x = blk(x)
        if self.transformer_block:
            x = window_reverse(x, self.window_size, Hp, Wp)
            if pad_r > 0 or pad_b > 0:
                x = x[:, :, :H, :W].contiguous()
        if self.downsample is None:
            return x
        return self.downsample(x)


class MambaVision(nn.Module):
    """
    MambaVision,
    """

    def __init__(self,
                 dim,
                 in_dim,
                 depths,
                 window_size,
                 mlp_ratio,
                 num_heads,
                 drop_path_rate=0.2,
                 in_chans=3,
                 num_classes=1000,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 layer_scale=None,
                 layer_scale_conv=None,
                 **kwargs):
        """
        Args:
            dim: feature size dimension.
            depths: number of layers in each stage.
            window_size: window size in each stage.
            mlp_ratio: MLP ratio.
            num_heads: number of heads in each stage.
            drop_path_rate: drop path rate.
            in_chans: number of input channels.
            num_classes: number of classes.
            qkv_bias: bool argument for query, key, value learnable bias.
            qk_scale: bool argument to scaling query, key.
            drop_rate: dropout rate.
            attn_drop_rate: attention dropout rate.
            norm_layer: normalization layer.
            layer_scale: layer scaling coefficient.
            layer_scale_conv: conv layer scaling coefficient.
        """
        super().__init__()
        num_features = int(dim * 2 ** (len(depths) - 1))
        self.num_classes = num_classes
        self.patch_embed = PatchEmbed(in_chans=in_chans, in_dim=in_dim, dim=dim)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.levels = nn.ModuleList()
        for i in range(len(depths)):
            conv = True if (i == 0 or i == 1) else False
            level = MambaVisionLayer(dim=int(dim * 2 ** i),
                                     depth=depths[i],
                                     num_heads=num_heads[i],
                                     window_size=window_size[i],
                                     mlp_ratio=mlp_ratio,
                                     qkv_bias=qkv_bias,
                                     qk_scale=qk_scale,
                                     conv=conv,
                                     drop=drop_rate,
                                     attn_drop=attn_drop_rate,
                                     drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                                     downsample=(i < 3),
                                     layer_scale=layer_scale,
                                     layer_scale_conv=layer_scale_conv,
                                     transformer_blocks=list(range(depths[i] // 2 + 1, depths[i])) if depths[
                                                                                                          i] % 2 != 0 else list(
                                         range(depths[i] // 2, depths[i])),
                                     )
            self.levels.append(level)
        self.norm = nn.BatchNorm2d(num_features)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, LayerNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'rpb'}

    def forward_features(self, x):
        x = self.patch_embed(x)
        for level in self.levels:
            x = level(x)
        x = self.norm(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        # x = self.head(x)
        return x

    def _load_state_dict(self,
                         pretrained,
                         strict: bool = False):
        _load_checkpoint(self,
                         pretrained,
                         strict=strict)


def loss_cluster(embeds, num_classes, device):
    import numpy as np
    from sklearnex.cluster import KMeans
    from sklearn.metrics import silhouette_score
    if isinstance(embeds, torch.Tensor):
        embeds = embeds.detach().cpu().numpy()
    kmeans = KMeans(n_clusters=num_classes, random_state=0).fit(embeds)
    labels = kmeans.labels_
    if len(np.unique(labels)) > 1:
       loss_cluster = silhouette_score(embeds, labels)
    else:
       loss_cluster = [0.]
    return torch.as_tensor(loss_cluster, device=device)




class BalancedSoftmaxLoss(nn.Module):
    def __init__(self, cls_num_list, total_tasks, class_size, which_task):
        super().__init__()
        # numerator = (class_size / total_tasks) * (which_task+1)
        cls_prior = cls_num_list / sum(cls_num_list)
        cls_prior = torch.FloatTensor(cls_prior).cuda()
        self.log_prior = torch.log(cls_prior).unsqueeze(0)

    def forward(self, logits, labels):
        # 扩展 self.log_prior 的第二维
        expanded_log_prior = torch.zeros_like(logits)
        expanded_log_prior[:, :self.log_prior.size(1)] = self.log_prior

        # 相加
        adjusted_logits = logits + expanded_log_prior

        #adjusted_logits = logits + self.log_prior
        # print('min and max target balanced softmax: ', labels.min(), labels.max())
        label_loss = F.cross_entropy(adjusted_logits, labels)

        return label_loss

class Classifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(Classifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.PReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)



class feature_extractor(nn.Module):

    def __init__(self,  wavename = 'haar'):

        super(feature_extractor, self).__init__()
        self.Downsample = DWT_2D(wavename=wavename)

        self.conv1_l = nn.Conv2d(1, 1, kernel_size=1, stride=1, bias=False)
        self.conv1_m = nn.Conv2d(4, 1, kernel_size=1, stride=1, bias=False)
        self.conv1_h = nn.Conv2d(3, 1, kernel_size=1, stride=1, bias=False)

        # 图像数据集的时候
        # self.conv1_l = nn.Conv2d(3, 1, kernel_size=1, stride=1, bias=False)
        # self.conv1_m = nn.Conv2d(12, 1, kernel_size=1, stride=1, bias=False)
        # self.conv1_h = nn.Conv2d(9, 1, kernel_size=1, stride=1, bias=False)

    def forward(self, x: torch.Tensor):
        """
        Compute a forward pass.
        :param x: input tensor (batch_size, *input_shape)
        :return: output tensor (output_classes)
        """

        ll,lh,hl,hh = self.Downsample(x)
        h = torch.cat((lh, hl, hh), 1)
        m = torch.cat((ll, lh, hl, hh), 1)
        x_l = self.conv1_l(ll)
        x_m = self.conv1_m(m)
        x_h = self.conv1_h(h)
        out = torch.cat((x_l, x_m, x_h), 1)

        return out,ll

class DWTNet_MambaVision(nn.Module):
    """
    MambaVision network architecture. Designed for complex datasets.
    """
    def __init__(self,
                 dim,
                 in_dim,
                 depths,
                 window_size,
                 mlp_ratio,
                 num_heads,
                 drop_path_rate=0.2,
                 in_chans=3,
                 num_classes=1000,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 layer_scale=None,
                 layer_scale_conv=None,
                 wavename='haar',
                 **kwargs) -> None:
        super(DWTNet_MambaVision, self).__init__()
        self.feature_extractor = feature_extractor(wavename=wavename)
        self.num_classes = num_classes
        self.model_l = MambaVision(dim,in_dim,depths,window_size,mlp_ratio,num_heads,drop_path_rate,
                                   in_chans,num_classes,qkv_bias,qk_scale,drop_rate,attn_drop_rate,
                                   layer_scale,layer_scale_conv)

        # self.nf= 1024 # 这个数字可能要改 dim=128
        self.nf = 512  # 这个数字可能要改 dim=64
        #nf = MambaVision_S--768 MambaVision_B--1024 MambaVision_L--1568 MambaVision_L3--2048

        self.classwise_select_counts = torch.zeros(num_classes, self.nf)
        self.select_probs = torch.zeros(self.nf)
        self.dropout_st = 0.6   # Select the top 60% of frequency domain features dropout
        self.select_probs[:min(int(self.nf * 1.1 * self.dropout_st), self.nf)] = 1
        self.classwise_select_probs = torch.zeros(num_classes, self.nf)
        # self.classifier = nn.Linear(self.nf, num_classes)

        #self.anchor_encoder = AnchorEncoder(self.nf, self.nf, num_layers=2, num_heads=8, num_classes=num_classes)
        self.classifier = Classifier(self.nf, num_classes)

    def forward(self, x:torch.Tensor, y=None):
        """
        Compute a forward pass.
        :param x: input tensor (batch_size, *input_shape)
        :return: output tensor (output_classes)
        """
        feature, _ = self.feature_extractor(x)
        x = self.model_l(feature)

        if self.training and y is not None:
            # 确保 self.classwise_select_probs 和 y 在同一设备上
            self.classwise_select_probs = self.classwise_select_probs.to(y.device)

            # 确保 torch.rand 生成的张量也在同一设备上
            random_tensor = torch.rand(x.shape[1], device=y.device)

            # 执行索引和比较操作
            classwise_mask = (random_tensor < self.classwise_select_probs[y.long()]).to(x.device)
            assert all(self.classwise_select_probs.sum(dim=1)[y.long()] > 0), "mask error"
            x *= classwise_mask
        x = self.get_kvalue(x, y, self.dropout_st)
        #x = self.anchor_encoder(x)
        out = self.classifier(x)

        return out

    def construct(self, x: torch.Tensor, y = None):
        """
        Compute a forward pass.
        :param x: input tensor (batch_size, *input_shape)
        :return: output tensor (output_classes)
        """
        x = self.model_l(x)
        x = self.get_kvalue(x, y, self.dropout_st)
        out = self.classifier(x)
        return out

    def construct_feature(self, x: torch.Tensor, y = None):
        """
        Compute a forward pass.
        :param x: input tensor (batch_size, *input_shape)
        :return: output tensor (output_classes)
        """
        x = self.model_l(x)
        return x

    def freeze_layers(self):
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def get_kvalue(self, x, y, k_percent_on):
        num_filters = x.shape[1]
        k = int(k_percent_on * num_filters)
        score = torch.abs(x)
        threshold = score.kthvalue(num_filters - k, dim = 1, keepdim=True)[0]
        mask = score > threshold
        x = x * mask
        if y is not None:
            for class_idx in range(self.num_classes):
                sel_idx = y == class_idx
                self.classwise_select_counts[class_idx] += (mask[sel_idx]).sum(
                    dim=0
                ).cpu()
        return x


def DWTNet_MambaVision_B(pretrained=False, wavename = 'haar',  **kwargs):
    # model_path = "/tmp/mamba_vision_B.pth.tar"

    depths = [1, 1, 6, 3]
    num_heads = [2, 4, 8, 8]
    window_size = [8, 16, 4, 2]
    dim = 64           # 256:84
    in_dim = 32
    mlp_ratio = 4
    resolution = 32
    drop_path_rate = 0.1
    layer_scale = 1e-5
    layer_scale_conv = None
    num_classes = 10  # 流量数据集
    # num_classes = 100 # 图像数据集
    in_chans = 3
    qkv_bias = True
    qk_scale=None
    drop_rate = 0.1
    attn_drop_rate = 0.1

    # pretrained_cfg = resolve_pretrained_cfg('mamba_vision_B').to_dict()
    # update_args(pretrained_cfg, kwargs, kwargs_filter=None)
    model = DWTNet_MambaVision(
                        dim,
                        in_dim,
                        depths,
                        window_size,
                        mlp_ratio,
                        num_heads,
                        drop_path_rate,
                        in_chans,
                        num_classes,
                        qkv_bias,
                        qk_scale,
                        drop_rate,
                        attn_drop_rate,
                        layer_scale,
                        layer_scale_conv,
                        wavename,
                        )
    return model




import datetime
import os

import torch
from torch import optim, nn
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import random
import numpy as np
import argparse
from torch.utils.data import TensorDataset, DataLoader



class DWTNet_MambaVision_pretrain(nn.Module):
    def __init__(self, hidden_dim: int,  decoder: bool = False, dtype: torch.dtype = None):
        super().__init__()

        self.mask_token = nn.Parameter(torch.zeros((1, 1, hidden_dim), dtype=dtype))
        self.mask_token._no_weight_decay = True
        is_pretrain = True
        wavename = 'haar'
        self.encoder = DWTNet_MambaVision_B(is_pretrain, wavename)

        # 初始化 replay_component
        self.encoder.replay_component = ReplayComponent(
            cbam_path=None,
            vae_path=None,
            warmup_mode=False
        )


        self.feature_extractor = feature_extractor()
        nn.init.normal_(self.mask_token, std=.02)
        if decoder:
            self.decoder = nn.Linear(hidden_dim * 32, in_channels * (32 ** 2), bias=False)
            nn.init.xavier_uniform_(self.decoder.weight)
            # nn.init.constant_(self.decoder.bias, 0)
            print(f'Parameters: Backbone: {sum([p.numel() for p in self.encoder.parameters()])}, '
                f'Total: {sum([p.numel() for p in self.parameters()])}')

    def random_mask(self, x, mask_ratio):
        N, C, H, W = x.shape
        x = x.reshape(N, C, -1).permute(0, 2, 1)  # 现在 x 的形状为 [N, L, C]
        L = x.shape[1]

        num_masks = int(L * mask_ratio)
        num_keeps = L - num_masks
        noise = torch.rand(N, L)
        
        # 排序噪声以确定哪些位置被保留，哪些被掩码
        ids_shuffle = torch.argsort(noise, dim=1)  # 升序排列: 小的是保留的，大的是移除的
        ids_restore = torch.argsort(ids_shuffle, dim=1).to(x.device)  # ids_restore[i] = 第i个噪声元素的排名

        # 保留前num_keeps个元素
        ids_keep = ids_shuffle[:, :num_keeps].to(x.device)
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, C))  # 实际上是非掩码的部分

        # 生成二进制掩码: 0表示保留，1表示移除
        mask = torch.ones((N, L), device=x.device)
        mask[:, :num_keeps] = 0
        # 反排序得到原始顺序的掩码
        mask = torch.gather(mask, dim=1, index=ids_restore)

        # 确保 mask_tokens 的特征维度与 x_masked 的特征维度一致
        if self.mask_token.shape[-1] != x.shape[-1]:
            raise ValueError(f"Mask token's feature dimension ({self.mask_token.shape[-1]}) does not match input's feature dimension ({x.shape[-1]})")

        mask_tokens = self.mask_token.repeat(N, num_masks, 1)

        x_ = torch.cat((x_masked, mask_tokens), dim=1)  # 没有cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, C))  # 反排序

        x = x_.permute(0, 2, 1).reshape(N, C, H, W)
        return x, mask

    def forward_loss(self, x, y, mask):
        if y.dim() == 4:
            y = y.squeeze(1)

        patch_size = 4
        N, H, W = y.shape
        h_patch = w_patch = H // patch_size

        y = y.reshape(N, 1, h_patch, patch_size, w_patch, patch_size)
        y = torch.einsum('nchpwq->nhwpqc', y)
        y = y.reshape(shape=(N, h_patch * w_patch, patch_size**2))
        x = x.reshape(shape=y.shape)
        mask = mask[:, :y.shape[1]]
        loss = ((x - y) ** 2).mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def pretrain_forward(self, x, mask_ratio):
        x = x.float()
        y = x.clone()
        x, mask = self.encode(x, mask_ratio)
        x = self.decoder(x)
        return self.forward_loss(x, y, mask)

    def encode(self, x, mask_ratio=0.):


        # Step 1: 提取特征和 patch embedding
        x = self.feature_extractor(x)

        # Step 2: 使用 replay_component 对图像进行增强（不需要 labels）
        if hasattr(self.encoder, 'replay_component') and self.encoder.replay_component is not None:
            with torch.no_grad():  # 不训练这部分
                enhanced_x = self.encoder.replay_component(x[0])  # 现在不需要 labels
            x = enhanced_x

        x = self.encoder.model_l.patch_embed(x)

        return self.encode_after_embed(x, mask_ratio)

    def encode_after_embed(self, x, mask_ratio=0.):
        if mask_ratio > 0:
            x, mask = self.random_mask(x, mask_ratio)
        else:
            mask = None
        for level in self.encoder.model_l.levels:
            # x, layer_out
            x = level(x)
        return torch.flatten(x, 1), mask

    def forward(self, x, mask_ratio=0.):
        return self.encode(x, mask_ratio)

def manual_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_pretrain_args():
    p = argparse.ArgumentParser()
    p.add_argument('-b', '--batch_size', type=int, default=128)
    p.add_argument('--mask_ratio', type=float, default=0.8)

    p.add_argument('-m', '--model', type=str, default='bimodal', choices=['bimodal', 'text', 'vision'])
    p.add_argument('--txt_lite', action='store_true', default=False)
    p.add_argument('--vision_lite', action='store_true', default=False)
    p.add_argument('--txt_type', type=str, default='mingru', choices=['mamba', 'mingru'])
    p.add_argument('--wave', action='store_true', default=False)
    p.add_argument('--hidden_dim', type=int, default=128)

    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=1e-4)

    p.add_argument('--seed', type=int, default=3407)
    p.add_argument('-d', '--device', type=int, default=0)

    p.add_argument('--ita', action='store_true', default=False)
    p.add_argument('--clip', action='store_true', default=False)
    p.add_argument('--pretrain_path', type=str, default=None)

        # 添加 warm-up 相关参数
    p.add_argument('--enable_warmup', action='store_true', help='Enable warm-up phase for ReplayComponent')
    p.add_argument('--pretrained_cbam', type=str, default=None, help='Path to pretrained CBAM weights')
    p.add_argument('--pretrained_vae', type=str, default=None, help='Path to pretrained VAE weights')
    p.add_argument('--first_task_data_path', type=str, required=False, help='Path to first task data for warm-up')
    p.add_argument('--warmup_epochs', type=int, default=5, help='Number of warm-up epochs')
    p.add_argument('--warmup_lr', type=float, default=1e-4, help='Learning rate for warm-up')
    p.add_argument('--vae_save_path', type=str, default='saved_models/vae_finetuned.pth', help='Path to save fine-tuned VAE')

    args = p.parse_args()
    print(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    return args

def load_pretrain_data(batch_size: int, payload_size: int = 96, num_window_packets: int = 8):
    # data = torch.load(f'./data/pretrain_data_p96_w8.pt')
    # data = data.reshape(-1, 1, 32, 32)
    # from torchvision.transforms import transforms
    # transforms = transforms.Compose([
    # 	transforms.ToPILImage(),
    # 	transforms.Grayscale(num_output_channels=1),
    # 	transforms.ToTensor(),
    # 	transforms.Normalize(mean=[0.5], std=[0.5]),
    # ])
    # data_list = []
    # for img_tensor in data:
    # 	data_list.append(transforms(img_tensor))
    # data = torch.stack(data_list)
    # torch.save(data, f'./data/pretrain_data_p96_w8_transformed.pt')

    # data = torch.load(f'./data/pretrain_data_p96_w8_transformed.pt')
    dataset_dir = './data/pretrain'
    data_list = []
    for data_file in os.listdir(dataset_dir):
        if not data_file.endswith(f'p{payload_size}_w{num_window_packets}.pt'):
            continue
        data = torch.load(f'{dataset_dir}/{data_file}')
        data = data.reshape(data.size(0), -1)
        if data.size(1) > 32**2:
            data = data[:, :32**2]
        data_list.append(data.view(-1, 1, 32, 32))
    data_list = torch.cat(data_list, dim=0).to(torch.float32) / 255.
    import torchvision.transforms.functional as F
    data_list = F.normalize(data_list, mean=[0.5], std=[0.5], inplace=True)
    data_loader = DataLoader(TensorDataset(data_list), batch_size=batch_size, shuffle=True)
    return data_loader


def pretrain(args, mask_ratio):
    now_time = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    save_dir = f'./output/pretrain/{now_time}/{mask_ratio}'
    os.makedirs(save_dir, exist_ok=True)
    has_nan = False

    # 加载数据集
    loader = load_pretrain_data(args.batch_size)

    # 初始化模型、优化器等
    model = DWTNet_MambaVision_pretrain(hidden_dim=args.hidden_dim, decoder=True).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = torch.nn.MSELoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler()

    # --- 新增：Warm-up 阶段 ---
    if args.enable_warmup:
        print("=> 开始 warm-up 阶段")

        # 构建第一个任务的数据集
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ])
        first_task_dataset = datasets.ImageFolder(root=args.first_task_data_path, transform=transform)
        first_task_loader = DataLoader(first_task_dataset, batch_size=args.batch_size, shuffle=True)

        # 初始化 replay_component（可能需要从 model 中提取）
        if not hasattr(model.encoder, 'replay_component'):
            model.encoder.replay_component = ReplayComponent(
                cbam_path=args.pretrained_cbam,
                vae_path=args.pretrained_vae,
                warmup_mode=True
            ).to(device)

        # Warm-up
        warmup_args = {
            'device': device,
            'warmup_epochs': args.warmup_epochs,
            'warmup_lr': args.warmup_lr,
            'batch_size': args.batch_size,
            'vae_save_path': args.vae_save_path if hasattr(args, 'vae_save_path') else 'saved_models/vae_finetuned.pth'
        }

        warmup_vae_with_cbam(model.encoder.replay_component, first_task_loader, warmup_args)

        # 将 warm-up 后的 VAE 权重保存并重新加载到模型中
        model.encoder.replay_component.vae.load_state_dict(torch.load(warmup_args['vae_save_path']))
        print("=> warm-up 完成")
    # ----------------------------

    for epoch in range(0, args.epochs):
        if has_nan:
            break
        model.train()
        epoch_progress = tqdm(enumerate(loader), total=len(loader), desc=f'Epoch {epoch}', disable=False)
        for batch_index, batch_data in epoch_progress:
            batch_data = batch_data[0].to(device, non_blocking=True)
            with autocast():
                loss = model.pretrain_forward(batch_data, mask_ratio)
                if torch.isnan(loss).any():
                    has_nan = True
                    print(f'Epoch {epoch} has NaN.')
                    break
            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0, 2)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
            epoch_progress.set_postfix(loss=loss.item())

        if (epoch + 1) % 1 == 0:
            model.cpu()
            state_dict = {
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'scaler': scaler.state_dict()
            }
            save_path = f'{save_dir}/h{args.hidden_dim}_e{epoch + 1}_d{args.device}.pt'
            torch.save(state_dict, save_path)
            model.to(device, non_blocking=True)


if __name__ == '__main__':
    args = parse_pretrain_args()
    args.hidden_dim = 64
    seed = args.seed
    manual_seed(seed)

    device = torch.device('cuda:0')
    loader = load_pretrain_data(args.batch_size)

    hidden_dim = 64
    decoder = True
    dtype = None
    #dtype = torch.float32

    in_channels = 1
    mask_ratio = args.mask_ratio
    for mask_ratio in [
        # 0.25,
        # 0.5,
        # 0.75,
        #0.8,
        0.9
    ]:
        args.mask_ratio = mask_ratio
        print(args)
        time = datetime.datetime.now().strftime('%m_%d_%H_%M')
        # logger = get_logger(f'pretrain{wave_str}_h{args.hidden_dim}_d{args.device}_t{time}')

        model = DWTNet_MambaVision_pretrain(hidden_dim, decoder, dtype).to(device)

        optimizer = optim.AdamW(model.parameters(),
                                lr=args.lr,
                                weight_decay=args.weight_decay)
        criterion = torch.nn.MSELoss()
        criterion_valid = torch.nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        scaler = GradScaler()
        pretrain(args, mask_ratio)