import math
import os
import random
import warnings
from collections import OrderedDict
from copy import deepcopy
from itertools import repeat
from os import path as osp

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from torch.nn import init as init

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

INFERENCE_SEED = 231


def dict2str(opt, indent_level=1):
    msg = "\n"
    for key, value in opt.items():
        if isinstance(value, dict):
            msg += " " * (indent_level * 2) + key + ":["
            msg += dict2str(value, indent_level + 1)
            msg += " " * (indent_level * 2) + "]\n"
        else:
            msg += " " * (indent_level * 2) + key + ": " + str(value) + "\n"
    return msg


def print_info(message):
    print(message, flush=True)


def print_warning(message):
    print(f"Warning: {message}", flush=True)


def set_deterministic_seed(seed=INFERENCE_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def img2tensor(imgs, bgr2rgb=True, float32=True):
    def _totensor(img):
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)
        if img.shape[2] == 3 and bgr2rgb:
            if img.dtype == np.float64:
                img = img.astype(np.float32)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        return img.float() if float32 else img

    if isinstance(imgs, list):
        return [_totensor(img) for img in imgs]
    return _totensor(imgs)


def tensor2img(tensor, rgb2bgr=True, out_type=np.uint8, min_max=(0, 1)):
    if torch.is_tensor(tensor):
        tensor = [tensor]
    result = []
    for item in tensor:
        item = item.squeeze(0).float().detach().cpu().clamp_(*min_max)
        item = (item - min_max[0]) / (min_max[1] - min_max[0])
        if item.dim() == 3:
            img_np = item.numpy().transpose(1, 2, 0)
            if img_np.shape[2] == 1:
                img_np = np.squeeze(img_np, axis=2)
            elif rgb2bgr:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif item.dim() == 2:
            img_np = item.numpy()
        else:
            raise TypeError(f"Unsupported tensor dim: {item.dim()}")
        if out_type == np.uint8:
            img_np = (img_np * 255.0).round()
        result.append(img_np.astype(out_type))
    return result[0] if len(result) == 1 else result


def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if mean < a - 2 * std or mean > b + 2 * std:
        warnings.warn("mean is more than 2 std from [a, b] in trunc_normal_", stacklevel=2)

    with torch.no_grad():
        low = norm_cdf((a - mean) / std)
        up = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * low - 1, 2 * up - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def _ntuple(n):
    def parse(x):
        if isinstance(x, (tuple, list)):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)


def index_reverse(index):
    index_r = torch.zeros_like(index)
    ind = torch.arange(0, index.shape[-1], device=index.device)
    for i in range(index.shape[0]):
        index_r[i, index[i, :]] = ind
    return index_r


def semantic_neighbor(x, index):
    dim = index.dim()
    assert x.shape[:dim] == index.shape, f"x ({x.shape}) and index ({index.shape}) shape incompatible"
    for _ in range(x.dim() - index.dim()):
        index = index.unsqueeze(-1)
    index = index.expand(x.shape)
    return torch.gather(x, dim=dim - 1, index=index)


class dwconv(nn.Module):
    def __init__(self, hidden_features, kernel_size=5):
        super().__init__()
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(
                hidden_features,
                hidden_features,
                kernel_size=kernel_size,
                stride=1,
                padding=(kernel_size - 1) // 2,
                groups=hidden_features,
            ),
            nn.GELU(),
        )
        self.hidden_features = hidden_features

    def forward(self, x, x_size):
        x = x.transpose(1, 2).view(x.shape[0], self.hidden_features, x_size[0], x_size[1]).contiguous()
        x = self.depthwise_conv(x)
        return x.flatten(2).transpose(1, 2).contiguous()


class ConvFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, kernel_size=5, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.dwconv = dwconv(hidden_features=hidden_features, kernel_size=kernel_size)
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x, x_size):
        x = self.fc1(x)
        x = self.act(x)
        x = x + self.dwconv(x, x_size)
        x = self.fc2(x)
        return x


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.qkv_bias = qkv_bias
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )
        self.proj = nn.Linear(dim, dim)
        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, qkv, rpi, mask=None):
        b_, n, c3 = qkv.shape
        c = c3 // 3
        qkv = qkv.reshape(b_, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[rpi.view(-1)].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1,
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b_ // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
        attn = self.softmax(attn)

        x = (attn @ v).transpose(1, 2).reshape(b_, n, c)
        return self.proj(x)


def window_partition(x, window_size):
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c)


def window_reverse(windows, window_size, h, w):
    b = int(windows.shape[0] / (h * w / window_size / window_size))
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)


class ASSM(nn.Module):
    def __init__(self, dim, d_state, input_resolution, num_tokens=64, inner_rank=128, mlp_ratio=2.0):
        super().__init__()
        del input_resolution
        self.dim = dim
        self.num_tokens = num_tokens
        self.inner_rank = inner_rank
        self.expand = mlp_ratio
        hidden = int(self.dim * self.expand)
        self.d_state = d_state
        self.selectiveScan = Selective_Scan(d_model=hidden, d_state=self.d_state, expand=1)
        self.out_norm = nn.LayerNorm(hidden)
        self.out_proj = nn.Linear(hidden, dim, bias=True)
        self.in_proj = nn.Sequential(nn.Conv2d(self.dim, hidden, 1, 1, 0))
        self.CPE = nn.Sequential(nn.Conv2d(hidden, hidden, 3, 1, 1, groups=hidden))
        self.embeddingB = nn.Embedding(self.num_tokens, self.inner_rank)
        self.embeddingB.weight.data.uniform_(-1 / self.num_tokens, 1 / self.num_tokens)
        self.route = nn.Sequential(
            nn.Linear(self.dim, self.dim // 3),
            nn.GELU(),
            nn.Linear(self.dim // 3, self.num_tokens),
            nn.LogSoftmax(dim=-1),
        )

    def forward(self, x, x_size, token):
        b, n, c = x.shape
        h, w = x_size
        full_embedding = self.embeddingB.weight @ token.weight
        pred_route = self.route(x)
        cls_policy = F.gumbel_softmax(pred_route, hard=True, dim=-1)
        prompt = torch.matmul(cls_policy, full_embedding).view(b, n, self.d_state)

        detached_index = torch.argmax(cls_policy.detach(), dim=-1, keepdim=False).view(b, n)
        _, x_sort_indices = torch.sort(detached_index, dim=-1, stable=False)
        x_sort_indices_reverse = index_reverse(x_sort_indices)

        x = x.permute(0, 2, 1).reshape(b, c, h, w).contiguous()
        x = self.in_proj(x)
        x = x * torch.sigmoid(self.CPE(x))
        cc = x.shape[1]
        x = x.view(b, cc, -1).contiguous().permute(0, 2, 1)
        semantic_x = semantic_neighbor(x, x_sort_indices)
        y = self.selectiveScan(semantic_x, prompt)
        y = self.out_proj(self.out_norm(y))
        return semantic_neighbor(y, x_sort_indices_reverse)


class Selective_Scan(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        expand=2.0,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        device=None,
        dtype=None,
        **kwargs,
    ):
        del kwargs
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.x_proj = (nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs),)
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        self.dt_projs = (
            self.dt_init(
                self.dt_rank,
                self.d_inner,
                dt_scale,
                dt_init,
                dt_min,
                dt_max,
                dt_init_floor,
                **factory_kwargs,
            ),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=1, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=1, merge=True)
        self.selective_scan = selective_scan_fn

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        a = torch.arange(1, d_state + 1, dtype=torch.float32, device=device).unsqueeze(0).repeat(d_inner, 1).contiguous()
        a_log = torch.log(a)
        if copies > 1:
            a_log = a_log.unsqueeze(0).repeat(copies, 1, 1)
            if merge:
                a_log = a_log.flatten(0, 1)
        a_log = nn.Parameter(a_log)
        a_log._no_weight_decay = True
        return a_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        d = torch.ones(d_inner, device=device)
        if copies > 1:
            d = d.unsqueeze(0).repeat(copies, 1)
            if merge:
                d = d.flatten(0, 1)
        d = nn.Parameter(d)
        d._no_weight_decay = True
        return d

    def forward_core(self, x, prompt):
        b, l, _ = x.shape
        k = 1
        xs = x.permute(0, 2, 1).view(b, 1, -1, l).contiguous()
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(b, k, -1, l), self.x_proj_weight)
        dts, bs, cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(b, k, -1, l), self.dt_projs_weight)
        xs = xs.float().view(b, -1, l)
        dts = dts.contiguous().float().view(b, -1, l)
        bs = bs.float().view(b, k, -1, l)
        cs = cs.float().view(b, k, -1, l) + prompt
        ds = self.Ds.float().view(-1)
        a_s = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_bias = self.dt_projs_bias.float().view(-1)
        out_y = self.selective_scan(
            xs,
            dts,
            a_s,
            bs,
            cs,
            ds,
            z=None,
            delta_bias=dt_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(b, k, -1, l)
        return out_y[:, 0]

    def forward(self, x, prompt, **kwargs):
        del kwargs
        b, l, c = prompt.shape
        prompt = prompt.permute(0, 2, 1).contiguous().view(b, 1, c, l)
        y = self.forward_core(x, prompt)
        return y.permute(0, 2, 1).contiguous()


class AttentiveLayer(nn.Module):
    def __init__(
        self,
        dim,
        d_state,
        input_resolution,
        num_heads,
        window_size,
        shift_size,
        inner_rank,
        num_tokens,
        convffn_kernel_size,
        mlp_ratio,
        qkv_bias=True,
        norm_layer=nn.LayerNorm,
        is_last=False,
    ):
        super().__init__()
        del input_resolution, is_last
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        self.norm4 = norm_layer(dim)
        layer_scale = 1e-4
        self.scale1 = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)
        self.scale2 = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)
        self.wqkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.win_mhsa = WindowAttention(dim, window_size=to_2tuple(window_size), num_heads=num_heads, qkv_bias=qkv_bias)
        self.assm = ASSM(dim, d_state, input_resolution=None, num_tokens=num_tokens, inner_rank=inner_rank, mlp_ratio=mlp_ratio)
        mlp_hidden_dim = int(dim * self.mlp_ratio)
        self.convffn1 = ConvFFN(in_features=dim, hidden_features=mlp_hidden_dim, kernel_size=convffn_kernel_size)
        self.convffn2 = ConvFFN(in_features=dim, hidden_features=mlp_hidden_dim, kernel_size=convffn_kernel_size)
        self.embeddingA = nn.Embedding(inner_rank, d_state)
        self.embeddingA.weight.data.uniform_(-1 / inner_rank, 1 / inner_rank)

    def forward(self, x, x_size, params):
        h, w = x_size
        b, n, c = x.shape
        shortcut = x
        x = self.norm1(x)
        qkv = self.wqkv(x).reshape(b, h, w, 3 * c)
        if self.shift_size > 0:
            shifted_qkv = torch.roll(qkv, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = params["attn_mask"]
        else:
            shifted_qkv = qkv
            attn_mask = None
        x_windows = window_partition(shifted_qkv, self.window_size).view(-1, self.window_size * self.window_size, 3 * c)
        attn_windows = self.win_mhsa(x_windows, rpi=params["rpi_sa"], mask=attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, c)
        shifted_x = window_reverse(attn_windows, self.window_size, h, w)
        if self.shift_size > 0:
            attn_x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            attn_x = shifted_x
        x_win = attn_x.view(b, n, c) + shortcut
        x_win = self.convffn1(self.norm2(x_win), x_size) + x_win
        x = shortcut * self.scale1 + x_win

        shortcut = x
        x_aca = self.assm(self.norm3(x), x_size, self.embeddingA) + x
        x = x_aca + self.convffn2(self.norm4(x_aca), x_size)
        x = shortcut * self.scale2 + x
        return x


class BasicBlock(nn.Module):
    def __init__(
        self,
        dim,
        d_state,
        input_resolution,
        idx,
        depth,
        num_heads,
        window_size,
        inner_rank,
        num_tokens,
        convffn_kernel_size,
        mlp_ratio=4.0,
        qkv_bias=True,
        norm_layer=nn.LayerNorm,
        downsample=None,
        use_checkpoint=False,
    ):
        super().__init__()
        del idx, use_checkpoint
        self.layers = nn.ModuleList(
            [
                AttentiveLayer(
                    dim=dim,
                    d_state=d_state,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if i % 2 == 0 else window_size // 2,
                    inner_rank=inner_rank,
                    num_tokens=num_tokens,
                    convffn_kernel_size=convffn_kernel_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    norm_layer=norm_layer,
                    is_last=i == depth - 1,
                )
                for i in range(depth)
            ]
        )
        self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer) if downsample is not None else None

    def forward(self, x, x_size, params):
        for layer in self.layers:
            x = layer(x, x_size, params)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        del in_chans
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]
        self.embed_dim = embed_dim
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchUnEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        del img_size, patch_size, in_chans, norm_layer
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        return x.transpose(1, 2).view(x.shape[0], self.embed_dim, x_size[0], x_size[1])


class Upsample(nn.Sequential):
    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f"Unsupported scale: {scale}")
        super().__init__(*m)


class UpsampleOneStep(nn.Sequential):
    def __init__(self, scale, num_feat, num_out_ch, input_resolution=None):
        del input_resolution
        super().__init__(nn.Conv2d(num_feat, (scale ** 2) * num_out_ch, 3, 1, 1), nn.PixelShuffle(scale))


class ASSB(nn.Module):
    def __init__(
        self,
        dim,
        d_state,
        idx,
        input_resolution,
        depth,
        num_heads,
        window_size,
        inner_rank,
        num_tokens,
        convffn_kernel_size,
        mlp_ratio,
        qkv_bias=True,
        norm_layer=nn.LayerNorm,
        downsample=None,
        use_checkpoint=False,
        img_size=224,
        patch_size=4,
        resi_connection="1conv",
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)
        self.patch_unembed = PatchUnEmbed(img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)
        self.residual_group = BasicBlock(
            dim=dim,
            d_state=d_state,
            input_resolution=input_resolution,
            idx=idx,
            depth=depth,
            num_heads=num_heads,
            window_size=window_size,
            inner_rank=inner_rank,
            num_tokens=num_tokens,
            convffn_kernel_size=convffn_kernel_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer,
            downsample=downsample,
            use_checkpoint=use_checkpoint,
        )
        if resi_connection == "1conv":
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == "3conv":
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim // 4, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim // 4, 1, 1, 0),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim, 3, 1, 1),
            )
        else:
            raise ValueError(f"Unsupported resi_connection: {resi_connection}")

    def forward(self, x, x_size, params):
        return self.patch_embed(self.conv(self.patch_unembed(self.residual_group(x, x_size, params), x_size))) + x


class MambaIRv2(nn.Module):
    def __init__(
        self,
        img_size=64,
        patch_size=1,
        in_chans=3,
        embed_dim=48,
        d_state=8,
        depths=(6, 6, 6, 6),
        num_heads=(4, 4, 4, 4),
        window_size=16,
        inner_rank=32,
        num_tokens=64,
        convffn_kernel_size=5,
        mlp_ratio=2.0,
        qkv_bias=True,
        norm_layer=nn.LayerNorm,
        ape=False,
        patch_norm=True,
        use_checkpoint=False,
        upscale=2,
        img_range=1.0,
        upsampler="",
        resi_connection="1conv",
        **kwargs,
    ):
        super().__init__()
        del kwargs
        num_in_ch = in_chans
        num_out_ch = in_chans
        num_feat = 64
        self.img_range = img_range
        self.mean = torch.Tensor((0.4488, 0.4371, 0.4040)).view(1, 3, 1, 1) if in_chans == 3 else torch.zeros(1, 1, 1, 1)
        self.upscale = upscale
        self.upsampler = upsampler
        self.window_size = window_size

        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
        )
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
        )

        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=0.02)

        relative_position_index_sa = self.calculate_rpi_sa()
        self.register_buffer("relative_position_index_SA", relative_position_index_sa)

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            self.layers.append(
                ASSB(
                    dim=embed_dim,
                    d_state=d_state,
                    idx=i_layer,
                    input_resolution=(patches_resolution[0], patches_resolution[1]),
                    depth=depths[i_layer],
                    num_heads=num_heads[i_layer],
                    window_size=window_size,
                    inner_rank=inner_rank,
                    num_tokens=num_tokens,
                    convffn_kernel_size=convffn_kernel_size,
                    mlp_ratio=self.mlp_ratio,
                    qkv_bias=qkv_bias,
                    norm_layer=norm_layer,
                    downsample=None,
                    use_checkpoint=use_checkpoint,
                    img_size=img_size,
                    patch_size=patch_size,
                    resi_connection=resi_connection,
                )
            )
        self.norm = norm_layer(self.num_features)

        if resi_connection == "1conv":
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == "3conv":
            self.conv_after_body = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1),
            )
        else:
            raise ValueError(f"Unsupported resi_connection: {resi_connection}")

        if self.upsampler == "pixelshuffle":
            self.conv_before_upsample = nn.Sequential(nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True))
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        elif self.upsampler == "pixelshuffledirect":
            self.upsample = UpsampleOneStep(upscale, embed_dim, num_out_ch, (patches_resolution[0], patches_resolution[1]))
        elif self.upsampler == "nearest+conv":
            assert self.upscale == 4, "only support x4 now."
            self.conv_before_upsample = nn.Sequential(nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True))
            self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        else:
            self.conv_last = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x, params):
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        for layer in self.layers:
            x = layer(x, x_size, params)
        x = self.norm(x)
        return self.patch_unembed(x, x_size)

    def calculate_rpi_sa(self):
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size - 1
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        return relative_coords.sum(-1)

    def calculate_mask(self, x_size):
        h, w = x_size
        img_mask = torch.zeros((1, h, w, 1))
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -(self.window_size // 2)),
            slice(-(self.window_size // 2), None),
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -(self.window_size // 2)),
            slice(-(self.window_size // 2), None),
        )
        cnt = 0
        for h_slice in h_slices:
            for w_slice in w_slices:
                img_mask[:, h_slice, w_slice, :] = cnt
                cnt += 1
        mask_windows = window_partition(img_mask, self.window_size).view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        return attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

    def forward(self, x):
        h_ori, w_ori = x.size()[-2], x.size()[-1]
        mod = self.window_size
        h_pad = ((h_ori + mod - 1) // mod) * mod - h_ori
        w_pad = ((w_ori + mod - 1) // mod) * mod - w_ori
        h, w = h_ori + h_pad, w_ori + w_pad
        x = torch.cat([x, torch.flip(x, [2])], 2)[:, :, :h, :]
        x = torch.cat([x, torch.flip(x, [3])], 3)[:, :, :, :w]

        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range
        attn_mask = self.calculate_mask([h, w]).to(x.device)
        params = {"attn_mask": attn_mask, "rpi_sa": self.relative_position_index_SA}

        if self.upsampler == "pixelshuffle":
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x, params)) + x
            x = self.conv_before_upsample(x)
            x = self.conv_last(self.upsample(x))
        elif self.upsampler == "pixelshuffledirect":
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x, params)) + x
            x = self.upsample(x)
        elif self.upsampler == "nearest+conv":
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x, params)) + x
            x = self.conv_before_upsample(x)
            x = self.lrelu(self.conv_up1(F.interpolate(x, scale_factor=2, mode="nearest")))
            x = self.lrelu(self.conv_up2(F.interpolate(x, scale_factor=2, mode="nearest")))
            x = self.conv_last(self.lrelu(self.conv_hr(x)))
        else:
            x_first = self.conv_first(x)
            res = self.conv_after_body(self.forward_features(x_first, params)) + x_first
            x = x + self.conv_last(res)

        x = x / self.img_range + self.mean
        return x[..., : h_ori * self.upscale, : w_ori * self.upscale]


class MambaIRv2Tester:
    def __init__(self, opt):
        self.opt = opt
        requested_device = opt.get("device")
        if requested_device is not None:
            self.device = torch.device(requested_device)
        else:
            use_cuda = torch.cuda.is_available() and opt["num_gpu"] != 0
            self.device = torch.device("cuda" if use_cuda else "cpu")

        network_opt = deepcopy(opt["network_g"])
        network_type = network_opt.pop("type")
        if network_type != "MambaIRv2":
            raise ValueError(f"Unsupported network type: {network_type}")
        self.net_g = MambaIRv2(**network_opt)
        self.net_g = self.net_g.to(self.device)
        if self.device.type == "cuda" and opt["num_gpu"] > 1 and torch.cuda.device_count() > 1:
            self.net_g = torch.nn.DataParallel(self.net_g)

        self.print_network()
        load_path = self.opt["path"].get("pretrain_network_g")
        if load_path is not None:
            param_key = self.opt["path"].get("param_key_g", "params")
            self.load_network(load_path, strict=self.opt["path"].get("strict_load_g", True), param_key=param_key)

    def get_bare_model(self):
        if isinstance(self.net_g, torch.nn.DataParallel):
            return self.net_g.module
        return self.net_g

    def print_network(self):
        net = self.get_bare_model()
        net_params = sum(p.numel() for p in net.parameters())
        print_info(f"Network: {net.__class__.__name__}, with parameters: {net_params:,d}")

    def load_network(self, load_path, strict=True, param_key="params"):
        net = self.get_bare_model()
        load_net = torch.load(load_path, map_location="cpu")
        if param_key is not None:
            if param_key not in load_net and "params" in load_net:
                param_key = "params"
                print_info("Loading: params_ema does not exist, use params.")
            load_net = load_net[param_key]
        for key, value in deepcopy(load_net).items():
            if key.startswith("module."):
                load_net[key[7:]] = value
                load_net.pop(key)
        if not strict:
            current = net.state_dict()
            for key in list(load_net.keys()):
                if key in current and current[key].shape != load_net[key].shape:
                    print_warning(
                        f"Size different, ignore [{key}]: crt_net: {current[key].shape}; load_net: {load_net[key].shape}"
                    )
                    load_net.pop(key)
        net.load_state_dict(load_net, strict=strict)
        print_info(f"Loading {net.__class__.__name__} model from {load_path}, with param key: [{param_key}].")

    def feed_data(self, data):
        self.lq = data["lq"].to(self.device)
        if "gt" in data:
            self.gt = data["gt"].to(self.device)

    def test(self):
        _, c, h, w = self.lq.size()
        split_token_h = h // 200 + 1
        split_token_w = w // 200 + 1
        mod_pad_h = 0 if h % split_token_h == 0 else split_token_h - h % split_token_h
        mod_pad_w = 0 if w % split_token_w == 0 else split_token_w - w % split_token_w
        img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), "reflect")
        _, _, H, W = img.size()
        split_h = H // split_token_h
        split_w = W // split_token_w
        shave_h = split_h // 10
        shave_w = split_w // 10
        scale = self.opt.get("scale", 1)
        rows = H // split_h
        cols = W // split_w
        slices = []
        for i in range(rows):
            for j in range(cols):
                if i == 0 and i == rows - 1:
                    top = slice(i * split_h, (i + 1) * split_h)
                elif i == 0:
                    top = slice(i * split_h, (i + 1) * split_h + shave_h)
                elif i == rows - 1:
                    top = slice(i * split_h - shave_h, (i + 1) * split_h)
                else:
                    top = slice(i * split_h - shave_h, (i + 1) * split_h + shave_h)

                if j == 0 and j == cols - 1:
                    left = slice(j * split_w, (j + 1) * split_w)
                elif j == 0:
                    left = slice(j * split_w, (j + 1) * split_w + shave_w)
                elif j == cols - 1:
                    left = slice(j * split_w - shave_w, (j + 1) * split_w)
                else:
                    left = slice(j * split_w - shave_w, (j + 1) * split_w + shave_w)
                slices.append((top, left))

        self.net_g.eval()
        with torch.no_grad():
            outputs = []
            for top, left in slices:
                outputs.append(self.net_g(img[..., top, left]))
            merged = torch.zeros(1, c, H * scale, W * scale, device=self.device)
            for i in range(rows):
                for j in range(cols):
                    top = slice(i * split_h * scale, (i + 1) * split_h * scale)
                    left = slice(j * split_w * scale, (j + 1) * split_w * scale)
                    crop_top = slice(0, split_h * scale) if i == 0 else slice(shave_h * scale, (shave_h + split_h) * scale)
                    crop_left = slice(0, split_w * scale) if j == 0 else slice(shave_w * scale, (shave_w + split_w) * scale)
                    merged[..., top, left] = outputs[i * cols + j][..., crop_top, crop_left]
            self.output = merged
        _, _, out_h, out_w = self.output.size()
        self.output = self.output[:, :, 0 : out_h - mod_pad_h * scale, 0 : out_w - mod_pad_w * scale]

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict["lq"] = self.lq.detach().cpu()
        out_dict["result"] = self.output.detach().cpu()
        if hasattr(self, "gt"):
            out_dict["gt"] = self.gt.detach().cpu()
        return out_dict

    def validation(self, dataloader, current_iter, save_img):
        dataset_name = dataloader.dataset.opt["name"]
        for idx, val_data in enumerate(dataloader):
            del idx
            img_name = osp.splitext(osp.basename(val_data["lq_path"][0]))[0]
            self.feed_data(val_data)
            self.test()
            visuals = self.get_current_visuals()
            sr_img = tensor2img(visuals["result"])

            if save_img:
                suffix = self.opt["val"].get("suffix")
                if suffix:
                    save_img_path = osp.join(self.opt["path"]["visualization"], dataset_name, f"{img_name}_{suffix}.png")
                else:
                    save_img_path = osp.join(
                        self.opt["path"]["visualization"], dataset_name, f"{img_name}_{self.opt['name']}.png"
                    )
                os.makedirs(osp.dirname(osp.abspath(save_img_path)), exist_ok=True)
                if not cv2.imwrite(save_img_path, sr_img):
                    raise IOError(f"Failed to write image: {save_img_path}")

            del self.lq
            del self.output
            if hasattr(self, "gt"):
                del self.gt
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        print_info(f"Finished testing {dataset_name}.")


def build_model(opt):
    model_type = opt["model_type"]
    if model_type != "MambaIRv2Model":
        raise ValueError(f"Unsupported model type: {model_type}")
    return MambaIRv2Tester(opt)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def resolve_device(device):
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(device, torch.device):
        return device
    return torch.device(device)


def resolve_checkpoint_path(model_dir):
    model_dir = osp.abspath(osp.expanduser(str(model_dir)))
    if osp.isfile(model_dir):
        return model_dir
    if not osp.isdir(model_dir):
        raise FileNotFoundError(f"Model path does not exist: {model_dir}")

    candidates = [
        osp.join(model_dir, "mambair_v2.pth")
    ]
    for candidate in candidates:
        if osp.isfile(candidate):
            return candidate
    raise FileNotFoundError(f"No Mamba checkpoint found under: {model_dir}")


def get_runtime_options(checkpoint_path, output_path, device):
    if device.type == "cuda":
        num_gpu = 1 if device.index is not None else max(1, torch.cuda.device_count())
    else:
        num_gpu = 0

    return {
        "name": "mamba_runtime",
        "model_type": "MambaIRv2Model",
        "scale": 4,
        "num_gpu": num_gpu,
        "device": device,
        "network_g": {
            "type": "MambaIRv2",
            "upscale": 4,
            "in_chans": 3,
            "img_size": 64,
            "img_range": 1.0,
            "embed_dim": 174,
            "d_state": 16,
            "depths": [6, 6, 6, 6, 6, 6],
            "num_heads": [6, 6, 6, 6, 6, 6],
            "window_size": 16,
            "inner_rank": 64,
            "num_tokens": 128,
            "convffn_kernel_size": 5,
            "mlp_ratio": 2.0,
            "upsampler": "pixelshuffle",
            "resi_connection": "1conv",
        },
        "path": {
            "pretrain_network_g": checkpoint_path,
            "strict_load_g": False,
            "results_root": output_path,
            "visualization": output_path,
        },
    }


def list_image_paths(input_path):
    input_path = osp.abspath(osp.expanduser(str(input_path)))
    if osp.isfile(input_path):
        return [input_path]
    if not osp.isdir(input_path):
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    image_paths = []
    for root, _, files in os.walk(input_path):
        for file_name in files:
            if osp.splitext(file_name)[1].lower() in IMAGE_EXTENSIONS:
                image_paths.append(osp.join(root, file_name))
    image_paths.sort(key=lambda path: osp.relpath(path, input_path))
    return image_paths


def load_image_tensor(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"Failed to read image: {image_path}")
    img = img.astype(np.float32) / 255.0
    return img2tensor(img, bgr2rgb=True, float32=True).unsqueeze(0)


def run_folder(model, input_path, output_path):
    input_path = osp.abspath(osp.expanduser(str(input_path)))
    output_path = osp.abspath(osp.expanduser(str(output_path)))
    os.makedirs(output_path, exist_ok=True)

    image_paths = list_image_paths(input_path)
    if not image_paths:
        raise ValueError(f"No images found under: {input_path}")

    root_path = input_path if osp.isdir(input_path) else osp.dirname(input_path)
    print_info(f"Found {len(image_paths)} images in {input_path}.")

    for index, image_path in enumerate(image_paths, start=1):
        rel_path = osp.relpath(image_path, root_path)
        save_path = osp.join(output_path, osp.splitext(rel_path)[0] + ".png")
        print_info(f"[{index}/{len(image_paths)}] Processing {rel_path}")

        model.feed_data({"lq": load_image_tensor(image_path)})
        model.test()
        sr_img = tensor2img(model.get_current_visuals()["result"])
        os.makedirs(osp.dirname(osp.abspath(save_path)), exist_ok=True)
        if not cv2.imwrite(save_path, sr_img):
            raise IOError(f"Failed to write image: {save_path}")

        del model.lq
        del model.output
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main_mamba(model_dir, input_path, output_path, device=None):
    set_deterministic_seed()
    device = resolve_device(device)
    if device.type == "cuda":
        if device.index is not None:
            torch.cuda.set_device(device)
        torch.cuda.empty_cache()

    checkpoint_path = resolve_checkpoint_path(model_dir)
    runtime_opt = get_runtime_options(checkpoint_path, output_path, device)
    runtime_opt["seed"] = INFERENCE_SEED
    print_info(dict2str(runtime_opt))

    model = build_model(runtime_opt)
    run_folder(model, input_path, output_path)
    print_info(f"Done. Results saved to {osp.abspath(osp.expanduser(str(output_path)))}")
