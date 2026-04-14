

import math
import torch
import torch.nn as nn


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample."""
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c)
    return windows


def window_reverse(windows, window_size, h, w):
    b = int(windows.shape[0] / (h * w / window_size / window_size))
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
    return x


def grid_shuffle(x, h, w, c, interval_size):
    x = x.view(-1, h // interval_size, interval_size, w // interval_size, interval_size, c)
    x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
    x = x.view(-1, h // interval_size, w // interval_size, c)
    return x


def grid_unshuffle(x, b, h, w, interval_size):
    x = x.view(b, interval_size, interval_size, h // interval_size, w // interval_size, -1)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous().view(b, h, w, -1)
    return x


def to_2tuple(x):
    if isinstance(x, (list, tuple)):
        return x
    return (x, x)


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    """Fills the input Tensor with values drawn from a truncated normal distribution."""
    with torch.no_grad():
        return tensor.normal_(mean, std)


class DynamicPosBias(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.pos_dim = dim // 4
        self.pos_proj = nn.Linear(2, self.pos_dim)
        self.pos1 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim),
        )
        self.pos2 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim)
        )
        self.pos3 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.num_heads)
        )

    def forward(self, biases):
        pos = self.pos3(self.pos2(self.pos1(self.pos_proj(biases))))
        return pos


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, rpi, mask=None):
        b_, n, c = x.shape
        qkv = x.reshape(b_, n, 3, self.num_heads, c // self.num_heads // 3).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[rpi.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b_ // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(b_, n, c // 3)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class FAB(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size

        self.norm1 = norm_layer(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, x_size, rpi_sa, attn_mask):
        h, w = x_size
        b, _, c = x.shape
        shortcut = x
        x = self.norm1(x)
        x = x.view(b, h, w, c)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = attn_mask
        else:
            shifted_x = x
            attn_mask = None

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, c)
        attn_windows = self.attn(self.qkv(x_windows), rpi=rpi_sa, mask=attn_mask)

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, c)
        shifted_x = window_reverse(attn_windows, self.window_size, h, w)

        if self.shift_size > 0:
            attn_x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            attn_x = shifted_x
        attn_x = attn_x.view(b, h * w, c)

        x = shortcut + self.drop_path(attn_x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class SEModule(nn.Module):
    def __init__(self, channels, rd_channels=None, bias=True):
        super(SEModule, self).__init__()
        self.fc1 = nn.Conv2d(channels, rd_channels, kernel_size=1, bias=bias)
        self.act = nn.SiLU(inplace=True)
        self.fc2 = nn.Conv2d(rd_channels, channels, kernel_size=1, bias=bias)
        self.gate = nn.Sigmoid()

    def forward(self, x):
        x_se = x.mean((2, 3), keepdim=True)
        x_se = self.fc1(x_se)
        x_se = self.act(x_se)
        x_se = self.fc2(x_se)
        return x * self.gate(x_se)


class FusedConv(nn.Module):
    def __init__(self, num_feat, expand_size=4, attn_ratio=4):
        super(FusedConv, self).__init__()
        mid_feat = num_feat * expand_size
        rd_feat = int(mid_feat / attn_ratio)
        self.pre_norm = nn.LayerNorm(num_feat)
        self.fused_conv = nn.Conv2d(num_feat, mid_feat, 3, 1, 1)
        self.norm1 = nn.LayerNorm(mid_feat)
        self.act1 = nn.GELU()
        self.se = SEModule(mid_feat, rd_feat, bias=True)
        self.conv3_1x1 = nn.Conv2d(mid_feat, num_feat, 1, 1)

    def forward(self, x, x_size, rpi, mask):
        shortcut = x
        h, w = x_size
        b, _, c = x.shape
        x = x.view(b, h, w, c)
        x = self.pre_norm(x).permute(0, 3, 1, 2)
        x = self.fused_conv(x).permute(0, 2, 3, 1).contiguous()
        x = self.act1(self.norm1(x).permute(0, 3, 1, 2).contiguous())
        x = self.se(x)
        x = self.conv3_1x1(x).permute(0, 2, 3, 1).contiguous().view(b, h * w, c)
        return x + shortcut


class AffineTransform(nn.Module):
    def __init__(self, dim, window_size, num_heads, qk_scale=None, attn_drop=0., position_bias=True):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.position_bias = position_bias
        if self.position_bias:
            self.pos = DynamicPosBias(self.dim // 4, self.num_heads)
        self.attn_drop = nn.Dropout(attn_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, h, w):
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        group_size = (h, w)
        if self.position_bias:
            position_bias_h = torch.arange(1 - group_size[0], group_size[0], device=attn.device)
            position_bias_w = torch.arange(1 - group_size[1], group_size[1], device=attn.device)
            biases = torch.stack(torch.meshgrid([position_bias_h, position_bias_w], indexing='ij'))
            biases = biases.flatten(1).transpose(0, 1).contiguous().float()

            coords_h = torch.arange(group_size[0], device=attn.device)
            coords_w = torch.arange(group_size[1], device=attn.device)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += group_size[0] - 1
            relative_coords[:, :, 1] += group_size[1] - 1
            relative_coords[:, :, 0] *= 2 * group_size[1] - 1
            relative_position_index = relative_coords.sum(-1)

            pos = self.pos(biases)
            relative_position_bias = pos[relative_position_index.view(-1)].view(
                group_size[0] * group_size[1], group_size[0] * group_size[1], -1)
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
            attn = attn + relative_position_bias.unsqueeze(0)

        attn = self.softmax(attn)
        x = self.attn_drop(attn)
        x = x @ v
        return x


class GridAttention(nn.Module):
    def __init__(self, window_size, dim, num_heads, qk_scale=None, attn_drop=0., position_bias=True):
        super().__init__()
        self.window_size = window_size
        self.dim = dim
        self.num_heads = num_heads
        self.attn_transform1 = AffineTransform(dim, window_size=to_2tuple(self.window_size),
                                               num_heads=num_heads, qk_scale=qk_scale,
                                               attn_drop=attn_drop, position_bias=position_bias)
        self.attn_transform2 = AffineTransform(dim, window_size=to_2tuple(self.window_size),
                                               num_heads=num_heads, qk_scale=qk_scale,
                                               attn_drop=attn_drop, position_bias=position_bias)

    def forward(self, qkv, grid, h, w):
        b_, n, c = grid.shape
        qkv = qkv.reshape(b_, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        grid = grid.reshape(b_, n, self.num_heads, -1).permute(0, 2, 1, 3)

        q, k, v = qkv[0], qkv[1], qkv[2]
        x = self.attn_transform1(grid, k, v, h, w)
        x = self.attn_transform2(q, grid, x, h, w)
        x = x.transpose(1, 2).reshape(b_, n, c)
        return x


class GAB(nn.Module):
    def __init__(self, window_size, interval_size, dim, num_heads, qkv_bias=True,
                 qk_scale=None, attn_drop=0., drop=0., drop_path=0., mlp_ratio=2):
        super().__init__()
        self.window_size = window_size
        self.interval_size = interval_size
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.grid_proj = nn.Linear(dim, dim // 2)
        self.shift_size = window_size // 2

        self.grid_attn = GridAttention(
            window_size, dim // 2, num_heads=num_heads // 2, qk_scale=qk_scale, attn_drop=attn_drop)
        self.window_attn = WindowAttention(
            dim // 4, window_size=to_2tuple(self.window_size), num_heads=num_heads // 2,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.window_attn_s = WindowAttention(
            dim // 4, window_size=to_2tuple(self.window_size), num_heads=num_heads // 2,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.fc = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        mip_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mip_hidden_dim, act_layer=nn.GELU, drop=drop)

    def forward(self, x, x_size, rpi_sa, mask):
        h, w = x_size
        b, _, c = x.shape
        shortcut = x

        qkv = self.qkv(x)
        x_window, x_qkv = torch.split(qkv, c * 3 // 2, dim=-1)

        x = x.view(b, h, w, c)
        Gh, Gw = h // self.interval_size, w // self.interval_size
        x_grid = self.grid_proj(grid_shuffle(x, h, w, c, self.interval_size).view(-1, Gh * Gw, c))
        x_qkv = grid_shuffle(x_qkv, h, w, c * 3 // 2, self.interval_size).view(-1, Gh * Gw, c * 3 // 2)

        x_grid_attn = self.grid_attn(x_qkv, x_grid, Gh, Gw).view(-1, Gh, Gw, c // 2)
        x_grid_attn = grid_unshuffle(x_grid_attn, b, h, w, self.interval_size).view(b, h * w, c // 2)

        x_window, x_window_s = torch.split(x_window.view(b, h, w, c * 3 // 2), c * 3 // 4, dim=-1)
        x_window = window_partition(x_window, self.window_size)
        x_window = x_window.view(-1, self.window_size * self.window_size, c * 3 // 4)

        x_window_s = torch.roll(x_window_s, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        x_window_s = x_window_s.view(-1, self.window_size * self.window_size, c * 3 // 4)

        x_win_attn = self.window_attn(x_window, rpi=rpi_sa, mask=None).view(-1, self.window_size, self.window_size, c // 4)
        x_win_attn = window_reverse(x_win_attn, self.window_size, h, w).view(b, h * w, c // 4)

        x_win_s_attn = self.window_attn_s(x_window_s, rpi=rpi_sa, mask=mask).view(-1, self.window_size, self.window_size, c // 4)
        x_win_s_attn = window_reverse(x_win_s_attn, self.window_size, h, w).view(b, h * w, c // 4)
        x_win_s_attn = torch.roll(x_win_s_attn, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        x_win_attn = torch.cat([x_win_attn, x_win_s_attn], dim=-1)
        x = torch.cat([x_win_attn, x_grid_attn], dim=-1)
        x = self.norm1(self.fc(x))

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.norm2(self.mlp(x)))
        return x


class AttenBlocks(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size, interval_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        blk = []

        for i in range(depth):
            if i % 2 == 0:
                blk.append(FusedConv(num_feat=dim, expand_size=6, attn_ratio=2))
                blk.append(FAB(
                    dim=dim, input_resolution=input_resolution, num_heads=num_heads,
                    window_size=window_size, shift_size=0, mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer))
            else:
                blk.append(FAB(
                    dim=dim, input_resolution=input_resolution, num_heads=num_heads,
                    window_size=window_size, shift_size=window_size // 2, mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer))
        self.blocks = nn.ModuleList(blk)
        self.gab = GAB(
            window_size=window_size, interval_size=interval_size, dim=dim, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, drop=drop,
            drop_path=0., mlp_ratio=mlp_ratio)

        self.scale = nn.Parameter(torch.empty(dim))
        trunc_normal_(self.scale, std=.02)
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, x_size, params):
        for blk in self.blocks:
            x = blk(x, x_size, params['rpi_sa'], params['attn_mask'])
        y = self.gab(x, x_size, params['rpi_sa'], params['attn_mask'])
        x = x + y * self.scale
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchUnEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        x = x.transpose(1, 2).contiguous().view(x.shape[0], self.embed_dim, x_size[0], x_size[1])
        return x


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
            raise ValueError(f'scale {scale} is not supported.')
        super(Upsample, self).__init__(*m)


class RHTB(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size, interval_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 img_size=224, patch_size=4, resi_connection='1conv'):
        super(RHTB, self).__init__()
        self.dim = dim
        self.input_resolution = input_resolution

        self.residual_group = AttenBlocks(
            dim=dim, input_resolution=input_resolution, depth=depth, num_heads=num_heads,
            window_size=window_size, interval_size=interval_size, mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop,
            drop_path=drop_path, norm_layer=norm_layer, downsample=downsample,
            use_checkpoint=use_checkpoint)

        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == 'identity':
            self.conv = nn.Identity()

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)
        self.patch_unembed = PatchUnEmbed(img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)

    def forward(self, x, x_size, params):
        return self.patch_embed(self.conv(self.patch_unembed(self.residual_group(x, x_size, params), x_size))) + x


class HMANet(nn.Module):
    def __init__(self, img_size=64, patch_size=1, in_chans=3, embed_dim=96, depths=(6, 6, 6, 6),
                 num_heads=(6, 6, 6, 6), window_size=7, interval_size=4, mlp_ratio=4.,
                 qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True, use_checkpoint=False,
                 upscale=4, img_range=1., upsampler='pixelshuffle', resi_connection='1conv', **kwargs):
        super(HMANet, self).__init__()

        self.window_size = window_size
        self.shift_size = window_size // 2

        num_in_ch = in_chans
        num_out_ch = in_chans
        num_feat = 64
        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        self.upscale = upscale
        self.upsampler = upsampler

        relative_position_index_SA = self.calculate_rpi_sa()
        self.register_buffer('relative_position_index_SA', relative_position_index_SA)

        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim,
            embed_dim=embed_dim, norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim,
            embed_dim=embed_dim, norm_layer=norm_layer if self.patch_norm else None)

        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = RHTB(
                dim=embed_dim, input_resolution=(patches_resolution[0], patches_resolution[1]),
                depth=depths[i_layer], num_heads=num_heads[i_layer], window_size=window_size,
                interval_size=interval_size, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer, downsample=None, use_checkpoint=use_checkpoint,
                img_size=img_size, patch_size=patch_size, resi_connection=resi_connection)
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)

        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == 'identity':
            self.conv_after_body = nn.Identity()

        if self.upsampler == 'pixelshuffle':
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True))
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
            assert upscale in [1, 4]
            if upscale == 1:
                self.conv_last_tail = nn.Sequential(
                    nn.Conv2d(num_feat, num_feat, 3, 1, 1),
                    nn.LeakyReLU(inplace=True),
                    nn.Conv2d(num_feat, num_feat, 3, 1, 1),
                    nn.LeakyReLU(inplace=True),
                    nn.Conv2d(num_feat, num_out_ch, 3, 1, 1),
                )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def calculate_rpi_sa(self):
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size - 1
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)
        return relative_position_index

    def calculate_mask(self, x_size):
        h, w = x_size
        img_mask = torch.zeros((1, h, w, 1))
        h_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        x_size = (x.shape[2], x.shape[3])
        attn_mask = self.calculate_mask(x_size).to(x.device)
        params = {'attn_mask': attn_mask, 'rpi_sa': self.relative_position_index_SA}

        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x, x_size, params)

        x = self.norm(x)
        x = self.patch_unembed(x, x_size)
        return x

    def forward(self, x):
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        if self.upsampler == 'pixelshuffle':
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.conv_before_upsample(x)
            x = self.conv_last(self.upsample(x))

        x = x / self.img_range + self.mean
        return x


def get_network(model_hp):
    """
    根据超参数创建 HMANet 网络
    model_hp: ModelParams 对象，包含 embed_dim, depths, num_heads 等参数
    """
    net = HMANet(
        img_size=64,
        patch_size=1,
        in_chans=3,
        embed_dim=model_hp.embed_dim,
        depths=tuple(model_hp.depths),
        num_heads=tuple(model_hp.num_heads),
        window_size=model_hp.window_size,
        interval_size=model_hp.interval_size,
        mlp_ratio=model_hp.mlp_ratio,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=model_hp.drop_path_rate,
        upscale=4,
        img_range=1.0,
        upsampler="pixelshuffle",
        resi_connection="1conv",
    )
    return net
