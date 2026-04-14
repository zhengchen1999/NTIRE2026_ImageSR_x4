"""
DAT - Dual Aggregation Transformer for Image Super-Resolution
==============================================================
ICCV 2023 Paper: https://arxiv.org/abs/2308.03364
Official Repo: https://github.com/zhengchen1999/DAT

Key Components:
- Spatial Window Self-Attention (SW-SA): Local window attention
- Channel-wise Self-Attention (CW-SA): Global channel attention
- Adaptive Interaction Module (AIM): Fuses spatial and channel features
- Spatial-Gate Feed-Forward Network (SGFN): Non-linear spatial gating
- DATB: Dual Aggregation Transformer Block (alternates spatial/channel)

Author: Adapted for NTIRE 2025 SR Challenge
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from typing import Optional, Tuple, List
import numpy as np

try:
    from timm.models.layers import DropPath, trunc_normal_
except ImportError:
    from torch.nn import Identity as DropPath
    def trunc_normal_(tensor, std=0.02):
        nn.init.normal_(tensor, std=std)

try:
    from einops import rearrange
    from einops.layers.torch import Rearrange
    EINOPS_AVAILABLE = True
except ImportError:
    EINOPS_AVAILABLE = False
    # Fallback implementations
    class Rearrange(nn.Module):
        def __init__(self, pattern):
            super().__init__()
            self.pattern = pattern
        def forward(self, x):
            if 'b c h w -> b (h w) c' in self.pattern:
                b, c, h, w = x.shape
                return x.permute(0, 2, 3, 1).reshape(b, h*w, c)
            return x
    
    def rearrange(x, pattern, **kwargs):
        if 'b (h w) c -> b c h w' in pattern:
            h = kwargs.get('h')
            w = kwargs.get('w', h)
            b, _, c = x.shape
            return x.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        return x


# =============================================================================
# Helper Functions
# =============================================================================

def img2windows(img: torch.Tensor, H_sp: int, W_sp: int) -> torch.Tensor:
    """
    Partition image into windows.
    
    Args:
        img: (B, C, H, W)
        H_sp: Window height
        W_sp: Window width
    
    Returns:
        Windows: (B', N, C) where B' = B * num_windows, N = H_sp * W_sp
    """
    B, C, H, W = img.shape
    img_reshape = img.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
    img_perm = img_reshape.permute(0, 2, 4, 3, 5, 1).contiguous()
    img_perm = img_perm.reshape(-1, H_sp * W_sp, C)
    return img_perm


def windows2img(img_splits: torch.Tensor, H_sp: int, W_sp: int, H: int, W: int) -> torch.Tensor:
    """
    Merge windows back to image.
    
    Args:
        img_splits: (B', N, C)
        H_sp, W_sp: Window size
        H, W: Original image size
    
    Returns:
        Image: (B, H, W, C)
    """
    B = int(img_splits.shape[0] / (H * W / H_sp / W_sp))
    img = img_splits.view(B, H // H_sp, W // W_sp, H_sp, W_sp, -1)
    img = img.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return img


# =============================================================================
# Spatial Gate (for SGFN)
# =============================================================================

class SpatialGate(nn.Module):
    """Spatial-Gate with depthwise convolution."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)
    
    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        # Split into two parts
        x1, x2 = x.chunk(2, dim=-1)
        B, N, C = x.shape
        
        # Apply depthwise conv to second part
        x2 = self.norm(x2)
        x2 = x2.transpose(1, 2).contiguous().view(B, C // 2, H, W)
        x2 = self.conv(x2)
        x2 = x2.flatten(2).transpose(-1, -2).contiguous()
        
        # Gating
        return x1 * x2


# =============================================================================
# Spatial-Gate Feed-Forward Network (SGFN)
# =============================================================================

class SGFN(nn.Module):
    """
    Spatial-Gate Feed-Forward Network.
    
    Introduces spatial information via gating mechanism.
    """
    
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: nn.Module = nn.GELU,
        drop: float = 0.0
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.sg = SpatialGate(hidden_features // 2)
        self.fc2 = nn.Linear(hidden_features // 2, out_features)
        self.drop = nn.Dropout(drop)
    
    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Args:
            x: (B, H*W, C)
            H, W: Spatial dimensions
        Returns:
            x: (B, H*W, C)
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.sg(x, H, W)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# =============================================================================
# Dynamic Position Bias
# =============================================================================

class DynamicPosBias(nn.Module):
    """Dynamic Relative Position Bias (from CrossFormer)."""
    
    def __init__(self, dim: int, num_heads: int, residual: bool = False):
        super().__init__()
        self.residual = residual
        self.num_heads = num_heads
        # Match official DAT pos_dim calculation exactly
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
    
    def forward(self, biases: torch.Tensor) -> torch.Tensor:
        if self.residual:
            pos = self.pos_proj(biases)
            pos = pos + self.pos1(pos)
            pos = pos + self.pos2(pos)
            pos = self.pos3(pos)
        else:
            pos = self.pos3(self.pos2(self.pos1(self.pos_proj(biases))))
        return pos


# =============================================================================
# Spatial Window Self-Attention
# =============================================================================

class SpatialAttention(nn.Module):
    """
    Spatial Window Self-Attention.
    
    Supports rectangular windows.
    """
    
    def __init__(
        self,
        dim: int,
        idx: int,
        split_size: List[int] = [8, 8],
        dim_out: Optional[int] = None,
        num_heads: int = 6,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        qk_scale: Optional[float] = None,
        position_bias: bool = True
    ):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out or dim
        self.split_size = split_size
        self.num_heads = num_heads
        self.idx = idx
        self.position_bias = position_bias
        
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        
        # Set window size based on index
        if idx == 0:
            self.H_sp, self.W_sp = split_size[0], split_size[1]
        else:
            self.W_sp, self.H_sp = split_size[0], split_size[1]
        
        # Dynamic position bias
        if self.position_bias:
            self.pos = DynamicPosBias(self.dim // 4, self.num_heads, residual=False)
            
            # Generate position bias table
            position_bias_h = torch.arange(1 - self.H_sp, self.H_sp)
            position_bias_w = torch.arange(1 - self.W_sp, self.W_sp)
            biases = torch.stack(torch.meshgrid([position_bias_h, position_bias_w], indexing='ij'))
            biases = biases.flatten(1).transpose(0, 1).contiguous().float()
            self.register_buffer('rpe_biases', biases)
            
            # Relative position index
            coords_h = torch.arange(self.H_sp)
            coords_w = torch.arange(self.W_sp)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += self.H_sp - 1
            relative_coords[:, :, 1] += self.W_sp - 1
            relative_coords[:, :, 0] *= 2 * self.W_sp - 1
            relative_position_index = relative_coords.sum(-1)
            self.register_buffer('relative_position_index', relative_position_index)
        
        self.attn_drop = nn.Dropout(attn_drop)
    
    def im2win(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """Convert image to windows."""
        B, N, C = x.shape
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        x = img2windows(x, self.H_sp, self.W_sp)
        x = x.reshape(-1, self.H_sp * self.W_sp, self.num_heads, C // self.num_heads)
        x = x.permute(0, 2, 1, 3).contiguous()
        return x
    
    def forward(
        self,
        qkv: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        H: int,
        W: int,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            qkv: Tuple of (q, k, v), each (B, L, C)
            H, W: Spatial dimensions
            mask: Attention mask
        Returns:
            x: (B, H, W, C)
        """
        q, k, v = qkv[0], qkv[1], qkv[2]
        B, L, C = q.shape
        
        # Partition to windows
        q = self.im2win(q, H, W)
        k = self.im2win(k, H, W)
        v = self.im2win(v, H, W)
        
        # Attention
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        
        # Add position bias
        if self.position_bias:
            pos = self.pos(self.rpe_biases)
            relative_position_bias = pos[self.relative_position_index.view(-1)].view(
                self.H_sp * self.W_sp, self.H_sp * self.W_sp, -1
            )
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
            attn = attn + relative_position_bias.unsqueeze(0)
        
        # Apply mask
        N = attn.shape[3]
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        
        attn = F.softmax(attn, dim=-1, dtype=attn.dtype)
        attn = self.attn_drop(attn)
        
        x = attn @ v
        x = x.transpose(1, 2).reshape(-1, self.H_sp * self.W_sp, C)
        
        # Merge windows
        x = windows2img(x, self.H_sp, self.W_sp, H, W)
        
        return x


# =============================================================================
# Adaptive Spatial Attention
# =============================================================================

class AdaptiveSpatialAttention(nn.Module):
    """
    Adaptive Spatial Self-Attention with AIM (Adaptive Interaction Module).
    
    Combines window attention with depthwise convolution via channel/spatial interaction.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        reso: int = 64,
        split_size: List[int] = [8, 8],
        shift_size: List[int] = [1, 2],
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        rg_idx: int = 0,
        b_idx: int = 0
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.split_size = split_size
        self.shift_size = shift_size
        self.b_idx = b_idx
        self.rg_idx = rg_idx
        self.patches_resolution = reso
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.branch_num = 2
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)
        
        # Two spatial attention branches
        self.attns = nn.ModuleList([
            SpatialAttention(
                dim // 2, idx=i,
                split_size=split_size, num_heads=num_heads // 2, dim_out=dim // 2,
                qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, position_bias=True
            ) for i in range(self.branch_num)
        ])
        
        # Attention mask for shifted windows
        if self._should_shift():
            attn_mask = self.calculate_mask(self.patches_resolution, self.patches_resolution)
            self.register_buffer("attn_mask_0", attn_mask[0])
            self.register_buffer("attn_mask_1", attn_mask[1])
        else:
            self.register_buffer("attn_mask_0", None)
            self.register_buffer("attn_mask_1", None)
        
        # Depthwise convolution branch
        self.dwconv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )
        
        # AIM: Channel interaction (matches official DAT architecture)
        self.channel_interaction = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 8, kernel_size=1),
            nn.BatchNorm2d(dim // 8),
            nn.GELU(),
            nn.Conv2d(dim // 8, dim, kernel_size=1),
        )
        
        # AIM: Spatial interaction (matches official DAT architecture)
        self.spatial_interaction = nn.Sequential(
            nn.Conv2d(dim, dim // 16, kernel_size=1),
            nn.BatchNorm2d(dim // 16),
            nn.GELU(),
            nn.Conv2d(dim // 16, 1, kernel_size=1)
        )
    
    def _should_shift(self) -> bool:
        """Determine if this block should use shifted windows."""
        return ((self.rg_idx % 2 == 0 and self.b_idx > 0 and (self.b_idx - 2) % 4 == 0) or
                (self.rg_idx % 2 != 0 and self.b_idx % 4 == 0))
    
    def calculate_mask(self, H: int, W: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate attention mask for shifted windows."""
        img_mask_0 = torch.zeros((1, H, W, 1))
        img_mask_1 = torch.zeros((1, H, W, 1))
        
        h_slices_0 = (
            slice(0, -self.split_size[0]),
            slice(-self.split_size[0], -self.shift_size[0]),
            slice(-self.shift_size[0], None)
        )
        w_slices_0 = (
            slice(0, -self.split_size[1]),
            slice(-self.split_size[1], -self.shift_size[1]),
            slice(-self.shift_size[1], None)
        )
        h_slices_1 = (
            slice(0, -self.split_size[1]),
            slice(-self.split_size[1], -self.shift_size[1]),
            slice(-self.shift_size[1], None)
        )
        w_slices_1 = (
            slice(0, -self.split_size[0]),
            slice(-self.split_size[0], -self.shift_size[0]),
            slice(-self.shift_size[0], None)
        )
        
        cnt = 0
        for h in h_slices_0:
            for w in w_slices_0:
                img_mask_0[:, h, w, :] = cnt
                cnt += 1
        
        cnt = 0
        for h in h_slices_1:
            for w in w_slices_1:
                img_mask_1[:, h, w, :] = cnt
                cnt += 1
        
        # Calculate mask for window-0
        img_mask_0 = img_mask_0.view(1, H // self.split_size[0], self.split_size[0],
                                      W // self.split_size[1], self.split_size[1], 1)
        img_mask_0 = img_mask_0.permute(0, 1, 3, 2, 4, 5).contiguous()
        img_mask_0 = img_mask_0.view(-1, self.split_size[0], self.split_size[1], 1)
        mask_windows_0 = img_mask_0.view(-1, self.split_size[0] * self.split_size[1])
        attn_mask_0 = mask_windows_0.unsqueeze(1) - mask_windows_0.unsqueeze(2)
        attn_mask_0 = attn_mask_0.masked_fill(attn_mask_0 != 0, float(-100.0))
        attn_mask_0 = attn_mask_0.masked_fill(attn_mask_0 == 0, float(0.0))
        
        # Calculate mask for window-1
        img_mask_1 = img_mask_1.view(1, H // self.split_size[1], self.split_size[1],
                                      W // self.split_size[0], self.split_size[0], 1)
        img_mask_1 = img_mask_1.permute(0, 1, 3, 2, 4, 5).contiguous()
        img_mask_1 = img_mask_1.view(-1, self.split_size[1], self.split_size[0], 1)
        mask_windows_1 = img_mask_1.view(-1, self.split_size[1] * self.split_size[0])
        attn_mask_1 = mask_windows_1.unsqueeze(1) - mask_windows_1.unsqueeze(2)
        attn_mask_1 = attn_mask_1.masked_fill(attn_mask_1 != 0, float(-100.0))
        attn_mask_1 = attn_mask_1.masked_fill(attn_mask_1 == 0, float(0.0))
        
        return attn_mask_0, attn_mask_1
    
    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Args:
            x: (B, H*W, C)
            H, W: Spatial dimensions
        Returns:
            x: (B, H*W, C)
        """
        B, L, C = x.shape
        
        qkv = self.qkv(x).reshape(B, -1, 3, C).permute(2, 0, 1, 3)
        v = qkv[2].transpose(-2, -1).contiguous().view(B, C, H, W)
        
        # Image padding
        max_split_size = max(self.split_size[0], self.split_size[1])
        pad_l = pad_t = 0
        pad_r = (max_split_size - W % max_split_size) % max_split_size
        pad_b = (max_split_size - H % max_split_size) % max_split_size
        
        qkv = qkv.reshape(3 * B, H, W, C).permute(0, 3, 1, 2)
        qkv = F.pad(qkv, (pad_l, pad_r, pad_t, pad_b))
        qkv = qkv.reshape(3, B, C, -1).transpose(-2, -1)
        _H = pad_b + H
        _W = pad_r + W
        
        if self._should_shift():
            qkv = qkv.view(3, B, _H, _W, C)
            qkv_0 = torch.roll(qkv[:, :, :, :, :C // 2], 
                               shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(2, 3))
            qkv_0 = qkv_0.view(3, B, _H * _W, C // 2)
            qkv_1 = torch.roll(qkv[:, :, :, :, C // 2:],
                               shifts=(-self.shift_size[1], -self.shift_size[0]), dims=(2, 3))
            qkv_1 = qkv_1.view(3, B, _H * _W, C // 2)
            
            if self.patches_resolution != _H or self.patches_resolution != _W:
                mask_tmp = self.calculate_mask(_H, _W)
                x1_shift = self.attns[0](qkv_0, _H, _W, mask=mask_tmp[0].to(x.device))
                x2_shift = self.attns[1](qkv_1, _H, _W, mask=mask_tmp[1].to(x.device))
            else:
                x1_shift = self.attns[0](qkv_0, _H, _W, mask=self.attn_mask_0)
                x2_shift = self.attns[1](qkv_1, _H, _W, mask=self.attn_mask_1)
            
            x1 = torch.roll(x1_shift, shifts=(self.shift_size[0], self.shift_size[1]), dims=(1, 2))
            x2 = torch.roll(x2_shift, shifts=(self.shift_size[1], self.shift_size[0]), dims=(1, 2))
            x1 = x1[:, :H, :W, :].reshape(B, L, C // 2)
            x2 = x2[:, :H, :W, :].reshape(B, L, C // 2)
            attened_x = torch.cat([x1, x2], dim=2)
        else:
            x1 = self.attns[0](qkv[:, :, :, :C // 2], _H, _W)[:, :H, :W, :].reshape(B, L, C // 2)
            x2 = self.attns[1](qkv[:, :, :, C // 2:], _H, _W)[:, :H, :W, :].reshape(B, L, C // 2)
            attened_x = torch.cat([x1, x2], dim=2)
        
        # Convolution branch
        conv_x = self.dwconv(v)
        
        # AIM: Adaptive Interaction Module
        # Channel interaction (C-Map)
        channel_map = self.channel_interaction(conv_x).permute(0, 2, 3, 1).contiguous().view(B, 1, C)
        # Spatial interaction (S-Map)
        attention_reshape = attened_x.transpose(-2, -1).contiguous().view(B, C, H, W)
        spatial_map = self.spatial_interaction(attention_reshape)
        
        # Apply interactions
        attened_x = attened_x * torch.sigmoid(channel_map)
        conv_x = torch.sigmoid(spatial_map) * conv_x
        conv_x = conv_x.permute(0, 2, 3, 1).contiguous().view(B, L, C)
        
        x = attened_x + conv_x
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


# =============================================================================
# Adaptive Channel Attention
# =============================================================================

class AdaptiveChannelAttention(nn.Module):
    """
    Adaptive Channel Self-Attention with AIM.
    
    Based on XCiT cross-covariance attention.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0
    ):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # Convolution branch
        self.dwconv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )
        
        # AIM: Channel interaction (matches official DAT architecture)
        self.channel_interaction = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 8, kernel_size=1),
            nn.BatchNorm2d(dim // 8),
            nn.GELU(),
            nn.Conv2d(dim // 8, dim, kernel_size=1),
        )
        # AIM: Spatial interaction (matches official DAT architecture)
        self.spatial_interaction = nn.Sequential(
            nn.Conv2d(dim, dim // 16, kernel_size=1),
            nn.BatchNorm2d(dim // 16),
            nn.GELU(),
            nn.Conv2d(dim // 16, 1, kernel_size=1)
        )
    
    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Args:
            x: (B, H*W, C)
            H, W: Spatial dimensions
        Returns:
            x: (B, H*W, C)
        """
        B, N, C = x.shape
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Channel attention (transpose for channel-wise)
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        
        v_ = v.reshape(B, C, N).contiguous().view(B, C, H, W)
        
        # Normalize
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        attened_x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
        
        # Convolution branch
        conv_x = self.dwconv(v_)
        
        # AIM
        attention_reshape = attened_x.transpose(-2, -1).contiguous().view(B, C, H, W)
        channel_map = self.channel_interaction(attention_reshape)
        spatial_map = self.spatial_interaction(conv_x).permute(0, 2, 3, 1).contiguous().view(B, N, 1)
        
        # Apply interactions (swapped for channel attention)
        attened_x = attened_x * torch.sigmoid(spatial_map)
        conv_x = conv_x * torch.sigmoid(channel_map)
        conv_x = conv_x.permute(0, 2, 3, 1).contiguous().view(B, N, C)
        
        x = attened_x + conv_x
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


# =============================================================================
# DATB - Dual Aggregation Transformer Block
# =============================================================================

class DATB(nn.Module):
    """
    Dual Aggregation Transformer Block.
    
    Alternates between spatial and channel attention.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        reso: int = 64,
        split_size: List[int] = [2, 4],
        shift_size: List[int] = [1, 2],
        expansion_factor: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
        rg_idx: int = 0,
        b_idx: int = 0
    ):
        super().__init__()
        
        self.norm1 = norm_layer(dim)
        
        # Alternate between spatial and channel attention
        if b_idx % 2 == 0:
            # Spatial attention block
            self.attn = AdaptiveSpatialAttention(
                dim, num_heads=num_heads, reso=reso, split_size=split_size,
                shift_size=shift_size, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop, attn_drop=attn_drop, rg_idx=rg_idx, b_idx=b_idx
            )
        else:
            # Channel attention block
            self.attn = AdaptiveChannelAttention(
                dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                attn_drop=attn_drop, proj_drop=drop
            )
        
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        
        # Feed-forward network
        ffn_hidden_dim = int(dim * expansion_factor)
        self.ffn = SGFN(in_features=dim, hidden_features=ffn_hidden_dim, 
                        out_features=dim, act_layer=act_layer)
        self.norm2 = norm_layer(dim)
    
    def forward(self, x: torch.Tensor, x_size: Tuple[int, int]) -> torch.Tensor:
        """
        Args:
            x: (B, H*W, C)
            x_size: (H, W)
        Returns:
            x: (B, H*W, C)
        """
        H, W = x_size
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.ffn(self.norm2(x), H, W))
        return x


# =============================================================================
# Residual Group
# =============================================================================

class ResidualGroup(nn.Module):
    """
    Residual Group containing multiple DATB blocks.
    """
    
    def __init__(
        self,
        dim: int,
        reso: int,
        num_heads: int,
        split_size: List[int] = [2, 4],
        expansion_factor: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_paths: Optional[List[float]] = None,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
        depth: int = 2,
        use_chk: bool = False,
        resi_connection: str = '1conv',
        rg_idx: int = 0
    ):
        super().__init__()
        self.use_chk = use_chk
        self.reso = reso
        
        self.blocks = nn.ModuleList([
            DATB(
                dim=dim,
                num_heads=num_heads,
                reso=reso,
                split_size=split_size,
                shift_size=[split_size[0] // 2, split_size[1] // 2],
                expansion_factor=expansion_factor,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_paths[i] if drop_paths else 0.0,
                act_layer=act_layer,
                norm_layer=norm_layer,
                rg_idx=rg_idx,
                b_idx=i,
            ) for i in range(depth)
        ])
        
        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == '3conv':
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim // 4, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim // 4, 1, 1, 0),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim, 3, 1, 1)
            )
    
    def forward(self, x: torch.Tensor, x_size: Tuple[int, int]) -> torch.Tensor:
        """
        Args:
            x: (B, H*W, C)
            x_size: (H, W)
        Returns:
            x: (B, H*W, C)
        """
        H, W = x_size
        res = x
        
        for blk in self.blocks:
            if self.use_chk:
                x = checkpoint.checkpoint(blk, x, x_size, use_reentrant=False)
            else:
                x = blk(x, x_size)
        
        # Reshape for conv
        x = x.view(-1, H, W, x.shape[-1]).permute(0, 3, 1, 2).contiguous()
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1).contiguous().view(-1, H * W, x.shape[1])
        
        x = res + x
        return x


# =============================================================================
# Upsample Modules
# =============================================================================

class Upsample(nn.Sequential):
    """Upsample module using PixelShuffle."""
    
    def __init__(self, scale: int, num_feat: int):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'Scale {scale} is not supported. Supported: 2^n and 3.')
        super().__init__(*m)


class UpsampleOneStep(nn.Sequential):
    """Lightweight upsample (single conv + pixelshuffle)."""
    
    def __init__(self, scale: int, num_feat: int, num_out_ch: int):
        m = [
            nn.Conv2d(num_feat, (scale ** 2) * num_out_ch, 3, 1, 1),
            nn.PixelShuffle(scale)
        ]
        super().__init__(*m)


# =============================================================================
# DAT - Main Model
# =============================================================================

class DAT(nn.Module):
    """
    Dual Aggregation Transformer for Image Super-Resolution.
    
    ICCV 2023: https://arxiv.org/abs/2308.03364
    
    Args:
        img_size: Input image size (default: 64)
        in_chans: Number of input channels (default: 3)
        embed_dim: Embedding dimension (default: 180)
        depths: Depth of each residual group
        num_heads: Number of attention heads in each group
        split_size: Window split size [H_sp, W_sp]
        expansion_factor: MLP expansion factor
        upscale: Upscale factor (2/3/4)
        img_range: Image range (1.0 or 255.0)
        upsampler: 'pixelshuffle' or 'pixelshuffledirect'
    """
    
    def __init__(
        self,
        img_size: int = 64,
        in_chans: int = 3,
        embed_dim: int = 180,
        split_size: List[int] = [2, 4],
        depth: List[int] = [6, 6, 6, 6, 6, 6],
        num_heads: List[int] = [6, 6, 6, 6, 6, 6],
        expansion_factor: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
        use_chk: bool = False,
        upscale: int = 4,
        img_range: float = 1.0,
        resi_connection: str = '1conv',
        upsampler: str = 'pixelshuffle',
        **kwargs
    ):
        super().__init__()
        
        num_in_ch = in_chans
        num_out_ch = in_chans
        num_feat = 64
        self.img_range = img_range
        self.upscale = upscale
        self.upsampler = upsampler
        
        # Mean for normalization
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        
        # 1. Shallow Feature Extraction
        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)
        
        # 2. Deep Feature Extraction
        self.num_layers = len(depth)
        self.use_chk = use_chk
        self.num_features = self.embed_dim = embed_dim
        
        self.before_RG = nn.Sequential(
            Rearrange('b c h w -> b (h w) c'),
            nn.LayerNorm(embed_dim)
        )
        
        # Stochastic depth decay
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]
        
        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            layer = ResidualGroup(
                dim=embed_dim,
                num_heads=num_heads[i],
                reso=img_size,
                split_size=split_size,
                expansion_factor=expansion_factor,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_paths=dpr[sum(depth[:i]):sum(depth[:i + 1])],
                act_layer=act_layer,
                norm_layer=norm_layer,
                depth=depth[i],
                use_chk=use_chk,
                resi_connection=resi_connection,
                rg_idx=i
            )
            self.layers.append(layer)
        
        self.norm = norm_layer(embed_dim)
        
        # Post-body conv
        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == '3conv':
            self.conv_after_body = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1)
            )
        
        # 3. Reconstruction
        if self.upsampler == 'pixelshuffle':
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1),
                nn.LeakyReLU(inplace=True)
            )
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        elif self.upsampler == 'pixelshuffledirect':
            self.upsample = UpsampleOneStep(upscale, embed_dim, num_out_ch)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Deep feature extraction."""
        _, _, H, W = x.shape
        x_size = (H, W)
        x = self.before_RG(x)
        for layer in self.layers:
            x = layer(x, x_size)
        x = self.norm(x)
        x = x.view(-1, H, W, x.shape[-1]).permute(0, 3, 1, 2).contiguous()
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) input LR image
        Returns:
            SR image (B, C, H*scale, W*scale)
        """
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range
        
        if self.upsampler == 'pixelshuffle':
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.conv_before_upsample(x)
            x = self.conv_last(self.upsample(x))
        elif self.upsampler == 'pixelshuffledirect':
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.upsample(x)
        
        x = x / self.img_range + self.mean
        return x


# =============================================================================
# Factory Functions
# =============================================================================

def create_dat_model(
    upscale: int = 4,
    embed_dim: int = 180,
    depths: List[int] = [6, 6, 6, 6, 6, 6],
    num_heads: List[int] = [6, 6, 6, 6, 6, 6],
    split_size: List[int] = [8, 16],
    img_size: int = 64,
    img_range: float = 1.0,
    expansion_factor: float = 2.0,
    resi_connection: str = '1conv',
    use_checkpoint: bool = False,
    **kwargs
) -> DAT:
    """
    Create DAT model with specified configuration.
    
    Default is DAT-S (small) configuration from the paper.
    
    Args:
        upscale: Super-resolution scale factor
        embed_dim: Embedding dimension
        depths: Number of blocks in each residual group
        num_heads: Number of attention heads in each group
        split_size: Window split size [H, W]
        img_size: Input image size
        img_range: Image value range
        expansion_factor: FFN expansion factor
        resi_connection: '1conv' or '3conv'
        use_checkpoint: Use gradient checkpointing
    
    Returns:
        DAT model instance
    """
    model = DAT(
        upscale=upscale,
        in_chans=3,
        img_size=img_size,
        img_range=img_range,
        depth=depths,
        embed_dim=embed_dim,
        num_heads=num_heads,
        expansion_factor=expansion_factor,
        resi_connection=resi_connection,
        split_size=split_size,
        use_chk=use_checkpoint,
        **kwargs
    )
    return model


def create_dat_light(upscale: int = 4, **kwargs) -> DAT:
    """Create DAT-light model (lightweight version)."""
    return create_dat_model(
        upscale=upscale,
        embed_dim=60,
        depths=[6, 6, 6, 6],
        num_heads=[6, 6, 6, 6],
        split_size=[8, 16],
        expansion_factor=2.0,
        upsampler='pixelshuffledirect',
        **kwargs
    )


def create_dat_s(upscale: int = 4, **kwargs) -> DAT:
    """Create DAT-S model (standard)."""
    return create_dat_model(
        upscale=upscale,
        embed_dim=180,
        depths=[6, 6, 6, 6, 6, 6],
        num_heads=[6, 6, 6, 6, 6, 6],
        split_size=[8, 16],
        expansion_factor=2.0,
        **kwargs
    )


def create_dat_2(upscale: int = 4, **kwargs) -> DAT:
    """Create DAT-2 model (larger version)."""
    return create_dat_model(
        upscale=upscale,
        embed_dim=180,
        depths=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
        num_heads=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
        split_size=[8, 32],
        expansion_factor=2.0,
        **kwargs
    )


# =============================================================================
# Test
# =============================================================================

if __name__ == '__main__':
    # Test DAT model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Testing DAT on {device}")
    
    model = create_dat_s(upscale=4).to(device)
    print(f"Model created: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    x = torch.randn(1, 3, 64, 64).to(device)
    with torch.no_grad():
        y = model(x)
    
    print(f"Input: {x.shape} -> Output: {y.shape}")
    assert y.shape == (1, 3, 256, 256), f"Expected (1, 3, 256, 256), got {y.shape}"
    print("✓ DAT test passed!")
