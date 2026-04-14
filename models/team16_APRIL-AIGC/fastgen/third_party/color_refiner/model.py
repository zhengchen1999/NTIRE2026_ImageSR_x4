from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.checkpoint import checkpoint
from einops import rearrange

from .model_rrdb import RRDB


def _resolve_base_indices(input_mask: torch.Tensor, base_input_index: int) -> torch.Tensor:
    valid_counts = input_mask.sum(dim=1).long().clamp_min(1)
    if base_input_index >= 0:
        return torch.minimum(torch.full_like(valid_counts, base_input_index), valid_counts - 1)
    return (valid_counts + base_input_index).clamp_min(0)


def _resolve_base_index(num_inputs: int, base_input_index: int) -> int:
    if num_inputs <= 0:
        raise ValueError("num_inputs must be positive")
    if base_input_index >= 0:
        return min(base_input_index, num_inputs - 1)
    return max(num_inputs + base_input_index, 0)


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        variance = (x - mean).pow(2).mean(dim=1, keepdim=True)
        normalized = (x - mean) / torch.sqrt(variance + self.eps)
        return normalized * self.weight[:, None, None] + self.bias[:, None, None]


class FeedForward(nn.Module):
    def __init__(self, dim: int, expansion_factor: float) -> None:
        super().__init__()
        hidden_dim = int(dim * expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_dim * 2, kernel_size=1)
        self.depthwise = nn.Conv2d(
            hidden_dim * 2,
            hidden_dim * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_dim * 2,
        )
        self.project_out = nn.Conv2d(hidden_dim, dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.project_in(x)
        x1, x2 = self.depthwise(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        return self.project_out(x)


def _normalize_stage_values(values: float | Sequence[float], *, stages: int, name: str) -> tuple[float, ...]:
    if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        normalized = tuple(float(value) for value in values)
    else:
        normalized = (float(values),)
    if len(normalized) == 1:
        return normalized * stages
    if len(normalized) != stages:
        raise ValueError(f"{name} expects either 1 value or {stages} values, got {len(normalized)}")
    return normalized


def _normalize_stage_dims(stage_dims: Sequence[int] | None, *, embed_dim: int, stages: int = 4) -> tuple[int, ...]:
    if stage_dims is None:
        return tuple(embed_dim * (2**index) for index in range(stages))

    normalized = tuple(int(value) for value in stage_dims)
    if len(normalized) != stages:
        raise ValueError(f"stage_dims expects exactly {stages} values, got {len(normalized)}")
    if any(value <= 0 for value in normalized):
        raise ValueError(f"stage_dims values must be positive, got {normalized}")
    return normalized


class RestormerAttnProcessor2_0:
    def __call__(self, attn: "Attention", q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        scale = attn.temperature.view(1, attn.num_heads, 1, 1) * (q.shape[-1] ** 0.5)
        query = q * scale
        with sdpa_kernel(SDPBackend.MATH):
            hidden_states = F.scaled_dot_product_attention(
                query,
                k,
                v,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
            )
        return hidden_states.type_as(q)


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1)
        self.processor = RestormerAttnProcessor2_0()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, height, width = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        k = rearrange(k, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        v = rearrange(v, "b (head c) h w -> b head c (h w)", head=self.num_heads)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        hidden_states = self.processor(self, q, k, v)
        hidden_states = hidden_states.type_as(x)
        hidden_states = rearrange(hidden_states, "b head c (h w) -> b (head c) h w", h=height, w=width)
        return self.project_out(hidden_states)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, expansion_factor: float) -> None:
        super().__init__()
        self.norm1 = LayerNorm2d(dim)
        self.attention = Attention(dim, num_heads)
        self.norm2 = LayerNorm2d(dim)
        self.feed_forward = FeedForward(dim, expansion_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm1(x))
        x = x + self.feed_forward(self.norm2(x))
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.body = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * 4, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class InputFusionStem2(nn.Module):
    def __init__(self, embed_dim: int, base_input_index: int = -1) -> None:
        super().__init__()
        self.base_input_index = base_input_index

        self.encoder = nn.Sequential(
            nn.Conv2d(3, embed_dim, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, 3, 1, 1),
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(embed_dim*3, embed_dim, 1),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, 3,1,1),
        )

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if inputs.ndim != 5:
            raise ValueError(f"Expected [B, N, C, H, W], got {inputs.shape}")
        if inputs.shape[1] < 2:
            raise ValueError("InputFusionStem2 requires at least two input images")

        base_index = _resolve_base_index(inputs.shape[1], self.base_input_index)
        aux_index = 0 if base_index != 0 else 1

        x1 = inputs[:, aux_index]
        x2 = inputs[:, base_index]

        f1 = self.encoder(x1)
        f2 = self.encoder(x2)

        fused = torch.cat([f1, f2, torch.abs(f1-f2)], dim=1)

        return self.fusion(fused), x2


class InputFusionStem3(nn.Module):
    def __init__(self, embed_dim: int, base_input_index: int = -1) -> None:
        super().__init__()
        self.base_input_index = base_input_index
        self.encoder = nn.Sequential(
            nn.Conv2d(3, embed_dim, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, 3, 1, 1),
        )

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if inputs.ndim != 5:
            raise ValueError(f"Expected [B, N, C, H, W], got {inputs.shape}")
        if inputs.shape[1] < 1:
            raise ValueError("InputFusionStem3 requires at least one input image")

        base_index = _resolve_base_index(inputs.shape[1], self.base_input_index)
        base_input = inputs[:, base_index]
        return self.encoder(base_input), base_input


class LocalRRDBEnhancer(nn.Module):
    def __init__(
        self,
        channels: int,
        num_blocks: int = 0,
        growth_channels: int = 32,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.gradient_checkpointing = bool(gradient_checkpointing)
        total_blocks = max(0, int(num_blocks))
        self.body = nn.ModuleList(RRDB(channels, int(growth_channels)) for _ in range(total_blocks))
        self.body_conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1) if total_blocks > 0 else None

    @property
    def enabled(self) -> bool:
        return len(self.body) > 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return x

        residual = x
        features = x
        for block in self.body:
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                features = checkpoint(block, features, use_reentrant=False)
            else:
                features = block(features)

        assert self.body_conv is not None
        return residual + self.body_conv(features)


class InputFusionStem(nn.Module):
    def __init__(self, embed_dim: int, base_input_index: int) -> None:
        super().__init__()
        hidden_dim = max(embed_dim // 4, 8)
        self.base_input_index = base_input_index
        self.shared_encoder = nn.Sequential(
            nn.Conv2d(3, embed_dim, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1),
        )
        self.score = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(embed_dim, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
        )
        self.mix = nn.Sequential(
            nn.Conv2d(embed_dim * 3 + 1, embed_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, inputs: torch.Tensor, input_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_inputs, channels, height, width = inputs.shape
        encoded = self.shared_encoder(inputs.view(batch_size * num_inputs, channels, height, width))
        encoded = encoded.view(batch_size, num_inputs, -1, height, width)

        valid_mask = input_mask[:, :, None, None, None].to(encoded.dtype)
        logits = self.score(encoded.view(batch_size * num_inputs, -1, height, width)).view(batch_size, num_inputs, 1, 1, 1)
        logits = logits.masked_fill(valid_mask == 0, -1e4)
        weights = torch.softmax(logits, dim=1) * valid_mask
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)

        fused = (encoded * weights).sum(dim=1)

        base_indices = _resolve_base_indices(input_mask, self.base_input_index)
        batch_indices = torch.arange(batch_size, device=inputs.device)
        base_features = encoded[batch_indices, base_indices]
        base_inputs = inputs[batch_indices, base_indices]

        valid_ratio = input_mask.float().mean(dim=1, keepdim=True).view(batch_size, 1, 1, 1).expand(-1, 1, height, width)
        mixed = torch.cat([fused, base_features, torch.abs(fused - base_features), valid_ratio], dim=1)
        return self.mix(mixed), base_inputs


class FusionRestormer(nn.Module):
    def __init__(
        self,
        embed_dim: int = 48,
        stage_dims: Sequence[int] | None = None,
        num_blocks: Sequence[int] = (4, 6, 6, 8),
        num_heads: Sequence[int] = (1, 2, 4, 8),
        expansion_factor: float | Sequence[float] = 2.66,
        refinement_blocks: int = 4,
        base_input_index: int = -1,
        use_residual: bool = True,
        gradient_checkpointing: bool = False,
        rrdb_num_blocks: int = 0,
        rrdb_growth_channels: int = 32,
    ) -> None:
        super().__init__()
        if len(num_blocks) != 4 or len(num_heads) != 4:
            raise ValueError("Restormer expects exactly 4 encoder/decoder stages")
        stage_expansion_factors = _normalize_stage_values(expansion_factor, stages=4, name="expansion_factor")
        level1_dim, level2_dim, level3_dim, level4_dim = _normalize_stage_dims(stage_dims, embed_dim=embed_dim)

        self.base_input_index = base_input_index
        self.use_residual = use_residual
        self.gradient_checkpointing = gradient_checkpointing
        self.pad_multiple = 8

        self.stem = InputFusionStem(embed_dim=level1_dim, base_input_index=base_input_index)
        self.local_enhancer = LocalRRDBEnhancer(
            level1_dim,
            num_blocks=rrdb_num_blocks,
            growth_channels=rrdb_growth_channels,
            gradient_checkpointing=gradient_checkpointing,
        )

        self.encoder_level1 = nn.Sequential(
            *[TransformerBlock(level1_dim, num_heads[0], stage_expansion_factors[0]) for _ in range(num_blocks[0])]
        )
        self.down1_2 = Downsample(level1_dim, level2_dim)

        self.encoder_level2 = nn.Sequential(
            *[TransformerBlock(level2_dim, num_heads[1], stage_expansion_factors[1]) for _ in range(num_blocks[1])]
        )
        self.down2_3 = Downsample(level2_dim, level3_dim)

        self.encoder_level3 = nn.Sequential(
            *[TransformerBlock(level3_dim, num_heads[2], stage_expansion_factors[2]) for _ in range(num_blocks[2])]
        )
        self.down3_4 = Downsample(level3_dim, level4_dim)

        self.latent = nn.Sequential(
            *[TransformerBlock(level4_dim, num_heads[3], stage_expansion_factors[3]) for _ in range(num_blocks[3])]
        )

        self.up4_3 = Upsample(level4_dim, level3_dim)
        self.reduce_level3 = nn.Conv2d(level3_dim * 2, level3_dim, kernel_size=1)
        self.decoder_level3 = nn.Sequential(
            *[TransformerBlock(level3_dim, num_heads[2], stage_expansion_factors[2]) for _ in range(num_blocks[2])]
        )

        self.up3_2 = Upsample(level3_dim, level2_dim)
        self.reduce_level2 = nn.Conv2d(level2_dim * 2, level2_dim, kernel_size=1)
        self.decoder_level2 = nn.Sequential(
            *[TransformerBlock(level2_dim, num_heads[1], stage_expansion_factors[1]) for _ in range(num_blocks[1])]
        )

        self.up2_1 = Upsample(level2_dim, level1_dim)
        self.reduce_level1 = nn.Conv2d(level1_dim * 2, level1_dim, kernel_size=1)
        self.decoder_level1 = nn.Sequential(
            *[TransformerBlock(level1_dim, num_heads[0], stage_expansion_factors[0]) for _ in range(num_blocks[0])]
        )
        self.refinement = nn.Sequential(
            *[TransformerBlock(level1_dim, num_heads[0], stage_expansion_factors[0]) for _ in range(refinement_blocks)]
        )
        self.output = nn.Conv2d(level1_dim, 3, kernel_size=3, stride=1, padding=1)

    def enable_gradient_checkpointing(self) -> None:
        self.gradient_checkpointing = True

    def disable_gradient_checkpointing(self) -> None:
        self.gradient_checkpointing = False

    def _run_blocks(self, blocks: nn.Sequential, x: torch.Tensor) -> torch.Tensor:
        for block in blocks:
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        return x

    def _pad_inputs(self, inputs: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        batch_size, num_inputs, channels, height, width = inputs.shape
        pad_h = (self.pad_multiple - height % self.pad_multiple) % self.pad_multiple
        pad_w = (self.pad_multiple - width % self.pad_multiple) % self.pad_multiple
        if pad_h == 0 and pad_w == 0:
            return inputs, 0, 0
        pad_mode = "reflect" if height > 1 and width > 1 else "replicate"
        flat = inputs.view(batch_size * num_inputs, channels, height, width)
        flat = F.pad(flat, (0, pad_w, 0, pad_h), mode=pad_mode)
        return flat.view(batch_size, num_inputs, channels, height + pad_h, width + pad_w), pad_h, pad_w

    def forward(self, inputs: torch.Tensor, input_mask: torch.Tensor | None = None) -> torch.Tensor:
        if inputs.ndim != 5:
            raise ValueError(f"Expected [B, N, C, H, W], got {inputs.shape}")

        original_height, original_width = inputs.shape[-2:]
        if input_mask is None:
            input_mask = torch.ones(inputs.shape[:2], device=inputs.device, dtype=inputs.dtype)

        padded_inputs, pad_h, pad_w = self._pad_inputs(inputs)
        features, base_input = self.stem(padded_inputs, input_mask.float())
        features = self.local_enhancer(features)

        level1 = self._run_blocks(self.encoder_level1, features)
        level2 = self._run_blocks(self.encoder_level2, self.down1_2(level1))
        level3 = self._run_blocks(self.encoder_level3, self.down2_3(level2))
        level4 = self._run_blocks(self.latent, self.down3_4(level3))

        level3_decode = self.up4_3(level4)
        level3_decode = self.reduce_level3(torch.cat([level3_decode, level3], dim=1))
        level3_decode = self._run_blocks(self.decoder_level3, level3_decode)

        level2_decode = self.up3_2(level3_decode)
        level2_decode = self.reduce_level2(torch.cat([level2_decode, level2], dim=1))
        level2_decode = self._run_blocks(self.decoder_level2, level2_decode)

        level1_decode = self.up2_1(level2_decode)
        level1_decode = self.reduce_level1(torch.cat([level1_decode, level1], dim=1))
        level1_decode = self._run_blocks(self.decoder_level1, level1_decode)
        level1_decode = self._run_blocks(self.refinement, level1_decode)

        prediction = self.output(level1_decode)
        if self.use_residual:
            prediction = prediction + base_input
        prediction = prediction.clamp(0.0, 1.0)

        if pad_h > 0 or pad_w > 0:
            prediction = prediction[..., :original_height, :original_width]
        return prediction



class FusionRestormer2(nn.Module):
    def __init__(
        self,
        embed_dim: int = 48,
        stage_dims: Sequence[int] | None = None,
        num_blocks: Sequence[int] = (4, 6, 6, 8),
        num_heads: Sequence[int] = (1, 2, 4, 8),
        expansion_factor: float | Sequence[float] = 2.66,
        refinement_blocks: int = 4,
        base_input_index: int = -1,
        use_residual: bool = True,
        gradient_checkpointing: bool = False,
        rrdb_num_blocks: int = 0,
        rrdb_growth_channels: int = 32,
    ) -> None:
        super().__init__()
        if len(num_blocks) != 4 or len(num_heads) != 4:
            raise ValueError("Restormer expects exactly 4 encoder/decoder stages")
        stage_expansion_factors = _normalize_stage_values(expansion_factor, stages=4, name="expansion_factor")
        level1_dim, level2_dim, level3_dim, level4_dim = _normalize_stage_dims(stage_dims, embed_dim=embed_dim)

        self.base_input_index = base_input_index
        self.use_residual = use_residual
        self.gradient_checkpointing = gradient_checkpointing
        self.pad_multiple = 8

        self.stem = InputFusionStem2(embed_dim=level1_dim, base_input_index=base_input_index)
        self.local_enhancer = LocalRRDBEnhancer(
            level1_dim,
            num_blocks=rrdb_num_blocks,
            growth_channels=rrdb_growth_channels,
            gradient_checkpointing=gradient_checkpointing,
        )

        self.encoder_level1 = nn.Sequential(
            *[TransformerBlock(level1_dim, num_heads[0], stage_expansion_factors[0]) for _ in range(num_blocks[0])]
        )
        self.down1_2 = Downsample(level1_dim, level2_dim)

        self.encoder_level2 = nn.Sequential(
            *[TransformerBlock(level2_dim, num_heads[1], stage_expansion_factors[1]) for _ in range(num_blocks[1])]
        )
        self.down2_3 = Downsample(level2_dim, level3_dim)

        self.encoder_level3 = nn.Sequential(
            *[TransformerBlock(level3_dim, num_heads[2], stage_expansion_factors[2]) for _ in range(num_blocks[2])]
        )
        self.down3_4 = Downsample(level3_dim, level4_dim)

        self.latent = nn.Sequential(
            *[TransformerBlock(level4_dim, num_heads[3], stage_expansion_factors[3]) for _ in range(num_blocks[3])]
        )

        self.up4_3 = Upsample(level4_dim, level3_dim)
        self.reduce_level3 = nn.Conv2d(level3_dim * 2, level3_dim, kernel_size=1)
        self.decoder_level3 = nn.Sequential(
            *[TransformerBlock(level3_dim, num_heads[2], stage_expansion_factors[2]) for _ in range(num_blocks[2])]
        )

        self.up3_2 = Upsample(level3_dim, level2_dim)
        self.reduce_level2 = nn.Conv2d(level2_dim * 2, level2_dim, kernel_size=1)
        self.decoder_level2 = nn.Sequential(
            *[TransformerBlock(level2_dim, num_heads[1], stage_expansion_factors[1]) for _ in range(num_blocks[1])]
        )

        self.up2_1 = Upsample(level2_dim, level1_dim)
        self.reduce_level1 = nn.Conv2d(level1_dim * 2, level1_dim, kernel_size=1)
        self.decoder_level1 = nn.Sequential(
            *[TransformerBlock(level1_dim, num_heads[0], stage_expansion_factors[0]) for _ in range(num_blocks[0])]
        )
        self.refinement = nn.Sequential(
            *[TransformerBlock(level1_dim, num_heads[0], stage_expansion_factors[0]) for _ in range(refinement_blocks)]
        )
        self.output = nn.Conv2d(level1_dim, 3, kernel_size=3, stride=1, padding=1)

    def enable_gradient_checkpointing(self) -> None:
        self.gradient_checkpointing = True

    def disable_gradient_checkpointing(self) -> None:
        self.gradient_checkpointing = False

    def _run_blocks(self, blocks: nn.Sequential, x: torch.Tensor) -> torch.Tensor:
        for block in blocks:
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        return x

    def _pad_inputs(self, inputs: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        batch_size, num_inputs, channels, height, width = inputs.shape
        pad_h = (self.pad_multiple - height % self.pad_multiple) % self.pad_multiple
        pad_w = (self.pad_multiple - width % self.pad_multiple) % self.pad_multiple
        if pad_h == 0 and pad_w == 0:
            return inputs, 0, 0
        pad_mode = "reflect" if height > 1 and width > 1 else "replicate"
        flat = inputs.view(batch_size * num_inputs, channels, height, width)
        flat = F.pad(flat, (0, pad_w, 0, pad_h), mode=pad_mode)
        return flat.view(batch_size, num_inputs, channels, height + pad_h, width + pad_w), pad_h, pad_w

    def forward(self, inputs: torch.Tensor, input_mask: torch.Tensor | None = None) -> torch.Tensor:
        if inputs.ndim != 5:
            raise ValueError(f"Expected [B, N, C, H, W], got {inputs.shape}")

        original_height, original_width = inputs.shape[-2:]
        if input_mask is None:
            input_mask = torch.ones(inputs.shape[:2], device=inputs.device, dtype=inputs.dtype)

        padded_inputs, pad_h, pad_w = self._pad_inputs(inputs)
        features, base_input = self.stem(padded_inputs, input_mask.float())
        features = self.local_enhancer(features)

        level1 = self._run_blocks(self.encoder_level1, features)
        level2 = self._run_blocks(self.encoder_level2, self.down1_2(level1))
        level3 = self._run_blocks(self.encoder_level3, self.down2_3(level2))
        level4 = self._run_blocks(self.latent, self.down3_4(level3))

        level3_decode = self.up4_3(level4)
        level3_decode = self.reduce_level3(torch.cat([level3_decode, level3], dim=1))
        level3_decode = self._run_blocks(self.decoder_level3, level3_decode)

        level2_decode = self.up3_2(level3_decode)
        level2_decode = self.reduce_level2(torch.cat([level2_decode, level2], dim=1))
        level2_decode = self._run_blocks(self.decoder_level2, level2_decode)

        level1_decode = self.up2_1(level2_decode)
        level1_decode = self.reduce_level1(torch.cat([level1_decode, level1], dim=1))
        level1_decode = self._run_blocks(self.decoder_level1, level1_decode)
        level1_decode = self._run_blocks(self.refinement, level1_decode)

        prediction = self.output(level1_decode)
        if self.use_residual:
            prediction = prediction + base_input
        prediction = prediction.clamp(0.0, 1.0)

        if pad_h > 0 or pad_w > 0:
            prediction = prediction[..., :original_height, :original_width]
        return prediction


class FusionRestormer3(FusionRestormer2):
    def __init__(
        self,
        embed_dim: int = 48,
        stage_dims: Sequence[int] | None = None,
        num_blocks: Sequence[int] = (4, 6, 6, 8),
        num_heads: Sequence[int] = (1, 2, 4, 8),
        expansion_factor: float | Sequence[float] = 2.66,
        refinement_blocks: int = 4,
        base_input_index: int = -1,
        use_residual: bool = True,
        gradient_checkpointing: bool = False,
        rrdb_num_blocks: int = 0,
        rrdb_growth_channels: int = 32,
    ) -> None:
        super().__init__(
            embed_dim=embed_dim,
            stage_dims=stage_dims,
            num_blocks=num_blocks,
            num_heads=num_heads,
            expansion_factor=expansion_factor,
            refinement_blocks=refinement_blocks,
            base_input_index=base_input_index,
            use_residual=use_residual,
            gradient_checkpointing=gradient_checkpointing,
            rrdb_num_blocks=rrdb_num_blocks,
            rrdb_growth_channels=rrdb_growth_channels,
        )
        self.stem = InputFusionStem3(embed_dim=self.output.in_channels, base_input_index=base_input_index)


def build_model(model_name: str = "restormer", **kwargs) -> nn.Module:
    normalized_name = model_name.lower()
    if normalized_name in {"fusion_restormer", "restormer"}:
        return FusionRestormer(**kwargs)

    if normalized_name in {"fusion_restormer2", "restormer2"}:
        return FusionRestormer2(**kwargs)
    if normalized_name in {"fusion_restormer3", "restormer3"}:
        return FusionRestormer3(**kwargs)
    raise ValueError(f"Unsupported model: {model_name}")
