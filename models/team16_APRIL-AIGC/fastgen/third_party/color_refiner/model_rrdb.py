from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


def _resolve_base_indices(input_mask: torch.Tensor, base_input_index: int) -> torch.Tensor:
    valid_counts = input_mask.sum(dim=1).long().clamp_min(1)
    if base_input_index >= 0:
        return torch.minimum(torch.full_like(valid_counts, base_input_index), valid_counts - 1)
    return (valid_counts + base_input_index).clamp_min(0)


def _resolve_pair_indices(input_mask: torch.Tensor, base_input_index: int) -> tuple[torch.Tensor, torch.Tensor]:
    base_indices = _resolve_base_indices(input_mask, base_input_index)
    aux_indices = base_indices.clone()
    batch_size = input_mask.shape[0]

    for batch_index in range(batch_size):
        valid_indices = torch.nonzero(input_mask[batch_index] > 0, as_tuple=False).flatten()
        if valid_indices.numel() <= 1:
            continue

        base_index = int(base_indices[batch_index].item())
        aux_index = next((int(index.item()) for index in valid_indices if int(index.item()) != base_index), base_index)
        aux_indices[batch_index] = aux_index

    return base_indices, aux_indices


class ResidualDenseBlock(nn.Module):
    def __init__(self, num_feat: int, growth_channels: int) -> None:
        super().__init__()
        self.activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv1 = nn.Conv2d(num_feat, growth_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(num_feat + growth_channels, growth_channels, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(num_feat + growth_channels * 2, growth_channels, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(num_feat + growth_channels * 3, growth_channels, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(num_feat + growth_channels * 4, num_feat, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.activation(self.conv1(x))
        x2 = self.activation(self.conv2(torch.cat([x, x1], dim=1)))
        x3 = self.activation(self.conv3(torch.cat([x, x1, x2], dim=1)))
        x4 = self.activation(self.conv4(torch.cat([x, x1, x2, x3], dim=1)))
        x5 = self.conv5(torch.cat([x, x1, x2, x3, x4], dim=1))
        return x + x5 * 0.2


class RRDB(nn.Module):
    def __init__(self, num_feat: int, growth_channels: int) -> None:
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, growth_channels)
        self.rdb2 = ResidualDenseBlock(num_feat, growth_channels)
        self.rdb3 = ResidualDenseBlock(num_feat, growth_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.rdb3(self.rdb2(self.rdb1(x)))
        return x + residual * 0.2


class RealESRGANRefiner(nn.Module):
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
        del stage_dims
        del num_heads
        del expansion_factor

        if not isinstance(num_blocks, Sequence) or len(num_blocks) == 0:
            raise ValueError(f"num_blocks must be a non-empty sequence, got {num_blocks}")

        trunk_blocks = int(rrdb_num_blocks) if int(rrdb_num_blocks) > 0 else int(sum(num_blocks)) + int(refinement_blocks)
        num_feat = int(embed_dim)
        growth_channels = int(rrdb_growth_channels)

        self.base_input_index = int(base_input_index)
        self.use_residual = bool(use_residual)
        self.gradient_checkpointing = bool(gradient_checkpointing)

        self.base_head = nn.Conv2d(3, num_feat, kernel_size=3, stride=1, padding=1)
        self.aux_head = nn.Conv2d(3, num_feat, kernel_size=3, stride=1, padding=1)
        self.diff_head = nn.Conv2d(3, num_feat, kernel_size=3, stride=1, padding=1)
        self.fusion = nn.Sequential(
            nn.Conv2d(num_feat * 3, num_feat, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(num_feat, num_feat, kernel_size=3, stride=1, padding=1),
        )
        self.body = nn.ModuleList(RRDB(num_feat, growth_channels) for _ in range(trunk_blocks))
        self.body_conv = nn.Conv2d(num_feat, num_feat, kernel_size=3, stride=1, padding=1)
        self.hr_conv = nn.Conv2d(num_feat, num_feat, kernel_size=3, stride=1, padding=1)
        self.out_conv = nn.Conv2d(num_feat, 3, kernel_size=3, stride=1, padding=1)
        self.activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def enable_gradient_checkpointing(self) -> None:
        self.gradient_checkpointing = True

    def disable_gradient_checkpointing(self) -> None:
        self.gradient_checkpointing = False

    def _run_body(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.body:
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        return x

    def forward(self, inputs: torch.Tensor, input_mask: torch.Tensor | None = None) -> torch.Tensor:
        if inputs.ndim != 5:
            raise ValueError(f"Expected [B, N, C, H, W], got {inputs.shape}")

        if input_mask is None:
            input_mask = torch.ones(inputs.shape[:2], device=inputs.device, dtype=inputs.dtype)

        base_indices, aux_indices = _resolve_pair_indices(input_mask, self.base_input_index)
        batch_indices = torch.arange(inputs.shape[0], device=inputs.device)

        base_input = inputs[batch_indices, base_indices]
        aux_input = inputs[batch_indices, aux_indices]
        diff_input = torch.abs(base_input - aux_input)

        fused = self.fusion(
            torch.cat(
                [
                    self.base_head(base_input),
                    self.aux_head(aux_input),
                    self.diff_head(diff_input),
                ],
                dim=1,
            )
        )
        body = self.body_conv(self._run_body(fused))
        features = fused + body
        prediction = self.out_conv(self.activation(self.hr_conv(features)))

        if self.use_residual:
            prediction = prediction + base_input

        return prediction.clamp(0.0, 1.0)


def build_model(model_name: str = "realesrgan_refiner", **kwargs) -> nn.Module:
    normalized_name = model_name.lower()
    if normalized_name in {
        "realesrgan",
        "realesrgan_refiner",
        "rrdb_refiner",
        "fusion_rrdb",
    }:
        return RealESRGANRefiner(**kwargs)
    raise ValueError(f"Unsupported model: {model_name}")

