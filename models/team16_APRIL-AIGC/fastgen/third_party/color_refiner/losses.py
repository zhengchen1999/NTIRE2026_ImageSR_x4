from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .metrics import LPIPSMetric, SSIM


class CharbonnierLoss(nn.Module):
    def __init__(self, epsilon: float = 1e-3) -> None:
        super().__init__()
        self.epsilon = epsilon

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.sqrt((prediction - target).pow(2) + self.epsilon**2).mean()


class ColorConsistencyLoss(nn.Module):
    def __init__(self, pooled_resolution: int = 8) -> None:
        super().__init__()
        self.pooled_resolution = pooled_resolution

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_mean = prediction.mean(dim=(-1, -2))
        target_mean = target.mean(dim=(-1, -2))
        pred_std = prediction.flatten(2).std(dim=-1, unbiased=False)
        target_std = target.flatten(2).std(dim=-1, unbiased=False)

        pooled_pred = F.adaptive_avg_pool2d(prediction, self.pooled_resolution)
        pooled_target = F.adaptive_avg_pool2d(target, self.pooled_resolution)

        mean_loss = F.l1_loss(pred_mean, target_mean)
        std_loss = F.l1_loss(pred_std, target_std)
        pooled_loss = F.l1_loss(pooled_pred, pooled_target)
        return mean_loss + 0.5 * std_loss + pooled_loss


class RefinerLoss(nn.Module):
    def __init__(
        self,
        pixel_weight: float = 1.0,
        ssim_weight: float = 0.2,
        color_weight: float = 0.1,
        lpips_weight: float = 0.0,
        lpips_net: str = "alex",
    ) -> None:
        super().__init__()
        self.pixel_weight = pixel_weight
        self.ssim_weight = ssim_weight
        self.color_weight = color_weight
        self.lpips_weight = lpips_weight
        self.pixel_loss = CharbonnierLoss()
        self.ssim_metric = SSIM()
        self.color_loss = ColorConsistencyLoss()
        self.lpips_metric = LPIPSMetric(lpips_net, as_loss=True) if lpips_weight > 0 else None
        if self.lpips_metric is not None and not self.lpips_metric.available:
            raise RuntimeError("lpips_weight > 0 but PyIQA LPIPS is unavailable. Install `pyiqa` first.")

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        pixel = self.pixel_loss(prediction, target)
        total = self.pixel_weight * pixel
        components: dict[str, torch.Tensor] = {"pixel": pixel.detach()}

        if self.ssim_weight > 0:
            ssim_loss = 1.0 - self.ssim_metric(prediction, target)
            total = total + self.ssim_weight * ssim_loss
            components["ssim"] = ssim_loss.detach()

        if self.color_weight > 0:
            color = self.color_loss(prediction, target)
            total = total + self.color_weight * color
            components["color"] = color.detach()

        if self.lpips_weight > 0 and self.lpips_metric is not None:
            lpips_value = self.lpips_metric(prediction, target)
            total = total + self.lpips_weight * lpips_value
            components["lpips"] = lpips_value.detach()

        components["total"] = total.detach()
        return total, components
