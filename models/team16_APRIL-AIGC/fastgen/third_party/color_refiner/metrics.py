from __future__ import annotations

import importlib

import torch
import torch.nn as nn


def _reduce_scores(score: torch.Tensor, reduction: str) -> torch.Tensor:
    if score.ndim == 0:
        if reduction == "none":
            return score.unsqueeze(0)
        if reduction == "mean":
            return score
        raise ValueError(f"Unsupported reduction: {reduction}")

    per_sample = score.reshape(score.shape[0], -1).mean(dim=1)
    if reduction == "none":
        return per_sample
    if reduction == "mean":
        return per_sample.mean()
    raise ValueError(f"Unsupported reduction: {reduction}")


def _resolve_lpips_metric_name(net: str) -> str:
    normalized = str(net).strip().lower()
    if normalized in {"vgg", "lpips-vgg"}:
        return "lpips-vgg"
    if normalized in {"alex", "lpips"}:
        return "lpips"
    raise ValueError(f"Unsupported LPIPS backend: {net}")


class PyIQAMetric(nn.Module):
    def __init__(
        self,
        metric_name: str,
        *,
        as_loss: bool = False,
        loss_weight: float | None = None,
        loss_reduction: str = "mean",
        **metric_kwargs,
    ) -> None:
        super().__init__()
        self.metric_name = metric_name
        self.as_loss = as_loss
        self.loss_weight = loss_weight
        self.loss_reduction = loss_reduction
        self.metric_kwargs = metric_kwargs
        self.metric: nn.Module | None = None
        self.metric_device: torch.device | None = None
        self.available = False
        self.lower_better = False
        try:
            self._pyiqa = importlib.import_module("pyiqa")
            self.available = True
        except ImportError:
            self._pyiqa = None

    def _build_metric(self, device: torch.device) -> nn.Module:
        if not self.available or self._pyiqa is None:
            raise RuntimeError("PyIQA is unavailable. Please install `pyiqa` in the runtime environment.")
        if self.metric is None:
            kwargs = dict(self.metric_kwargs)
            if self.as_loss:
                kwargs["loss_weight"] = self.loss_weight
                kwargs["loss_reduction"] = self.loss_reduction
            self.metric = self._pyiqa.create_metric(
                self.metric_name,
                as_loss=self.as_loss,
                device=device,
                **kwargs,
            )
            self.lower_better = bool(getattr(self.metric, "lower_better", False))
            self.metric_device = torch.device(device)
        elif self.metric_device != torch.device(device):
            self.metric = self.metric.to(device)
            self.metric_device = torch.device(device)
        return self.metric

    def forward(self, prediction: torch.Tensor, target: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
        metric = self._build_metric(prediction.device)
        score = metric(prediction.float(), target.float())
        if isinstance(score, tuple):
            score = score[0]
        if not torch.is_tensor(score):
            score = torch.as_tensor(score, device=prediction.device, dtype=prediction.dtype)
        if self.as_loss:
            if reduction != "mean":
                raise ValueError("PyIQA loss metrics only support reduction='mean' in this wrapper.")
            return score.mean()
        return _reduce_scores(score, reduction)


class PSNRMetric(PyIQAMetric):
    def __init__(self) -> None:
        super().__init__("psnr")


def compute_psnr(
    prediction: torch.Tensor,
    target: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    return PSNRMetric()(prediction, target, reduction=reduction)


class SSIM(PyIQAMetric):
    def __init__(self) -> None:
        super().__init__("ssim")


class LPIPSMetric(PyIQAMetric):
    def __init__(self, net: str = "alex", as_loss: bool = False) -> None:
        super().__init__(_resolve_lpips_metric_name(net), as_loss=as_loss)
