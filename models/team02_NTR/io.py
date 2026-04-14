"""
NTR Team (team02) — NTIRE 2026 Image Super-Resolution (x4) Challenge
Method: TimeDiffiT_ResNet_color_128 with MDAE pretraining + SFT
        8-fold geometric self-ensemble, tiled inference
"""

import os
import re
import glob
import numpy as np
import torch
import torch.nn as nn
import cv2

from .arch import TimeDiffiT_ResNet_color_128  # our SR architecture


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_model(model_dir: str, device: torch.device) -> nn.Module:
    """Load TimeDiffiT_ResNet_color_128 from checkpoint."""

    class _Args:
        task = "sr"
        sr_scale = 4

    model = TimeDiffiT_ResNet_color_128(_Args())
    state = torch.load(model_dir, map_location=device, weights_only=False)
    clean = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
    model.load_state_dict(clean, strict=False)
    model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Image utilities
# ---------------------------------------------------------------------------

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def _scan_images(root: str):
    imgs = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if fn.lower().endswith(IMG_EXT):
                imgs.append(os.path.join(dp, fn))
    return sorted(imgs)


def _read_rgb(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to read: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _to_tensor(img: np.ndarray, device: torch.device) -> torch.Tensor:
    arr = np.ascontiguousarray(img.transpose(2, 0, 1))
    t = torch.from_numpy(arr).float().unsqueeze(0).to(device) / 255.0
    return t * 2.0 - 1.0  # [0,1] -> [-1,1] (auto_normalize)


def _to_uint8(tensor: torch.Tensor) -> np.ndarray:
    img = tensor.squeeze(0).permute(1, 2, 0).detach().cpu().float()
    img = (img + 1.0) * 0.5  # [-1,1] -> [0,1]
    img = torch.clamp(img, 0.0, 1.0).numpy()
    return (img * 255.0 + 0.5).astype(np.uint8)


# ---------------------------------------------------------------------------
# Tiled inference
# ---------------------------------------------------------------------------

TILE = 128
OVERLAP = 32
SCALE = 4


def _make_tiles(length: int, tile: int, overlap: int):
    stride = tile - overlap
    starts = list(range(0, length - tile + 1, stride))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


def _run_model(model, x):
    t = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
    with torch.no_grad():
        out = model(x1=x, time=t)
    return torch.clamp(out, -1.0, 1.0)


def _run_tiled(model, lr_tensor):
    _, _, h, w = lr_tensor.shape
    if h <= TILE and w <= TILE:
        return _run_model(model, lr_tensor)

    ys = _make_tiles(h, TILE, OVERLAP)
    xs = _make_tiles(w, TILE, OVERLAP)
    out = torch.zeros((1, 3, h * SCALE, w * SCALE),
                      device=lr_tensor.device, dtype=lr_tensor.dtype)
    weight = torch.zeros_like(out)

    for y in ys:
        for x in xs:
            patch = lr_tensor[:, :, y:y + TILE, x:x + TILE]
            pr = _run_model(model, patch)
            oy, ox = y * SCALE, x * SCALE
            ph, pw = pr.shape[2], pr.shape[3]
            out[:, :, oy:oy + ph, ox:ox + pw] += pr
            weight[:, :, oy:oy + ph, ox:ox + pw] += 1.0

    return out / weight.clamp_min(1.0)


# ---------------------------------------------------------------------------
# 8-fold geometric self-ensemble (TTA)
# ---------------------------------------------------------------------------

def _apply_transform(x, t):
    if t >= 4:
        x = torch.flip(x, dims=[3])
    k = t % 4
    if k > 0:
        x = torch.rot90(x, k=k, dims=[2, 3])
    return x


def _invert_transform(x, t):
    k = t % 4
    if k > 0:
        x = torch.rot90(x, k=4 - k, dims=[2, 3])
    if t >= 4:
        x = torch.flip(x, dims=[3])
    return x


def _run_ensemble(model, lr_tensor):
    accum = None
    for t_idx in range(8):
        aug = _apply_transform(lr_tensor, t_idx)
        sr = _run_tiled(model, aug)
        sr = _invert_transform(sr, t_idx)
        accum = sr if accum is None else accum + sr
    return accum / 8.0


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def main(model_dir: str, input_path: str, output_path: str,
         device: torch.device) -> None:
    """
    Run SR inference on all images in input_path and save results to output_path.

    Args:
        model_dir:   Path to net_best.pth checkpoint.
        input_path:  Directory containing LR input images.
        output_path: Directory to save SR output images.
        device:      torch.device to use for inference.
    """
    os.makedirs(output_path, exist_ok=True)
    model = _load_model(model_dir, device)

    lr_paths = _scan_images(input_path)
    print(f"[team02_NTR] {len(lr_paths)} images | tile={TILE} overlap={OVERLAP} "
          f"ensemble=8-fold | device={device}")

    for i, lr_path in enumerate(lr_paths):
        name = os.path.basename(lr_path)
        lr_img = _read_rgb(lr_path)
        lr_tensor = _to_tensor(lr_img, device)
        sr_tensor = _run_ensemble(model, lr_tensor)
        sr_uint8 = _to_uint8(sr_tensor)
        save_path = os.path.join(output_path, name)
        cv2.imwrite(save_path, cv2.cvtColor(sr_uint8, cv2.COLOR_RGB2BGR))
        print(f"  [{i+1}/{len(lr_paths)}] {name}")

    print(f"[team02_NTR] Saved {len(lr_paths)} SR images to {output_path}")
