"""
NTIRE 2026 Image SR x4 Challenge — Team 29: FreqFusion
=======================================================
Lightweight IO wrapper for the official NTIRE2026_ImageSR_x4 test runner.

This module exposes main(model_dir, input_path, output_path, device) which:
  1. Loads the expert ensemble (HAT-L, DAT, NAFNet)
  2. Builds CompleteEnhancedFusionSR and loads the fusion checkpoint
  3. Runs 4× super-resolution on all PNG images in input_path
  4. Saves results to output_path

The original model implementation lives in src/ and is NOT modified.
"""

import os
import sys
import glob
import logging

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Ensure the project root (parent of this file's grandparent) is on sys.path
# so that `from src.models.enhanced_fusion import ...` works.
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.models.enhanced_fusion import CompleteEnhancedFusionSR
from src.models import expert_loader

# ---------------------------------------------------------------------------
# Model configuration (must match the training config)
# ---------------------------------------------------------------------------
MODEL_CONFIG = {
    "scale": 4,
    "num_experts": 3,
    "fusion_dim": 64,
    "num_heads": 4,
    "refine_depth": 4,
    "refine_channels": 64,
    "num_bands": 3,
    "block_size": 8,
    "enable_hierarchical": True,
    "enable_multi_domain_freq": True,
    "enable_lka": True,
    "enable_edge_enhance": True,
    "enable_dynamic_selection": True,
    "enable_cross_band_attn": True,
    "enable_adaptive_bands": True,
    "enable_multi_resolution": True,
    "enable_collaborative": True,
}


# ---------------------------------------------------------------------------
# Image I/O helpers
# ---------------------------------------------------------------------------
def _load_image(path: str) -> torch.Tensor:
    """Load PNG image → [1, 3, H, W] float32 tensor in [0, 1]."""
    img = Image.open(path).convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def _save_image(tensor: torch.Tensor, path: str):
    """Save [1, 3, H, W] or [3, H, W] tensor as PNG."""
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    arr = (tensor.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)
    Image.fromarray(arr).save(path, format="PNG")


# ---------------------------------------------------------------------------
# Tiled inference (OOM safety for large images)
# ---------------------------------------------------------------------------
def _tiled_forward(model, lr_img, tile_size=64, overlap=8, scale=4, device="cuda"):
    """Process a large image by splitting into overlapping tiles."""
    _, _, h, w = lr_img.shape
    sr_h, sr_w = h * scale, w * scale
    sr_output = torch.zeros(1, 3, sr_h, sr_w, device=device)
    weight_map = torch.zeros(1, 1, sr_h, sr_w, device=device)

    step = tile_size - overlap
    y_positions = list(range(0, max(h - tile_size + 1, 1), step))
    if y_positions[-1] + tile_size < h:
        y_positions.append(h - tile_size)
    x_positions = list(range(0, max(w - tile_size + 1, 1), step))
    if x_positions[-1] + tile_size < w:
        x_positions.append(w - tile_size)

    for y in y_positions:
        for x in x_positions:
            lr_tile = lr_img[:, :, y:y + tile_size, x:x + tile_size]
            torch.cuda.empty_cache()
            sr_tile = model(lr_tile)
            sy, sx = y * scale, x * scale
            st = tile_size * scale
            wy = torch.ones(st, device=device)
            wx = torch.ones(st, device=device)
            blend = min(overlap * scale, st // 4)
            if blend > 0:
                ramp = torch.linspace(0, 1, blend, device=device)
                if y > 0:
                    wy[:blend] = ramp
                if y + tile_size < h:
                    wy[-blend:] = 1 - ramp
                if x > 0:
                    wx[:blend] = ramp
                if x + tile_size < w:
                    wx[-blend:] = 1 - ramp
            weight = (wy.unsqueeze(1) * wx.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
            sr_output[:, :, sy:sy + st, sx:sx + st] += sr_tile * weight
            weight_map[:, :, sy:sy + st, sx:sx + st] += weight

    return sr_output / weight_map.clamp(min=1e-8)


# ---------------------------------------------------------------------------
# Build model + load checkpoint
# ---------------------------------------------------------------------------
def _build_and_load(model_dir: str, device):
    """Build the full model and load the fusion checkpoint."""

    # --- Expert ensemble ---
    pretrained_dir = os.path.join(_PROJECT_ROOT, "pretrained")
    ensemble = expert_loader.ExpertEnsemble(upscale=MODEL_CONFIG["scale"], device=device)
    expert_paths = {
        "hat": os.path.join(pretrained_dir, "hat", "HAT-L_SRx4_ImageNet-pretrain.pth"),
        "dat": os.path.join(pretrained_dir, "dat", "DAT_x4.pth"),
        "nafnet": os.path.join(pretrained_dir, "nafnet", "NAFNet-SIDD-width64.pth"),
    }
    ensemble.load_all_experts(checkpoint_paths=expert_paths, freeze=True)

    # --- Fusion model ---
    model = CompleteEnhancedFusionSR(
        expert_ensemble=ensemble,
        num_experts=MODEL_CONFIG["num_experts"],
        num_bands=MODEL_CONFIG["num_bands"],
        block_size=MODEL_CONFIG["block_size"],
        upscale=MODEL_CONFIG["scale"],
        fusion_dim=MODEL_CONFIG["fusion_dim"],
        num_heads=MODEL_CONFIG["num_heads"],
        refine_depth=MODEL_CONFIG["refine_depth"],
        refine_channels=MODEL_CONFIG["refine_channels"],
        enable_hierarchical=MODEL_CONFIG["enable_hierarchical"],
        enable_multi_domain_freq=MODEL_CONFIG["enable_multi_domain_freq"],
        enable_lka=MODEL_CONFIG["enable_lka"],
        enable_edge_enhance=MODEL_CONFIG["enable_edge_enhance"],
        enable_dynamic_selection=MODEL_CONFIG["enable_dynamic_selection"],
        enable_cross_band_attn=MODEL_CONFIG["enable_cross_band_attn"],
        enable_adaptive_bands=MODEL_CONFIG["enable_adaptive_bands"],
        enable_multi_resolution=MODEL_CONFIG["enable_multi_resolution"],
        enable_collaborative=MODEL_CONFIG["enable_collaborative"],
    )
    model = model.to(device)

    # --- Load fusion checkpoint ---
    checkpoint = torch.load(model_dir, map_location=device, weights_only=False)
    ckpt_state = checkpoint.get("model_state_dict", checkpoint)
    model_state = model.state_dict()
    loaded = 0
    for key, param in ckpt_state.items():
        clean_key = key
        for prefix in ["module.", "model."]:
            if clean_key.startswith(prefix):
                clean_key = clean_key[len(prefix):]
        if clean_key in model_state and param.shape == model_state[clean_key].shape:
            model_state[clean_key] = param
            loaded += 1
    model.load_state_dict(model_state, strict=False)
    print(f"[team29_FreqFusion] Loaded {loaded} fusion weight tensors from checkpoint")

    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


# ---------------------------------------------------------------------------
# Public API — called by test.py
# ---------------------------------------------------------------------------
@torch.no_grad()
def main(model_dir: str, input_path: str, output_path: str, device=None):
    """
    NTIRE2026 official interface.

    Args:
        model_dir:   Path to the pretrained fusion checkpoint (.pth).
        input_path:  Folder containing LR input images (PNG).
        output_path: Folder where SR output images will be saved (PNG).
        device:      torch.device or None (auto-detect).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[team29_FreqFusion] Device: {device}")

    # Build model
    model = _build_and_load(model_dir, device)

    # Scan input images
    input_imgs = sorted(glob.glob(os.path.join(input_path, "*.[pP][nN][gG]")))
    if not input_imgs:
        input_imgs = sorted(glob.glob(os.path.join(input_path, "*.[jJ][pP]*[gG]")))
    print(f"[team29_FreqFusion] Found {len(input_imgs)} images in {input_path}")

    os.makedirs(output_path, exist_ok=True)

    for img_path in input_imgs:
        img_name = os.path.basename(img_path)
        lr_img = _load_image(img_path).to(device)

        # Try full-image forward; fall back to tiled if OOM
        try:
            torch.cuda.empty_cache()
            sr_img = model(lr_img)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                print(f"  OOM on {img_name}, switching to tiled inference (128px)...")
                sr_img = _tiled_forward(model, lr_img, tile_size=128, overlap=32, scale=4, device=device)
            else:
                raise

        _save_image(sr_img, os.path.join(output_path, img_name))
        del sr_img, lr_img
        torch.cuda.empty_cache()

    print(f"[team29_FreqFusion] Done. {len(input_imgs)} images saved to {output_path}")
