import os
import glob
import logging

import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
from tqdm import tqdm

from utils import utils_logger
from utils import utils_image as util

logger = logging.getLogger(__name__)


# =============================================================================
# Model Loaders  (mirrors inference.py)
# =============================================================================

def _load_hat_model(weights_path: str, device: torch.device) -> torch.nn.Module:
    from models.team25_IKLAB.hat import HATIQCMix, DropPath

    model = HATIQCMix(
        img_size=64, patch_size=1, in_chans=3, embed_dim=180,
        depths=(6,) * 12, num_heads=(6,) * 12, window_size=16,
        compress_ratio=3, squeeze_factor=30, conv_scale=0.01,
        overlap_ratio=0.5, mlp_ratio=2., qkv_bias=True, qk_scale=None,
        drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
        ape=False, patch_norm=True, use_checkpoint=False,
        upscale=4, img_range=1., upsampler='pixelshuffle', resi_connection='1conv',
    )
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    key = ('params_ema' if 'params_ema' in ckpt
           else ('params' if 'params' in ckpt else None))
    model.load_state_dict(ckpt[key] if key else ckpt, strict=False)
    model = model.to(device).train()   # intentional — IQA-adaptive branching requires train()
    for p in model.parameters():
        p.requires_grad = False
    for m in model.modules():
        if isinstance(m, DropPath):
            m.drop_prob = 0.0           # disable stochastic depth for determinism
    logger.info(f"  [HAT] loaded from {weights_path}")
    return model


def _load_dat_model(weights_path: str, device: torch.device) -> torch.nn.Module:
    from models.team25_IKLAB.dat import DAT

    model = DAT(
        img_size=64, in_chans=3, embed_dim=180,
        split_size=[8, 16], depth=[6] * 6, num_heads=[6] * 6,
        expansion_factor=2., upscale=4, img_range=1., resi_connection='1conv',
    )
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    key = ('params_ema' if 'params_ema' in ckpt
           else ('params' if 'params' in ckpt else None))
    model.load_state_dict(ckpt[key] if key else ckpt, strict=True)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    logger.info(f"  [DAT] loaded from {weights_path}")
    return model


def _load_fusion_net(weights_path: str, device: torch.device):
    from models.team25_IKLAB.fusion_net import FusionNet

    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    state = ckpt.get('fusion_net', ckpt)
    state = {k.replace('module.', ''): v for k, v in state.items()}
    num_models = ckpt.get('num_models', 2)

    net = FusionNet(num_models=num_models, base_channels=32)
    use_var_map = False

    net.load_state_dict(state, strict=True)
    net = net.to(device).eval()
    for p in net.parameters():
        p.requires_grad = False
    logger.info(f"  [FusionNet] loaded from {weights_path}  (var_map={use_var_map})")
    return net, use_var_map


# =============================================================================
# Inference helpers  (mirrors inference.py)
# =============================================================================

def _pad_to_multiple(img: torch.Tensor, window_size: int = 16):
    _, _, h, w = img.size()
    h_pad = (window_size - h % window_size) % window_size
    w_pad = (window_size - w % window_size) % window_size
    if h_pad == 0 and w_pad == 0:
        return img, 0, 0
    if h_pad:
        img = torch.cat([img, torch.flip(img, [2])], dim=2)[:, :, :h + h_pad, :]
    if w_pad:
        img = torch.cat([img, torch.flip(img, [3])], dim=3)[:, :, :, :w + w_pad]
    return img, h_pad, w_pad


@torch.no_grad()
def _run_sr_single(img_lr, sr_models, fusion_net, scale=4, use_var_map=False):
    _, _, h, w = img_lr.size()
    img_pad, _, _ = _pad_to_multiple(img_lr)
    sr_outputs = [m(img_pad)[:, :, :h * scale, :w * scale] for m in sr_models]

    if use_var_map and len(sr_outputs) > 1:
        var_map  = torch.var(torch.stack(sr_outputs, dim=0), dim=0).mean(dim=1, keepdim=True)
        combined = torch.cat(sr_outputs + [var_map], dim=1)
    else:
        combined = torch.cat(sr_outputs, dim=1)

    return fusion_net(combined).clamp(0, 1)


@torch.no_grad()
def _run_sr_tta8(img_lr, sr_models, fusion_net, scale=4, use_var_map=False):
    outputs = []
    for rot_k in range(4):
        for flip in (False, True):
            x = img_lr.clone()
            if flip:
                x = torch.flip(x, [3])
            if rot_k:
                x = torch.rot90(x, rot_k, dims=[2, 3])

            out = _run_sr_single(x, sr_models, fusion_net, scale, use_var_map)

            if rot_k:
                out = torch.rot90(out, 4 - rot_k, dims=[2, 3])
            if flip:
                out = torch.flip(out, [3])

            outputs.append(out)

    return torch.stack(outputs, dim=0).mean(dim=0)


# =============================================================================
# Public entry point — matches the interface expected by test.py
# =============================================================================

def main(model_dir: str, input_path: str, output_path: str,
         device: torch.device = None, use_tta: bool = True):
    """Run HAT+DAT fusion SR for all images in *input_path*, saving results to *output_path*.

    Args:
        model_dir:   Directory containing hat.pth, dat.pth, fusion_best.pth.
        input_path:  Directory with LR input images.
        output_path: Directory for SR output images.
        device:      torch.device (auto-detected if None).
        use_tta:     Whether to apply 8-way TTA (default True).
    """
    utils_logger.logger_info("NTIRE2026-ImageSRx4-team25", log_path="NTIRE2026-ImageSRx4.log")

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- load weights ----
    w_hat    = os.path.join(model_dir, 'hat.pth')
    w_dat    = os.path.join(model_dir, 'dat.pth')
    w_fusion = os.path.join(model_dir, 'fusion_best.pth')
    for p in (w_hat, w_dat, w_fusion):
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Weight file not found: {p}")

    sr_models = [
        _load_hat_model(w_hat, device),
        _load_dat_model(w_dat, device),
    ]
    fusion_net, use_var_map = _load_fusion_net(w_fusion, device)

    # ---- image files ----
    util.mkdir(output_path)
    lr_files = sorted(glob.glob(os.path.join(input_path, '*.[jpJP][pnPN]*[gG]')))
    if not lr_files:
        raise FileNotFoundError(f"No images found in {input_path}")

    run_fn = _run_sr_tta8 if use_tta else _run_sr_single

    for lr_path in tqdm(lr_files, desc='SR inference'):
        img_name, ext = os.path.splitext(os.path.basename(lr_path))

        lr_img = Image.open(lr_path).convert('RGB')
        lr_t   = TF.to_tensor(lr_img).unsqueeze(0).to(device)

        sr_t = run_fn(lr_t, sr_models, fusion_net, scale=4, use_var_map=use_var_map)

        sr_np = (sr_t.squeeze(0).permute(1, 2, 0)
                 .cpu().numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
        Image.fromarray(sr_np).save(os.path.join(output_path, img_name + '.png'))
