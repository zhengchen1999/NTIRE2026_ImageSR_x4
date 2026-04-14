"""
HAT-L with L2 Fine-tuning and 8x Geometric Self-Ensemble (TTA).
NTIRE 2026 Image Super-Resolution (x4) Challenge.
"""
import os
import glob
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

from models.team31_HAT_L2FT.hat_arch import HAT


def build_model(model_dir, device):
    """Build HAT-L model and load weights."""
    model = HAT(
        upscale=4, in_chans=3, img_size=64, window_size=16,
        compress_ratio=3, squeeze_factor=30, conv_scale=0.01,
        overlap_ratio=0.5, img_range=1.0,
        depths=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
        embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
        mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv',
    )
    state = torch.load(model_dir, map_location='cpu', weights_only=False)
    key = 'params_ema' if 'params_ema' in state else 'params'
    model.load_state_dict(state[key], strict=True)
    model.eval().to(device)
    return model


def pad_and_forward(model, img_t):
    """Pad to multiple of 16, forward, crop back."""
    _, _, h, w = img_t.shape
    ph = (16 - h % 16) % 16
    pw = (16 - w % 16) % 16
    if ph > 0 or pw > 0:
        img_t = F.pad(img_t, (0, pw, 0, ph), mode='reflect')
    with torch.no_grad():
        out = model(img_t)
    return out[:, :, :h * 4, :w * 4]


def forward_tta(model, img_t, device):
    """8x geometric self-ensemble (TTA)."""
    transforms = [
        lambda x: x,
        lambda x: x.flip(-1),
        lambda x: x.flip(-2),
        lambda x: x.flip(-1).flip(-2),
        lambda x: x.permute(0, 1, 3, 2),
        lambda x: x.permute(0, 1, 3, 2).flip(-1),
        lambda x: x.permute(0, 1, 3, 2).flip(-2),
        lambda x: x.permute(0, 1, 3, 2).flip(-1).flip(-2),
    ]
    inv_transforms = [
        lambda x: x,
        lambda x: x.flip(-1),
        lambda x: x.flip(-2),
        lambda x: x.flip(-1).flip(-2),
        lambda x: x.permute(0, 1, 3, 2),
        lambda x: x.flip(-1).permute(0, 1, 3, 2),
        lambda x: x.flip(-2).permute(0, 1, 3, 2),
        lambda x: x.flip(-1).flip(-2).permute(0, 1, 3, 2),
    ]

    outputs = []
    for t, inv_t in zip(transforms, inv_transforms):
        aug = t(img_t).to(device)
        out = pad_and_forward(model, aug)
        outputs.append(inv_t(out).cpu())

    return torch.mean(torch.stack(outputs), dim=0)


def main(model_dir, input_path, output_path, device=None):
    """Main entry point for NTIRE2026 test.py."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = build_model(model_dir, device)
    os.makedirs(output_path, exist_ok=True)

    img_paths = sorted(glob.glob(os.path.join(input_path, '*.[jpJP][pnPN]*[gG]')))
    print(f"Processing {len(img_paths)} images with HAT-L L2-FT + 8x TTA")

    for i, path in enumerate(img_paths):
        name = os.path.splitext(os.path.basename(path))[0]
        ext = os.path.splitext(path)[1]

        img = np.array(Image.open(path).convert('RGB')).astype(np.float32) / 255.0
        img_t = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)

        output = forward_tta(model, img_t, device)

        output = output.squeeze(0).clamp(0, 1).numpy()
        output = (output * 255.0).round().astype(np.uint8).transpose(1, 2, 0)
        Image.fromarray(output).save(os.path.join(output_path, name + ext))

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(img_paths)}]")

    print(f"Done! Results saved to {output_path}")
