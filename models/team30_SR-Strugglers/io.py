"""
FusionHero (Team 30): NTIRE2026 test.py interface.
Two-branch transformer fusion with global weight w=0.04 for the second branch.
model_func signature: main(model_dir, input_path, output_path, device)
"""
import os
import glob
import logging

import torch
import torch.nn.functional as F

from utils import utils_image as util

from .arch_hat import HAT
from .arch_smt import MSHAT

# Fusion weight (best on validation)
W_BBOX = 0.04


def _load_hat(model_dir, device):
    hat_path = os.path.join(model_dir, "branch_a.pth")
    net = HAT(
        img_size=64, patch_size=1, in_chans=3, embed_dim=180,
        depths=(6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6),
        num_heads=(6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6),
        window_size=16, compress_ratio=3, squeeze_factor=30, conv_scale=0.01,
        overlap_ratio=0.5, mlp_ratio=2.0, qkv_bias=True, qk_scale=None,
        drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.1, ape=False,
        patch_norm=True, use_checkpoint=False, upscale=4, img_range=1.0,
        upsampler="pixelshuffle", resi_connection="1conv",
    )
    state = torch.load(hat_path, map_location="cpu")
    net.load_state_dict(state["params_ema"] if "params_ema" in state else state, strict=True)
    net = net.eval().to(device)
    for p in net.parameters():
        p.requires_grad = False
    return net


def _load_bbox(model_dir, device):
    bbox_path = os.path.join(model_dir, "branch_b.pth")
    net = MSHAT(
        upscale=4, in_chans=3, img_size=64, window_size=16,
        compress_ratio=3, squeeze_factor=30, conv_scale=0.01, overlap_ratio=0.5,
        img_range=1.0, depths=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
        embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
        mlp_ratio=2, upsampler="pixelshuffle", resi_connection="1conv",
    )
    state = torch.load(bbox_path, map_location="cpu")
    net.load_state_dict(state["params"] if "params" in state else state, strict=False)
    net = net.eval().to(device)
    for p in net.parameters():
        p.requires_grad = False
    return net


def _pad16(x):
    _, _, h, w = x.shape
    ph = (16 - h % 16) % 16
    pw = (16 - w % 16) % 16
    if ph > 0 or pw > 0:
        x = F.pad(x, (0, pw, 0, ph), mode="reflect")
    return x, h, w


def _infer_hat(model, lr):
    lr_p, h, w = _pad16(lr)
    with torch.no_grad():
        sr = model(lr_p)
    return sr[:, :, : h * 4, : w * 4].clamp(0, 1)


def _infer_bbox(model, lr):
    _, _, h_old, w_old = lr.size()
    h_pad = (h_old // 16 + 1) * 16 - h_old
    w_pad = (w_old // 16 + 1) * 16 - w_old
    lr_p = torch.cat([lr, torch.flip(lr, [2])], 2)[:, :, : h_old + h_pad, :]
    lr_p = torch.cat([lr_p, torch.flip(lr_p, [3])], 3)[:, :, :, : w_old + w_pad]
    with torch.no_grad():
        sr = model(lr_p)
    return sr[..., : h_old * 4, : w_old * 4].clamp(0, 1)


def _infer_bbox_tta(model, lr):
    tfs = [
        lambda x: x, lambda x: x.flip(-1), lambda x: x.flip(-2), lambda x: x.flip(-1).flip(-2),
        lambda x: x.permute(0, 1, 3, 2), lambda x: x.permute(0, 1, 3, 2).flip(-1),
        lambda x: x.permute(0, 1, 3, 2).flip(-2), lambda x: x.permute(0, 1, 3, 2).flip(-1).flip(-2),
    ]
    inv = [
        lambda x: x, lambda x: x.flip(-1), lambda x: x.flip(-2), lambda x: x.flip(-1).flip(-2),
        lambda x: x.permute(0, 1, 3, 2), lambda x: x.flip(-1).permute(0, 1, 3, 2),
        lambda x: x.flip(-2).permute(0, 1, 3, 2), lambda x: x.flip(-1).flip(-2).permute(0, 1, 3, 2),
    ]
    outs = []
    for tf, itf in zip(tfs, inv):
        outs.append(itf(_infer_bbox(model, tf(lr))))
    return torch.mean(torch.stack(outs, dim=0), dim=0).clamp(0, 1)


def main(model_dir, input_path, output_path, device=None):
    """
    Required interface for test.py.
    model_dir: path to model_zoo/team30_FusionHero (contains branch_a.pth, branch_b.pth)
    input_path: folder of LR PNG images
    output_path: folder to save SR PNG images (same filenames as input)
    device: torch device
    """
    logging.getLogger("NTIRE2026-ImageSRx4")
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = os.path.abspath(model_dir)
    if input_path.endswith("/"):
        input_path = input_path[:-1]
    util.mkdir(output_path)

    hat = _load_hat(model_dir, device)
    bbox = _load_bbox(model_dir, device)

    data_range = 1.0
    input_list = sorted(glob.glob(os.path.join(input_path, "*.[jpJP][pnPN]*[gG]")))
    for img_lr_path in input_list:
        img_name = os.path.basename(img_lr_path)
        name_stem, ext = os.path.splitext(img_name)
        img_lr = util.imread_uint(img_lr_path, n_channels=3)
        lr = util.uint2tensor4(img_lr, data_range).to(device)

        sr_hat = _infer_hat(hat, lr)
        sr_bbox = _infer_bbox_tta(bbox, lr)
        sr = (1.0 - W_BBOX) * sr_hat + W_BBOX * sr_bbox

        sr_uint = util.tensor2uint(sr, data_range)
        util.imsave(sr_uint, os.path.join(output_path, img_name))
