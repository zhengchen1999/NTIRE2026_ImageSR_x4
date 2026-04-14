import glob
import os
import random

import cv2
import numpy as np
import torch

from models.team15_escxl.model import ESC


def set_deterministic(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def img_to_tensor(img_bgr_uint8: np.ndarray, device: torch.device):
    img = img_bgr_uint8.astype(np.float32) / 255.0
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float().unsqueeze(0).to(device)
    return img


def tensor_to_img_uint8(output: torch.Tensor):
    out = output.detach().float().cpu().clamp_(0, 1).numpy()
    out = np.transpose(out[[2, 1, 0], :, :], (1, 2, 0))
    out = (out * 255.0).round().astype(np.uint8)
    return out


def run_tta_x8(model, x: torch.Tensor):
    outputs = []
    for k in range(4):
        rot = torch.rot90(x, k, [-2, -1])
        out = model(rot)
        outputs.append(torch.rot90(out, -k, [-2, -1]))
        flip = rot.flip(-1)
        out_f = model(flip)
        outputs.append(torch.rot90(out_f.flip(-1), -k, [-2, -1]))
    return torch.stack(outputs, dim=0).mean(dim=0)


def build_model(model_path: str, device: torch.device):
    model = ESC(
        dim=192,
        pdim=48,
        kernel_size=13,
        n_blocks=8,
        conv_blocks=5,
        window_size=48,
        num_heads=12,
        upscaling_factor=4,
        exp_ratio=1.25,
        use_ln=True,
        attn_type="Flex",
    )
    ckpt = torch.load(model_path, map_location="cpu")
    state_dict = ckpt["params_ema"] if isinstance(ckpt, dict) and "params_ema" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    for _, param in model.named_parameters():
        param.requires_grad = False
    model.to(device)
    return model


def run(model, input_path: str, output_path: str, device: torch.device):
    os.makedirs(output_path, exist_ok=True)
    input_img_list = sorted(glob.glob(os.path.join(input_path, "*.[jpJP][pnPN]*[gG]")))
    for idx, img_path in enumerate(input_img_list):
        img_name, ext = os.path.splitext(os.path.basename(img_path))
        img_lq = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img_lq is None:
            raise RuntimeError(f"Failed to read image: {img_path}")

        img_lq = img_to_tensor(img_lq, device)
        with torch.no_grad():
            img_sr = run_tta_x8(model, img_lq)
        img_sr = tensor_to_img_uint8(img_sr.squeeze(0))
        cv2.imwrite(os.path.join(output_path, img_name + ext), img_sr)

        if (idx + 1) % 10 == 0 or (idx + 1) == len(input_img_list):
            print(f"[team15_escxl] {idx + 1}/{len(input_img_list)} done")


def main(model_dir, input_path, output_path, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_deterministic(0)
    model = build_model(model_dir, device)
    run(model, input_path, output_path, device)
