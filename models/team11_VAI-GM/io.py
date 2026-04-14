import glob
import os
import os.path as osp

import torch
from PIL import Image
from torchvision import transforms

from basicsr.archs.pft_arch import PFT


def _load_state_dict(model: torch.nn.Module, model_path: str, device: torch.device):
    ckpt = torch.load(model_path, map_location=device)
    if isinstance(ckpt, dict):
        if "params_ema" in ckpt:
            state_dict = ckpt["params_ema"]
        elif "params" in ckpt:
            state_dict = ckpt["params"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt
    model.load_state_dict(state_dict, strict=True)


def _build_model(scale: int = 4) -> torch.nn.Module:
    return PFT(
        upscale=scale,
        embed_dim=240,
        depths=[4, 4, 4, 6, 6, 6],
        num_heads=6,
        num_topk=[
            1024, 1024, 1024, 1024,
            256, 256, 256, 256,
            128, 128, 128, 128,
            64, 64, 64, 64, 64, 64,
            32, 32, 32, 32, 32, 32,
            16, 16, 16, 16, 16, 16,
        ],
        window_size=32,
        convffn_kernel_size=7,
        mlp_ratio=2,
        upsampler="pixelshuffle",
        use_checkpoint=False,
    )


def _infer_one(model: torch.nn.Module, image_path: str, save_path: str, device: torch.device):
    with torch.no_grad():
        img = Image.open(image_path).convert("RGB")
        tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)
        output = model(tensor).clamp(0.0, 1.0)[0].cpu()
        Image.fromarray((output.permute(1, 2, 0).numpy() * 255.0).round().astype("uint8")).save(save_path)


def main(model_dir: str, input_path: str, output_path: str, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.isfile(model_dir):
        raise FileNotFoundError(
            f"Checkpoint not found: {model_dir}. "
            "Place your model checkpoint (.pth) in model_zoo/"
        )
    if not os.path.isdir(input_path):
        raise FileNotFoundError(f"Input folder not found: {input_path}")

    os.makedirs(output_path, exist_ok=True)

    model = _build_model(scale=4).to(device)
    _load_state_dict(model, model_dir, device)
    model.eval()

    img_list = sorted(glob.glob(osp.join(input_path, "*.[jpJP][pnPN]*[gG]")))
    if len(img_list) == 0:
        raise RuntimeError(f"No input images found in: {input_path}")
    for img_path in img_list:
        name = osp.basename(img_path)
        out_path = osp.join(output_path, name)
        _infer_one(model, img_path, out_path, device)
