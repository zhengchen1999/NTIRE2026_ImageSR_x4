import os.path
import logging
import torch
import json
import glob
import importlib.util

from utils.model_summary import get_model_flops
from utils import utils_logger
from utils import utils_image as util


def _load_vepg_class():
    model_path = os.path.join(os.path.dirname(__file__), "model.py")
    spec = importlib.util.spec_from_file_location("vepg_model", model_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load VEPG model from {model_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "VEPG"):
        raise ImportError("VEPG class not found in model.py")
    return module.VEPG


VEPG = _load_vepg_class()


def forward(img_lq, model, tile=None, tile_overlap=32, scale=4):
    if tile is None:
        output = model(img_lq)
    else:
        b, c, h, w = img_lq.size()
        tile = min(tile, h, w)
        sf = scale

        stride = tile - tile_overlap
        h_idx_list = list(range(0, h - tile, stride)) + [h - tile]
        w_idx_list = list(range(0, w - tile, stride)) + [w - tile]
        E = torch.zeros(b, c, h * sf, w * sf).type_as(img_lq)
        W = torch.zeros_like(E)

        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                in_patch = img_lq[..., h_idx:h_idx + tile, w_idx:w_idx + tile]
                out_patch = model(in_patch)
                out_patch_mask = torch.ones_like(out_patch)

                E[..., h_idx * sf:(h_idx + tile) * sf, w_idx * sf:(w_idx + tile) * sf].add_(out_patch)
                W[..., h_idx * sf:(h_idx + tile) * sf, w_idx * sf:(w_idx + tile) * sf].add_(out_patch_mask)
        output = E.div_(W)

    return output


def run(model, data_path, save_path, tile, device):
    data_range = 1.0
    sf = 4
    border = sf

    if data_path.endswith('/'):
        data_path = data_path[:-1]
    input_img_list = sorted(glob.glob(os.path.join(data_path, '*.[jpJP][pnPN]*[gG]')))
    util.mkdir(save_path)

    for i, img_lr in enumerate(input_img_list):
        img_name, ext = os.path.splitext(os.path.basename(img_lr))
        img_lr = util.imread_uint(img_lr, n_channels=3)
        img_lr = util.uint2tensor4(img_lr, data_range)
        img_lr = img_lr.to(device)

        img_sr = forward(img_lr, model, tile)
        img_sr = util.tensor2uint(img_sr, data_range)

        util.imsave(img_sr, os.path.join(save_path, img_name + ext))


def main(model_dir, input_path, output_path, device=None):
    # utils_logger.logger_info("NTIRE2026-ImageSRx4", log_path="NTIRE2026-ImageSRx4.log")
    logger = logging.getLogger("NTIRE2026-ImageSRx4")

    torch.cuda.current_device()
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = False
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'Running on device: {device}')

    json_dir = os.path.join(os.getcwd(), "results.json")
    if not os.path.exists(json_dir):
        results = dict()
    else:
        with open(json_dir, "r", encoding="utf-8") as f:
            results = json.load(f)

    model = VEPG(model_dir)
    model.eval()
    tile = None
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)

    run(model, input_path, output_path, tile, device)
