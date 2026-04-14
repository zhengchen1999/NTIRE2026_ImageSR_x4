import os.path
import logging
import torch
import argparse
import json
import glob

import os
from PIL import Image
import torchvision.transforms.functional as F

from pprint import pprint
# from utils.model_summary import get_model_flops
# from utils import utils_logger
# from utils import utils_image as util

from models.team19_D2SR.model import DRRE as D2SR
from torchvision import transforms

from utils.wavelet_color_fix import adain_color_fix, wavelet_color_fix

def forward(img_lq, model, tile=None, tile_overlap=32, scale=4):
    if tile is None:
        # test the image as a whole
        output = model(img_lq)
    else:
        # test the image tile by tile
        b, c, h, w = img_lq.size()
        tile = min(tile, h, w)
        tile_overlap = tile_overlap
        sf = scale

        stride = tile - tile_overlap
        h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
        w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
        E = torch.zeros(b, c, h*sf, w*sf).type_as(img_lq)
        W = torch.zeros_like(E)

        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                in_patch = img_lq[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
                out_patch = model(in_patch)
                out_patch_mask = torch.ones_like(out_patch)

                E[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch)
                W[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch_mask)
        output = E.div_(W)

    return output


# def run(model, data_path, save_path, tile, device):
#     data_range = 1.0
#     sf = 4
#     border = sf

#     if data_path.endswith('/'):  # solve when path ends with /
#         data_path = data_path[:-1]
#     # scan all the jpg and png images
#     input_img_list = sorted(glob.glob(os.path.join(data_path, '*.[jpJP][pnPN]*[gG]')))
#     # save_path = os.path.join(args.save_dir, model_name, mode)
#     util.mkdir(save_path)

#     for i, img_lr in enumerate(input_img_list):

#         # --------------------------------
#         # (1) img_lr
#         # --------------------------------
#         img_name, ext = os.path.splitext(os.path.basename(img_lr))
#         img_lr = util.imread_uint(img_lr, n_channels=3)
#         img_lr = util.uint2tensor4(img_lr, data_range)
#         img_lr = img_lr.to(device)

#         # --------------------------------
#         # (2) img_sr
#         # --------------------------------
#         img_sr = forward(img_lr, model, tile)
#         img_sr = util.tensor2uint(img_sr, data_range)

#         util.imsave(img_sr, os.path.join(save_path, img_name+ext))


def main(model_dir, input_path, output_path, device=None):
    # utils_logger.logger_info("NTIRE2024-ImageSRx4", log_path="NTIRE2024-ImageSRx4.log")
    logger = logging.getLogger("NTIRE2024-ImageSRx4")

    # --------------------------------
    # basic settings
    # --------------------------------
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
        with open(json_dir, "r") as f:
            results = json.load(f)

    # --------------------------------
    # load model
    # --------------------------------
    # DAT baseline, ICCV 2023
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_path", type=str, default='/data2/NTIRE26/NTIRE2026/SR/Team19/D2SR/pretrained_models/stable-diffusion-2-1-base')
    parser.add_argument('--pretrained_path', type=str, default='/data2/NTIRE26/NTIRE2026/SR/Team19/D2SR/model_zoo/team19_D2SR/team00_dat.pth', help="path to a model state dict to be used")
    parser.add_argument("--process_size", type=int, default=512)
    parser.add_argument("--upscale", type=int, default=4)
    parser.add_argument("--align_method", type=str, choices=['wavelet', 'adain', 'nofix'], default="wavelet")
    parser.add_argument("--timestep", default=1, type=int, help="more bigger more smooth")
    parser.add_argument("--lambda_pix", default=1.25, type=float, help="the scale for pixel-level enhancement")
    parser.add_argument("--lambda_sem", default=1.0, type=float, help="the scale for sementic-level enhancements")
    parser.add_argument("--vae_decoder_tiled_size", type=int, default=512)
    parser.add_argument("--vae_encoder_tiled_size", type=int, default=512)
    parser.add_argument("--latent_tiled_size", type=int, default=512) 
    parser.add_argument("--latent_tiled_overlap", type=int, default=8) 
    parser.add_argument("--mixed_precision", type=str, default="fp16")
    parser.add_argument("--default",  action="store_true", help="use default or adjustale setting?") 

    parser.add_argument("--verbose", action='store_true')

    model_agrs = parser.parse_known_args()[0]

    # Call the processing function
    print(model_agrs)
    model_agrs.pretrained_path = model_dir
    model = D2SR(model_agrs)
    model.set_eval()
   
    input_img_list = sorted(glob.glob(os.path.join(input_path, '*.[jpJP][pnPN]*[gG]')))

    # Make the output directory
    os.makedirs(output_path, exist_ok=True)
    print(f'There are {len(input_img_list)} images.')

    for image_name in input_img_list:
        # Ensure the input image is a multiple of 8
        
        input_image = Image.open(image_name).convert('RGB')
        ori_width, ori_height = input_image.size
        rscale = model_agrs.upscale
        resize_flag = True

        if ori_width < model_agrs.process_size // rscale or ori_height < model_agrs.process_size // rscale:
            scale = (model_agrs.process_size // rscale) / min(ori_width, ori_height)
            input_image = input_image.resize((int(scale * ori_width), int(scale * ori_height)))
            resize_flag = True

        input_image = input_image.resize((input_image.size[0] * rscale, input_image.size[1] * rscale))
        new_width = input_image.width - input_image.width % 8
        new_height = input_image.height - input_image.height % 8
        input_image = input_image.resize((new_width, new_height), Image.LANCZOS)
        bname = os.path.basename(image_name)

        # Get caption (you can add the text prompt here)
        validation_prompt = ''

        # Translate the image
        with torch.no_grad():
            c_t = F.to_tensor(input_image).unsqueeze(0).cuda() * 2 - 1
            inference_time, output_image = model(model_agrs.default, c_t, prompt=validation_prompt)

        print(f"Inference time: {inference_time:.4f} seconds")

        output_image = output_image * 0.5 + 0.5
        output_image = torch.clip(output_image, 0, 1)
        output_pil = transforms.ToPILImage()(output_image[0].cpu())

        if model_agrs.align_method == 'adain':
            output_pil = adain_color_fix(target=output_pil, source=input_image)
        elif model_agrs.align_method == 'wavelet':
            output_pil = wavelet_color_fix(target=output_pil, source=input_image)

        if resize_flag:
            output_pil = output_pil.resize((int(model_agrs.upscale * ori_width), int(model_agrs.upscale * ori_height)))
        output_pil.save(os.path.join(output_path, bname))