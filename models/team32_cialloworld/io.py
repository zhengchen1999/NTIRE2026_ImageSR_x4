"""
Inference module for Multi-Model Fusion
Team: cialloworld
"""
import os.path as osp
import logging
import torch
import argparse
import json
import glob
import numpy as np

import sys
sys.path.insert(0, osp.dirname(osp.dirname(osp.dirname(__file__))))

from utils import utils_image as util


def main(model_dir, input_path, output_path, device=None):
    """
    Main inference function

    Args:
        model_dir: Path to model weights
        input_path: Path to input images
        output_path: Path to save output images
        device: Device to run inference
    """
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    # Setup device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f'Running on device: {device}')
    logger.info(f'Loading model from: {model_dir}')

    # Load model architecture
    from .model import HAT_DAT_NAFNet_Fusion_3x3

    model = HAT_DAT_NAFNet_Fusion_3x3(
        upscale=4,
        in_chans=3,
        img_range=1.,
        freeze_backbone=False,
    )

    # Load weights
    checkpoint = torch.load(model_dir, map_location='cpu')

    # Handle different checkpoint formats
    if 'params' in checkpoint:
        state_dict = checkpoint['params']
    elif 'params_ema' in checkpoint:
        state_dict = checkpoint['params_ema']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # Load pretrained backbones if available
    if hasattr(model, 'load_pretrain'):
        hat_path = osp.join(osp.dirname(model_dir), 'HAT-L_SRx4_ImageNet-pretrain.pth')
        dat_path = osp.join(osp.dirname(model_dir), 'DAT_2_x4.pth')

        if osp.exists(hat_path):
            logger.info(f'Loading HAT pretrained from: {hat_path}')
            model.load_pretrain(hat_path=hat_path)

        if osp.exists(dat_path):
            logger.info(f'Loading DAT pretrained from: {dat_path}')
            model.load_pretrain(dat_path=dat_path)

    # Load fusion weights
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    for k, v in model.named_parameters():
        v.requires_grad = False

    model = model.to(device)
    logger.info("Model loaded successfully")

    # Run inference
    util.mkdir(output_path)

    # Get all images
    input_img_list = sorted(glob.glob(osp.join(input_path, '*.[jpJP][pnPN]*[gG]')))
    logger.info(f'Found {len(input_img_list)} images in {input_path}')

    for i, img_lr_path in enumerate(input_img_list):
        img_name = osp.splitext(osp.basename(img_lr_path))[0]
        ext = osp.splitext(img_lr_path)[1]

        output_file = osp.join(output_path, img_name + ext)

        if osp.exists(output_file):
            logger.info(f'[{i+1}/{len(input_img_list)}] Skipping {img_name} (already exists)')
            continue

        # Read image (model uses img_range=1.0, input in [0,1])
        img_lr = util.imread_uint(img_lr_path, n_channels=3)
        img_lr = util.uint2tensor4(img_lr, 1.0)
        img_lr = img_lr.to(device)

        # Pad to multiple of 16 (required by HAT window_size and NAFNet)
        _, _, h, w = img_lr.shape
        pad_h = (16 - h % 16) % 16
        pad_w = (16 - w % 16) % 16
        if pad_h > 0 or pad_w > 0:
            img_lr = torch.nn.functional.pad(img_lr, (0, pad_w, 0, pad_h), mode='reflect')

        # Inference
        with torch.no_grad():
            img_sr = model(img_lr)

        # Crop to original size * 4
        if pad_h > 0 or pad_w > 0:
            img_sr = img_sr[:, :, :h * 4, :w * 4]

        # Save
        img_sr = util.tensor2uint(img_sr, 1.0)
        util.imsave(img_sr, output_file)

        logger.info(f'[{i+1}/{len(input_img_list)}] Processed {img_name}')

    logger.info(f'Inference complete. Results saved to: {output_path}')
