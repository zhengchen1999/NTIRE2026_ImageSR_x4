from typing import Dict, Optional
import math
import random
import time
import io
import os

import numpy as np
import torch
from torch.utils import data
from PIL import Image

from HYPIR.dataset.utils import augment, random_crop_arr, center_crop_arr, load_file_meta
# 这里的 import 可以保留，或者如果你想清理得更干净，可以把不再用的 degradation import 删掉
from HYPIR.utils.common import instantiate_from_config


class RealESRGANDataset(data.Dataset):

    def __init__(
        self,
        file_meta,
        file_backend_cfg,
        out_size,
        crop_type,
        use_hflip,
        use_rot,
        # ----------------------------------------------------------------
        # 下面这些参数全部保留，为了兼容你的 Config 文件不报错
        # 但我们在内部不再使用它们
        # ----------------------------------------------------------------
        blur_kernel_size=None,
        kernel_list=None,
        kernel_prob=None,
        blur_sigma=None,
        betag_range=None,
        betap_range=None,
        sinc_prob=None,
        blur_kernel_size2=None,
        kernel_list2=None,
        kernel_prob2=None,
        blur_sigma2=None,
        betag_range2=None,
        betap_range2=None,
        sinc_prob2=None,
        final_sinc_prob=None,
        p_empty_prompt=0.0,
        return_file_name=False,
    ):
        super(RealESRGANDataset, self).__init__()
        self.file_meta = file_meta
        self.image_files = load_file_meta(file_meta)
        self.file_backend = instantiate_from_config(file_backend_cfg)
        self.out_size = out_size
        self.crop_type = crop_type
        assert self.crop_type in ["none", "center", "random"]

        self.use_hflip = use_hflip
        self.use_rot = use_rot

        self.scale = 4  # 固定 4 倍降采样
        self.p_empty_prompt = p_empty_prompt
        self.return_file_name = return_file_name

    def load_gt_image(self, image_path: str, max_retry: int = 5) -> Optional[np.ndarray]:
        image_bytes = None
        while image_bytes is None:
            if max_retry == 0:
                return None
            try:
                image_bytes = self.file_backend.get(image_path)
            except:
                # file does not exist
                return None
            max_retry -= 1
            if image_bytes is None:
                time.sleep(0.5)

        try:
            # failed to decode image bytes
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except:
            return None

        if self.crop_type != "none":
            if image.height == self.out_size and image.width == self.out_size:
                image = np.array(image)
            else:
                if self.crop_type == "center":
                    image = center_crop_arr(image, self.out_size)
                elif self.crop_type == "random":
                    image = random_crop_arr(image, self.out_size, min_crop_frac=0.7)
        else:
            assert image.height == self.out_size and image.width == self.out_size
            image = np.array(image)
        # hwc, rgb, 0,255, uint8
        return image

    @torch.no_grad()
    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        # -------------------------------- Load hq images -------------------------------- #
        # load gt image
        img_gt = None
        while img_gt is None:
            # load meta file
            image_file = self.image_files[index]
            gt_path = image_file["image_path"]
            prompt = image_file["prompt"]
            img_gt = self.load_gt_image(gt_path)
            if img_gt is None:
                print(f"failed to load {gt_path}, try another image")
                index = random.randint(0, len(self) - 1)

        # -------------------------------- Pre-process -------------------------------- #
        # 原代码逻辑：PIL(RGB) -> numpy(RGB) -> float(BGR) -> Augment -> float(BGR) -> Tensor(RGB)
        # 我们保持这个颜色转换逻辑，以免影响 augment 的行为
        
        # [0, 255] RGB -> [0, 1] BGR float32
        img_hq = (img_gt[..., ::-1] / 255.0).astype(np.float32)
        
        if np.random.uniform() < self.p_empty_prompt:
            prompt = ""

        # -------------------- Do augmentation: flip, rotation -------------------- #
        img_hq = augment(img_hq, self.use_hflip, self.use_rot)

        # -------------------- Generate LQ using Bicubic Downsample -------------------- #
        # 1. 转回 PIL Image (RGB) 以便使用标准的 PIL Bicubic
        # img_hq 目前是 BGR [0,1]，先转成 RGB uint8
        img_hq_rgb_uint8 = (img_hq[..., ::-1] * 255.0).round().astype(np.uint8)
        img_pil = Image.fromarray(img_hq_rgb_uint8)
        
        h, w = img_pil.height, img_pil.width
        
        # 2. 4倍下采样
        img_lr_pil = img_pil.resize(
            (w // self.scale, h // self.scale), 
            resample=Image.BICUBIC
        )
        
        # -------------------- Convert to Tensor -------------------- #
        
        # HQ: BGR [0,1] -> RGB Tensor [C, H, W]
        img_hq_tensor = torch.from_numpy(
            img_hq[..., ::-1].transpose(2, 0, 1).copy()
        ).float()
        
        # LQ: PIL RGB -> numpy RGB -> Tensor [C, H, W] [0, 1]
        img_lr_np = np.array(img_lr_pil).astype(np.float32) / 255.0
        img_lr_tensor = torch.from_numpy(
            img_lr_np.transpose(2, 0, 1).copy()
        ).float()

        # -------------------- Return Data -------------------- #
        # 不再返回 kernel1, kernel2, sinc_kernel
        # 直接返回 hq 和 lq
        data = {
            "hq": img_hq_tensor,
            "lq": img_lr_tensor, # 这里直接给出了 Bicubic 的 LQ
            "txt": prompt,
        }
        
        if self.return_file_name:
            data["filename"] = os.path.basename(gt_path)
            
        return data

    def __len__(self) -> int:
        return len(self.image_files)
