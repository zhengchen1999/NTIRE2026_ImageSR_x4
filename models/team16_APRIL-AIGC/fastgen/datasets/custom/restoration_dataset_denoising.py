import os
import random

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from fastgen.datasets.custom.basicsr.realesrgan import RealESRGAN_degradation
from fastgen.datasets.custom.restoration_dataset import restoration_transform


DEFAULT_PROMPT = (
    "Remove strong Gaussian noise from the input image while preserving the exact scene content, "
    "natural colors, clean edges, fine textures, and realistic details. High-fidelity denoising, "
    "no plastic smoothing, no ringing, photorealistic restoration."
)


class ImageDenoisingDataset(Dataset):
    def __init__(
        self,
        dataset_path: str,
        root_dir: str = "",
        cfg_prob: float = 0.0,
        sigma: float = 50.0,
        deterministic_noise: bool = True,
        noise_seed: int = 3407,
        realesrgan_prob: float = 0.1,
        degrade_resize_bak: bool = True,
        degradation_device: str = "cpu",
        transform=None,
    ):
        self.dataset_path = dataset_path
        self.root_dir = root_dir
        self.cfg_prob = cfg_prob
        self.sigma = float(sigma)
        self.deterministic_noise = deterministic_noise
        self.noise_seed = int(noise_seed)
        self.realesrgan_prob = float(realesrgan_prob)
        if not 0.0 <= self.realesrgan_prob <= 1.0:
            raise ValueError(f"realesrgan_prob must be in [0, 1], got {realesrgan_prob}")
        self.degrade_resize_bak = degrade_resize_bak
        self.degradation_device = degradation_device
        self._degradation = None
        self.transform = transform if transform is not None else restoration_transform
        self.default_caption = DEFAULT_PROMPT

        self.data = pd.read_csv(dataset_path)
        if "id" not in self.data.columns:
            raise ValueError(f"Missing required column `id` in {dataset_path}")
        self.data_by_id = self.data.set_index("id", drop=False)
        print(f"[ImageDenoisingDataset] Loaded {len(self.data)} GT samples from {dataset_path}")

    def __len__(self):
        return len(self.data)

    def _parse_index_str(self, index_str: str):
        parts = index_str.split("-")
        if len(parts) != 4:
            raise ValueError(f"Invalid index_str format: {index_str}")
        idx = int(parts[0])
        target_h = int(parts[1])
        target_w = int(parts[2])
        condition_num = int(parts[3])
        return idx, target_h, target_w, condition_num

    def _resolve_path(self, rel_path: str) -> str:
        return os.path.join(self.root_dir, rel_path) if self.root_dir else rel_path

    def _load_gt_image(self, rel_path: str, target_h: int, target_w: int) -> Image.Image:
        full_path = self._resolve_path(rel_path)
        try:
            img = Image.open(full_path).convert("RGB")
        except Exception as exc:
            raise RuntimeError(f"Failed to open image: {full_path} -> {exc}") from exc

        w, h = img.size
        if (h, w) == (target_h, target_w):
            return img

        if h / w > target_h / target_w:
            new_w = int(w)
            new_h = int(new_w * target_h / target_w)
        else:
            new_h = int(h)
            new_w = int(new_h * target_w / target_h)

        left = (w - new_w) / 2
        top = (h - new_h) / 2
        right = (w + new_w) / 2
        bottom = (h + new_h) / 2

        img = img.crop((left, top, right, bottom))
        return img.resize((target_w, target_h), Image.LANCZOS)

    def _sample_noise(self, row_id: int, shape):
        if self.deterministic_noise:
            rng = np.random.default_rng(self.noise_seed + int(row_id))
            return rng.normal(0.0, self.sigma, size=shape)
        return np.random.normal(0.0, self.sigma, size=shape)

    def _get_degradation(self):
        if self._degradation is None:
            self._degradation = RealESRGAN_degradation(device=self.degradation_device)
        return self._degradation

    def _should_use_realesrgan(self) -> bool:
        return self.realesrgan_prob > 0.0 and random.random() < self.realesrgan_prob

    def _degrade_with_realesrgan(self, gt_img: Image.Image):
        gt_np = np.asarray(gt_img, dtype=np.float32) / 255.0
        gt_tensor, condition_tensor = self._get_degradation().degrade_process(
            gt_np,
            resize_bak=self.degrade_resize_bak,
        )
        if gt_tensor.ndim == 4:
            gt_tensor = gt_tensor.squeeze(0)
        if condition_tensor.ndim == 4:
            condition_tensor = condition_tensor.squeeze(0)
        # RealESRGAN may flip the GT internally, so keep the returned GT tensor aligned with the condition.
        return gt_tensor * 2.0 - 1.0, condition_tensor * 2.0 - 1.0

    def _get_caption(self, row) -> str:
        caption = row.get("instruction", self.default_caption)
        if pd.isna(caption) or caption == "":
            caption = self.default_caption
        if random.random() < self.cfg_prob:
            return ""
        return str(caption)

    def _get_denoising_sample(self, row, target_h: int, target_w: int):
        try:
            gt_img = self._load_gt_image(row["gt"], target_h, target_w)
            if self._should_use_realesrgan():
                pixel_values, condition_img = self._degrade_with_realesrgan(gt_img)
            else:
                gt_np = np.asarray(gt_img, dtype=np.float32)
                noisy_np = np.clip(gt_np + self._sample_noise(int(row["id"]), gt_np.shape), 0.0, 255.0).astype(
                    np.uint8
                )
                pixel_values = self.transform(gt_img)
                condition_img = self.transform(Image.fromarray(noisy_np))

            return {
                "real": pixel_values,
                "image_condition": condition_img.unsqueeze(1),
                "condition": self._get_caption(row),
                "neg_condition": "",
            }
        except Exception as exc:
            print(f"⚠️ Error processing denoising sample (id={row.get('id', 'unknown')}): {exc}")
            next_idx = (int(row.get("id", 0)) + 1) % len(self.data)
            next_row = self.data_by_id.loc[next_idx]
            return self._get_denoising_sample(next_row, target_h, target_w)

    def __getitem__(self, index_str: str):
        idx, target_h, target_w, condition_num = self._parse_index_str(index_str)
        if condition_num != 1:
            print(f"Warning: condition_num={condition_num} not supported yet, using one noisy image.")
        row = self.data_by_id.loc[idx]
        return self._get_denoising_sample(row, target_h, target_w)
