import os
import csv
import random
import torch
from PIL import Image
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset
import pandas as pd
from .basicsr.realesrgan import RealESRGAN_degradation 

def to_rgb_if_rgba(img: Image.Image) -> Image.Image:
    if img.mode.upper() == "RGBA":
        rgb_img = Image.new("RGB", img.size, (255, 255, 255))
        rgb_img.paste(img, mask=img.split()[3])
        return rgb_img
    return img

# 和你原来一样的 [-1,1] 变换
restoration_transform = transforms.Compose([
    transforms.Lambda(to_rgb_if_rgba),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x * 2.0 - 1.0),  # [0,1] → [-1,1]
])

# 可以考虑给 face restoration 也准备一个默认 prompt
DEFAULT_PROMPTS = {
    "real_world_face_restoration": (
        "High-quality real-world face restoration, recover clear, sharp, natural-looking facial details from degraded, low-resolution, blurry, noisy, compressed, or old photos, "
        "realistic skin texture, accurate skin tone, sharp eyes and facial features, remove artifacts, no over-smoothing, no plastic look, photorealistic portrait, "
        "professional face enhancement, high fidelity, natural lighting"
    ),
}

class FaceRestorationDegradeDataset(Dataset):
    """
    用于 Real-World Face Restoration 的动态退化 Dataset
    - CSV 只需提供高质量 GT 图像路径
    - 低质量条件图 (input_image) 通过 RealESRGAN degradation 现场生成
    - 支持可变分辨率、动态条件数量（这里暂时固定为1张退化图，可扩展）
    
    CSV 示例：
    id,gt
    1,faces/high_quality/001.jpg
    2,faces/high_quality/002.png
    ...
    """
    def __init__(
        self,
        dataset_path: str,              # csv 路径
        root_dir: str = "",             # gt 图像的根目录（可选）
        cfg_prob: float = 0.0,          # classifier-free guidance 空文本概率
        transform=None,
        degrade_resize_bak: bool = True,   # 是否在 degrade_process 里 resize 回原大小
        device: str = 'cpu',          
    ):
        self.dataset_path = dataset_path
        self.root_dir = root_dir
        self.cfg_prob = cfg_prob
        self.transform = transform if transform is not None else restoration_transform
        self.default_caption = DEFAULT_PROMPTS["real_world_face_restoration"]
        self.degradation = RealESRGAN_degradation(device=device)
        self.degrade_resize_bak = degrade_resize_bak

        self.data = pd.read_csv(dataset_path)
        print(f"[FaceRestorationDegradeDataset] Loaded {len(self.data)} GT samples from {dataset_path}")

    def __len__(self):
        return len(self.data)

    def _parse_index_str(self, index_str: str):
        parts = index_str.split('-')
        assert len(parts) == 4, f"Invalid index_str format: {index_str}"
        
        idx = int(parts[0])
        target_h = int(parts[1])
        target_w = int(parts[2])
        condition_num = int(parts[3])
        return idx, target_h, target_w, condition_num

    def _degrade_image(self, gt_pil: Image.Image) -> torch.Tensor:
        """对 GT 进行 RealESRGAN 风格真实世界退化，返回 [-1,1] 的 tensor"""
        # 转成 numpy [0,1]
        gt_np = np.asarray(gt_pil) / 255.0
        
        # 调用 degrade_process
        # 假设它返回 (GT_tensor, LR_tensor)，形状 [1,C,H,W] 或 [C,H,W]
        GT_t, LR_t = self.degradation.degrade_process(
            gt_np, 
            resize_bak=self.degrade_resize_bak
        )
        
        # 通常 degrade_process 里的 GT 还是 [0,1]，我们转成 [-1,1]
        if GT_t.ndim == 4:
            GT_t = GT_t.squeeze(0)
        GT_t = GT_t * 2.0 - 1.0
        
        # LR 作为条件，通常也希望是 [-1,1]，保持一致
        if LR_t.ndim == 4:
            LR_t = LR_t.squeeze(0)
        LR_t = LR_t * 2.0 - 1.0
        
        return GT_t, LR_t

    def _get_face_restoration_sample(self, row, target_h, target_w):
        try:
            gt_path = row['gt']
            gt_img_pil = Image.open(os.path.join(self.root_dir, gt_path) if self.root_dir else gt_path).convert("RGB")
            
            # center crop resize
            w, h = gt_img_pil.size
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
            gt_cropped = gt_img_pil.crop((left, top, right, bottom))
            gt_resized = gt_cropped.resize((target_w, target_h), Image.LANCZOS)
            
            pixel_values, condition_img = self._degrade_image(gt_resized)
            
            caption = row.get('instruction', self.default_caption)
            if random.random() < self.cfg_prob:
                caption = ""
            
            image_condition = condition_img.unsqueeze(1)  # [C, 1, H, W]
            
            return {
                "real": pixel_values,                    # 干净 GT [-1,1]
                "image_condition": image_condition,      # 退化条件 [C, 1, H, W]
                "condition": caption,
                "neg_condition": "",
            }
        
        except Exception as e:
            print(f"⚠️ Error processing face restoration sample (id={row.get('id', 'unknown')}): {e}")
            next_idx = (int(row.get('id', 0)) + 1) % len(self.data)
            next_row = self.data.iloc[next_idx]
            return self._get_face_restoration_sample(next_row, target_h, target_w)

    def __getitem__(self, index_str: str) -> dict:
        idx, target_h, target_w, condition_num = self._parse_index_str(index_str)
        
        if condition_num != 1:
            print(f"Warning: condition_num={condition_num} not supported yet, using 1 degraded image.")
        
        row = self.data[self.data['id'] == idx]
        if len(row) == 0:
            raise ValueError(f"No row found for id={idx}")
        row = row.iloc[0]
        
        return self._get_face_restoration_sample(row, target_h, target_w)