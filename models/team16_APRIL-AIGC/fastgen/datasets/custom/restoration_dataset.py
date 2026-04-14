# 用于 SIRR / Shadow Removal / AI Flash Portrait
import os
import csv
import random
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import pandas as pd

def to_rgb_if_rgba(img: Image.Image) -> Image.Image:
    if img.mode.upper() == "RGBA":
        rgb_img = Image.new("RGB", img.size, (255, 255, 255))
        rgb_img.paste(img, mask=img.split()[3])
        return rgb_img
    return img

restoration_transform = transforms.Compose([
    transforms.Lambda(to_rgb_if_rgba),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x * 2.0 - 1.0),  # [0,1] → [-1,1]
])

DEFAULT_PROMPTS = {
    "sirr": "High-quality single image reflection removal in the wild, remove complex real-world reflections from glass windows, mirrors, or transparent surfaces, recover clear transmission layer with natural colors, sharp details, realistic textures, no ghosting, no residual reflections, photorealistic, professional photography, perfect alignment, high fidelity",
    "shadow_removal": "High-quality image shadow removal in real scenes, accurately detect and remove cast shadows while preserving original scene illumination, natural lighting consistency, maintain texture details, colors, and object boundaries, no over-brightening, no halo artifacts, photorealistic result, clean shadow-free image, high realism, high fidelity",
    "ai_flash_portrait": "AI flash portrait restoration for low-light real-world scenes, enhance underexposed portrait with natural flash lighting, clean skin, sharp facial details, accurate skin tone, pleasant background preservation, balanced exposure, no noise, no overexposure, realistic portrait photography, high quality, professional look, high fidelity",

    "shadow_removal_v2": "Remove cast shadows accurately while keeping original scene illumination, colors, and textures intact. Natural even lighting, no darkening or overexposure, photorealistic result with clean boundaries and high detail fidelity.",
    "sirr_v2": "Remove reflections from glass windows, mirrors, and transparent surfaces in the input image, accurately recovering the clear transmission layer with perfect color fidelity and natural textures. Preserve original scene illumination, colors, sharp details, and exact tones in non-reflective areas without any shifts, distortions, or alterations. Subtly reveal true underlying content only where reflections exist, photorealistic, high-fidelity restoration, clean seamless result.",

    "raindrop_v2" : "the same scene as the input image but with all raindrops completely removed from the glass or lens, crystal clear view without any water streaks or distortions, no refraction artifacts, sharp details on the background objects, unchanged composition and lighting, hyperrealistic restoration, high resolution, no blur, professional image editing,  high-fidelity restoration",
    "lowlightv2_v2" : "low-light fix, reduce noise, keep sharp details, enhance shadow visibility, mild contrast brightness boost, true natural colors, no artifacts over-enhance, preserve original look",

    "transfer_v2" : "Learn the retouch delta between reference pair image 3 and image 2, then apply the same retouch delta to image 1. Preserve all scene content from image 1 exactly; modify only photographic rendering attributes (color tone, contrast, dynamic range, luminance distribution, highlight/shadow behavior). High consistency, natural colors, seamless professional retouch.",
}

class RestorationMultiCondDataset(Dataset):
    """
    用于图像恢复任务的多条件 Dataset
    - 示例 CSV 内容：
      input_image,gt
      "img1.jpg,img2.jpg,img3.jpg",gt.jpg
      "cond1.png",gt2.jpg
    - real:           [C, H, W]        
    - image_condition:[C, T, H, W]      
    - condition:      str           
    """
    def __init__(
        self,
        dataset_path: str,
        task: str,
        root_dir: str = "",
        cfg_prob: float = 0.0,
        transform=None,
    ):
        self.dataset_path = dataset_path
        self.root_dir = root_dir
        self.cfg_prob = cfg_prob
        self.transform = restoration_transform
        self.default_caption = DEFAULT_PROMPTS[task]

        self.data = pd.read_csv(dataset_path)
        print(f"[RestorationMultiCondDataset] Loaded {len(self.data)} valid samples from {dataset_path}")

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


    def _load_and_centercrop_resize_image(self, rel_path, target_h, target_w) -> torch.Tensor:
        full_path = os.path.join(self.root_dir, rel_path) if self.root_dir else rel_path
        try:
            img = Image.open(full_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Failed to open image: {full_path} → {e}")

        w, h = img.size
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
        img = img.resize((target_w, target_h), Image.LANCZOS)
        
        return self.transform(img)

    def _get_ti2i_sample(self, row, target_h, target_w):
        try:
            output_image = row['gt']
            input_images_str = row.get('input_image', '')
            
            if pd.isna(input_images_str) or input_images_str == '':
                input_images = []
            else:
                input_images = [img.strip() for img in str(input_images_str).split(',')]
            
        
            gt_img = self._load_and_centercrop_resize_image(output_image, target_h, target_w)
            C, H, W = gt_img.shape
            
            caption = row.get('instruction', self.default_caption)
            
            pixel_values = gt_img
        
            
      
            cond_imgs = []
            for img_path in input_images:
                cond_img = self._load_and_centercrop_resize_image(img_path, H, W)
                cond_imgs.append(cond_img)
            
            num_cond_images = len(cond_imgs)
            cond = torch.stack(cond_imgs, dim=0)  # [m, C, H, W]
            image_condition = cond.permute(1, 0, 2, 3)  # [C, m, H, W]
        
            
            
            return {
                "real": pixel_values,
                "image_condition": image_condition,
                "condition": caption,
                "neg_condition" : "",
            }
        
        except Exception as e:
            print(f"⚠️  Error loading T2I/TI2I sample (id={row['id']}): {e}")
            # 返回下一个样本
            next_idx = (int(row['id']) + 1) % len(self.data)
            next_row = self.data.iloc[next_idx]
            return self._get_ti2i_sample(next_row, target_h, target_w)


    def __getitem__(self, index_str: str) -> dict:
        idx, target_h, target_w, condition_num = self._parse_index_str(index_str)

        row = self.data[self.data['id'] == idx]
        row = row.iloc[0]  # 取第一行
        
        
        return self._get_ti2i_sample(row, target_h, target_w)

   