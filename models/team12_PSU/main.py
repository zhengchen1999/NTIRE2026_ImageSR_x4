import os
import time
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from tqdm import tqdm
from .model import DUSKAN

def main(model_dir, input_path, output_path, device):
    print(f"--- Initializing DUSKAN ---")
    model = DUSKAN(
        in_channels=3,
        base_width=48,
        middle_blk_num=1,
        enc_blk_nums=[1, 1, 1, 1],
        dec_blk_nums=[1, 1, 1, 1],
        d_state=64
    )
    
    print(f"Loading weights from: {model_dir}")
    checkpoint = torch.load(model_dir, map_location=device)
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'], strict=True)
    else:
        model.load_state_dict(checkpoint, strict=True)
        
    model.to(device)
    model.eval()

    os.makedirs(output_path, exist_ok=True)
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    image_files = sorted([f for f in os.listdir(input_path) if os.path.splitext(f)[1].lower() in valid_exts])
    
    transform = transforms.ToTensor()
    total_time = 0.0
    
    for img_name in tqdm(image_files, desc="DUSKAN Inferencing"):
        img_p = os.path.join(input_path, img_name)
        img = Image.open(img_p).convert("RGB")
        input_tensor = transform(img).unsqueeze(0).to(device)
        
        # --- SR SPECIFIC: Upscale x4 by Bicubic before the model ---
        input_tensor = F.interpolate(
            input_tensor, scale_factor=4, mode='bicubic', align_corners=False
        )
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        start_time = time.time()
        
        with torch.no_grad():
            output_tensor = model(input_tensor)
            output_tensor = torch.clamp(output_tensor, 0, 1)
            
        if device.type == 'cuda':
            torch.cuda.synchronize()
        total_time += (time.time() - start_time)
        
        # Save output with exactly the same name as the input
        save_path = os.path.join(output_path, img_name)
        save_image(output_tensor, save_path)
        
    avg_time = total_time / len(image_files) if image_files else 0
    print(f"\n--- Inference Complete ---")
    print(f"Average time per image (for readme.txt): {avg_time:.4f}s")