import os
import glob
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from models.network_swinir import SwinIR

def tensor2img(tensor):
    t = tensor.float().detach().cpu().clamp(0, 1)
    if t.ndim == 4:
        t = t[0]
    img = t.permute(1, 2, 0).numpy()
    return (img * 255.0).round().astype(np.uint8)[:, :, ::-1]  # RGB -> BGR for opencv

def main(model_dir, input_path, output_path, device):
    print(f"Loading SwinIR 180 from {model_dir}")
    model = SwinIR(
        upscale=4,
        img_size=48, # Required for correct attention mask
        window_size=8,
        embed_dim=180,
        depths=[6, 6, 6, 6, 6, 6],
        num_heads=[6, 6, 6, 6, 6, 6],
        mlp_ratio=2.0,
        img_range=1.0,
        upsampler='pixelshuffle',
        resi_connection='1conv'
    )

    ckpt = torch.load(model_dir, map_location='cpu')
    if 'ema_state_dict' in ckpt:
        state_dict = ckpt['ema_state_dict']
    elif 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
    else:
        state_dict = ckpt
        
    # Remove prefixes if any
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('ema_model.'):
            new_state_dict[k[10:]] = v
        elif k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
            
    model.load_state_dict(new_state_dict, strict=True)
    model.eval()
    model.to(device)

    img_list = sorted(glob.glob(os.path.join(input_path, "*")))
    print(f"Found {len(img_list)} images in {input_path}")
    
    with torch.no_grad():
        for p in tqdm(img_list, desc="Processing"):
            img_bgr = cv2.imread(p, cv2.IMREAD_COLOR)
            if img_bgr is None:
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            
            lr_t = torch.from_numpy(np.ascontiguousarray(img_rgb)).permute(2, 0, 1).unsqueeze(0).to(device)
            
            _, _, h_old, w_old = lr_t.size()
            window_size = 8
            
            # Pad to window_size multiple
            h_pad = (window_size - h_old % window_size) % window_size
            w_pad = (window_size - w_old % window_size) % window_size
            if h_pad > 0 or w_pad > 0:
                lr_t = F.pad(lr_t, (0, w_pad, 0, h_pad), mode="reflect")
                
            sr = model(lr_t)
            
            # Crop back to original scale
            sr = sr[:, :, :h_old * 4, :w_old * 4]
            
            sr_img = tensor2img(sr)
            
            basename = os.path.basename(p)
            save_path = os.path.join(output_path, basename)
            cv2.imwrite(save_path, sr_img)
