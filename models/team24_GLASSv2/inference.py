import os
import glob
import cv2
import torch
import numpy as np

# Import your architecture directly
from basicsr.archs.glassv2_arch import GLASSv2

def main(model_dir, input_path, output_path, device):
    """
    This function acts as the bridge for the NTIRE evaluation script.
    """
    # 1. Initialize your model with exact parameters from your config
    model = GLASSv2(
        upscale=4,
        in_chans=3,
        img_size=64,
        img_range=1.,
        embed_dim=180,
        d_state=16,
        depths=[6, 6, 6, 6, 6, 6],
        num_heads=[6, 6, 6, 6, 6, 6],
        window_size=16,
        inner_rank=64,
        num_tokens=128,
        convffn_kernel_size=5,
        mlp_ratio=2.,
        upsampler='pixelshuffle',
        resi_connection='1conv'
    ).to(device)

    # 2. Load the trained weights (.pth file)
    state_dict = torch.load(model_dir, map_location=device)
    
    # BasicSR usually saves weights under 'params_ema' or 'params'
    if 'params_ema' in state_dict:
        state_dict = state_dict['params_ema']
    elif 'params' in state_dict:
        state_dict = state_dict['params']
        
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    # 3. Find all images in the input directory
    img_paths = sorted(glob.glob(os.path.join(input_path, '*.[pP][nN][gG]')) + 
                       glob.glob(os.path.join(input_path, '*.[jJ][pP][gG]')))

    # 4. Process each image
    for img_path in img_paths:
        img_name = os.path.basename(img_path)
        
        # Read image
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Convert to tensor [1, 3, H, W] and normalize to [0, 1]
        img_tensor = torch.from_numpy(img).float() / 255.0
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(device)

        # Forward pass
        with torch.no_grad():
            output_tensor = model(img_tensor)

        # Convert back to numpy array [H, W, 3] and scale to [0, 255]
        output = output_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        output = np.clip(output * 255.0, 0, 255).astype(np.uint8)
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

        # Save output image
        save_path = os.path.join(output_path, img_name)
        cv2.imwrite(save_path, output)