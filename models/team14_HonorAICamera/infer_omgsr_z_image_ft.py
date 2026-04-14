import sys
import os
import argparse
from PIL import Image
import torch
from torchvision import transforms
import torchvision.transforms.functional as F
from tqdm import tqdm
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from models.HonorAICamera00_OmgZimage.wavelet_color_fix import wavelet_reconstruction
import glob
from models.HonorAICamera00_OmgZimage.omgsr_z_image_infer_model_ft import OMGSR_ZImage_Infer
from diffusers.training_utils import free_memory
from diffusers import ZImagePipeline
import cv2
import numpy as np

def sliding_windows(w: int, h: int, tile_width: int, tile_height: int, tile_stridew: int, tile_strideh: int) :
    # 生成分块坐标
    hi_list = list(range(0, h - tile_height + 1, tile_strideh))
    if (h - tile_height) % tile_strideh != 0:
        hi_list.append(h - tile_height)

    wi_list = list(range(0, w - tile_width + 1, tile_stridew))
    if (w - tile_width) % tile_stridew != 0:
        wi_list.append(w - tile_width)

    coords = []
    for hi in hi_list:
        for wi in wi_list:
            coords.append((hi, hi + tile_height, wi, wi + tile_width))
    return coords

def gaussian_weights(tile_width: int, tile_height: int) -> np.ndarray:
    '''Generates a gaussian mask of weights for tile contributions'''
    latent_width = tile_width
    latent_height = tile_height
    var = 0.01
    midpoint = (latent_width - 1) / 2  # -1 because index goes from 0 to latent_width - 1
    x_probs = [
        np.exp(-(x - midpoint) * (x - midpoint) / (latent_width * latent_width) / (2 * var)) / np.sqrt(2 * np.pi * var)
        for x in range(latent_width)]
    midpoint = latent_height / 2
    y_probs = [
        np.exp(-(y - midpoint) * (y - midpoint) / (latent_height * latent_height) / (2 * var)) / np.sqrt(2 * np.pi * var)
        for y in range(latent_height)]
    weights = np.outer(y_probs, x_probs)
    return weights

def wavelet_normalize(source, target, eps=1e-12):

    source = torch.from_numpy(source.copy()[None].transpose(0,3,1,2))
    target = torch.from_numpy(target.copy()[None].transpose(0,3,1,2))
    normalized = wavelet_reconstruction(source,target)

    normalized = normalized[0].data.numpy().transpose(1,2,0)
    return normalized

def main(model_dir, input_path, output_path, device):
    # Initialize the model
    mid_timestep = 121
    guidance_scale = 1.0
    z_image_turbo_path = './Z-Image-Turbo'
    weight_dtype = torch.bfloat16
    prompt = ''
    upscale = 4
    omgsr = OMGSR_ZImage_Infer(
        z_image_turbo_path, model_dir, device=device, 
        guidance_scale=guidance_scale, mid_timestep=mid_timestep)


    text_encoding_pipeline = ZImagePipeline.from_pretrained(
        z_image_turbo_path, transformer=None, vae=None, torch_dtype=weight_dtype
    )
    text_encoding_pipeline = text_encoding_pipeline.to("cuda")
    with torch.no_grad():
        prompt_embeds,_= text_encoding_pipeline.encode_prompt(
            prompt
        )
    

    # Release the pipeline
    text_encoding_pipeline = text_encoding_pipeline.to("cpu")  # Move to CPU first
    del text_encoding_pipeline
    free_memory()


   
    image_names = sorted(glob.glob(f"{input_path}/*.png") + glob.glob(f"{input_path}/*.jpg") + glob.glob(f"{input_path}/*.jpeg"))
    if len(image_names) == 0:
        print('Warning : No image !!!')

    # Make the output directory
    os.makedirs(output_path, exist_ok=True)
    print(f"There are {len(image_names)} images.")

    for image_name in tqdm(image_names):
        lr = cv2.imread(image_name).astype(np.float32)
        lr = lr[:, :, ::-1] / 255.0

        if upscale != 1:
            lr = cv2.resize(lr,(lr.shape[1] * upscale, lr.shape[0] * upscale,))

        x = lr.copy()

        h, w, _ = lr.shape
        ovp = 128
        tile_w, tile_h, tile_stridew, tile_strideh = 1024 , 1024 , 1024 - 2 * ovp + ovp, 1024 - 2 * ovp + ovp

        padding_width, padding_height = max(0, tile_w - w), max(0, tile_h - h)
        lr = np.pad(lr, ((0, padding_height), (0, padding_width), (0, 0)), mode="constant",
                    constant_values=(0.8, 0.8))  # 0808
        lr = np.pad(lr, ((128, 128), (128, 128), (0, 0)), mode="constant", constant_values=(0.8, 0.8))  # 0808
        h, w, _ = lr.shape
        lr_py = torch.from_numpy(lr[np.newaxis, :].transpose((0, 3, 1, 2)).copy())

        x = lr_py

        _, _, h, w = x.shape

        '''fix分辨率'''

        tiles = tqdm(sliding_windows(w, h, tile_w, tile_h, tile_stridew, tile_strideh), unit="tile", leave=False)
        eps = torch.zeros_like(x)
        count = torch.zeros_like(x, dtype=torch.float32)
        weights = gaussian_weights(tile_w, tile_h)[None, None]
        pad_size = 128

        
        
        for hi, hi_end, wi, wi_end in tiles:
            tiles.set_description(f"Process tile ({hi} {hi_end}), ({wi} {wi_end})")

            # 1. 原始分块提取
            tile_x = x[:, :, hi:hi_end, wi:wi_end]

            with torch.no_grad():
                lq_img = tile_x.to(device=device, dtype=weight_dtype) * 2. - 1.
                output_image = omgsr(lq_img.to(), prompt_embeds)
            output_image = output_image.float()
            hr_patch = (output_image.cpu().numpy()[0][:,0:hi_end - hi, 0:wi_end - wi] / 2 + 0.5)
            
            hr_patch = hr_patch.transpose(1, 2, 0)

            lr_patch = lr[hi:hi_end, wi:wi_end, :]

            hr_patch = wavelet_normalize(hr_patch, lr_patch)

            tile_eps = np.expand_dims(hr_patch, axis=0)  # NHWC
            tile_eps = torch.from_numpy(tile_eps.transpose((0, 3, 1, 2)).copy())  # N3HW

            # accumulate noise
            eps[:, :, hi:hi_end, wi:wi_end] += tile_eps * weights
            count[:, :, hi:hi_end, wi:wi_end] += weights

            # average on noise (score)
        eps.div_(count)
        eps = eps.float().squeeze(0).numpy()
        out_image = eps.transpose((1, 2, 0))  # out_image: (4096, 3072, 3)

        # 如果需要旋转回来

        
        out_image = out_image[:lr.shape[0], :lr.shape[1], :]
        hr = wavelet_normalize(out_image, lr)
        

        lr_py = lr[None, :]  # NHWC
        lr_py = torch.from_numpy(lr_py.transpose(0, 3, 1, 2))  # NCHW
        hr_py = hr[None, :]
        hr_py = torch.from_numpy(hr_py.transpose(0, 3, 1, 2))  # NCHW

        
        output = hr_py
        
        output = output.clamp(0.0, 1.0)
        output = output.cpu().permute(0, 2, 3, 1).float().detach().numpy()[0]
        

        output = output[pad_size : -(pad_size + padding_height),
                 pad_size : -(pad_size + padding_width), :]

        
        image = Image.fromarray(np.uint8(output * 255))  # RGB排列
        lr_img = Image.fromarray(np.uint8(lr * 255))
        
        save_name = os.path.basename(image_name)[:-3] + "png"
        image.save(os.path.join(output_path, save_name))
        
        print(save_name, 'done')

    
