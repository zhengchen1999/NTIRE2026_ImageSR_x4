import sys
from typing import Callable
import torch
import os
import time
import torch
from peft import PeftModel
from diffusers import AutoencoderKL, ZImageTransformer2DModel
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import math
import numpy as np


def encode_images(pixels: torch.Tensor, vae: torch.nn.Module, weight_dtype):
    pixel_latents = vae.encode(pixels.to(vae.dtype)).latent_dist.sample()
    pixel_latents = (pixel_latents - vae.config.shift_factor) * vae.config.scaling_factor
    return pixel_latents.to(weight_dtype)



class OMGSR_ZImage_Infer(torch.nn.Module):
    def __init__(self, z_image_path, lora_path, device, weight_dtype=torch.bfloat16, mid_timestep=295, guidance_scale=1.0):
        super().__init__()

        vae = AutoencoderKL.from_pretrained(z_image_path, subfolder="vae")

        z_image_transformer = ZImageTransformer2DModel.from_pretrained(z_image_path, subfolder="transformer") 

        
        vae.requires_grad_(False)
        z_image_transformer.requires_grad_(False)

        vae.to(device=device, dtype=weight_dtype)  
        z_image_transformer.to(dtype=weight_dtype, device=device)

        print("Loading adapers...")
        # stage2 可能是 31k 或 32k (readme 写 32k，代码原为 31k)
        stage2_dir = os.path.join(lora_path, 'stage2_32k')
        if not os.path.exists(stage2_dir):
            stage2_dir = os.path.join(lora_path, 'stage2_31k')
        stage2_name = os.path.basename(stage2_dir)
        
        z_image_transformer = PeftModel.from_pretrained(z_image_transformer, os.path.join(lora_path, 'stage1_6k', "z_image_turbo_adapter"), is_trainable=False)
        z_image_transformer.merge_and_unload()
        z_image_transformer = PeftModel.from_pretrained(z_image_transformer, os.path.join(stage2_dir, "z_image_turbo_adapter"), is_trainable=False)
        z_image_transformer.merge_and_unload()
        z_image_transformer = PeftModel.from_pretrained(z_image_transformer, os.path.join(lora_path, 'stage3_17k', "z_image_turbo_adapter"), is_trainable=False)
        z_image_transformer.merge_and_unload()
        

        
        self.guidance_scale = guidance_scale
        self.mid_timestep = mid_timestep
        self.weight_dtype = weight_dtype

        
        self.t_curr = (1000 - 121) / 1000

        print(f"Current One mid-timestep settings: ",self.t_curr)

        self.vae = vae
        self.z_image_transformer = z_image_transformer
        self.device = device
        # self._init_tiled_vae(encoder_tile_size=1024, decoder_tile_size=224)

    

    def _forward_no_tile(self, lq_latent, prompt_embeds):
        bsz, c, h, w = lq_latent.shape
        
        lq_latent_ = lq_latent.unsqueeze(2)
        lq_latent_list = list(lq_latent_.unbind(dim=0))
        timestep_input = torch.tensor([self.t_curr], device=lq_latent.device)
        
        # One-Step Predict.
        model_pred = self.z_image_transformer(
            lq_latent_list,timestep_input , prompt_embeds, return_dict=False
        )[0]

        noise_pred = torch.stack([t.float() for t in model_pred], dim=0)
        noise_pred = noise_pred.squeeze(2)
        noise_pred = - noise_pred

        # lq_latent = lq_latent - self.t_curr * noise_pred  
        lq_latent = lq_latent - (1 - self.t_curr) * noise_pred 
    
        lq_latent = (lq_latent / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        pred_img = self.vae.decode(lq_latent.to(self.vae.dtype), return_dict=False)[0]
        return pred_img

    

    def forward(self, lq_img, prompt_embeds):
        # torch.cuda.synchronize()
        # start_time = time.time()
        lq_latent = encode_images(
            lq_img.to(self.vae.dtype), self.vae, self.weight_dtype
        )
        _, _, h, w = lq_latent.shape
        
        pred_img = self._forward_no_tile(lq_latent, prompt_embeds)
        
        return pred_img
    
