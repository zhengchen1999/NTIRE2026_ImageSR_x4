'''
 * SeeSR: Towards Semantics-Aware Real-World Image Super-Resolution 
 * Modified from diffusers by Rongyuan Wu
 * 24/12/2023
'''
import os
import sys
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
import glob
import argparse
import numpy as np
from PIL import Image

import torch
import torch.utils.checkpoint

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor

from pipelines.pipeline_seesr import StableDiffusionControlNetPipeline
from utils.wavelet_color_fix import wavelet_color_fix, adain_color_fix
from utils.degradation_features import compute_degradation_features, DegradationTokenAdapter

from ram.models.ram_lora import ram
from ram import inference_ram as inference

from typing import Mapping, Any
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

logger = get_logger(__name__, log_level="INFO")


tensor_transforms = transforms.Compose([
                transforms.ToTensor(),
            ])

ram_transforms = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
def load_state_dict_diffbirSwinIR(model: nn.Module, state_dict: Mapping[str, Any], strict: bool=False) -> None:
    state_dict = state_dict.get("state_dict", state_dict)
    
    is_model_key_starts_with_module = list(model.state_dict().keys())[0].startswith("module.")
    is_state_dict_key_starts_with_module = list(state_dict.keys())[0].startswith("module.")
    
    if (
        is_model_key_starts_with_module and
        (not is_state_dict_key_starts_with_module)
    ):
        state_dict = {f"module.{key}": value for key, value in state_dict.items()}
    if (
        (not is_model_key_starts_with_module) and
        is_state_dict_key_starts_with_module
    ):
        state_dict = {key[len("module."):]: value for key, value in state_dict.items()}
    
    model.load_state_dict(state_dict, strict=strict)


def load_seesr_pipeline(args, accelerator, enable_xformers_memory_efficient_attention):
    
    from models.controlnet import ControlNetModel
    from models.unet_2d_condition import UNet2DConditionModel

    # Load scheduler, tokenizer and models.
    
    scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_path, subfolder="scheduler")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_path, subfolder="text_encoder")
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_path, subfolder="tokenizer")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae")
    feature_extractor = CLIPImageProcessor.from_pretrained(f"{args.pretrained_model_path}/feature_extractor")
    unet = UNet2DConditionModel.from_pretrained(args.seesr_model_path, subfolder="unet")
    controlnet = ControlNetModel.from_pretrained(args.seesr_model_path, subfolder="controlnet")
    
    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    controlnet.requires_grad_(False)

    if enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
            controlnet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # Get the validation pipeline
    validation_pipeline = StableDiffusionControlNetPipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, feature_extractor=feature_extractor, 
        unet=unet, controlnet=controlnet, scheduler=scheduler, safety_checker=None, requires_safety_checker=False,
    )
    
    validation_pipeline._init_tiled_vae(encoder_tile_size=args.vae_encoder_tiled_size, decoder_tile_size=args.vae_decoder_tiled_size)

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move text_encode and vae to gpu and cast to weight_dtype
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    controlnet.to(accelerator.device, dtype=weight_dtype)

    return validation_pipeline

def load_tag_model(args, device='cuda'):
    pretrained_path = args.ram_pretrained_path or 'present/models/ram_swin_large_14m.pth'
    model = ram(pretrained=pretrained_path,
                pretrained_condition=args.ram_ft_path,
                image_size=384,
                vit='swin_l')
    model.eval()
    model.to(device)
    
    return model

def load_degradation_token_adapter(args, device='cuda', dtype=torch.float16):
    if not args.use_degradation_token:
        return None
    if not args.degradation_token_adapter_path:
        raise ValueError("--use_degradation_token requires --degradation_token_adapter_path.")

    adapter = DegradationTokenAdapter(
        in_dim=args.degradation_feat_dim,
        out_dim=args.degradation_token_dim,
        dropout=args.degradation_token_dropout,
        use_timestep_condition=args.use_dynamic_degradation_token,
        timestep_dim=args.degradation_token_timestep_dim,
    )
    state_dict = torch.load(args.degradation_token_adapter_path, map_location='cpu')
    adapter.load_state_dict(state_dict, strict=False)
    adapter.eval()
    adapter.to(device=device, dtype=dtype)
    return adapter


def get_validation_prompt(args, image, model, degradation_token_adapter=None, device='cuda'):
    validation_prompt = ""
 
    lq = tensor_transforms(image).unsqueeze(0).to(device)
    lq_for_ram = ram_transforms(lq)
    res = inference(lq_for_ram, model)
    ram_encoder_hidden_states = model.generate_image_embeds(lq_for_ram)
    degradation_features = None
    if degradation_token_adapter is not None:
        adapter_dtype = next(degradation_token_adapter.parameters()).dtype
        degradation_features = compute_degradation_features(lq).to(device=device, dtype=adapter_dtype)
        if args.use_dynamic_degradation_token:
            start_timestep = torch.full(
                (degradation_features.shape[0],),
                int(args.start_steps),
                device=device,
                dtype=torch.long,
            )
            degradation_token = degradation_token_adapter(
                degradation_features,
                timesteps=start_timestep,
            ).unsqueeze(1)
        else:
            degradation_token = degradation_token_adapter(degradation_features).unsqueeze(1)
        degradation_token = degradation_token.to(dtype=ram_encoder_hidden_states.dtype)
        if degradation_token.shape[-1] != ram_encoder_hidden_states.shape[-1]:
            raise ValueError(
                f"degradation token dim ({degradation_token.shape[-1]}) != RAM embed dim ({ram_encoder_hidden_states.shape[-1]}). "
                "Please set --degradation_token_dim to match training."
            )
        ram_encoder_hidden_states = torch.cat([ram_encoder_hidden_states, degradation_token], dim=1)

    validation_prompt = f"{res[0]}, {args.prompt},"

    return validation_prompt, ram_encoder_hidden_states, degradation_features

def main(args, enable_xformers_memory_efficient_attention=True,):
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
    )

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the output folder creation
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("SeeSR")

    pipeline = load_seesr_pipeline(args, accelerator, enable_xformers_memory_efficient_attention)
    model = load_tag_model(args, accelerator.device)
    degradation_token_adapter = load_degradation_token_adapter(
        args,
        device=accelerator.device,
        dtype=torch.float16 if args.mixed_precision == "fp16" else torch.float32,
    )
 
    if accelerator.is_main_process:
        generator = torch.Generator(device=accelerator.device)
        if args.seed is not None:
            generator.manual_seed(args.seed)

        if os.path.isdir(args.image_path):
            image_names = sorted(glob.glob(f'{args.image_path}/*.*'))
        else:
            image_names = [args.image_path]

        for image_idx, image_name in enumerate(image_names[:]):
            print(f'================== process {image_idx} imgs... ===================')
            validation_image = Image.open(image_name).convert("RGB")

            validation_prompt, ram_encoder_hidden_states, degradation_features = get_validation_prompt(
                args,
                validation_image,
                model,
                degradation_token_adapter=degradation_token_adapter,
                device=accelerator.device,
            )
            validation_prompt += args.added_prompt # clean, extremely detailed, best quality, sharp, clean
            negative_prompt = args.negative_prompt #dirty, messy, low quality, frames, deformed, 
            
            if args.save_prompts:
                name_for_prompt, _ = os.path.splitext(os.path.basename(image_name))
                txt_save_path = os.path.join(args.output_dir, f"{name_for_prompt}.txt")
                with open(txt_save_path, "w", encoding="utf-8") as file:
                    file.write(validation_prompt)
            print(f'{validation_prompt}')

            ori_width, ori_height = validation_image.size
            resize_flag = False
            rscale = args.upscale
            if ori_width < args.process_size//rscale or ori_height < args.process_size//rscale:
                scale = (args.process_size//rscale)/min(ori_width, ori_height)
                tmp_image = validation_image.resize((int(scale*ori_width), int(scale*ori_height)))

                validation_image = tmp_image
                resize_flag = True

            validation_image = validation_image.resize((validation_image.size[0]*rscale, validation_image.size[1]*rscale))
            validation_image = validation_image.resize((validation_image.size[0]//8*8, validation_image.size[1]//8*8))
            width, height = validation_image.size
            resize_flag = True #

            print(f'input size: {height}x{width}')

            for sample_idx in range(args.sample_times):  
                with torch.autocast("cuda"):
                    image = pipeline(
                            validation_prompt, validation_image, num_inference_steps=args.num_inference_steps, generator=generator, height=height, width=width,
                            guidance_scale=args.guidance_scale, negative_prompt=negative_prompt, conditioning_scale=args.conditioning_scale,
                            start_point=args.start_point, ram_encoder_hidden_states=ram_encoder_hidden_states,
                            latent_tiled_size=args.latent_tiled_size, latent_tiled_overlap=args.latent_tiled_overlap,
                            degradation_token_adapter=degradation_token_adapter,
                            degradation_features=degradation_features,
                            use_dynamic_degradation_token=args.use_dynamic_degradation_token,
                            args=args,
                        ).images[0]
                
                if args.align_method == 'nofix':
                    image = image
                else:
                    if args.align_method == 'wavelet':
                        image = wavelet_color_fix(image, validation_image)
                    elif args.align_method == 'adain':
                        image = adain_color_fix(image, validation_image)

                if resize_flag: 
                    image = image.resize((ori_width*rscale, ori_height*rscale))
                    
                name, ext = os.path.splitext(os.path.basename(image_name))
                if args.sample_times > 1:
                    out_name = f"{name}_s{sample_idx:02d}.png"
                else:
                    out_name = f"{name}.png"
                image.save(os.path.join(args.output_dir, out_name))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seesr_model_path", type=str, default=None)
    parser.add_argument("--ram_ft_path", type=str, default=None)
    parser.add_argument("--ram_pretrained_path", type=str, default=None)
    parser.add_argument("--pretrained_model_path", type=str, default=None)
    parser.add_argument("--prompt", type=str, default="") # user can add self-prompt to improve the results
    parser.add_argument("--added_prompt", type=str, default="clean, high-resolution, 8k")
    parser.add_argument("--negative_prompt", type=str, default="dotted, noise, blur, lowres, smooth")
    parser.add_argument("--image_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--mixed_precision", type=str, default="fp16") # no/fp16/bf16
    parser.add_argument("--guidance_scale", type=float, default=5.5)
    parser.add_argument("--conditioning_scale", type=float, default=1.0)
    parser.add_argument("--blending_alpha", type=float, default=1.0)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--process_size", type=int, default=512)
    parser.add_argument("--vae_decoder_tiled_size", type=int, default=224) # latent size, for 24G
    parser.add_argument("--vae_encoder_tiled_size", type=int, default=1024) # image size, for 13G
    parser.add_argument("--latent_tiled_size", type=int, default=96) 
    parser.add_argument("--latent_tiled_overlap", type=int, default=4) 
    parser.add_argument("--upscale", type=int, default=4)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--sample_times", type=int, default=1)
    parser.add_argument("--align_method", type=str, choices=['wavelet', 'adain', 'nofix'], default='adain')
    parser.add_argument("--start_steps", type=int, default=999) # defaults set to 999.
    parser.add_argument("--start_point", type=str, choices=['lr', 'noise'], default='lr') # LR Embedding Strategy, choose 'lr latent + 999 steps noise' as diffusion start point. 
    parser.add_argument("--save_prompts", action='store_true')
    parser.add_argument("--use_degradation_token", action='store_true')
    parser.add_argument("--degradation_token_adapter_path", type=str, default=None)
    parser.add_argument("--degradation_feat_dim", type=int, default=6)
    parser.add_argument("--degradation_token_dim", type=int, default=512)
    parser.add_argument("--degradation_token_dropout", type=float, default=0.1)
    parser.add_argument("--use_dynamic_degradation_token", action='store_true')
    parser.add_argument("--degradation_token_timestep_dim", type=int, default=128)
    args = parser.parse_args()
    main(args)



