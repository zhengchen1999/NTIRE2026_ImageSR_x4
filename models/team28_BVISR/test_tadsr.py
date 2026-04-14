import os
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.abspath(__file__))) # Add current directory to path
import glob
import argparse
import torch
from torchvision import transforms
import torchvision.transforms.functional as F
import numpy as np
from PIL import Image

from tadsr import TADSR_test
from my_utils.wavelet_color_fix import adain_color_fix, wavelet_color_fix

from ram.models.ram_lora import ram
from ram import inference_ram as inference

tensor_transforms = transforms.Compose([
                transforms.ToTensor(),
            ])

ram_transforms = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def get_validation_prompt(args, image, model, device='cuda'):
    validation_prompt = ""
    lq = tensor_transforms(image).unsqueeze(0).to(device)
    lq_ram = ram_transforms(lq).to(dtype=weight_dtype)
    captions = inference(lq_ram, model)
    validation_prompt = f"{captions[0]}, {args.prompt},"
    
    return validation_prompt, lq


class DummyArgs:
    pass

def main(model_dir, input_path, output_path, device):
    args = DummyArgs()
    args.input_image = input_path
    args.output_dir = output_path
    
    current_dir = os.path.dirname(os.path.realpath(__file__))
    print(current_dir)

    weights_root = 'model_zoo/team28_BVISR/weights'
    weights_tadsr = os.path.join(weights_root, 'TADSR')
    pretrained_dir = weights_tadsr if os.path.isdir(weights_tadsr) else weights_root

    args.pretrained_model_name_or_path = pretrained_dir
    args.ram_path = os.path.join(pretrained_dir, 'ram_swin_large_14m.pth')
    args.ram_ft_path = os.path.join(pretrained_dir, 'DAPE.pth')
    
    args.seed = 42
    args.process_size = 512
    args.upscale = 4
    args.align_method = 'adain'
    args.tadsr_path = model_dir
    args.prompt = ''
    args.save_prompts = False
    args.mixed_precision = 'fp16'
    args.merge_and_unload_lora = False
    args.vae_decoder_tiled_size = 224
    args.vae_encoder_tiled_size = 1024
    args.latent_tiled_size = 96
    args.latent_tiled_overlap = 32
    args.timesteps = 500

    global weight_dtype
    weight_dtype = torch.float16 if args.mixed_precision == "fp16" else torch.float32


    # initialize the model
    model = TADSR_test(args)

    image_names = sorted(glob.glob(f'{args.input_image}/*.*'))

    # get ram model
    DAPE = ram(pretrained=args.ram_path,
            pretrained_condition=args.ram_ft_path,
            image_size=384,
            vit='swin_l')
    DAPE.eval()
    DAPE.to(device)
    DAPE = DAPE.to(dtype=weight_dtype)
    
    os.makedirs(args.output_dir, exist_ok=True)
    print(f'There are {len(image_names)} images in {args.input_image}.')

    for i in range(len(image_names)):
        image_name = image_names[i]
        input_image = Image.open(image_name).convert('RGB')

        ori_width, ori_height = input_image.size
        rscale = args.upscale
        resize_flag = False
        if ori_width < args.process_size//rscale or ori_height < args.process_size//rscale:
            scale = (args.process_size//rscale)/min(ori_width, ori_height)
            input_image = input_image.resize((int(scale*ori_width), int(scale*ori_height)))
            resize_flag = True
        input_image = input_image.resize((input_image.size[0]*rscale, input_image.size[1]*rscale))

        new_width = input_image.width - input_image.width % 8
        new_height = input_image.height - input_image.height % 8
        input_image = input_image.resize((new_width, new_height), Image.LANCZOS)
        bname = os.path.basename(image_name)

        # get caption
        validation_prompt, lq = get_validation_prompt(args, input_image, DAPE, device=device)
        print(f"process {image_name}, tag: {validation_prompt}")

        # translate the image
        with torch.no_grad():
            lq = lq*2-1
            timesteps_tensor = torch.tensor([args.timesteps], device=device)
            output_image = model(lq, prompt=validation_prompt, timesteps=timesteps_tensor)
            output_pil = transforms.ToPILImage()(output_image[0].cpu() * 0.5 + 0.5)
            if args.align_method == 'adain':
                output_pil = adain_color_fix(target=output_pil, source=input_image)
            elif args.align_method == 'wavelet':
                output_pil = wavelet_color_fix(target=output_pil, source=input_image)
            
            target_width = ori_width * args.upscale
            target_height = ori_height * args.upscale
            if output_pil.size != (target_width, target_height):
                output_pil = output_pil.resize((target_width, target_height), Image.LANCZOS)

        output_pil.save(os.path.join(args.output_dir, bname))


if __name__ == "__main__":
    pass

