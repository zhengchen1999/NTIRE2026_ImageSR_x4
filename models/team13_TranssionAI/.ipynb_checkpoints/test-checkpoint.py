import argparse
import os
from pathlib import Path
from time import time

from accelerate.utils import set_seed
from PIL import Image
from torchvision import transforms
from torch.nn import functional as F
import torch
from HYPIR.enhancer.sd2 import SD2Enhancer
from HYPIR.utils.captioner import EmptyCaptioner, FixedCaptioner

def tensor2image(img_tensor):
        return (
            (img_tensor * 255.0)
            .clamp(0, 255)
            .to(torch.uint8)
            .permute(0, 2, 3, 1)
            .contiguous()
            .cpu()
            .numpy()
        )

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_type", type=str, required=True, choices=["sd2"],
                        help="Type of the base model. Currently only 'sd2' is supported.")
    parser.add_argument("--base_model_path", type=str, required=True,
                        help="Path to the base model directory.")
    parser.add_argument("--model_t", type=int, required=True,
                        help="Model input timestep.")
    parser.add_argument("--coeff_t", type=int, required=True,
                        help="Timestep used to calculate the conversion coefficients from noise to data.")
    parser.add_argument("--lora_rank", type=int, required=True,
                        help="Rank of the LoRA modules.")
    parser.add_argument("--lora_modules", type=str, required=True,
                        help="Comma-separated list of LoRA module names.")
    parser.add_argument("--weight_path", type=str, required=True,
                        help="Path to the LoRA weight file.")
    parser.add_argument("--patch_size", type=int, default=512,
                        help="Size of the patches to process.")
    parser.add_argument("--stride", type=int, default=256,
                        help="Stride for the patches.")
    parser.add_argument("--lq_dir", type=str, required=True,
                        help="Directory containing low-quality images. Support nested directories.")
    parser.add_argument("--scale_by", type=str, default="factor", choices=["factor", "longest_side"],
                        help=(
                            "Method to scale the input images. "
                            "'factor' scales by a fixed factor, 'longest_side' scales by the longest side (to a fixed size)."
                        ))
    parser.add_argument("--upscale", type=int, default=4,
                        help="Upscaling factor.")
    parser.add_argument("--target_longest_side", type=int, default=None,
                        help="Target longest side for scaling if 'scale_by' is set to 'longest_side'.")
    parser.add_argument("--txt_dir", type=str, default=None,
                        help=(
                            "Directory containing text prompts for images. "
                            "The structure of the directory should match the structure of 'lq_dir'. "
                            "e.g. if image path is 'lq_dir/a/b/c.png', then the prompt should be in 'txt_dir/a/b/c.txt'. "
                            "If txt_dir is None, will use captioner."
                        ))
    parser.add_argument("--captioner", type=str, choices=["empty", "fixed"], default="empty",
                        help="Captioner to use. 'empty' for no captions, 'fixed' for a fixed caption.")
    parser.add_argument("--fixed_caption", type=str, default=None,
                        help="Fixed caption to use if 'captioner' is set to 'fixed'.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save the results.")
    parser.add_argument("--seed", type=int, default=231,
                        help="Random seed.")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run the model on (e.g., 'cuda', 'cpu').")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)

    if args.base_model_type == "sd2":
        model = SD2Enhancer(
            base_model_path=args.base_model_path,
            weight_path=args.weight_path,
            lora_modules=args.lora_modules.split(","),
            lora_rank=args.lora_rank,
            model_t=args.model_t,
            coeff_t=args.coeff_t,
            device=args.device,
        )
        print("Start loading models")
        load_start = time()
        model.init_models()
        print(f"Models loaded in {time() - load_start:.2f} seconds.")
    else:
        raise ValueError(f"Unsupported model type: {args.base_model_type}")

    input_dir = Path(args.lq_dir)
    output_dir = Path(args.output_dir)

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    images = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in image_extensions:
                full_path = Path(root) / file
                images.append(full_path)
    images.sort(key=lambda x: str(x.relative_to(input_dir)))
    print(f"Found {len(images)} images in {input_dir}.")

    if args.txt_dir is None:
        if args.captioner is None:
            raise ValueError("Either 'txt_dir' or 'captioner' must be specified.")
        elif args.captioner == "empty":
            captioner = EmptyCaptioner(args.device)
        elif args.captioner == "fixed":
            if args.fixed_caption is None:
                raise ValueError("Fixed caption must be provided when 'captioner' is set to 'fixed'.")
            captioner = FixedCaptioner(args.device, args.fixed_caption)

    to_tensor = transforms.ToTensor()

    result_dir = output_dir / "result"
    prompt_dir = output_dir / "prompt"
    for file_path in images:
        t1=time()
        print(f"Process file: \033[92m{os.path.basename(file_path)}\033[0m")

        relative_path = file_path.relative_to(input_dir)
        result_path = result_dir / relative_path.with_suffix(".jpg")
        result_path.parent.mkdir(parents=True, exist_ok=True)
        prompt_path = prompt_dir / relative_path.with_suffix(".txt")
        prompt_path.parent.mkdir(parents=True, exist_ok=True)

        lq_pil = Image.open(file_path).convert("RGB")
        lq_tensor = to_tensor(lq_pil).unsqueeze(0)
        
        # simple bicubic upscale
        '''
        lq_tensor = F.interpolate(lq_tensor, scale_factor=4, mode="bicubic")
        result = [Image.fromarray(img) for img in tensor2image(lq_tensor)][0]
        result.save(result_path)
        '''
        if args.txt_dir is not None:
            with open(args.txt_dir / relative_path.with_suffix(".txt"), "r") as fp:
                prompt = fp.read().strip()
        else:
            prompt = captioner(lq_pil)
        with open(prompt_path, "w") as fp:
            fp.write(prompt)
        print(f"Prompt: \033[94m{prompt}\033[0m")

        result = model.enhance(
            lq=lq_tensor,
            prompt=prompt,
            scale_by=args.scale_by,
            upscale=1,
            target_longest_side=args.target_longest_side,
            patch_size=args.patch_size,
            stride=args.stride,
            return_type="pil",
        )[0]
        result.save('tmp1.png')
        
        lq_pil2 = Image.open('tmp1.png').convert("RGB")
        lq_tensor2 = to_tensor(lq_pil2).unsqueeze(0)        
        result2 = model.enhance(
            lq=lq_tensor2,
            prompt=prompt,
            scale_by=args.scale_by,
            upscale=4,
            target_longest_side=args.target_longest_side,
            patch_size=1280,
            stride=640,
            return_type="pil",
        )[0]
        result2.save(result_path)
        print(f"inference an image in {time() - t1:.3f} seconds.")
         
    print(f"Done. \033[92mEnjoy your results in {result_dir}.\033[0m")
