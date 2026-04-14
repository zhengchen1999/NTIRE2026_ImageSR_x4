import argparse
import os
import io
from pathlib import Path
from time import time

from accelerate.utils import set_seed
from PIL import Image
from torchvision import transforms
import torch
import numpy as np

# 确保 HYPIR 包在你的 Python 路径中
from HYPIR.enhancer.sd2 import SD2Enhancer
from HYPIR.utils.captioner import EmptyCaptioner, FixedCaptioner

def parse_args():
    parser = argparse.ArgumentParser(description="NITRE 2026 Ensemble Inference (H100 Optimized)")
    
    # === 基础配置 ===
    parser.add_argument("--base_model_type", type=str, default="sd2", choices=["sd2"])
    parser.add_argument("--base_model_path", type=str, required=True, help="Base SD2 path")
    parser.add_argument("--lq_dir", type=str, required=True, help="Input LR images directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Final output directory")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=231)
    
    # === 融合参数 ===
    parser.add_argument("--alpha", type=float, default=0.6, 
                        help="Weight for Model A. 0.6 means 60% Model A + 40% Model B.")

    # === Model A 配置 (Perception / Texture) ===
    parser.add_argument("--weight_path_a", type=str, required=True, help="Weights for Model A")
    parser.add_argument("--lora_rank_a", type=int, required=True)
    parser.add_argument("--lora_modules_a", type=str, required=True)
    
    # === Model B 配置 (Fidelity / Structure) ===
    parser.add_argument("--weight_path_b", type=str, required=True, help="Weights for Model B")
    parser.add_argument("--lora_rank_b", type=int, required=True)
    parser.add_argument("--lora_modules_b", type=str, required=True)

    # === 推理参数 (假设 A/B 参数一致，若不一致请拆分) ===
    parser.add_argument("--model_t", type=int, required=True, help="Timestep t")
    parser.add_argument("--coeff_t", type=int, required=True, help="Coefficient timestep")
    parser.add_argument("--patch_size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--scale_by", type=str, default="factor", choices=["factor", "longest_side"])
    parser.add_argument("--upscale", type=int, default=4)
    parser.add_argument("--target_longest_side", type=int, default=None)
    
    # === Prompt 配置 ===
    parser.add_argument("--txt_dir", type=str, default=None)
    parser.add_argument("--captioner", type=str, choices=["empty", "fixed"], default="empty")
    parser.add_argument("--fixed_caption", type=str, default=None)

    args = parser.parse_args()
    return args

def apply_jpeg_trick(pil_image):
    """
    内存中转 JPEG -> PNG Trick
    去掉 quality 参数，使用 PIL 默认的 JPEG 压缩系数 (75)
    与单模型原生测试保持 100% 一致
    """
    buffer = io.BytesIO()
    # 保存为 JPEG (使用默认有损压缩)
    pil_image.save(buffer, format="JPEG")
    buffer.seek(0)
    # 重新读取 (带上了 JPEG 的平滑效果)
    return Image.open(buffer).convert("RGB")

def blend_images(img_a, img_b, alpha):
    """
    加权融合: alpha * A + (1 - alpha) * B
    """
    arr_a = np.array(img_a).astype(np.float32)
    arr_b = np.array(img_b).astype(np.float32)
    
    # 融合计算
    blended = arr_a * alpha + arr_b * (1.0 - alpha)
    
    # 截断并转回 uint8
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    return Image.fromarray(blended)

if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    
    input_dir = Path(args.lq_dir)
    output_dir = Path(args.output_dir)
    result_dir = output_dir / "result"
    result_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # 1. 准备数据
    # =========================================================================
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

    # =========================================================================
    # 2. 加载模型 (H100 80G 显存足够同时加载两个)
    # =========================================================================
    print(f"\n\033[96m=== Loading Models on {args.device} ===\033[0m")
    
    print("Loading Model A (Perception)...")
    model_a = SD2Enhancer(
        base_model_path=args.base_model_path,
        weight_path=args.weight_path_a,
        lora_modules=args.lora_modules_a.split(","),
        lora_rank=args.lora_rank_a,
        model_t=args.model_t,
        coeff_t=args.coeff_t,
        device=args.device,
    )
    model_a.init_models()

    print("Loading Model B (Fidelity)...")
    model_b = SD2Enhancer(
        base_model_path=args.base_model_path,
        weight_path=args.weight_path_b,
        lora_modules=args.lora_modules_b.split(","),
        lora_rank=args.lora_rank_b,
        model_t=args.model_t,
        coeff_t=args.coeff_t,
        device=args.device,
    )
    model_b.init_models()
    print("\033[92mModels Loaded Successfully.\033[0m")

    # =========================================================================
    # 3. 准备 Captioner
    # =========================================================================
    if args.txt_dir is None:
        if args.captioner == "empty":
            captioner = EmptyCaptioner(args.device)
        elif args.captioner == "fixed":
            captioner = FixedCaptioner(args.device, args.fixed_caption)

    to_tensor = transforms.ToTensor()

    print(f"\nConfiguration: Alpha={args.alpha}, Upscale={args.upscale}, Patch={args.patch_size}")

    # =========================================================================
    # 4. 推理循环
    # =========================================================================
    total_time = 0
    for i, file_path in enumerate(images):
        t1 = time()
        print(f"[{i+1}/{len(images)}] Process: \033[92m{os.path.basename(file_path)}\033[0m")

        relative_path = file_path.relative_to(input_dir)
        # 强制输出为 PNG
        save_path = result_dir / relative_path.with_suffix(".png")
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # 读取输入
        lq_pil = Image.open(file_path).convert("RGB")
        lq_tensor = to_tensor(lq_pil).unsqueeze(0)
        
        # 获取 Prompt
        if args.txt_dir is not None:
            txt_p = args.txt_dir / relative_path.with_suffix(".txt")
            if txt_p.exists():
                with open(txt_p, "r") as fp: prompt = fp.read().strip()
            else:
                prompt = ""
        else:
            prompt = captioner(lq_pil)

        # --- Model A 推理 ---
        res_a = model_a.enhance(
            lq=lq_tensor, prompt=prompt, scale_by=args.scale_by,
            upscale=args.upscale, target_longest_side=args.target_longest_side,
            patch_size=args.patch_size, stride=args.stride, return_type="pil"
        )[0]
        # Trick A (使用默认质量)
        res_a = apply_jpeg_trick(res_a)

        # --- Model B 推理 ---
        res_b = model_b.enhance(
            lq=lq_tensor, prompt=prompt, scale_by=args.scale_by,
            upscale=args.upscale, target_longest_side=args.target_longest_side,
            patch_size=args.patch_size, stride=args.stride, return_type="pil"
        )[0]
        # Trick B (使用默认质量)
        res_b = apply_jpeg_trick(res_b)

        # --- 融合 ---
        final_img = blend_images(res_a, res_b, args.alpha)
        
        # 保存 (PNG)
        final_img.save(save_path)
        
        cost = time() - t1
        total_time += cost
        print(f"Saved to {save_path} | Time: {cost:.3f}s")

    print(f"\nDone. Average time per image: {total_time/len(images):.3f}s")
    print(f"\033[92mAll results saved in {result_dir}\033[0m")