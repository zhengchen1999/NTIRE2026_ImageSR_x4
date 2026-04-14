#!/bin/bash

# === 1. 定义 LoRA Modules (保持与你原脚本一致) ===
LORA_MODULES_LIST=(to_k to_q to_v to_out.0 conv conv1 conv2 conv_shortcut conv_out proj_in proj_out ff.net.2 ff.net.0.proj)
IFS=','
LORA_MODULES="${LORA_MODULES_LIST[*]}"
unset IFS

# === 2. 定义路径 ===
# Base Model 路径
BASE_MODEL="/data/pretrained/stable-diffusion-2-1-base/"

# 输入 LR 图像路径
LQ_DIR="/home/NTIRE26/data2/NTIRE2026/SR/dataset/DIV2K/Test/LR/X4"

# 输出路径 (建议区分 alpha 值)
OUTPUT_DIR="results_ensemble_alpha0.9_JPEG_75"

# === 3. 模型权重路径 ===
# [Model A] 感知强 (Perception) -> 请在此处填入模型 A 的权重路径
WEIGHT_A="/home/NTIRE26/data2/NTIRE2026/SR/Team13/NTIRE2026_SRx4_TranssionAI/model_zoo/A模型复杂退化的权重-checkpoint-77000/checkpoint-77000/ema_state_dict.pth" 

# [Model B] 结构好 (Fidelity/PSNR) -> 这里是你刚才提供的模型 B 路径
WEIGHT_B="/home/NTIRE26/data2/NTIRE2026/SR/Team13/NTIRE2026_SRx4_TranssionAI/model_zoo/B模型简单退化的权重-checkpoint-70000/checkpoint-70000/ema_state_dict.pth"


python test.py \
    --base_model_path "$BASE_MODEL" \
    --lq_dir "$LQ_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --device cuda \
    --seed 231 \
    --alpha 0.9 \
    \
    --weight_path_a "$WEIGHT_A" \
    --lora_rank_a 256 \
    --lora_modules_a "$LORA_MODULES" \
    \
    --weight_path_b "$WEIGHT_B" \
    --lora_rank_b 256 \
    --lora_modules_b "$LORA_MODULES" \
    \
    --model_t 200 \
    --coeff_t 200 \
    --patch_size 512 \
    --stride 256 \
    --scale_by factor \
    --upscale 4 \
    --base_model_type sd2