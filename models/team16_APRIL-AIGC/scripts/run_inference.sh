#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INPUT_DIR="${INPUT_DIR:-$ROOT_DIR/../../input}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/../../results}"

REFINER_CKPT="${REFINER_CKPT:-$ROOT_DIR/../../model_zoo/team16_APRIL-AIGC/checkpoints/refiner/best_lpips_model_only.pt}"

REFINER_TILE_SIZE="${REFINER_TILE_SIZE:-768}"
REFINER_TILE_OVERLAP="${REFINER_TILE_OVERLAP:-64}"

export PYTHONPATH="$ROOT_DIR"
export FASTGEN_OUTPUT_ROOT="$ROOT_DIR/runtime"
export HF_HOME="${HF_HOME:-$FASTGEN_OUTPUT_ROOT/.cache}"
export LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-true}"

mkdir -p "$ROOT_DIR/outputs" "$FASTGEN_OUTPUT_ROOT"

cd "$ROOT_DIR"

python scripts/inference/image_model_inferenceTI2I_benchmark_teacher_refiner_integrated.py \
  --config fastgen/configs/experiments/Flux2_klein/config_submit_sr.py \
  --do_teacher_sampling True \
  --do_student_sampling False \
  --num_steps 10 \
  --denoising_strength 1.0 \
  --guidance_scale 2.0 \
  --ti2i \
  --task sr \
  --input_upscale 4 \
  --input_image_dir "$INPUT_DIR" \
  --image_save_dir "$OUTPUT_DIR" \
  --refiner_ckpt "$REFINER_CKPT" \
  --refiner_tile_size "$REFINER_TILE_SIZE" \
  --refiner_tile_overlap "$REFINER_TILE_OVERLAP"
