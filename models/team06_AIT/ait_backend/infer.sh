#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# ----------------------------
# SeeSR inference configuration
# ----------------------------
GPU="1"
SEESR_ROOT="/home/jiyang/jiyang/Projects/SeeSR"
PRETRAINED_MODEL_PATH="${SEESR_ROOT}/preset/models/stable-diffusion-2-base"
SEESR_MODEL_PATH="/home/jiyang/jiyang/Projects/SeeSR/experience/seesr_noise_deg_token_dynamic/checkpoint-30000"
# Keep empty to avoid using DAPE or any RAM finetuned checkpoint.
RAM_FT_PATH=""

IMAGE_PATH="/home/jiyang/jiyang/Projects/SeeSR/preset/datasets/contest/test_data_contest"
OUTPUT_DIR="/home/jiyang/jiyang/Projects/inference_only/output/contest_seesr_noise_deg_token_dynamic_30000_guidance_4"

PROMPT=""
ADDED_PROMPT="clean, high-resolution, 8k"
NEGATIVE_PROMPT="dotted, noise, blur, lowres, smooth"

START_POINT="lr"
NUM_INFERENCE_STEPS=50
GUIDANCE_SCALE=4
CONDITIONING_SCALE=1.0
PROCESS_SIZE=512
UPSCALE=4
MIXED_PRECISION="fp16"
ALIGN_METHOD="adain"

# Tiled inference (OOM fallback will auto reduce these)
VAE_DECODER_TILED_SIZE=128
VAE_ENCODER_TILED_SIZE=1024
LATENT_TILED_SIZE=64
LATENT_TILED_OVERLAP=4

# Mitigate CUDA memory fragmentation (keep compatible with older PyTorch)
PYTORCH_CUDA_ALLOC_CONF_VALUE="max_split_size_mb:128"

# Degradation-aware token (must match training)
USE_DEGRADATION_TOKEN=true
DEGRADATION_ADAPTER_PATH="${SEESR_MODEL_PATH}/degradation_token_adapter.bin"
DEGRADATION_FEAT_DIM=6
DEGRADATION_TOKEN_DIM=512
DEGRADATION_TOKEN_DROPOUT=0.1
USE_DYNAMIC_DEGRADATION_TOKEN=true
DEGRADATION_TOKEN_TIMESTEP_DIM=128

BASE_CMD=(
  python "${SCRIPT_DIR}/test_seesr.py"
  --pretrained_model_path "${PRETRAINED_MODEL_PATH}"
  --seesr_model_path "${SEESR_MODEL_PATH}"
  --image_path "${IMAGE_PATH}"
  --output_dir "${OUTPUT_DIR}"
  --prompt "${PROMPT}"
  --added_prompt "${ADDED_PROMPT}"
  --negative_prompt "${NEGATIVE_PROMPT}"
  --start_point "${START_POINT}"
  --num_inference_steps "${NUM_INFERENCE_STEPS}"
  --guidance_scale "${GUIDANCE_SCALE}"
  --conditioning_scale "${CONDITIONING_SCALE}"
  --process_size "${PROCESS_SIZE}"
  --upscale "${UPSCALE}"
  --mixed_precision "${MIXED_PRECISION}"
  --align_method "${ALIGN_METHOD}"
  --degradation_feat_dim "${DEGRADATION_FEAT_DIM}"
  --degradation_token_dim "${DEGRADATION_TOKEN_DIM}"
  --degradation_token_dropout "${DEGRADATION_TOKEN_DROPOUT}"
  --degradation_token_timestep_dim "${DEGRADATION_TOKEN_TIMESTEP_DIM}"
)

if [ "${USE_DEGRADATION_TOKEN}" = "true" ]; then
  BASE_CMD+=(
    --use_degradation_token
    --degradation_token_adapter_path "${DEGRADATION_ADAPTER_PATH}"
  )
fi

if [ "${USE_DYNAMIC_DEGRADATION_TOKEN}" = "true" ]; then
  BASE_CMD+=(--use_dynamic_degradation_token)
fi

if [ -n "${RAM_FT_PATH}" ]; then
  BASE_CMD+=(--ram_ft_path "${RAM_FT_PATH}")
fi

run_with_tiles() {
  local vae_dec="$1"
  local vae_enc="$2"
  local latent_tile="$3"
  local latent_overlap="$4"
  local log_file="$5"

  local cmd=("${BASE_CMD[@]}")
  cmd+=(--vae_decoder_tiled_size "${vae_dec}")
  cmd+=(--vae_encoder_tiled_size "${vae_enc}")
  cmd+=(--latent_tiled_size "${latent_tile}")
  cmd+=(--latent_tiled_overlap "${latent_overlap}")

  echo "Try tiles: vae_dec=${vae_dec}, vae_enc=${vae_enc}, latent_tile=${latent_tile}, overlap=${latent_overlap}"

  set +e
  PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF_VALUE}" CUDA_VISIBLE_DEVICES="${GPU}" "${cmd[@]}" 2>&1 | tee "${log_file}"
  local code=${PIPESTATUS[0]}
  set -e
  return ${code}
}

OOM_LOG="$(mktemp /tmp/seesr_infer_oom_XXXX.log)"

# Progressive fallback profiles: quality -> conservative memory
if run_with_tiles 128 1024 64 4 "${OOM_LOG}"; then
  echo "Inference finished with profile #1."
  exit 0
fi
if grep -Eqi "out of memory|cublas_status_alloc_failed|cuda error: an illegal memory access was encountered" "${OOM_LOG}"; then
  echo "OOM detected. Retrying with smaller tiles..."
else
  echo "Inference failed (non-OOM). See log: ${OOM_LOG}"
  exit 1
fi

if run_with_tiles 96 768 48 8 "${OOM_LOG}"; then
  echo "Inference finished with profile #2."
  exit 0
fi
if grep -Eqi "out of memory|cublas_status_alloc_failed|cuda error: an illegal memory access was encountered" "${OOM_LOG}"; then
  echo "OOM detected again. Retrying with smaller tiles..."
else
  echo "Inference failed (non-OOM). See log: ${OOM_LOG}"
  exit 1
fi

if run_with_tiles 64 512 32 8 "${OOM_LOG}"; then
  echo "Inference finished with profile #3."
  exit 0
fi
if grep -Eqi "out of memory|cublas_status_alloc_failed|cuda error: an illegal memory access was encountered" "${OOM_LOG}"; then
  echo "OOM detected again. Retrying with smallest tiles..."
else
  echo "Inference failed (non-OOM). See log: ${OOM_LOG}"
  exit 1
fi

if run_with_tiles 48 384 24 8 "${OOM_LOG}"; then
  echo "Inference finished with profile #4 (most conservative)."
  exit 0
fi

echo "Inference still failed after all OOM fallbacks. See log: ${OOM_LOG}"
exit 1