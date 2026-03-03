#!/bin/bash
set -e

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DIMS="object_shape_error,overexposure"
PROMPTS="/root/ImageReward/data_generation/data/prompts_tagged_sdxl_v3.json"
SDXL_PATH="/root/autodl-tmp/huggingface_cache/hub/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b"
FLUX_PATH="/root/autodl-tmp/flux-schnell"
BASE_OUT="/root/autodl-tmp/diagnostic_v3"

cd /root/ImageReward/data_generation

echo "=== SDXL run starting at $(date) ==="
python scripts/pipeline.py \
  --source_prompts "$PROMPTS" \
  --output_dir "${BASE_OUT}/sdxl_${TIMESTAMP}" \
  --model_id sdxl \
  --model_path "$SDXL_PATH" \
  --model_filter sdxl \
  --attribute_filter "$DIMS" \
  --severities "moderate,severe" \
  --num_pairs_per_prompt 2 \
  --systematic \
  --max_retries 2 \
  --seed 42

echo "=== SDXL done at $(date) ==="

echo "=== Flux-schnell run starting at $(date) ==="
python scripts/pipeline.py \
  --source_prompts "$PROMPTS" \
  --output_dir "${BASE_OUT}/flux_${TIMESTAMP}" \
  --model_id flux-schnell \
  --model_path "$FLUX_PATH" \
  --model_filter sdxl \
  --attribute_filter "$DIMS" \
  --severities "moderate,severe" \
  --num_pairs_per_prompt 2 \
  --systematic \
  --max_retries 2 \
  --steps 4 \
  --cfg 0.0 \
  --seed 42

echo "=== Flux-schnell done at $(date) ==="
echo "=== All done at $(date) ==="

