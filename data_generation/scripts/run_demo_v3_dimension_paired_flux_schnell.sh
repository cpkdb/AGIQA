#!/bin/bash
set -euo pipefail

# 设置 PyTorch 显存分配策略，减少碎片化
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# V3 prompt_templates_v3 维度 demo（正负图像对）- Flux.1-schnell 版本
# - 从 image_quality_train.json 中筛选 model 含 "sdxl" 的正样本 prompt
# - 对每个维度的子属性（subcategory/attribute）各选 3 个正样本
# - 每个正样本对 mild/moderate/severe 各生成一次退化 prompt + 负样本图
# - 输出写入 /root/autodl-tmp（每个 subcategory/attribute 单独一个子目录，内含 images/ + dataset.json + prompts_cache.json）
# 注意：脚本会调用 LLM API 生成退化 prompt，需要网络

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/3.10/bin/python}"

POS_SOURCE="${POS_SOURCE:-/root/autodl-tmp/AGIQA/data/prompt_sources_workspace/backfill_merge_runs/all_dimensions_v1_full/merged_working_pool_cleaned_v1.jsonl}"
TAGGED_PROMPTS="${TAGGED_PROMPTS:-/root/autodl-tmp/AGIQA/data/prompt_sources_workspace/backfill_merge_runs/all_dimensions_v1_full/merged_working_pool_cleaned_v1.jsonl}"
TEMPLATE_DIR="${TEMPLATE_DIR:-$PROJECT_ROOT/config/prompt_templates_v3}"
QUALITY_DIMENSIONS="${QUALITY_DIMENSIONS:-$PROJECT_ROOT/config/quality_dimensions.json}"
LLM_CONFIG="${LLM_CONFIG:-$PROJECT_ROOT/config/llm_config.yaml}"

OUTPUT_PREFIX="${OUTPUT_PREFIX:-/root/autodl-tmp/demo_v3_dimension_paired_flux_schnell}"

NUM_PROMPTS_PER_DIMENSION="${NUM_PROMPTS_PER_DIMENSION:-2}"
SEED="${SEED:-42}"
MODEL_FILTER="${MODEL_FILTER:-sdxl}"
PROMPT_SAMPLING="${PROMPT_SAMPLING:-random}" # random | topk_random
SEVERITIES="${SEVERITIES:-moderate,severe}"
SUBCATEGORY_FILTER="${SUBCATEGORY_FILTER:-semantic_anatomy}"
ATTRIBUTE_FILTER="${ATTRIBUTE_FILTER:-body_proportion_error}"

# Flux.1-schnell 推荐参数（4步快速生成，guidance_scale=0.0）
STEPS="${STEPS:-4}"
CFG="${CFG:-0.0}"
WIDTH="${WIDTH:-1024}"
HEIGHT="${HEIGHT:-1024}"
OPTIMIZE="${OPTIMIZE:-true}"  # 启用优化模式（T5 4-bit + FP8 + compile）

CMD_ARGS=(
  --positive_source "$POS_SOURCE"
  --tagged_prompts "$TAGGED_PROMPTS"
  --template_dir "$TEMPLATE_DIR"
  --quality_dimensions "$QUALITY_DIMENSIONS"
  --llm_config "$LLM_CONFIG"
  --output_dir "$OUTPUT_PREFIX"
  --num_prompts_per_dimension "$NUM_PROMPTS_PER_DIMENSION"
  --seed "$SEED"
  --prompt_sampling "$PROMPT_SAMPLING"
  --model_filter "$MODEL_FILTER"
  --severities "$SEVERITIES"
  --steps "$STEPS"
  --cfg "$CFG"
  --width "$WIDTH"
  --height "$HEIGHT"
  --model_id flux-schnell
)

# 可选参数
[[ -n "$SUBCATEGORY_FILTER" ]] && CMD_ARGS+=(--subcategory_filter "$SUBCATEGORY_FILTER")
[[ -n "$ATTRIBUTE_FILTER" ]] && CMD_ARGS+=(--attribute_filter "$ATTRIBUTE_FILTER")
[[ "$OPTIMIZE" == "true" ]] && CMD_ARGS+=(--optimize)

echo "=========================================="
echo "V3 Dimension Paired Demo - Flux.1-schnell"
echo "=========================================="
echo "模型: Flux.1-schnell (4步快速生成)"
echo "Steps: $STEPS, CFG: $CFG"
echo "优化模式: $OPTIMIZE (T5 4-bit + FP8 + compile)"
echo "输出目录: $OUTPUT_PREFIX"
echo "=========================================="

"$PYTHON_BIN" "$SCRIPT_DIR/demo_v3_dimension_paired.py" "${CMD_ARGS[@]}"
