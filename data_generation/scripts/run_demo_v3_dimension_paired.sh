#!/bin/bash
set -euo pipefail

# V3 prompt_templates_v3 维度 demo（正负图像对）
# - 从 image_quality_train.json 中筛选 model 含 "sdxl" 的正样本 prompt
# - 对每个维度的子属性（subcategory/attribute）各选 3 个正样本
# - 每个正样本对 mild/moderate/severe 各生成一次退化 prompt + 负样本图
# - 输出写入 /root/autodl-tmp（每个 subcategory/attribute 单独一个子目录，内含 images/ + dataset.json + prompts_cache.json）
# 注意：脚本会调用 LLM API 生成退化 prompt，需要网络；首次运行可能需要你授权网络/写入 /root/autodl-tmp
# --attribute_filter "color_cast"
# nohup python demo_v3_dimension_paired.py --subcategory_filter "technical_quality" --attribute_filter "blur" --severities "moderate,severe" --num_prompts_per_dimension 3  --output_dir /root/autodl-tmp/ --tagged_prompts /root/ImageReward/data_generation/data/prompts_tagged_sdxl_v2.json  &
# nohup python demo_v3_dimension_paired.py --subcategory_filter "aesthetic_quality" --attribute_filter "flat_lighting" --severities "moderate,severe" --num_prompts_per_dimension 3  --output_dir /root/autodl-tmp/ --tagged_prompts /root/ImageReward/data_generation/data/prompts_tagged_sdxl_v2.json  &
# nohup python demo_v3_dimension_paired.py --subcategory_filter "semantic_anatomy" --attribute_filter "impossible_pose" --severities "moderate,severe" --num_prompts_per_dimension 3  --output_dir /root/autodl-tmp/ --tagged_prompts /root/ImageReward/data_generation/data/prompts_tagged_sdxl_v2.json  &
# nohup python demo_v3_dimension_paired.py --subcategory_filter "semantic_object" --attribute_filter "attribute_mismatch" --severities "moderate,severe" --num_prompts_per_dimension 3 --output_dir /root/autodl-tmp/ --tagged_prompts /root/ImageReward/data_generation/data/prompts_tagged_sdxl_v2.json  &
# nohup python demo_v3_dimension_paired.py --subcategory_filter "semantic_spatial" --attribute_filter "missing_key_element" --severities "moderate,severe" --num_prompts_per_dimension 3 --output_dir /root/autodl-tmp/ --tagged_prompts /root/ImageReward/data_generation/data/prompts_tagged_sdxl_v2.json  &
# nohup python demo_v3_dimension_paired.py --subcategory_filter "semantic_text" --attribute_filter "text_error" --severities "moderate,severe" --num_prompts_per_dimension 3 --output_dir /root/autodl-tmp/ --tagged_prompts /root/ImageReward/data_generation/data/prompts_tagged_sdxl_v2.json --steps 60 --cfg 10 &
#nohup python demo_v3_dimension_paired.py --subcategory_filter "semantic_text" --attribute_filter "text_gibberish" --severities "moderate,severe" --num_prompts_per_dimension 3 --output_dir /root/autodl-tmp/ --tagged_prompts /root/ImageReward/data_generation/data/prompts_tagged_sdxl_v2.json  &
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/3.10/bin/python}"

POS_SOURCE="${POS_SOURCE:-/root/autodl-tmp/AGIQA/data/prompt_sources_workspace/backfill_merge_runs/all_dimensions_v1_full/merged_working_pool_cleaned_v1.jsonl}"
TEMPLATE_DIR="${TEMPLATE_DIR:-$PROJECT_ROOT/config/prompt_templates_v3}"
QUALITY_DIMENSIONS="${QUALITY_DIMENSIONS:-$PROJECT_ROOT/config/quality_dimensions.json}"
LLM_CONFIG="${LLM_CONFIG:-$PROJECT_ROOT/config/llm_config.yaml}"

OUTPUT_PREFIX="${OUTPUT_PREFIX:-/root/autodl-tmp/demo_v3_dimension_paired}"
MODEL_PATH="${MODEL_PATH:-/root/ckpts/sd_xl_base_1.0.safetensors}"

NUM_PROMPTS_PER_DIMENSION="${NUM_PROMPTS_PER_DIMENSION:-3}"
SEED="${SEED:-42}"
MODEL_FILTER="${MODEL_FILTER:-sdxl}"
PROMPT_SAMPLING="${PROMPT_SAMPLING:-random}" # random | topk_random
SEVERITIES="${SEVERITIES:-moderate,severe}"
SUBCATEGORY_FILTER="${SUBCATEGORY_FILTER:-}"
ATTRIBUTE_FILTER="${ATTRIBUTE_FILTER:-}"

STEPS="${STEPS:-35}"
CFG="${CFG:-7.5}"
WIDTH="${WIDTH:-1024}"
HEIGHT="${HEIGHT:-1024}"

CMD_ARGS=(
  --positive_source "$POS_SOURCE"
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
  --model_path "$MODEL_PATH"
)

# 可选参数
[[ -n "$SUBCATEGORY_FILTER" ]] && CMD_ARGS+=(--subcategory_filter "$SUBCATEGORY_FILTER")
[[ -n "$ATTRIBUTE_FILTER" ]] && CMD_ARGS+=(--attribute_filter "$ATTRIBUTE_FILTER")

"$PYTHON_BIN" "$SCRIPT_DIR/demo_v3_dimension_paired.py" "${CMD_ARGS[@]}"
