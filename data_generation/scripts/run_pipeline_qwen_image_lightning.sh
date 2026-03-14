#!/bin/bash
# 闭环 Pipeline 运行脚本 - Qwen-Image-Lightning 版本

# 设置 PyTorch 显存分配策略，减少碎片化
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME="${HF_HOME:-/root/autodl-tmp/huggingface_cache}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/3.10/bin/python}"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export JUDGE_CONFIG_PATH="${JUDGE_CONFIG_PATH:-/root/ImageReward/data_generation/config/judge_config_api_gpt_ge.yaml}"

cd /root/ImageReward/data_generation

# 基础配置
BASE_SOURCE_PROMPTS="/root/autodl-tmp/AGIQA/data/prompt_sources_workspace/backfill_merge_runs/all_dimensions_v1_full/merged_working_pool.jsonl"
CLEANED_SOURCE_PROMPTS="/root/autodl-tmp/AGIQA/data/prompt_sources_workspace/backfill_merge_runs/all_dimensions_v1_full/merged_working_pool_cleaned_v1.jsonl"
SOURCE_PROMPTS="$BASE_SOURCE_PROMPTS"
if [ -f "$CLEANED_SOURCE_PROMPTS" ]; then
    SOURCE_PROMPTS="$CLEANED_SOURCE_PROMPTS"
fi
BASE_DIMENSION_SUBPOOL_INDEX="/root/autodl-tmp/AGIQA/data/prompt_sources_workspace/backfill_merge_runs/all_dimensions_v1_full/dimension_subpools/index.json"
BASE_CLEANED_DIMENSION_SUBPOOL_INDEX="/root/autodl-tmp/AGIQA/data/prompt_sources_workspace/backfill_merge_runs/all_dimensions_v1_full/dimension_subpools_cleaned_v1/index.json"
SEMANTIC_SCREENED_DIMENSION_SUBPOOL_INDEX_V1="/root/autodl-tmp/AGIQA/data/prompt_sources_workspace/backfill_merge_runs/all_dimensions_v1_full/semantic_screened_dimension_subpools_cleaned_v1/index.json"
SCREENED_CLEANED_DIMENSION_SUBPOOL_INDEX_V2="/root/autodl-tmp/AGIQA/data/prompt_sources_workspace/backfill_merge_runs/all_dimensions_v1_full/anatomy_screened_dimension_subpools_cleaned_v2/index.json"
SCREENED_CLEANED_DIMENSION_SUBPOOL_INDEX="/root/autodl-tmp/AGIQA/data/prompt_sources_workspace/backfill_merge_runs/all_dimensions_v1_full/anatomy_screened_dimension_subpools_cleaned_v1/index.json"
SCREENED_DIMENSION_SUBPOOL_INDEX="/root/autodl-tmp/AGIQA/data/prompt_sources_workspace/backfill_merge_runs/all_dimensions_v1_full/anatomy_screened_dimension_subpools/index.json"
DIMENSION_SUBPOOL_INDEX="$BASE_DIMENSION_SUBPOOL_INDEX"
if [ -f "$SEMANTIC_SCREENED_DIMENSION_SUBPOOL_INDEX_V1" ]; then
    DIMENSION_SUBPOOL_INDEX="$SEMANTIC_SCREENED_DIMENSION_SUBPOOL_INDEX_V1"
elif [ -f "$SCREENED_CLEANED_DIMENSION_SUBPOOL_INDEX_V2" ]; then
    DIMENSION_SUBPOOL_INDEX="$SCREENED_CLEANED_DIMENSION_SUBPOOL_INDEX_V2"
elif [ -f "$SCREENED_CLEANED_DIMENSION_SUBPOOL_INDEX" ]; then
    DIMENSION_SUBPOOL_INDEX="$SCREENED_CLEANED_DIMENSION_SUBPOOL_INDEX"
elif [ -f "$BASE_CLEANED_DIMENSION_SUBPOOL_INDEX" ]; then
    DIMENSION_SUBPOOL_INDEX="$BASE_CLEANED_DIMENSION_SUBPOOL_INDEX"
elif [ -f "$SCREENED_DIMENSION_SUBPOOL_INDEX" ]; then
    DIMENSION_SUBPOOL_INDEX="$SCREENED_DIMENSION_SUBPOOL_INDEX"
fi
OUTPUT_DIR="/root/autodl-tmp/Aesthetic_Quality_closed_loop/pipeline_output_qwen_image_lightning_$(date +%Y%m%d_%H%M%S)"
MODEL_PATH="/root/autodl-tmp/AGIQA/Qwen-Image/snapshots/75e0b4be04f60ec59a75f475837eced720f823b6"
NUNCHAKU_MODEL_PATH="${NUNCHAKU_MODEL_PATH:-/root/autodl-tmp/AGIQA/Nunchaku/svdq-int4_r32-qwen-image-lightningv1.0-4steps.safetensors}"

# 生成参数
NUM_PAIRS=3
MAX_PROMPTS=3
MAX_RETRIES=2
SEED=42
SHUFFLE=true
SYSTEMATIC=true

# 退化配置
SEVERITIES="moderate,severe"
SUBCATEGORY="aesthetic_quality"
#ATTRIBUTE="blur"

# 4090 默认优先使用 GPU；若显存不足可手动切回 fit-24g + offload
USE_CPU_OFFLOAD=false
RUNTIME_PROFILE="fast-gpu-24g"

echo "=========================================="
echo "Closed-Loop Pipeline - Qwen-Image-Lightning"
echo "=========================================="
echo "输出目录: $OUTPUT_DIR"
echo "Prompts 文件: $SOURCE_PROMPTS"
echo "模型: Qwen-Image-Lightning"
echo "Base 模型: $MODEL_PATH"
if [ "$RUNTIME_PROFILE" = "nunchaku-int4" ]; then
    echo "Nunchaku INT4: $NUNCHAKU_MODEL_PATH"
fi
echo "每 prompt 生成: $NUM_PAIRS 对"
echo "最大重试: $MAX_RETRIES 次"
echo "Shuffle: $SHUFFLE"
echo "Systematic: $SYSTEMATIC"
echo "Runtime Profile: $RUNTIME_PROFILE"
echo "CPU Offload: $USE_CPU_OFFLOAD"
echo ""

# 官方推荐快路径: steps=4, true_cfg_scale=1.0
CMD="$PYTHON_BIN scripts/pipeline.py \
    --source_prompts $SOURCE_PROMPTS \
    --output_dir $OUTPUT_DIR \
    --model_id qwen-image-lightning \
    --model_path $MODEL_PATH \
    --runtime_profile $RUNTIME_PROFILE \
    --num_pairs_per_prompt $NUM_PAIRS \
    --max_prompts $MAX_PROMPTS \
    --max_retries $MAX_RETRIES \
    --seed $SEED \
    --severities $SEVERITIES \
    --steps 4 \
    --cfg 1.0 \
    --model_filter sdxl \
    --dimension_subpool_index $DIMENSION_SUBPOOL_INDEX"

if [ "$RUNTIME_PROFILE" = "nunchaku-int4" ]; then
    CMD="$CMD --nunchaku_model_path $NUNCHAKU_MODEL_PATH"
fi

if [ "$SHUFFLE" = true ]; then
    CMD="$CMD --shuffle"
fi

if [ "$SYSTEMATIC" = true ]; then
    CMD="$CMD --systematic"
fi

if [ -n "$SUBCATEGORY" ]; then
    CMD="$CMD --subcategory_filter $SUBCATEGORY"
fi

if [ -n "$ATTRIBUTE" ]; then
    CMD="$CMD --attribute_filter $ATTRIBUTE"
fi

if [ "$USE_CPU_OFFLOAD" = true ]; then
    CMD="$CMD --use_cpu_offload"
fi

eval $CMD

echo ""
echo "=========================================="
echo "完成！结果保存在: $OUTPUT_DIR"
echo "  dataset.json    - 成功的训练对"
echo "  full_log.json   - 完整实验日志"
echo "  validation_report.json - 统计报告"
echo "=========================================="
