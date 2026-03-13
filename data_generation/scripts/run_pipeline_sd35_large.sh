#!/bin/bash
# 闭环 Pipeline 运行脚本 - SD3.5 Large Turbo 版本

# 设置 PyTorch 显存分配策略，减少碎片化
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export JUDGE_CONFIG_PATH="${JUDGE_CONFIG_PATH:-/root/ImageReward/data_generation/config/judge_config_api_gpt_ge.yaml}"

cd /root/ImageReward/data_generation

# 基础配置
BASE_SOURCE_PROMPTS="/root/autodl-tmp/AGIQA/data/prompt_sources_workspace/backfill_merge_runs/all_dimensions_v1_full/merged_working_pool_cleaned_v1.jsonl"
SD35_TURBO_SOURCE_PROMPTS="/root/autodl-tmp/AGIQA/data/prompt_sources_workspace/backfill_merge_runs/all_dimensions_v1_full/merged_working_pool_sd35_turbo_clipsafe_v1.jsonl"
SOURCE_PROMPTS="$BASE_SOURCE_PROMPTS"
if [ -f "$SD35_TURBO_SOURCE_PROMPTS" ]; then
    SOURCE_PROMPTS="$SD35_TURBO_SOURCE_PROMPTS"
fi
BASE_DIMENSION_SUBPOOL_INDEX="/root/autodl-tmp/AGIQA/data/prompt_sources_workspace/backfill_merge_runs/all_dimensions_v1_full/anatomy_screened_dimension_subpools_cleaned_v2/index.json"
BASE_DIMENSION_SUBPOOL_INDEX_V1="/root/autodl-tmp/AGIQA/data/prompt_sources_workspace/backfill_merge_runs/all_dimensions_v1_full/anatomy_screened_dimension_subpools_cleaned_v1/index.json"
SD35_TURBO_DIMENSION_SUBPOOL_INDEX_V2="/root/autodl-tmp/AGIQA/data/prompt_sources_workspace/backfill_merge_runs/all_dimensions_v1_full/sd35_turbo_dimension_subpools_clipsafe_v2/index.json"
SD35_TURBO_DIMENSION_SUBPOOL_INDEX="/root/autodl-tmp/AGIQA/data/prompt_sources_workspace/backfill_merge_runs/all_dimensions_v1_full/sd35_turbo_dimension_subpools_clipsafe_v1/index.json"
DIMENSION_SUBPOOL_INDEX="$BASE_DIMENSION_SUBPOOL_INDEX"
if [ ! -f "$DIMENSION_SUBPOOL_INDEX" ] && [ -f "$BASE_DIMENSION_SUBPOOL_INDEX_V1" ]; then
    DIMENSION_SUBPOOL_INDEX="$BASE_DIMENSION_SUBPOOL_INDEX_V1"
fi
if [ -f "$SD35_TURBO_DIMENSION_SUBPOOL_INDEX_V2" ]; then
    DIMENSION_SUBPOOL_INDEX="$SD35_TURBO_DIMENSION_SUBPOOL_INDEX_V2"
elif [ -f "$SD35_TURBO_DIMENSION_SUBPOOL_INDEX" ]; then
    DIMENSION_SUBPOOL_INDEX="$SD35_TURBO_DIMENSION_SUBPOOL_INDEX"
fi
OUTPUT_DIR="/root/autodl-tmp/Aesthetic_Quality_closed_loop/pipeline_output_sd35_large_turbo_$(date +%Y%m%d_%H%M%S)"
MODEL_PATH="/root/autodl-tmp/AGIQA/sd3.5-large-turbo"

# 生成参数
NUM_PAIRS=3
MAX_PROMPTS=3
MAX_RETRIES=2
SEED=42
SHUFFLE=true

# 退化配置
SEVERITIES="moderate,severe"
SUBCATEGORY="aesthetic_quality"   # perspective 过滤；实际输出目录名会是具体 attribute，例如 cluttered_scene
#ATTRIBUTE="blur"

# 速度优先默认关闭；显存不足时改为 true
USE_CPU_OFFLOAD=false
RUNTIME_PROFILE="fit-24g"

echo "=========================================="
echo "Closed-Loop Pipeline - SD3.5 Large Turbo"
echo "=========================================="
echo "输出目录: $OUTPUT_DIR"
echo "Prompts 文件: $SOURCE_PROMPTS"
echo "模型: SD3.5 Large Turbo"
echo "模型路径: $MODEL_PATH"
echo "每 prompt 生成: $NUM_PAIRS 对"
echo "最大重试: $MAX_RETRIES 次"
echo "Shuffle: $SHUFFLE"
echo "Runtime Profile: $RUNTIME_PROFILE"
echo "CPU Offload: $USE_CPU_OFFLOAD"
echo ""

# 推荐参数: Turbo steps=4, cfg=0.0
CMD="python scripts/pipeline.py \
    --source_prompts $SOURCE_PROMPTS \
    --output_dir $OUTPUT_DIR \
    --model_id sd3.5-large-turbo \
    --model_path $MODEL_PATH \
    --runtime_profile fit-24g \
    --num_pairs_per_prompt $NUM_PAIRS \
    --max_prompts $MAX_PROMPTS \
    --max_retries $MAX_RETRIES \
    --seed $SEED \
    --severities $SEVERITIES \
    --steps 4 \
    --cfg 0.0 \
    --dimension_subpool_index $DIMENSION_SUBPOOL_INDEX"

if [ "$SHUFFLE" = true ]; then
    CMD="$CMD --shuffle"
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
