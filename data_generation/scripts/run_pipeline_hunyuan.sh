#!/bin/bash
# 闭环 Pipeline 运行脚本 - Hunyuan-DiT 版本

# 设置 PyTorch 显存分配策略，减少碎片化
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /root/ImageReward/data_generation

# 基础配置
SOURCE_PROMPTS="/root/autodl-tmp/AGIQA/data/prompt_sources_workspace/backfill_merge_runs/all_dimensions_v1_full/merged_working_pool_cleaned_v1.jsonl"
DIMENSION_SUBPOOL_INDEX="/root/autodl-tmp/AGIQA/data/prompt_sources_workspace/backfill_merge_runs/all_dimensions_v1_full/anatomy_screened_dimension_subpools_cleaned_v2/index.json"
if [ ! -f "$DIMENSION_SUBPOOL_INDEX" ]; then
  DIMENSION_SUBPOOL_INDEX="/root/autodl-tmp/AGIQA/data/prompt_sources_workspace/backfill_merge_runs/all_dimensions_v1_full/anatomy_screened_dimension_subpools_cleaned_v1/index.json"
fi
OUTPUT_DIR="/root/autodl-tmp/Aesthetic_Quality_closed_loop/pipeline_output_hunyuan_$(date +%Y%m%d_%H%M%S)"
MODEL_PATH="/root/autodl-tmp/hunyuan-dit"

# 生成参数
NUM_PAIRS=3
MAX_PROMPTS=3
MAX_RETRIES=2
SEED=42
SHUFFLE=true

# 退化配置
SEVERITIES="moderate,severe"
SUBCATEGORY="aesthetic_quality"
#ATTRIBUTE="blur"
USE_CPU_OFFLOAD=false

echo "=========================================="
echo "Closed-Loop Pipeline - Hunyuan-DiT"
echo "=========================================="
echo "输出目录: $OUTPUT_DIR"
echo "Prompts 文件: $SOURCE_PROMPTS"
echo "模型: Hunyuan-DiT"
echo "模型路径: $MODEL_PATH"
echo "每 prompt 生成: $NUM_PAIRS 对"
echo "最大重试: $MAX_RETRIES 次"
echo "Shuffle: $SHUFFLE"
echo "CPU Offload: $USE_CPU_OFFLOAD"
echo ""

# 推荐参数: steps=30, cfg=5.0
CMD="python scripts/pipeline.py \
    --source_prompts $SOURCE_PROMPTS \
    --output_dir $OUTPUT_DIR \
    --model_id hunyuan-dit \
    --model_path $MODEL_PATH \
    --num_pairs_per_prompt $NUM_PAIRS \
    --max_prompts $MAX_PROMPTS \
    --max_retries $MAX_RETRIES \
    --seed $SEED \
    --severities $SEVERITIES \
    --steps 30 \
    --cfg 5.0 \
    --model_filter sdxl \
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
