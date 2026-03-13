#!/bin/bash
# 闭环 Pipeline 运行脚本 - Flux.1-dev 版本

# 设置 PyTorch 显存分配策略，减少碎片化
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /root/ImageReward/data_generation

# 基础配置
SOURCE_PROMPTS="/root/autodl-tmp/AGIQA/data/prompt_sources_workspace/backfill_merge_runs/all_dimensions_v1_full/merged_working_pool_cleaned_v1.jsonl"
DIMENSION_SUBPOOL_INDEX="/root/autodl-tmp/AGIQA/data/prompt_sources_workspace/backfill_merge_runs/all_dimensions_v1_full/anatomy_screened_dimension_subpools_cleaned_v2/index.json"
if [ ! -f "$DIMENSION_SUBPOOL_INDEX" ]; then
  DIMENSION_SUBPOOL_INDEX="/root/autodl-tmp/AGIQA/data/prompt_sources_workspace/backfill_merge_runs/all_dimensions_v1_full/anatomy_screened_dimension_subpools_cleaned_v1/index.json"
fi
OUTPUT_DIR="/root/autodl-tmp/Aesthetic_Quality_closed_loop/pipeline_output_flux_$(date +%Y%m%d_%H%M%S)"
MODEL_PATH="/root/autodl-tmp/flux-fp8"

# 生成参数
NUM_PAIRS=3          # 每个 prompt 生成的负样本数
MAX_PROMPTS=3        # 限制 prompt 数量（去掉此行则处理全部）
MAX_RETRIES=2        # 每对失败后最多重试次数
SEED=42
SHUFFLE=true         # 是否随机打乱 prompt 顺序

# 退化配置
SEVERITIES="moderate,severe"
# aesthetic_quality
# semantic_anatomy
# semantic_object
# semantic_spatial
# semantic_text
# technical_quality
SUBCATEGORY="aesthetic_quality"    # 可选：指定 perspective
#ATTRIBUTE="blur"                   # 可选：指定退化维度

echo "=========================================="
echo "Closed-Loop Pipeline - Flux.1-dev"
echo "=========================================="
echo "输出目录: $OUTPUT_DIR"
echo "Prompts 文件: $SOURCE_PROMPTS"
echo "模型: Flux.1-dev"
echo "每 prompt 生成: $NUM_PAIRS 对"
echo "最大重试: $MAX_RETRIES 次"
echo "Shuffle: $SHUFFLE"
echo ""

# 构建命令
CMD="python scripts/pipeline.py \
    --source_prompts $SOURCE_PROMPTS \
    --output_dir $OUTPUT_DIR \
    --model_id flux \
    --model_path $MODEL_PATH \
    --num_pairs_per_prompt $NUM_PAIRS \
    --max_prompts $MAX_PROMPTS \
    --max_retries $MAX_RETRIES \
    --seed $SEED \
    --severities $SEVERITIES \
    --steps 28 \
    --cfg 3.5 \
    --model_filter sdxl \
    --dimension_subpool_index $DIMENSION_SUBPOOL_INDEX"

# 可选参数
if [ "$SHUFFLE" = true ]; then
    CMD="$CMD --shuffle"
fi

if [ -n "$SUBCATEGORY" ]; then
    CMD="$CMD --subcategory_filter $SUBCATEGORY"
fi

if [ -n "$ATTRIBUTE" ]; then
    CMD="$CMD --attribute_filter $ATTRIBUTE"
fi

eval $CMD

echo ""
echo "=========================================="
echo "完成！结果保存在: $OUTPUT_DIR"
echo "  dataset.json    - 成功的训练对"
echo "  full_log.json   - 完整实验日志"
echo "  validation_report.json - 统计报告"
echo "=========================================="
