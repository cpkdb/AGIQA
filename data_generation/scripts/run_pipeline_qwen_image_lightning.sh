#!/bin/bash
# 闭环 Pipeline 运行脚本 - Qwen-Image-Lightning 版本

# 设置 PyTorch 显存分配策略，减少碎片化
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /root/ImageReward/data_generation

# 基础配置
SOURCE_PROMPTS="data/prompts_tagged_sdxl_v2.json"
OUTPUT_DIR="/root/autodl-tmp/Aesthetic_Quality_closed_loop/pipeline_output_qwen_image_lightning_$(date +%Y%m%d_%H%M%S)"
MODEL_PATH="Qwen/Qwen-Image"

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

# 若显存紧张可打开
USE_CPU_OFFLOAD=false

echo "=========================================="
echo "Closed-Loop Pipeline - Qwen-Image-Lightning"
echo "=========================================="
echo "输出目录: $OUTPUT_DIR"
echo "Prompts 文件: $SOURCE_PROMPTS"
echo "模型: Qwen-Image-Lightning"
echo "Base 模型: $MODEL_PATH"
echo "每 prompt 生成: $NUM_PAIRS 对"
echo "最大重试: $MAX_RETRIES 次"
echo "Shuffle: $SHUFFLE"
echo "CPU Offload: $USE_CPU_OFFLOAD"
echo ""

# 官方推荐快路径: steps=4, true_cfg_scale=1.0
CMD="python scripts/pipeline.py \
    --source_prompts $SOURCE_PROMPTS \
    --output_dir $OUTPUT_DIR \
    --model_id qwen-image-lightning \
    --model_path $MODEL_PATH \
    --num_pairs_per_prompt $NUM_PAIRS \
    --max_prompts $MAX_PROMPTS \
    --max_retries $MAX_RETRIES \
    --seed $SEED \
    --severities $SEVERITIES \
    --steps 4 \
    --cfg 1.0 \
    --model_filter sdxl"

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
