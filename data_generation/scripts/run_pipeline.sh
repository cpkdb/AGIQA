#!/bin/bash
# 闭环 Pipeline 运行脚本

cd /root/ImageReward/data_generation

# 基础配置
SOURCE_PROMPTS="data/prompts_tagged_sdxl_v2.json"
OUTPUT_DIR="/root/autodl-tmp/Aesthetic_Quality_closed_loop/pipeline_output_$(date +%Y%m%d_%H%M%S)"
MODEL_PATH="/root/ckpts/sd_xl_base_1.0.safetensors"

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
echo "Closed-Loop Pipeline"
echo "=========================================="
echo "输出目录: $OUTPUT_DIR"
echo "Prompts 文件: $SOURCE_PROMPTS"
echo "每 prompt 生成: $NUM_PAIRS 对"
echo "最大重试: $MAX_RETRIES 次"
echo "Shuffle: $SHUFFLE"
echo ""

# 构建命令
CMD="python scripts/pipeline.py \
    --source_prompts $SOURCE_PROMPTS \
    --output_dir $OUTPUT_DIR \
    --model_path $MODEL_PATH \
    --num_pairs_per_prompt $NUM_PAIRS \
    --max_prompts $MAX_PROMPTS \
    --max_retries $MAX_RETRIES \
    --seed $SEED \
    --severities $SEVERITIES \
    --steps 35 \
    --cfg 7.5 \
    --model_filter sdxl"

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
