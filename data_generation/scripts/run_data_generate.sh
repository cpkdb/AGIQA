#!/bin/bash
# ============================================================
# 数据集生成脚本
# 使用 LLM 进行质量退化 + SDXL 生成图像
# ============================================================

# 配置参数
SOURCE_PROMPTS="../data/example_prompts.json"          # 源prompt文件
OUTPUT_DIR="/root/autodl-tmp/dataset_demo_60pairs_v2"               # 输出目录
NUM_NEGATIVES=3                                        # 每个正样本配对的负样本数量
NUM_INFERENCE_STEPS=40                                 # SDXL推理步数
GUIDANCE_SCALE=7.5                                     # CFG scale
BASE_SEED=42                                           # 基础随机种子

# LLM 配置
LLM_CONFIG="../config/llm_config.yaml"

# 质量维度配置
QUALITY_DIMENSIONS="../config/quality_dimensions.json"

# SDXL 模型路径
MODEL_PATH="/root/ckpts/sd_xl_base_1.0.safetensors"

# ============================================================
# 运行数据集生成
# ============================================================
echo "============================================================"
echo "开始生成数据集"
echo "============================================================"
echo "源prompt文件: $SOURCE_PROMPTS"
echo "输出目录: $OUTPUT_DIR"
echo "每个正样本负样本数: $NUM_NEGATIVES"
echo "SDXL推理步数: $NUM_INFERENCE_STEPS"
echo "============================================================"

python generate_dataset.py \
    --source_prompts "$SOURCE_PROMPTS" \
    --output_dir "$OUTPUT_DIR" \
    --num_negatives_per_positive $NUM_NEGATIVES \
    --num_inference_steps $NUM_INFERENCE_STEPS \
    --guidance_scale $GUIDANCE_SCALE \
    --base_seed $BASE_SEED \
    --model_path "$MODEL_PATH" \
    --quality_dimensions "$QUALITY_DIMENSIONS" \
    --llm_config "$LLM_CONFIG"

echo "============================================================"
echo "数据集生成完成！"
echo "输出目录: $OUTPUT_DIR"
echo "============================================================"
