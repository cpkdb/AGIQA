#!/bin/bash
# 正负对比自监督AIGC数据集生成Demo快速运行脚本

# 设置默认参数
OUTPUT_DIR="${OUTPUT_DIR:-/root/ImageReward/data_generation/demo_output}"
MODEL_PATH="${MODEL_PATH:-/root/ckpts/sd_xl_base_1.0.safetensors}"
NUM_SAMPLES="${NUM_SAMPLES:-3}"  # 默认只生成3个样本用于快速测试
SEED="${SEED:-42}"

echo "======================================"
echo "正负对比自监督AIGC数据集生成Demo"
echo "======================================"
echo "输出目录: $OUTPUT_DIR"
echo "模型路径: $MODEL_PATH"
echo "样本数量: $NUM_SAMPLES"
echo "随机种子: $SEED"
echo "======================================"

# 切换到脚本目录
cd "$(dirname "$0")"

# 运行Demo
python contrastive_dataset_demo.py \
    --output_dir "$OUTPUT_DIR" \
    --model_path "$MODEL_PATH" \
    --num_samples "$NUM_SAMPLES" \
    --seed "$SEED" \
    "$@"

echo ""
echo "Demo完成！查看结果:"
echo "  - 图像: $OUTPUT_DIR/images/"
echo "  - 数据集: $OUTPUT_DIR/dataset.json"
echo "  - 统计: $OUTPUT_DIR/summary.json"
