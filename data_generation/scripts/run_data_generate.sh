#!/bin/bash
# ============================================================
# AIGC图像质量退化数据集生成脚本
# 只需设置 ATTRIBUTE，其他路径自动推导
# ============================================================

# ============================================================
# 【核心配置】只需修改这里
# ============================================================

# 选择要生成的属性
ATTRIBUTE="object_deformation"
# 可选值（23个属性）:
#   technical_quality:       blur, exposure_issues, low_contrast, color_distortion, color_banding
#   texture_detail:          over_smoothing, texture_artifacts, repetitive_patterns, edge_artifacts
#   aesthetic_quality:       poor_composition, poor_lighting, unharmonious_colors, lack_of_visual_appeal
#   structural_plausibility: object_deformation, perspective_error, physical_implausibility, hallucinated_objects, illogical_colors, contextual_mismatch
#   anatomical_accuracy:     hand_deformity, facial_anomaly, body_proportion, limb_abnormality

# 退化程度（留空则随机选择 mild/moderate/severe）
SEVERITY="severe"

# 每个正样本配对的负样本数量
NUM_NEGATIVES=2

# 正样本数量（留空使用全部）
NUM_POSITIVE_PROMPTS="3"

# ============================================================
# 【生成参数】一般不需要修改
# ============================================================
RANDOM_SELECT_PROMPTS=true    # 是否随机选择正样本
NUM_INFERENCE_STEPS=40         # SDXL推理步数
GUIDANCE_SCALE=7.5             # CFG scale
BASE_SEED=42                   # 基础随机种子

# ============================================================
# 【路径配置】一般不需要修改
# ============================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
OUTPUT_BASE="/root/autodl-tmp/demo"

LLM_CONFIG="$PROJECT_ROOT/config/llm_config.yaml"
QUALITY_DIMENSIONS="$PROJECT_ROOT/config/quality_dimensions.json"
MODEL_PATH="/root/ckpts/sd_xl_base_1.0.safetensors"

# ============================================================
# 【自动推导】根据属性自动设置类别和路径
# ============================================================

get_category() {
    local attr=$1
    case $attr in
        blur|exposure_issues|low_contrast|color_distortion|color_banding)
            echo "technical_quality" ;;
        over_smoothing|texture_artifacts|repetitive_patterns|edge_artifacts)
            echo "texture_detail" ;;
        poor_composition|poor_lighting|unharmonious_colors|lack_of_visual_appeal)
            echo "aesthetic_quality" ;;
        object_deformation|perspective_error|physical_implausibility|hallucinated_objects|illogical_colors|contextual_mismatch)
            echo "structural_plausibility" ;;
        hand_deformity|facial_anomaly|body_proportion|limb_abnormality)
            echo "anatomical_accuracy" ;;
        *)
            echo "" ;;
    esac
}

# 验证属性
CATEGORY=$(get_category "$ATTRIBUTE")
if [ -z "$CATEGORY" ]; then
    echo "错误: 未知属性 '$ATTRIBUTE'"
    echo ""
    echo "可用属性:"
    echo "  technical_quality:       blur, exposure_issues, low_contrast, color_distortion, color_banding"
    echo "  texture_detail:          over_smoothing, texture_artifacts, repetitive_patterns, edge_artifacts"
    echo "  aesthetic_quality:       poor_composition, poor_lighting, unharmonious_colors, lack_of_visual_appeal"
    echo "  structural_plausibility: object_deformation, perspective_error, physical_implausibility, hallucinated_objects, illogical_colors, contextual_mismatch"
    echo "  anatomical_accuracy:     hand_deformity, facial_anomaly, body_proportion, limb_abnormality"
    exit 1
fi

# 自动设置源prompt文件
SOURCE_PROMPTS="$PROJECT_ROOT/data/prompts/$CATEGORY/$ATTRIBUTE.json"
if [ ! -f "$SOURCE_PROMPTS" ]; then
    echo "错误: 源prompt文件不存在: $SOURCE_PROMPTS"
    exit 1
fi

# 自动生成输出目录（带时间戳和退化程度）
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
if [ -n "$SEVERITY" ]; then
    OUTPUT_DIR="${OUTPUT_BASE}/_${ATTRIBUTE}_${SEVERITY}_${TIMESTAMP}"
else
    OUTPUT_DIR="${OUTPUT_BASE}/_${ATTRIBUTE}_${TIMESTAMP}"
fi

# ============================================================
# 显示配置信息
# ============================================================
echo "============================================================"
echo "AIGC图像质量退化数据集生成"
echo "============================================================"
echo "属性:         $ATTRIBUTE"
echo "类别:         $CATEGORY"
echo "退化程度:     ${SEVERITY:-随机}"
echo "源prompt:     $SOURCE_PROMPTS"
echo "输出目录:     $OUTPUT_DIR"
echo "正样本数量:   ${NUM_POSITIVE_PROMPTS:-全部}"
echo "负样本数量:   $NUM_NEGATIVES (每个正样本)"
echo "推理步数:     $NUM_INFERENCE_STEPS"
echo "============================================================"

# ============================================================
# 构建并执行命令
# ============================================================
CMD="python $SCRIPT_DIR/generate_dataset.py"
CMD="$CMD --source_prompts \"$SOURCE_PROMPTS\""
CMD="$CMD --output_dir \"$OUTPUT_DIR\""
CMD="$CMD --num_negatives_per_positive $NUM_NEGATIVES"
CMD="$CMD --num_inference_steps $NUM_INFERENCE_STEPS"
CMD="$CMD --guidance_scale $GUIDANCE_SCALE"
CMD="$CMD --base_seed $BASE_SEED"
CMD="$CMD --model_path \"$MODEL_PATH\""
CMD="$CMD --quality_dimensions \"$QUALITY_DIMENSIONS\""
CMD="$CMD --llm_config \"$LLM_CONFIG\""
CMD="$CMD --subcategory_filter $CATEGORY"
CMD="$CMD --attribute_filter $ATTRIBUTE"

if [ -n "$NUM_POSITIVE_PROMPTS" ]; then
    CMD="$CMD --num_positive_prompts $NUM_POSITIVE_PROMPTS"
fi

if [ "$RANDOM_SELECT_PROMPTS" = true ]; then
    CMD="$CMD --random_select_prompts"
fi

if [ -n "$SEVERITY" ]; then
    CMD="$CMD --severity $SEVERITY"
fi

# 执行
eval $CMD

echo "============================================================"
echo "完成！输出目录: $OUTPUT_DIR"
echo "============================================================"
