#!/bin/bash
# 诊断脚本：按子组 × 模型 系统化跑闭环
# 目的：验证 Judge 有效性、定位需调整的 system prompt、发现模型-维度兼容性
#
# 用法:
#   bash scripts/run_diagnostic.sh                              # 全部 (6子组 × 多模型)
#   bash scripts/run_diagnostic.sh sdxl                         # 只跑 SDXL (6子组)
#   bash scripts/run_diagnostic.sh sdxl technical_quality       # SDXL + 技术质量
#   bash scripts/run_diagnostic.sh sdxl semantic_anatomy        # SDXL + 语义-解剖
#   bash run_diagnostic.sh all semantic_spatial        # 两模型 + 指定维度
#
# 支持的子组:
#   technical_quality    (7 dims: blur, overexposure, underexposure, ...)
#   aesthetic_quality     (9 dims: awkward_positioning, flat_lighting, ...)
#   semantic_anatomy      (7 dims: hand_malformation, face_asymmetry, ...)
#   semantic_object       (6 dims: object_shape_error, object_fusion, ...)
#   semantic_spatial     (11 dims: perspective_error, floating_objects, ...)
#   semantic_text         (2 dims: text_error, logo_symbol_error)
#
# 也可直接传单维度名: blur, hand_malformation, ...

set -e
cd /root/ImageReward/data_generation

# ===== 配置 =====
# 使用 autodl-tmp 中已有的 HuggingFace 缓存，避免重复下载
export HF_HOME="/root/autodl-tmp/huggingface_cache"

SOURCE_PROMPTS="data/prompts_tagged_sdxl_v3.json"
OUTPUT_ROOT="/root/autodl-tmp/diagnostic_v2_$(date +%Y%m%d)"

PROMPTS_PER_DIM=3
MAX_RETRIES=2
SEED=42
SEVERITIES="moderate,severe"

# 使用模型 ID，让 diffusers 自动从 HF_HOME 缓存加载
SDXL_MODEL_PATH="stabilityai/stable-diffusion-xl-base-1.0"
FLUX_SCHNELL_MODEL_PATH="/root/autodl-tmp/flux-1-schnell"
HUNYUAN_DIT_MODEL_PATH="/root/autodl-tmp/hunyuan-dit"
SD35_LARGE_MODEL_PATH="/root/autodl-tmp/sd3.5-large"
QWEN_IMAGE_LIGHTNING_MODEL_PATH="Qwen/Qwen-Image"

ALL_GROUPS="technical_quality aesthetic_quality semantic_anatomy semantic_object semantic_spatial semantic_text"

# ===== 参数解析 =====
FILTER_MODEL="${1:-all}"       # all / sdxl / flux-schnell
FILTER_GROUP="${2:-all}"       # all / 子组名 / 逗号分隔的维度名

run_one() {
    local model_id=$1
    local group=$2
    local model_path=$3
    local steps=$4
    local cfg=$5

    local out_dir="${OUTPUT_ROOT}/${model_id}/${group}"

    echo ""
    echo "============================================================"
    echo "  ${model_id} / ${group}"
    echo "  output: ${out_dir}"
    echo "============================================================"

    local cmd="python scripts/pipeline.py \
        --source_prompts $SOURCE_PROMPTS \
        --output_dir $out_dir \
        --model_id $model_id \
        --model_path $model_path \
        --num_pairs_per_prompt $PROMPTS_PER_DIM \
        --max_retries $MAX_RETRIES \
        --seed $SEED \
        --severities $SEVERITIES \
        --steps $steps \
        --cfg $cfg \
        --model_filter sdxl \
        --shuffle \
        --systematic"

    # 判断是子组/perspective 还是逗号分隔的维度名
    if echo "$group" | grep -q ','; then
        # 包含逗号 → 视为维度列表
        cmd="$cmd --attribute_filter $group"
    else
        cmd="$cmd --subcategory_filter $group"
    fi

    if [ "$model_id" = "flux-schnell" ]; then
        cmd="$cmd --optimize"
    fi

    eval $cmd
    local status=$?

    if [ $status -eq 0 ]; then
        echo "[OK] ${model_id}/${group}"
    else
        echo "[FAIL] ${model_id}/${group} exit=$status"
    fi

    return $status
}

run_model() {
    local model_id=$1
    local model_path=$2
    local steps=$3
    local cfg=$4

    if [ "$FILTER_GROUP" = "all" ]; then
        for group in $ALL_GROUPS; do
            run_one "$model_id" "$group" "$model_path" "$steps" "$cfg"
        done
    else
        run_one "$model_id" "$FILTER_GROUP" "$model_path" "$steps" "$cfg"
    fi
}

echo "=============================="
echo "  Diagnostic Pipeline"
echo "=============================="
echo "日期: $(date)"
echo "模型: $FILTER_MODEL"
echo "子组: $FILTER_GROUP"
echo "每维度: ${PROMPTS_PER_DIM} prompts × severities=[${SEVERITIES}]"
echo "输出: $OUTPUT_ROOT"
echo ""

# SDXL
if [ "$FILTER_MODEL" = "all" ] || [ "$FILTER_MODEL" = "sdxl" ]; then
    run_model sdxl "$SDXL_MODEL_PATH" 35 7.5
fi

# Flux-Schnell
if [ "$FILTER_MODEL" = "all" ] || [ "$FILTER_MODEL" = "flux-schnell" ]; then
    run_model flux-schnell "$FLUX_SCHNELL_MODEL_PATH" 4 0.0
fi

# Hunyuan-DiT
if [ "$FILTER_MODEL" = "all" ] || [ "$FILTER_MODEL" = "hunyuan-dit" ]; then
    run_model hunyuan-dit "$HUNYUAN_DIT_MODEL_PATH" 30 5.0
fi

# SD3.5 Large
if [ "$FILTER_MODEL" = "all" ] || [ "$FILTER_MODEL" = "sd3.5-large" ]; then
    run_model sd3.5-large "$SD35_LARGE_MODEL_PATH" 28 4.5
fi

# Qwen-Image-Lightning
if [ "$FILTER_MODEL" = "all" ] || [ "$FILTER_MODEL" = "qwen-image-lightning" ]; then
    run_model qwen-image-lightning "$QWEN_IMAGE_LIGHTNING_MODEL_PATH" 4 1.0
fi

echo ""
echo "=============================="
echo "  完成 — $OUTPUT_ROOT"
echo "=============================="
