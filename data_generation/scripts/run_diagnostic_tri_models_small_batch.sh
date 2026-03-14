#!/bin/bash
# 三模型小批量全维度退化测试（速度优先）
# 目标模型：
#   - flux-schnell
#   - sd3.5-large (建议加载 Turbo 权重)
#   - qwen-image-lightning
#
# 用法:
#   bash scripts/run_diagnostic_tri_models_small_batch.sh
#   bash scripts/run_diagnostic_tri_models_small_batch.sh flux-schnell
#   bash scripts/run_diagnostic_tri_models_small_batch.sh sd3.5-large-turbo semantic_spatial
#   bash scripts/run_diagnostic_tri_models_small_batch.sh all blur,text_error
#
# 第1参数: 模型过滤
#   all / flux-schnell / sd3.5-large-turbo / sd3.5-large / qwen-image-lightning
# 第2参数: 维度过滤
#   all / 子组名 / 逗号分隔维度名

set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/huggingface_cache}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/3.10/bin/python}"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export JUDGE_CONFIG_PATH="${JUDGE_CONFIG_PATH:-/root/ImageReward/data_generation/config/judge_config_api_gpt_ge.yaml}"

cd /root/ImageReward/data_generation

# ===== Prompt 池配置 =====
# 默认使用你当前构建的全量 merged prompt 池（JSONL）
BASE_SOURCE_PROMPTS="/root/autodl-tmp/AGIQA/data/prompt_sources_workspace/backfill_merge_runs/all_dimensions_v1_full/merged_working_pool.jsonl"
CLEANED_SOURCE_PROMPTS="/root/autodl-tmp/AGIQA/data/prompt_sources_workspace/backfill_merge_runs/all_dimensions_v1_full/merged_working_pool_cleaned_v1.jsonl"
SD35_TURBO_SOURCE_PROMPTS="/root/autodl-tmp/AGIQA/data/prompt_sources_workspace/backfill_merge_runs/all_dimensions_v1_full/merged_working_pool_sd35_turbo_clipsafe_v1.jsonl"
SOURCE_PROMPTS="${SOURCE_PROMPTS:-$BASE_SOURCE_PROMPTS}"
if [[ -f "$CLEANED_SOURCE_PROMPTS" ]]; then
    SOURCE_PROMPTS="$CLEANED_SOURCE_PROMPTS"
fi
BASE_DIMENSION_SUBPOOL_INDEX="/root/autodl-tmp/AGIQA/data/prompt_sources_workspace/backfill_merge_runs/all_dimensions_v1_full/dimension_subpools/index.json"
BASE_CLEANED_DIMENSION_SUBPOOL_INDEX="/root/autodl-tmp/AGIQA/data/prompt_sources_workspace/backfill_merge_runs/all_dimensions_v1_full/dimension_subpools_cleaned_v1/index.json"
SEMANTIC_SCREENED_DIMENSION_SUBPOOL_INDEX_V1="/root/autodl-tmp/AGIQA/data/prompt_sources_workspace/backfill_merge_runs/all_dimensions_v1_full/semantic_screened_dimension_subpools_cleaned_v1/index.json"
SCREENED_CLEANED_DIMENSION_SUBPOOL_INDEX_V2="/root/autodl-tmp/AGIQA/data/prompt_sources_workspace/backfill_merge_runs/all_dimensions_v1_full/anatomy_screened_dimension_subpools_cleaned_v2/index.json"
SCREENED_CLEANED_DIMENSION_SUBPOOL_INDEX="/root/autodl-tmp/AGIQA/data/prompt_sources_workspace/backfill_merge_runs/all_dimensions_v1_full/anatomy_screened_dimension_subpools_cleaned_v1/index.json"
SCREENED_DIMENSION_SUBPOOL_INDEX="/root/autodl-tmp/AGIQA/data/prompt_sources_workspace/backfill_merge_runs/all_dimensions_v1_full/anatomy_screened_dimension_subpools/index.json"
SD35_TURBO_SEMANTIC_SCREENED_DIMENSION_SUBPOOL_INDEX_V1="/root/autodl-tmp/AGIQA/data/prompt_sources_workspace/backfill_merge_runs/all_dimensions_v1_full/sd35_turbo_semantic_screened_dimension_subpools_clipsafe_v1/index.json"
SD35_TURBO_DIMENSION_SUBPOOL_INDEX_V2="/root/autodl-tmp/AGIQA/data/prompt_sources_workspace/backfill_merge_runs/all_dimensions_v1_full/sd35_turbo_dimension_subpools_clipsafe_v2/index.json"
SD35_TURBO_DIMENSION_SUBPOOL_INDEX="/root/autodl-tmp/AGIQA/data/prompt_sources_workspace/backfill_merge_runs/all_dimensions_v1_full/sd35_turbo_dimension_subpools_clipsafe_v1/index.json"
DIMENSION_SUBPOOL_INDEX="${DIMENSION_SUBPOOL_INDEX:-$BASE_DIMENSION_SUBPOOL_INDEX}"
if [[ -f "$SEMANTIC_SCREENED_DIMENSION_SUBPOOL_INDEX_V1" ]]; then
    DIMENSION_SUBPOOL_INDEX="$SEMANTIC_SCREENED_DIMENSION_SUBPOOL_INDEX_V1"
elif [[ -f "$SCREENED_CLEANED_DIMENSION_SUBPOOL_INDEX_V2" ]]; then
    DIMENSION_SUBPOOL_INDEX="$SCREENED_CLEANED_DIMENSION_SUBPOOL_INDEX_V2"
elif [[ -f "$SCREENED_CLEANED_DIMENSION_SUBPOOL_INDEX" ]]; then
    DIMENSION_SUBPOOL_INDEX="$SCREENED_CLEANED_DIMENSION_SUBPOOL_INDEX"
elif [[ -f "$BASE_CLEANED_DIMENSION_SUBPOOL_INDEX" ]]; then
    DIMENSION_SUBPOOL_INDEX="$BASE_CLEANED_DIMENSION_SUBPOOL_INDEX"
elif [[ -f "$SCREENED_DIMENSION_SUBPOOL_INDEX" ]]; then
    DIMENSION_SUBPOOL_INDEX="$SCREENED_DIMENSION_SUBPOOL_INDEX"
fi
# 如果样本中含 model 字段且你想筛选，可设置 MODEL_FILTER_IN_PROMPTS=sdxl
MODEL_FILTER_IN_PROMPTS="${MODEL_FILTER_IN_PROMPTS:-}"

# ===== 输出与采样配置 =====
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/autodl-tmp/tri_model_small_batch_$(date +%Y%m%d_%H%M%S)}"
PROMPTS_PER_DIM="${PROMPTS_PER_DIM:-2}"
MAX_RETRIES="${MAX_RETRIES:-1}"
SEED="${SEED:-42}"
SEVERITIES="${SEVERITIES:-moderate,severe}"

# ===== 模型路径（按需改成本地目录）=====
FLUX_SCHNELL_MODEL_PATH="${FLUX_SCHNELL_MODEL_PATH:-/root/autodl-tmp/flux-schnell}"
# 这里建议放 Turbo 权重目录或 HF model id，例如 stabilityai/stable-diffusion-3.5-large-turbo
SD35_LARGE_TURBO_MODEL_PATH="${SD35_LARGE_TURBO_MODEL_PATH:-/root/autodl-tmp/AGIQA/sd3.5-large-turbo}"
# Qwen-Image-Lightning 采用 base + LoRA，默认指向已验证通过的官方 snapshot
QWEN_IMAGE_LIGHTNING_MODEL_PATH="${QWEN_IMAGE_LIGHTNING_MODEL_PATH:-/root/autodl-tmp/AGIQA/Qwen-Image/snapshots/75e0b4be04f60ec59a75f475837eced720f823b6}"
QWEN_IMAGE_LIGHTNING_NUNCHAKU_MODEL_PATH="${QWEN_IMAGE_LIGHTNING_NUNCHAKU_MODEL_PATH:-/root/autodl-tmp/AGIQA/Nunchaku/svdq-int4_r32-qwen-image-lightningv1.0-4steps.safetensors}"

# 显存不足时可改为 true（速度会下降）
SD35_USE_CPU_OFFLOAD="${SD35_USE_CPU_OFFLOAD:-false}"
QWEN_USE_CPU_OFFLOAD="${QWEN_USE_CPU_OFFLOAD:-false}"

ALL_GROUPS="technical_quality aesthetic_quality semantic_anatomy semantic_object semantic_spatial semantic_text"

# ===== 参数解析 =====
FILTER_MODEL="${1:-all}"
FILTER_GROUP="${2:-all}"

run_one() {
    local model_id="$1"
    local group="$2"
    local model_path="$3"
    local steps="$4"
    local cfg="$5"
    local runtime_profile="$6"
    local source_prompts="$SOURCE_PROMPTS"
    local dimension_subpool_index="$DIMENSION_SUBPOOL_INDEX"

    if [[ "$model_id" == "sd3.5-large-turbo" ]]; then
        if [[ -f "$SD35_TURBO_SOURCE_PROMPTS" ]]; then
            source_prompts="$SD35_TURBO_SOURCE_PROMPTS"
        fi
        if [[ -f "$SD35_TURBO_SEMANTIC_SCREENED_DIMENSION_SUBPOOL_INDEX_V1" ]]; then
            dimension_subpool_index="$SD35_TURBO_SEMANTIC_SCREENED_DIMENSION_SUBPOOL_INDEX_V1"
        elif [[ -f "$SD35_TURBO_DIMENSION_SUBPOOL_INDEX_V2" ]]; then
            dimension_subpool_index="$SD35_TURBO_DIMENSION_SUBPOOL_INDEX_V2"
        elif [[ -f "$SD35_TURBO_DIMENSION_SUBPOOL_INDEX" ]]; then
            dimension_subpool_index="$SD35_TURBO_DIMENSION_SUBPOOL_INDEX"
        fi
    fi

    local out_dir="${OUTPUT_ROOT}/${model_id}/${group}"

    echo ""
    echo "============================================================"
    echo "  ${model_id} / ${group}"
    echo "  output: ${out_dir}"
    echo "============================================================"

    local -a cmd=(
        "$PYTHON_BIN" scripts/pipeline.py
        --source_prompts "$source_prompts"
        --output_dir "$out_dir"
        --model_id "$model_id"
        --model_path "$model_path"
        --runtime_profile "$runtime_profile"
        --num_pairs_per_prompt "$PROMPTS_PER_DIM"
        --max_retries "$MAX_RETRIES"
        --seed "$SEED"
        --severities "$SEVERITIES"
        --steps "$steps"
        --cfg "$cfg"
        --shuffle
        --systematic
        --dimension_subpool_index "$dimension_subpool_index"
    )

    if [[ -n "$MODEL_FILTER_IN_PROMPTS" ]]; then
        cmd+=(--model_filter "$MODEL_FILTER_IN_PROMPTS")
    fi

    if [[ "$group" == *,* ]]; then
        cmd+=(--attribute_filter "$group")
    else
        cmd+=(--subcategory_filter "$group")
    fi

    if [[ "$model_id" == "flux-schnell" ]]; then
        cmd+=(--optimize)
    fi

    if [[ "$model_id" == "sd3.5-large" && "$SD35_USE_CPU_OFFLOAD" == "true" ]]; then
        cmd+=(--use_cpu_offload)
    fi

    if [[ "$model_id" == "qwen-image-lightning" && "$QWEN_USE_CPU_OFFLOAD" == "true" ]]; then
        cmd+=(--use_cpu_offload)
    fi

    if [[ "$runtime_profile" == "nunchaku-int4" ]]; then
        cmd+=(--nunchaku_model_path "$QWEN_IMAGE_LIGHTNING_NUNCHAKU_MODEL_PATH")
    fi

    "${cmd[@]}"
    local status=$?

    if [[ $status -eq 0 ]]; then
        echo "[OK] ${model_id}/${group}"
    else
        echo "[FAIL] ${model_id}/${group} exit=${status}"
    fi

    return "$status"
}

run_model() {
    local model_id="$1"
    local model_path="$2"
    local steps="$3"
    local cfg="$4"
    local runtime_profile="$5"

    if [[ "$FILTER_GROUP" == "all" ]]; then
        for group in $ALL_GROUPS; do
            run_one "$model_id" "$group" "$model_path" "$steps" "$cfg" "$runtime_profile"
        done
    else
        run_one "$model_id" "$FILTER_GROUP" "$model_path" "$steps" "$cfg" "$runtime_profile"
    fi
}

echo "=============================="
echo "  Tri-Model Small-Batch Diagnostic"
echo "=============================="
echo "日期: $(date)"
echo "模型过滤: $FILTER_MODEL"
echo "维度过滤: $FILTER_GROUP"
echo "prompts: $SOURCE_PROMPTS"
echo "每维度: ${PROMPTS_PER_DIM} prompts × severities=[${SEVERITIES}]"
echo "输出: $OUTPUT_ROOT"
echo ""

# Flux-Schnell (4-step 快路径)
if [[ "$FILTER_MODEL" == "all" || "$FILTER_MODEL" == "flux-schnell" ]]; then
    run_model flux-schnell "$FLUX_SCHNELL_MODEL_PATH" 4 0.0 fast-gpu
fi

# SD3.5 Large (使用 Turbo 权重，沿用 sd3.5-large 生成器接口)
if [[ "$FILTER_MODEL" == "all" || "$FILTER_MODEL" == "sd3.5-large-turbo" || "$FILTER_MODEL" == "sd3.5-large" ]]; then
    run_model sd3.5-large-turbo "$SD35_LARGE_TURBO_MODEL_PATH" 4 1.0 fit-24g
fi

# Qwen-Image-Lightning (Nunchaku INT4 快路径)
if [[ "$FILTER_MODEL" == "all" || "$FILTER_MODEL" == "qwen-image-lightning" ]]; then
    run_model qwen-image-lightning "$QWEN_IMAGE_LIGHTNING_MODEL_PATH" 4 1.0 nunchaku-int4
fi

echo ""
echo "=============================="
echo "  完成 — $OUTPUT_ROOT"
echo "=============================="
