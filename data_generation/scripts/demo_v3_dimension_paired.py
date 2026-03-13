#!/usr/bin/env python3
"""
V3 prompt_templates_v3 维度 demo：从 image_quality_train.json 选正样本 prompt，
对每个维度的子属性（YAML: subcategory -> attribute）各选 3 个正样本，
分别生成 mild/moderate/severe 退化 prompt，
并用 SDXL 生成正/负图像对。

输出目录结构：
  <output_dir>/
    <subcategory>/
      <attribute>/
        images/
          positive_<seed>.png
          negative_<seed>_<severity>.png
        dataset.json
        prompts_cache.json
"""

import sys
import os

# 检查必要的依赖
try:
    import torch
except ImportError:
    print("错误: 未找到 torch 模块。", file=sys.stderr)
    print("请确保使用正确的 conda 环境运行此脚本。", file=sys.stderr)
    print("建议使用: /root/miniconda3/envs/3.10/bin/python", file=sys.stderr)
    print("或者运行: conda activate 3.10", file=sys.stderr)
    sys.exit(1)

import argparse
import json
import logging
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import hashlib
from concurrent.futures import ThreadPoolExecutor
import yaml

from llm_prompt_degradation import LLMPromptDegradation
from sdxl_generator import SDXLGenerator
from flux_generator import FluxGenerator
from flux_schnell_generator import FluxSchnellGenerator
from hunyuan_dit_generator import HunyuanDiTGenerator
from sd35_large_generator import SD35LargeGenerator
from qwen_image_lightning_generator import QwenImageLightningGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_tagged_prompts(tagged_path: Path) -> Dict[str, List[str]]:
    """
    加载带语义标签的正样本数据。
    返回: { prompt_text: [tag1, tag2, ...] }
    """
    with open(tagged_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    prompts = data.get("prompts", data if isinstance(data, list) else [])
    prompt_to_tags: Dict[str, List[str]] = {}
    for item in prompts:
        prompt = item.get("prompt", item.get("text", ""))
        tags = item.get("semantic_tags", [])
        if isinstance(prompt, str) and prompt.strip():
            prompt_to_tags[prompt.strip()] = tags
    return prompt_to_tags


def load_tag_requirements(tag_config_path: Path) -> Dict[str, Dict]:
    """
    加载语义标签配置，解析维度对标签的要求。
    返回: { dimension_name: {"required": [...], "alternative": [...], "preferred": [...]} }
    """
    with open(tag_config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    dim_reqs = config.get("dimension_requirements", {})
    result: Dict[str, Dict] = {}

    for perspective, dims in dim_reqs.items():
        if not isinstance(dims, dict):
            continue
        for dim_name, req in dims.items():
            if req is None:
                result[dim_name] = {"required": None}
            elif isinstance(req, dict):
                result[dim_name] = {
                    "required": req.get("required"),
                    "alternative": req.get("alternative"),
                    "preferred": req.get("preferred"),
                }
            else:
                result[dim_name] = {"required": None}
    return result


def filter_prompts_by_tags(
    prompt_pool: List[str],
    prompt_to_tags: Dict[str, List[str]],
    dimension_name: str,
    tag_requirements: Dict[str, Dict],
) -> List[str]:
    """
    根据维度的标签要求过滤正样本。
    - 若维度无要求(required=None)，返回全部 prompt_pool
    - 若维度有required标签要求：
      1. prompt 必须存在于 prompt_to_tags 中（有标签数据）
      2. required 标签必须**全部满足**（AND 逻辑）
      3. alternative 标签满足任意一个即可（OR 逻辑）
      4. 找不到标签数据的 prompt 会被直接拒绝
    """
    req = tag_requirements.get(dimension_name)
    if req is None or req.get("required") is None:
        # 该维度无标签要求，返回全部
        return prompt_pool

    required_tags = set(req.get("required") or [])
    alternative_tags = set(req.get("alternative") or [])

    if not required_tags and not alternative_tags:
        return prompt_pool

    filtered = []
    skipped_no_tag_data = 0
    skipped_wrong_tags = 0
    
    for p in prompt_pool:
        # 检查该 prompt 是否在标签数据中
        if p not in prompt_to_tags:
            # 该维度有标签要求，但这个 prompt 没有标签数据，拒绝
            skipped_no_tag_data += 1
            continue
        
        tags = set(prompt_to_tags.get(p, []))
        
        # 检查 required 标签：必须全部满足（AND 逻辑）
        has_all_required = required_tags.issubset(tags) if required_tags else True
        
        # 检查 alternative 标签：满足任意一个即可（OR 逻辑）
        has_any_alternative = bool(tags & alternative_tags) if alternative_tags else False
        
        # 通过条件：满足所有 required，或者满足任意 alternative
        if has_all_required or has_any_alternative:
            # 针对 text_error 的额外严格过滤：
            # 必须包含引号，且包含暗示"视觉文字"的关键词，避免 "theme of '...'" 这种非视觉文本
            if dimension_name == "text_error":
                lower_p = p.lower()
                has_quote = "'" in p or '"' in p
                # 视觉文字指示词列表 (使用 regex word boundary 防止匹配 design, texture 等)
                import re
                visual_indicators = [
                    "sign", "text", "written", "label", "board", "banner", 
                    "poster", "reading", "says", "inscribed", "logo", 
                    "marked", "branded", "printed", "stamp"
                ]
                # 构建 regex: \b(sign|text|...)\b
                pattern = r"\b(" + "|".join(visual_indicators) + r")\b"
                has_indicator = bool(re.search(pattern, lower_p))
                
                if not (has_quote and has_indicator):
                    skipped_wrong_tags += 1
                    continue

            # 针对 face_asymmetry 和 extra_limbs 的额外严格过滤：拒绝背影，拒绝动物，必须有明确人类面部/身体特征词
            if dimension_name in ["face_asymmetry", "extra_limbs"]:
                lower_p = p.lower()
                # 拒绝背影/远景
                negative_keywords = ["back view", "from behind", "facing away", "rear view", "walking away"]
                # 拒绝动物/非人
                animal_keywords = [
                    "cat", "dog", "tiger", "lion", "bear", "wolf", "fox", "animal", 
                    "creature", "robot", "statue", "monkey", "ape", "bird", 
                    "capybara", "rodent", "fish", "insect", "beast", "monster"
                ]
                
                # 使用 Regex 确保全词匹配
                import re
                def has_word(text, words):
                    pattern = r"\b(" + "|".join(re.escape(w) for w in words) + r")\b"
                    return bool(re.search(pattern, text))

                if has_word(lower_p, negative_keywords) or has_word(lower_p, animal_keywords):
                    skipped_wrong_tags += 1
                    continue
                
                # 必须包含明确面部词
                strong_face_keywords = [
                    "face", "eyes", "mouth", "nose", "portrait", "close-up", "headshot", 
                    "looking at viewer", "looking at camera", "smile", "laugh", "cry", "expression"
                ]
                # 必须包含明确人类词
                human_keywords = [
                    "man", "woman", "boy", "girl", "child", "kid", "person", "people", "human", 
                    "guy", "lady", "men", "women", "male", "female"
                ]

                has_face_kw = has_word(lower_p, strong_face_keywords)
                has_human_kw = has_word(lower_p, human_keywords)

                if not (has_face_kw and has_human_kw):
                    skipped_wrong_tags += 1
                    continue
            filtered.append(p)
        else:
            skipped_wrong_tags += 1
    
    if skipped_no_tag_data > 0 or skipped_wrong_tags > 0:
        logger.info(
            f"[{dimension_name}] 标签过滤: 通过 {len(filtered)}, "
            f"无标签数据拒绝 {skipped_no_tag_data}, 标签不符拒绝 {skipped_wrong_tags}"
        )
    
    return filtered


SEVERITIES = ["mild", "moderate", "severe"]


def stable_u32(text: str) -> int:
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)


def load_subcategory_attributes_from_templates(template_dir: Path) -> Dict[str, List[str]]:
    """
    解析 prompt_templates_v3 的结构:
      { subcategory: { attribute: { mild/moderate/severe: prompt } } }
    返回:
      { subcategory: [attribute1, attribute2, ...] }
    """
    mapping: Dict[str, List[str]] = {}
    for yaml_file in sorted(template_dir.glob("*.yaml")):
        with open(yaml_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            continue
        for subcategory, attrs in data.items():
            if not isinstance(subcategory, str) or not subcategory.strip():
                continue
            if not isinstance(attrs, dict):
                continue
            for attribute in attrs.keys():
                if not isinstance(attribute, str) or not attribute.strip():
                    continue
                mapping.setdefault(subcategory, []).append(attribute)

    # 去重并排序，保证稳定
    for subcategory, attrs in list(mapping.items()):
        mapping[subcategory] = sorted(list(dict.fromkeys(attrs)))
    return mapping


def load_positive_prompts_from_image_quality_train(
    json_path: Path,
    min_prompt_len: int = 10,
    top_k_by_score: int = 20000,
    model_substring: str = "",
) -> List[Tuple[str, float]]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    items: List[Tuple[str, float]] = []
    for row in data:
        prompt = row.get("prompt")
        score = row.get("gt_score")
        model = row.get("model", "")
        if model_substring:
            if not isinstance(model, str) or model_substring.lower() not in model.lower():
                continue
        if not isinstance(prompt, str) or not prompt.strip():
            continue
        if len(prompt.strip()) < min_prompt_len:
            continue
        if not isinstance(score, (int, float)):
            continue
        items.append((prompt.strip(), float(score)))

    # 分数从高到低，截取 top_k
    items.sort(key=lambda x: x[1], reverse=True)
    return items[: min(top_k_by_score, len(items))]


def sample_unique_prompts(
    prompt_score_list: List[Tuple[str, float]],
    total_needed: int,
    seed: int,
) -> List[str]:
    rng = random.Random(seed)
    # 先去重（按 prompt 文本）
    deduped: List[str] = []
    seen = set()
    for p, _s in prompt_score_list:
        if p not in seen:
            deduped.append(p)
            seen.add(p)
    if total_needed > len(deduped):
        raise ValueError(f"可用正样本 prompt 数量不足: need={total_needed}, have={len(deduped)}")
    return rng.sample(deduped, total_needed)

def sample_unique_prompts_from_all(
    all_prompts: List[str],
    total_needed: int,
    seed: int,
) -> List[str]:
    rng = random.Random(seed)
    deduped: List[str] = []
    seen = set()
    for p in all_prompts:
        if p not in seen:
            deduped.append(p)
            seen.add(p)
    if total_needed > len(deduped):
        raise ValueError(f"可用正样本 prompt 数量不足: need={total_needed}, have={len(deduped)}")
    return rng.sample(deduped, total_needed)


def call_llm_with_retry(
    degrader: LLMPromptDegradation,
    *,
    positive_prompt: str,
    subcategory: str,
    attribute: str,
    severity: str,
    retries: int,
    retry_sleep: float,
) -> Tuple[str, Dict]:
    last_err: Optional[Exception] = None
    for attempt in range(1, max(1, retries) + 1):
        try:
            return degrader.generate_negative_prompt(
                positive_prompt=positive_prompt,
                subcategory=subcategory,
                attribute=attribute,
                severity=severity,
            )
        except Exception as e:
            last_err = e
            if attempt < retries:
                logger.warning(
                    "[%s/%s] LLM失败 (%s/%s) severity=%s: %s",
                    subcategory,
                    attribute,
                    attempt,
                    retries,
                    severity,
                    e,
                )
                time.sleep(retry_sleep)
            else:
                break
    assert last_err is not None
    raise last_err


def main():
    parser = argparse.ArgumentParser(description="V3维度：LLM prompt退化 + SDXL 正负图像对生成 demo")
    parser.add_argument(
        "--positive_source",
        type=str,
        default="/root/autodl-tmp/AGIQA/data/prompt_sources_workspace/backfill_merge_runs/all_dimensions_v1_full/merged_working_pool_cleaned_v1.jsonl",
        help="正样本来源（image_quality_train.json）",
    )
    parser.add_argument(
        "--template_dir",
        type=str,
        default="/root/ImageReward/data_generation/config/prompt_templates_v3",
        help="v3 system prompt 模板目录",
    )
    parser.add_argument(
        "--quality_dimensions",
        type=str,
        default="/root/ImageReward/data_generation/config/quality_dimensions_active.json",
        help="退化维度元信息（用于 category/description 回填）",
    )
    parser.add_argument(
        "--tagged_prompts",
        type=str,
        default=None,
        help="带语义标签的正样本数据路径（由tag_positive_prompts.py生成）",
    )
    parser.add_argument(
        "--tag_config",
        type=str,
        default="/root/ImageReward/data_generation/config/semantic_tag_requirements.json",
        help="语义标签配置文件路径",
    )
    parser.add_argument(
        "--llm_config",
        type=str,
        default="/root/ImageReward/data_generation/config/llm_config.yaml",
        help="LLM 配置",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/root/autodl-tmp/demo_v3_dimension_paired",
        help="输出目录前缀（会自动追加时间戳）",
    )
    parser.add_argument(
        "--append_timestamp",
        action="store_true",
        default=True,
        help="是否在输出目录末尾追加时间戳（默认追加）",
    )
    parser.add_argument(
        "--no_append_timestamp",
        action="store_false",
        dest="append_timestamp",
        help="不追加时间戳（直接使用 --output_dir 作为最终输出目录）",
    )
    parser.add_argument("--num_prompts_per_dimension", type=int, default=3, help="每个维度选几个正样本")
    parser.add_argument("--seed", type=int, default=42, help="随机种子（选prompt + 图像seed基准）")
    parser.add_argument(
        "--prompt_sampling",
        type=str,
        default="random",
        choices=["topk_random", "random"],
        help="正样本选取策略：topk_random=按gt_score排序取top_k后随机；random=全量随机",
    )
    parser.add_argument(
        "--model_filter",
        type=str,
        default="sdxl",
        help="只从 model 字段匹配该子串的样本里选正样本（例如 'sdxl'）",
    )
    parser.add_argument("--top_k_by_score", type=int, default=20000, help="topk_random 策略的 top_k 截断")
    parser.add_argument("--min_prompt_len", type=int, default=10, help="过滤过短 prompt（字符数）")
    parser.add_argument("--llm_workers", type=int, default=3, help="并行 LLM 请求线程数（建议 3~6）")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="断点续跑：尽量复用已生成的 prompts_cache.json / 图片文件，只补齐缺失项",
    )
    parser.add_argument(
        "--continue_on_error",
        action="store_true",
        default=True,
        help="遇到单个属性/样本失败时继续后续生成（默认开启）",
    )
    parser.add_argument(
        "--no_continue_on_error",
        action="store_false",
        dest="continue_on_error",
        help="遇到错误立即退出",
    )
    parser.add_argument("--llm_retries", type=int, default=5, help="LLM调用失败重试次数")
    parser.add_argument("--llm_retry_sleep", type=float, default=3.0, help="LLM重试间隔（秒）")
    parser.add_argument(
        "--severities",
        type=str,
        default="mild,moderate,severe",
        help="退化程度列表，逗号分隔（如 'moderate,severe'）",
    )
    parser.add_argument(
        "--subcategory_filter",
        type=str,
        default=None,
        help="只处理指定的subcategory，逗号分隔（如 'technical_quality,aesthetic_quality'）",
    )
    parser.add_argument(
        "--attribute_filter",
        type=str,
        default=None,
        help="只处理指定的attribute，逗号分隔（如 'blur,noise'）",
    )
    parser.add_argument(
        "--seed_strategy",
        type=str,
        default="legacy_global",
        choices=["legacy_global", "stable_hash"],
        help="seed 生成策略：legacy_global=全局递增(兼容旧输出)；stable_hash=按(subcat/attr/prompt)哈希稳定生成",
    )

    # 和 run_data_generate.sh 保持一致
    parser.add_argument("--steps", type=int, default=40, help="SDXL推理步数")
    parser.add_argument("--cfg", type=float, default=7.5, help="SDXL CFG scale")
    parser.add_argument("--width", type=int, default=1024, help="生成分辨率宽")
    parser.add_argument("--height", type=int, default=1024, help="生成分辨率高")
    parser.add_argument(
        "--model_path",
        type=str,
        default="/root/ckpts/sd_xl_base_1.0.safetensors",
        help="模型权重路径",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="sdxl",
        choices=["sdxl", "flux", "flux-schnell", "hunyuan-dit", "sd3.5-large", "qwen-image-lightning"],
        help="使用的生成模型 ID (sdxl, flux, flux-schnell, hunyuan-dit, sd3.5-large 或 qwen-image-lightning)",
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="启用速度优化模式（仅 flux-schnell，T5 4-bit + FP8，需要 ~17GB 显存）",
    )
    parser.add_argument(
        "--use_cpu_offload",
        action="store_true",
        help="启用 CPU offload（推荐仅在 SD3.5 Large 显存不足时使用）",
    )
    args = parser.parse_args()

    template_dir = Path(args.template_dir)
    if not template_dir.exists():
        raise FileNotFoundError(f"template_dir 不存在: {template_dir}")

    subcat_to_attrs = load_subcategory_attributes_from_templates(template_dir)
    if not subcat_to_attrs:
        raise ValueError(f"未从模板目录解析到任何 subcategory/attribute: {template_dir}")

    # 解析 severities 参数
    severities = [s.strip() for s in args.severities.split(",") if s.strip()]
    if not severities:
        severities = SEVERITIES
    logger.info(f"使用退化程度: {severities}")

    # 按 subcategory 过滤
    if args.subcategory_filter:
        filter_set = set(s.strip() for s in args.subcategory_filter.split(",") if s.strip())
        subcat_to_attrs = {k: v for k, v in subcat_to_attrs.items() if k in filter_set}
        if not subcat_to_attrs:
            raise ValueError(f"过滤后无有效 subcategory: {filter_set}")
        logger.info(f"过滤后的 subcategory: {list(subcat_to_attrs.keys())}")

    # 按 attribute 过滤
    if args.attribute_filter:
        attr_filter_set = set(s.strip() for s in args.attribute_filter.split(",") if s.strip())
        for subcat in list(subcat_to_attrs.keys()):
            filtered_attrs = [a for a in subcat_to_attrs[subcat] if a in attr_filter_set]
            if filtered_attrs:
                subcat_to_attrs[subcat] = filtered_attrs
            else:
                del subcat_to_attrs[subcat]
        if not subcat_to_attrs:
            raise ValueError(f"过滤后无有效 attribute: {attr_filter_set}")
        logger.info(f"过滤后的 attribute: {attr_filter_set}")

    logger.info(
        "解析到 %d 个 subcategory，共 %d 个 attribute",
        len(subcat_to_attrs),
        sum(len(v) for v in subcat_to_attrs.values()),
    )

    # 构建候选正样本池（然后每个 subcategory/attribute 各自随机抽样 N 个）
    if args.prompt_sampling == "topk_random":
        pos_items = load_positive_prompts_from_image_quality_train(
            Path(args.positive_source),
            min_prompt_len=args.min_prompt_len,
            top_k_by_score=args.top_k_by_score,
            model_substring=args.model_filter or None,
        )
        prompt_pool = [p for p, _s in pos_items]
        logger.info(f"正样本池: topk_random (top_k={args.top_k_by_score}) -> 候选 {len(prompt_pool)}")
    else:
        with open(args.positive_source, "r", encoding="utf-8") as f:
            all_rows = json.load(f)
        prompt_pool = []
        for row in all_rows:
            model = row.get("model", "")
            if args.model_filter:
                if not isinstance(model, str) or args.model_filter.lower() not in model.lower():
                    continue
            p = row.get("prompt")
            if isinstance(p, str) and p.strip() and len(p.strip()) >= args.min_prompt_len:
                prompt_pool.append(p.strip())
        logger.info(f"正样本池: random (model_filter={args.model_filter!r}) -> 候选 {len(prompt_pool)}")

    # 加载标签数据（用于语义过滤）
    prompt_to_tags: Dict[str, List[str]] = {}
    tag_requirements: Dict[str, Dict] = {}
    if args.tagged_prompts:
        tagged_path = Path(args.tagged_prompts)
        if tagged_path.exists():
            prompt_to_tags = load_tagged_prompts(tagged_path)
            logger.info(f"加载标签数据: {len(prompt_to_tags)} 条带标签的 prompt")
        else:
            logger.warning(f"标签数据文件不存在: {tagged_path}，将跳过标签过滤")

    tag_requirements = {}
    if args.tag_config:
        try:
            raw_reqs = json.load(open(args.tag_config, "r", encoding="utf-8"))
            # 支持新版结构，取 'dimension_requirements'
            dim_reqs = raw_reqs.get("dimension_requirements", raw_reqs) if isinstance(raw_reqs, dict) and "dimension_requirements" in raw_reqs else raw_reqs
            
            # 扁平化 tag_requirements: {category: {attr: req}} -> {attr: req}
            # 假设 attribute 名称在整个配置中是唯一的
            for cat, attrs in dim_reqs.items():
                if isinstance(attrs, dict):
                    for attr, req in attrs.items():
                        tag_requirements[attr] = req
            logger.info(f"加载标签配置: {len(tag_requirements)} 个维度的标签要求")
        except Exception as e:
            logger.warning(f"加载 tag_config 失败: {e}，将不使用标签过滤")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 初始化 LLM 退化器（会自动读取 prompt_templates_v3）
    degrader = LLMPromptDegradation(
        llm_config_path=args.llm_config,
        quality_dimensions_path=args.quality_dimensions,
    )

    # 初始化图像生成器
    if args.model_id == "flux":
        logger.info(f"正在初始化 Flux.1-dev 生成器: {args.model_path}")
        sdxl = FluxGenerator(model_path=args.model_path)
    elif args.model_id == "flux-schnell":
        logger.info("正在初始化 Flux.1-schnell 生成器")
        sdxl = FluxSchnellGenerator(optimize_for_speed=args.optimize, enable_compile=args.optimize)
    elif args.model_id == "hunyuan-dit":
        logger.info(f"正在初始化 Hunyuan-DiT 生成器: {args.model_path}")
        sdxl = HunyuanDiTGenerator(model_path=args.model_path, use_cpu_offload=args.use_cpu_offload)
    elif args.model_id == "sd3.5-large":
        logger.info(f"正在初始化 SD3.5 Large 生成器: {args.model_path}")
        sdxl = SD35LargeGenerator(model_path=args.model_path, use_cpu_offload=args.use_cpu_offload)
    elif args.model_id == "qwen-image-lightning":
        logger.info(f"正在初始化 Qwen-Image-Lightning 生成器: {args.model_path}")
        sdxl = QwenImageLightningGenerator(model_path=args.model_path, use_low_mem=args.use_cpu_offload)
    else:
        logger.info(f"正在初始化 SDXL 生成器: {args.model_path}")
        sdxl = SDXLGenerator(model_path=args.model_path)

    llm_workers = max(1, int(args.llm_workers))

    # 每个 (subcategory/attribute) 单独输出（属性内图片/JSON在一起，不同属性分开）
    base_seed = args.seed
    expected_pairs_per_attr = args.num_prompts_per_dimension * len(severities)
    global_pair_id = 0
    global_positive_seed_counter = 0

    with ThreadPoolExecutor(max_workers=llm_workers) as llm_pool:
        for subcategory, attributes in subcat_to_attrs.items():
            for attribute in attributes:
                attr_dir = out_dir / subcategory / f"{attribute}_{timestamp}" if args.append_timestamp else out_dir / subcategory / attribute
                images_dir = attr_dir / "images"
                images_dir.mkdir(parents=True, exist_ok=True)

                key = f"{subcategory}::{attribute}"
                key_hash = stable_u32(key)

                # 按标签要求过滤正样本池
                if prompt_to_tags and tag_requirements:
                    filtered_pool = filter_prompts_by_tags(
                        prompt_pool, prompt_to_tags, attribute, tag_requirements
                    )
                    if len(filtered_pool) < args.num_prompts_per_dimension:
                        logger.warning(
                            f"[{subcategory}/{attribute}] 过滤后正样本不足: "
                            f"需要 {args.num_prompts_per_dimension}, 可用 {len(filtered_pool)}"
                        )
                        if len(filtered_pool) == 0:
                            logger.error(f"[{subcategory}/{attribute}] 无符合标签要求的正样本，跳过")
                            continue
                else:
                    filtered_pool = prompt_pool

                try:
                    # 使用时间戳确保每次运行随机抽取不同的正样本
                    random_seed = int(time.time() * 1000) % (2**31) + key_hash
                    attr_prompts = sample_unique_prompts_from_all(
                        filtered_pool,
                        total_needed=min(args.num_prompts_per_dimension, len(filtered_pool)),
                        seed=random_seed,
                    )
                except ValueError as e:
                    logger.error(f"[{subcategory}/{attribute}] 采样失败: {e}，跳过")
                    continue
                logger.info(f"[{subcategory}/{attribute}] 选取正样本 {len(attr_prompts)} 条")

                dataset_path = attr_dir / "dataset.json"
                cache_path = attr_dir / "prompts_cache.json"

                if args.resume and dataset_path.exists() and cache_path.exists():
                    try:
                        existing_dataset = json.load(open(dataset_path, "r", encoding="utf-8"))
                        existing_pairs = existing_dataset.get("pairs", [])
                        existing_cache = json.load(open(cache_path, "r", encoding="utf-8"))
                        actual_expected_pairs = len(attr_prompts) * len(severities)
                        if (
                            isinstance(existing_pairs, list)
                            and isinstance(existing_cache, list)
                            and len(existing_pairs) == actual_expected_pairs
                            and len(existing_cache) == actual_expected_pairs
                        ):
                            # 进一步检查文件是否都在，避免“JSON写了但图片缺失”
                            all_ok = True
                            for idx, p in enumerate(attr_prompts):
                                if args.seed_strategy == "stable_hash":
                                    pos_seed = base_seed + (stable_u32(f"{key}::pos::{p}") % 1_000_000_000)
                                else:
                                    pos_seed = base_seed + (global_positive_seed_counter + idx)
                                if not (images_dir / f"positive_{pos_seed}.png").exists():
                                    all_ok = False
                                    break
                                for sev in severities:
                                    if not (images_dir / f"negative_{pos_seed}_{sev}.png").exists():
                                        all_ok = False
                                        break
                                if not all_ok:
                                    break
                            if all_ok:
                                logger.info(f"[{subcategory}/{attribute}] resume: 已完成，跳过")
                                if args.seed_strategy == "legacy_global":
                                    global_positive_seed_counter += len(attr_prompts)
                                    global_pair_id += len(attr_prompts) * len(severities)
                                continue
                    except Exception as e:
                        logger.warning(f"[{subcategory}/{attribute}] resume: 读取已存在数据失败，将尝试补齐: {e}")

                # 加载已有 prompts_cache，用于复用负样本 prompt（避免断点续跑时重新请求 LLM）
                existing_map: Dict[Tuple[str, str], Dict[str, Any]] = {}
                if args.resume and cache_path.exists():
                    try:
                        existing_cache = json.load(open(cache_path, "r", encoding="utf-8"))
                        if isinstance(existing_cache, list):
                            for e in existing_cache:
                                if not isinstance(e, dict):
                                    continue
                                pp = e.get("positive_prompt")
                                sev = e.get("severity")
                                np = e.get("negative_prompt")
                                if isinstance(pp, str) and isinstance(sev, str) and isinstance(np, str):
                                    existing_map[(pp, sev)] = e
                            logger.info(f"[{subcategory}/{attribute}] resume: 复用 {len(existing_map)} 条 cached 负样本 prompt")
                    except Exception as e:
                        logger.warning(f"[{subcategory}/{attribute}] resume: 读取 prompts_cache.json 失败，将重新生成: {e}")

                prompts_cache: List[Dict] = []
                pairs: List[Dict] = []
                positive_image_cache: Dict[str, str] = {}  # prompt -> relpath (within attr_dir)

                for p in attr_prompts:
                    if args.seed_strategy == "stable_hash":
                        pos_seed = base_seed + (stable_u32(f"{key}::pos::{p}") % 1_000_000_000)
                    else:
                        pos_seed = base_seed + global_positive_seed_counter
                        global_positive_seed_counter += 1

                    # 针对缺失项并行触发 LLM 退化（I/O 可与 GPU 生成并行）
                    severity_futures = {}
                    for severity in severities:
                        cached = existing_map.get((p, severity))
                        if cached and isinstance(cached.get("negative_prompt"), str) and cached["negative_prompt"].strip():
                            continue
                        severity_futures[severity] = llm_pool.submit(
                            call_llm_with_retry,
                            degrader,
                            positive_prompt=p,
                            subcategory=subcategory,
                            attribute=attribute,
                            severity=severity,
                            retries=int(args.llm_retries),
                            retry_sleep=float(args.llm_retry_sleep),
                        )

                    if p not in positive_image_cache:
                        pos_path = images_dir / f"positive_{pos_seed}.png"
                        if args.resume and pos_path.exists():
                            pos_info = {"seed": pos_seed, "reused": True, "reused_from_disk": True}
                            positive_image_cache[p] = str(pos_path.relative_to(attr_dir))
                        else:
                            try:
                                pos_img, pos_info = sdxl.generate(
                                    prompt=p,
                                    negative_prompt="",
                                    num_inference_steps=args.steps,
                                    guidance_scale=args.cfg,
                                    width=args.width,
                                    height=args.height,
                                    seed=pos_seed,
                                )
                                pos_img.save(pos_path)
                                positive_image_cache[p] = str(pos_path.relative_to(attr_dir))
                                logger.info(f"[{subcategory}/{attribute}] 保存正样本: {pos_path}")
                            except Exception as e:
                                logger.error(f"[{subcategory}/{attribute}] 正样本生成失败: {e}")
                                if args.continue_on_error:
                                    continue
                                raise
                    else:
                        pos_info = {"seed": pos_seed, "reused": True}

                    for severity in severities:
                        cached = existing_map.get((p, severity))
                        if cached and isinstance(cached.get("negative_prompt"), str) and cached["negative_prompt"].strip():
                            neg_prompt = cached["negative_prompt"]
                            deg_info = cached.get("degradation_info") or cached.get("degradation") or {}
                        else:
                            try:
                                neg_prompt, deg_info = severity_futures[severity].result()
                            except Exception as e:
                                logger.error(f"[{subcategory}/{attribute}] LLM生成失败 severity={severity}: {e}")
                                if args.continue_on_error:
                                    continue
                                raise

                        neg_seed = pos_seed  # 正负样本使用相同seed，确保唯一变量是prompt差异
                        neg_path = images_dir / f"negative_{pos_seed}_{severity}.png"
                        if args.resume and neg_path.exists():
                            neg_info = {"seed": neg_seed, "reused": True, "reused_from_disk": True}
                        else:
                            try:
                                neg_img, neg_info = sdxl.generate(
                                    prompt=neg_prompt,
                                    negative_prompt="",
                                    num_inference_steps=args.steps,
                                    guidance_scale=args.cfg,
                                    width=args.width,
                                    height=args.height,
                                    seed=neg_seed,
                                )
                                neg_img.save(neg_path)
                                logger.info(f"[{subcategory}/{attribute}] 保存负样本: {neg_path}")
                            except Exception as e:
                                logger.error(f"[{subcategory}/{attribute}] 负样本生成失败 severity={severity}: {e}")
                                if args.continue_on_error:
                                    continue
                                raise

                        prompts_cache.append(
                            {
                                "subcategory": subcategory,
                                "attribute": attribute,
                                "severity": severity,
                                "positive_prompt": p,
                                "negative_prompt": neg_prompt,
                                "degradation_info": deg_info,
                                "positive_seed": pos_seed,
                                "negative_seed": neg_seed,
                            }
                        )

                        if args.seed_strategy == "stable_hash":
                            pair_id = hashlib.md5(f"{key}::{p}::{severity}".encode("utf-8")).hexdigest()[:12]
                        else:
                            pair_id = f"{global_pair_id:07d}"
                        pairs.append(
                            {
                                "pair_id": pair_id,
                                "subcategory": subcategory,
                                "attribute": attribute,
                                "severity": severity,
                                "positive": {
                                    "prompt": p,
                                    "image_path": positive_image_cache[p],
                                    "seed": pos_seed,
                                },
                                "negative": {
                                    "prompt": neg_prompt,
                                    "image_path": str(neg_path.relative_to(attr_dir)),
                                    "seed": neg_seed,
                                },
                                "degradation": deg_info,
                                "generation_info": {
                                    "positive": pos_info,
                                    "negative": neg_info,
                                },
                            }
                        )
                        if args.seed_strategy == "legacy_global":
                            global_pair_id += 1

                with open(attr_dir / "prompts_cache.json", "w", encoding="utf-8") as f:
                    json.dump(prompts_cache, f, ensure_ascii=False, indent=2)

                dataset = {
                    "metadata": {
                        "generated_at": datetime.now().isoformat(),
                        "positive_source": args.positive_source,
                        "template_dir": str(template_dir),
                        "subcategory": subcategory,
                        "attribute": attribute,
                        "num_prompts": len(attr_prompts),
                        "severities": severities,
                        "prompt_sampling": args.prompt_sampling,
                        "model_filter": args.model_filter,
                        "total_pairs": len(pairs),
                        "total_positive_images": len(set(positive_image_cache.values())),
                        "total_negative_images": len(pairs),
                        "sdxl_params": {
                            "steps": args.steps,
                            "cfg": args.cfg,
                            "width": args.width,
                            "height": args.height,
                            "model_path": args.model_path,
                        },
                        "seed": base_seed,
                        "llm_workers": llm_workers,
                    },
                    "pairs": pairs,
                }

                with open(attr_dir / "dataset.json", "w", encoding="utf-8") as f:
                    json.dump(dataset, f, ensure_ascii=False, indent=2)

    logger.info(f"完成：输出目录 {out_dir}")


if __name__ == "__main__":
    main()
