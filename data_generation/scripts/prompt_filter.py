#!/usr/bin/env python3
"""
Prompt Filter (Stage 1)
从 demo_v3_dimension_paired.py 提取的正样本-维度兼容性过滤模块。

基于预标签化系统 (tag_positive_prompts.py + semantic_tag_requirements.json)，
在生成前主动判断 (prompt, dimension) 是否兼容，避免无效生成。

可独立运行测试:
    python scripts/prompt_filter.py --test hand_malformation
    python scripts/prompt_filter.py --test text_error
    python scripts/prompt_filter.py --stats
"""

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_tagged_prompts(tagged_path: Path) -> Dict[str, List[str]]:
    """
    加载预标签化的 prompt → tags 映射。

    Args:
        tagged_path: 带语义标签的正样本数据路径 (e.g. prompts_tagged_sdxl_v2.json)

    Returns:
        { prompt_text: [tag1, tag2, ...] }
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
    扁平化处理: {category: {attr: req}} → {attr: req}

    Args:
        tag_config_path: semantic_tag_requirements.json 路径

    Returns:
        { dimension_name: {"required": [...], "alternative": [...], "preferred": [...]} | None }
    """
    with open(tag_config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    dim_reqs_raw = config.get("dimension_requirements", config)
    result: Dict[str, Dict] = {}

    for category, dims in dim_reqs_raw.items():
        if category.startswith("_"):
            continue
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


# 预编译的正则：维度特殊过滤规则
_VISUAL_INDICATORS_PATTERN = re.compile(
    r"\b(sign|text|written|label|board|banner|poster|reading|says|inscribed|logo|marked|branded|printed|stamp)\b"
)
_NEGATIVE_VIEW_PATTERN = re.compile(
    r"\b(back view|from behind|facing away|rear view|walking away)\b"
)
_ANIMAL_PATTERN = re.compile(
    r"\b(cat|dog|tiger|lion|bear|wolf|fox|animal|creature|robot|statue|monkey|ape|bird|capybara|rodent|fish|insect|beast|monster)\b"
)
_STRONG_FACE_PATTERN = re.compile(
    r"\b(face|eyes|mouth|nose|portrait|close-up|headshot|looking at viewer|looking at camera|smile|laugh|cry|expression)\b"
)
_HUMAN_PATTERN = re.compile(
    r"\b(man|woman|boy|girl|child|kid|person|people|human|guy|lady|men|women|male|female)\b"
)


def _check_dimension_special_rules(prompt: str, dimension: str) -> Tuple[bool, str]:
    """
    检查维度特殊过滤规则（硬编码规则，从 demo_v3 提取）。

    Returns:
        (passed, reason): 是否通过, 不通过时的原因
    """
    lower_p = prompt.lower()

    if dimension == "text_error":
        has_quote = "'" in prompt or '"' in prompt
        has_indicator = bool(_VISUAL_INDICATORS_PATTERN.search(lower_p))
        if not (has_quote and has_indicator):
            return False, "text_error: 缺少引号或视觉文字指示词"

    if dimension in ("face_asymmetry", "extra_limbs"):
        if _NEGATIVE_VIEW_PATTERN.search(lower_p):
            return False, f"{dimension}: 包含背影/远景关键词"
        if _ANIMAL_PATTERN.search(lower_p):
            return False, f"{dimension}: 包含动物/非人关键词"
        if not (_STRONG_FACE_PATTERN.search(lower_p) and _HUMAN_PATTERN.search(lower_p)):
            return False, f"{dimension}: 缺少明确人脸+人类关键词"

    return True, ""


def is_prompt_compatible(
    prompt: str,
    tags: List[str],
    dimension: str,
    tag_requirements: Dict[str, Dict],
) -> Tuple[bool, str]:
    """
    单条 prompt 兼容性检查（pipeline 运行时使用）。

    Args:
        prompt: 正样本文本
        tags: 该 prompt 的语义标签列表
        dimension: 目标维度名
        tag_requirements: 维度标签要求字典

    Returns:
        (compatible, reason): 是否兼容, 不兼容时的原因
    """
    req = tag_requirements.get(dimension)

    # 无要求 → 兼容
    if req is None or req.get("required") is None:
        # 仍需过特殊规则
        return _check_dimension_special_rules(prompt, dimension)

    required_tags = set(req.get("required") or [])
    alternative_tags = set(req.get("alternative") or [])

    if not required_tags and not alternative_tags:
        return _check_dimension_special_rules(prompt, dimension)

    tag_set = set(tags)

    has_all_required = required_tags.issubset(tag_set) if required_tags else True
    has_any_alternative = bool(tag_set & alternative_tags) if alternative_tags else False

    if not (has_all_required or has_any_alternative):
        missing = required_tags - tag_set
        return False, f"标签不满足: 缺少 {missing}"

    return _check_dimension_special_rules(prompt, dimension)


def filter_prompts_by_tags(
    prompt_pool: List[str],
    prompt_to_tags: Dict[str, List[str]],
    dimension_name: str,
    tag_requirements: Dict[str, Dict],
) -> List[str]:
    """
    根据维度的标签要求批量过滤正样本。

    Args:
        prompt_pool: 候选正样本列表
        prompt_to_tags: prompt → tags 映射
        dimension_name: 目标维度
        tag_requirements: 维度标签要求字典

    Returns:
        过滤后的兼容 prompt 列表
    """
    req = tag_requirements.get(dimension_name)
    if req is None or req.get("required") is None:
        # 无标签要求，仍检查特殊规则
        filtered = []
        for p in prompt_pool:
            passed, _ = _check_dimension_special_rules(p, dimension_name)
            if passed:
                filtered.append(p)
        if len(filtered) < len(prompt_pool):
            logger.info(
                f"[{dimension_name}] 特殊规则过滤: {len(prompt_pool)} → {len(filtered)}"
            )
        return filtered

    required_tags = set(req.get("required") or [])
    alternative_tags = set(req.get("alternative") or [])

    if not required_tags and not alternative_tags:
        return prompt_pool

    filtered = []
    skipped_no_tag = 0
    skipped_tags = 0
    skipped_special = 0

    for p in prompt_pool:
        if p not in prompt_to_tags:
            skipped_no_tag += 1
            continue

        tags = set(prompt_to_tags.get(p, []))
        has_all_required = required_tags.issubset(tags) if required_tags else True
        has_any_alternative = bool(tags & alternative_tags) if alternative_tags else False

        if not (has_all_required or has_any_alternative):
            skipped_tags += 1
            continue

        passed, _ = _check_dimension_special_rules(p, dimension_name)
        if not passed:
            skipped_special += 1
            continue

        filtered.append(p)

    if skipped_no_tag > 0 or skipped_tags > 0 or skipped_special > 0:
        logger.info(
            f"[{dimension_name}] 标签过滤: 通过 {len(filtered)}, "
            f"无标签 {skipped_no_tag}, 标签不符 {skipped_tags}, 特殊规则 {skipped_special}"
        )

    return filtered


def _run_test(dimension: str, tagged_path: Path, tag_config_path: Path):
    """对指定维度执行筛选测试"""
    prompt_to_tags = load_tagged_prompts(tagged_path)
    tag_requirements = load_tag_requirements(tag_config_path)
    prompt_pool = list(prompt_to_tags.keys())

    print(f"\n=== 测试维度: {dimension} ===")
    print(f"总 prompt 数: {len(prompt_pool)}")

    filtered = filter_prompts_by_tags(
        prompt_pool, prompt_to_tags, dimension, tag_requirements
    )

    print(f"兼容 prompt 数: {len(filtered)}")
    if filtered:
        print(f"示例 (前3条):")
        for p in filtered[:3]:
            tags = prompt_to_tags.get(p, [])
            print(f"  [{', '.join(tags)}] {p[:80]}...")


def _run_stats(tagged_path: Path, tag_config_path: Path, dimensions_path: Path = None):
    """统计所有维度的兼容 prompt 数量"""
    prompt_to_tags = load_tagged_prompts(tagged_path)
    tag_requirements = load_tag_requirements(tag_config_path)
    prompt_pool = list(prompt_to_tags.keys())

    # 收集所有维度名
    all_dims = list(tag_requirements.keys())
    if dimensions_path and dimensions_path.exists():
        with open(dimensions_path, "r", encoding="utf-8") as f:
            dim_config = json.load(f)
        for persp_data in dim_config.get("perspectives", {}).values():
            for dim_name in persp_data.get("dimensions", {}).keys():
                if dim_name not in all_dims:
                    all_dims.append(dim_name)

    print(f"\n=== 维度兼容性统计 ===")
    print(f"总 prompt 数: {len(prompt_pool)}")
    print(f"{'维度':<30} {'兼容数':>8} {'比例':>8}")
    print("-" * 50)

    for dim in sorted(all_dims):
        filtered = filter_prompts_by_tags(
            prompt_pool, prompt_to_tags, dim, tag_requirements
        )
        ratio = len(filtered) / len(prompt_pool) * 100 if prompt_pool else 0
        print(f"{dim:<30} {len(filtered):>8} {ratio:>7.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prompt Filter (Stage 1) 独立测试")
    parser.add_argument("--test", type=str, default=None, help="测试指定维度的过滤效果")
    parser.add_argument("--stats", action="store_true", help="统计所有维度的兼容 prompt 数量")
    parser.add_argument(
        "--tagged_prompts",
        type=str,
        default="/root/ImageReward/data_generation/data/prompts_tagged_sdxl_v2.json",
    )
    parser.add_argument(
        "--tag_config",
        type=str,
        default="/root/ImageReward/data_generation/config/semantic_tag_requirements.json",
    )
    parser.add_argument(
        "--quality_dimensions",
        type=str,
        default="/root/ImageReward/data_generation/config/quality_dimensions_v3.json",
    )
    args = parser.parse_args()

    tagged = Path(args.tagged_prompts)
    tag_cfg = Path(args.tag_config)
    dim_cfg = Path(args.quality_dimensions)

    if args.test:
        _run_test(args.test, tagged, tag_cfg)
    elif args.stats:
        _run_stats(tagged, tag_cfg, dim_cfg)
    else:
        print("用法:")
        print("  python scripts/prompt_filter.py --test hand_malformation")
        print("  python scripts/prompt_filter.py --stats")
