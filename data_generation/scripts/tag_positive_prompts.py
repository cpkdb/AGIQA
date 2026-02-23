#!/usr/bin/env python3
"""
正样本Prompt语义标注脚本

根据semantic_tag_requirements.json中定义的标签体系，
使用LLM对正样本prompt进行批量语义标注。

输出带标签的正样本数据，供后续维度筛选使用。
"""

import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Set, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time
import re

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# LLM标注的System Prompt
TAGGING_SYSTEM_PROMPT = """You are an expert at analyzing text-to-image prompts. Your task is to identify what semantic elements will be present in the generated image.

Given a prompt, output ONLY the matching tags from this list (comma-separated, no explanations):

AVAILABLE TAGS:
- has_person: Image will contain human figure(s) - full body, half body, or partial
- has_face: Image will contain clearly visible human face(s)
- has_hand: Image will contain clearly visible hand(s) - including hands holding objects
- has_full_body: Image will contain complete human body (head to toe visible)
- has_animal: Image will contain animal(s) - real or anthropomorphic
- has_multiple_objects: Image will contain multiple distinct/separable objects
- has_countable_objects: Image contains specific countable items (two apples, three birds, etc.)
- has_indoor_scene: Image is an indoor scene (room, building interior)
- has_background: Image has clear foreground/background separation
- has_text: Image will contain text, signs, books, posters, labels, etc.
- has_reflective_surface: Image contains mirrors, water, glass, or other reflective surfaces
- has_logo_or_symbol: Image contains logos, brand marks, icons, or symbols
- has_deformable_object: Image contains objects with clear structural features that can be deformed (chair, bench, table, desk, shelf, ladder, fence, pole, mug, cup, bicycle, cart, car, vehicle, door, window, sofa, bed, cabinet, bookshelf, lamppost, railing)
- has_shape_rigid_object: Image contains objects with fixed natural shapes (wheel, tire, ball, globe, coin, plate, dish, pizza, donut, clock, sun, moon, egg, orange, apple, mirror, book)

RULES:
1. Only output tags that are CLEARLY implied by the prompt
2. If a tag is uncertain, do NOT include it
3. Output format: tag1, tag2, tag3 (or "none" if no tags match)
4. For "has_person", include if prompt mentions: person, man, woman, boy, girl, portrait, figure, people, someone, etc.
5. For "has_hand", include if: hands are explicitly mentioned OR person is holding/grabbing/pointing at something
6. For "has_face", include if: face, portrait, headshot, close-up of person, expression, etc.
7. For "has_deformable_object", include if: prompt contains furniture (chair, table, bench, desk, shelf, sofa, bed), containers (mug, cup, bottle), vehicles (car, bicycle, cart), or structural objects (fence, pole, ladder, door, window)
8. For "has_shape_rigid_object", include if: prompt contains objects with inherently fixed shapes like wheels, balls, plates, coins, eggs, clock, sun, moon, pizza

Examples:
- "a beautiful woman holding a red rose" → has_person, has_hand, has_multiple_objects
- "a cat sleeping on a couch in a living room" → has_animal, has_indoor_scene, has_background, has_deformable_object
- "portrait of an old man with wrinkled face" → has_person, has_face
- "a sunset over the ocean" → has_background
- "two birds sitting on a branch" → has_animal, has_countable_objects, has_multiple_objects
- "a coffee shop with menu board on the wall" → has_indoor_scene, has_text, has_background
- "a wooden chair in the corner of the room" → has_indoor_scene, has_deformable_object
- "a child kicking a soccer ball" → has_person, has_full_body, has_shape_rigid_object
- "breakfast with eggs on a plate" → has_shape_rigid_object, has_multiple_objects
"""


def load_tag_config(config_path: str) -> Dict:
    """加载标签配置"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_positive_prompts(source_path: str) -> List[Dict]:
    """加载正样本数据"""
    with open(source_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 支持多种格式
    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and 'data' in data:
        return data['data']
    else:
        raise ValueError(f"Unsupported data format in {source_path}")


def create_llm_client():
    """创建LLM客户端"""
    try:
        from openai import OpenAI
        import yaml

        # 尝试从配置文件读取
        config_path = Path(__file__).parent.parent / "config" / "llm_config.yaml"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)

            llm_config = config.get('llm', {})
            return OpenAI(
                api_key=llm_config.get('api_key'),
                base_url=llm_config.get('api_base')
            ), llm_config.get('model', 'gpt-4o')
        else:
            # 使用环境变量
            import os
            return OpenAI(api_key=os.getenv('OPENAI_API_KEY')), 'gpt-4o'
    except Exception as e:
        logger.error(f"Failed to create LLM client: {e}")
        raise


def tag_prompt_with_llm(
    client,
    model: str,
    prompt: str,
    valid_tags: Set[str],
    max_retries: int = 3,
    retry_delay: float = 2.0
) -> List[str]:
    """使用LLM对单个prompt进行标注"""

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": TAGGING_SYSTEM_PROMPT},
                    {"role": "user", "content": f"Prompt: {prompt}"}
                ],
                temperature=0.1,
                max_tokens=100
            )

            result = response.choices[0].message.content.strip().lower()

            # 解析返回的标签
            if result == "none" or not result:
                return []

            # 提取有效标签
            tags = []
            for tag in re.split(r'[,\s]+', result):
                tag = tag.strip()
                if tag in valid_tags:
                    tags.append(tag)

            return list(set(tags))  # 去重

        except Exception as e:
            logger.warning(f"LLM call failed (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                logger.error(f"Failed to tag prompt after {max_retries} attempts: {prompt[:50]}...")
                return []

    return []


def tag_prompt_with_keywords(prompt: str, tag_config: Dict) -> List[str]:
    """使用关键词匹配进行快速标注（作为LLM的补充或备选）"""
    prompt_lower = prompt.lower()
    tags = []

    for tag_name, tag_info in tag_config['tags'].items():
        # 特殊处理：has_quoted_text 使用正则检测引号内文字
        if tag_name == 'has_quoted_text':
            # 检测 '...' 或 "..." 模式
            if re.search(r'[\'"][^\'"]{2,}[\'"]', prompt):
                tags.append(tag_name)
            continue

        keywords = tag_info.get('detection_keywords', [])
        for keyword in keywords:
            if keyword.lower() in prompt_lower:
                tags.append(tag_name)
                break

    return list(set(tags))


def batch_tag_prompts(
    prompts: List[Dict],
    tag_config: Dict,
    use_llm: bool = True,
    llm_workers: int = 5,
    use_keywords_fallback: bool = True
) -> List[Dict]:
    """批量标注prompts"""

    valid_tags = set(tag_config['tags'].keys())
    results = []

    if use_llm:
        client, model = create_llm_client()
        logger.info(f"Using LLM model: {model} with {llm_workers} workers")

        # 并行处理
        with ThreadPoolExecutor(max_workers=llm_workers) as executor:
            future_to_idx = {}

            for idx, item in enumerate(prompts):
                prompt_text = item.get('prompt', item.get('text', ''))
                future = executor.submit(
                    tag_prompt_with_llm,
                    client, model, prompt_text, valid_tags
                )
                future_to_idx[future] = idx

            # 收集结果
            tagged_results = [None] * len(prompts)

            for future in tqdm(as_completed(future_to_idx), total=len(prompts), desc="Tagging prompts"):
                idx = future_to_idx[future]
                item = prompts[idx].copy()

                try:
                    llm_tags = future.result()
                except Exception as e:
                    logger.error(f"Error processing prompt {idx}: {e}")
                    llm_tags = []

                # 可选：使用关键词补充
                if use_keywords_fallback:
                    prompt_text = item.get('prompt', item.get('text', ''))
                    keyword_tags = tag_prompt_with_keywords(prompt_text, tag_config)
                    # 合并LLM和关键词结果
                    all_tags = list(set(llm_tags + keyword_tags))
                else:
                    all_tags = llm_tags

                item['semantic_tags'] = all_tags
                tagged_results[idx] = item

            results = tagged_results
    else:
        # 仅使用关键词匹配
        logger.info("Using keyword-based tagging only")
        for item in tqdm(prompts, desc="Tagging prompts"):
            item = item.copy()
            prompt_text = item.get('prompt', item.get('text', ''))
            item['semantic_tags'] = tag_prompt_with_keywords(prompt_text, tag_config)
            results.append(item)

    return results


def compute_tag_statistics(tagged_prompts: List[Dict], tag_config: Dict) -> Dict:
    """计算标签统计信息"""
    stats = {tag: 0 for tag in tag_config['tags'].keys()}
    total = len(tagged_prompts)

    for item in tagged_prompts:
        for tag in item.get('semantic_tags', []):
            if tag in stats:
                stats[tag] += 1

    # 计算百分比
    stats_with_pct = {}
    for tag, count in stats.items():
        stats_with_pct[tag] = {
            "count": count,
            "percentage": round(count / total * 100, 2) if total > 0 else 0
        }

    return {
        "total_prompts": total,
        "tag_distribution": stats_with_pct
    }


def check_dimension_coverage(tagged_prompts: List[Dict], tag_config: Dict) -> Dict:
    """检查各维度的可用正样本覆盖情况"""
    dim_requirements = tag_config.get('dimension_requirements', {})
    coverage = {}

    total = len(tagged_prompts)

    for perspective, dims in dim_requirements.items():
        if not isinstance(dims, dict):
            continue
        coverage[perspective] = {}

        for dim_name, req in dims.items():
            if req is None:
                # 无要求，所有样本都可用
                coverage[perspective][dim_name] = {
                    "available": total,
                    "percentage": 100.0,
                    "requirement": "none"
                }
            elif isinstance(req, dict):
                required_tags = req.get('required', [])
                alternative_tags = req.get('alternative', [])

                if not required_tags:
                    coverage[perspective][dim_name] = {
                        "available": total,
                        "percentage": 100.0,
                        "requirement": "preferred_only"
                    }
                else:
                    # 统计满足要求的样本数
                    count = 0
                    for item in tagged_prompts:
                        item_tags = set(item.get('semantic_tags', []))
                        # 检查必需标签或备选标签
                        if any(tag in item_tags for tag in required_tags):
                            count += 1
                        elif alternative_tags and any(tag in item_tags for tag in alternative_tags):
                            count += 1

                    coverage[perspective][dim_name] = {
                        "available": count,
                        "percentage": round(count / total * 100, 2) if total > 0 else 0,
                        "requirement": f"required: {required_tags}" + (f", alt: {alternative_tags}" if alternative_tags else "")
                    }

    return coverage


def main():
    parser = argparse.ArgumentParser(description="正样本Prompt语义标注")
    parser.add_argument(
        "--source",
        type=str,
        default="/root/ImageReward/data_generation/data/image_quality_train.json",
        help="正样本数据源路径"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/root/ImageReward/data_generation/data/prompts_tagged.json",
        help="标注结果输出路径"
    )
    parser.add_argument(
        "--tag_config",
        type=str,
        default="/root/ImageReward/data_generation/config/semantic_tag_requirements.json",
        help="标签配置文件路径"
    )
    parser.add_argument(
        "--use_llm",
        action="store_true",
        default=True,
        help="使用LLM进行标注（默认开启）"
    )
    parser.add_argument(
        "--no_llm",
        action="store_true",
        help="仅使用关键词匹配（不使用LLM）"
    )
    parser.add_argument(
        "--llm_workers",
        type=int,
        default=5,
        help="LLM并行请求数"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="限制处理的样本数量（用于测试）"
    )
    parser.add_argument(
        "--stats_only",
        action="store_true",
        help="仅统计已标注数据，不重新标注"
    )

    args = parser.parse_args()

    # 加载标签配置
    logger.info(f"Loading tag config from {args.tag_config}")
    tag_config = load_tag_config(args.tag_config)

    if args.stats_only:
        # 仅统计模式
        logger.info(f"Loading tagged data from {args.output}")
        with open(args.output, 'r', encoding='utf-8') as f:
            data = json.load(f)
        tagged_prompts = data.get('prompts', data)
    else:
        # 加载正样本
        logger.info(f"Loading prompts from {args.source}")
        prompts = load_positive_prompts(args.source)
        logger.info(f"Loaded {len(prompts)} prompts")

        # 限制数量（用于测试）
        if args.limit:
            prompts = prompts[:args.limit]
            logger.info(f"Limited to {len(prompts)} prompts for testing")

        # 执行标注
        use_llm = args.use_llm and not args.no_llm
        tagged_prompts = batch_tag_prompts(
            prompts,
            tag_config,
            use_llm=use_llm,
            llm_workers=args.llm_workers
        )

        # 保存结果
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        output_data = {
            "version": "1.0",
            "source": args.source,
            "total_prompts": len(tagged_prompts),
            "prompts": tagged_prompts
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved tagged prompts to {args.output}")

    # 计算统计信息
    logger.info("Computing statistics...")
    stats = compute_tag_statistics(tagged_prompts, tag_config)

    print("\n" + "=" * 60)
    print("标签分布统计")
    print("=" * 60)
    print(f"总样本数: {stats['total_prompts']}")
    print("\n标签 | 数量 | 占比")
    print("-" * 40)
    for tag, info in sorted(stats['tag_distribution'].items(), key=lambda x: -x[1]['count']):
        print(f"{tag}: {info['count']} ({info['percentage']}%)")

    # 检查维度覆盖
    logger.info("Checking dimension coverage...")
    coverage = check_dimension_coverage(tagged_prompts, tag_config)

    print("\n" + "=" * 60)
    print("维度覆盖情况（需要特定标签的维度）")
    print("=" * 60)

    for perspective, dims in coverage.items():
        print(f"\n### {perspective} ###")
        for dim_name, info in dims.items():
            if info['requirement'] != 'none' and info['requirement'] != 'preferred_only':
                status = "✅" if info['percentage'] >= 10 else "⚠️" if info['percentage'] >= 1 else "❌"
                print(f"  {status} {dim_name}: {info['available']} samples ({info['percentage']}%)")
                print(f"      要求: {info['requirement']}")

    # 保存统计信息
    stats_output = args.output.replace('.json', '_stats.json')
    with open(stats_output, 'w', encoding='utf-8') as f:
        json.dump({
            "tag_statistics": stats,
            "dimension_coverage": coverage
        }, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved statistics to {stats_output}")


if __name__ == "__main__":
    main()
