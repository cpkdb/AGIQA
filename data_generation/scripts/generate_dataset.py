"""
主数据集生成脚本
整合SDXL图像生成和提示词退化，生成符合schema的数据集
"""

import os
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging
from tqdm import tqdm

from sdxl_generator import SDXLGenerator
from llm_prompt_degradation import LLMPromptDegradation

# 可选生成器导入
try:
    from flux_generator import FluxGenerator
    FLUX_DEV_AVAILABLE = True
except ImportError:
    FLUX_DEV_AVAILABLE = False

try:
    from flux_schnell_generator import FluxSchnellGenerator
    FLUX_SCHNELL_AVAILABLE = True
except ImportError:
    FLUX_SCHNELL_AVAILABLE = False

# 检查LLM是否可用
try:
    import openai
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetGenerator:
    """数据集生成器"""

    def __init__(
        self,
        output_dir: str = "/root/autodl-tmp/dataset_v1",
        quality_dimensions_path: str = "/root/ImageReward/data_generation/config/quality_dimensions.json",
        model_path: str = "/root/ckpts/sd_xl_base_1.0.safetensors",
        schema_path: str = "/root/ImageReward/data_generation/schema/dataset_schema.json",
        llm_config_path: str = "/root/ImageReward/data_generation/config/llm_config.yaml",
        generator_type: str = "sdxl"
    ):
        """
        初始化数据集生成器

        Args:
            output_dir: 输出目录 (默认: /root/autodl-tmp/dataset_v1)
            quality_dimensions_path: quality_dimensions.json路径
            model_path: SDXL模型路径
            schema_path: schema文件路径
            llm_config_path: LLM配置文件路径
            generator_type: 生成器类型 (sdxl/flux-dev/flux-schnell)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 创建子目录
        self.images_dir = self.output_dir / "images"
        self.images_dir.mkdir(exist_ok=True)

        # 加载schema
        with open(schema_path, 'r', encoding='utf-8') as f:
            self.schema = json.load(f)

        # 初始化图像生成器
        self.generator_type = generator_type
        if generator_type == "flux-schnell":
            if not FLUX_SCHNELL_AVAILABLE:
                raise ImportError("FluxSchnellGenerator 不可用，请检查 flux_schnell_generator.py")
            self.image_generator = FluxSchnellGenerator()
            generator_model_name = "flux-1-schnell"
        elif generator_type == "flux-dev":
            if not FLUX_DEV_AVAILABLE:
                raise ImportError("FluxGenerator 不可用，请检查 flux_generator.py")
            self.image_generator = FluxGenerator()
            generator_model_name = "flux-1-dev"
        else:
            self.image_generator = SDXLGenerator(model_path=model_path)
            generator_model_name = "stable-diffusion-xl-base-1.0"

        logger.info(f"✓ 使用图像生成器: {generator_type}")

        # 只使用LLM方法
        if not LLM_AVAILABLE:
            raise ImportError("LLM退化生成器不可用，请安装: pip install openai")

        logger.info("✓ 使用LLM方法生成退化prompt")
        self.degradation_generator = LLMPromptDegradation(
            llm_config_path=llm_config_path,
            quality_dimensions_path=quality_dimensions_path
        )

        # 数据集元数据
        self.dataset = {
            "metadata": {
                "version": "2.0",
                "created_at": datetime.now().isoformat(),
                "total_pairs": 0,
                "total_positive_images": 0,
                "total_negative_images": 0,
                "generator_model": generator_model_name,
                "degradation_method": "LLM-based",
                "description": "AIGC图像质量评估自监督数据集 (LLM-based)",
                "positive_reuse_strategy": "每个正样本配对多个负样本"
            },
            "pairs": []
        }

    def load_source_prompts(self, source_file: str) -> List[str]:
        """
        加载源提示词

        Args:
            source_file: 源提示词文件路径（JSON格式）

        Returns:
            提示词列表

        支持的格式:
            格式1: ["prompt1", "prompt2", ...]
            格式2: [{"text": "prompt1"}, ...] 或 [{"prompt": "prompt1"}, ...]
            格式3: {"prompts": ["prompt1", ...]}
            格式4: {"prompts": [{"prompt": "prompt1"}, ...]}  # 新维度格式
        """
        with open(source_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        prompts = []

        # 支持多种格式
        if isinstance(data, list):
            prompts = data
        elif isinstance(data, dict):
            prompts = data.get('prompts', [])

        # 提取 prompt 文本
        result = []
        for item in prompts:
            if isinstance(item, str):
                result.append(item)
            elif isinstance(item, dict):
                # 支持 "text" 或 "prompt" 字段
                prompt_text = item.get('prompt', item.get('text', ''))
                if prompt_text:
                    result.append(prompt_text)

        return result

    def _generate_negative_prompt(
        self,
        positive_prompt: str,
        degradation_type: Optional[Dict] = None,
        severity: Optional[str] = None,
        attribute_filter: Optional[str] = None
    ) -> Tuple[str, Dict]:
        """
        生成退化的负样本prompt（LLM方法）

        Args:
            positive_prompt: 正样本提示词
            degradation_type: 退化类型（包含subcategory信息）
            severity: 退化程度
            attribute_filter: 指定的属性（如 blur, noise）

        Returns:
            (negative_prompt, degradation_info)
        """
        # LLM方法：使用子类别级别
        subcategory = degradation_type['subcategory']
        return self.degradation_generator.generate_negative_prompt(
            positive_prompt,
            subcategory,
            attribute=attribute_filter,  # 使用指定属性或随机选择
            severity=severity
        )

    def _get_all_degradation_types(
        self, 
        category_filter: Optional[str] = None,
        subcategory_filter: Optional[str] = None
    ) -> List[Dict]:
        """
        获取所有退化类型（LLM方法）
        
        Args:
            category_filter: 可选的类别过滤器，如 "visual_quality" 或 "alignment"
            subcategory_filter: 可选的子类别过滤器，如 "low_visual_quality" 或 "aesthetic_quality"
        """
        all_types = self.degradation_generator.get_all_subcategories()
        
        # 子类别过滤优先级更高
        if subcategory_filter:
            filtered = [t for t in all_types if t['subcategory'] == subcategory_filter]
            logger.info(f"子类别过滤: {subcategory_filter}, 筛选后 {len(filtered)}/{len(all_types)} 个子类别")
            return filtered
        
        if category_filter:
            filtered = [t for t in all_types if t['category'] == category_filter]
            logger.info(f"类别过滤: {category_filter}, 筛选后 {len(filtered)}/{len(all_types)} 个子类别")
            return filtered
        
        return all_types

    def generate_dataset_with_reuse(
        self,
        source_prompts: List[str],
        num_negatives_per_positive: int = 10,
        source: str = "custom",
        balance_severity: bool = False,
        balance_category: bool = True,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        base_seed: int = 42,
        category_filter: Optional[str] = None,
        subcategory_filter: Optional[str] = None,
        attribute_filter: Optional[str] = None,
        fixed_severity: Optional[str] = None
    ):
        """
        生成数据集（正样本复用策略，分阶段执行）

        执行流程：
        阶段1: LLM批量生成所有退化prompt
        阶段2: SDXL批量生成所有图像

        Args:
            source_prompts: 源提示词列表
            num_negatives_per_positive: 每个正样本配对的负样本数量（默认10）
            source: 提示词来源标识
            balance_severity: 是否平衡退化程度（默认False，使用20%/40%/40%分布）
            balance_category: 是否平衡退化类别（默认True）
            num_inference_steps: 推理步数
            guidance_scale: CFG scale
            base_seed: 基础随机种子
            category_filter: 类别过滤器，如 "visual_quality" 或 "alignment"
            subcategory_filter: 子类别过滤器，如 "low_visual_quality"
            attribute_filter: 属性过滤器，如 "blur" 或 "noise"
            fixed_severity: 固定退化程度，如 "mild", "moderate", "severe"
        """
        import random

        # 获取所有退化类型（可选过滤）
        all_degradation_types = self._get_all_degradation_types(category_filter, subcategory_filter)
        
        # 如果指定了属性过滤，记录日志
        if attribute_filter:
            logger.info(f"属性过滤: 所有样本将使用属性 '{attribute_filter}'")
        total_pairs = len(source_prompts) * num_negatives_per_positive

        logger.info(f"=" * 60)
        logger.info(f"开始生成数据集（分阶段执行）")
        logger.info(f"=" * 60)
        logger.info(f"  - 正样本prompt数量: {len(source_prompts)}")
        logger.info(f"  - 每个正样本配对负样本数: {num_negatives_per_positive}")
        logger.info(f"  - 预计生成pair总数: {total_pairs}")
        logger.info(f"  - 退化类型总数: {len(all_degradation_types)}")

        # ============================================================
        # 阶段1: LLM批量生成所有退化prompt
        # ============================================================
        logger.info(f"\n{'=' * 60}")
        logger.info(f"阶段1: LLM批量生成退化prompt")
        logger.info(f"{'=' * 60}")

        # 存储所有生成任务的信息
        generation_tasks = []

        for positive_idx, positive_prompt in enumerate(tqdm(source_prompts, desc="LLM退化生成")):
            seed = base_seed + positive_idx

            # 选择退化类型
            if balance_category and num_negatives_per_positive <= len(all_degradation_types):
                selected_degradation_types = random.sample(
                    all_degradation_types,
                    num_negatives_per_positive
                )
            elif num_negatives_per_positive > len(all_degradation_types):
                selected_degradation_types = random.choices(
                    all_degradation_types,
                    k=num_negatives_per_positive
                )
            else:
                selected_degradation_types = random.choices(
                    all_degradation_types,
                    k=num_negatives_per_positive
                )

            for neg_idx, degradation_type in enumerate(selected_degradation_types):
                # 选择退化程度（优先级：fixed_severity > balance_severity > random）
                if fixed_severity:
                    severity = fixed_severity
                elif balance_severity:
                    severities = ["mild", "moderate", "severe"]
                    severity = severities[neg_idx % len(severities)]
                else:
                    severity = self.degradation_generator.select_severity_random()

                # 生成退化prompt
                negative_prompt_text, degradation_info = self._generate_negative_prompt(
                    positive_prompt,
                    degradation_type,
                    severity,
                    attribute_filter=attribute_filter
                )

                # 存储任务信息
                generation_tasks.append({
                    'positive_idx': positive_idx,
                    'neg_idx': neg_idx,
                    'positive_prompt': positive_prompt,
                    'negative_prompt': negative_prompt_text,
                    'degradation_info': degradation_info,
                    'seed': seed
                })

                logger.debug(f"  [{positive_idx+1}-{neg_idx+1}] {degradation_info['subcategory']} ({severity})")

        logger.info(f"\n✓ 阶段1完成: 生成了 {len(generation_tasks)} 个退化prompt")

        # 保存退化prompt到临时文件（可选，用于检查）
        prompts_cache_file = self.output_dir / "prompts_cache.json"
        with open(prompts_cache_file, 'w', encoding='utf-8') as f:
            json.dump(generation_tasks, f, ensure_ascii=False, indent=2)
        logger.info(f"  退化prompt已缓存到: {prompts_cache_file}")

        # ============================================================
        # 阶段2: SDXL批量生成所有图像
        # ============================================================
        logger.info(f"\n{'=' * 60}")
        logger.info(f"阶段2: SDXL批量生成图像")
        logger.info(f"{'=' * 60}")

        pair_id_counter = 0
        generated_positive_seeds = set()  # 记录已生成的正样本

        for task in tqdm(generation_tasks, desc="SDXL图像生成"):
            positive_idx = task['positive_idx']
            neg_idx = task['neg_idx']
            positive_prompt = task['positive_prompt']
            negative_prompt_text = task['negative_prompt']
            degradation_info = task['degradation_info']
            seed = task['seed']

            # === 生成正样本图像（每个seed只生成一次） ===
            positive_image_path = self.images_dir / f"positive_{seed}.png"

            if seed not in generated_positive_seeds:
                logger.info(f"\n[正样本 {positive_idx+1}/{len(source_prompts)}] 生成正样本图像...")
                positive_image, positive_gen_info = self.image_generator.generate(
                    prompt=positive_prompt,
                    negative_prompt="low quality, worst quality",
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    seed=seed
                )
                positive_image.save(positive_image_path)
                self.dataset['metadata']['total_positive_images'] += 1
                generated_positive_seeds.add(seed)
                logger.info(f"  保存正样本: {positive_image_path.name}")
            else:
                # 复用已有的正样本信息
                positive_gen_info = {'model': 'stable-diffusion-xl-base-1.0'}

            # === 生成负样本图像 ===
            degradation_desc = degradation_info.get('subcategory', 'unknown')
            logger.info(f"  [负样本 {neg_idx+1}] 退化: {degradation_desc} ({degradation_info['severity']})")

            if degradation_info['category'] == 'visual_quality':
                negative_image, negative_gen_info = self.image_generator.generate(
                    prompt=negative_prompt_text,
                    negative_prompt="",
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    seed=seed
                )
            else:
                negative_image, negative_gen_info = self.image_generator.generate(
                    prompt=negative_prompt_text,
                    negative_prompt="low quality, worst quality",
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    seed=seed
                )

            # 保存负样本图像
            negative_image_path = self.images_dir / f"negative_{seed}_{neg_idx}.png"
            negative_image.save(negative_image_path)
            self.dataset['metadata']['total_negative_images'] += 1

            # === 构建pair数据 ===
            pair_data = {
                "pair_id": f"{pair_id_counter:07d}",
                "positive": {
                    "prompt": positive_prompt,
                    "image_path": str(positive_image_path.relative_to(self.output_dir)),
                    "source": source,
                    "shared_across_pairs": True,
                    "shared_seed": seed
                },
                "negative": {
                    "prompt": negative_prompt_text,
                    "image_path": str(negative_image_path.relative_to(self.output_dir))
                },
                "degradation": degradation_info,
                "generation_info": {
                    "model": positive_gen_info['model'],
                    "seed": seed,
                    "steps": num_inference_steps,
                    "cfg_scale": guidance_scale,
                    "generated_at": datetime.now().isoformat()
                }
            }

            self.dataset['pairs'].append(pair_data)
            self.dataset['metadata']['total_pairs'] += 1
            pair_id_counter += 1

            # 定期保存
            if pair_id_counter % 10 == 0:
                self.save_dataset()

        # 最终保存
        self.save_dataset()

        # 删除临时缓存文件
        if prompts_cache_file.exists():
            prompts_cache_file.unlink()

        logger.info(f"\n{'=' * 60}")
        logger.info(f"数据集生成完成！")
        logger.info(f"{'=' * 60}")
        logger.info(f"  - 正样本图像: {self.dataset['metadata']['total_positive_images']}")
        logger.info(f"  - 负样本图像: {self.dataset['metadata']['total_negative_images']}")
        logger.info(f"  - 总pair数: {self.dataset['metadata']['total_pairs']}")

    def save_dataset(self):
        """保存数据集到JSON文件"""
        output_file = self.output_dir / "dataset.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.dataset, f, ensure_ascii=False, indent=2)
        logger.info(f"数据集已保存到: {output_file}")

    def cleanup(self):
        """清理资源"""
        if hasattr(self, 'image_generator'):
            self.image_generator.cleanup()


def main():
    parser = argparse.ArgumentParser(description="生成AIGC图像质量评估数据集（正样本复用策略）")

    parser.add_argument(
        "--source_prompts",
        type=str,
        required=True,
        help="源提示词JSON文件路径"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/root/autodl-tmp/dataset_v1",
        help="输出目录（默认：/root/autodl-tmp/dataset_v1）"
    )
    parser.add_argument(
        "--num_negatives_per_positive",
        type=int,
        default=10,
        help="每个正样本配对的负样本数量（默认10）"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/root/ckpts/sd_xl_base_1.0.safetensors",
        help="SDXL模型路径"
    )
    parser.add_argument(
        "--generator",
        type=str,
        default="sdxl",
        choices=["sdxl", "flux-dev", "flux-schnell"],
        help="图像生成器类型（默认：sdxl）"
    )
    parser.add_argument(
        "--quality_dimensions",
        type=str,
        default="/root/ImageReward/data_generation/config/quality_dimensions.json",
        help="质量维度配置文件路径"
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="推理步数"
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="CFG scale"
    )
    parser.add_argument(
        "--base_seed",
        type=int,
        default=42,
        help="基础随机种子"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="custom",
        help="提示词来源标识"
    )
    parser.add_argument(
        "--balance_severity",
        action="store_true",
        help="是否平衡退化程度分布（默认使用20%%/40%%/40%%随机分布）"
    )

    parser.add_argument(
        "--llm_config",
        type=str,
        default="/root/ImageReward/data_generation/config/llm_config.yaml",
        help="LLM配置文件路径"
    )
    parser.add_argument(
        "--category_filter",
        type=str,
        default=None,
        choices=["visual_quality", "alignment"],
        help="只使用指定类别的退化维度（可选：visual_quality, alignment）"
    )
    parser.add_argument(
        "--subcategory_filter",
        type=str,
        default=None,
        help="只使用指定子类别（如：low_visual_quality, aesthetic_quality, semantic_plausibility, basic_recognition, attribute_alignment, composition_interaction, external_knowledge）"
    )
    parser.add_argument(
        "--attribute_filter",
        type=str,
        default=None,
        help="只使用指定属性（如：blur, noise, exposure_issues, low_contrast, low_sharpness, color_distortion 等）"
    )
    parser.add_argument(
        "--num_positive_prompts",
        type=int,
        default=None,
        help="限制使用的正样本prompt数量（默认使用全部）"
    )
    parser.add_argument(
        "--random_select_prompts",
        action="store_true",
        help="随机选择正样本prompt（配合 --num_positive_prompts 使用）"
    )
    parser.add_argument(
        "--severity",
        type=str,
        default=None,
        choices=["mild", "moderate", "severe"],
        help="指定固定的退化程度（不指定则随机或平衡选择）"
    )

    args = parser.parse_args()

    # 创建生成器
    generator = DatasetGenerator(
        output_dir=args.output_dir,
        quality_dimensions_path=args.quality_dimensions,
        model_path=args.model_path,
        llm_config_path=args.llm_config,
        generator_type=args.generator
    )

    # 加载源提示词
    source_prompts = generator.load_source_prompts(args.source_prompts)
    logger.info(f"加载了 {len(source_prompts)} 个源提示词")
    
    # 限制正样本数量
    if args.num_positive_prompts and args.num_positive_prompts < len(source_prompts):
        import random as rand_module
        if args.random_select_prompts:
            # 随机选择
            source_prompts = rand_module.sample(source_prompts, args.num_positive_prompts)
            logger.info(f"随机选择了 {args.num_positive_prompts} 个正样本prompt")
        else:
            # 顺序选择前N个
            source_prompts = source_prompts[:args.num_positive_prompts]
            logger.info(f"顺序选择前 {args.num_positive_prompts} 个正样本prompt")

    # 生成数据集（使用正样本复用策略）
    logger.info(f"每个正样本配对 {args.num_negatives_per_positive} 个负样本")
    generator.generate_dataset_with_reuse(
        source_prompts=source_prompts,
        num_negatives_per_positive=args.num_negatives_per_positive,
        source=args.source,
        balance_severity=args.balance_severity,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        base_seed=args.base_seed,
        category_filter=args.category_filter,
        subcategory_filter=args.subcategory_filter,
        attribute_filter=args.attribute_filter,
        fixed_severity=args.severity
    )

    # 清理
    generator.cleanup()


if __name__ == "__main__":
    main()
