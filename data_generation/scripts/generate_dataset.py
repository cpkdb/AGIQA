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
        llm_config_path: str = "/root/ImageReward/data_generation/config/llm_config.yaml"
    ):
        """
        初始化数据集生成器

        Args:
            output_dir: 输出目录 (默认: /root/autodl-tmp/dataset_v1)
            quality_dimensions_path: quality_dimensions.json路径
            model_path: SDXL模型路径
            schema_path: schema文件路径
            llm_config_path: LLM配置文件路径
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 创建子目录
        self.images_dir = self.output_dir / "images"
        self.images_dir.mkdir(exist_ok=True)

        # 加载schema
        with open(schema_path, 'r', encoding='utf-8') as f:
            self.schema = json.load(f)

        # 初始化组件
        self.sdxl_generator = SDXLGenerator(model_path=model_path)

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
                "generator_model": "stable-diffusion-xl-base-1.0",
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
        """
        with open(source_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 支持多种格式
        if isinstance(data, list):
            # 格式1: ["prompt1", "prompt2", ...]
            if all(isinstance(item, str) for item in data):
                return data
            # 格式2: [{"text": "prompt1"}, {"text": "prompt2"}, ...]
            elif all(isinstance(item, dict) for item in data):
                return [item.get('text', item.get('prompt', '')) for item in data]
        elif isinstance(data, dict):
            # 格式3: {"prompts": [...]}
            return data.get('prompts', [])

        return []

    def _generate_negative_prompt(
        self,
        positive_prompt: str,
        degradation_type: Optional[Dict] = None,
        severity: Optional[str] = None
    ) -> Tuple[str, Dict]:
        """
        生成退化的负样本prompt（LLM方法）

        Args:
            positive_prompt: 正样本提示词
            degradation_type: 退化类型（包含subcategory信息）
            severity: 退化程度

        Returns:
            (negative_prompt, degradation_info)
        """
        # LLM方法：使用子类别级别
        subcategory = degradation_type['subcategory']
        return self.degradation_generator.generate_negative_prompt(
            positive_prompt,
            subcategory,
            attribute=None,  # 随机选择属性
            severity=severity
        )

    def _get_all_degradation_types(self) -> List[Dict]:
        """获取所有退化类型（LLM方法）"""
        # LLM方法：返回子类别级别的退化类型
        return self.degradation_generator.get_all_subcategories()

    def generate_dataset_with_reuse(
        self,
        source_prompts: List[str],
        num_negatives_per_positive: int = 10,
        source: str = "custom",
        balance_severity: bool = False,
        balance_category: bool = True,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        base_seed: int = 42
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
        """
        import random

        # 获取所有退化类型
        all_degradation_types = self._get_all_degradation_types()
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
                # 选择退化程度
                if balance_severity:
                    severities = ["mild", "moderate", "severe"]
                    severity = severities[neg_idx % len(severities)]
                else:
                    severity = self.degradation_generator.select_severity_random()

                # 生成退化prompt
                negative_prompt_text, degradation_info = self._generate_negative_prompt(
                    positive_prompt,
                    degradation_type,
                    severity
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
                positive_image, positive_gen_info = self.sdxl_generator.generate(
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
                negative_image, negative_gen_info = self.sdxl_generator.generate(
                    prompt=negative_prompt_text,
                    negative_prompt="",
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    seed=seed
                )
            else:
                negative_image, negative_gen_info = self.sdxl_generator.generate(
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
        if hasattr(self, 'sdxl_generator'):
            self.sdxl_generator.cleanup()


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

    args = parser.parse_args()

    # 创建生成器
    generator = DatasetGenerator(
        output_dir=args.output_dir,
        quality_dimensions_path=args.quality_dimensions,
        model_path=args.model_path,
        llm_config_path=args.llm_config
    )

    # 加载源提示词
    source_prompts = generator.load_source_prompts(args.source_prompts)
    logger.info(f"加载了 {len(source_prompts)} 个源提示词")

    # 生成数据集（使用正样本复用策略）
    logger.info(f"每个正样本配对 {args.num_negatives_per_positive} 个负样本")
    generator.generate_dataset_with_reuse(
        source_prompts=source_prompts,
        num_negatives_per_positive=args.num_negatives_per_positive,
        source=args.source,
        balance_severity=args.balance_severity,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        base_seed=args.base_seed
    )

    # 清理
    generator.cleanup()


if __name__ == "__main__":
    main()
