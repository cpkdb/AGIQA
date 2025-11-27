"""
正负对比自监督AIGC图像数据集生成Demo（使用硬编码关键词）
使用SDXL生成正负样本对，并使用ImageReward进行质量评分验证

⚠️ 注意：此demo使用硬编码的退化关键词，不依赖 quality_dimensions.json
建议用于快速测试，生产环境请使用 llm_prompt_degradation.py

核心思路：
1. 正样本：使用高质量提示词 + 标准negative prompt生成
2. 负样本：通过以下方式退化：
   - 视觉质量退化：去除质量提升词/添加退化关键词
   - 语义对齐退化：修改提示词关键元素
3. 正样本复用：一个正样本图像对应多个不同退化的负样本
4. 使用ImageReward验证正负样本质量差异（可选）
"""

import os
import sys
import json
import torch
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging
from PIL import Image

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# 注意：此demo不再使用 PromptDegradation（已弃用）
# from data_generation.scripts.prompt_degradation import PromptDegradation

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ContrastiveDatasetDemo:
    """正负对比自监督数据集生成Demo"""

    def __init__(
        self,
        sdxl_model_path: str = "/root/ckpts/sd_xl_base_1.0.safetensors",
        output_dir: str = "/root/autodl-tmp/demo_output",
        device: str = "cuda",
        use_image_reward: bool = False,
        num_negatives_per_positive: int = 3
    ):
        """
        初始化Demo

        Args:
            sdxl_model_path: SDXL模型路径
            output_dir: 输出目录
            device: 运行设备
            use_image_reward: 是否使用ImageReward评分
            num_negatives_per_positive: 每个正样本对应的负样本数量
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir = self.output_dir / "images"
        self.images_dir.mkdir(exist_ok=True)
        self.device = device
        self.use_image_reward = use_image_reward
        self.num_negatives_per_positive = num_negatives_per_positive

        # 初始化SDXL生成器
        logger.info("正在初始化SDXL生成器...")
        self._init_sdxl(sdxl_model_path)

        # 初始化ImageReward（可选）
        if use_image_reward:
            logger.info("正在初始化ImageReward模型...")
            self._init_image_reward()

        # 初始化提示词退化生成器
        logger.info("正在初始化提示词退化生成器...")
        quality_dimensions_path = Path(__file__).parent.parent / "config" / "quality_dimensions.json"
        self.prompt_degradation = PromptDegradation(str(quality_dimensions_path))

        # 数据集存储
        self.dataset = {
            "metadata": {
                "version": "2.0",
                "created_at": datetime.now().isoformat(),
                "description": "正负对比自监督AIGC图像数据集Demo（正样本复用版）",
                "generator_model": "stable-diffusion-xl-base-1.0",
                "total_pairs": 0,
                "total_positive_images": 0,
                "total_negative_images": 0,
                "positive_reuse_strategy": "每个正样本配对多个负样本",
                "num_negatives_per_positive": num_negatives_per_positive
            },
            "pairs": []
        }

    def _init_sdxl(self, model_path: str):
        """初始化SDXL模型"""
        from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler

        if os.path.exists(model_path):
            logger.info(f"从本地加载SDXL模型: {model_path}")
            self.pipe = StableDiffusionXLPipeline.from_single_file(
                model_path,
                torch_dtype=torch.float16,
                use_safetensors=True
            )
        else:
            logger.info("从HuggingFace下载SDXL模型...")
            self.pipe = StableDiffusionXLPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0",
                torch_dtype=torch.float16,
                use_safetensors=True
            )

        self.pipe = self.pipe.to(self.device)

        # 内存优化
        self.pipe.enable_model_cpu_offload()
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
            logger.info("已启用xformers内存优化")
        except Exception as e:
            logger.warning(f"无法启用xformers: {e}")

        logger.info("SDXL模型加载成功")

    def _init_image_reward(self):
        """初始化ImageReward模型"""
        try:
            import ImageReward as RM
            self.reward_model = RM.load("ImageReward-v1.0", device=self.device)
            logger.info("ImageReward模型加载成功")
        except Exception as e:
            logger.warning(f"无法加载ImageReward模型: {e}")
            self.use_image_reward = False
            self.reward_model = None

    def generate_image(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        width: int = 1024,
        height: int = 1024,
        seed: Optional[int] = None
    ) -> Tuple[Image.Image, Dict]:
        """
        生成单张图像

        Args:
            prompt: 正向提示词
            negative_prompt: 负向提示词
            num_inference_steps: 推理步数
            guidance_scale: CFG scale
            width: 图像宽度
            height: 图像高度
            seed: 随机种子

        Returns:
            (生成的图像, 生成信息)
        """
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        output = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            generator=generator
        )

        image = output.images[0]

        gen_info = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "seed": seed,
            "steps": num_inference_steps,
            "cfg_scale": guidance_scale,
            "width": width,
            "height": height
        }

        return image, gen_info

    def compute_reward_score(self, image: Image.Image, prompt: str) -> float:
        """
        计算ImageReward分数

        Args:
            image: PIL图像
            prompt: 提示词

        Returns:
            ImageReward分数
        """
        if not self.use_image_reward or self.reward_model is None:
            return 0.0

        try:
            score = self.reward_model.score(prompt, image)
            return float(score)
        except Exception as e:
            logger.warning(f"计算ImageReward分数失败: {e}")
            return 0.0

    def generate_contrastive_pair_visual_quality(
        self,
        prompt: str,
        pair_id: str,
        seed: int = 42,
        degradation_type: str = "blur",
        severity: str = "moderate"
    ) -> Dict:
        """
        生成视觉质量对比样本对

        正样本：高质量提示词生成
        负样本：添加退化关键词或去除质量提升词

        Args:
            prompt: 基础提示词
            pair_id: 样本对ID
            seed: 随机种子
            degradation_type: 退化类型
            severity: 退化程度

        Returns:
            样本对数据
        """
        # 退化关键词映射
        degradation_keywords = {
            "blur": {
                "mild": "slightly blurry, soft focus",
                "moderate": "blurry, out of focus",
                "severe": "extremely blurry, heavily out of focus"
            },
            "noise": {
                "mild": "subtle noise, slight grain",
                "moderate": "noisy, visible grain",
                "severe": "very noisy, heavy grain"
            },
            "low_quality": {
                "mild": "slightly low quality",
                "moderate": "low quality, mediocre",
                "severe": "very low quality, poor quality, ugly"
            },
            "artifacts": {
                "mild": "minor artifacts",
                "moderate": "visible artifacts, compression artifacts",
                "severe": "heavy artifacts, jpeg artifacts, distorted"
            }
        }

        # 质量提升词
        quality_boost = ", masterpiece, best quality, high resolution, detailed, sharp, professional"

        # 标准negative prompt
        standard_negative = "low quality, worst quality, blurry, ugly, deformed"

        # === 生成正样本 ===
        positive_prompt = prompt + quality_boost
        logger.info(f"[{pair_id}] 生成正样本...")

        positive_image, positive_info = self.generate_image(
            prompt=positive_prompt,
            negative_prompt=standard_negative,
            seed=seed
        )

        positive_path = self.images_dir / f"{pair_id}_positive.png"
        positive_image.save(positive_path)

        # === 生成负样本 ===
        # 方式1：添加退化关键词到prompt
        degradation_kw = degradation_keywords.get(degradation_type, {}).get(severity, "low quality")
        negative_prompt_text = f"{prompt}, {degradation_kw}"

        logger.info(f"[{pair_id}] 生成负样本 (退化类型: {degradation_type}, 程度: {severity})...")

        negative_image, negative_info = self.generate_image(
            prompt=negative_prompt_text,
            negative_prompt="",  # 不使用negative prompt以允许退化
            seed=seed  # 使用相同seed保持主体一致
        )

        negative_path = self.images_dir / f"{pair_id}_negative.png"
        negative_image.save(negative_path)

        # === 计算ImageReward分数 ===
        positive_score = self.compute_reward_score(positive_image, prompt)
        negative_score = self.compute_reward_score(negative_image, prompt)

        logger.info(f"[{pair_id}] ImageReward分数 - 正样本: {positive_score:.4f}, 负样本: {negative_score:.4f}")

        # 构建样本对数据
        pair_data = {
            "pair_id": pair_id,
            "type": "visual_quality",
            "positive": {
                "prompt": positive_prompt,
                "image_path": str(positive_path.relative_to(self.output_dir)),
                "generation_info": positive_info,
                "reward_score": positive_score
            },
            "negative": {
                "prompt": negative_prompt_text,
                "image_path": str(negative_path.relative_to(self.output_dir)),
                "generation_info": negative_info,
                "reward_score": negative_score
            },
            "degradation": {
                "category": "visual_quality",
                "type": degradation_type,
                "severity": severity,
                "keywords_added": degradation_kw
            },
            "score_difference": positive_score - negative_score
        }

        return pair_data

    def generate_contrastive_pair_semantic(
        self,
        prompt: str,
        pair_id: str,
        seed: int = 42,
        modification: Dict = None
    ) -> Dict:
        """
        生成语义对齐对比样本对

        正样本：原始提示词生成
        负样本：修改提示词关键元素后生成

        Args:
            prompt: 基础提示词
            pair_id: 样本对ID
            seed: 随机种子
            modification: 修改配置 {"original": "xxx", "replaced": "yyy", "type": "object/color/action"}

        Returns:
            样本对数据
        """
        if modification is None:
            modification = {"original": "", "replaced": "", "type": "none"}

        quality_boost = ", masterpiece, best quality, high resolution, detailed"
        standard_negative = "low quality, worst quality, blurry, ugly"

        # === 生成正样本 ===
        positive_prompt = prompt + quality_boost
        logger.info(f"[{pair_id}] 生成正样本 (语义对齐)...")

        positive_image, positive_info = self.generate_image(
            prompt=positive_prompt,
            negative_prompt=standard_negative,
            seed=seed
        )

        positive_path = self.images_dir / f"{pair_id}_positive.png"
        positive_image.save(positive_path)

        # === 生成负样本 ===
        # 替换提示词中的关键元素
        if modification["original"] and modification["original"] in prompt:
            negative_prompt_base = prompt.replace(modification["original"], modification["replaced"])
        else:
            negative_prompt_base = prompt

        negative_prompt_text = negative_prompt_base + quality_boost

        logger.info(f"[{pair_id}] 生成负样本 (语义修改: {modification['original']} → {modification['replaced']})...")

        negative_image, negative_info = self.generate_image(
            prompt=negative_prompt_text,
            negative_prompt=standard_negative,
            seed=seed
        )

        negative_path = self.images_dir / f"{pair_id}_negative.png"
        negative_image.save(negative_path)

        # === 计算ImageReward分数（使用原始prompt评估） ===
        positive_score = self.compute_reward_score(positive_image, prompt)
        negative_score = self.compute_reward_score(negative_image, prompt)  # 用原始prompt评估负样本

        logger.info(f"[{pair_id}] ImageReward分数 - 正样本: {positive_score:.4f}, 负样本(对齐度): {negative_score:.4f}")

        pair_data = {
            "pair_id": pair_id,
            "type": "semantic_alignment",
            "positive": {
                "prompt": positive_prompt,
                "image_path": str(positive_path.relative_to(self.output_dir)),
                "generation_info": positive_info,
                "reward_score": positive_score
            },
            "negative": {
                "prompt": negative_prompt_text,
                "image_path": str(negative_path.relative_to(self.output_dir)),
                "generation_info": negative_info,
                "reward_score": negative_score,
                "evaluated_with_prompt": prompt  # 注明使用原始prompt评估
            },
            "degradation": {
                "category": "alignment",
                "modification": modification
            },
            "score_difference": positive_score - negative_score
        }

        return pair_data

    def generate_contrastive_pairs_with_reuse(
        self,
        prompt: str,
        pair_id_start: int,
        seed: int,
        num_negatives: int = None
    ) -> List[Dict]:
        """
        使用正样本复用策略生成多个对比样本对

        一个正样本图像对应多个不同退化的负样本图像

        Args:
            prompt: 基础提示词
            pair_id_start: 起始pair_id编号
            seed: 随机种子
            num_negatives: 负样本数量（默认使用self.num_negatives_per_positive）

        Returns:
            样本对数据列表
        """
        if num_negatives is None:
            num_negatives = self.num_negatives_per_positive

        quality_boost = ", masterpiece, best quality, high resolution, detailed, sharp, professional"
        standard_negative = "low quality, worst quality, blurry, ugly, deformed"

        pairs = []

        # === 生成正样本（只生成一次） ===
        positive_prompt = prompt + quality_boost
        logger.info(f"[正样本 seed={seed}] 生成正样本图像...")

        positive_image, positive_info = self.generate_image(
            prompt=positive_prompt,
            negative_prompt=standard_negative,
            seed=seed
        )

        positive_image_path = self.images_dir / f"positive_{seed}.png"
        positive_image.save(positive_image_path)

        # 计算正样本的ImageReward分数
        positive_score = self.compute_reward_score(positive_image, prompt)

        # === 生成多个负样本 ===
        # 获取所有退化类型
        all_degradation_types = self.prompt_degradation.get_all_degradation_types()

        # 随机选择N个退化类型（允许重复，但每次随机选择severity）
        import random
        selected_degradation_types = random.sample(
            all_degradation_types,
            min(num_negatives, len(all_degradation_types))
        )

        # 如果需要更多负样本，重复采样
        while len(selected_degradation_types) < num_negatives:
            selected_degradation_types.append(random.choice(all_degradation_types))

        for neg_idx, degradation_type in enumerate(selected_degradation_types[:num_negatives]):
            # 随机选择退化程度
            severity = self.prompt_degradation.select_severity_random()

            # 生成退化的负样本提示词
            negative_prompt_text, degradation_info = self.prompt_degradation.generate_negative_prompt(
                prompt, degradation_type, severity
            )

            logger.info(f"[负样本 seed={seed}_{neg_idx}] 生成负样本 ({degradation_info['dimension']}/{degradation_info['attribute']}/{severity})...")

            # 生成负样本图像（使用相同seed保持主体相似）
            negative_image, negative_info = self.generate_image(
                prompt=negative_prompt_text,
                negative_prompt="",  # 不使用negative prompt以允许退化
                seed=seed,
                num_inference_steps=30,
                guidance_scale=7.5
            )

            negative_image_path = self.images_dir / f"negative_{seed}_{neg_idx}.png"
            negative_image.save(negative_image_path)

            # 计算负样本的ImageReward分数
            negative_score = self.compute_reward_score(negative_image, prompt)

            # 构建样本对数据
            pair_id = pair_id_start + neg_idx
            pair_data = {
                "pair_id": f"{pair_id:07d}",
                "positive": {
                    "prompt": positive_prompt,
                    "image_path": str(positive_image_path.relative_to(self.output_dir)),
                    "generation_info": positive_info,
                    "reward_score": positive_score,
                    "shared_across_pairs": True,
                    "shared_seed": seed
                },
                "negative": {
                    "prompt": negative_prompt_text,
                    "image_path": str(negative_image_path.relative_to(self.output_dir)),
                    "generation_info": negative_info,
                    "reward_score": negative_score,
                    "negative_index": neg_idx
                },
                "degradation": degradation_info,
                "score_difference": positive_score - negative_score
            }

            pairs.append(pair_data)

            logger.info(f"[pair_id={pair_id:07d}] ImageReward分数 - 正: {positive_score:.4f}, 负: {negative_score:.4f}, 差值: {pair_data['score_difference']:.4f}")

        return pairs

    def run_demo(
        self,
        prompts: List[Dict],
        base_seed: int = 42,
        use_reuse_strategy: bool = True
    ):
        """
        运行完整Demo

        Args:
            prompts: 提示词列表，每个元素包含 prompt 和可选的 modification
            base_seed: 基础随机种子
            use_reuse_strategy: 是否使用正样本复用策略
        """
        logger.info(f"开始生成Demo数据集，共 {len(prompts)} 个样本...")
        logger.info(f"正样本复用策略: {'启用' if use_reuse_strategy else '禁用'}")

        if use_reuse_strategy:
            # === 使用正样本复用策略 ===
            pair_id_counter = 0

            for i, prompt_config in enumerate(prompts):
                prompt = prompt_config["prompt"]
                seed = base_seed + i

                try:
                    # 为每个正样本生成多个负样本
                    pairs = self.generate_contrastive_pairs_with_reuse(
                        prompt=prompt,
                        pair_id_start=pair_id_counter,
                        seed=seed
                    )

                    self.dataset["pairs"].extend(pairs)
                    self.dataset["metadata"]["total_pairs"] += len(pairs)
                    self.dataset["metadata"]["total_positive_images"] += 1
                    self.dataset["metadata"]["total_negative_images"] += len(pairs)

                    pair_id_counter += len(pairs)

                except Exception as e:
                    logger.error(f"生成样本 {i} 失败: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

                # 定期保存
                if (i + 1) % 3 == 0:
                    self._save_dataset()

        else:
            # === 旧方法：每对独立生成 ===
            degradation_types = ["blur", "noise", "low_quality", "artifacts"]
            severities = ["mild", "moderate", "severe"]

            for i, prompt_config in enumerate(prompts):
                prompt = prompt_config["prompt"]
                pair_type = prompt_config.get("type", "visual_quality")

                # 生成两个样本对ID
                pair_id_visual = f"demo_{i:04d}_visual"
                pair_id_semantic = f"demo_{i:04d}_semantic"

                try:
                    # === 视觉质量对比 ===
                    if pair_type in ["visual_quality", "both"]:
                        degradation_type = degradation_types[i % len(degradation_types)]
                        severity = severities[i % len(severities)]

                        pair_data = self.generate_contrastive_pair_visual_quality(
                            prompt=prompt,
                            pair_id=pair_id_visual,
                            seed=base_seed + i,
                            degradation_type=degradation_type,
                            severity=severity
                        )
                        self.dataset["pairs"].append(pair_data)
                        self.dataset["metadata"]["total_pairs"] += 1

                    # === 语义对齐对比 ===
                    if pair_type in ["semantic", "both"] and "modification" in prompt_config:
                        pair_data = self.generate_contrastive_pair_semantic(
                            prompt=prompt,
                            pair_id=pair_id_semantic,
                            seed=base_seed + i + 1000,
                            modification=prompt_config["modification"]
                        )
                        self.dataset["pairs"].append(pair_data)
                        self.dataset["metadata"]["total_pairs"] += 1

                except Exception as e:
                    logger.error(f"生成样本 {i} 失败: {e}")
                    continue

                # 定期保存
                if (i + 1) % 5 == 0:
                    self._save_dataset()

        # 最终保存
        self._save_dataset()
        self._generate_summary()

        logger.info(f"Demo完成！共生成 {self.dataset['metadata']['total_pairs']} 对样本")
        logger.info(f"输出目录: {self.output_dir}")

    def _save_dataset(self):
        """保存数据集"""
        output_file = self.output_dir / "dataset.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.dataset, f, ensure_ascii=False, indent=2)
        logger.info(f"数据集已保存到: {output_file}")

    def _generate_summary(self):
        """生成数据集统计摘要"""
        if not self.dataset["pairs"]:
            return

        # 统计
        total_pairs = len(self.dataset["pairs"])
        visual_pairs = sum(1 for p in self.dataset["pairs"] if p["type"] == "visual_quality")
        semantic_pairs = sum(1 for p in self.dataset["pairs"] if p["type"] == "semantic_alignment")

        # 分数统计
        score_diffs = [p["score_difference"] for p in self.dataset["pairs"] if "score_difference" in p]
        avg_score_diff = sum(score_diffs) / len(score_diffs) if score_diffs else 0

        # 正样本平均分数
        positive_scores = [p["positive"]["reward_score"] for p in self.dataset["pairs"]]
        negative_scores = [p["negative"]["reward_score"] for p in self.dataset["pairs"]]

        avg_positive = sum(positive_scores) / len(positive_scores) if positive_scores else 0
        avg_negative = sum(negative_scores) / len(negative_scores) if negative_scores else 0

        summary = {
            "total_pairs": total_pairs,
            "visual_quality_pairs": visual_pairs,
            "semantic_alignment_pairs": semantic_pairs,
            "scores": {
                "average_positive_score": round(avg_positive, 4),
                "average_negative_score": round(avg_negative, 4),
                "average_score_difference": round(avg_score_diff, 4)
            },
            "generated_at": datetime.now().isoformat()
        }

        summary_file = self.output_dir / "summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        logger.info("\n" + "="*50)
        logger.info("数据集统计摘要")
        logger.info("="*50)
        logger.info(f"总样本对数: {total_pairs}")
        logger.info(f"  - 视觉质量对比: {visual_pairs}")
        logger.info(f"  - 语义对齐对比: {semantic_pairs}")
        logger.info(f"ImageReward平均分数:")
        logger.info(f"  - 正样本: {avg_positive:.4f}")
        logger.info(f"  - 负样本: {avg_negative:.4f}")
        logger.info(f"  - 平均差值: {avg_score_diff:.4f}")
        logger.info("="*50)

    def cleanup(self):
        """清理资源"""
        if hasattr(self, 'pipe'):
            del self.pipe
        if hasattr(self, 'reward_model'):
            del self.reward_model
        torch.cuda.empty_cache()
        logger.info("资源已清理")


def get_demo_prompts() -> List[Dict]:
    """
    获取Demo用的提示词列表

    Returns:
        提示词配置列表
    """
    prompts = [
        # 视觉质量对比示例
        {
            "prompt": "a beautiful sunset over the ocean with colorful clouds",
            "type": "both",
            "modification": {
                "original": "sunset",
                "replaced": "sunrise",
                "type": "scene"
            }
        },
        {
            "prompt": "a cute orange cat sitting on a red velvet chair",
            "type": "both",
            "modification": {
                "original": "cat",
                "replaced": "dog",
                "type": "object"
            }
        },
        {
            "prompt": "portrait of a young woman with blue eyes and blonde hair",
            "type": "both",
            "modification": {
                "original": "blue eyes",
                "replaced": "brown eyes",
                "type": "attribute"
            }
        },
        {
            "prompt": "a modern glass building reflecting the city skyline",
            "type": "visual_quality"
        },
        {
            "prompt": "a red sports car driving on a mountain road",
            "type": "both",
            "modification": {
                "original": "red",
                "replaced": "blue",
                "type": "color"
            }
        },
        {
            "prompt": "a white horse running through a green meadow",
            "type": "both",
            "modification": {
                "original": "running",
                "replaced": "standing",
                "type": "action"
            }
        },
        {
            "prompt": "a cozy coffee shop interior with warm lighting",
            "type": "visual_quality"
        },
        {
            "prompt": "an astronaut floating in space with Earth in the background",
            "type": "visual_quality"
        }
    ]
    return prompts


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="正负对比自监督AIGC数据集生成Demo")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/root/autodl-tmp/demo_output",
        help="输出目录"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/root/ckpts/sd_xl_base_1.0.safetensors",
        help="SDXL模型路径"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="生成样本数量（默认使用全部demo prompts）"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子"
    )
    parser.add_argument(
        "--no_reward",
        action="store_true",
        help="禁用ImageReward评分"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="运行设备"
    )
    parser.add_argument(
        "--num_negatives_per_positive",
        type=int,
        default=3,
        help="每个正样本对应的负样本数量（默认3）"
    )
    parser.add_argument(
        "--use_old_method",
        action="store_true",
        help="使用旧方法（不复用正样本）"
    )

    args = parser.parse_args()

    # 获取Demo提示词
    prompts = get_demo_prompts()
    if args.num_samples:
        prompts = prompts[:args.num_samples]

    logger.info(f"将生成 {len(prompts)} 个基础样本的对比数据集")
    if not args.use_old_method:
        logger.info(f"每个正样本将生成 {args.num_negatives_per_positive} 个负样本")

    # 创建Demo实例
    demo = ContrastiveDatasetDemo(
        sdxl_model_path=args.model_path,
        output_dir=args.output_dir,
        device=args.device,
        use_image_reward=not args.no_reward,
        num_negatives_per_positive=args.num_negatives_per_positive
    )

    try:
        # 运行Demo
        demo.run_demo(
            prompts=prompts,
            base_seed=args.seed,
            use_reuse_strategy=not args.use_old_method
        )
    finally:
        demo.cleanup()


if __name__ == "__main__":
    main()
