"""
SDXL图像生成器
使用diffusers库调用SDXL模型生成图像
"""

import torch
from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler
from PIL import Image
import os
from typing import Optional, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SDXLGenerator:
    """SDXL图像生成器类"""

    def __init__(
        self,
        model_path: str = "/root/ckpts/sd_xl_base_1.0.safetensors",
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        use_safetensors: bool = True
    ):
        """
        初始化SDXL生成器

        Args:
            model_path: SDXL模型路径
            device: 运行设备 (cuda/cpu)
            dtype: 数据类型
            use_safetensors: 是否使用safetensors格式
        """
        self.device = device
        self.dtype = dtype
        self.model_path = model_path

        logger.info(f"正在加载SDXL模型: {model_path}")

        # 检查模型路径是否存在
        if not os.path.exists(model_path):
            logger.warning(f"本地模型不存在: {model_path}")
            logger.info("将从HuggingFace下载模型: stabilityai/stable-diffusion-xl-base-1.0")
            model_path = "stabilityai/stable-diffusion-xl-base-1.0"

        # 加载pipeline
        try:
            self.pipe = StableDiffusionXLPipeline.from_single_file(
                model_path,
                torch_dtype=dtype,
                use_safetensors=use_safetensors
            ) if os.path.exists(self.model_path) else StableDiffusionXLPipeline.from_pretrained(
                model_path,
                torch_dtype=dtype,
                use_safetensors=use_safetensors
            )

            self.pipe = self.pipe.to(device)

            # 启用内存优化
            self.pipe.enable_model_cpu_offload()
            if hasattr(self.pipe, 'enable_xformers_memory_efficient_attention'):
                try:
                    self.pipe.enable_xformers_memory_efficient_attention()
                    logger.info("已启用xformers内存优化")
                except Exception as e:
                    logger.warning(f"无法启用xformers: {e}")

            logger.info("SDXL模型加载成功")

        except Exception as e:
            logger.error(f"加载SDXL模型失败: {e}")
            raise

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        width: int = 1024,
        height: int = 1024,
        seed: Optional[int] = None,
        **kwargs
    ) -> tuple[Image.Image, Dict[str, Any]]:
        """
        生成图像

        Args:
            prompt: 正向提示词
            negative_prompt: 负向提示词
            num_inference_steps: 推理步数
            guidance_scale: CFG scale
            width: 图像宽度
            height: 图像高度
            seed: 随机种子
            **kwargs: 其他参数

        Returns:
            (生成的图像, 生成信息字典)
        """
        # 设置随机种子
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        # 生成图像
        logger.info(f"正在生成图像 - Prompt: {prompt[:50]}...")

        try:
            output = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
                generator=generator,
                **kwargs
            )

            image = output.images[0]

            # 生成信息
            generation_info = {
                "model": "stable-diffusion-xl-base-1.0",
                "seed": seed,
                "steps": num_inference_steps,
                "cfg_scale": guidance_scale,
                "width": width,
                "height": height,
                "prompt": prompt,
                "negative_prompt": negative_prompt
            }

            logger.info("图像生成成功")
            return image, generation_info

        except Exception as e:
            logger.error(f"生成图像失败: {e}")
            raise

    def generate_batch(
        self,
        prompts: list[str],
        negative_prompts: Optional[list[str]] = None,
        **kwargs
    ) -> list[tuple[Image.Image, Dict[str, Any]]]:
        """
        批量生成图像

        Args:
            prompts: 提示词列表
            negative_prompts: 负向提示词列表
            **kwargs: 其他生成参数

        Returns:
            [(图像, 生成信息), ...]
        """
        if negative_prompts is None:
            negative_prompts = [""] * len(prompts)

        results = []
        for prompt, negative_prompt in zip(prompts, negative_prompts):
            try:
                result = self.generate(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    **kwargs
                )
                results.append(result)
            except Exception as e:
                logger.error(f"批量生成失败 - Prompt: {prompt[:50]}... - 错误: {e}")
                continue

        return results

    def cleanup(self):
        """清理资源"""
        if hasattr(self, 'pipe'):
            del self.pipe
            torch.cuda.empty_cache()
            logger.info("已清理SDXL模型资源")


if __name__ == "__main__":
    # 测试代码
    generator = SDXLGenerator()

    # 测试单张图像生成
    image, info = generator.generate(
        prompt="a beautiful sunset over the ocean, masterpiece, high quality, detailed",
        negative_prompt="low quality, blurry, bad anatomy",
        seed=42
    )

    # 保存测试图像
    os.makedirs("/root/ImageReward/data_generation/test_output", exist_ok=True)
    image.save("/root/ImageReward/data_generation/test_output/test_sdxl.png")
    print(f"测试图像已保存，生成信息: {info}")

    generator.cleanup()
