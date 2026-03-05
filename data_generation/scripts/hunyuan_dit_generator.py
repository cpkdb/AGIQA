"""
Hunyuan-DiT 图像生成器
优先为单卡 4090 提供全 GPU 的速度优先推理路径。
"""

import sys

try:
    import torch
except ImportError as e:
    print("错误: 未找到 torch 模块。", file=sys.stderr)
    print(f"详细错误: {e}", file=sys.stderr)
    sys.exit(1)

try:
    from diffusers import HunyuanDiTPipeline
except ImportError as e:
    print("错误: 未找到 diffusers 中的 HunyuanDiTPipeline。", file=sys.stderr)
    print("请安装较新的 diffusers 版本。", file=sys.stderr)
    print(f"详细错误: {e}", file=sys.stderr)
    sys.exit(1)

from PIL import Image
import os
from typing import Optional, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


DEFAULT_MODEL_ID = "Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers"


class HunyuanDiTGenerator:
    """Hunyuan-DiT 图像生成器类。"""

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_ID,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        use_cpu_offload: bool = False,
    ):
        self.device = device
        self.dtype = dtype
        self.model_path = model_path
        self.use_cpu_offload = use_cpu_offload

        load_path = model_path
        if model_path and os.path.exists(model_path):
            logger.info(f"加载本地 Hunyuan-DiT 模型: {model_path}")
        else:
            if model_path != DEFAULT_MODEL_ID:
                logger.warning(f"本地模型不存在: {model_path}，回退到默认模型 ID: {DEFAULT_MODEL_ID}")
            load_path = DEFAULT_MODEL_ID
            logger.info(f"加载 HuggingFace Hunyuan-DiT 模型: {load_path}")

        try:
            self.pipe = HunyuanDiTPipeline.from_pretrained(
                load_path,
                torch_dtype=dtype,
            )

            if use_cpu_offload:
                self.pipe.enable_model_cpu_offload()
                logger.info("Hunyuan-DiT 已启用 CPU offload")
            else:
                self.pipe = self.pipe.to(device)

            if hasattr(self.pipe, "enable_xformers_memory_efficient_attention"):
                try:
                    self.pipe.enable_xformers_memory_efficient_attention()
                    logger.info("Hunyuan-DiT 已启用 xFormers")
                except Exception as exc:
                    logger.warning(f"Hunyuan-DiT 无法启用 xFormers: {exc}")

            if hasattr(self.pipe, "enable_vae_tiling"):
                self.pipe.enable_vae_tiling()
            if hasattr(self.pipe, "enable_vae_slicing"):
                self.pipe.enable_vae_slicing()

            logger.info("Hunyuan-DiT 模型加载成功")
        except Exception as exc:
            logger.error(f"加载 Hunyuan-DiT 模型失败: {exc}")
            raise

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = 30,
        guidance_scale: float = 5.0,
        width: int = 1024,
        height: int = 1024,
        seed: Optional[int] = None,
        **kwargs,
    ) -> tuple[Image.Image, Dict[str, Any]]:
        generator = None
        if seed is not None:
            generator = torch.Generator(device="cpu").manual_seed(seed)

        logger.info(f"Hunyuan-DiT 生成中 (steps={num_inference_steps}, seed={seed})")

        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            generator=generator,
        )

        image = result.images[0]
        gen_info = {
            "model": "hunyuan-dit",
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "width": width,
            "height": height,
            "seed": seed,
            "use_cpu_offload": self.use_cpu_offload,
        }

        return image, gen_info

    def cleanup(self):
        if hasattr(self, "pipe"):
            del self.pipe
            torch.cuda.empty_cache()
            logger.info("Hunyuan-DiT 生成器资源已清理")
