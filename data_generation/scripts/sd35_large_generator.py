"""
SD3.5 Large 图像生成器
默认采用速度优先的量化加载策略，并保留 CPU offload 回退选项。
"""

import sys

try:
    import torch
except ImportError as e:
    print("错误: 未找到 torch 模块。", file=sys.stderr)
    print(f"详细错误: {e}", file=sys.stderr)
    sys.exit(1)

try:
    from diffusers import StableDiffusion3Pipeline, SD3Transformer2DModel
except ImportError as e:
    print("错误: 未找到 diffusers 中的 SD3 相关模块。", file=sys.stderr)
    print("请安装较新的 diffusers 版本。", file=sys.stderr)
    print(f"详细错误: {e}", file=sys.stderr)
    sys.exit(1)

try:
    from transformers import BitsAndBytesConfig
    BNB_AVAILABLE = True
except ImportError:
    BNB_AVAILABLE = False

from PIL import Image
import os
from typing import Optional, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


DEFAULT_MODEL_ID = "stabilityai/stable-diffusion-3.5-large"


class SD35LargeGenerator:
    """Stable Diffusion 3.5 Large 生成器类。"""

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_ID,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        use_cpu_offload: bool = False,
        prefer_quantized: bool = True,
    ):
        self.device = device
        self.dtype = dtype
        self.model_path = model_path
        self.use_cpu_offload = use_cpu_offload
        self.prefer_quantized = prefer_quantized

        load_path = model_path
        is_local = bool(model_path and os.path.isdir(model_path))
        if not is_local and model_path != DEFAULT_MODEL_ID and not os.path.exists(model_path):
            logger.warning(f"本地模型不存在: {model_path}，回退到默认模型 ID: {DEFAULT_MODEL_ID}")
            load_path = DEFAULT_MODEL_ID
            is_local = False

        logger.info(f"加载 SD3.5 Large 模型: {load_path}")

        try:
            transformer = None
            if prefer_quantized and BNB_AVAILABLE:
                logger.info("尝试使用 4-bit transformer 量化加载 SD3.5 Large")
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=dtype,
                    bnb_4bit_use_double_quant=True,
                )
                transformer = SD3Transformer2DModel.from_pretrained(
                    load_path,
                    subfolder="transformer",
                    quantization_config=quant_config,
                    torch_dtype=dtype,
                    local_files_only=is_local,
                )

            self.pipe = StableDiffusion3Pipeline.from_pretrained(
                load_path,
                transformer=transformer,
                torch_dtype=dtype,
                local_files_only=is_local,
            )

            if use_cpu_offload:
                self.pipe.enable_model_cpu_offload()
                logger.info("SD3.5 Large 已启用 CPU offload")
            else:
                self.pipe = self.pipe.to(device)

            if hasattr(self.pipe, "enable_attention_slicing"):
                self.pipe.enable_attention_slicing()
            if hasattr(self.pipe, "enable_vae_tiling"):
                self.pipe.enable_vae_tiling()
            if hasattr(self.pipe, "enable_vae_slicing"):
                self.pipe.enable_vae_slicing()

            logger.info("SD3.5 Large 模型加载成功")
        except Exception as exc:
            logger.error(f"加载 SD3.5 Large 模型失败: {exc}")
            raise

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = 28,
        guidance_scale: float = 4.5,
        width: int = 1024,
        height: int = 1024,
        seed: Optional[int] = None,
        max_sequence_length: int = 512,
        **kwargs,
    ) -> tuple[Image.Image, Dict[str, Any]]:
        generator = None
        if seed is not None:
            generator = torch.Generator(device="cpu").manual_seed(seed)

        logger.info(f"SD3.5 Large 生成中 (steps={num_inference_steps}, seed={seed})")

        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            max_sequence_length=max_sequence_length,
            generator=generator,
        )

        image = result.images[0]
        gen_info = {
            "model": "sd3.5-large",
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "width": width,
            "height": height,
            "seed": seed,
            "use_cpu_offload": self.use_cpu_offload,
            "prefer_quantized": self.prefer_quantized,
        }

        return image, gen_info

    def cleanup(self):
        if hasattr(self, "pipe"):
            del self.pipe
            torch.cuda.empty_cache()
            logger.info("SD3.5 Large 生成器资源已清理")
