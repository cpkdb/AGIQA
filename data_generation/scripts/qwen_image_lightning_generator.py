"""
Qwen-Image-Lightning 图像生成器
基于 Qwen/Qwen-Image base pipeline + Lightning LoRA，优先支持 4-step 快路径。
"""

import math
import sys

try:
    import torch
except ImportError as e:
    print("错误: 未找到 torch 模块。", file=sys.stderr)
    print(f"详细错误: {e}", file=sys.stderr)
    sys.exit(1)

try:
    from diffusers import DiffusionPipeline, FlowMatchEulerDiscreteScheduler
except ImportError as e:
    print("错误: 未找到 diffusers 中的 Qwen-Image 相关模块。", file=sys.stderr)
    print("请安装较新的 diffusers 版本（Qwen-Image-Lightning 建议直接使用 main 分支）。", file=sys.stderr)
    print(f"详细错误: {e}", file=sys.stderr)
    sys.exit(1)

from PIL import Image
from typing import Optional, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


DEFAULT_BASE_MODEL_ID = "Qwen/Qwen-Image"
DEFAULT_LORA_REPO = "lightx2v/Qwen-Image-Lightning"
DEFAULT_WEIGHT_NAME = "Qwen-Image-Lightning-4steps-V2.0.safetensors"


class QwenImageLightningGenerator:
    """Qwen-Image-Lightning 图像生成器类。"""

    def __init__(
        self,
        model_path: str = DEFAULT_BASE_MODEL_ID,
        lora_repo: str = DEFAULT_LORA_REPO,
        weight_name: str = DEFAULT_WEIGHT_NAME,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        use_low_mem: bool = False,
    ):
        self.device = device
        self.dtype = dtype
        self.model_path = model_path or DEFAULT_BASE_MODEL_ID
        self.lora_repo = lora_repo
        self.weight_name = weight_name
        self.use_low_mem = use_low_mem

        logger.info(f"加载 Qwen-Image base: {self.model_path}")
        logger.info(f"加载 Qwen-Image-Lightning LoRA: {self.lora_repo} / {self.weight_name}")

        scheduler = FlowMatchEulerDiscreteScheduler.from_config(
            {
                "base_image_seq_len": 256,
                "base_shift": math.log(3),
                "invert_sigmas": False,
                "max_image_seq_len": 8192,
                "max_shift": math.log(3),
                "num_train_timesteps": 1000,
                "shift": 3.0,
                "shift_terminal": None,
                "stochastic_sampling": False,
                "time_shift_type": "exponential",
                "use_beta_sigmas": False,
                "use_dynamic_shifting": False,
                "use_exponential_sigmas": False,
                "use_karras_sigmas": False,
            }
        )

        try:
            self.pipe = DiffusionPipeline.from_pretrained(
                self.model_path,
                scheduler=scheduler,
                torch_dtype=dtype,
            )

            if use_low_mem and hasattr(self.pipe, "enable_model_cpu_offload"):
                self.pipe.enable_model_cpu_offload()
                logger.info("Qwen-Image-Lightning 已启用 CPU offload")
            else:
                self.pipe = self.pipe.to(device)

            if hasattr(self.pipe, "enable_attention_slicing"):
                self.pipe.enable_attention_slicing()
            if hasattr(self.pipe, "enable_vae_tiling"):
                self.pipe.enable_vae_tiling()
            if hasattr(self.pipe, "enable_vae_slicing"):
                self.pipe.enable_vae_slicing()

            self.pipe.load_lora_weights(self.lora_repo, weight_name=self.weight_name)
            logger.info("Qwen-Image-Lightning 模型加载成功")
        except Exception as exc:
            logger.error(f"加载 Qwen-Image-Lightning 失败: {exc}")
            raise

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = 4,
        guidance_scale: float = 1.0,
        width: int = 1024,
        height: int = 1024,
        seed: Optional[int] = None,
        **kwargs,
    ) -> tuple[Image.Image, Dict[str, Any]]:
        if negative_prompt:
            logger.debug("Qwen-Image-Lightning 当前实现忽略 negative_prompt")

        generator = None
        if seed is not None:
            generator = torch.Generator(device="cpu").manual_seed(seed)

        logger.info(f"Qwen-Image-Lightning 生成中 (steps={num_inference_steps}, seed={seed})")

        result = self.pipe(
            prompt=prompt,
            true_cfg_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            width=width,
            height=height,
            generator=generator,
        )

        image = result.images[0]
        gen_info = {
            "model": "qwen-image-lightning",
            "base_model": self.model_path,
            "lora_repo": self.lora_repo,
            "weight_name": self.weight_name,
            "prompt": prompt,
            "steps": num_inference_steps,
            "true_cfg_scale": guidance_scale,
            "width": width,
            "height": height,
            "seed": seed,
            "use_low_mem": self.use_low_mem,
        }

        return image, gen_info

    def cleanup(self):
        if hasattr(self, "pipe"):
            del self.pipe
            torch.cuda.empty_cache()
            logger.info("Qwen-Image-Lightning 生成器资源已清理")
