"""
Qwen-Image-Lightning 图像生成器
默认对齐官方 diffusers 加载方式；实验量化路径仅保留为可选分支。
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
    from diffusers import DiffusionPipeline, FlowMatchEulerDiscreteScheduler, PipelineQuantizationConfig
    from diffusers.models import QwenImageTransformer2DModel
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


DEFAULT_BASE_MODEL_ID = "/root/autodl-tmp/AGIQA/Qwen-Image/snapshots/75e0b4be04f60ec59a75f475837eced720f823b6"
DEFAULT_LORA_REPO = "/root/autodl-tmp/AGIQA/Qwen-Image-Lightning"
DEFAULT_WEIGHT_NAME = "Qwen-Image-Lightning-4steps-V2.0.safetensors"
DEFAULT_NUNCHAKU_MODEL = "nunchaku-ai/nunchaku-qwen-image/svdq-int4_r32-qwen-image-lightningv1.0-4steps.safetensors"


class QwenImageLightningGenerator:
    """Qwen-Image-Lightning 图像生成器类。"""

    def _enable_low_mem_runtime(self) -> None:
        if hasattr(self.pipe, "enable_model_cpu_offload"):
            self.pipe.enable_model_cpu_offload()
            logger.info("Qwen-Image-Lightning 已启用 model CPU offload")
            return
        self.pipe.enable_sequential_cpu_offload()
        logger.info("Qwen-Image-Lightning 已启用 sequential CPU offload")

    def _load_nunchaku_pipeline(
        self,
        scheduler: FlowMatchEulerDiscreteScheduler,
        dtype: torch.dtype,
    ) -> None:
        try:
            from diffusers import QwenImagePipeline
            from nunchaku.models.transformers.transformer_qwenimage import (
                NunchakuQwenImageTransformer2DModel,
            )
        except ImportError as exc:
            raise ImportError(
                "nunchaku-int4 运行档位需要安装 Nunchaku 及支持 QwenImagePipeline 的 diffusers。"
            ) from exc

        transformer = NunchakuQwenImageTransformer2DModel.from_pretrained(
            self.nunchaku_model_path
        )
        self.pipe = QwenImagePipeline.from_pretrained(
            self.model_path,
            transformer=transformer,
            scheduler=scheduler,
            torch_dtype=dtype,
        )
        self.pipe.enable_model_cpu_offload()
        logger.info(
            "Qwen-Image-Lightning 使用 Nunchaku INT4 融合 4-step checkpoint: %s",
            self.nunchaku_model_path,
        )

    def __init__(
        self,
        model_path: str = DEFAULT_BASE_MODEL_ID,
        lora_repo: str = DEFAULT_LORA_REPO,
        weight_name: str = DEFAULT_WEIGHT_NAME,
        nunchaku_model_path: str = DEFAULT_NUNCHAKU_MODEL,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        use_nf4: bool = False,
        use_low_mem: bool = False,
        runtime_profile: str = "fit-24g",
    ):
        self.device = device
        self.dtype = dtype
        self.model_path = model_path or DEFAULT_BASE_MODEL_ID
        self.lora_repo = lora_repo
        self.weight_name = weight_name
        self.nunchaku_model_path = nunchaku_model_path or DEFAULT_NUNCHAKU_MODEL
        self.use_nf4 = use_nf4
        self.use_low_mem = use_low_mem
        self.runtime_profile = runtime_profile

        logger.info(f"加载 Qwen-Image base: {self.model_path}")
        if runtime_profile == "nunchaku-int4":
            logger.info(
                "加载 Qwen-Image-Lightning Nunchaku INT4 fused checkpoint: %s",
                self.nunchaku_model_path,
            )
        else:
            logger.info(f"加载 Qwen-Image-Lightning LoRA: {self.lora_repo} / {self.weight_name}")

        scheduler = FlowMatchEulerDiscreteScheduler.from_config(
            {
                "base_image_seq_len": 256,
                "base_shift": math.log(3),
                "invert_sigmas": False,
                "max_image_seq_len": 8192,
                "max_shift": math.log(3),
                "num_train_timesteps": 1000,
                "shift": 1.0,
                "shift_terminal": None,
                "stochastic_sampling": False,
                "time_shift_type": "exponential",
                "use_beta_sigmas": False,
                "use_dynamic_shifting": True,
                "use_exponential_sigmas": False,
                "use_karras_sigmas": False,
            }
        )

        try:
            if runtime_profile == "nunchaku-int4":
                self._load_nunchaku_pipeline(scheduler=scheduler, dtype=dtype)
                self.use_low_mem = True
            elif runtime_profile == "fast-gpu-24g" and not use_low_mem and not use_nf4:
                model = QwenImageTransformer2DModel.from_pretrained(
                    self.model_path,
                    subfolder="transformer",
                    torch_dtype=dtype,
                )
                self.pipe = DiffusionPipeline.from_pretrained(
                    self.model_path,
                    transformer=model,
                    scheduler=scheduler,
                    torch_dtype=dtype,
                )
                logger.info("Qwen-Image-Lightning 使用 fast-gpu-24g 加载路径")
            else:
                load_kwargs = {
                    "scheduler": scheduler,
                    "torch_dtype": dtype,
                }

                if use_nf4:
                    load_kwargs["quantization_config"] = PipelineQuantizationConfig(
                        quant_backend="bitsandbytes_4bit",
                        quant_kwargs={
                            "load_in_4bit": True,
                            "bnb_4bit_compute_dtype": dtype,
                            "bnb_4bit_quant_type": "nf4",
                        },
                        components_to_quantize=["transformer", "text_encoder"],
                    )
                    logger.info("Qwen-Image-Lightning 使用实验性 scoped nf4 量化加载")

                self.pipe = DiffusionPipeline.from_pretrained(
                    self.model_path, **load_kwargs
                )

            if runtime_profile != "nunchaku-int4":
                self.pipe.load_lora_weights(self.lora_repo, weight_name=self.weight_name)
                logger.info(f"Qwen-Image-Lightning LoRA 已加载: {self.weight_name}")

            if runtime_profile == "nunchaku-int4":
                self.pipe._exclude_from_cpu_offload.append("transformer")
            elif use_low_mem:
                self._enable_low_mem_runtime()
            else:
                try:
                    self.pipe = self.pipe.to(device)
                    logger.info("Qwen-Image-Lightning 已将整条 pipeline 放入 GPU")
                except torch.OutOfMemoryError:
                    if runtime_profile != "fast-gpu-24g":
                        raise
                    logger.warning("Qwen-Image-Lightning fast-gpu-24g 在当前 GPU 上 OOM，自动回退到 offload")
                    torch.cuda.empty_cache()
                    self.use_low_mem = True
                    self.runtime_profile = "fit-24g"
                    self._enable_low_mem_runtime()

            if hasattr(self.pipe, "vae"):
                if hasattr(self.pipe.vae, "enable_tiling"):
                    self.pipe.vae.enable_tiling()
                if hasattr(self.pipe.vae, "enable_slicing"):
                    self.pipe.vae.enable_slicing()
            else:
                if hasattr(self.pipe, "enable_vae_tiling"):
                    self.pipe.enable_vae_tiling()
                if hasattr(self.pipe, "enable_vae_slicing"):
                    self.pipe.enable_vae_slicing()

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
            "use_nf4": self.use_nf4,
            "use_low_mem": self.use_low_mem,
            "runtime_profile": self.runtime_profile,
            "backend": "nunchaku" if self.runtime_profile == "nunchaku-int4" else "diffusers",
            "fused_checkpoint": (
                self.nunchaku_model_path if self.runtime_profile == "nunchaku-int4" else None
            ),
        }

        return image, gen_info

    def cleanup(self):
        if hasattr(self, "pipe"):
            del self.pipe
            torch.cuda.empty_cache()
            logger.info("Qwen-Image-Lightning 生成器资源已清理")
