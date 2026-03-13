"""
SD3.5 Large Turbo 图像生成器
先实现稳定的官方基础加载路径，不复用通用 SD3.5 的量化实现。
"""

import logging
import os
import sys
from typing import Optional

try:
    import torch
except ImportError as e:
    print("错误: 未找到 torch 模块。", file=sys.stderr)
    print(f"详细错误: {e}", file=sys.stderr)
    sys.exit(1)

try:
    from diffusers import StableDiffusion3Pipeline
except ImportError as e:
    print("错误: 未找到 diffusers 中的 SD3 相关模块。", file=sys.stderr)
    print("请安装较新的 diffusers 版本。", file=sys.stderr)
    print(f"详细错误: {e}", file=sys.stderr)
    sys.exit(1)

try:
    from transformers import BitsAndBytesConfig, T5EncoderModel
    BNB_AVAILABLE = True
except ImportError:
    BitsAndBytesConfig = None
    T5EncoderModel = None
    BNB_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


DEFAULT_TURBO_MODEL_ID = "stabilityai/stable-diffusion-3.5-large-turbo"


class SD35LargeTurboGenerator:
    """Stable Diffusion 3.5 Large Turbo 生成器类。"""

    def __init__(
        self,
        model_path: str = DEFAULT_TURBO_MODEL_ID,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        runtime_profile: str = "fast-gpu",
        use_cpu_offload: bool = False,
        prefer_quantized: bool = False,
    ):
        self.device = device
        self.dtype = dtype
        self.runtime_profile = runtime_profile
        self.model_path = model_path
        self.use_cpu_offload = use_cpu_offload or runtime_profile == "fit-24g"
        self.use_group_offload = self.use_cpu_offload and runtime_profile == "fit-24g"
        self.use_8bit_text_encoder_3 = runtime_profile == "fit-24g"
        # 暂时统一走稳定的非量化路径，后续再单独恢复 Turbo 低显存量化实现。
        self.prefer_quantized = bool(prefer_quantized and runtime_profile == "experimental")

        load_path = model_path or DEFAULT_TURBO_MODEL_ID
        is_local = bool(load_path and os.path.isdir(load_path))
        if not is_local and load_path != DEFAULT_TURBO_MODEL_ID and not os.path.exists(load_path):
            logger.warning(f"本地 Turbo 模型不存在: {load_path}，回退到默认模型 ID: {DEFAULT_TURBO_MODEL_ID}")
            load_path = DEFAULT_TURBO_MODEL_ID
            is_local = False

        logger.info(f"加载 SD3.5 Large Turbo 模型: {load_path}")

        pipe_kwargs = {
            "torch_dtype": dtype,
            "local_files_only": is_local,
        }

        text_encoder_3 = self._load_8bit_text_encoder_3(load_path, is_local)
        if text_encoder_3 is not None:
            pipe_kwargs["text_encoder_3"] = text_encoder_3

        self.pipe = StableDiffusion3Pipeline.from_pretrained(
            load_path,
            **pipe_kwargs,
        )

        if self.use_cpu_offload:
            self._enable_low_memory_execution()
        else:
            self.pipe = self.pipe.to(device)

        # Turbo 在 24GB 卡上的主路径已经依赖 CPU offload；默认再叠加这些省显存开关
        # 会明显拖慢 4-step 推理，所以只在后续需要时再显式开启。

        logger.info("SD3.5 Large Turbo 模型加载成功")

    def _load_8bit_text_encoder_3(self, load_path: str, is_local: bool):
        if not self.use_8bit_text_encoder_3:
            return None
        if not BNB_AVAILABLE:
            logger.warning("未安装 bitsandbytes/transformers 量化依赖，跳过 8-bit text_encoder_3")
            return None

        try:
            quant_config = BitsAndBytesConfig(load_in_8bit=True)
            text_encoder_3 = T5EncoderModel.from_pretrained(
                load_path,
                subfolder="text_encoder_3",
                quantization_config=quant_config,
                torch_dtype=self.dtype,
                local_files_only=is_local,
            )
            logger.info("SD3.5 Large Turbo 已启用 8-bit text_encoder_3")
            return text_encoder_3
        except Exception as exc:
            logger.warning(f"SD3.5 Large Turbo 启用 8-bit text_encoder_3 失败，回退到默认文本编码器: {exc}")
            return None

    def _enable_low_memory_execution(self) -> None:
        if self.use_group_offload and hasattr(self.pipe, "enable_group_offload"):
            try:
                self.pipe.enable_group_offload(
                    onload_device=torch.device(self.device),
                    offload_type="block_level",
                    num_blocks_per_group=1,
                    use_stream=True,
                )
                logger.info("SD3.5 Large Turbo 已启用 group offload")
                return
            except Exception as exc:
                logger.warning(f"SD3.5 Large Turbo 启用 group offload 失败，回退到 CPU offload: {exc}")

        self.pipe.enable_model_cpu_offload()
        logger.info("SD3.5 Large Turbo 已启用 CPU offload")

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = 4,
        guidance_scale: float = 0.0,
        width: int = 1024,
        height: int = 1024,
        seed: Optional[int] = None,
        max_sequence_length: int = 256,
        **kwargs,
    ) -> tuple:
        if self.runtime_profile != "fit-24g" and "max_sequence_length" not in kwargs:
            max_sequence_length = 512

        generator = None
        if seed is not None:
            generator = torch.Generator(device="cpu").manual_seed(seed)

        logger.info(f"SD3.5 Large Turbo 生成中 (steps={num_inference_steps}, seed={seed})")
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
            "model": "sd3.5-large-turbo",
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "width": width,
            "height": height,
            "seed": seed,
            "use_cpu_offload": self.use_cpu_offload,
            "use_group_offload": self.use_group_offload,
            "use_8bit_text_encoder_3": self.use_8bit_text_encoder_3,
            "prefer_quantized": self.prefer_quantized,
            "runtime_profile": self.runtime_profile,
        }
        return image, gen_info

    def cleanup(self):
        if hasattr(self, "pipe"):
            del self.pipe
            torch.cuda.empty_cache()
            logger.info("SD3.5 Large Turbo 生成器资源已清理")
