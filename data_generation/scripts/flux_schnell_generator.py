"""
Flux.1-schnell 图像生成器
基于 diffusers 库，仅需 4 步即可生成高质量图像
官方模型：black-forest-labs/FLUX.1-schnell

支持两种模式：
  - 标准模式：使用 CPU offload，兼容性好
  - 优化模式：T5 4-bit + Transformer FP8 + torch.compile，速度快 2-3x
"""

import sys

try:
    import torch
except ImportError as e:
    print("错误: 未找到 torch 模块。", file=sys.stderr)
    print("请确保使用正确的 conda 环境运行此脚本。", file=sys.stderr)
    print(f"详细错误: {e}", file=sys.stderr)
    sys.exit(1)

try:
    from diffusers import FluxPipeline, FluxTransformer2DModel
    from transformers import T5EncoderModel
except ImportError as e:
    print("错误: 未找到 diffusers 模块。", file=sys.stderr)
    print("请安装: pip install -U diffusers", file=sys.stderr)
    print(f"详细错误: {e}", file=sys.stderr)
    sys.exit(1)

# 检查量化库
try:
    import bitsandbytes as bnb
    from transformers import BitsAndBytesConfig
    BNB_AVAILABLE = True
except ImportError:
    BNB_AVAILABLE = False

try:
    from torchao.quantization import quantize_, float8_weight_only
    TORCHAO_AVAILABLE = True
except ImportError:
    TORCHAO_AVAILABLE = False

from PIL import Image
import os
import gc
from typing import Optional, Dict, Any, Literal
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_HF_CACHE = "/root/autodl-tmp/huggingface"
DEFAULT_LOCAL_MODEL_PATH = "/root/autodl-tmp/flux-schnell"


def setup_hf_cache(cache_dir: str = DEFAULT_HF_CACHE):
    """设置 HuggingFace 缓存目录"""
    os.environ["HF_HOME"] = cache_dir
    os.environ["HF_HUB_CACHE"] = os.path.join(cache_dir, "hub")
    os.makedirs(cache_dir, exist_ok=True)


class FluxSchnellGenerator:
    """Flux.1-schnell 图像生成器类"""

    def __init__(
        self,
        model_path: str = DEFAULT_LOCAL_MODEL_PATH,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        optimize_for_speed: bool = False,
        enable_compile: bool = False
    ):
        """
        初始化 Flux.1-schnell 生成器

        Args:
            model_path: 本地模型路径或 HuggingFace 模型 ID
            device: 运行设备 (cuda/cpu)
            dtype: 数据类型 (推荐 bfloat16)
            optimize_for_speed: 启用速度优化模式（T5 4-bit + Transformer FP8，需要 ~15GB 显存）
            enable_compile: 启用 torch.compile（仅在 optimize_for_speed=True 时生效）
        """
        self.device = device
        self.dtype = dtype
        self.model_path = model_path
        self.optimize_for_speed = optimize_for_speed
        self.enable_compile = enable_compile and optimize_for_speed

        setup_hf_cache()

        logger.info(f"加载 Flux.1-schnell 模型: {model_path}")
        if optimize_for_speed:
            logger.info("优化模式: T5 4-bit + Transformer FP8")

        torch.cuda.empty_cache()
        gc.collect()

        try:
            if optimize_for_speed:
                self._load_optimized_pipeline(model_path, dtype)
            else:
                self._load_standard_pipeline(model_path, dtype)

            self._enable_optimizations()
            logger.info("✓ Flux.1-schnell 模型加载成功")

        except Exception as e:
            logger.error(f"加载 Flux.1-schnell 模型失败: {e}")
            raise

    def _load_standard_pipeline(self, model_path: str, dtype: torch.dtype):
        """标准加载模式（使用 CPU offload）"""
        if os.path.isdir(model_path):
            logger.info("从本地目录加载模型...")
            self.pipe = FluxPipeline.from_pretrained(
                model_path,
                torch_dtype=dtype,
                local_files_only=True
            )
        else:
            logger.info("从 HuggingFace 下载模型...")
            self.pipe = FluxPipeline.from_pretrained(
                model_path,
                torch_dtype=dtype
            )

        self.pipe.enable_model_cpu_offload()
        logger.info("✓ CPU offload 已启用")

    def _load_optimized_pipeline(self, model_path: str, dtype: torch.dtype):
        """优化加载模式（T5 4-bit + Transformer FP8，无 CPU offload）"""
        if not BNB_AVAILABLE:
            raise ImportError("优化模式需要 bitsandbytes，请安装: pip install bitsandbytes")

        # 1. 加载 4-bit 量化的 T5
        logger.info("加载 T5 (4-bit 量化)...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True
        )
        text_encoder_2 = T5EncoderModel.from_pretrained(
            model_path,
            subfolder="text_encoder_2",
            quantization_config=bnb_config,
            torch_dtype=dtype,
            local_files_only=os.path.isdir(model_path)
        )
        logger.info("✓ T5 4-bit 加载完成")

        # 2. 加载 Pipeline（使用量化的 T5）
        logger.info("加载 Pipeline...")
        if os.path.isdir(model_path):
            self.pipe = FluxPipeline.from_pretrained(
                model_path,
                text_encoder_2=text_encoder_2,
                torch_dtype=dtype,
                local_files_only=True
            )
        else:
            self.pipe = FluxPipeline.from_pretrained(
                model_path,
                text_encoder_2=text_encoder_2,
                torch_dtype=dtype
            )

        # 3. Transformer FP8 量化（可选）
        if TORCHAO_AVAILABLE:
            logger.info("应用 Transformer FP8 量化...")
            quantize_(self.pipe.transformer, float8_weight_only())
            logger.info("✓ Transformer FP8 量化完成")

        # 4. 移动到 GPU（不使用 CPU offload）
        logger.info("移动模型到 GPU...")
        self.pipe.to(self.device)
        torch.cuda.empty_cache()
        gc.collect()

        # 显示显存使用
        allocated = torch.cuda.memory_allocated() / 1024**3
        logger.info(f"✓ 显存使用: {allocated:.1f} GB")

        # 5. torch.compile（可选）
        if self.enable_compile:
            logger.info("编译 Transformer（首次运行需要 2-3 分钟）...")
            self.pipe.transformer = torch.compile(
                self.pipe.transformer,
                mode="reduce-overhead",
                fullgraph=True
            )
            logger.info("✓ Transformer 编译完成")

    def _enable_optimizations(self):
        """启用加速优化"""
        if hasattr(self.pipe, 'enable_attention_slicing'):
            self.pipe.enable_attention_slicing()

        try:
            self.pipe.enable_xformers_memory_efficient_attention()
            logger.info("✓ xFormers 已启用")
        except Exception:
            pass

        if hasattr(self.pipe, 'enable_vae_slicing'):
            self.pipe.enable_vae_slicing()

        if hasattr(self.pipe, 'enable_vae_tiling'):
            self.pipe.enable_vae_tiling()

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = 4,
        guidance_scale: float = 0.0,
        width: int = 1024,
        height: int = 1024,
        seed: Optional[int] = None,
        max_sequence_length: int = 512,
        **kwargs
    ) -> tuple[Image.Image, Dict[str, Any]]:
        """
        生成图像

        Args:
            prompt: 正向提示词
            negative_prompt: 负向提示词（Schnell 不支持，将被忽略）
            num_inference_steps: 推理步数（推荐 4）
            guidance_scale: CFG scale（必须为 0.0）
            width: 图像宽度
            height: 图像高度
            seed: 随机种子
            max_sequence_length: 最大序列长度（默认 256）
            **kwargs: 其他参数

        Returns:
            (image, generation_info) 元组
        """
        if negative_prompt:
            logger.debug("Flux.1-schnell 不支持 negative_prompt，已忽略")

        if guidance_scale != 0.0:
            logger.warning(f"Flux.1-schnell 要求 guidance_scale=0.0，当前值 {guidance_scale} 将被覆盖")
            guidance_scale = 0.0

        generator = None
        if seed is not None:
            generator = torch.Generator(device="cpu").manual_seed(seed)

        logger.info(f"Schnell 生成中 (steps={num_inference_steps}, seed={seed})")
        logger.info(f"Prompt: {prompt[:100]}...")

        torch.cuda.empty_cache()

        result = self.pipe(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            max_sequence_length=max_sequence_length,
            generator=generator
        )

        image = result.images[0]

        torch.cuda.empty_cache()

        gen_info = {
            "model": "flux-1-schnell" + ("-optimized" if self.optimize_for_speed else ""),
            "prompt": prompt,
            "negative_prompt": "",
            "steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "width": width,
            "height": height,
            "seed": seed,
            "max_sequence_length": max_sequence_length
        }

        logger.info("Schnell 生成完成")

        return image, gen_info

    def cleanup(self):
        """清理资源"""
        if hasattr(self, 'pipe'):
            del self.pipe
            torch.cuda.empty_cache()
            gc.collect()
            logger.info("Flux.1-schnell 生成器资源已清理")


if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser(description="Flux.1-schnell 图像生成器测试")
    parser.add_argument("--prompt", type=str,
                        default="a beautiful sunset over the ocean, masterpiece",
                        help="测试提示词")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--steps", type=int, default=4, help="推理步数")
    parser.add_argument("--output", type=str, default="/tmp/flux_schnell_test.png",
                        help="输出路径")
    parser.add_argument("--optimize", action="store_true",
                        help="启用速度优化模式（T5 4-bit + Transformer FP8）")
    parser.add_argument("--compile", action="store_true",
                        help="启用 torch.compile（需配合 --optimize）")
    args = parser.parse_args()

    print(f"Flux.1-schnell 测试")
    print(f"  Prompt: {args.prompt}")
    print(f"  Steps: {args.steps}")
    print(f"  Seed: {args.seed}")
    print(f"  Optimize: {args.optimize}")
    print(f"  Compile: {args.compile}")

    generator = FluxSchnellGenerator(
        optimize_for_speed=args.optimize,
        enable_compile=args.compile
    )

    # 预热（如果启用 compile）
    if args.compile:
        print("\n预热中（编译）...")
        warmup_start = time.time()
        _, _ = generator.generate(prompt="warmup", seed=0)
        warmup_time = time.time() - warmup_start
        print(f"预热完成: {warmup_time:.1f}s")

    # 正式生成
    print("\n正式生成...")
    start_time = time.time()
    image, info = generator.generate(
        prompt=args.prompt,
        seed=args.seed,
        num_inference_steps=args.steps
    )
    gen_time = time.time() - start_time

    image.save(args.output)
    print(f"\n测试图像已保存到: {args.output}")
    print(f"生成时间: {gen_time:.2f}s")
    print(f"生成信息: {info}")

    # 显示显存使用
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"显存: {allocated:.1f} GB (allocated) / {reserved:.1f} GB (reserved)")

    generator.cleanup()
