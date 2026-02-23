"""
Flux.1-dev 图像生成器
使用 diffusers 库调用 Flux.1-dev 模型生成图像
支持多种量化模式：
  - official: 官方预量化模型（推荐，加载最快）
  - local_fp8: 本地 FP8 模型 + 运行时量化
  - auto: 自动选择最佳模式
"""

import sys

try:
    import torch
except ImportError as e:
    print("错误: 未找到 torch 模块。", file=sys.stderr)
    print("请确保使用正确的 conda 环境运行此脚本。", file=sys.stderr)
    print("建议使用: /root/miniconda3/envs/3.10/bin/python", file=sys.stderr)
    print("或者运行: conda activate 3.10", file=sys.stderr)
    print(f"详细错误: {e}", file=sys.stderr)
    sys.exit(1)

try:
    from diffusers import FluxPipeline, FluxTransformer2DModel, AutoencoderKL, FlowMatchEulerDiscreteScheduler
    from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
except ImportError as e:
    print("错误: 未找到 diffusers 或 transformers 模块。", file=sys.stderr)
    print("请安装: pip install diffusers transformers", file=sys.stderr)
    print(f"详细错误: {e}", file=sys.stderr)
    sys.exit(1)

# 检查 torchao 是否可用
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

# 官方预量化模型配置
OFFICIAL_QUANTIZED_MODELS = {
    "fp8": "diffusers/FLUX.1-dev-torchao-fp8",
    "int8": "diffusers/FLUX.1-dev-torchao-int8",
    "bnb-4bit": "diffusers/FLUX.1-dev-bnb-4bit",
    "bnb-8bit": "diffusers/FLUX.1-dev-bnb-8bit",
}

# 默认缓存目录（大容量磁盘）
DEFAULT_HF_CACHE = "/root/autodl-tmp/huggingface"


def setup_hf_cache(cache_dir: str = DEFAULT_HF_CACHE):
    """设置 HuggingFace 缓存目录"""
    os.environ["HF_HOME"] = cache_dir
    os.environ["HF_HUB_CACHE"] = os.path.join(cache_dir, "hub")
    os.makedirs(cache_dir, exist_ok=True)


class FluxGenerator:
    """Flux.1-dev 图像生成器类"""

    def __init__(
        self,
        model_path: str = "/root/autodl-tmp/flux-fp8",
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        quantization_mode: Literal["auto", "official", "local_fp8"] = "auto",
        official_quant_type: Literal["fp8", "int8", "bnb-4bit", "bnb-8bit"] = "int8"
    ):
        """
        初始化 Flux.1-dev 生成器

        Args:
            model_path: 本地 Flux 模型路径（仅 local_fp8 模式使用）
            device: 运行设备 (cuda/cpu)
            dtype: 数据类型 (推荐 bfloat16)
            quantization_mode: 量化模式
                - "auto": 自动选择（优先官方预量化）
                - "official": 使用官方预量化模型（加载最快）
                - "local_fp8": 使用本地 FP8 模型 + 运行时量化
            official_quant_type: 官方预量化类型（仅 official/auto 模式）
                - "fp8": TorchAO FP8（推荐，速度快）
                - "int8": TorchAO INT8（最流行，质量好）
                - "bnb-4bit": BitsAndBytes 4-bit（显存最小）
                - "bnb-8bit": BitsAndBytes 8-bit
        """
        self.device = device
        self.dtype = dtype
        self.model_path = model_path
        self.quantization_mode = quantization_mode
        self.official_quant_type = official_quant_type

        # 自动选择模式
        if quantization_mode == "auto":
            # 优先使用官方预量化模型（加载更快）
            quantization_mode = "official"
            logger.info("自动选择模式：使用官方预量化模型（加载更快）")

        # 加载 pipeline
        try:
            if quantization_mode == "official":
                self.pipe = self._load_official_quantized(official_quant_type)
            else:  # local_fp8
                self.pipe = self._load_local_fp8(model_path, dtype)

            # 启用加速优化
            self._enable_optimizations()

            logger.info("Flux.1-dev 模型加载成功")

        except Exception as e:
            logger.error(f"加载 Flux.1-dev 模型失败: {e}")
            raise

    def _load_official_quantized(self, quant_type: str) -> FluxPipeline:
        """
        加载官方预量化模型（推荐，加载最快）

        Args:
            quant_type: 量化类型 (fp8/int8/bnb-4bit/bnb-8bit)

        Returns:
            FluxPipeline 实例
        """
        # 设置缓存目录
        setup_hf_cache()

        model_id = OFFICIAL_QUANTIZED_MODELS.get(quant_type)
        if not model_id:
            raise ValueError(f"未知的量化类型: {quant_type}，可选: {list(OFFICIAL_QUANTIZED_MODELS.keys())}")

        logger.info(f"加载官方预量化模型: {model_id}")
        logger.info(f"缓存目录: {DEFAULT_HF_CACHE}")

        torch.cuda.empty_cache()
        gc.collect()

        # torchao 模型需要 use_safetensors=False
        is_torchao = quant_type in ["fp8", "int8"]

        if is_torchao:
            pipe = FluxPipeline.from_pretrained(
                model_id,
                torch_dtype=self.dtype,
                use_safetensors=False,  # torchao 模型必须设置
                device_map="balanced"
            )
            logger.info(f"✓ TorchAO {quant_type.upper()} 预量化模型加载完成")
        else:
            # bitsandbytes 模型
            pipe = FluxPipeline.from_pretrained(
                model_id,
                torch_dtype=self.dtype,
                device_map="balanced"
            )
            logger.info(f"✓ BitsAndBytes {quant_type} 预量化模型加载完成")

        torch.cuda.empty_cache()
        gc.collect()

        return pipe

    def _load_local_fp8(self, model_path: str, dtype: torch.dtype) -> FluxPipeline:
        """
        加载本地 FP8 模型（需要运行时量化）
        """
        logger.info(f"加载本地 FP8 模型: {model_path}")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"本地模型不存在: {model_path}")

        if TORCHAO_AVAILABLE:
            return self._load_fp8_pipeline(model_path, dtype)
        else:
            logger.warning("torchao 未安装，使用 CPU offload 模式")
            return self._load_fp8_pipeline_cpu_offload(model_path, dtype)

    def _enable_optimizations(self):
        """启用加速优化"""
        logger.info("启用加速优化...")

        if hasattr(self.pipe, 'enable_attention_slicing'):
            self.pipe.enable_attention_slicing()
            logger.info("✓ Attention slicing 已启用")

        try:
            self.pipe.enable_xformers_memory_efficient_attention()
            logger.info("✓ xFormers 内存高效注意力已启用")
        except Exception as e:
            logger.debug(f"xFormers 不可用: {str(e)[:50]}")

        if hasattr(self.pipe, 'enable_vae_slicing'):
            self.pipe.enable_vae_slicing()
            logger.info("✓ VAE slicing 已启用")

        if hasattr(self.pipe, 'enable_vae_tiling'):
            self.pipe.enable_vae_tiling()
            logger.info("✓ VAE tiling 已启用")

    def _load_fp8_pipeline(self, model_path: str, dtype: torch.dtype) -> FluxPipeline:
        """
        优化的 FP8 版本 Flux pipeline 加载

        使用 torchao FP8 量化实现真正的 8-bit 推理加速

        Args:
            model_path: FP8 模型目录路径
            dtype: 数据类型

        Returns:
            FluxPipeline 实例
        """
        # 检查 torchao 是否可用
        if not TORCHAO_AVAILABLE:
            logger.warning("torchao 未安装，将使用 CPU offload 模式（速度较慢）")
            return self._load_fp8_pipeline_cpu_offload(model_path, dtype)

        logger.info("使用 torchao FP8 量化加载模型...")

        # 清理显存
        torch.cuda.empty_cache()
        gc.collect()

        # 1. 加载 Text Encoder 1 (CLIP) - 较小，直接移到 GPU
        logger.info("加载 Text Encoder 1 (CLIP)...")
        text_encoder = CLIPTextModel.from_pretrained(
            model_path, subfolder="text_encoder", torch_dtype=dtype
        ).to(self.device)
        torch.cuda.empty_cache()
        logger.info("✓ Text Encoder 1 加载完成")

        # 2. 加载 Text Encoder 2 (T5) - 在 CPU 上量化后移到 GPU
        logger.info("加载 Text Encoder 2 (T5)...")
        text_encoder_2 = T5EncoderModel.from_pretrained(
            model_path, subfolder="text_encoder_2", torch_dtype=dtype
        )
        logger.info("  应用 FP8 量化...")
        quantize_(text_encoder_2, float8_weight_only())
        text_encoder_2 = text_encoder_2.to(self.device)
        torch.cuda.empty_cache()
        gc.collect()
        logger.info("✓ Text Encoder 2 (FP8) 加载完成")

        # 3. 加载 Tokenizers
        logger.info("加载 Tokenizers...")
        tokenizer = CLIPTokenizer.from_pretrained(
            model_path, subfolder="tokenizer"
        )
        tokenizer_2 = T5TokenizerFast.from_pretrained(
            model_path, subfolder="tokenizer_2"
        )
        logger.info("✓ Tokenizers 加载完成")

        # 4. 加载 VAE - 使用 from_single_file
        logger.info("加载 VAE...")
        vae_path = os.path.join(model_path, "flux-vae-bf16.safetensors")
        vae_config_path = os.path.join(model_path, "vae", "config.json")
        vae = AutoencoderKL.from_single_file(
            vae_path,
            config=vae_config_path,
            torch_dtype=dtype
        ).to(self.device)
        torch.cuda.empty_cache()
        logger.info("✓ VAE 加载完成")

        # 5. 加载 Scheduler
        logger.info("加载 Scheduler...")
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model_path, subfolder="scheduler"
        )
        logger.info("✓ Scheduler 加载完成")

        # 6. 加载 Transformer - 在 CPU 上量化后移到 GPU
        logger.info("加载 Transformer (FP8)...")
        transformer_path = os.path.join(model_path, "flux1-dev-fp8.safetensors")
        config_path = os.path.join(model_path, "transformer_config.json")

        transformer = FluxTransformer2DModel.from_single_file(
            transformer_path,
            config=config_path,
            torch_dtype=dtype
        )
        logger.info("  应用 FP8 量化...")
        quantize_(transformer, float8_weight_only())
        transformer = transformer.to(self.device)
        torch.cuda.empty_cache()
        gc.collect()
        logger.info("✓ Transformer (FP8) 加载完成")

        # 7. 手动组装 Pipeline
        logger.info("组装 Pipeline...")
        pipe = FluxPipeline(
            scheduler=scheduler,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            vae=vae,
            transformer=transformer
        )
        logger.info("✓ FP8 Pipeline 组装完成")

        # 清理显存
        torch.cuda.empty_cache()
        gc.collect()

        return pipe

    def _load_fp8_pipeline_cpu_offload(self, model_path: str, dtype: torch.dtype) -> FluxPipeline:
        """
        备用方案：使用 CPU offload 加载 FP8 模型（速度较慢但显存占用低）
        """
        logger.info("使用 CPU offload 模式加载 FP8 模型...")

        # 清理显存
        torch.cuda.empty_cache()
        gc.collect()

        # 1. 加载 Text Encoder 1
        logger.info("加载 Text Encoder 1 (CLIP)...")
        text_encoder = CLIPTextModel.from_pretrained(
            model_path, subfolder="text_encoder", torch_dtype=dtype
        )
        logger.info("✓ Text Encoder 1 加载完成")

        # 2. 加载 Text Encoder 2
        logger.info("加载 Text Encoder 2 (T5)...")
        text_encoder_2 = T5EncoderModel.from_pretrained(
            model_path, subfolder="text_encoder_2", torch_dtype=dtype
        )
        logger.info("✓ Text Encoder 2 加载完成")

        # 3. 加载 Tokenizers
        logger.info("加载 Tokenizers...")
        tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")
        tokenizer_2 = T5TokenizerFast.from_pretrained(model_path, subfolder="tokenizer_2")
        logger.info("✓ Tokenizers 加载完成")

        # 4. 加载 VAE
        logger.info("加载 VAE...")
        vae_path = os.path.join(model_path, "flux-vae-bf16.safetensors")
        vae_config_path = os.path.join(model_path, "vae", "config.json")
        vae = AutoencoderKL.from_single_file(vae_path, config=vae_config_path, torch_dtype=dtype)
        logger.info("✓ VAE 加载完成")

        # 5. 加载 Scheduler
        logger.info("加载 Scheduler...")
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(model_path, subfolder="scheduler")
        logger.info("✓ Scheduler 加载完成")

        # 6. 加载 Transformer
        logger.info("加载 Transformer (FP8)...")
        transformer_path = os.path.join(model_path, "flux1-dev-fp8.safetensors")
        config_path = os.path.join(model_path, "transformer_config.json")
        transformer = FluxTransformer2DModel.from_single_file(
            transformer_path, config=config_path, torch_dtype=dtype
        )
        logger.info("✓ Transformer 加载完成")

        # 7. 组装 Pipeline
        logger.info("组装 Pipeline...")
        pipe = FluxPipeline(
            scheduler=scheduler,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            vae=vae,
            transformer=transformer
        )

        # 8. 启用 CPU offload
        logger.info("启用 model CPU offload...")
        pipe.enable_model_cpu_offload()
        logger.info("✓ CPU offload 已启用")

        torch.cuda.empty_cache()
        gc.collect()

        return pipe

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = 28,
        guidance_scale: float = 3.5,
        width: int = 1024,
        height: int = 1024,
        seed: Optional[int] = None,
        **kwargs
    ) -> tuple[Image.Image, Dict[str, Any]]:
        """
        生成图像

        Args:
            prompt: 正向提示词
            negative_prompt: 负向提示词（Flux 可能不支持）
            num_inference_steps: 推理步数（推荐 28）
            guidance_scale: CFG scale（推荐 3.5）
            width: 图像宽度
            height: 图像高度
            seed: 随机种子
            **kwargs: 其他参数

        Returns:
            (image, generation_info) 元组
        """
        # 设置随机种子
        # 注意：Flux Pipeline 内部某些操作需要 CPU Generator，不能用 CUDA
        if seed is not None:
            generator = torch.Generator(device="cpu").manual_seed(seed)
        else:
            generator = None

        logger.info(f"Flux 生成中 (steps={num_inference_steps}, cfg={guidance_scale}, seed={seed})")
        logger.info(f"Prompt: {prompt[:100]}...")

        # 清理显存
        torch.cuda.empty_cache()

        # 生成图像
        result = self.pipe(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            generator=generator
        )

        image = result.images[0]

        # 生成后清理显存
        torch.cuda.empty_cache()

        # 生成信息
        if self.quantization_mode == "official" or (self.quantization_mode == "auto"):
            model_name = f"flux-1-dev-{self.official_quant_type}"
        elif "fp8" in self.model_path.lower():
            model_name = "flux-1-dev-fp8-local"
        else:
            model_name = "flux-1-dev"
        gen_info = {
            "model": model_name,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "width": width,
            "height": height,
            "seed": seed
        }

        logger.info("Flux 生成完成")

        return image, gen_info

    def cleanup(self):
        """清理资源"""
        if hasattr(self, 'pipe'):
            del self.pipe
            torch.cuda.empty_cache()
            logger.info("Flux 生成器资源已清理")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Flux.1-dev 图像生成器测试")
    parser.add_argument("--mode", type=str, default="auto",
                        choices=["auto", "official", "local_fp8"],
                        help="量化模式")
    parser.add_argument("--quant-type", type=str, default="int8",
                        choices=["fp8", "int8", "bnb-4bit", "bnb-8bit"],
                        help="官方预量化类型")
    parser.add_argument("--prompt", type=str,
                        default="a beautiful sunset over the ocean, masterpiece",
                        help="测试提示词")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--output", type=str, default="/tmp/flux_test.png",
                        help="输出路径")
    args = parser.parse_args()

    print(f"使用模式: {args.mode}, 量化类型: {args.quant_type}")

    generator = FluxGenerator(
        quantization_mode=args.mode,
        official_quant_type=args.quant_type
    )
    image, info = generator.generate(
        prompt=args.prompt,
        seed=args.seed
    )

    image.save(args.output)
    print(f"测试图像已保存到: {args.output}")
    print(f"生成信息: {info}")
