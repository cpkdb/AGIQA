"""
Image Generator Tool
封装 SDXL 和未来其他生成模型，使用 smolagents @tool 装饰器
支持多模型注册机制
"""

import json
import os
import sys
import uuid
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Type

sys.path.insert(0, str(Path(__file__).parent.parent))

from smolagents import tool

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageGeneratorRegistry:
    """
    图像生成器注册表，支持多模型
    使用注册模式，方便未来添加新的生成模型
    """

    _generators: Dict[str, Type] = {}
    _instances: Dict[str, Any] = {}

    @classmethod
    def register(cls, model_id: str, generator_class: Type):
        """注册生成器类"""
        cls._generators[model_id] = generator_class
        logger.info(f"Registered generator: {model_id}")

    @classmethod
    def get(cls, model_id: str):
        """获取生成器实例（单例）"""
        if model_id not in cls._generators:
            available = list(cls._generators.keys())
            raise ValueError(f"Unknown model: {model_id}. Available: {available}")

        if model_id not in cls._instances:
            logger.info(f"Initializing generator: {model_id}")
            cls._instances[model_id] = cls._generators[model_id]()

        return cls._instances[model_id]

    @classmethod
    def available_models(cls) -> list:
        """获取可用模型列表"""
        return list(cls._generators.keys())

    @classmethod
    def cleanup(cls, model_id: str = None):
        """清理生成器实例"""
        if model_id:
            if model_id in cls._instances:
                instance = cls._instances.pop(model_id)
                if hasattr(instance, 'cleanup'):
                    instance.cleanup()
        else:
            for mid, instance in cls._instances.items():
                if hasattr(instance, 'cleanup'):
                    instance.cleanup()
            cls._instances.clear()


# SDXL 生成器配置
_sdxl_config = {}

def _register_sdxl(model_path: str = None):
    """延迟注册 SDXL，避免启动时加载模型"""
    global _sdxl_config
    if model_path:
        _sdxl_config['model_path'] = model_path
    try:
        from sdxl_generator import SDXLGenerator
        ImageGeneratorRegistry.register("sdxl", SDXLGenerator)
    except ImportError as e:
        logger.warning(f"Failed to register SDXL: {e}")


# Flux 生成器配置
_flux_config = {}

def _register_flux(model_path: str = None):
    """延迟注册 Flux，避免启动时加载模型"""
    global _flux_config
    if model_path:
        _flux_config['model_path'] = model_path
    try:
        from flux_generator import FluxGenerator
        ImageGeneratorRegistry.register("flux", FluxGenerator)
    except ImportError as e:
        logger.warning(f"Failed to register Flux: {e}")


# Flux-Schnell 生成器配置
_flux_schnell_config = {}

def _register_flux_schnell(model_path: str = None, optimize: bool = False):
    """延迟注册 Flux-Schnell，避免启动时加载模型"""
    global _flux_schnell_config
    if model_path:
        _flux_schnell_config['model_path'] = model_path
    _flux_schnell_config['optimize'] = optimize
    try:
        from flux_schnell_generator import FluxSchnellGenerator
        # 注册带优化配置的生成器
        ImageGeneratorRegistry.register("flux-schnell", lambda: FluxSchnellGenerator(
            optimize_for_speed=_flux_schnell_config.get('optimize', False),
            enable_compile=_flux_schnell_config.get('optimize', False)
        ))
    except ImportError as e:
        logger.warning(f"Failed to register Flux-Schnell: {e}")


# 默认输出目录
DEFAULT_OUTPUT_DIR = Path("/tmp/generated_images")


@tool
def image_generator(
    prompt: str,
    seed: int,
    model_id: str = "sdxl",
    negative_prompt: str = "low quality, worst quality",
    output_dir: str = None,
    output_path: str = None,
    model_path: str = None,
    steps: int = 35,
    cfg: float = 7.5,
    width: int = 1024,
    height: int = 1024,
    optimize: bool = False
) -> str:
    """
    Generate an image using the specified model.

    This tool generates images using text-to-image models. Currently supports
    SDXL and Flux.1-dev.

    Args:
        prompt: The text prompt describing the image to generate.
        seed: Random seed for reproducibility. Use the same seed for positive
              and negative images to ensure content consistency between pairs.
        model_id: The generation model to use. Currently supported:
                  - "sdxl": Stable Diffusion XL (default)
                  - "flux": Flux.1-dev
                  - "flux-schnell": Flux.1-schnell (4-step fast generation)
        negative_prompt: Negative prompt to avoid certain qualities in the image.
        output_dir: Directory to save generated images (used if output_path not specified).
        output_path: Exact path to save the image (overrides output_dir if provided).
        model_path: Path to model weights.
                    - SDXL: /root/ckpts/sd_xl_base_1.0.safetensors
                    - Flux: /root/autodl-tmp/flux-1-dev
        steps: Number of inference steps (default: 35 for SDXL, 28 for Flux, 4 for Flux-Schnell).
        cfg: CFG scale (default: 7.5 for SDXL, 3.5 for Flux, 0.0 for Flux-Schnell).
        width: Image width (default: 1024).
        height: Image height (default: 1024).
        optimize: Enable speed optimization for Flux-Schnell (T5 4-bit + FP8 + compile).

    Returns:
        Path to the generated image file as a string.
    """
    # 确保模型已注册
    if model_id == "sdxl" and "sdxl" not in ImageGeneratorRegistry._generators:
        _register_sdxl(model_path)
    elif model_id == "flux" and "flux" not in ImageGeneratorRegistry._generators:
        _register_flux(model_path)
    elif model_id == "flux-schnell" and "flux-schnell" not in ImageGeneratorRegistry._generators:
        _register_flux_schnell(model_path, optimize=optimize)

    # 获取生成器
    generator = ImageGeneratorRegistry.get(model_id)

    # 确定输出路径
    if output_path:
        # 使用指定的精确路径
        final_path = Path(output_path)
        final_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        # 使用 output_dir + 自动生成文件名
        out_dir = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        unique_id = uuid.uuid4().hex[:8]
        filename = f"img_{seed}_{unique_id}.png"
        final_path = out_dir / filename

    # 生成图像（传递所有参数）
    # 根据 model_id 选择正确的配置
    if model_id == "sdxl":
        final_model_path = model_path or _sdxl_config.get('model_path')
    elif model_id == "flux":
        final_model_path = model_path or _flux_config.get('model_path')
    elif model_id == "flux-schnell":
        final_model_path = model_path or _flux_schnell_config.get('model_path')
    else:
        final_model_path = model_path

    image, gen_info = generator.generate(
        prompt=prompt,
        negative_prompt=negative_prompt,
        seed=seed,
        num_inference_steps=steps,
        guidance_scale=cfg,
        width=width,
        height=height,
        model_path=final_model_path
    )

    # 保存图像
    image.save(str(final_path))
    logger.info(f"Image saved to: {final_path}")

    return str(final_path)


def cleanup():
    """清理所有生成器资源"""
    ImageGeneratorRegistry.cleanup()


if __name__ == "__main__":
    # 测试
    _register_sdxl()
    result = image_generator(
        prompt="a beautiful sunset over the ocean, masterpiece",
        seed=42,
        model_id="sdxl"
    )
    print("Generated image:", result)
