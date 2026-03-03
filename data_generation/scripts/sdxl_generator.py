"""
SDXL图像生成器
使用diffusers库调用SDXL模型生成图像
支持长prompt（突破CLIP 77 token限制）
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
    from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler
except ImportError as e:
    print("错误: 未找到 diffusers 模块。", file=sys.stderr)
    print("请安装: pip install diffusers", file=sys.stderr)
    print(f"详细错误: {e}", file=sys.stderr)
    sys.exit(1)

# 尝试导入compel用于长prompt支持
try:
    from compel import Compel, ReturnedEmbeddingsType
    COMPEL_AVAILABLE = True
except ImportError:
    COMPEL_AVAILABLE = False

from PIL import Image
import os
from typing import Optional, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SDXLGenerator:
    """SDXL图像生成器类，支持长prompt"""

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
        self.compel = None

        logger.info(f"正在加载SDXL模型: {model_path}")

        # 检查模型路径是否存在
        if not os.path.exists(model_path):
            logger.warning(f"本地模型不存在: {model_path}")
            logger.info("将从HuggingFace下载模型: stabilityai/stable-diffusion-xl-base-1.0")
            model_path = "stabilityai/stable-diffusion-xl-base-1.0"

        # 加载pipeline
        try:
            if os.path.isfile(model_path):
                self.pipe = StableDiffusionXLPipeline.from_single_file(
                    model_path,
                    torch_dtype=dtype,
                    use_safetensors=use_safetensors
                )
            else:
                self.pipe = StableDiffusionXLPipeline.from_pretrained(
                    model_path,
                    torch_dtype=dtype,
                    use_safetensors=use_safetensors
                )

            self.pipe = self.pipe.to(device)

            # 启用内存优化（注意：不使用 cpu_offload，因为它与 Compel 不兼容）
            # self.pipe.enable_model_cpu_offload()  # 禁用：会导致 Compel 设备不匹配
            if hasattr(self.pipe, 'enable_xformers_memory_efficient_attention'):
                try:
                    self.pipe.enable_xformers_memory_efficient_attention()
                    logger.info("已启用xformers内存优化")
                except Exception as e:
                    logger.warning(f"无法启用xformers: {e}")

            # 初始化Compel用于长prompt支持
            if COMPEL_AVAILABLE:
                try:
                    self.compel = Compel(
                        tokenizer=[self.pipe.tokenizer, self.pipe.tokenizer_2],
                        text_encoder=[self.pipe.text_encoder, self.pipe.text_encoder_2],
                        returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                        requires_pooled=[False, True],
                        truncate_long_prompts=False  # 关键：不截断长prompt
                    )
                    logger.info("已启用Compel长prompt支持")
                except Exception as e:
                    logger.warning(f"无法初始化Compel: {e}，将使用默认模式（可能截断长prompt）")
                    self.compel = None
            else:
                logger.warning("Compel未安装，长prompt可能被截断。建议运行: pip install compel")

            logger.info("SDXL模型加载成功")

        except Exception as e:
            logger.error(f"加载SDXL模型失败: {e}")
            raise

    def _encode_prompt(self, prompt: str, negative_prompt: str = ""):
        """
        使用Compel编码prompt，支持长文本
        自动padding确保prompt和negative_prompt编码后shape一致
        
        Returns:
            (prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds)
        """
        if self.compel is None:
            return None, None, None, None
        
        try:
            # 确保 text_encoder 在正确的设备上（解决 cpu_offload 导致的设备不匹配）
            # cpu_offload 会将模型移到 CPU，但 Compel 需要它们在 GPU 上
            self.pipe.text_encoder.to(self.device)
            self.pipe.text_encoder_2.to(self.device)
            
            # 编码正向prompt
            conditioning, pooled = self.compel(prompt)
            
            # 编码负向prompt
            if negative_prompt:
                negative_conditioning, negative_pooled = self.compel(negative_prompt)
            else:
                negative_conditioning, negative_pooled = self.compel("")
            
            # 确保两个编码的序列长度一致（padding到相同长度）
            pos_len = conditioning.shape[1]
            neg_len = negative_conditioning.shape[1]
            
            if pos_len != neg_len:
                max_len = max(pos_len, neg_len)
                
                if pos_len < max_len:
                    # Pad positive embedding with zeros
                    padding = torch.zeros(
                        (conditioning.shape[0], max_len - pos_len, conditioning.shape[2]),
                        dtype=conditioning.dtype,
                        device=conditioning.device
                    )
                    conditioning = torch.cat([conditioning, padding], dim=1)
                
                if neg_len < max_len:
                    # Pad negative embedding with zeros
                    padding = torch.zeros(
                        (negative_conditioning.shape[0], max_len - neg_len, negative_conditioning.shape[2]),
                        dtype=negative_conditioning.dtype,
                        device=negative_conditioning.device
                    )
                    negative_conditioning = torch.cat([negative_conditioning, padding], dim=1)
                
                logger.info(f"Embedding长度对齐: prompt {pos_len} tokens, negative {neg_len} tokens -> {max_len} tokens")
            
            return conditioning, negative_conditioning, pooled, negative_pooled
            
        except Exception as e:
            logger.warning(f"Compel编码失败: {e}，回退到默认模式")
            return None, None, None, None

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
        生成图像（支持长prompt）

        Args:
            prompt: 正向提示词（无长度限制）
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

        logger.info(f"正在生成图像 - Prompt长度: {len(prompt)} 字符")

        try:
            # 尝试使用Compel处理长prompt
            prompt_embeds, negative_embeds, pooled_embeds, negative_pooled = self._encode_prompt(
                prompt, negative_prompt
            )
            
            if prompt_embeds is not None:
                # 使用预计算的embeddings（支持长prompt）
                output = self.pipe(
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=negative_embeds,
                    pooled_prompt_embeds=pooled_embeds,
                    negative_pooled_prompt_embeds=negative_pooled,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    width=width,
                    height=height,
                    generator=generator,
                    **kwargs
                )
            else:
                # 回退到默认模式（可能截断）
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
                "negative_prompt": negative_prompt,
                "long_prompt_support": prompt_embeds is not None
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

    # 测试长prompt
    long_prompt = """a beautiful sunset over the ocean, masterpiece, high quality, detailed, 
    golden hour lighting, dramatic clouds, reflections on water, photorealistic, 
    8k resolution, cinematic composition, vibrant colors, ethereal atmosphere,
    professional photography, award winning, national geographic style,
    peaceful scene, natural beauty, breathtaking view, serene mood"""
    
    image, info = generator.generate(
        prompt=long_prompt,
        negative_prompt="low quality, blurry, bad anatomy, watermark, text",
        seed=42
    )

    # 保存测试图像
    os.makedirs("/root/ImageReward/data_generation/test_output", exist_ok=True)
    image.save("/root/ImageReward/data_generation/test_output/test_sdxl.png")
    print(f"测试图像已保存，生成信息: {info}")
    print(f"长prompt支持: {info.get('long_prompt_support', False)}")

    generator.cleanup()
