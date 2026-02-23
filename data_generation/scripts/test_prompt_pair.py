#!/usr/bin/env python
"""
快速测试脚本：直接提供正负 prompt 生成图像对比
用于验证退化效果是否符合预期
"""

import argparse
from pathlib import Path
from datetime import datetime
from sdxl_generator import SDXLGenerator


def main():
    parser = argparse.ArgumentParser(description="测试正负 prompt 对生成效果")
    
    parser.add_argument(
        "--positive", "-p",
        type=str,
        required=True,
        help="正样本 prompt"
    )
    parser.add_argument(
        "--negative", "-n",
        type=str,
        required=True,
        help="负样本 prompt（退化后的）"
    )
    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        default="/root/autodl-tmp/test_pairs",
        help="输出目录"
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=42,
        help="随机种子（确保正负样本可比较）"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=40,
        help="推理步数"
    )
    parser.add_argument(
        "--cfg",
        type=float,
        default=7.5,
        help="CFG scale"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/root/ckpts/sd_xl_base_1.0.safetensors",
        help="SDXL 模型路径"
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="输出文件名前缀（可选）"
    )
    parser.add_argument(
        "--no_neg_prompt",
        action="store_true",
        help="负样本生成时不使用 negative prompt（用于 visual_quality 退化）"
    )
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成文件名前缀（始终包含时间戳）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.name:
        prefix = f"{args.name}_{timestamp}"
    else:
        prefix = f"test_{timestamp}"
    
    print("=" * 60)
    print("Prompt 对比测试")
    print("=" * 60)
    print(f"正样本: {args.positive}")
    print(f"负样本: {args.negative}")
    print(f"Seed: {args.seed}")
    print(f"输出目录: {output_dir}")
    print("=" * 60)
    
    # 初始化生成器
    print("\n初始化 SDXL 生成器...")
    generator = SDXLGenerator(model_path=args.model_path)
    
    # 生成正样本
    print("\n[1/2] 生成正样本图像...")
    positive_image, _ = generator.generate(
        prompt=args.positive,
        negative_prompt="low quality, worst quality",
        num_inference_steps=args.steps,
        guidance_scale=args.cfg,
        seed=args.seed
    )
    positive_path = output_dir / f"{prefix}_positive.png"
    positive_image.save(positive_path)
    print(f"  保存: {positive_path}")
    
    # 生成负样本
    print("\n[2/2] 生成负样本图像...")
    neg_prompt_for_gen = "" if args.no_neg_prompt else "low quality, worst quality"
    negative_image, _ = generator.generate(
        prompt=args.negative,
        negative_prompt=neg_prompt_for_gen,
        num_inference_steps=args.steps,
        guidance_scale=args.cfg,
        seed=args.seed
    )
    negative_path = output_dir / f"{prefix}_negative.png"
    negative_image.save(negative_path)
    print(f"  保存: {negative_path}")
    
    # 清理
    generator.cleanup()
    
    print("\n" + "=" * 60)
    print("生成完成！")
    print(f"正样本: {positive_path}")
    print(f"负样本: {negative_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
