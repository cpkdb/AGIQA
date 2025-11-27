#!/usr/bin/env python3
"""
LLM质量退化 + SDXL图像生成联动Demo
读取LLM生成的正负prompt对，使用SDXL生成对应的正负图像对

使用方法：
    python demo_llm_sdxl_paired.py \
        --llm_results /path/to/llm_degradation_results.json \
        --output_dir /root/autodl-tmp/llm_sdxl_demo \
        --num_steps 20 \
        --cfg_scale 5.0
"""

import os
import sys
import json
import argparse
import torch
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import time

# 添加脚本目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from sdxl_generator import SDXLGenerator


def load_llm_results(json_path: str) -> dict:
    """加载LLM生成的正负prompt对结果"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def filter_successful_pairs(llm_data: dict) -> list:
    """过滤出成功生成的正负prompt对"""
    successful_pairs = []
    for result in llm_data.get('results', []):
        if result.get('negative_prompt'):  # 只保留有负样本的
            successful_pairs.append(result)
    return successful_pairs


def save_image_pair(
    positive_image,
    negative_image,
    pair_id: int,
    positive_seed: int,
    negative_seed: int,
    severity: str,
    output_dir: str
) -> dict:
    """保存正负图像对并返回路径信息"""
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    # 保存正样本图像
    positive_filename = f"positive_{pair_id:03d}_{positive_seed}.png"
    positive_path = os.path.join(images_dir, positive_filename)
    positive_image.save(positive_path)

    # 保存负样本图像
    negative_filename = f"negative_{pair_id:03d}_{severity}_{negative_seed}.png"
    negative_path = os.path.join(images_dir, negative_filename)
    negative_image.save(negative_path)

    return {
        "positive_path": os.path.join("images", positive_filename),
        "negative_path": os.path.join("images", negative_filename)
    }


def main():
    parser = argparse.ArgumentParser(description="LLM+SDXL联动Demo")
    parser.add_argument(
        "--llm_results",
        type=str,
        default="/root/ImageReward/data_generation/demo_output/llm_degradation_results_20251124_184621.json",
        help="LLM生成结果JSON路径"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/root/autodl-tmp/llm_sdxl_demo",
        help="输出目录"
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=20,
        help="SDXL推理步数（20=快速，50=标准，80=高质量）"
    )
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=5.0,
        help="CFG scale（5.0=快速，7.5=标准，9.0=高质量）"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="基础随机种子"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/root/ckpts/sd_xl_base_1.0.safetensors",
        help="SDXL模型路径"
    )

    args = parser.parse_args()

    # 创建带时间戳的输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{args.output_dir}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("LLM质量退化 + SDXL图像生成联动Demo")
    print("=" * 80)
    print(f"LLM结果文件: {args.llm_results}")
    print(f"输出目录: {output_dir}")
    print(f"生成参数: {args.num_steps}步, CFG={args.cfg_scale}")
    print("=" * 80)

    # 1. 加载LLM结果
    print("\n[1/3] 加载LLM生成的正负prompt对...")
    llm_data = load_llm_results(args.llm_results)
    successful_pairs = filter_successful_pairs(llm_data)
    print(f"成功加载 {len(successful_pairs)} 个正负prompt对")

    # 2. 初始化SDXL生成器
    print("\n[2/3] 初始化SDXL生成器...")
    print("正在加载SDXL模型（可能需要几分钟）...")
    generator = SDXLGenerator(
        model_path=args.model_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype=torch.float16
    )
    print("SDXL生成器初始化完成")

    # 3. 生成图像对
    print(f"\n[3/3] 开始生成 {len(successful_pairs)} 对正负图像...")
    print("=" * 80)

    dataset_pairs = []
    success_count = 0
    fail_count = 0
    total_time = 0

    for i, pair_data in enumerate(tqdm(successful_pairs, desc="生成进度")):
        pair_id = pair_data['id']
        positive_prompt = pair_data['positive_prompt']
        negative_prompt = pair_data['negative_prompt']
        degradation_info = pair_data.get('degradation_info', {})
        severity = degradation_info.get('severity', 'unknown')

        try:
            # 生成正样本图像
            positive_seed = args.seed + pair_id
            start_time = time.time()

            positive_image, positive_info = generator.generate(
                prompt=positive_prompt,
                num_inference_steps=args.num_steps,
                guidance_scale=args.cfg_scale,
                seed=positive_seed
            )

            positive_time = time.time() - start_time

            # 生成负样本图像
            negative_seed = args.seed + pair_id + 10000
            start_time = time.time()

            negative_image, negative_info = generator.generate(
                prompt=negative_prompt,
                num_inference_steps=args.num_steps,
                guidance_scale=args.cfg_scale,
                seed=negative_seed
            )

            negative_time = time.time() - start_time
            total_time += (positive_time + negative_time)

            # 保存图像
            image_paths = save_image_pair(
                positive_image,
                negative_image,
                pair_id,
                positive_seed,
                negative_seed,
                severity,
                output_dir
            )

            # 记录到数据集
            dataset_pairs.append({
                "id": pair_id,
                "positive": {
                    "prompt": positive_prompt,
                    "image_path": image_paths['positive_path'],
                    "seed": positive_seed,
                    "generation_time": round(positive_time, 2)
                },
                "negative": {
                    "prompt": negative_prompt,
                    "image_path": image_paths['negative_path'],
                    "seed": negative_seed,
                    "generation_time": round(negative_time, 2),
                    "degradation_info": degradation_info
                }
            })

            success_count += 1

        except Exception as e:
            print(f"\n生成第 {pair_id} 对图像失败: {e}")
            fail_count += 1
            continue

    # 4. 保存数据集JSON
    print("\n" + "=" * 80)
    print("保存数据集...")

    dataset_json = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "total_pairs": len(successful_pairs),
            "success_count": success_count,
            "fail_count": fail_count,
            "model": "stabilityai/stable-diffusion-xl-base-1.0",
            "generation_params": {
                "num_inference_steps": args.num_steps,
                "guidance_scale": args.cfg_scale,
                "resolution": "1024x1024"
            },
            "llm_source": args.llm_results,
            "degradation_config": llm_data.get('metadata', {}),
            "total_generation_time": round(total_time, 2),
            "avg_time_per_image": round(total_time / (success_count * 2) if success_count > 0 else 0, 2)
        },
        "pairs": dataset_pairs
    }

    dataset_file = os.path.join(output_dir, "dataset.json")
    with open(dataset_file, 'w', encoding='utf-8') as f:
        json.dump(dataset_json, f, ensure_ascii=False, indent=2)

    print(f"数据集已保存: {dataset_file}")

    # 5. 保存统计摘要
    summary_file = os.path.join(output_dir, "summary.txt")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("LLM质量退化 + SDXL图像生成Demo - 统计摘要\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"输出目录: {output_dir}\n\n")
        f.write(f"总prompt对数: {len(successful_pairs)}\n")
        f.write(f"成功生成: {success_count} 对 ({success_count*2} 张图像)\n")
        f.write(f"失败: {fail_count} 对\n")
        f.write(f"成功率: {success_count/len(successful_pairs)*100:.1f}%\n\n")
        f.write(f"SDXL参数:\n")
        f.write(f"  - 推理步数: {args.num_steps}\n")
        f.write(f"  - CFG Scale: {args.cfg_scale}\n")
        f.write(f"  - 分辨率: 1024x1024\n\n")
        f.write(f"生成耗时统计:\n")
        f.write(f"  - 总耗时: {total_time:.1f} 秒 ({total_time/60:.1f} 分钟)\n")
        f.write(f"  - 平均每张图像: {total_time/(success_count*2) if success_count > 0 else 0:.1f} 秒\n\n")

        # 退化程度分布
        severity_counts = {}
        for pair in dataset_pairs:
            severity = pair['negative']['degradation_info'].get('severity', 'unknown')
            severity_counts[severity] = severity_counts.get(severity, 0) + 1

        f.write(f"退化程度分布:\n")
        for severity, count in sorted(severity_counts.items()):
            f.write(f"  - {severity}: {count}\n")

    print(f"统计摘要已保存: {summary_file}")

    # 6. 打印最终统计
    print("\n" + "=" * 80)
    print("统计信息")
    print("=" * 80)
    print(f"  总数: {len(successful_pairs)} 对")
    print(f"  成功: {success_count} 对 ({success_count*2} 张图像)")
    print(f"  失败: {fail_count} 对")
    print(f"  成功率: {success_count/len(successful_pairs)*100:.1f}%")
    print(f"  总耗时: {total_time:.1f} 秒 ({total_time/60:.1f} 分钟)")
    print(f"  平均每张: {total_time/(success_count*2) if success_count > 0 else 0:.1f} 秒")

    print("\n" + "=" * 80)
    print("Demo完成!")
    print("=" * 80)
    print(f"\n输出目录: {output_dir}")
    print(f"  - dataset.json: 完整数据集元数据")
    print(f"  - summary.txt: 统计摘要")
    print(f"  - images/: {success_count*2} 张生成的图像")

    # 显示几个示例
    if dataset_pairs:
        print("\n示例对预览（前3个）:")
        print("-" * 80)
        for pair in dataset_pairs[:3]:
            print(f"ID {pair['id']}:")
            print(f"  正: {pair['positive']['prompt'][:60]}...")
            print(f"  负: {pair['negative']['prompt'][:60]}...")
            print(f"  程度: {pair['negative']['degradation_info'].get('severity')}")
            print(f"  图像: {pair['positive']['image_path']} | {pair['negative']['image_path']}")
            print()


if __name__ == "__main__":
    main()
