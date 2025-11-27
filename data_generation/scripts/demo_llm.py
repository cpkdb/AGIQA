#!/usr/bin/env python3
"""
LLM退化Prompt生成快速演示脚本
运行前请确保已设置: export OPENAI_API_KEY="your-api-key"
"""

import os
import sys
from pathlib import Path

# 添加脚本目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from llm_prompt_degradation import LLMPromptDegradation


def main():
    # 检查API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ 错误: 未设置OPENAI_API_KEY环境变量")
        print("\n请先设置API密钥:")
        print("  export OPENAI_API_KEY='your-api-key-here'")
        print("\n或者添加到 ~/.bashrc:")
        print("  echo 'export OPENAI_API_KEY=\"your-api-key\"' >> ~/.bashrc")
        print("  source ~/.bashrc")
        sys.exit(1)

    # 配置路径
    llm_config_path = "/root/ImageReward/data_generation/config/llm_config.yaml"
    quality_dimensions_path = "/root/ImageReward/data_generation/config/quality_dimensions.json"

    print("=" * 80)
    print("LLM退化Prompt生成演示")
    print("=" * 80)

    # 初始化生成器
    print("\n📦 正在初始化LLM生成器...")
    generator = LLMPromptDegradation(
        llm_config_path=llm_config_path,
        quality_dimensions_path=quality_dimensions_path
    )
    print("✅ 初始化完成\n")

    # 演示用例
    demos = [
        {
            "name": "视觉质量退化 (低技术质量)",
            "prompt": "a beautiful sunset over the ocean, masterpiece, high quality, detailed",
            "subcategory": "low_visual_quality",
            "severity": "moderate"
        },
        {
            "name": "对齐度退化 (颜色替换)",
            "prompt": "a red apple on a wooden table, professional photography",
            "subcategory": "attribute_alignment",
            "severity": "moderate"
        },
        {
            "name": "语义合理性退化 (人体结构)",
            "prompt": "a person waving hello with both hands, realistic, detailed",
            "subcategory": "semantic_plausibility",
            "severity": "severe"
        }
    ]

    for i, demo in enumerate(demos, 1):
        print(f"【演示 {i}/{len(demos)}】{demo['name']}")
        print("-" * 80)
        print(f"💬 正样本prompt: {demo['prompt']}")
        print(f"🎯 退化维度: {demo['subcategory']}")
        print(f"📊 退化程度: {demo['severity']}")
        print()

        try:
            print("🤖 调用LLM生成负样本...")
            negative_prompt, degradation_info = generator.generate_negative_prompt(
                demo['prompt'],
                demo['subcategory'],
                demo['severity']
            )

            print(f"✅ 生成成功!")
            print(f"💬 负样本prompt: {negative_prompt}")
            print(f"📋 退化信息: {degradation_info}")
            print()

        except Exception as e:
            print(f"❌ 生成失败: {e}")
            print()

        print("=" * 80)
        print()

    print("✨ 演示完成！")
    print("\n💡 提示:")
    print("  - 查看完整使用指南: /root/ImageReward/data_generation/LLM_DEGRADATION_GUIDE.md")
    print("  - 运行完整测试: python test_llm_degradation.py --test all")
    print("  - 在数据集生成中使用: python generate_dataset.py --use_llm ...")


if __name__ == "__main__":
    main()
