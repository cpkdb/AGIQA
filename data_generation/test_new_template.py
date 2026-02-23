"""
测试新的动作导向模板格式
针对 blur, exposure_issues, low_contrast 三个属性
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scripts.llm_prompt_degradation import LLMPromptDegradation

# 配置路径
LLM_CONFIG_PATH = "/root/ImageReward/data_generation/config/llm_config.yaml"
QUALITY_DIMENSIONS_PATH = "/root/ImageReward/data_generation/config/quality_dimensions.json"

# 测试prompts
TEST_PROMPTS = [
    "a beautiful sunset over the ocean with golden light",
    "portrait of a young woman in a garden, natural lighting",
    "city skyline at night with bright lights",
    "a cat sitting on a wooden chair in a cozy room",
    "mountain landscape with snow-capped peaks and clear sky",
]

# 测试的属性和severity
TEST_CASES = [
    {"subcategory": "low_visual_quality", "attribute": "blur", "severity": "mild"},
    {"subcategory": "low_visual_quality", "attribute": "blur", "severity": "moderate"},
    {"subcategory": "low_visual_quality", "attribute": "blur", "severity": "severe"},
    {"subcategory": "low_visual_quality", "attribute": "exposure_issues", "severity": "mild"},
    {"subcategory": "low_visual_quality", "attribute": "exposure_issues", "severity": "moderate"},
    {"subcategory": "low_visual_quality", "attribute": "exposure_issues", "severity": "severe"},
    {"subcategory": "low_visual_quality", "attribute": "low_contrast", "severity": "mild"},
    {"subcategory": "low_visual_quality", "attribute": "low_contrast", "severity": "moderate"},
    {"subcategory": "low_visual_quality", "attribute": "low_contrast", "severity": "severe"},
]


def main():
    # API key 从配置文件读取，无需环境变量
    pass

    print("=" * 80)
    print("新模板格式测试 - 动作导向版本")
    print("测试属性: blur, exposure_issues, low_contrast")
    print("=" * 80)

    # 初始化生成器
    print("\n正在初始化LLM退化生成器...")
    generator = LLMPromptDegradation(
        llm_config_path=LLM_CONFIG_PATH,
        quality_dimensions_path=QUALITY_DIMENSIONS_PATH
    )
    print("初始化完成!\n")

    # 存储结果
    results = []

    # 测试每个case
    for test_case in TEST_CASES:
        subcategory = test_case["subcategory"]
        attribute = test_case["attribute"]
        severity = test_case["severity"]

        print("-" * 80)
        print(f"测试: {attribute} ({severity})")
        print("-" * 80)

        # 选择一个测试prompt
        test_prompt = TEST_PROMPTS[TEST_CASES.index(test_case) % len(TEST_PROMPTS)]

        print(f"原始Prompt: {test_prompt}")

        try:
            negative_prompt, degradation_info = generator.generate_negative_prompt(
                positive_prompt=test_prompt,
                subcategory=subcategory,
                attribute=attribute,
                severity=severity
            )

            print(f"退化Prompt: {negative_prompt}")
            print(f"退化信息: {degradation_info}")

            results.append({
                "attribute": attribute,
                "severity": severity,
                "original": test_prompt,
                "degraded": negative_prompt,
                "success": True
            })

        except Exception as e:
            print(f"生成失败: {e}")
            results.append({
                "attribute": attribute,
                "severity": severity,
                "original": test_prompt,
                "degraded": None,
                "success": False,
                "error": str(e)
            })

        print()

    # 汇总结果
    print("=" * 80)
    print("测试结果汇总")
    print("=" * 80)

    success_count = sum(1 for r in results if r["success"])
    total_count = len(results)

    print(f"\n成功率: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)")

    print("\n详细结果:")
    for r in results:
        status = "✅" if r["success"] else "❌"
        print(f"{status} {r['attribute']} ({r['severity']})")
        if r["success"]:
            print(f"   原始: {r['original'][:50]}...")
            print(f"   退化: {r['degraded'][:50]}...")


if __name__ == "__main__":
    main()
