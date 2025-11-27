"""
测试LLM退化prompt生成功能
对比LLM方法与关键词方法的生成结果
"""

import os
import sys
import json
import logging
from pathlib import Path

# 添加脚本目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from llm_prompt_degradation import LLMPromptDegradation
from prompt_degradation import PromptDegradation

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_llm_degradation():
    """测试LLM退化生成器"""

    # 配置路径
    llm_config_path = "/root/ImageReward/data_generation/config/llm_config.yaml"
    quality_dimensions_path = "/root/ImageReward/data_generation/config/quality_dimensions.json"

    # 检查API key
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("未设置OPENAI_API_KEY环境变量")
        logger.info("请设置环境变量: export OPENAI_API_KEY='your-api-key'")
        return False

    print("=" * 100)
    print("LLM退化Prompt生成功能测试")
    print("=" * 100)

    try:
        # 初始化生成器
        logger.info("初始化LLM退化生成器...")
        llm_generator = LLMPromptDegradation(
            llm_config_path=llm_config_path,
            quality_dimensions_path=quality_dimensions_path
        )

        # 初始化关键词生成器（用于对比）
        logger.info("初始化关键词退化生成器...")
        keyword_generator = PromptDegradation(quality_dimensions_path)

        print("\n" + "=" * 100)

        # 测试用例
        test_cases = [
            {
                "name": "视觉质量-技术质量",
                "prompt": "a beautiful sunset over the ocean, masterpiece, high quality, detailed",
                "subcategory": "low_visual_quality",
                "severity": "moderate"
            },
            {
                "name": "视觉质量-审美质量",
                "prompt": "portrait of a woman, professional photography, natural lighting",
                "subcategory": "aesthetic_quality",
                "severity": "moderate"
            },
            {
                "name": "视觉质量-语义合理性",
                "prompt": "a person waving hello, clear facial features, realistic proportions",
                "subcategory": "semantic_plausibility",
                "severity": "severe"
            },
            {
                "name": "对齐度-基础识别",
                "prompt": "a red apple on a wooden table",
                "subcategory": "basic_recognition",
                "severity": "moderate"
            },
            {
                "name": "对齐度-属性对齐",
                "prompt": "a blue car parked on the street",
                "subcategory": "attribute_alignment",
                "severity": "moderate"
            },
            {
                "name": "对齐度-组合交互",
                "prompt": "three cats sitting on the left side of the sofa",
                "subcategory": "composition_interaction",
                "severity": "severe"
            }
        ]

        results = []

        for i, test in enumerate(test_cases, 1):
            print(f"\n【测试 {i}/{len(test_cases)}】{test['name']}")
            print("-" * 100)
            print(f"子类别: {test['subcategory']}")
            print(f"退化程度: {test['severity']}")
            print(f"正样本: {test['prompt']}")
            print()

            # LLM方法
            try:
                llm_negative, llm_info = llm_generator.generate_negative_prompt(
                    test['prompt'],
                    test['subcategory'],
                    test['severity']
                )
                print(f"✓ LLM方法:")
                print(f"  负样本: {llm_negative}")
                print(f"  退化信息: {llm_info}")

                # 保存结果
                results.append({
                    "test_name": test['name'],
                    "positive_prompt": test['prompt'],
                    "subcategory": test['subcategory'],
                    "severity": test['severity'],
                    "llm_negative": llm_negative,
                    "llm_info": llm_info,
                    "llm_success": True
                })

            except Exception as e:
                print(f"✗ LLM方法失败: {e}")
                results.append({
                    "test_name": test['name'],
                    "positive_prompt": test['prompt'],
                    "subcategory": test['subcategory'],
                    "severity": test['severity'],
                    "llm_success": False,
                    "llm_error": str(e)
                })

            print()

        # 保存测试结果
        output_file = "/root/ImageReward/data_generation/test_llm_degradation_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print("\n" + "=" * 100)
        print(f"测试完成！结果已保存到: {output_file}")
        print("=" * 100)

        # 统计
        success_count = sum(1 for r in results if r.get('llm_success', False))
        print(f"\n成功率: {success_count}/{len(results)} ({success_count/len(results)*100:.1f}%)")

        return True

    except Exception as e:
        logger.error(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_comparison():
    """对比LLM方法与关键词方法"""

    llm_config_path = "/root/ImageReward/data_generation/config/llm_config.yaml"
    quality_dimensions_path = "/root/ImageReward/data_generation/config/quality_dimensions.json"

    if not os.getenv("OPENAI_API_KEY"):
        logger.error("未设置OPENAI_API_KEY，跳过对比测试")
        return

    print("\n" + "=" * 100)
    print("LLM方法 vs 关键词方法对比测试")
    print("=" * 100)

    # 初始化两个生成器
    llm_gen = LLMPromptDegradation(
        llm_config_path=llm_config_path,
        quality_dimensions_path=quality_dimensions_path
    )
    keyword_gen = PromptDegradation(quality_dimensions_path)

    # 测试prompt
    test_prompt = "a cute cat sitting on a red velvet chair, professional photography, high quality"
    subcategory = "low_visual_quality"
    severity = "moderate"

    print(f"\n正样本prompt: {test_prompt}")
    print(f"退化维度: {subcategory}")
    print(f"退化程度: {severity}\n")

    # LLM方法生成3个样本
    print("【LLM方法】生成3个负样本:")
    for i in range(3):
        negative, info = llm_gen.generate_negative_prompt(test_prompt, subcategory, severity)
        print(f"  {i+1}. {negative}")

    print()

    # 关键词方法生成3个样本
    print("【关键词方法】生成3个负样本:")
    all_types = keyword_gen.get_all_degradation_types()
    # 筛选出low_visual_quality的退化类型
    low_quality_types = [t for t in all_types if t['subcategory'] == subcategory]

    for i, deg_type in enumerate(low_quality_types[:3]):
        negative, info = keyword_gen.generate_negative_prompt(test_prompt, deg_type, severity)
        print(f"  {i+1}. [{deg_type['attribute']}] {negative}")

    print("\n" + "=" * 100)


def test_batch_generation():
    """测试批量生成"""

    llm_config_path = "/root/ImageReward/data_generation/config/llm_config.yaml"
    quality_dimensions_path = "/root/ImageReward/data_generation/config/quality_dimensions.json"

    if not os.getenv("OPENAI_API_KEY"):
        logger.error("未设置OPENAI_API_KEY，跳过批量生成测试")
        return

    print("\n" + "=" * 100)
    print("批量生成测试")
    print("=" * 100)

    llm_gen = LLMPromptDegradation(
        llm_config_path=llm_config_path,
        quality_dimensions_path=quality_dimensions_path
    )

    # 测试prompts
    test_prompts = [
        "a beautiful landscape with mountains and rivers",
        "portrait of a smiling person",
        "a modern building in the city"
    ]

    print(f"\n批量生成 {len(test_prompts)} 个负样本...")

    results = llm_gen.generate_batch_negatives(test_prompts)

    for i, (pos, neg, info) in enumerate(results, 1):
        print(f"\n{i}. 正样本: {pos}")
        print(f"   负样本: {neg}")
        print(f"   退化: {info['dimension']} ({info['severity']})")

    print("\n" + "=" * 100)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="测试LLM退化prompt生成功能")
    parser.add_argument(
        "--test",
        choices=["basic", "comparison", "batch", "all"],
        default="all",
        help="测试类型: basic=基础测试, comparison=对比测试, batch=批量测试, all=全部测试"
    )

    args = parser.parse_args()

    if args.test in ["basic", "all"]:
        test_llm_degradation()

    if args.test in ["comparison", "all"]:
        test_comparison()

    if args.test in ["batch", "all"]:
        test_batch_generation()
