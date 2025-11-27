#!/usr/bin/env python3
"""
测试 API 配置是否正确
验证 LLM 退化生成器是否可以正常工作
"""

import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from llm_prompt_degradation import LLMPromptDegradation

def test_api_config():
    """测试 API 配置"""
    
    print("=" * 80)
    print("测试 API 配置")
    print("=" * 80)
    
    # 配置路径
    llm_config_path = Path(__file__).parent.parent / "config" / "llm_config.yaml"
    quality_dimensions_path = Path(__file__).parent.parent / "config" / "quality_dimensions.json"
    
    print(f"\n配置文件路径:")
    print(f"  LLM Config: {llm_config_path}")
    print(f"  Quality Dimensions: {quality_dimensions_path}")
    
    # 初始化生成器
    print("\n正在初始化 LLM 退化生成器...")
    try:
        generator = LLMPromptDegradation(
            llm_config_path=str(llm_config_path),
            quality_dimensions_path=str(quality_dimensions_path)
        )
        print("✅ 初始化成功！")
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        return False
    
    # 测试生成
    print("\n" + "=" * 80)
    print("测试退化 prompt 生成")
    print("=" * 80)
    
    test_cases = [
        {
            "prompt": "a beautiful sunset over the ocean, masterpiece, high quality",
            "subcategory": "low_visual_quality",
            "severity": "moderate"
        },
        {
            "prompt": "a red apple on a wooden table",
            "subcategory": "attribute_alignment",
            "severity": "moderate"
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n【测试 {i}】")
        print(f"正样本: {test['prompt']}")
        print(f"退化维度: {test['subcategory']}")
        print(f"退化程度: {test['severity']}")
        
        try:
            negative_prompt, degradation_info = generator.generate_negative_prompt(
                test['prompt'],
                test['subcategory'],
                test['severity']
            )
            
            print(f"✅ 生成成功！")
            print(f"负样本: {negative_prompt}")
            print(f"退化信息: {degradation_info}")
            
        except Exception as e:
            print(f"❌ 生成失败: {e}")
            return False
    
    print("\n" + "=" * 80)
    print("✅ 所有测试通过！API 配置正确，可以正常使用。")
    print("=" * 80)
    
    return True


if __name__ == "__main__":
    success = test_api_config()
    sys.exit(0 if success else 1)
