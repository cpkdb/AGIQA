#!/usr/bin/env python3
import sys
import json
import random
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from llm_prompt_degradation import LLMPromptDegradation

# 加载正样本
with open('/root/ImageReward/data_generation/data/example_prompts.json', 'r') as f:
    positive_prompts = json.load(f)['prompts'][:20]

# 初始化生成器
generator = LLMPromptDegradation(
    llm_config_path='/root/ImageReward/data_generation/config/llm_config.yaml',
    quality_dimensions_path='/root/ImageReward/data_generation/config/quality_dimensions.json'
)

# 生成退化程度分布
severities = ['mild'] * 2 + ['moderate'] * 12 + ['severe'] * 6
random.shuffle(severities)

print("=" * 80)
print("开始生成20对LLM退化prompt...")
print("=" * 80)

results = []
for i, prompt in enumerate(positive_prompts, 1):
    severity = severities[i-1]
    print(f"\n[{i}/20] {severity}")
    try:
        neg_prompt, deg_info = generator.generate_negative_prompt(
            prompt, 'low_visual_quality', severity
        )
        results.append({
            'id': i,
            'positive_prompt': prompt,
            'negative_prompt': neg_prompt,
            'degradation_info': deg_info
        })
        print(f"✓ {neg_prompt[:60]}...")
    except Exception as e:
        print(f"✗ 失败: {e}")

# 保存结果
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = f'/root/ImageReward/data_generation/demo_output/llm_degradation_results_{timestamp}.json'

severity_counts = {}
for r in results:
    s = r['degradation_info']['severity']
    severity_counts[s] = severity_counts.get(s, 0) + 1

output_data = {
    'metadata': {
        'generated_at': datetime.now().isoformat(),
        'subcategory': 'low_visual_quality',
        'severity_distribution': severity_counts,
        'total_prompts': 20,
        'success_count': len(results),
        'fail_count': 20 - len(results)
    },
    'results': results
}

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(output_data, f, ensure_ascii=False, indent=2)

print(f"\n保存到: {output_file}")
print(f"成功: {len(results)}/20")
