#!/usr/bin/env python3
"""
验证语义标签的准确性
用法: python verify_tags.py --tag has_hand --sample 10
"""

import json
import random
import argparse

def main():
    parser = argparse.ArgumentParser(description='验证语义标签准确性')
    parser.add_argument('--tag', type=str, required=True, help='要验证的标签')
    parser.add_argument('--sample', type=int, default=10, help='抽样数量')
    parser.add_argument('--file', type=str, 
                       default='/root/ImageReward/data_generation/data/prompts_tagged_sdxl_v2.json',
                       help='标注文件路径')
    args = parser.parse_args()
    
    # 读取数据
    with open(args.file, 'r') as f:
        data = json.load(f)
    
    # 筛选包含指定标签的prompt
    tagged_prompts = [p for p in data['prompts'] if args.tag in p.get('semantic_tags', [])]
    
    print(f"\n{'='*80}")
    print(f"标签验证: {args.tag}")
    print(f"{'='*80}")
    print(f"总数: {len(tagged_prompts)}")
    print(f"抽样数: {min(args.sample, len(tagged_prompts))}")
    print(f"{'='*80}\n")
    
    # 随机抽样
    samples = random.sample(tagged_prompts, min(args.sample, len(tagged_prompts)))
    
    for i, p in enumerate(samples, 1):
        print(f"\n[样本 {i}]")
        print(f"Prompt: {p['prompt']}")
        print(f"所有标签: {', '.join(p.get('semantic_tags', []))}")
        print(f"{'-'*80}")

if __name__ == '__main__':
    main()
