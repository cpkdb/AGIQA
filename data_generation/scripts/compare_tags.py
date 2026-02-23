#!/usr/bin/env python3
"""
对比新旧标注文件的差异
用法: python compare_tags.py --tag has_hand --show-removed 10
"""

import json
import argparse
import random

def main():
    parser = argparse.ArgumentParser(description='对比新旧标注差异')
    parser.add_argument('--tag', type=str, required=True, help='要对比的标签')
    parser.add_argument('--show-removed', type=int, default=10, help='显示被移除标签的样本数')
    parser.add_argument('--show-added', type=int, default=10, help='显示新增标签的样本数')
    args = parser.parse_args()
    
    # 读取新旧文件
    with open('/root/ImageReward/data_generation/data/prompts_tagged_sdxl.json', 'r') as f:
        old_data = json.load(f)
    
    with open('/root/ImageReward/data_generation/data/prompts_tagged_sdxl_v2.json', 'r') as f:
        new_data = json.load(f)
    
    # 创建prompt到标签的映射
    old_tags_map = {p['prompt']: set(p.get('semantic_tags', [])) for p in old_data['prompts']}
    new_tags_map = {p['prompt']: set(p.get('semantic_tags', [])) for p in new_data['prompts']}
    
    # 找出被移除和新增的样本
    removed_prompts = []
    added_prompts = []
    
    for prompt in old_tags_map.keys():
        old_has_tag = args.tag in old_tags_map[prompt]
        new_has_tag = args.tag in new_tags_map.get(prompt, set())
        
        if old_has_tag and not new_has_tag:
            removed_prompts.append(prompt)
        elif not old_has_tag and new_has_tag:
            added_prompts.append(prompt)
    
    # 统计
    old_count = sum(1 for tags in old_tags_map.values() if args.tag in tags)
    new_count = sum(1 for tags in new_tags_map.values() if args.tag in tags)
    
    print(f"\n{'='*80}")
    print(f"标签对比: {args.tag}")
    print(f"{'='*80}")
    print(f"旧版本数量: {old_count}")
    print(f"新版本数量: {new_count}")
    print(f"移除数量: {len(removed_prompts)}")
    print(f"新增数量: {len(added_prompts)}")
    print(f"{'='*80}\n")
    
    # 显示被移除的样本（这些是误标）
    if removed_prompts and args.show_removed > 0:
        print(f"\n{'='*80}")
        print(f"被移除的标签（可能是误标）- 随机显示 {min(args.show_removed, len(removed_prompts))} 个")
        print(f"{'='*80}\n")
        
        samples = random.sample(removed_prompts, min(args.show_removed, len(removed_prompts)))
        for i, prompt in enumerate(samples, 1):
            print(f"[{i}] {prompt[:200]}...")
            print(f"{'-'*80}\n")
    
    # 显示新增的样本（这些是之前漏标的）
    if added_prompts and args.show_added > 0:
        print(f"\n{'='*80}")
        print(f"新增的标签（之前漏标）- 随机显示 {min(args.show_added, len(added_prompts))} 个")
        print(f"{'='*80}\n")
        
        samples = random.sample(added_prompts, min(args.show_added, len(added_prompts)))
        for i, prompt in enumerate(samples, 1):
            print(f"[{i}] {prompt[:200]}...")
            print(f"{'-'*80}\n")

if __name__ == '__main__':
    main()
