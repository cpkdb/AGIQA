# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# 永远使用中文回答

## 项目概述

本项目是 ImageReward 的数据生成模块，用于构建 AIGC 图像质量评估的自监督数据集。核心思路是通过 LLM 生成退化的 prompt，然后使用 SDXL 模型生成正负样本图像对，用于 BT 偏好模型训练。

## 核心架构

### 数据流程

```
正样本 Prompt → LLM 退化 → 负样本 Prompt
      ↓                         ↓
    SDXL                      SDXL
      ↓                         ↓
  正样本图像    +    负样本图像  =  训练对
```

### 主要组件

1. **LLMPromptDegradation** (`scripts/llm_prompt_degradation.py`)
   - 基于 GPT-4o 的 prompt 退化生成器
   - 输入：正样本 prompt + 子类别 + 退化程度
   - 输出：退化后的负样本 prompt

2. **SDXLGenerator** (`scripts/sdxl_generator.py`)
   - 封装 SDXL 图像生成
   - 支持相同 seed 保证正负样本内容一致性

3. **DatasetGenerator** (`scripts/generate_dataset.py`)
   - 整合 LLM 和 SDXL，实现正样本复用策略
   - 一个正样本对应 N 个不同退化的负样本

### 退化维度层级
```

```
### 退化维度原则
```
注意生成的负样本，还是需要保证在内容和风格尽量差不多的情况下，有一个明显的相应维度的退化才行，然后还需要考虑的是，你生成出来的负样本，应该尽量接近于生成图像的质量退化，而不是模拟拍摄图像的退化，这两个是不一样的。
你的核心是创造出符合人眼视觉一致的正负样本出来，只要生成出来的符合正样本的视觉质量（不是图文一致性）大于负样本，那就是有价值的训练样本
```

## 常用命令

### 生成数据集

```bash
# 基本用法：使用 LLM 生成退化 prompt
python scripts/generate_dataset.py \
    --source_prompts data/example_prompts.json \
    --output_dir /root/autodl-tmp/dataset_v1 \
    --num_negatives_per_positive 10 \
    --num_inference_steps 50 \
    --base_seed 42

# 指定退化类别
python scripts/generate_dataset.py \
    --source_prompts data/example_prompts.json \
    --output_dir /root/autodl-tmp/dataset_visual \
    --category_filter visual_quality \
    --num_negatives_per_positive 5

# 指定子类别和属性
python scripts/generate_dataset.py \
    --source_prompts data/example_prompts.json \
    --subcategory_filter low_visual_quality \
    --attribute_filter blur \
    --severity moderate

# 限制正样本数量
python scripts/generate_dataset.py \
    --source_prompts data/example_prompts.json \
    --num_positive_prompts 100 \
    --random_select_prompts
```

### 测试 LLM 退化

```bash
# 直接运行模块测试
python scripts/llm_prompt_degradation.py

# 使用测试脚本
python scripts/test_llm_degradation.py --test all
```

### 测试 SDXL 生成

```bash
python scripts/sdxl_generator.py
```

## 关键配置文件

- `config/llm_config.yaml`: LLM API 配置（模型、API key、参数）
- `config/quality_dimensions.json`: 退化维度定义
- `config/prompt_templates/*.yaml`: 属性级别的 System Prompt 模板
- `schema/dataset_schema.json`: 输出数据集格式定义

## 输出目录结构

```
/root/autodl-tmp/dataset_v1/
├── images/
│   ├── positive_{seed}.png      # 正样本（每个 seed 一张）
│   ├── negative_{seed}_0.png    # 负样本 1
│   ├── negative_{seed}_1.png    # 负样本 2
│   └── ...
├── dataset.json                 # 完整元数据
└── prompts_cache.json          # LLM 生成的退化 prompt 缓存（生成完成后删除）
```

## 依赖要求

- Python 3.10+
- PyTorch + CUDA
- diffusers (SDXL)
- openai (LLM API)
- 显存需求：最低 12GB（仅 SDXL），推荐 16GB+

## 注意事项

1. **API Key 配置**：可在 `config/llm_config.yaml` 中直接配置，或通过环境变量 `OPENAI_API_KEY` 设置
2. **正样本复用**：默认策略是一个正样本生成多个负样本，减少 SDXL 调用次数
3. **退化程度分布**：mild 20%, moderate 40%, severe 40%
4. **SDXL 模型路径**：默认 `/root/ckpts/sd_xl_base_1.0.safetensors`，不存在时自动从 HuggingFace 下载
