# 简化版使用指南

## 代码改进说明

已经简化了 `generate_dataset.py` 的代码逻辑：

### 主要改进

1. **统一的退化生成器接口**
   - 使用 `self.degradation_generator` 统一管理
   - 根据 `--use_llm` 参数自动选择 LLM 或关键词方法
   - 不再需要同时初始化两个生成器

2. **清晰的方法选择**
   - 默认：关键词方法（快速、免费）
   - 加 `--use_llm`：LLM 方法（更自然、有成本）

3. **移除了旧代码**
   - 删除了 `generate_dataset()` 旧方法
   - 删除了 `--use_old_method` 参数
   - 只保留正样本复用策略

## 使用方法

### 1. 使用关键词方法（默认，推荐）

```bash
python scripts/generate_dataset.py \
    --source_prompts data/example_prompts.json \
    --output_dir /root/autodl-tmp/dataset_keyword \
    --num_negatives_per_positive 10 \
    --num_inference_steps 50 \
    --base_seed 42
```

**特点**：
- ✅ 快速（无 API 调用）
- ✅ 免费
- ✅ 已验证有效
- ✅ 12GB 显存即可运行

### 2. 使用 LLM 方法（更自然）

```bash
# 确保设置了 API key
export OPENAI_API_KEY="your-api-key"

python scripts/generate_dataset.py \
    --source_prompts data/example_prompts.json \
    --output_dir /root/autodl-tmp/dataset_llm \
    --num_negatives_per_positive 10 \
    --use_llm \
    --num_inference_steps 50 \
    --base_seed 42
```

**特点**：
- ✅ 更自然的语言表达
- ✅ 更好的语义理解
- ✅ 可处理复杂的对齐度退化
- ⚠️ 有 API 成本（GPT-4o: ~$0.003/次）
- ⚠️ 速度较慢（需要 API 调用）

## 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--source_prompts` | 正样本 prompt 文件（JSON） | 必需 |
| `--output_dir` | 输出目录 | `/root/autodl-tmp/dataset_v1` |
| `--num_negatives_per_positive` | 每个正样本的负样本数 | 10 |
| `--use_llm` | 使用 LLM 方法 | False（默认关键词） |
| `--num_inference_steps` | SDXL 推理步数 | 50 |
| `--guidance_scale` | CFG scale | 7.5 |
| `--base_seed` | 随机种子 | 42 |
| `--balance_severity` | 平衡退化程度 | False（使用 20%/40%/40%） |

## 代码结构

```python
DatasetGenerator
├── __init__()
│   ├── 初始化 SDXL 生成器
│   └── 根据 use_llm 选择退化生成器
│       ├── LLM 方法: LLMPromptDegradation
│       └── 关键词方法: PromptDegradation
│
├── _get_all_degradation_types()  # 统一接口
│   ├── LLM: 返回子类别级别
│   └── 关键词: 返回属性级别
│
├── _generate_negative_prompt()  # 统一接口
│   ├── LLM: 调用 generate_negative_prompt()
│   └── 关键词: 调用 generate_prompt_pair()
│
└── generate_dataset_with_reuse()  # 主生成方法
    ├── 生成正样本图像（每个 prompt 一次）
    └── 生成 N 个负样本图像（使用统一接口）
```

## 快速测试

```bash
# 1. 进入目录
cd /root/ImageReward/data_generation

# 2. 测试关键词方法（2 个正样本 × 3 个负样本 = 6 对）
python scripts/generate_dataset.py \
    --source_prompts data/demo_prompts.json \
    --output_dir /root/autodl-tmp/test_keyword \
    --num_negatives_per_positive 3 \
    --num_inference_steps 30

# 3. 测试 LLM 方法（需要 API key）
export OPENAI_API_KEY="sk-..."
python scripts/generate_dataset.py \
    --source_prompts data/demo_prompts.json \
    --output_dir /root/autodl-tmp/test_llm \
    --num_negatives_per_positive 3 \
    --use_llm \
    --num_inference_steps 30

# 4. 查看结果
ls /root/autodl-tmp/test_keyword/images/
cat /root/autodl-tmp/test_keyword/dataset.json | jq '.metadata'
```

## 输出结构

```
/root/autodl-tmp/dataset_v1/
├── images/
│   ├── positive_42.png          # 正样本
│   ├── negative_42_0.png        # 负样本 1
│   ├── negative_42_1.png        # 负样本 2
│   ├── negative_42_2.png        # 负样本 3
│   └── ...
├── dataset.json                 # 完整元数据
│   ├── metadata
│   │   ├── degradation_method: "LLM-based" 或 "keyword-based"
│   │   ├── total_positive_images
│   │   ├── total_negative_images
│   │   └── total_pairs
│   └── pairs[]
│       ├── pair_id: "0000000"
│       ├── positive: {prompt, image_path, shared_seed}
│       ├── negative: {prompt, image_path}
│       └── degradation: {category, subcategory/attribute, severity}
└── summary.json
```

## 常见问题

### Q: 如何只使用 LLM 方法？

**A**: 加上 `--use_llm` 参数即可，代码会自动只使用 LLM 方法：

```bash
python scripts/generate_dataset.py \
    --source_prompts data/example_prompts.json \
    --use_llm \
    --output_dir /root/autodl-tmp/dataset_llm
```

### Q: LLM 方法的成本是多少？

**A**: 
- GPT-4o: ~$0.003/次，100 万对约 $3,000
- GPT-4o-mini: ~$0.0002/次，100 万对约 $200
- GPT-3.5-turbo: ~$0.0006/次，100 万对约 $600

在 `config/llm_config.yaml` 中修改模型：

```yaml
llm:
  model: "gpt-4o-mini"  # 更便宜的选择
```

### Q: 代码还有关键词方法的代码吗？

**A**: 有，但已经简化：
- 只有一个统一的 `self.degradation_generator`
- 根据 `--use_llm` 参数自动选择
- 不会同时加载两个生成器

### Q: 如何验证 LLM 是否正常工作？

**A**: 查看生成的 `dataset.json`：

```bash
cat dataset.json | jq '.metadata.degradation_method'
# 输出: "LLM-based" 或 "keyword-based"

cat dataset.json | jq '.pairs[0].degradation.method'
# 输出: "llm" 或 "keyword"
```

## 总结

现在的代码更简洁：
- ✅ 统一的接口设计
- ✅ 清晰的方法选择（一个参数控制）
- ✅ 移除了冗余代码
- ✅ 保持了灵活性

你只需要记住：
- **默认 = 关键词方法**（快速、免费）
- **加 `--use_llm` = LLM 方法**（更自然、有成本）
