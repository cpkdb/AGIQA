# LLM退化Prompt生成功能使用指南

## 概述

本项目新增了基于LLM（如GPT-4o）的质量退化prompt生成功能，作为原有关键词方法的增强版本。

### 两种方法对比

| 维度 | 关键词方法 | **LLM方法** |
|------|-----------|-------------|
| **多样性** | 固定关键词列表，多样性有限 | 每次生成不同表达，多样性极高 |
| **自然性** | 简单拼接关键词，可能不够自然 | LLM保证语言流畅自然 |
| **对齐度退化** | 未真正实现，返回原prompt | 可智能替换对象/颜色/动作等 |
| **成本** | 免费 | 需要API调用费用 |
| **速度** | 极快（毫秒级） | 较慢（秒级，依赖API） |
| **可控性** | 高度可控 | 依赖LLM理解能力 |

### 输入层级

LLM方法使用**子类别（subcategory）级别**作为输入，而不是具体的属性：

- **Visual Quality类（3个子类别）**：
  - `low_visual_quality` - 技术质量问题（blur/noise/exposure等）
  - `aesthetic_quality` - 审美质量（composition/lighting等）
  - `semantic_plausibility` - 语义合理性（anatomy/physics等）

- **Alignment类（4个子类别）**：
  - `basic_recognition` - 基础识别（主要/次要对象）
  - `attribute_alignment` - 属性对齐（颜色/形状/动作等）
  - `composition_interaction` - 组合交互（数量/位置/大小）
  - `external_knowledge` - 外部知识（地理/品牌/风格）

## 安装依赖

```bash
# 安装OpenAI库
pip install openai

# 或使用项目requirements
pip install -r requirements.txt
```

## 配置API密钥

### 方法1: 环境变量（推荐）

```bash
export OPENAI_API_KEY="your-api-key-here"
```

将此行添加到 `~/.bashrc` 或 `~/.zshrc` 以永久保存：

```bash
echo 'export OPENAI_API_KEY="your-api-key-here"' >> ~/.bashrc
source ~/.bashrc
```

### 方法2: 直接修改配置文件（不推荐）

编辑 `data_generation/config/llm_config.yaml`:

```yaml
llm:
  api_key: "your-api-key-here"  # 直接填写（不要提交到git）
```

⚠️ **注意**：不要将包含真实API密钥的配置文件提交到git仓库！

## 配置说明

编辑 `data_generation/config/llm_config.yaml` 根据需求调整：

```yaml
llm:
  provider: "openai"              # API提供商
  model: "gpt-4o"                 # 使用的模型
  temperature: 0.7                # 创造性控制（0.5-0.9推荐）
  max_tokens: 150                 # 最大生成长度
  timeout: 30                     # 请求超时时间
  max_retries: 3                  # 失败重试次数

degradation:
  fallback_to_keywords: true      # API失败时回退到关键词方法
  validate_output: true           # 验证LLM输出质量
```

### 推荐配置

- **高质量 + 高成本**: `gpt-4o`, temperature=0.7
- **平衡**: `gpt-4o-mini`, temperature=0.7
- **低成本**: `gpt-3.5-turbo`, temperature=0.8

## 使用方法

### 1. 测试LLM功能

```bash
cd /root/ImageReward/data_generation/scripts

# 基础测试
python test_llm_degradation.py --test basic

# 对比LLM vs 关键词方法
python test_llm_degradation.py --test comparison

# 批量生成测试
python test_llm_degradation.py --test batch

# 全部测试
python test_llm_degradation.py --test all
```

### 2. 在数据集生成中使用LLM

```bash
cd /root/ImageReward/data_generation/scripts

# 使用LLM生成退化prompt
python generate_dataset.py \
  --source_prompts /path/to/prompts.json \
  --output_dir /root/autodl-tmp/dataset_llm \
  --num_negatives_per_positive 10 \
  --use_llm  # 启用LLM模式

# 仍然使用关键词方法（默认）
python generate_dataset.py \
  --source_prompts /path/to/prompts.json \
  --output_dir /root/autodl-tmp/dataset_keyword \
  --num_negatives_per_positive 10
  # 不加--use_llm参数
```

### 3. Python代码中使用

```python
from llm_prompt_degradation import LLMPromptDegradation

# 初始化
generator = LLMPromptDegradation(
    llm_config_path="/root/ImageReward/data_generation/config/llm_config.yaml",
    quality_dimensions_path="/root/ImageReward/data_generation/config/quality_dimensions.json"
)

# 生成单个负样本
positive_prompt = "a beautiful sunset over the ocean, high quality"
negative_prompt, degradation_info = generator.generate_negative_prompt(
    positive_prompt=positive_prompt,
    subcategory="low_visual_quality",  # 子类别级别
    severity="moderate"
)

print(f"正样本: {positive_prompt}")
print(f"负样本: {negative_prompt}")
print(f"退化信息: {degradation_info}")

# 批量生成
prompts = [
    "a red apple on the table",
    "portrait of a smiling woman",
    "modern architecture in the city"
]

results = generator.generate_batch_negatives(prompts)
for pos, neg, info in results:
    print(f"正: {pos}")
    print(f"负: {neg}")
    print(f"退化: {info['dimension']} ({info['severity']})")
```

## 成本估算

基于OpenAI官方定价（2024年价格，可能有变化）：

| 模型 | 输入价格 | 输出价格 | 每次调用预估 | 10万次调用 |
|------|---------|---------|-------------|-----------|
| gpt-4o | $2.50/1M tokens | $10.00/1M tokens | ~$0.003 | ~$300 |
| gpt-4o-mini | $0.15/1M tokens | $0.60/1M tokens | ~$0.0002 | ~$20 |
| gpt-3.5-turbo | $0.50/1M tokens | $1.50/1M tokens | ~$0.0006 | ~$60 |

**示例计算**（生成100万对数据集）:
- 10万个正样本 × 每个10个负样本 = 100万次API调用
- 使用gpt-4o: ~$3,000
- 使用gpt-4o-mini: ~$200
- 使用gpt-3.5-turbo: ~$600

💡 **节省成本的建议**:
1. 先用少量数据测试（100个样本）验证效果
2. 使用 `gpt-4o-mini` 而非 `gpt-4o`
3. 启用缓存（在配置中设置 `enable_cache: true`）
4. 仅对alignment退化使用LLM，visual_quality仍用关键词方法

## 故障排除

### 问题1: ImportError: No module named 'openai'

**解决**:
```bash
pip install openai
```

### 问题2: API调用失败 - 401 Unauthorized

**原因**: API密钥未设置或无效

**解决**:
```bash
# 检查环境变量
echo $OPENAI_API_KEY

# 重新设置
export OPENAI_API_KEY="sk-your-real-api-key"
```

### 问题3: API调用超时

**原因**: 网络问题或API服务繁忙

**解决**: 配置文件中增加超时时间和重试次数
```yaml
llm:
  timeout: 60        # 增加到60秒
  max_retries: 5     # 增加重试次数
```

### 问题4: 生成的负样本质量不佳

**解决方法**:
1. 调整temperature（0.5-0.9之间尝试）
2. 检查System Prompt是否合理
3. 尝试不同的模型（gpt-4o vs gpt-3.5-turbo）
4. 启用输出验证 `validate_output: true`

### 问题5: Fallback到关键词方法

**原因**: LLM API调用失败后自动降级

**解决**:
- 检查API密钥和网络
- 如不希望fallback，设置 `fallback_to_keywords: false`

## 性能优化

### 1. 批处理

使用 `generate_batch_negatives()` 而非多次调用单个生成：

```python
# ❌ 慢：多次单独调用
for prompt in prompts:
    negative, info = generator.generate_negative_prompt(prompt, subcategory, severity)

# ✅ 快：批量调用
results = generator.generate_batch_negatives(prompts)
```

### 2. 并发处理

在配置中设置合理的速率限制：

```yaml
llm:
  batch_size: 10           # 每批处理10个
  rate_limit_rpm: 60       # 每分钟最多60个请求（根据API套餐调整）
```

### 3. 缓存

启用缓存避免重复调用相同prompt：

```yaml
degradation:
  enable_cache: true
  cache_dir: "/root/ImageReward/data_generation/.cache/llm_degradation"
```

## 与关键词方法的混合使用

推荐策略：**Visual Quality用关键词，Alignment用LLM**

原因：
- Visual Quality的关键词方法已经很有效
- Alignment退化的关键词方法未真正实现，需要LLM

实现方式（修改代码）：

```python
# 在generate_dataset_with_reuse中
if degradation_info['category'] == 'alignment':
    # 对齐度退化使用LLM
    negative_prompt, info = llm_generator.generate_negative_prompt(...)
else:
    # 视觉质量退化使用关键词
    negative_prompt, info = keyword_generator.generate_negative_prompt(...)
```

## 质量验证

生成后验证负样本质量：

```bash
# 检查生成结果
python -c "
import json
with open('/root/autodl-tmp/dataset_llm/dataset.json') as f:
    data = json.load(f)

# 统计LLM vs 关键词方法
llm_count = sum(1 for p in data['pairs'] if p['degradation'].get('method') == 'llm')
keyword_count = sum(1 for p in data['pairs'] if p['degradation'].get('method') == 'keyword')

print(f'LLM方法: {llm_count}')
print(f'关键词方法: {keyword_count}')
print(f'总计: {len(data[\"pairs\"])}')

# 查看示例
for pair in data['pairs'][:3]:
    print(f\"正: {pair['positive']['prompt']}\")
    print(f\"负: {pair['negative']['prompt']}\")
    print(f\"退化: {pair['degradation']}\")
    print()
"
```

## 文件结构

```
data_generation/
├── config/
│   ├── llm_config.yaml                 # LLM配置文件
│   └── quality_dimensions.json         # 质量维度定义
├── scripts/
│   ├── llm_prompt_degradation.py       # LLM退化生成器
│   ├── prompt_degradation.py           # 关键词退化生成器
│   ├── generate_dataset.py             # 主生成脚本（已集成LLM）
│   └── test_llm_degradation.py         # 测试脚本
└── LLM_DEGRADATION_GUIDE.md           # 本文档
```

## 下一步

1. ✅ 测试LLM功能: `python test_llm_degradation.py --test all`
2. ✅ 小规模验证: 生成100对样本验证效果和成本
3. ⏳ 大规模生成: 确认效果后生成完整数据集
4. ⏳ 对比评估: 使用ImageReward评估两种方法的效果差异

## 参考

- OpenAI API文档: https://platform.openai.com/docs/api-reference
- 本项目文档: `/root/ImageReward/data_generation/DATASET_GENERATION_PLAN.md`
- 质量维度定义: Section 2.2 (Visual Quality & Alignment)
