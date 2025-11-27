# API 配置指南

## 问题说明

你提到的两个问题已经修复：

### 1. ✅ 退化类型和程度已作为输入传给 LLM

**修改前**：User prompt 只包含正样本提示词

**修改后**：User prompt 明确包含：
- 正样本提示词
- 退化维度（subcategory）
- 退化程度（severity）

```python
user_prompt = f"""请为以下正样本提示词生成质量退化的负样本提示词：

**正样本提示词**: {positive_prompt}

**退化要求**:
- 退化维度: {subcategory}
- 退化程度: {severity}

请根据上述退化原则和具体指令，生成一个退化后的负样本提示词。只返回修改后的提示词文本，不要有任何解释。"""
```

同时，System prompt 中也包含了详细的退化原则和具体指令。

### 2. ✅ API key 支持永久配置

**修改后的配置优先级**：

1. **配置文件中直接填写**（推荐，永久配置）
2. 环境变量 `OPENAI_API_KEY`
3. 配置文件中使用占位符 `${OPENAI_API_KEY}`

## API 配置方法

### 方法 1：直接在配置文件中填写（推荐）

编辑 `config/llm_config.yaml`：

```yaml
llm:
  provider: "openai"
  model: "gpt-4o"
  
  # 直接填写你的 API key（永久配置）
  api_key: "sk-your-actual-api-key-here"
  
  # 如果使用代理或自定义 endpoint
  api_base: "https://api.chatanywhere.org"
```

**优点**：
- ✅ 一次配置，永久有效
- ✅ 不需要每次运行都设置环境变量
- ✅ 代码会自动使用配置文件中的 key

**注意**：
- ⚠️ 不要将包含真实 API key 的配置文件提交到公开的 git 仓库
- ⚠️ 建议将 `llm_config.yaml` 添加到 `.gitignore`

### 方法 2：使用环境变量（临时使用）

```bash
# 临时设置（当前终端会话有效）
export OPENAI_API_KEY="sk-your-api-key"

# 永久设置（添加到 ~/.bashrc 或 ~/.zshrc）
echo 'export OPENAI_API_KEY="sk-your-api-key"' >> ~/.bashrc
source ~/.bashrc
```

配置文件中留空或使用占位符：

```yaml
llm:
  # 方式 A: 留空，从环境变量读取
  api_key: ""
  
  # 方式 B: 使用占位符
  api_key: "${OPENAI_API_KEY}"
```

### 方法 3：混合使用

配置文件中填写默认 key，需要时用环境变量覆盖：

```bash
# 使用配置文件中的 key
python scripts/generate_dataset.py --use_llm ...

# 临时使用其他 key
export OPENAI_API_KEY="sk-another-key"
python scripts/generate_dataset.py --use_llm ...
```

## 当前配置状态

查看你的 `config/llm_config.yaml`：

```yaml
llm:
  provider: "openai"
  model: "gpt-4o"
  api_key: "sk-OFeNoWrvYQr52dKoSWefZZOdweq4hQLiigkSGSDtTxWiGyU7"
  api_base: "https://api.chatanywhere.org"
```

**状态**：✅ 已经永久配置好了！

你现在可以直接运行，不需要每次设置环境变量：

```bash
# 直接运行，会自动使用配置文件中的 API key
python scripts/generate_dataset.py \
    --source_prompts data/example_prompts.json \
    --use_llm \
    --output_dir /root/autodl-tmp/dataset_llm
```

## 验证配置

### 测试 API 连接

```bash
cd /root/ImageReward/data_generation/scripts

# 运行测试脚本
python llm_prompt_degradation.py
```

如果配置正确，会看到：

```
初始化LLM退化生成器...
使用配置文件中的 API key
初始化 OpenAI 客户端 - Model: gpt-4o
使用自定义 API Base: https://api.chatanywhere.org
LLM退化生成器初始化完成 - Model: gpt-4o

================================================================================
测试LLM退化生成
================================================================================

【测试 1】
子类别: low_visual_quality
退化程度: moderate
正样本: a beautiful sunset over the ocean, masterpiece, high quality, detailed
负样本: a beautiful sunset over the ocean, noticeably blurred and out of focus
...
```

### 检查配置加载

在代码中添加日志查看：

```python
# 会自动打印使用的配置方式
# "使用配置文件中的 API key" 或 "使用环境变量 OPENAI_API_KEY"
```

## 配置文件说明

### 完整配置示例

```yaml
llm:
  provider: "openai"
  model: "gpt-4o"              # 或 gpt-4o-mini（更便宜）
  api_key: "sk-your-key"       # 直接填写
  api_base: "https://api.chatanywhere.org"  # 可选，代理地址
  
  # 生成参数
  temperature: 0.7             # 创造性，0.5-0.9
  max_tokens: 150              # 最大生成长度
  top_p: 0.95
  
  # 请求配置
  timeout: 30                  # 超时时间
  max_retries: 3               # 重试次数
  retry_delay: 2               # 重试延迟

degradation:
  fallback_to_keywords: true   # API 失败时回退到关键词方法
  validate_output: true        # 验证 LLM 输出
  min_length: 10               # 最小长度
  max_length: 200              # 最大长度

logging:
  level: "INFO"
  log_api_calls: true          # 记录 API 调用
```

## 常见问题

### Q1: 每次运行都要设置环境变量吗？

**A**: 不需要！直接在 `config/llm_config.yaml` 中填写 API key，一次配置永久有效。

### Q2: 如何切换不同的 API key？

**A**: 三种方式：
1. 修改配置文件中的 `api_key`
2. 临时设置环境变量 `export OPENAI_API_KEY="new-key"`
3. 使用不同的配置文件 `--llm_config path/to/another_config.yaml`

### Q3: 配置文件中的 key 会被环境变量覆盖吗？

**A**: 不会。优先级是：
1. 配置文件中的有效 key（以 `sk-` 开头）
2. 环境变量 `OPENAI_API_KEY`
3. 配置文件中的占位符 `${OPENAI_API_KEY}`

### Q4: 如何使用不同的模型？

**A**: 修改配置文件：

```yaml
llm:
  model: "gpt-4o-mini"  # 更便宜
  # 或
  model: "gpt-3.5-turbo"  # 最便宜
```

### Q5: 退化类型和程度是如何传给 LLM 的？

**A**: 通过两个地方：

1. **System Prompt**：包含详细的退化原则和具体指令
   ```python
   system_prompt = f"""
   # Degradation Dimension
   - 类别: {category}
   - 子类别: {subcategory}
   - 退化程度: {severity}
   
   # Specific Instructions
   ...（根据 subcategory 和 severity 生成的具体指令）
   """
   ```

2. **User Prompt**：明确说明退化要求
   ```python
   user_prompt = f"""
   正样本提示词: {positive_prompt}
   
   退化要求:
   - 退化维度: {subcategory}
   - 退化程度: {severity}
   """
   ```

这样 LLM 可以清楚地知道：
- 要退化哪个维度（如 low_visual_quality）
- 退化到什么程度（mild/moderate/severe）
- 具体应该如何修改（详细的指令）

## 使用示例

### 基本使用（使用配置文件中的 key）

```bash
python scripts/generate_dataset.py \
    --source_prompts data/example_prompts.json \
    --use_llm \
    --output_dir /root/autodl-tmp/dataset_llm \
    --num_negatives_per_positive 10
```

### 使用不同的配置文件

```bash
python scripts/generate_dataset.py \
    --source_prompts data/example_prompts.json \
    --use_llm \
    --llm_config config/llm_config_gpt35.yaml \
    --output_dir /root/autodl-tmp/dataset_gpt35
```

### 临时覆盖 API key

```bash
export OPENAI_API_KEY="sk-temporary-key"
python scripts/generate_dataset.py \
    --source_prompts data/example_prompts.json \
    --use_llm \
    --output_dir /root/autodl-tmp/dataset_temp
```

## 总结

现在的配置方式：

✅ **API key 永久配置**：直接写在 `llm_config.yaml` 中，不需要每次设置环境变量

✅ **退化信息完整传递**：退化类型、程度、原则都通过 System Prompt 和 User Prompt 传给 LLM

✅ **灵活的配置方式**：支持配置文件、环境变量、占位符三种方式

✅ **清晰的日志输出**：会显示使用的是配置文件还是环境变量

你现在可以直接运行代码，不需要任何额外配置！
