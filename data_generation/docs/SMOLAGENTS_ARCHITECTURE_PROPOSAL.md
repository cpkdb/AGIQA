# smolagents 架构详细方案 (v2)

**状态**: 待确认
**日期**: 2024-XX-XX
**更新**: 根据用户反馈调整

---

## 0. 核心设计决策

### 0.1 模型分工

| 组件 | 模型 | 用途 |
|-----|------|------|
| **prompt_degrader** | GPT-4o | 生成退化 prompt（现有机制保留） |
| **image_generator** | SDXL / 未来其他模型 | 生成图像（多模型支持） |
| **degradation_judge** | Gemini | VLM 判别图像对是否有效 |
| **编排层** | 无（Python 循环） | 流程固定，无需 Agent 自主决策 |

### 0.2 配置统一

所有 LLM/VLM 调用使用**相同的配置格式**（`config/llm_config.yaml`），只需修改 `model` 和 `api_key`：

```yaml
# GPT-4o 配置（prompt 退化）
degradation_llm:
  provider: openai
  model: gpt-4o
  api_key: ${OPENAI_API_KEY}
  api_base: https://api.openai.com/v1  # 可选

# Gemini 配置（VLM 判别）
judge_vlm:
  provider: google
  model: gemini-1.5-pro-vision
  api_key: ${GOOGLE_API_KEY}
```

### 0.3 维度配置

使用最新的 v3 配置：
- **维度定义**: `config/quality_dimensions_v3.json`（3 perspectives, 50 dimensions）
- **策略模板**: `config/prompt_templates_v3/*.yaml`（6 个文件）

---

## 1. 架构总览

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        smolagents Pipeline                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌──────────────────────────────────────────────────────────────────────┐  │
│   │                         Tool Layer (@tool)                            │  │
│   │                                                                       │  │
│   │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐   │  │
│   │  │ prompt_degrader │  │ image_generator │  │ degradation_judge   │   │  │
│   │  │ (现有代码封装)   │  │ (现有代码封装)   │  │ (新增 VLM 判别)     │   │  │
│   │  └─────────────────┘  └─────────────────┘  └─────────────────────┘   │  │
│   └──────────────────────────────────────────────────────────────────────┘  │
│                                      │                                       │
│                                      ↓                                       │
│   ┌──────────────────────────────────────────────────────────────────────┐  │
│   │                      CodeAgent (编排层)                               │  │
│   │                                                                       │  │
│   │   输入: positive_prompts, dimension, severity                        │  │
│   │                                                                       │  │
│   │   循环:                                                               │  │
│   │     1. 调用 prompt_degrader → 生成退化 prompt                        │  │
│   │     2. 调用 image_generator → 生成正负图像                            │  │
│   │     3. 调用 degradation_judge → 判断是否有效                          │  │
│   │     4. 有效 → 保存 | 无效 → 记录问题                                  │  │
│   │                                                                       │  │
│   │   输出: 有效的图像对 + 统计报告                                       │  │
│   └──────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Tool 层详细设计

### 2.1 Tool 1: `prompt_degrader`

**功能**: 将正样本 Prompt 退化为负样本 Prompt（封装现有 `LLMPromptDegradation`）

```python
from smolagents import tool

@tool
def prompt_degrader(
    positive_prompt: str,
    subcategory: str,
    attribute: str = None,
    severity: str = "moderate"
) -> str:
    """
    Generate a degraded (lower quality) version of the input prompt.

    Args:
        positive_prompt: The original high-quality prompt to degrade.
        subcategory: The degradation category (e.g., 'technical_quality', 'anatomical_accuracy').
        attribute: Specific attribute to degrade (e.g., 'blur', 'hand_deformity').
                   If None, randomly selects one from the subcategory.
        severity: Degradation intensity - 'mild', 'moderate', or 'severe'.

    Returns:
        JSON string containing:
        - negative_prompt: The degraded prompt
        - degradation_info: Metadata about the degradation applied
    """
    # 内部实现: 调用现有 LLMPromptDegradation
```

**输入/输出示例**:
```python
# 输入
positive_prompt = "a beautiful sunset over the ocean, masterpiece, high quality"
subcategory = "technical_quality"
attribute = "blur"
severity = "moderate"

# 输出 (JSON string)
{
    "negative_prompt": "a beautiful sunset over the ocean, blurry, out of focus",
    "degradation_info": {
        "category": "technical_quality",
        "subcategory": "technical_quality",
        "attribute": "blur",
        "severity": "moderate",
        "method": "llm"
    }
}
```

---

### 2.2 Tool 2: `image_generator` (多模型支持)

**功能**: 生成图像，支持多种生成模型（当前 SDXL，未来可扩展）

```python
@tool
def image_generator(
    prompt: str,
    seed: int,
    model_id: str = "sdxl",
    negative_prompt: str = "low quality, worst quality"
) -> str:
    """
    Generate an image using the specified model.

    Args:
        prompt: The text prompt describing the image to generate.
        seed: Random seed for reproducibility. Use the same seed for positive
              and negative images to ensure content consistency.
        model_id: The generation model to use. Currently supported:
                  - "sdxl": Stable Diffusion XL (default)
                  - Future: "playground", "flux", "midjourney-api", etc.
        negative_prompt: Negative prompt to avoid certain qualities.

    Returns:
        Path to the generated image file.
    """
    # 内部实现: 根据 model_id 调用不同的生成器
```

**多模型架构设计**:

```python
# scripts/tools/image_generator.py

class ImageGeneratorRegistry:
    """图像生成器注册表，支持多模型"""

    _generators = {}

    @classmethod
    def register(cls, model_id: str, generator_class):
        cls._generators[model_id] = generator_class

    @classmethod
    def get(cls, model_id: str):
        if model_id not in cls._generators:
            raise ValueError(f"Unknown model: {model_id}. Available: {list(cls._generators.keys())}")
        return cls._generators[model_id]

# 注册现有 SDXL
ImageGeneratorRegistry.register("sdxl", SDXLGenerator)

# 未来添加新模型只需:
# ImageGeneratorRegistry.register("playground", PlaygroundGenerator)
# ImageGeneratorRegistry.register("flux", FluxGenerator)
```

**输入/输出示例**:
```python
# 输入
prompt = "a beautiful sunset over the ocean, masterpiece"
seed = 42
model_id = "sdxl"  # 或未来的 "playground", "flux" 等

# 输出
"/tmp/generated_images/img_42_abc123.png"
```

---

### 2.3 Tool 3: `degradation_judge` (核心新增，使用 Gemini)

**功能**: 使用 Gemini VLM 判断图像对是否展示了有效的质量退化

```python
@tool
def degradation_judge(
    positive_image_path: str,
    negative_image_path: str,
    expected_dimension: str,
    expected_attribute: str = None
) -> str:
    """
    Judge whether an image pair shows valid visual quality degradation.

    This tool uses a Vision-Language Model (GPT-4o) to evaluate if:
    1. Both images depict the same/similar content (content consistency)
    2. The negative image shows actual quality degradation in the expected dimension

    Args:
        positive_image_path: Path to the high-quality (positive) image.
        negative_image_path: Path to the degraded (negative) image.
        expected_dimension: The degradation category that was applied
                           (e.g., 'technical_quality', 'anatomical_accuracy').
        expected_attribute: The specific attribute degraded (e.g., 'blur', 'hand_deformity').

    Returns:
        JSON string containing:
        - valid: bool - Whether this is a valid training pair
        - content_consistent: bool - Whether both images show similar content
        - degradation_detected: bool - Whether quality degradation is visible
        - detected_dimension: str - What type of degradation was actually observed
        - degradation_level: str - 'none', 'subtle', 'moderate', 'obvious'
        - notes: str - Additional observations
    """
```

**输入/输出示例**:
```python
# 输入
positive_image_path = "/tmp/images/positive_42.png"
negative_image_path = "/tmp/images/negative_42.png"
expected_dimension = "technical_quality"
expected_attribute = "blur"

# 输出 (JSON string)
{
    "valid": true,
    "content_consistent": true,
    "degradation_detected": true,
    "detected_dimension": "technical_quality",
    "detected_attribute": "blur",
    "degradation_level": "moderate",  # none / subtle / moderate / obvious
    "notes": "The negative image shows clear blur effect while maintaining the same sunset scene."
}
```

---

### 2.4 VLM 判别准则 (degradation_judge 的 System Prompt)

```yaml
# config/judge_prompt.yaml

system_prompt: |
  You are an expert image quality assessor. Your task is to evaluate whether
  a pair of AI-generated images constitutes a valid training sample for a
  visual quality preference model.

  ## Input
  - Image A (Left): Positive sample - expected to have HIGHER visual quality
  - Image B (Right): Negative sample - expected to have LOWER visual quality
  - Expected degradation dimension: {dimension}
  - Expected degradation attribute: {attribute}

  ## Evaluation Criteria

  ### 1. Content Consistency
  Do both images depict the same or very similar scene/subject?
  - YES: Same subject, similar composition, minor variations acceptable
  - NO: Completely different scenes or major subject changes

  ### 2. Quality Degradation Detection
  Does Image B (negative) show noticeably lower visual quality than Image A (positive)?

  For dimension "{dimension}", look for:
  {dimension_specific_guidelines}

  ### 3. Degradation Level Assessment
  How obvious is the quality difference?
  - none: No visible difference
  - subtle: Requires careful observation to notice
  - moderate: Clearly noticeable but not extreme
  - obvious: Immediately apparent, significant quality gap

  ## Output Format (JSON only)
  ```json
  {
    "valid": true/false,
    "content_consistent": true/false,
    "degradation_detected": true/false,
    "detected_dimension": "what you actually observed",
    "detected_attribute": "specific attribute observed",
    "degradation_level": "none/subtle/moderate/obvious",
    "notes": "brief explanation"
  }
  ```

  ## Decision Rules
  - valid = true IF: content_consistent AND degradation_detected
  - valid = false IF: NOT content_consistent OR NOT degradation_detected

  IMPORTANT:
  - We accept ALL degradation levels (subtle, moderate, obvious) as valid
  - Only reject if there is NO degradation or content is inconsistent
  - This is about VISUAL QUALITY, not text-image alignment

dimension_guidelines:
  technical_quality: |
    - blur: Check for loss of sharpness, soft edges, motion blur
    - exposure_issues: Look for overexposed (washed out) or underexposed (too dark) areas
    - low_contrast: Look for flat, hazy appearance
    - color_distortion: Check for unnatural color casts

  texture_detail: |
    - over_smoothing: Look for plastic/waxy appearance, lack of natural texture
    - texture_artifacts: Check for unnatural patterns, repetitive elements

  aesthetic_quality: |
    - poor_composition: Check if subject placement or balance is worse
    - poor_lighting: Look for unflattering or unnatural lighting

  structural_plausibility: |
    - object_deformation: Check for distorted shapes
    - perspective_error: Look for spatial inconsistencies
    - physical_implausibility: Check for impossible physics

  anatomical_accuracy: |
    - hand_deformity: Count fingers, check joint positions
    - facial_anomaly: Check facial symmetry and feature positions
    - body_proportion: Assess limb lengths, head-to-body ratio
```

---

## 3. Agent 编排设计

### 3.1 单 Agent 方案 (推荐)

使用单个 `CodeAgent` 编排整个流程，简单高效。

```python
from smolagents import CodeAgent, LiteLLMModel

# 初始化模型（用于 Agent 推理，非 VLM 判别）
model = LiteLLMModel(model_id="gpt-4o")

# 创建 Agent
agent = CodeAgent(
    tools=[prompt_degrader, image_generator, degradation_judge],
    model=model,
    max_steps=100  # 根据数据量调整
)

# 运行任务
result = agent.run("""
Generate quality degradation image pairs for the following prompts.

Positive prompts:
{prompts_list}

For each prompt:
1. Use prompt_degrader to create a degraded version
2. Use image_generator to generate both positive and negative images (same seed)
3. Use degradation_judge to validate the pair
4. Record results

Degradation settings:
- Subcategories to cover: {subcategories}
- Severity distribution: mild (20%), moderate (40%), severe (40%)

Output a summary of valid pairs generated.
""")
```

### 3.2 执行流程

```
┌─────────────────────────────────────────────────────────────────┐
│                     Agent 执行流程                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  FOR each positive_prompt in prompts:                           │
│      FOR each (subcategory, attribute, severity) combination:   │
│                                                                  │
│          ┌─────────────────────────────────────────────────┐    │
│          │ Step 1: prompt_degrader                          │    │
│          │   输入: positive_prompt, subcategory, severity   │    │
│          │   输出: negative_prompt, degradation_info        │    │
│          └─────────────────────────────────────────────────┘    │
│                              ↓                                   │
│          ┌─────────────────────────────────────────────────┐    │
│          │ Step 2: image_generator (x2)                     │    │
│          │   生成 positive_image (seed=N)                   │    │
│          │   生成 negative_image (seed=N, 同 seed)          │    │
│          └─────────────────────────────────────────────────┘    │
│                              ↓                                   │
│          ┌─────────────────────────────────────────────────┐    │
│          │ Step 3: degradation_judge                        │    │
│          │   输入: positive_image, negative_image           │    │
│          │   输出: valid, degradation_level, notes          │    │
│          └─────────────────────────────────────────────────┘    │
│                              ↓                                   │
│          ┌─────────────────────────────────────────────────┐    │
│          │ Step 4: 结果处理                                  │    │
│          │   IF valid:                                       │    │
│          │       保存图像对到数据集                           │    │
│          │       记录 degradation_level (用于后续分析)       │    │
│          │   ELSE:                                           │    │
│          │       记录失败原因 (用于模板优化)                  │    │
│          └─────────────────────────────────────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. 文件结构

```
data_generation/
├── config/
│   ├── llm_config.yaml              # 现有：LLM API 配置
│   ├── quality_dimensions.json      # 现有：退化维度定义
│   ├── prompt_templates_v3/         # 现有：退化 prompt 模板
│   └── judge_prompt.yaml            # 新增：VLM 判别 prompt 配置
│
├── scripts/
│   ├── llm_prompt_degradation.py    # 现有：保留，作为 Tool 的底层实现
│   ├── sdxl_generator.py            # 现有：保留，作为 Tool 的底层实现
│   │
│   ├── tools/                       # 新增：smolagents Tool 定义
│   │   ├── __init__.py
│   │   ├── prompt_degrader.py       # @tool 封装 LLMPromptDegradation
│   │   ├── image_generator.py       # @tool 封装 SDXLGenerator
│   │   └── degradation_judge.py     # @tool VLM 判别逻辑
│   │
│   └── agent_pipeline.py            # 新增：Agent 编排主脚本
│
├── outputs/                         # 输出目录
│   ├── images/
│   ├── dataset.json
│   └── validation_report.json       # 新增：验证统计报告
│
└── docs/
    ├── AGENT_ARCHITECTURE.md        # 已有
    └── SMOLAGENTS_ARCHITECTURE_PROPOSAL.md  # 本文档
```

---

## 5. 输出数据格式

### 5.1 dataset.json (扩展)

```json
{
  "metadata": {
    "version": "2.0",
    "generator": "smolagents_pipeline",
    "total_pairs": 1000,
    "valid_pairs": 950,
    "validation_rate": 0.95
  },
  "pairs": [
    {
      "id": "pair_0001",
      "positive": {
        "prompt": "a beautiful sunset over the ocean, masterpiece",
        "image_path": "images/positive_0001.png",
        "seed": 42
      },
      "negative": {
        "prompt": "a beautiful sunset over the ocean, blurry, out of focus",
        "image_path": "images/negative_0001.png",
        "seed": 42
      },
      "degradation": {
        "subcategory": "technical_quality",
        "attribute": "blur",
        "severity": "moderate"
      },
      "validation": {
        "valid": true,
        "content_consistent": true,
        "degradation_detected": true,
        "degradation_level": "moderate",
        "notes": "Clear blur effect observed"
      }
    }
  ]
}
```

### 5.2 validation_report.json (新增)

```json
{
  "summary": {
    "total_generated": 1000,
    "valid_pairs": 950,
    "invalid_pairs": 50,
    "validation_rate": 0.95
  },
  "by_degradation_level": {
    "obvious": 380,
    "moderate": 350,
    "subtle": 220,
    "none": 50
  },
  "by_subcategory": {
    "technical_quality": {"valid": 200, "invalid": 10},
    "texture_detail": {"valid": 180, "invalid": 8},
    "anatomical_accuracy": {"valid": 150, "invalid": 20}
  },
  "failure_reasons": {
    "content_inconsistent": 30,
    "no_degradation_detected": 15,
    "wrong_dimension": 5
  }
}
```

---

## 6. 依赖要求

```txt
# requirements_smolagents.txt

# smolagents 核心
smolagents>=1.0.0

# LLM 后端
openai>=1.0.0
litellm>=1.0.0  # 可选，用于多模型支持

# 现有依赖
torch>=2.0.0
diffusers>=0.25.0
Pillow>=10.0.0
pyyaml>=6.0
```

---

## 7. 使用方式

### 7.1 命令行

```bash
# 基本用法
python scripts/agent_pipeline.py \
    --source_prompts data/prompts.json \
    --output_dir outputs/dataset_v2 \
    --num_pairs_per_prompt 5

# 指定维度
python scripts/agent_pipeline.py \
    --source_prompts data/prompts.json \
    --subcategory technical_quality \
    --output_dir outputs/technical_only
```

### 7.2 Python API

```python
from scripts.agent_pipeline import DataGenerationPipeline

pipeline = DataGenerationPipeline(
    output_dir="outputs/my_dataset",
    llm_config_path="config/llm_config.yaml"
)

# 运行生成
results = pipeline.run(
    prompts=["prompt1", "prompt2", ...],
    subcategories=["technical_quality", "anatomical_accuracy"],
    num_negatives_per_positive=5
)

# 获取报告
print(results.validation_report)
```

---

## 8. 现有代码迁移对照表

| 现有功能 | 现有位置 | 迁移后位置 | 迁移方式 |
|---------|---------|-----------|---------|
| LLM Prompt 退化 | `llm_prompt_degradation.py` | `tools/prompt_degrader.py` | 封装为 @tool |
| System Prompt 缓存 | `LLMPromptDegradation.__init__` | 保留在原类中 | 无需改动 |
| 线程安全 Client | `LLMPromptDegradation._get_client` | 保留在原类中 | 无需改动 |
| SDXL 生成 | `sdxl_generator.py` | `tools/image_generator.py` | 封装为 @tool |
| 批量生成逻辑 | `generate_dataset.py` | `agent_pipeline.py` | Agent 编排替代 |
| 数据集保存 | `generate_dataset.py` | `agent_pipeline.py` | 保留逻辑 |

**核心原则**: 现有的 `LLMPromptDegradation` 和 `SDXLGenerator` 类**保持不变**，Tool 只是薄封装层。

---

## 9. 待确认事项

请确认以下设计决策：

### 9.1 判别准则
- [x] 内容一致性：正负样本描绘相同/相似场景
- [x] 退化有效性：负样本在预期维度上有可见的质量下降
- [x] **接受所有退化程度**（subtle/moderate/obvious 都算有效）
  - ✅ 已确认：多样化的退化程度有利于对比学习

### 9.2 失败处理
- [x] **无效 pair 只记录，不自动重试**
  - ✅ 已确认：记录失败原因供后续分析和模板优化

### 9.3 模型配置（已确认）
| 组件 | 模型 | 备注 |
|-----|------|------|
| prompt_degrader | GPT-4o | 现有配置保留 |
| degradation_judge | Gemini | 新增，配置格式统一 |
| 编排层 | 无 | Python 循环，无需额外模型 |

### 9.4 配置文件（已确认）
- [x] 维度定义：`config/quality_dimensions_v3.json`
- [x] 策略模板：`config/prompt_templates_v3/*.yaml`
- [x] 现有机制完全保留（不同维度加载不同策略 prompt）

### 9.5 是否使用 smolagents 库？

**两种选择**：

| 方案 | 说明 | 优点 | 缺点 |
|-----|------|------|------|
| **A: 使用 smolagents** | 用 `@tool` 装饰器定义工具 | 代码规范，未来可扩展为自主 Agent | 需要安装额外库 |
| **B: 纯 Python 类** | 直接用 Python 类封装 | 无额外依赖，更简单 | 无框架约束 |

**您的选择**: ✅ **A: 使用 smolagents**

---

## 10. 参考资料

- [smolagents 官方文档](https://huggingface.co/docs/smolagents/en/index)
- [HuggingFace Agents Course - Tools](https://huggingface.co/learn/agents-course/en/unit2/smolagents/tools)
- [HuggingFace Agents Course - Code Agents](https://huggingface.co/learn/agents-course/en/unit2/smolagents/code_agents)
- [DataCamp smolagents Tutorial](https://www.datacamp.com/tutorial/smolagents)
