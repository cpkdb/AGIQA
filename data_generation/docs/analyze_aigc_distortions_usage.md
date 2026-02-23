# AIGC图像失真分析工具使用说明

## 功能概述

该工具自动从SDXL生成的图像中发现和归类AIGC特有的视觉失真，通过GPT-4o进行开放式分析，最终归纳出新的质量维度分类体系。

## 核心流程

```
1. 下载数据集 (image_quality_train.json)
   ↓
2. 采样 prompts (50-100个)
   ↓
3. SDXL 批量生成图像
   ↓
4. GPT-4o 开放式失真分析
   ↓
5. 提取失真模式并聚类
   ↓
6. 归纳质量维度分类体系
   ↓
7. 生成 JSON + Markdown 报告
```

## 快速开始

### 基础用法

```bash
cd /root/ImageReward/data_generation

# 使用默认参数（80个样本，自动下载数据集）
python scripts/analyze_aigc_distortions.py

# 指定样本数量
python scripts/analyze_aigc_distortions.py --num-samples 100

# 使用已下载的数据集（跳过下载）
python scripts/analyze_aigc_distortions.py --dataset-path data/image_quality_train.json
```

### 恢复中断的运行

```bash
# 从中断处继续（跳过已完成的步骤）
python scripts/analyze_aigc_distortions.py --resume
```

### 自定义配置

```bash
python scripts/analyze_aigc_distortions.py \
  --num-samples 50 \
  --num-inference-steps 20 \
  --guidance-scale 7.0 \
  --width 512 \
  --height 512 \
  --device cuda \
  --request-interval 2.0
```

## 重要参数说明

### 数据集参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dataset-url` | HuggingFace URL | 数据集下载地址 |
| `--dataset-path` | `data/image_quality_train.json` | 本地数据集路径 |
| `--force-download` | False | 强制重新下载数据集 |
| `--num-samples` | 80 | 采样的prompt数量（建议50-100） |
| `--seed` | 42 | 随机种子（用于采样和生成） |

### SDXL生成参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--num-inference-steps` | 30 | SDXL推理步数（越大质量越高但越慢） |
| `--guidance-scale` | 7.5 | CFG scale |
| `--width` | 1024 | 图像宽度 |
| `--height` | 1024 | 图像高度 |
| `--device` | cuda | 计算设备 |

### GPT-4o参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--llm-config-path` | `config/llm_config.yaml` | LLM配置文件路径 |
| `--model` | 从配置读取 | GPT模型名称（可覆盖配置） |
| `--request-interval` | 自动计算 | GPT请求间隔（秒），0则从配置自动推算 |

### 输出路径参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--sampled-prompts-path` | `outputs/sampled_prompts.json` | 采样的prompts保存路径 |
| `--image-output-dir` | `outputs/sdxl_aigc_samples/` | 生成图像保存目录 |
| `--raw-analysis-path` | `outputs/raw_gpt4o_analysis.json` | GPT-4o原始分析结果 |
| `--patterns-path` | `outputs/distortion_patterns.json` | 提取的失真模式 |
| `--taxonomy-path` | `outputs/new_quality_dimensions.json` | 归纳的质量维度体系 |
| `--report-path` | `outputs/dimension_report.md` | Markdown可视化报告 |

## 输出文件说明

### 1. `sampled_prompts.json`
采样的prompt列表，包含id和prompt文本。

```json
[
  {"id": 42, "prompt": "a beautiful sunset..."},
  {"id": 137, "prompt": "portrait of a woman..."}
]
```

### 2. `sdxl_aigc_samples/metadata.json`
生成图像的元数据，包含prompt、路径、生成参数。

```json
[
  {
    "id": 42,
    "prompt": "a beautiful sunset...",
    "image_path": "outputs/sdxl_aigc_samples/sample_00042.png",
    "generation_info": {
      "model": "stable-diffusion-xl-base-1.0",
      "seed": 1276,
      "steps": 30,
      "cfg_scale": 7.5
    }
  }
]
```

### 3. `raw_gpt4o_analysis.json`
GPT-4o的原始分析结果，每张图一条记录。

```json
[
  {
    "id": 42,
    "prompt": "...",
    "image_path": "...",
    "status": "success",
    "response_text": "{...}",
    "parsed": {
      "summary": "Image shows anatomical issues...",
      "distortions": [
        {
          "label": "hand deformity",
          "description": "six fingers on left hand",
          "severity": "high",
          "confidence": 0.95,
          "evidence": "clearly visible extra digit"
        }
      ]
    }
  }
]
```

### 4. `distortion_patterns.json`
聚合的失真模式，按出现频率排序。

```json
{
  "generated_at": "2026-01-05T12:00:00Z",
  "total_images": 80,
  "total_findings": 247,
  "patterns": [
    {
      "pattern": "hand deformity",
      "canonical_label": "Hand Deformity",
      "count": 23,
      "average_confidence": 0.88,
      "severity_histogram": {
        "high": 15,
        "medium": 6,
        "critical": 2
      },
      "examples": [...]
    }
  ]
}
```

### 5. `new_quality_dimensions.json`
归纳的质量维度分类体系。

```json
{
  "generated_at": "2026-01-05T12:00:00Z",
  "note": "Induced taxonomy from open-ended GPT-4o distortion analysis",
  "dimensions": [
    {
      "name": "anatomy_and_form",
      "description": "Body/structure integrity...",
      "keywords": ["anatom", "limb", "finger", "hand", "face"],
      "directions": [
        {
          "name": "Hand Deformity",
          "count": 23,
          "average_confidence": 0.88,
          "severity_histogram": {...},
          "examples": [...]
        }
      ]
    }
  ]
}
```

### 6. `dimension_report.md`
Markdown格式的可视化报告，包含维度统计和示例。

## 注意事项

### 1. API配置
确保`config/llm_config.yaml`中配置了有效的GPT-4o API密钥：

```yaml
llm:
  provider: "openai"
  model: "gpt-4o"
  api_key: "sk-xxx..."
  api_base: "https://api.chatanywhere.org"  # 可选
  rate_limit_rpm: 60
```

### 2. 硬件需求
- **GPU**: 最低12GB显存（SDXL），推荐16GB+
- **磁盘**: 每100张图像约需5-10GB空间
- **网络**: 需要访问HuggingFace和OpenAI API

### 3. 运行时间估算
- 80个样本，30步推理：
  - SDXL生成：约20-40分钟
  - GPT-4o分析：约8-16分钟（取决于API速率）
  - 总计：约30-60分钟

### 4. 故障恢复
- 所有中间结果自动保存
- 使用`--resume`可从中断处继续
- 失败的GPT分析会自动重试3次
- 图像缺失会自动重新生成

### 5. 成本估算
- GPT-4o Vision API：每张图像约$0.01-0.02
- 80张图像总成本：约$0.80-1.60

## 故障排除

### 问题：`requests module not found`
```bash
pip install requests
```

### 问题：`PIL module not found`
```bash
pip install pillow
```

### 问题：GPT API 401 错误
检查`config/llm_config.yaml`中的API密钥是否有效。

### 问题：CUDA out of memory
降低图像分辨率或推理步数：
```bash
python scripts/analyze_aigc_distortions.py \
  --width 768 --height 768 \
  --num-inference-steps 20
```

### 问题：GPT请求速率限制
增加请求间隔：
```bash
python scripts/analyze_aigc_distortions.py --request-interval 2.0
```

## 高级用法

### 仅运行特定步骤

```bash
# 1. 仅下载和采样
python scripts/analyze_aigc_distortions.py --num-samples 10
# 然后手动中止，检查 outputs/sampled_prompts.json

# 2. 使用已有图像运行GPT分析
# 修改代码跳过生成步骤，或手动编辑metadata.json
```

### 批量处理

```bash
# 处理多个数据集
for dataset in dataset1.json dataset2.json; do
  python scripts/analyze_aigc_distortions.py \
    --dataset-path "$dataset" \
    --image-output-dir "outputs/${dataset%.json}" \
    --num-samples 50
done
```

### 与现有维度对比

生成的`new_quality_dimensions.json`可以与`config/quality_dimensions.json`对比：

```python
import json

with open("outputs/new_quality_dimensions.json") as f:
    new_dims = json.load(f)

with open("config/quality_dimensions.json") as f:
    old_dims = json.load(f)

# 对比分析代码...
```

## 后续研究方向

1. **维度优化**：根据发现的新维度更新`quality_dimensions.json`
2. **Prompt退化策略**：为新维度设计退化模板（`config/prompt_templates/`）
3. **数据集生成**：使用新维度生成对比数据集进行BT模型训练
4. **人工标注**：对发现的失真进行人工验证和细化

## 联系与反馈

如有问题或建议，请参考主项目文档或提交issue。
