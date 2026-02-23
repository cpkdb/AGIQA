# AIGC 图像质量数据生成 — 6 阶段闭环 Workflow 架构

## 系统架构总览

```
┌────────────────────────────────────────────────────────────────────────────┐
│                      Orchestrator  (pipeline.py)                           │
│                                                                            │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │ Stage 1   │→│ Stage 2   │→│ Stage 3   │→│ Stage 4   │→│ Stage 5   │   │
│  │ 数据筛选  │  │ 策略选择  │  │ Prompt   │  │ 图像生成  │  │ VLM 检验 │   │
│  │SemanticR. │  │ Strategy  │  │ 退化     │  │SDXL/Flux │  │ Judge    │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘  └────┬─────┘   │
│       ↑                                                        │          │
│       │              ┌──────────────────┐                      │          │
│       └──────────────│    Stage 6       │←─────────────────────┘          │
│                      │  KnowledgeBase   │                                 │
│                      └──────────────────┘                                 │
└────────────────────────────────────────────────────────────────────────────┘
```

### 反馈回路

| 回路 | 路径 | 作用 |
|------|------|------|
| 局部重试 | Stage 5 → Stage 3/4 | 单对失败的即时修复（style_drift / content_drift / insufficient_effect） |
| Circuit Breaker | Stage 5 → Stage 6 → Stage 1 | 连续失败 ≥10 次 → 暂停维度 |
| 模型兼容 | Stage 5 → Stage 6 → Stage 4 | 成功率 < 30% (≥10 次) → 跳过 model×dimension |
| 策略进化 | Stage 5 → Stage 6 → Stage 2 | 成功率 < 60% (≥5 次) → 触发 StrategyOptimizer 进化 |

### 设计原则

1. **向后兼容**: 新功能通过可选参数 (`enable_routing`, `enable_feedback`) 启用，不传则行为等同原有脚本。
2. **单阶段可独立运行**: 每个 Stage 对应独立模块，可单独测试。
3. **渐进式集成**: 在 `pipeline.py` 基础上增量增强，不重写。

---

## Stage 1: 数据筛选 — SemanticRouter

在生成前主动判断 `(prompt, dimension)` 兼容性，跳过不可能成功的组合。

| 属性 | 值 |
|------|---|
| **类名** | `SemanticRouter` |
| **文件** | `scripts/semantic_router.py` |
| **配置** | `config/semantic_tag_requirements.json` + `config/dimension_requirements.yaml` |
| **方法** | 纯关键词匹配，无 LLM 调用，零延迟 |

### 核心 API

```python
router = SemanticRouter()

# 分析 prompt 语义标签
sig = router.analyze("a woman holding a cup, portrait, close-up")
# → {"tags": ["has_person", "has_face", "has_hand"], "has_person": True, ...}

# 判断兼容性
compatible, reason = router.is_compatible(sig, "hand_malformation")
# → (True, "ok")

compatible, reason = router.is_compatible(sig, "animal_anatomy_error")
# → (False, "missing_required:has_animal")
```

### 兼容性规则 (`dimension_requirements.yaml`)

17 个维度配置了标签要求，其余维度默认兼容：

| 维度 | 必需标签 (OR) | 优先标签 |
|------|-------------|---------|
| `hand_malformation` | `has_hand` | `has_person` |
| `face_asymmetry` | `has_face` | — |
| `body_proportion_error` | `has_person` | `has_full_body` |
| `animal_anatomy_error` | `has_animal` | — |
| `text_error` | `has_text` | — |
| `reflection_error` | `has_reflective_surface` | — |
| ... | | |

### Pipeline 集成

```python
pipeline = DataGenerationPipeline(
    output_dir="...",
    enable_routing=True,       # 启用 Stage 1
)
```

启用后，`run()` 在每轮生成前调用 `router.analyze()` + `router.is_compatible()`，不兼容直接跳过。

### 独立运行

```bash
python scripts/semantic_router.py --input data/prompts.json --report
python scripts/semantic_router.py --input data/prompts.json --dimension hand_malformation
```

### 辅助模块: prompt_filter.py

从 `demo_v3_dimension_paired.py` 提取的批量过滤模块，面向预标签化数据：

```bash
python scripts/prompt_filter.py --test hand_malformation
python scripts/prompt_filter.py --stats
```

含硬编码特殊规则（text_error 需引号+指示词、face_asymmetry 排斥动物/背影等）。

---

## Stage 2: 策略选择 — StrategyOptimizer

为每个 `(dimension, severity, model)` 选择最优策略模板，对低效模板进行约束微调。

| 属性 | 值 |
|------|---|
| **类名** | `StrategyOptimizer` |
| **文件** | `scripts/llm_prompt_degradation.py:613` |
| **约束配置** | `config/template_constraints.yaml` |

### 三级模板查找

```
Level 1: KnowledgeBase 高成功率变体（预留，当前直接走 Level 2）
     ↓
Level 2: YAML 基础模板
         cache_key = f"{subcategory}_{dimension}_{severity}"
         → prompt_templates_v3/*.yaml 中的属性级模板
     ↓
Level 3: 动态回退 (简单模板)
```

### 防漂移约束 (`template_constraints.yaml`)

7 个维度配置了进化约束：

```yaml
blur:
  evolution_enabled: true
  must_keep: ["blur", "out of focus"]
  must_avoid: ["sharp", "high resolution", "crystal clear"]
```

`check_guardrails()` 验证模板是否满足 `must_keep` / `must_avoid` 约束。

### 核心 API

```python
optimizer = StrategyOptimizer(degrader, knowledge_base=kb)
template = optimizer.select_template("blur", "moderate", "sdxl")
passed = optimizer.check_guardrails("blur", template)
```

---

## Stage 3: Prompt 退化 — LLMPromptDegradation

使用 GPT-4o 将正样本 prompt 改写为带质量退化的负样本 prompt。

| 属性 | 值 |
|------|---|
| **类名** | `LLMPromptDegradation` |
| **文件** | `scripts/llm_prompt_degradation.py:38` |
| **Tool 封装** | `scripts/tools/prompt_degrader.py` |
| **LLM** | GPT-4o (OpenAI API) |
| **模板** | `config/prompt_templates_v3/*.yaml` — 50 维度 × 3 严重度 ≈ 150 个策略模板 |

### 核心方法签名

```python
def generate_negative_prompt(
    self,
    positive_prompt: str,
    subcategory: str,
    attribute: Optional[str] = None,
    severity: str = "moderate",
    # 重试模式（可选）
    failed_negative_prompt: Optional[str] = None,
    feedback: Optional[str] = None,
    judge_scores: Optional[Dict] = None,      # Stage 5 多维评分
    # 上下文参数（可选）
    model_id: str = "sdxl",
    prompt_signature: Optional[Dict] = None,   # Stage 1 语义标签
) -> Tuple[str, Dict]:
```

### 重试模式增强

- **judge_scores 透传**: 重试时 user_prompt 包含上次 VLM Judge 的 4 维评分，LLM 可针对性修正。
- **style_lock**: 首次生成时，若 `prompt_signature` 含 `has_person/has_face/has_hand` 标签，添加风格锁定指令。
- **style_anchor 保留**: 重试 prompt 中要求 LLM 保留开头的画风锚定前缀。

### 模板系统

```
config/prompt_templates_v3/
├── technical_quality.yaml      # blur, overexposure, underexposure, ...
├── aesthetic_quality.yaml      # flat_lighting, color_clash, ...
├── anatomical_accuracy.yaml    # hand_malformation, face_asymmetry, ...
├── structural_plausibility.yaml
├── texture_detail.yaml
└── ...
```

缓存机制: `_build_system_prompt_cache()` 在初始化时一次性加载所有模板到内存。

---

## Stage 4: 图像生成

| 属性 | 值 |
|------|---|
| **Tool** | `scripts/tools/image_generator.py` |
| **支持模型** | SDXL, Flux, Flux-Schnell |
| **关键策略** | 正负样本使用相同 seed 保证内容一致性 |

Pipeline 中的使用:
- 正样本: 每个 prompt 生成一次，跨 pair 复用
- 负样本: 每次 attempt 单独生成

Style Anchoring (`pipeline.py:185-196`):

```python
STYLE_ANCHORS = {
    "realistic": "realistic style, photorealistic, lifelike,",
    "illustration": "illustration style, digital art, stylized,",
    "painting": "painting style, artistic,",
}
```

`positive_incompatible` 失败时，根据 VLM 推荐的风格给正样本 prompt 添加锚定前缀并重新生成。

---

## Stage 5: VLM 检验 — degradation_judge

使用 VLM (Gemini) 判别图像对是否展示了有效退化。

| 属性 | 值 |
|------|---|
| **Tool** | `scripts/tools/degradation_judge.py` |
| **VLM** | Gemini (OpenAI 兼容 API) |
| **输入** | 左右拼接图像 (左=正样本, 右=负样本) |
| **输出** | 多维度评分 + 失败类型 + 诊断 |

### 评分 Schema

```json
{
  "valid": true,
  "scores": {
    "content_preservation": 0.92,
    "style_consistency": 0.88,
    "degradation_intensity": 0.75,
    "dimension_accuracy": 0.85
  },
  "style_type": "realistic",
  "failure": null,
  "recommended_style": null,
  "notes": "模糊退化清晰可见，内容保持一致"
}
```

### 4 种失败类型

| failure_type | 含义 | Pipeline 处理 |
|-------------|------|--------------|
| `positive_incompatible` | 正样本风格不适合该维度 | 添加 style anchor → 重新生成正样本 |
| `style_drift` | 负样本画风偏离 | `_apply_style_anchor()` → 直接添加前缀，不调 LLM |
| `content_drift` | 主体/内容变化过大 | LLM 重写 + 反馈 |
| `insufficient_effect` | 退化效果不明显 | LLM 重写 + 反馈 |

### 覆盖的退化维度

50 个维度分为 5 大类，每个维度在 Judge Prompt 中有独立的检查指引 (`DIMENSION_GUIDELINES` dict)：

- **Technical Quality** (7): blur, overexposure, underexposure, low_contrast, color_cast, desaturation, plastic_waxy_texture
- **Aesthetic Quality** (9): awkward_positioning, awkward_framing, unbalanced_layout, cluttered_scene, lack_of_depth, flat_lighting, lighting_imbalance, color_clash, dull_palette
- **Semantic: Anatomy** (7): hand_malformation, face_asymmetry, expression_mismatch, body_proportion_error, extra_limbs, impossible_pose, animal_anatomy_error
- **Semantic: Object** (6): object_shape_error, object_fusion, missing_parts, extra_objects, count_error, illogical_colors
- **Semantic: Spatial/Physical/Scene/Text** (5+): perspective_error, scale_inconsistency, floating_objects, shadow_mismatch, text_error, ...

---

## Stage 6: 反馈优化 — KnowledgeBase

跨运行的持久化知识库，记录统计数据并驱动系统决策。

| 属性 | 值 |
|------|---|
| **类名** | `KnowledgeBase` |
| **文件** | `scripts/knowledge_base.py` |
| **持久化** | `{kb_dir}/knowledge_base.json` |

### 数据结构

```python
dimension_stats    # {dim: {attempts, successes, total_scores, consecutive_failures, failure_types}}
strategy_stats     # {"dim|sev|model": {template_id, attempts, successes}}
compat_matrix      # {"model|dim": {attempts, successes}}
paused_dimensions  # Set[str] — Circuit Breaker 暂停的维度
```

### 可配置阈值

| 参数 | 默认值 | 作用 |
|------|--------|------|
| `evolution_threshold` | 0.6 | 成功率低于此值触发策略进化 |
| `incompatible_threshold` | 0.3 | 成功率低于此值标记 model×dim 不兼容 |
| `min_attempts_evolution` | 5 | 触发进化的最少尝试次数 |
| `min_attempts_compat` | 10 | 判断兼容性的最少尝试次数 |
| `circuit_breaker_limit` | 10 | 连续失败次数上限 |

### 核心 API

```python
kb = KnowledgeBase(path="outputs/knowledge_base/")

# 每次 VLM 判定后
kb.report_outcome(dimension="blur", severity="moderate", model_id="sdxl",
                  template_id="blur_moderate", success=True,
                  scores={"content_preservation": 0.9, ...})

# 系统决策
kb.is_dimension_paused("blur")                    # Circuit Breaker
kb.is_model_compatible("flux-schnell", "text_error")  # 模型兼容性
kb.should_trigger_evolution("blur", "moderate", "sdxl")  # 策略进化

# 报告
kb.get_dimension_health()   # 各维度成功率 + 平均评分
kb.get_compat_report()      # model×dim 兼容性矩阵
kb.get_strategy_report()    # 策略效果 + 是否需要进化
```

### 独立运行

```bash
python scripts/knowledge_base.py --report outputs/knowledge_base/
```

---

## Pipeline 调度核心 (`pipeline.py`)

### 初始化

```python
class DataGenerationPipeline:
    def __init__(
        self,
        output_dir: str,
        quality_dimensions_path: str = None,
        model_id: str = "sdxl",
        model_path: str = None,
        max_retries: int = 2,
        enable_routing: bool = False,      # Stage 1
        enable_feedback: bool = False,     # Stage 6
        knowledge_base_dir: str = None,    # Stage 6 存储路径
    )
```

### `run()` 主循环流程

```
for prompt in prompts:
    sig = router.analyze(prompt)                          # Stage 1 (可选)

    for pair_idx in range(num_pairs_per_prompt):
        dimension, severity = random_select()

        # 检查点跳过
        if pair_key in _completed_pairs: continue

        # Stage 1: 兼容性
        if router and not is_compatible(sig, dim): continue

        # Stage 6: Circuit Breaker + 模型兼容
        if kb.is_dimension_paused(dim): continue
        if not kb.is_model_compatible(model, dim): continue

        # Stage 2~5: 闭环生成
        result = generate_pair_with_retry(prompt, dim, sev, seed, sig)

        # Stage 6: 记录结果
        kb.report_outcome(dim, sev, model, template, success, scores)

        # 定期保存
        if counter % 10 == 0: save_results(); kb.save()
```

### 闭环重试 (`generate_pair_with_retry`)

```
生成正样本 (一次)
for attempt in range(1 + max_retries):
    if retry && style_drift:   直接 style_anchor (不调 LLM)
    elif retry:                LLM 重写 (带 feedback + judge_scores)
    else:                      LLM 首次退化 (带 prompt_signature)

    生成负样本 → VLM Judge

    if valid:   break
    if positive_incompatible:  添加 style anchor → 重新生成正样本 → continue
    else:       记录 failed_prompt, feedback → 下次重试
```

### 断点续跑

基于 `full_log.json` 实现：
- `_load_checkpoint()`: 读取已有日志，提取 `success=True` 的 pair 到 `_completed_pairs` set。
- `_make_pair_key()`: `f"{prompt[:100]}|{dimension}|{severity}"`
- 恢复 `pair_counter`, `full_log`, `dataset` 状态。

### CLI

```bash
# 基础（向后兼容，无新功能）
python scripts/pipeline.py \
    --source_prompts data/prompts_tagged_sdxl_v2.json \
    --model_id flux-schnell --max_prompts 5

# 启用 Stage 1 + Stage 6
python scripts/pipeline.py \
    --source_prompts data/prompts_tagged_sdxl_v2.json \
    --model_id flux-schnell \
    --enable_routing \
    --enable_feedback \
    --knowledge_base_dir outputs/knowledge_base/
```

---

## 配置文件总览

| 文件 | 用途 | 使用方 |
|------|------|-------|
| `config/llm_config.yaml` | GPT-4o API 配置 | Stage 3 |
| `config/judge_config.yaml` | VLM (Gemini) API 配置 | Stage 5 |
| `config/quality_dimensions_v3.json` | 50 维度定义 (含 L1/L2/L3 可控性) | Stage 2, Pipeline |
| `config/prompt_templates_v3/*.yaml` | 150 个策略模板 | Stage 3 |
| `config/semantic_tag_requirements.json` | 14 种语义标签定义 + 关键词 | Stage 1 |
| `config/dimension_requirements.yaml` | 17 个维度的标签兼容规则 | Stage 1 |
| `config/template_constraints.yaml` | 7 个维度的防漂移约束 | Stage 2 |

---

## 输出结构

```
{output_dir}/
├── images/
│   └── pair_NNNN/
│       ├── positive_seed42.png
│       ├── attempt_0_negative_seed42.png
│       └── attempt_1_negative_seed42.png     # (重试时)
├── dataset.json          # 仅成功 pair
├── full_log.json         # 所有 pair (含全部 attempts，支持断点续跑)
├── validation_report.json # 统计报告
└── knowledge_base/       # (Stage 6 启用时)
    └── knowledge_base.json
```

---

## 文件清单

| 文件 | 角色 | Stage |
|------|------|-------|
| `scripts/pipeline.py` | Orchestrator, 闭环调度 | 全局 |
| `scripts/semantic_router.py` | 关键词标签化 + 维度兼容筛选 | 1 |
| `scripts/prompt_filter.py` | 批量过滤（面向预标签数据） | 1 |
| `scripts/llm_prompt_degradation.py` | LLM 退化引擎 + StrategyOptimizer | 2, 3 |
| `scripts/tools/prompt_degrader.py` | smolagents Tool 封装 | 3 |
| `scripts/tools/image_generator.py` | 图像生成 Tool | 4 |
| `scripts/tools/degradation_judge.py` | VLM Judge Tool | 5 |
| `scripts/knowledge_base.py` | 持久化知识库 | 6 |

---

## 版本历史

| 版本 | 日期 | 变更 |
|------|------|------|
| v1.0 | 2024-11 | Agent 1 (LLMPromptDegradation) 完整实现 |
| v1.1 | 2024-11 | Agent 2 (VLM Judge) 雏形设计 |
| v2.0 | 2025-01 | 闭环 pipeline + 50 维度 + VLM Judge 上线 |
| v3.0 | 2025-02 | 6 阶段 Workflow: SemanticRouter, StrategyOptimizer, KnowledgeBase, 多维评分, 断点续跑 |
