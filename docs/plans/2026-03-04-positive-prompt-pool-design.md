# Positive Prompt Pool Design For Quality Degradation Agent Workflow

## Goal

为当前 AIGC 质量退化数据生成框架补充一层可独立管理的正样本 Prompt 供给系统，使正样本不再是“原始 prompt 列表”，而是“面向退化维度可调度的 Prompt Pool”。

本设计目标：

- 提升正样本对特定退化维度的可承载性，减少无效组合和空转重试
- 在保持类别广度的同时，让正样本与质量退化维度强绑定
- 复用现有代码中的 tagging / filtering / routing / closed-loop judge 机制，避免平行重造
- 为后续百万级数据生成提供可统计、可补洞、可反馈回流的上游 Prompt 基础设施

本设计不直接修改 Stage 2-6 的退化、生成、判别逻辑，重点补齐正样本入口层。

---

## Current Framework Summary

当前仓库已具备完整的闭环数据生成主流程：

1. Stage 1: 数据筛选 / SemanticRouter
2. Stage 2: 策略选择 / StrategyOptimizer
3. Stage 3: Prompt 退化 / LLMPromptDegradation
4. Stage 4: 图像生成 / SDXL, Flux, 其他生成器
5. Stage 5: VLM 检验 / degradation_judge
6. Stage 6: 反馈优化 / KnowledgeBase

对应实现入口：

- `data_generation/scripts/pipeline.py`
- `data_generation/docs/AGENT_ARCHITECTURE.md`

现有正样本相关能力已经存在，但分散在多个模块中：

- `data_generation/scripts/tag_positive_prompts.py`
  - 对正样本做 LLM / 关键词语义标注
  - 产出 `semantic_tags`
  - 可统计各维度覆盖率
- `data_generation/scripts/semantic_router.py`
  - 在线生成 `signature`
  - 根据 `dimension_requirements.yaml` 做维度兼容性过滤
- `data_generation/scripts/prompt_filter.py`
  - 在 tag 过滤之外，补充维度级特殊规则
  - 如 `text_error` 的引号文本约束、部分人体维度对背影/动物的排斥
- `data_generation/config/semantic_tag_requirements.json`
  - 定义现有 tag 体系和大部分维度要求
- `data_generation/config/dimension_requirements.yaml`
  - 定义 Stage 1 使用的 `required_tags` / `preferred_tags`

结论：当前系统缺的不是“正样本筛选能力”，而是一个统一的正样本 Prompt Pool 层，用来把这些能力组织成上游供给系统。

---

## Core Problem

当前 `pipeline.py` 的运行方式，本质上仍是：

- 先拿到正样本 prompt
- 再随机选择退化维度
- 然后才由 Stage 1 判断该 `(prompt, dimension)` 是否兼容

这种顺序的主要问题：

- 会产生大量先抽样、后跳过的无效组合
- `required_tags` 之外的 `preferred_tags` 没有真正参与采样排序
- `prompt_filter.py` 中更严格的特殊规则没有并入主 pipeline
- VLM 在 Stage 5 返回的 `positive_content_mismatch` / `positive_incompatible` 还没有回流到正样本供给层

因此，现有闭环在“负样本修正”上很强，但在“正样本前置路由”上仍偏被动。

---

## Design Principle

正样本方案遵循以下原则：

1. 正样本必须是“维度感知”的，而不是全局通用的
2. 类别广度与维度适配性必须同时优化
3. 现有 AIGC prompt 库作为主来源，LLM 只负责定向补洞
4. 实现层复用现有 tag 体系，不建立平行标签命名
5. 正样本供给必须进入闭环反馈，而不是一次性静态预处理

---

## Proposed Architecture

在现有 6 阶段 workflow 前新增：

- Stage 0: Positive Prompt Pool Builder

新增后的整体流程：

1. Stage 0: Positive Prompt Pool Builder
2. Stage 1: Semantic Routing / Prompt Selection
3. Stage 2: StrategyOptimizer
4. Stage 3: LLMPromptDegradation
5. Stage 4: Image Generation
6. Stage 5: VLM Judge
7. Stage 6: KnowledgeBase Feedback

其中 Stage 0 是离线构建 + 周期性增量更新；Stage 1 是运行时路由与选样。

---

## Stage 0: Positive Prompt Pool Builder

### 0.1 Responsibilities

Stage 0 负责把多来源原始 prompt 构建成一个可供 agent 调用的正样本池。

输出不再是简单的 `List[str]`，而是带元数据的 Prompt Registry。

### 0.2 Input Sources

正样本来源采用双轨策略：

- Existing libraries
  - 现有 AIGC prompt 库
  - 历史自有 prompt 数据
  - 可公开获取的真实 T2I prompt 数据集
- LLM synthesis
  - 仅对覆盖缺口做定向补池
  - 不做大规模无约束自由生成

推荐比例：

- 70%: 现有库抽取与清洗
- 30%: LLM 按缺口补齐

这个比例是初始建议，后续可由覆盖率统计调整。

### 0.3 Stage 0 Sub-Steps

#### Step A: Ingestion

导入原始 prompt，并执行：

- 去重
- 规范化大小写与标点
- 过滤明显无意义或极短 prompt
- 去除高噪声质量增强词

重点清理词包括但不限于：

- `masterpiece`
- `best quality`
- `high quality`
- `ultra detailed`
- `8k`
- `award winning`

原则：去除会干扰“质量退化判断”的无效风格加成，但保留必要的语义内容与画风类别。

#### Step B: Tagging

复用现有 `tag_positive_prompts.py` 作为主工具，为每条 prompt 生成：

- `semantic_tags`

具体策略：

- 默认使用 LLM 打标
- 关键词匹配作为 fallback / 补充
- 输出统一存回 Prompt Registry

复用现有 tag 体系，避免新建平行标签：

- `has_person`
- `has_face`
- `has_hand`
- `has_full_body`
- `has_animal`
- `has_multiple_objects`
- `has_countable_objects`
- `has_indoor_scene`
- `has_background`
- `has_text`
- `has_reflective_surface`
- `has_logo_or_symbol`
- `has_structured_object`
- 以及现有 schema 中已定义的其它标签

#### Step C: Prompt Signature Enrichment

仅依靠 `semantic_tags` 粒度不够，需要增加更强的结构信号 `prompt_signature`。

建议新增或固化以下 signature 字段：

- `contains_hands_visible`
- `contains_full_body`
- `contains_closeup_face`
- `contains_explicit_text_literal`
- `contains_reflection_surface`
- `contains_count_constraint`
- `contains_multiple_distinct_objects`
- `contains_indoor_layout`
- `contains_action_pose`
- `contains_human_subject`
- `contains_animal_subject`

这些 signature 不替代现有 `tags`，而是用于更细粒度的维度匹配与优先级排序。

#### Step D: Coverage Audit

复用 `tag_positive_prompts.py` 中现有覆盖率统计思路，对每个维度统计：

- 可用 prompt 数
- 总覆盖率
- required 覆盖率
- preferred 覆盖率
- 特殊规则过滤后真实可用数

此处不应只看 tag 层覆盖，还要叠加特殊规则过滤后的有效覆盖。

#### Step E: LLM Backfill

对覆盖不足的维度，触发定向合成。

原则：

- 只补缺口，不重建全库
- 按“缺什么补什么”的方式定向生成
- 先补 required 维度，再补 preferred 稀缺维度

例如：

- `text_error` 缺口：生成带引号文本、带文字载体的 prompt
- `hand_malformation` 缺口：生成明确出现手部动作的 prompt
- `reflection_error` 缺口：生成含镜子、水面、玻璃反射的 prompt
- `count_error` 缺口：生成带明确数量约束的 prompt

---

## Prompt Taxonomy

### Macro Taxonomy (for coverage balancing)

宏观分类用于保证类别广度，不直接作为硬过滤条件：

- Indoor
- Urban
- Nature
- People & Activities
- Objects & Artifacts
- Food
- Events

作用：

- 保证样本分布不过度偏向人物或静物
- 为 LLM 补池提供场景上下文
- 为后续采样提供类别均衡能力

### Micro Semantic Constraints (for dimension compatibility)

微观约束用于保证退化维度的可触发性：

- 人体相关
- 动物相关
- 多物体/数量相关
- 文本/标识相关
- 反射/光学相关
- 室内结构相关
- 姿态/动作相关

在实现层，这些微观约束应直接映射到现有 `semantic_tags + prompt_signature`，而不是另起一套业务标签。

---

## Prompt Registry Schema

建议 Stage 0 产出的统一数据结构如下：

```json
{
  "id": "prompt_000001",
  "prompt": "a woman holding a coffee cup in a cafe, close-up portrait",
  "source": "library",
  "source_dataset": "internal_prompt_corpus_v1",
  "scene_taxonomy": ["people_activities", "indoor"],
  "semantic_tags": ["has_person", "has_face", "has_hand", "has_indoor_scene"],
  "prompt_signature": {
    "contains_hands_visible": true,
    "contains_closeup_face": true,
    "contains_full_body": false,
    "contains_explicit_text_literal": false,
    "contains_reflection_surface": false,
    "contains_count_constraint": false,
    "contains_multiple_distinct_objects": true,
    "contains_indoor_layout": true,
    "contains_action_pose": true
  },
  "compatible_dimensions": [
    "hand_malformation",
    "face_asymmetry",
    "expression_mismatch",
    "plastic_waxy_texture"
  ],
  "preferred_dimensions": [
    "hand_malformation",
    "face_asymmetry"
  ],
  "style_profile": {
    "dominant_style": "realistic",
    "style_anchor_hint": "realistic style, photorealistic"
  },
  "exclusion_reasons": {},
  "quality_notes": {
    "cleaned": true,
    "removed_quality_boosters": true
  }
}
```

设计要求：

- `compatible_dimensions` 是运行时快速路由索引
- `preferred_dimensions` 用于优先采样
- `exclusion_reasons` 记录特殊规则淘汰原因，便于审计
- `style_profile` 供 `positive_incompatible` 失败后的 style anchor 调整使用

---

## Dimension-Aware Positive Prompt Policy

正样本并非所有维度通用，必须按维度定义“required / preferred / special rules / fallback”。

### Policy Layers

对每个退化维度，定义四层策略：

1. `required`
   - 必须满足，否则不可进入该维度
2. `preferred`
   - 满足则优先采样，但不是硬约束
3. `special_rules`
   - 额外正则 / 逻辑规则
4. `fallback`
   - 池子不足时的降级策略

### Example Mapping

#### Human Anatomy

- `hand_malformation`
  - required: `has_hand`
  - preferred: `has_person`, `contains_hands_visible`
  - special rules: 优先有动作词，如 holding / pointing / gesture
- `face_asymmetry`
  - required: `has_face`
  - preferred: `contains_closeup_face`
  - special rules: 排除背影、动物、弱面部描述
- `body_proportion_error`
  - required: `has_person`
  - preferred: `has_full_body`, `contains_full_body`
- `impossible_pose`
  - required: `has_person`
  - preferred: `has_full_body`, `contains_action_pose`

#### Objects / Counting / Spatial

- `count_error`
  - required: 无硬性标签，但应优先 `has_countable_objects`
  - preferred: `contains_count_constraint`
  - fallback: 不使用无数量约束的 prompt 做高优先级样本
- `scale_inconsistency`
  - required: 无硬性标签
  - preferred: `has_multiple_objects`, `contains_multiple_distinct_objects`
- `penetration_overlap`
  - required: 无硬性标签
  - preferred: `has_multiple_objects`, 结构化物体
- `scene_layout_error`
  - required: 无硬性标签
  - preferred: `has_indoor_scene`, `contains_indoor_layout`

#### Optical / Text

- `reflection_error`
  - required: `has_reflective_surface`
  - preferred: `contains_reflection_surface`
- `text_error`
  - required: `has_text`
  - preferred: `contains_explicit_text_literal`
  - special rules: 必须带引号内容和文字载体
- `logo_symbol_error`
  - required: `has_logo_or_symbol` 或 `has_text`
  - preferred: 明确 logo / sign / brand 场景

#### Technical / Aesthetic

技术和美学维度总体上对正样本要求较弱，可使用更大范围通用正样本池，但仍建议做以下偏好路由：

- `plastic_waxy_texture`
  - preferred: `has_person`, `has_face`, `contains_closeup_face`
- `lighting_imbalance`
  - preferred: 有明确光源或室内/舞台/窗边场景
- `awkward_framing`
  - preferred: 有人物、主体明显的构图型 prompt

---

## Integration With Existing Agent Workflow

### New Workflow

推荐工作流调整为：

1. Stage 0 构建 Prompt Pool
2. 根据目标维度，从 Prompt Pool 获取候选正样本
3. Stage 1 再做运行时校验与排序
4. Stage 2 选择退化模板
5. Stage 3 改写负样本 prompt
6. Stage 4 生成正负图像
7. Stage 5 判别
8. Stage 6 记录反馈并回流

### Changes To Stage 1

Stage 1 不再只是“过滤器”，而应升级为“路由 + 选样器”：

- 输入：目标维度 + 候选 Prompt Pool
- 输出：按优先级排序的兼容正样本候选

运行逻辑建议：

1. 先从 `compatible_dimensions` 反查候选集合
2. 再按 `preferred_dimensions` 排序
3. 再执行特殊规则二次过滤
4. 如果池子不足，按 fallback 规则降级

这比当前“随机抽维度后再 skip”更高效，也更符合百万级生成需求。

### Changes To Stage 3

Stage 3 已支持透传 `prompt_signature` 给 `LLMPromptDegradation`。

建议继续保留，并逐步增强其用途：

- 当前用途：人物类 prompt 的 style lock
- 后续用途：
  - 让 LLM 明确知道哪些结构必须保留
  - 在重试时避免破坏可触发退化的关键语义元素

示例：

- 对 `text_error`，要求保留原始引号文字载体
- 对 `count_error`，要求保留原始数量约束
- 对 `reflection_error`，要求保留原始反射介质

### Changes To Stage 5 / Stage 6

将 VLM 失败类型转化为 Prompt Pool 反馈信号：

- `positive_content_mismatch`
  - 记录为 `prompt × dimension` 不兼容
  - 直接降低该 prompt 在该维度下的可用性
- `positive_incompatible`
  - 记录 style profile 与推荐 style anchor
- 同维度持续失败
  - 触发 Stage 0 定向补池，而不是只做模板修补

也就是说，KnowledgeBase 不只记录“退化模板是否有效”，还应逐步记录“正样本是否适合这个维度”。

---

## LLM Backfill Prompting Strategy

LLM 合成遵循“定向补池”原则。

### Synthesis Template

建议统一使用约束式模板，而非开放式自由生成：

```text
Write a text-to-image prompt for a {macro_scene}.

Requirements:
- The prompt must clearly include: {required_elements}
- The prompt should preferably include: {preferred_elements}
- Keep it natural and realistic
- Keep it concise but descriptive (15-40 words; up to 50 words for complex spatial scenes)
- Do not use quality-boosting keywords such as masterpiece, best quality, 8k, ultra detailed
- Preserve only one clear semantic focus

Output only the prompt text.
```

### Generation Rules

- 优先补 required 缺口
- 单次生成目标是“补齐某类结构”，不是追求文风多样性
- 同一轮补池应限制在单一目标维度或同类维度群

例如：

- 手部补池：优先生成人物手部交互 prompt
- 文本补池：优先生成牌匾、招牌、封面、包装、界面文案
- 反射补池：优先生成镜子、水面、玻璃橱窗

---

## Data Flow

### Offline Flow

1. 导入现有 prompt 源
2. 清洗与归一化
3. LLM / 关键词打标
4. 生成 `prompt_signature`
5. 覆盖率统计
6. LLM 定向补池
7. 构建 `Prompt Registry`
8. 预计算 `dimension -> candidate pool` 索引

### Runtime Flow

1. Agent 决定目标维度与严重度
2. 从 `dimension-compatible pool` 取候选正样本
3. Stage 1 进行最终检查与排序
4. 执行 Stage 2-5 的闭环生成
5. 将失败类型回写到 Stage 6
6. 定期回流到 Stage 0 更新 Prompt Pool

---

## Rollout Plan

### Phase 1: Minimal Integration

目标：不改核心闭环，只新增离线 Prompt Pool 产物。

- 复用 `tag_positive_prompts.py`
- 统一产出带 tag 的正样本 JSON
- 生成维度覆盖率报告
- 用于人工和脚本筛选正样本源

收益：

- 快速看到哪些维度缺正样本
- 为 LLM 补池提供明确目标

### Phase 2: Recommended Integration

目标：把 Prompt Pool 真正接到 pipeline 前端。

- 新增 Stage 0 构建脚本
- 在运行前加载 Prompt Registry
- 将 Stage 1 改为“按维度取样 + 校验 + 排序”
- 并入 `prompt_filter.py` 的特殊规则

收益：

- 显著减少无效 `(prompt, dimension)` 组合
- 提高成功率与生成效率

### Phase 3: Full Closed-Loop Integration

目标：让 Prompt Pool 进入反馈闭环。

- 记录 `prompt × dimension × model` 成功率
- 将 `positive_content_mismatch` 回写到 blacklist
- 对低覆盖、高失败维度自动触发补池

收益：

- 正样本供给成为自进化系统
- 更适合百万级持续生成任务

---

## Non-Goals

本设计当前不覆盖：

- 修改现有 Judge Prompt 的评分逻辑
- 重新定义质量退化 taxonomy
- 重写 Stage 2 的模板选择器
- 改动图像生成器本身

这些部分继续复用现有实现。

---

## Risks And Notes

### Risk 1: Tag 粒度不够

若仅靠现有 `semantic_tags`，部分维度的可承载性仍会判断失真。

应对：

- 增加 `prompt_signature`
- 将特殊规则显式纳入筛选逻辑

### Risk 2: 两套规则源不一致

当前存在：

- `semantic_tag_requirements.json`
- `dimension_requirements.yaml`
- `prompt_filter.py` 硬编码特殊规则

这三处规则可能漂移。

应对：

- 后续需要统一“单一维度约束源”
- 至少保证生成时的规则来源可追踪

### Risk 3: LLM 补池带来风格漂移

如果让 LLM 自由生成，容易产生模板化或不自然 prompt。

应对：

- LLM 只补缺口
- 严格约束长度、结构与禁用词
- 用现有库 prompt 作为主分布

---

## Recommended Next Step

在本设计基础上，下一步应输出两份可执行产物：

1. `Prompt Pool JSON Schema`
   - 明确 Stage 0 输出数据结构
2. `Dimension → Tag / Signature Mapping Table`
   - 明确每个退化维度的 required / preferred / special rules / fallback

完成这两项后，即可进入具体实现规划，并最小化与现有 pipeline 的集成风险。
