# Positive Prompt Pool Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 在现有质量退化数据生成框架中增量引入 Stage 0 正样本 Prompt Pool，使正样本按维度可路由、可统计、可补池，并逐步接入现有 pipeline。

**Architecture:** 先实现离线 Prompt Pool 构建链路，复用现有 tag/filter 逻辑输出统一 Prompt Registry；再补充维度映射和特殊规则收敛；最后把运行时从“随机抽维度后过滤”升级为“按维度取正样本后闭环生成”。整个过程保持 Stage 2-6 主闭环不重写，优先做前置增量集成。

**Tech Stack:** Python, json, yaml, pytest

---

### Task 0: 估算正样本需求并确定 Prompt 来源策略

**Files:**
- Create: `data_generation/scripts/prompt_pool_sizing.py`
- Create: `tests/test_prompt_pool_sizing.py`
- Optional Create: `data_generation/data/prompt_source_catalog.json`

**Step 1: Write the failing test**

为新的 `prompt_pool_sizing.py` 写测试，验证：

- 输入目标生成规模（如总 pair 数、每个正样本复用次数）
- 输入维度清单与维度采样权重
- 输出每个维度所需的“最低可用正样本数”估算
- 输出总体建议正样本池规模

同时验证可对候选 prompt 源做最小覆盖评估：

- 输入多个来源的样本统计
- 输出推荐导入优先级

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_prompt_pool_sizing.py -q`

Expected:

- FAIL，因为 `prompt_pool_sizing.py` 还不存在

**Step 3: Write minimal implementation**

新增 `prompt_pool_sizing.py`，先实现纯规则估算版本：

- 根据总 pair 数、目标维度数、单正样本复用上限，估算每个维度所需正样本基数
- 支持区分 `required-heavy` 维度和通用维度
- 支持对候选 prompt 库做“已知标签覆盖率”或“预估覆盖率”的简单评分

此阶段不接真实外部数据集下载，只做本地估算与来源选择逻辑。

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_prompt_pool_sizing.py -q`

Expected:

- PASS

**Step 5: Commit**

```bash
git add tests/test_prompt_pool_sizing.py data_generation/scripts/prompt_pool_sizing.py
git commit -m "feat: add prompt pool sizing estimator"
```

### Task 1: 定义 Prompt Pool Schema 与最小离线构建器

**Files:**
- Create: `data_generation/scripts/prompt_pool_builder.py`
- Create: `tests/test_prompt_pool_builder.py`
- Optional Modify: `data_generation/schema/dataset_schema.json`

**Step 1: Write the failing test**

为新的 `prompt_pool_builder.py` 写最小行为测试，验证：

- 输入原始 prompt 列表
- 输出包含 `prompt`, `source`, `semantic_tags`, `prompt_signature`, `compatible_dimensions`, `preferred_dimensions`
- 默认可调用现有 `SemanticRouter.analyze()` 构造基础 signature

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_prompt_pool_builder.py -q`

Expected:

- FAIL，因为 `prompt_pool_builder.py` 还不存在

**Step 3: Write minimal implementation**

新增 `prompt_pool_builder.py`，先实现最小可用版本：

- 接收原始 prompt 列表
- 调用 `SemanticRouter.analyze()` 生成基础 tag/signature
- 基于维度规则生成初版 `compatible_dimensions` / `preferred_dimensions`
- 输出 JSON 结构

此阶段不引入 LLM 补池，只做本地离线构建。

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_prompt_pool_builder.py -q`

Expected:

- PASS

**Step 5: Commit**

```bash
git add tests/test_prompt_pool_builder.py data_generation/scripts/prompt_pool_builder.py
git commit -m "feat: add minimal prompt pool builder"
```

### Task 2: 收敛维度映射规则为统一 Prompt Pool Policy

**Files:**
- Create: `data_generation/config/prompt_pool_policy.yaml`
- Modify: `data_generation/scripts/prompt_pool_builder.py`
- Create: `tests/test_prompt_pool_policy.py`

**Step 1: Write the failing test**

为 `prompt_pool_policy.yaml` 和构建器增加测试，验证：

- 每个维度可以读取 `required`, `preferred`, `special_rules`, `fallback`
- 构建器按该配置生成 `compatible_dimensions` 和 `preferred_dimensions`

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_prompt_pool_policy.py -q`

Expected:

- FAIL，因为策略配置和解析逻辑还不存在

**Step 3: Write minimal implementation**

新增 `prompt_pool_policy.yaml`，将现有规则统一收敛：

- 兼容 `dimension_requirements.yaml` 中的 `required_tags` / `preferred_tags`
- 把 `prompt_filter.py` 中的关键特殊规则映射为可配置策略
- 在 `prompt_pool_builder.py` 中读取该策略并生成更精确的维度映射

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_prompt_pool_policy.py tests/test_prompt_pool_builder.py -q`

Expected:

- PASS

**Step 5: Commit**

```bash
git add data_generation/config/prompt_pool_policy.yaml data_generation/scripts/prompt_pool_builder.py tests/test_prompt_pool_policy.py tests/test_prompt_pool_builder.py
git commit -m "feat: add prompt pool policy mapping"
```

### Task 3: 增强 Prompt Signature 粒度

**Files:**
- Modify: `data_generation/scripts/prompt_pool_builder.py`
- Optional Modify: `data_generation/scripts/semantic_router.py`
- Create: `tests/test_prompt_signatures.py`

**Step 1: Write the failing test**

增加测试，验证以下增强字段可被正确推断：

- `contains_hands_visible`
- `contains_full_body`
- `contains_closeup_face`
- `contains_explicit_text_literal`
- `contains_reflection_surface`
- `contains_count_constraint`

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_prompt_signatures.py -q`

Expected:

- FAIL，因为增强 signature 尚未实现

**Step 3: Write minimal implementation**

在 `prompt_pool_builder.py` 中增加规则推断逻辑，必要时仅抽取 `semantic_router.py` 的公共辅助逻辑，不扩散修改范围。

此阶段只做 deterministic 规则，不引入额外 LLM 推断。

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_prompt_signatures.py tests/test_prompt_pool_builder.py -q`

Expected:

- PASS

**Step 5: Commit**

```bash
git add data_generation/scripts/prompt_pool_builder.py data_generation/scripts/semantic_router.py tests/test_prompt_signatures.py tests/test_prompt_pool_builder.py
git commit -m "feat: enrich prompt signatures"
```

### Task 4: 增加覆盖率审计与缺口报告

**Files:**
- Modify: `data_generation/scripts/prompt_pool_builder.py`
- Create: `tests/test_prompt_pool_coverage.py`

**Step 1: Write the failing test**

增加测试，验证构建器可输出：

- 每个维度的可用 prompt 数
- `required` 命中数
- `preferred` 命中数
- 特殊规则过滤后的有效数

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_prompt_pool_coverage.py -q`

Expected:

- FAIL，因为 coverage 汇总尚未完整实现

**Step 3: Write minimal implementation**

在构建器中新增 coverage 报告生成逻辑，尽量复用现有 `tag_positive_prompts.py` / `prompt_filter.py` 的统计思路。

输出格式应能直接用于后续 LLM 补池决策。

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_prompt_pool_coverage.py tests/test_prompt_pool_builder.py -q`

Expected:

- PASS

**Step 5: Commit**

```bash
git add data_generation/scripts/prompt_pool_builder.py tests/test_prompt_pool_coverage.py
git commit -m "feat: add prompt pool coverage report"
```

### Task 5: 接入 Pipeline 的运行时正样本选择（只做最小集成）

**Files:**
- Modify: `data_generation/scripts/pipeline.py`
- Create: `tests/test_pipeline_prompt_pool_routing.py`

**Step 1: Write the failing test**

增加测试，验证 `pipeline.py` 在启用 Prompt Pool 时：

- 优先从 `dimension-compatible pool` 选择正样本
- 不再仅依赖“随机维度后 Stage 1 skip”
- 仍保持未启用 Prompt Pool 时的向后兼容行为

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_pipeline_prompt_pool_routing.py -q`

Expected:

- FAIL，因为 pipeline 还不支持 Prompt Pool 输入

**Step 3: Write minimal implementation**

在 `pipeline.py` 增加可选 Prompt Pool 接口：

- 支持加载 Prompt Registry
- 在目标维度已知时先取候选 prompt
- 保留原有 `enable_routing` 行为作为运行时二次校验

此阶段不重写 Stage 2-6，仅改正样本选取入口。

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_pipeline_prompt_pool_routing.py tests/test_prompt_pool_builder.py -q`

Expected:

- PASS

**Step 5: Commit**

```bash
git add data_generation/scripts/pipeline.py tests/test_pipeline_prompt_pool_routing.py
git commit -m "feat: route pipeline through prompt pool"
```

### Task 6: 将 Stage 5/6 反馈接回 Prompt Pool 元数据

**Files:**
- Modify: `data_generation/scripts/pipeline.py`
- Optional Modify: `data_generation/scripts/knowledge_base.py`
- Create: `tests/test_prompt_pool_feedback.py`

**Step 1: Write the failing test**

增加测试，验证：

- `positive_content_mismatch` 可记录为 prompt-dimension 不兼容
- `positive_incompatible` 可记录推荐 style anchor
- 反馈结构可被 Prompt Pool 消费

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_prompt_pool_feedback.py -q`

Expected:

- FAIL，因为现有反馈没有回写 Prompt Pool 元数据

**Step 3: Write minimal implementation**

在 pipeline 侧先最小落地：

- 为成功/失败记录追加 prompt 级反馈字段
- 以 JSON 形式保存 `prompt_feedback`，先不做自动补池

后续再决定是否将其吸收到 `KnowledgeBase`。

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_prompt_pool_feedback.py tests/test_pipeline_prompt_pool_routing.py -q`

Expected:

- PASS

**Step 5: Commit**

```bash
git add data_generation/scripts/pipeline.py data_generation/scripts/knowledge_base.py tests/test_prompt_pool_feedback.py
git commit -m "feat: record prompt pool feedback signals"
```

### Task 7: 端到端静态验证

**Files:**
- Verify only

**Step 1: Run targeted tests**

Run:

```bash
pytest tests/test_prompt_pool_builder.py \
  tests/test_prompt_pool_policy.py \
  tests/test_prompt_signatures.py \
  tests/test_prompt_pool_coverage.py \
  tests/test_pipeline_prompt_pool_routing.py \
  tests/test_prompt_pool_feedback.py -q
```

Expected:

- PASS

**Step 2: Run syntax validation**

Run: `python -m compileall data_generation/scripts`

Expected:

- PASS

**Step 3: Review diffs**

确认：

- Stage 2-6 核心逻辑未被重写
- Prompt Pool 集成保持为前置增量
- 未引入不必要的外部依赖

**Step 4: Commit**

```bash
git add docs/plans/2026-03-04-positive-prompt-pool-implementation-plan.md
git commit -m "docs: add prompt pool implementation plan"
```
