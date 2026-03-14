# Prompt Pool Agent 整理与收口实现计划

> **执行说明：** 按任务顺序逐步实现，并在每一步完成后进行最小验证；任何外部 pool 删除动作都必须在你确认后再执行。

**目标：** 整理当前正样本 prompt pool / subpool 体系，明确当前主链路实际使用的 pool，识别冗余与历史产物，并为 32 个退化维度制定最终版 pool routing 方案，特别补齐 `scale_inconsistency`、`penetration_overlap`、`object_structure_error`、`material_mismatch`、`floating_objects` 等高风险维度的专门子池路线。

**架构：** 先做 inventory 和 active-manifest 层，不直接删数据；再在现有 cleaned / anatomy screened / turbo clipsafe 基础上收口出一份统一 routing 方案；最后只对确实需要的维度新增轻量规则召回或 LLM 语义筛选子池。整个过程优先复用现有 builder 和 index 体系，不重造平行系统。

**技术栈：** Python 3.10、JSON / JSONL、YAML、现有 pool builder 脚本、`unittest`

---

## 当前现状结论（作为实施前提）

### 当前主链路实际使用

- 通用模型（`flux-schnell` / `qwen-image-lightning` / 其他默认模型）当前实际使用：
  - source pool：
    - `merged_working_pool_cleaned_v1.jsonl`
  - dimension index：
    - `anatomy_screened_dimension_subpools_cleaned_v2/index.json`
    - 若不存在则回退 `cleaned_v1` / `base cleaned`

- `sd3.5-large-turbo` 当前实际使用：
  - source pool：
    - `merged_working_pool_sd35_turbo_clipsafe_v1.jsonl`
  - dimension index：
    - `sd35_turbo_dimension_subpools_clipsafe_v2/index.json`
    - 若不存在则回退 `clipsafe_v1`

### 当前结构问题

1. 当前主链路使用的 index 仍然保留旧 35 维体系痕迹：
   - 仍含 `face_asymmetry`
   - 仍含 `object_shape_error`
   - 不含当前 active taxonomy 中的 `object_structure_error`
   - `material_mismatch` 也没有真正接进当前主 index

2. 结果上会出现：
   - `object_structure_error`
   - `material_mismatch`
   - `scale_inconsistency`
   - `floating_objects`
   - `penetration_overlap`
   在主链路中退回到“全局 pool + 运行时 tag 过滤”甚至“几乎全池”的情况

3. 当前外部目录中存在多份历史 / 半成品产物：
   - `anatomy_screened_dimension_subpools`
   - `anatomy_screened_dimension_subpools_cleaned_v1`
   - `anatomy_screened_dimension_subpools_cleaned_v2`
   - `dimension_subpools`
   - `dimension_subpools_cleaned_v1`
   - `sd35_turbo_dimension_subpools_clipsafe_v1`
   - `sd35_turbo_dimension_subpools_clipsafe_v2`
   - `targeted_dimension_subpools_cleaned_v1`
   - `sd35_turbo_targeted_dimension_subpools_clipsafe_v1`
   - `sd35_turbo_clipsafe_v2_tmp`

4. 当前已明显不适合直接删除，但可以先标记为“主链路在用 / fallback / 历史待确认”三类。

### 当前关键维度 pool 规模（已知）

- 通用 active index：
  - `hand_malformation = 8864`
  - `expression_mismatch = 2414`
  - `body_proportion_error = 2176`
  - `extra_limbs = 2176`
  - `animal_anatomy_error = 2527`
  - `text_error = 1791`
  - `logo_symbol_error = 7748`

- turbo active index：
  - `hand_malformation = 6560`
  - `expression_mismatch = 1673`
  - `body_proportion_error = 1513`
  - `extra_limbs = 1784`
  - `animal_anatomy_error = 2085`
  - `text_error = 1767`
  - `logo_symbol_error = 6662`

- 但以下维度当前并没有真正接入专门子池：
  - `object_structure_error`
  - `material_mismatch`
  - `scale_inconsistency`
  - `floating_objects`
  - `penetration_overlap`

---

## 最终版 pool routing 目标

### A. 全池或 cleaned 全池可用

适合继续使用通用 cleaned pool，不需要专门筛池：

- `blur`
- `overexposure`
- `underexposure`
- `low_contrast`
- `color_cast`
- `desaturation`
- `plastic_waxy_texture`
- `awkward_positioning`
- `awkward_framing`
- `unbalanced_layout`
- `cluttered_scene`
- `lighting_imbalance`
- `color_clash`
- `dull_palette`
- `extra_objects`
- `count_error`
- `illogical_colors`
- `context_mismatch`
- `time_inconsistency`
- `scene_layout_error`

### B. 保留现有专门子池

继续沿用现有 anatomy / text / logo 路线，但统一进新的 active manifest：

- `hand_malformation`
- `expression_mismatch`
- `animal_anatomy_error`
- `text_error`
- `logo_symbol_error`

### C. 需要新增或重建专门子池

这几个维度应进入 Prompt Pool Agent 的下一轮重点收口：

0. `body_proportion_error`
- 现有 anatomy screened 池仍然过宽
- 想要稳定退化时，正样本更应偏向单人、真实、全身、头身和腿长可读的人物
- 铠甲、厚重服饰、强透视、重度遮挡都应排除
- 推荐路线：
  - 与 `extra_limbs` 优先复用同一个 `human_full_body_realistic` 父池
  - 先规则召回：`has_person`
  - 再做 LLM 语义筛选

0b. `extra_limbs`
- 与其单独维护一个宽泛的人体池，不如优先复用 `body_proportion_error` 的真实全身人物父池
- 再按肢体可读性做维度级二次筛选

1. `object_structure_error`
- 不再依赖旧 `object_shape_error`
- 需要“单个显著结构化非生物物体”候选池
- 推荐路线：
  - 规则召回：`has_structured_object`
  - 再做 LLM 语义筛选
  - 并与 `material_mismatch` 优先复用同一个 `structured_object_primary` 父池

2. `material_mismatch`
- 需要“材质观感明显可读”的结构化物体
- 推荐路线：
  - 先从 `object_structure_error` 候选池派生
  - 再做轻量 LLM 语义筛选

3. `scale_inconsistency`
- prompt 需要有可比较对象，不应再走全池
- 推荐路线：
  - 规则召回：`has_multiple_objects`
  - 再用 LLM 筛选“存在明确参照物、局部尺度错容易生成”的 prompt

4. `penetration_overlap`
- 需要两个实体/物体边界可见，不能继续几乎全池
- 推荐路线：
  - 规则召回：`has_multiple_objects`，并优先 `has_structured_object`
  - 再用 LLM 筛选“边界和接触关系明确”的 prompt

5. `floating_objects`
- 需要有明确主体，但主体不只限于非生物物体，人或动物也可以
- 推荐路线：
  - 规则召回：`has_structured_object` / `has_countable_objects` / `has_person` / `has_animal`
  - 当前优先做轻量规则池，不强制 LLM 语义筛选
  - 后续如效果仍不稳定，再考虑是否单独加轻量语义筛查

---

## 100 万对规模下的 prompt 数量考虑

### 基本公式

如果总目标是 100 万对图像，32 个维度平均分配，则每个维度平均约：

- `1,000,000 / 32 ≈ 31,250 pairs`

如果每个正样本 prompt 生成 `k` 个负样本 pair，则每个维度理论最少需要的正样本数约为：

- `31,250 / k`

例如：
- `k = 2` 时，约需 `15,625` 个正样本 prompt
- `k = 4` 时，约需 `7,813` 个正样本 prompt

### 工程解释

这不是硬性“必须有这么多 prompt 才能生产”，因为：
- 同一 prompt 可以跨模型使用
- 同一 prompt 可以多 seed 复用
- 实际生产不会平均分配到每个维度

但这意味着：
- 高约束维度若长期只有 `1k ~ 3k` prompt，做百万对时多样性一定不够
- 后续必须：
  - 降低这些维度的采样权重
  - 或继续补池 / 语义筛选扩池

---

## 任务拆分

### 任务 1：产出当前 active pool manifest

**涉及文件：**
- 新建：`data_generation/scripts/prompt_pool_agent.py`
- 新建：`data_generation/scripts/prompt_pool_agent_tools.py`
- 新建：`tests/test_prompt_pool_agent.py`

**步骤 1：先写失败测试**

补充测试，明确：
- 能解析当前主脚本真实会用的 source pool 和 dimension index
- 能输出“主链路在用 / fallback / 历史待确认”的 pool 清单

**步骤 2：运行测试并确认失败**

执行：

```bash
python -m unittest tests.test_prompt_pool_agent -v
```

预期：FAIL，因为新 agent 尚未实现。

**步骤 3：编写最小实现**

实现：
- `resolve_active_prompt_pools(...)`
- `scan_prompt_pool_inventory(...)`
- `classify_pool_artifacts(...)`
- `write_prompt_pool_manifest(...)`

**步骤 4：再次运行测试并确认通过**

执行：

```bash
python -m unittest tests.test_prompt_pool_agent -v
```

预期：PASS。

### 任务 2：为 32 维输出最终版 routing 方案

**涉及文件：**
- 新建：`data_generation/config/prompt_pool_routing_v1.json`
- 修改：`data_generation/scripts/prompt_pool_agent.py`
- 测试：`tests/test_prompt_pool_agent.py`

**步骤 1：先写失败测试**

补充测试，明确：
- 每个 active 维度都能映射到一种 pool routing 策略
- 高风险维度被标记为：
  - `existing_special_pool`
  - `rule_recall_then_llm_screen`
  - `global_cleaned_pool`

**步骤 2：运行测试并确认失败**

执行：

```bash
python -m unittest tests.test_prompt_pool_agent -v
```

预期：FAIL，因为 routing 配置尚未建立。

**步骤 3：编写最小实现**

输出：
- 32 维 routing 配置
- 每个维度当前 pool 来源与未来目标 pool 路线
- 每个维度当前 pool 数量与风险标记

**步骤 4：再次运行测试并确认通过**

执行：

```bash
python -m unittest tests.test_prompt_pool_agent -v
```

预期：PASS。

### 任务 3：补高风险维度的子池构建计划（不立即删除旧池）

**涉及文件：**
- 新建：`data_generation/config/prompt_pool_screening_plan_v1.json`
- 修改：`data_generation/scripts/prompt_pool_agent.py`
- 测试：`tests/test_prompt_pool_agent.py`

**步骤 1：先写失败测试**

补充测试，明确：
- 以下维度被标记为需要新建专门子池：
  - `object_structure_error`
  - `material_mismatch`
  - `scale_inconsistency`
  - `penetration_overlap`
  - `floating_objects`

**步骤 2：运行测试并确认失败**

执行：

```bash
python -m unittest tests.test_prompt_pool_agent -v
```

预期：FAIL，因为 screening 计划文件还不存在。

**步骤 3：编写最小实现**

输出：
- 每个维度的规则召回底池
- 是否需要 LLM 语义筛选
- 目标 pool 名称
- 是否需要 turbo 专用 clipsafe 派生

**步骤 4：再次运行测试并确认通过**

执行：

```bash
python -m unittest tests.test_prompt_pool_agent -v
```

预期：PASS。

### 任务 4：清理计划仅输出，不直接删除

**涉及文件：**
- 新建：`data_generation/config/prompt_pool_cleanup_candidates_v1.json`
- 修改：`data_generation/scripts/prompt_pool_agent.py`
- 测试：`tests/test_prompt_pool_agent.py`

**步骤 1：先写失败测试**

补充测试，明确：
- agent 只输出 cleanup candidates
- 不直接执行任何外部目录删除

候选应至少包含：
- `sd35_turbo_clipsafe_v2_tmp`
- `targeted_dimension_subpools_cleaned_v1`
- `sd35_turbo_targeted_dimension_subpools_clipsafe_v1`

其中 `anatomy_screened_dimension_subpools_v2` 不应直接归为 stale candidate。
它更适合作为 `historical_candidate` / lineage 产物保留，直到新版 cleaned routing 完全收口后再决定是否清理。

**步骤 2：运行测试并确认失败**

执行：

```bash
python -m unittest tests.test_prompt_pool_agent -v
```

预期：FAIL，因为 cleanup candidate 产物尚未实现。

**步骤 3：编写最小实现**

输出：
- `active`
- `fallback`
- `stale_candidate`
- `delete_requires_confirmation`

四类清单。

**步骤 4：再次运行测试并确认通过**

执行：

```bash
python -m unittest tests.test_prompt_pool_agent -v
```

预期：PASS。

---

## 范围边界

本轮先做到：
- 整理当前 pool 体系
- 输出 active manifest
- 输出 32 维最终版 routing 方案
- 输出需要 LLM 语义筛选的新维度计划
- 输出待确认的 cleanup candidates

本轮**不直接做**：
- 外部 pool 删除
- 大规模重建全部子池
- 改写主运行脚本去消费新 manifest
- 自动决定 100 万对的最终采样权重
