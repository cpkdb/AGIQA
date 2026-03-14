# Taxonomy & Allocation Agent 实现计划

> **执行说明：** 按任务顺序逐步实现，并在每一步完成后进行最小验证。

**目标：** 实现 `Taxonomy & Allocation Agent` 的第一版。该 Agent 是一个离线、确定性的分析/规划组件，用于读取当前生效的 taxonomy、当前运行时实际使用的 prompt 资源，以及历史 run 产物，然后为后续 orchestration 输出结构化分析 artifact。

**架构：** 第一版是一个配置驱动的分析型 Agent，而不是在线策略优化器。它复用当前 shell 脚本的运行时资源优先级、生效中的 taxonomy JSON，以及历史 `dataset.json` / `full_log.json` / `validation_report.json`，输出事实层摘要和一个可人工编辑的 allocation plan 模板。

**技术栈：** Python 3.10、标准库（`json`、`argparse`、`pathlib`、`datetime`）、现有数据生成脚本/配置、`unittest`

---

### 任务 1：先补资源解析与 pool 覆盖率的失败测试

**涉及文件：**
- 新建：`tests/test_taxonomy_allocation_tools.py`
- 修改：无
- 测试：`tests/test_taxonomy_allocation_tools.py`

**步骤 1：先写失败测试**

补充测试，明确以下行为：
- `flux-schnell`、`qwen-image-lightning`、`sd3.5-large-turbo` 的运行时资源优先级
- 从 source prompt JSONL 和 `index.json` 中提取 pool 覆盖率

测试应断言：
- 通用模型优先使用 cleaned source prompts 和 anatomy screened cleaned v2 index
- turbo 优先使用自己的 clipsafe source prompts 和 clipsafe v2 index
- 覆盖率报告中包含 `total_source_prompts`、`dimension_counts` 和 low-pool warnings

**步骤 2：运行测试并确认失败**

执行：

```bash
python -m unittest tests.test_taxonomy_allocation_tools -v
```

预期：FAIL，因为新模块/函数尚不存在。

**步骤 3：编写最小实现**

新建 tools 模块，并实现：
- 运行时资源候选常量
- `resolve_runtime_resources(...)`
- `inspect_pool_coverage(...)`

**步骤 4：再次运行测试并确认通过**

执行：

```bash
python -m unittest tests.test_taxonomy_allocation_tools -v
```

预期：PASS。

### 任务 2：补历史 run 聚合与 artifact 输出的失败测试

**涉及文件：**
- 修改：`tests/test_taxonomy_allocation_tools.py`
- 新建：`tests/test_taxonomy_allocation_agent.py`
- 测试：`tests/test_taxonomy_allocation_agent.py`

**步骤 1：先写失败测试**

补充测试，明确以下行为：
- 从 `full_log.json` 聚合历史 run 成功率
- 基于 `dataset.json.metadata.created_at/completed_at` 与 `validation_report.json.summary` 估算粗粒度速度
- Agent 入口端到端写出 artifact

测试应断言：
- `model_dimension_stats.json` 采用 `model -> dimension -> severity` 的嵌套结构
- `speed_summary.json` 包含 `avg_pair_seconds`、`avg_success_pair_seconds`、`avg_image_seconds`
- Agent 会写出：
  - `coverage_summary.json`
  - `model_dimension_stats.json`
  - `speed_summary.json`
  - `allocation_plan.template.json`
  - `manifest.json`

**步骤 2：运行测试并确认失败**

执行：

```bash
python -m unittest tests.test_taxonomy_allocation_agent -v
```

预期：FAIL，因为聚合逻辑和 CLI/entrypoint 仍未实现。

**步骤 3：编写最小实现**

实现：
- `aggregate_run_statistics(...)`
- `summarize_generation_speed(...)`
- `load_taxonomy_summary(...)`
- `build_allocation_plan_template(...)`
- `run_taxonomy_allocation_agent(...)`
- `data_generation/scripts/taxonomy_allocation_agent.py` 中的 CLI 包装

**步骤 4：再次运行测试并确认通过**

执行：

```bash
python -m unittest tests.test_taxonomy_allocation_agent -v
```

预期：PASS。

### 任务 3：校验与当前代码库假设的一致性

**涉及文件：**
- 修改：`tests/test_run_scripts.py`
- 修改：`tests/test_dimension_taxonomy_refresh.py`（仅在共享 fixture/import 必要时；否则不动）
- 测试：`tests/test_run_scripts.py`

**步骤 1：先写失败测试**

添加一个轻量集成测试，确保新的 agent 模块在当前代码布局下存在且可导入，并且不会破坏现有 run script 的预期。

**步骤 2：运行测试并确认失败**

执行：

```bash
python -m unittest tests.test_run_scripts -v
```

预期：如果新脚本/模块缺失或不可导入，则 FAIL。

**步骤 3：编写最小实现**

补全 public exports 与 CLI wiring，确保脚本存在且可导入。

**步骤 4：再次运行测试并确认通过**

执行：

```bash
python -m unittest tests.test_run_scripts -v
```

预期：PASS。

### 任务 4：对第一版 Agent 做聚焦验证

**涉及文件：**
- 修改：`data_generation/scripts/taxonomy_allocation_tools.py`
- 修改：`data_generation/scripts/taxonomy_allocation_agent.py`
- 测试：`tests/test_taxonomy_allocation_tools.py`
- 测试：`tests/test_taxonomy_allocation_agent.py`
- 测试：`tests/test_run_scripts.py`

**步骤 1：运行聚焦验证**

执行：

```bash
python -m unittest \
  tests.test_taxonomy_allocation_tools \
  tests.test_taxonomy_allocation_agent \
  tests.test_run_scripts \
  -v
```

预期：PASS，且没有失败用例。

**步骤 2：仅在必要时做收口修正**

如果有测试失败，只做最小修复，然后重复执行同一条命令，直到全部通过。

**步骤 3：确认范围边界**

确保第一版**不做**以下事情：
- 不自动做生产策略决策
- 不修改在线生成行为
- 不依赖 LLM

第一版只负责写出事实层分析 artifact 和一个可编辑的 allocation plan 模板。
