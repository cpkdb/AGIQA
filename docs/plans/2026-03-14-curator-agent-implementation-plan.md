# Curator Agent 实现计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 实现一个离线、无模型、cross-run 的 `Curator Agent`，对已有 run artifact 做数据整理，输出 blacklist 候选、curation 决策和跨 run 统计记忆。

**Architecture:** 采用 deterministic 的规则式 agent。输入是现有 run 目录中的 `dataset.json`、`full_log.json`、`validation_report.json`，输出结构化 artifact；不引入新的 LLM/VLM，不直接删除样本，只做整理和记忆沉淀。

**Tech Stack:** Python 3.10、JSON/JSONL、`pathlib`、`argparse`、`unittest`

---

### Task 1: 定义 Curator Agent 输出 schema

**Files:**
- Create: `data_generation/docs/curator_agent.md`
- Create: `tests/test_curator_agent.py`

**Step 1: 写失败测试，约束最小输出文件集合**

测试内容：
- Agent CLI 存在
- 运行后至少输出：
  - `curation_decisions.jsonl`
  - `blacklist.json`
  - `memory_stats.json`
  - `failure_pattern_summary.json`
  - `manifest.json`

**Step 2: 运行测试，确认失败**

Run:
```bash
python -m unittest tests.test_curator_agent -v
```

**Step 3: 写最小文档和占位实现**

先创建文档和脚本占位，保证测试能导入和调用。

**Step 4: 重新运行测试，确认进入下一个失败点**

Run:
```bash
python -m unittest tests.test_curator_agent -v
```

### Task 2: 实现 run artifact 读取与 failure pattern 聚合

**Files:**
- Create: `data_generation/scripts/curator_agent.py`
- Create: `data_generation/scripts/curator_agent_tools.py`
- Modify: `tests/test_curator_agent.py`

**Step 1: 写失败测试，覆盖 artifact 读取与聚合**

测试内容：
- 能扫描 `runs_root` 下的 run 目录
- 识别 `dataset.json`、`full_log.json`、`validation_report.json`
- 输出 `failure_pattern_summary.json`
- 聚合到：
  - `model_id`
  - `dimension`
  - `failure_type`
  - `count`

**Step 2: 运行测试，确认失败**

Run:
```bash
python -m unittest tests.test_curator_agent -v
```

**Step 3: 写最小实现**

实现：
- `discover_run_artifacts(...)`
- `aggregate_failure_patterns(...)`
- 写出 `failure_pattern_summary.json`

**Step 4: 重新运行测试，确认通过**

Run:
```bash
python -m unittest tests.test_curator_agent -v
```

### Task 3: 实现 blacklist 候选生成

**Files:**
- Modify: `data_generation/scripts/curator_agent_tools.py`
- Modify: `tests/test_curator_agent.py`

**Step 1: 写失败测试**

测试内容：
- 同一 prompt 在同一 `model × dimension` 下反复失败时，进入 `blacklist.json`
- 输出字段包括：
  - `prompt_text`
  - `model_id`
  - `dimension`
  - `failure_count`
  - `failure_types`
  - `candidate_reason`

**Step 2: 运行测试，确认失败**

Run:
```bash
python -m unittest tests.test_curator_agent -v
```

**Step 3: 写最小实现**

规则建议：
- 默认阈值：`failure_count >= 2`
- 只生成候选，不直接删或硬禁用

**Step 4: 重新运行测试，确认通过**

Run:
```bash
python -m unittest tests.test_curator_agent -v
```

### Task 4: 实现 curation decisions 和 memory stats

**Files:**
- Modify: `data_generation/scripts/curator_agent_tools.py`
- Modify: `data_generation/scripts/curator_agent.py`
- Modify: `tests/test_curator_agent.py`

**Step 1: 写失败测试**

测试内容：
- 生成 `curation_decisions.jsonl`
- 每条 decision 至少包含：
  - `sample_id` 或 `pair_id`
  - `model_id`
  - `dimension`
  - `decision`
  - `reason`
- 生成 `memory_stats.json`
- 至少统计：
  - run 数量
  - pair 数量
  - invalid 数量
  - blacklist 候选数量

**Step 2: 运行测试，确认失败**

Run:
```bash
python -m unittest tests.test_curator_agent -v
```

**Step 3: 写最小实现**

建议 decision 集合：
- `keep`
- `review`
- `blacklist_candidate`

第一版规则：
- valid pair → `keep`
- invalid pair → `review`
- 命中 blacklist 候选的 prompt → `blacklist_candidate`

**Step 4: 重新运行测试，确认通过**

Run:
```bash
python -m unittest tests.test_curator_agent -v
```

### Task 5: CLI 收口、文档补齐与 smoke

**Files:**
- Modify: `data_generation/scripts/curator_agent.py`
- Modify: `data_generation/docs/curator_agent.md`
- Test: `tests/test_curator_agent.py`

**Step 1: 写失败测试**

测试内容：
- CLI 支持：
  - `--runs_root`
  - `--output_dir`
  - `--failure_threshold`
- 运行后输出完整 artifact

**Step 2: 运行测试，确认失败**

Run:
```bash
python -m unittest tests.test_curator_agent -v
```

**Step 3: 写最小实现**

补齐：
- CLI 参数
- `manifest.json`
- 文档示例

**Step 4: 运行单测与 smoke**

Run:
```bash
python -m unittest tests.test_curator_agent -v
python data_generation/scripts/curator_agent.py --runs_root /root/autodl-tmp --output_dir /tmp/curator_agent_smoke --failure_threshold 2
```

**Step 5: 提交**

```bash
git add docs/plans/2026-03-14-curator-agent-implementation-plan.md \
        data_generation/scripts/curator_agent.py \
        data_generation/scripts/curator_agent_tools.py \
        data_generation/docs/curator_agent.md \
        tests/test_curator_agent.py
git commit -m "feat: add first curator agent"
```
