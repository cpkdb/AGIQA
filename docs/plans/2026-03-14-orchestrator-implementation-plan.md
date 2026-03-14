# Orchestrator 实现计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 实现一个无模型、轻量的 Orchestrator，统一读取 active pool / allocation 配置，生成标准化 run config，驱动现有运行入口，并写出 `run_registry.json`。

**Architecture:** Orchestrator 不替代现有 `pipeline.py`，而是在其之上做编排层。第一版只负责配置收口、artifact 路径规范化、run registry 记录和命令构造，不引入新的生成逻辑。

**Tech Stack:** Python 3.10、JSON、`argparse`、`pathlib`、`subprocess`、`unittest`

---

### Task 1: 定义 Orchestrator 的最小输入输出

**Files:**
- Create: `data_generation/docs/orchestrator.md`
- Create: `tests/test_orchestrator.py`

**Step 1: 写失败测试**

测试约束：
- 脚本入口存在
- 最少输出：
  - `run_config.json`
  - `run_registry.json`
  - `launch_command.sh`

**Step 2: 运行测试，确认失败**

Run:
```bash
python -m unittest tests.test_orchestrator -v
```

**Step 3: 写文档和占位脚本**

先把 CLI 和产物约定写出来。

**Step 4: 重新运行测试，确认进入下一个失败点**

Run:
```bash
python -m unittest tests.test_orchestrator -v
```

### Task 2: 实现 active resource 解析与 run config 生成

**Files:**
- Create: `data_generation/scripts/orchestrator.py`
- Create: `data_generation/scripts/orchestrator_tools.py`
- Modify: `tests/test_orchestrator.py`

**Step 1: 写失败测试**

测试内容：
- 默认能解析当前 active common / turbo source pool 和 dimension index
- 能根据模型生成一份标准化 `run_config.json`
- 至少包含：
  - `model_id`
  - `source_prompts`
  - `dimension_subpool_index`
  - `steps`
  - `cfg`
  - `output_dir`

**Step 2: 运行测试，确认失败**

Run:
```bash
python -m unittest tests.test_orchestrator -v
```

**Step 3: 写最小实现**

实现：
- 复用现有 prompt pool / runtime resource 解析逻辑
- 为常用模型生成默认 run config

**Step 4: 重新运行测试，确认通过**

Run:
```bash
python -m unittest tests.test_orchestrator -v
```

### Task 3: 实现 launch command 和 run registry

**Files:**
- Modify: `data_generation/scripts/orchestrator_tools.py`
- Modify: `data_generation/scripts/orchestrator.py`
- Modify: `tests/test_orchestrator.py`

**Step 1: 写失败测试**

测试内容：
- 输出 `launch_command.sh`
- 输出 `run_registry.json`
- registry 至少包含：
  - `run_id`
  - `created_at`
  - `status`
  - `model_id`
  - `source_prompts`
  - `dimension_subpool_index`

**Step 2: 运行测试，确认失败**

Run:
```bash
python -m unittest tests.test_orchestrator -v
```

**Step 3: 写最小实现**

第一版只生成命令和 registry，不强制立即执行。

**Step 4: 重新运行测试，确认通过**

Run:
```bash
python -m unittest tests.test_orchestrator -v
```

### Task 4: 增加 dry-run / execute 开关

**Files:**
- Modify: `data_generation/scripts/orchestrator.py`
- Modify: `tests/test_orchestrator.py`

**Step 1: 写失败测试**

测试内容：
- `--dry_run` 只写 artifact，不执行
- `--execute` 时构造可执行命令
- 当前测试只验证命令生成，不实际启动长任务

**Step 2: 运行测试，确认失败**

Run:
```bash
python -m unittest tests.test_orchestrator -v
```

**Step 3: 写最小实现**

第一版只支持：
- dry-run
- 生成 shell launch script
- 为后续真正执行留接口

**Step 4: 重新运行测试，确认通过**

Run:
```bash
python -m unittest tests.test_orchestrator -v
```

### Task 5: 文档与 smoke

**Files:**
- Modify: `data_generation/docs/orchestrator.md`
- Modify: `data_generation/scripts/orchestrator.py`

**Step 1: 补文档示例**

说明：
- 它不是内容生成 agent
- 只做运行编排和状态记录

**Step 2: 运行单测与 smoke**

Run:
```bash
python -m unittest tests.test_orchestrator -v
python data_generation/scripts/orchestrator.py --output_dir /tmp/orchestrator_smoke --model_id flux-schnell --dry_run
```

**Step 3: 提交**

```bash
git add docs/plans/2026-03-14-orchestrator-implementation-plan.md \
        data_generation/scripts/orchestrator.py \
        data_generation/scripts/orchestrator_tools.py \
        data_generation/docs/orchestrator.md \
        tests/test_orchestrator.py
git commit -m "feat: add first orchestrator"
```
