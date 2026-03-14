# Taxonomy & Allocation Agent

`Taxonomy & Allocation Agent` 是当前 AIGC 质量数据生产流程中的离线分析型 agent。

第一版目标很收敛：
- 自动读取当前生效的 taxonomy
- 自动解析当前脚本实际使用的 prompt pool / subpool
- 自动聚合历史 run 的成功率与失败类型
- 自动估算各模型的粗粒度速度
- 输出事实层 artifact 与可编辑的 allocation 计划模板

它**不会**在第一版中自动替你做生产策略决策。

## CLI

```bash
python scripts/taxonomy_allocation_agent.py \
  --output_dir /root/autodl-tmp/agent_runs/taxonomy_allocation/run_20260314_120000 \
  --runs_root /root/autodl-tmp \
  --recent_run_limit 20
```

可选参数：
- `--taxonomy_path`
- `--model_filter flux-schnell,sd3.5-large-turbo`
- `--recent_run_limit 20`

## 输入

默认会读取：
- active taxonomy：
  - `data_generation/config/quality_dimensions_active.json`
- 当前实际使用的 runtime prompt resources：
  - 通用 cleaned prompt pool
  - anatomy screened cleaned v2 index
  - turbo clipsafe pool / clipsafe v2 index
- 历史 run artifacts：
  - `dataset.json`
  - `full_log.json`
  - `validation_report.json`

## 输出

运行后会写出：
- `coverage_summary.json`
- `model_dimension_stats.json`
- `speed_summary.json`
- `allocation_plan.template.json`
- `allocation_insights.md`
- `manifest.json`

其中：
- `allocation_plan.template.json` 是后续人工补策略的模板
- `allocation_insights.md` 是便于快速阅读的摘要视图

## 当前实现状态

第一版已经实现为一个独立、可运行的离线分析型 Agent：

- 入口：
  - [taxonomy_allocation_agent.py](/root/ImageReward/data_generation/scripts/taxonomy_allocation_agent.py)
- 工具层：
  - [taxonomy_allocation_tools.py](/root/ImageReward/data_generation/scripts/taxonomy_allocation_tools.py)

它当前会自动：
- 解析各模型当前真实会使用的 source pool / subpool index
- 读取 active taxonomy
- 统计每维度 pool 大小
- 聚合历史 run 的成功率、失败类型、平均尝试次数
- 粗粒度估算各模型每对 pair / 每张图的生成速度

但它目前**还没有接入主运行流程**，也不会自动替你做调度决策。

## 当前范围

第一版只做分析，不做自动决策：
- 不自动决定哪些维度多采样
- 不自动决定哪些维度只跑某些模型
- 不自动决定哪些维度只保留 `severe`
- 不自动决定哪些维度开启 judge

这些策略应在正式启动大规模生产前，由人工基于 agent 输出进行配置。
