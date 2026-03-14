# Curator Agent

`Curator Agent` 是当前 AIGC 质量数据生产流程中的离线、无模型、cross-run 数据整理 agent。

第一版目标：
- 读取已有 run artifact
- 汇总失败模式
- 生成 blacklist 候选
- 生成 curation decisions
- 输出跨 run 统计记忆

当前实现优先以 `full_log.json` 作为 pair-level 成败与 failure type 的主来源，因为主 pipeline 的 `dataset.json` 只保存成功 pair；`dataset.json` 在这版里主要用于补充成功样本信息和 metadata。

它不负责：
- 单 pair 在线判别
- 新的 LLM/VLM 二次评审
- 直接删除样本
- 报告样本挑选

## CLI

```bash
python scripts/curator_agent.py \
  --runs_root /root/autodl-tmp \
  --output_dir /tmp/curator_agent_smoke \
  --failure_threshold 2
```

## 输出产物

- `curation_decisions.jsonl`
- `blacklist.json`
- `memory_stats.json`
- `failure_pattern_summary.json`
- `failure_pattern_by_prompt.json`
- `manifest.json`

## 当前实现状态

它当前已经能：
- 扫描 `runs_root` 下同时包含 `dataset.json`、`full_log.json`、`validation_report.json` 的 run 目录
- 从 `full_log.json` 聚合 `failure_type` 统计
- 生成 `model × dimension × prompt` 级别的 `failure_pattern_by_prompt.json`
- 生成 prompt 级 `blacklist.json`
- 生成 pair 级 `curation_decisions.jsonl`
- 生成 `memory_stats.json` 和 `failure_pattern_summary.json`

当前 decision 只有三类：
- `keep`
- `review`
- `blacklist_candidate`

它当前不会：
- 引入新的 LLM/VLM 二次判别
- 自动删除样本
- 自动做报告样本挑选
