# Orchestrator

`Orchestrator` 是当前 AIGC 质量数据生产流程中的无模型、轻量运行编排层。

第一版目标：
- 读取 active prompt pool / dimension index
- 生成标准化 `run_config.json`
- 生成 `launch_command.sh`
- 生成 `run_registry.json`
- 支持 `dry_run` 和最小版 `execute`

它不负责：
- 生成内容
- 判别图像质量
- 替代 `pipeline.py`

## CLI

```bash
python scripts/orchestrator.py \
  --output_dir /tmp/orchestrator_smoke \
  --model_id flux-schnell \
  --dry_run
```

也支持直接执行：

```bash
python scripts/orchestrator.py \
  --output_dir /tmp/orchestrator_execute \
  --model_id flux-schnell \
  --execute
```

当前 `execute` 的边界是：
- 直接执行生成好的 `launch_command.sh`
- 在 `run_registry.json` 中更新 `planned -> running -> completed/failed`
- 记录 `log_path` 和 `return_code`
- 不负责更复杂的 resume / queue / 多任务调度
