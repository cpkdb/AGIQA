# 三模型小批量全维度退化测试说明（4090，速度优先）

本文说明如何使用新脚本对当前三模型进行小批量维度退化测试：

- `flux-schnell`
- `sd3.5-large`（建议加载 Turbo 权重）
- `qwen-image-lightning`

脚本路径：

- `scripts/run_diagnostic_tri_models_small_batch.sh`

---

## 1. 设计目标

面向你当前的 4 万级正样本 prompt 池，先做“低成本、可比对”的小批量验证：

- 覆盖全部维度组（默认）
- 每个维度只抽少量 prompt（默认 2 条）
- severity 默认 `moderate,severe`
- 单模型顺序执行，避免同卡并发导致显存冲突

---

## 2. 默认 Prompt 源

脚本默认读取：

```bash
data/prompt_sources_workspace/backfill_merge_runs/all_dimensions_v1_full/merged_working_pool.jsonl
```

`pipeline.py` 已更新为支持 JSONL，并兼容以下字段：

- `prompt` / `text`
- `semantic_tags` / `tags`
- `prompt_signature` / `signature`

如果记录没有 `model` 字段，不会被 `--model_filter` 误过滤。

---

## 3. 推荐运行方式

全模型 + 全维度组（默认）：

```bash
bash scripts/run_diagnostic_tri_models_small_batch.sh
```

只跑某个模型：

```bash
bash scripts/run_diagnostic_tri_models_small_batch.sh flux-schnell
bash scripts/run_diagnostic_tri_models_small_batch.sh sd3.5-large-turbo
bash scripts/run_diagnostic_tri_models_small_batch.sh qwen-image-lightning
```

只跑某个维度组：

```bash
bash scripts/run_diagnostic_tri_models_small_batch.sh all semantic_spatial
```

只跑指定维度列表（逗号分隔）：

```bash
bash scripts/run_diagnostic_tri_models_small_batch.sh all blur,text_error,hand_malformation
```

---

## 4. 关键默认参数（速度优先）

- `PROMPTS_PER_DIM=2`
- `MAX_RETRIES=1`
- `SEVERITIES=moderate,severe`
- `flux-schnell`: `steps=4`, `cfg=0.0`, 自动 `--optimize`
- `sd3.5-large`（Turbo 权重）：`steps=4`, `cfg=0.0`
- `qwen-image-lightning`: `steps=4`, `cfg=1.0`

显存不足时可通过环境变量启用 offload（会明显变慢）：

```bash
SD35_USE_CPU_OFFLOAD=true \
QWEN_USE_CPU_OFFLOAD=true \
bash scripts/run_diagnostic_tri_models_small_batch.sh
```

---

## 5. 服务器部署时需要改的最小项

下载模型后，通常只需确认 3 个路径：

- `FLUX_SCHNELL_MODEL_PATH`
- `SD35_LARGE_TURBO_MODEL_PATH`
- `QWEN_IMAGE_LIGHTNING_MODEL_PATH`

示例：

```bash
FLUX_SCHNELL_MODEL_PATH=/root/autodl-tmp/flux-1-schnell \
SD35_LARGE_TURBO_MODEL_PATH=/root/autodl-tmp/sd3.5-large-turbo \
QWEN_IMAGE_LIGHTNING_MODEL_PATH=Qwen/Qwen-Image \
bash scripts/run_diagnostic_tri_models_small_batch.sh
```

---

## 6. 输出结构

输出根目录默认为：

```bash
/root/autodl-tmp/tri_model_small_batch_YYYYmmdd_HHMMSS
```

子目录组织：

```text
<output_root>/
  flux-schnell/<group>/
  sd3.5-large/<group>/
  qwen-image-lightning/<group>/
```

每个实验目录会生成：

- `dataset.json`
- `full_log.json`
- `validation_report.json`

可直接用于后续对比三模型在各维度下的有效退化率、失败类型和重试代价。
