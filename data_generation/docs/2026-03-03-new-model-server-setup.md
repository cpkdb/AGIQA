# 新增生成模型服务器部署与运行说明

本文面向单卡 RTX 4090 服务器，说明如何为当前项目准备并运行新增的生成模型：

- `hunyuan-dit`
- `sd3.5-large`
- `qwen-image-lightning`

对应的运行脚本已经接入仓库：

- `scripts/run_pipeline_hunyuan.sh`
- `scripts/run_pipeline_sd35_large.sh`
- `scripts/run_pipeline_qwen_image_lightning.sh`
- `scripts/run_diagnostic_tri_models_small_batch.sh`（三模型小批量全维度测试）

本文不负责自动下载模型，只提供推荐目录、依赖和首跑方式。

如果你当前主线是 `Flux Schnell + SD3.5 Large Turbo + Qwen-Image-Lightning`，建议直接看：

- `docs/2026-03-05-tri-model-small-batch-diagnostic.md`

---

## 1. 适用范围

当前说明适用于以下场景：

- 单卡 `RTX 4090 (24GB VRAM)`
- 使用当前仓库中的闭环生成入口 `scripts/pipeline.py`
- 模型文件放在本地磁盘目录中，由脚本通过 `--model_path` 读取

建议的默认理解：

- `Hunyuan-DiT` 优先走全 GPU，追求速度
- `SD3.5 Large` 默认也先按速度优先尝试；如果显存吃紧，再开启 `--use_cpu_offload`
- `Qwen-Image-Lightning` 默认按官方推荐的 4-step 快路径运行

---

## 2. 环境准备

建议先确认 Python 环境中至少有以下依赖：

```bash
pip install -U torch diffusers transformers accelerate pillow pyyaml
```

如果要跑 `SD3.5 Large`，建议额外安装：

```bash
pip install -U bitsandbytes
```

如果你希望部分模型启用更好的显存优化，还可以安装：

```bash
pip install -U xformers
```

建议先确认基础环境：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
python -c "import diffusers, transformers; print(diffusers.__version__); print(transformers.__version__)"
```

如果 `torch.cuda.is_available()` 返回 `False`，先不要继续跑生成脚本。

---

## 3. 模型下载目录建议

当前代码中默认建议把模型下载到以下目录：

```bash
/root/autodl-tmp/hunyuan-dit
/root/autodl-tmp/sd3.5-large
```

对应关系如下：

- `Hunyuan-DiT`: 建议下载到 `/root/autodl-tmp/hunyuan-dit`
- `SD3.5 Large`: 建议下载到 `/root/autodl-tmp/sd3.5-large`
- `Qwen-Image-Lightning`: 当前实现默认使用 `Qwen/Qwen-Image` 作为 base 模型，并从 `lightx2v/Qwen-Image-Lightning` 加载 Lightning LoRA

如果你使用 Hugging Face CLI 或其他方式下载，只要最终目录结构完整，并把脚本中的 `MODEL_PATH` 改到对应目录即可。

注意：

- `SD3.5 Large` 在 Hugging Face 上通常需要先接受模型许可后才能下载。
- 如果你不想放在上述目录，也可以直接修改运行脚本里的 `MODEL_PATH`。

---

## 4. 首次运行方式

### 4.1 Hunyuan-DiT

推荐直接使用现成脚本：

```bash
bash scripts/run_pipeline_hunyuan.sh
```

该脚本默认参数：

- `--model_id hunyuan-dit`
- `--model_path /root/autodl-tmp/hunyuan-dit`
- `--steps 30`
- `--cfg 5.0`
- 默认不启用 `--use_cpu_offload`

如果你要手动运行，等价命令大致如下：

```bash
python scripts/pipeline.py \
  --source_prompts data/prompts_tagged_sdxl_v2.json \
  --output_dir /root/autodl-tmp/hunyuan_test \
  --model_id hunyuan-dit \
  --model_path /root/autodl-tmp/hunyuan-dit \
  --num_pairs_per_prompt 3 \
  --max_prompts 3 \
  --max_retries 2 \
  --seed 42 \
  --severities moderate,severe \
  --steps 30 \
  --cfg 5.0 \
  --shuffle \
  --subcategory_filter aesthetic_quality \
  --model_filter sdxl
```

### 4.2 SD3.5 Large

推荐直接使用现成脚本：

```bash
bash scripts/run_pipeline_sd35_large.sh
```

该脚本默认参数：

- `--model_id sd3.5-large`
- `--model_path /root/autodl-tmp/sd3.5-large`
- `--steps 28`
- `--cfg 4.5`
- 默认不启用 `--use_cpu_offload`

如果你要手动运行，等价命令大致如下：

```bash
python scripts/pipeline.py \
  --source_prompts data/prompts_tagged_sdxl_v2.json \
  --output_dir /root/autodl-tmp/sd35_test \
  --model_id sd3.5-large \
  --model_path /root/autodl-tmp/sd3.5-large \
  --num_pairs_per_prompt 3 \
  --max_prompts 3 \
  --max_retries 2 \
  --seed 42 \
  --severities moderate,severe \
  --steps 28 \
  --cfg 4.5 \
  --shuffle \
  --subcategory_filter aesthetic_quality \
  --model_filter sdxl
```

如果显存不足，再在命令最后加上：

```bash
--use_cpu_offload
```

### 4.3 Qwen-Image-Lightning

推荐直接使用现成脚本：

```bash
bash scripts/run_pipeline_qwen_image_lightning.sh
```

该脚本默认参数：

- `--model_id qwen-image-lightning`
- `--model_path Qwen/Qwen-Image`
- `--steps 4`
- `--cfg 1.0`
- 默认不启用 `--use_cpu_offload`

当前代码会在生成器内部：

- 使用 `Qwen/Qwen-Image` 作为 base pipeline
- 从 `lightx2v/Qwen-Image-Lightning` 加载 `Qwen-Image-Lightning-4steps-V2.0.safetensors`

如果你要手动运行，等价命令大致如下：

```bash
python scripts/pipeline.py \
  --source_prompts data/prompts_tagged_sdxl_v2.json \
  --output_dir /root/autodl-tmp/qwen_image_lightning_test \
  --model_id qwen-image-lightning \
  --model_path Qwen/Qwen-Image \
  --num_pairs_per_prompt 3 \
  --max_prompts 3 \
  --max_retries 2 \
  --seed 42 \
  --severities moderate,severe \
  --steps 4 \
  --cfg 1.0 \
  --shuffle \
  --subcategory_filter aesthetic_quality \
  --model_filter sdxl
```

如果显存不足，也可以尝试：

```bash
--use_cpu_offload
```

---

## 5. 调参建议

### 5.1 Hunyuan-DiT

建议优先保持默认：

- `steps=30`
- `cfg=5.0`
- 不开 `use_cpu_offload`

如果你发现速度可以接受，想进一步提质，可以尝试：

- 增加 `steps` 到 `35` 或 `40`

如果你更关心吞吐量，可以先尝试：

- 降低 `max_prompts`
- 降低 `num_pairs_per_prompt`
- 维持 `1024x1024` 不变，先不要同时提高分辨率和步数

### 5.2 SD3.5 Large

建议先按默认值启动：

- `steps=28`
- `cfg=4.5`
- 默认不开 `use_cpu_offload`

如果出现 OOM，建议按这个顺序回退：

1. 打开 `--use_cpu_offload`
2. 降低 `max_prompts`
3. 降低 `num_pairs_per_prompt`
4. 视情况把 `steps` 降到 `24` 或 `20`

不建议一开始就同时把步数拉高并扩大批量。

### 5.3 Qwen-Image-Lightning

第一版建议保持官方快路径：

- `steps=4`
- `cfg=1.0`
- 默认不开 `use_cpu_offload`

如果服务器上的 `diffusers` 版本较旧，优先升级而不是先改参数。

---

## 6. 常见问题

### 6.1 `diffusers` 版本过低

如果报错里出现以下类型的信息：

- 找不到 `HunyuanDiTPipeline`
- 找不到 `StableDiffusion3Pipeline`
- 找不到 `SD3Transformer2DModel`
- 找不到 `FlowMatchEulerDiscreteScheduler`
- `Qwen-Image` / `Qwen-Image-Lightning` 相关 pipeline 初始化失败

通常说明 `diffusers` 版本偏旧。先升级：

```bash
pip install -U diffusers transformers accelerate
```

`Qwen-Image-Lightning` 官方更偏向较新的 `diffusers`，如果普通升级仍不够，可按官方建议安装更新版本。

### 6.2 `bitsandbytes` 缺失

如果 `SD3.5 Large` 相关报错提到 `BitsAndBytesConfig` 或 4bit 量化不可用，执行：

```bash
pip install -U bitsandbytes
```

即使没有 `bitsandbytes`，代码也会尝试回到非量化路径，但在 4090 上更容易吃满显存。

### 6.3 模型未授权或下载不完整

如果 `SD3.5 Large` 无法从 Hugging Face 拉取，优先检查：

- 是否已经在 Hugging Face 页面接受许可
- 本地目录是否下载完整
- 传入的 `--model_path` 是否指向模型根目录，而不是子目录

### 6.4 4090 显存不足

如果出现 `CUDA out of memory`：

- `Hunyuan-DiT`：先减少任务规模，再考虑临时打开 `use_cpu_offload`
- `SD3.5 Large`：优先打开 `--use_cpu_offload`
- `Qwen-Image-Lightning`：先打开 `--use_cpu_offload`，再降低任务规模

不要第一步就修改太多参数，否则很难定位瓶颈。

---

## 7. 建议的首次验证流程

建议第一次上线时按下面顺序做：

1. 先跑 `Hunyuan-DiT`，确认新模型路径、依赖和 pipeline 通路都正常
2. 再跑 `SD3.5 Large` 默认配置
3. 如果 `SD3.5 Large` OOM，再只加 `--use_cpu_offload` 重试
4. 再跑 `Qwen-Image-Lightning`
5. 跑通后再逐步扩大 `MAX_PROMPTS` 和 `NUM_PAIRS`

这样最容易把问题定位在“模型依赖”、“模型权限”还是“显存压力”。

---

## 8. 当前仓库内相关文件

如果后续你要继续调整，优先看这些文件：

- `scripts/hunyuan_dit_generator.py`
- `scripts/sd35_large_generator.py`
- `scripts/qwen_image_lightning_generator.py`
- `scripts/tools/image_generator.py`
- `scripts/pipeline.py`
- `scripts/run_pipeline_hunyuan.sh`
- `scripts/run_pipeline_sd35_large.sh`
- `scripts/run_pipeline_qwen_image_lightning.sh`

这几处已经覆盖了新增模型的主要接入点。
