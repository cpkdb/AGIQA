# Qwen-Image-Lightning Integration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 为数据生成流程新增 `qwen-image-lightning` 作为第三个主力生成模型，基于 `Qwen/Qwen-Image` base pipeline + `Qwen-Image-Lightning` LoRA 权重接入，优先支持 4090 上的 4-step 快路径。

**Architecture:** 复用现有生成器注册表，在 `data_generation/scripts/` 下新增 `QwenImageLightningGenerator`，内部封装 `DiffusionPipeline.from_pretrained("Qwen/Qwen-Image")` + `load_lora_weights(...)`。第一版仅支持官方推荐的 4-step、`true_cfg_scale=1.0` 路径，并在 `tools/image_generator.py`、`pipeline.py`、演示和诊断脚本中加入新 `model_id`。测试仅覆盖本地可验证的入口与字符串检查，不依赖真实模型下载。

**Tech Stack:** Python, diffusers, transformers, torch, unittest

---

### Task 1: 入口测试先行

**Files:**
- Modify: `tests/test_image_generator_registry.py`
- Modify: `tests/test_pipeline_model_choices.py`
- Modify: `tests/test_run_scripts.py`

**Step 1: Write the failing test**

为 `image_generator.py`、`pipeline.py`、`demo_v3_dimension_paired.py` 和新运行脚本添加 `qwen-image-lightning` 的断言。

**Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests.test_image_generator_registry tests.test_pipeline_model_choices tests.test_run_scripts -v`
Expected: FAIL，因为当前代码还未接入 `qwen-image-lightning`。

**Step 3: Write minimal implementation**

先只补 `model_id` 字符串、脚本路径和 CLI 枚举，让测试转绿。

**Step 4: Run test to verify it passes**

Run: `python3 -m unittest tests.test_image_generator_registry tests.test_pipeline_model_choices tests.test_run_scripts -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_image_generator_registry.py tests/test_pipeline_model_choices.py tests/test_run_scripts.py
git commit -m "test: cover qwen image lightning entrypoints"
```

### Task 2: 新增 Qwen-Image-Lightning 生成器

**Files:**
- Create: `data_generation/scripts/qwen_image_lightning_generator.py`
- Modify: `data_generation/scripts/tools/image_generator.py`

**Step 1: Write the failing test**

扩展注册测试，要求 `image_generator.py` 暴露 `qwen-image-lightning`，并包含专用注册函数。

**Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests.test_image_generator_registry -v`
Expected: FAIL

**Step 3: Write minimal implementation**

新增 `QwenImageLightningGenerator`：
- 使用 `DiffusionPipeline` + `FlowMatchEulerDiscreteScheduler`
- 默认 base: `Qwen/Qwen-Image`
- 默认 LoRA repo: `lightx2v/Qwen-Image-Lightning`
- 默认权重: `Qwen-Image-Lightning-4steps-V2.0.safetensors`
- 统一 `generate(...)` / `cleanup()` 接口
- 内部把外部 `guidance_scale` 映射为 `true_cfg_scale`

**Step 4: Run test to verify it passes**

Run: `python3 -m unittest tests.test_image_generator_registry -v`
Expected: PASS

**Step 5: Commit**

```bash
git add data_generation/scripts/qwen_image_lightning_generator.py data_generation/scripts/tools/image_generator.py tests/test_image_generator_registry.py
git commit -m "feat: add qwen image lightning generator"
```

### Task 3: 同步更新主入口和辅助脚本

**Files:**
- Modify: `data_generation/scripts/pipeline.py`
- Modify: `data_generation/scripts/demo_v3_dimension_paired.py`
- Modify: `data_generation/scripts/run_diagnostic.sh`
- Create: `data_generation/scripts/run_pipeline_qwen_image_lightning.sh`

**Step 1: Write the failing test**

为 CLI 枚举和运行脚本新增 `qwen-image-lightning` 断言。

**Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests.test_pipeline_model_choices tests.test_run_scripts -v`
Expected: FAIL

**Step 3: Write minimal implementation**

更新：
- `pipeline.py` 的 `--model_id`
- `demo_v3_dimension_paired.py` 的模型初始化分支
- `run_diagnostic.sh` 的模型矩阵
- 新增 `run_pipeline_qwen_image_lightning.sh`，默认 `steps=4`、`cfg=1.0`

**Step 4: Run test to verify it passes**

Run: `python3 -m unittest tests.test_pipeline_model_choices tests.test_run_scripts -v`
Expected: PASS

**Step 5: Commit**

```bash
git add data_generation/scripts/pipeline.py data_generation/scripts/demo_v3_dimension_paired.py data_generation/scripts/run_diagnostic.sh data_generation/scripts/run_pipeline_qwen_image_lightning.sh tests/test_pipeline_model_choices.py tests/test_run_scripts.py
git commit -m "feat: expose qwen image lightning in entrypoints"
```

### Task 4: 更新服务器说明文档

**Files:**
- Modify: `data_generation/docs/2026-03-03-new-model-server-setup.md`

**Step 1: Write the failing test**

无需自动化测试；通过文档审查验证。

**Step 2: Write minimal implementation**

补充：
- `qwen-image-lightning` 的依赖和版本要求
- base / LoRA 权重说明
- 首跑命令与建议参数
- 常见问题：diffusers 版本、LoRA 权重匹配

**Step 3: Commit**

```bash
git add data_generation/docs/2026-03-03-new-model-server-setup.md
git commit -m "docs: add qwen image lightning server setup notes"
```

### Task 5: 静态验证

**Files:**
- Verify only

**Step 1: Run targeted tests**

Run: `python3 -m unittest tests.test_image_generator_registry tests.test_pipeline_model_choices tests.test_run_scripts -v`

**Step 2: Run syntax validation**

Run: `PYTHONPYCACHEPREFIX=/tmp/pycache python3 -m compileall data_generation/scripts`

**Step 3: Confirm no runtime download was triggered**

检查 diff，确认未在实现阶段执行任何模型下载命令。

**Step 4: Commit**

```bash
git add docs/plans/2026-03-04-qwen-image-lightning-integration.md
git commit -m "docs: add qwen image lightning integration plan"
```
