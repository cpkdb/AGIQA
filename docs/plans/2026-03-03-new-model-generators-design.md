# New Model Generators Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 为数据生成流程新增 `hunyuan-dit` 和 `sd3.5-large` 两个可选生成模型，并保持在单卡 4090 场景下优先追求推理速度。

**Architecture:** 复用现有生成器注册表模式，在 `data_generation/scripts/` 下新增两个生成器类，并在 `tools/image_generator.py` 中统一注册和分发。`Hunyuan-DiT` 默认走全 GPU 快路径；`SD3.5 Large` 默认走速度优先量化路径，并保留显存不足时的 offload 回退参数。测试只覆盖本地可验证的注册和 CLI 路径，不依赖真实模型下载。

**Tech Stack:** Python, diffusers, transformers, torch, bitsandbytes, pytest

---

### Task 1: 计划外的本地可测面整理

**Files:**
- Create: `tests/test_image_generator_registry.py`
- Create: `tests/test_pipeline_model_choices.py`

**Step 1: Write the failing test**

为 `tools/image_generator.py` 写注册分发测试，验证传入 `model_id="hunyuan-dit"` 和 `model_id="sd3.5-large"` 时会走到对应注册函数；为 `pipeline.py` 写 CLI choices 测试，验证新模型出现在 `--model_id` 枚举中。

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_image_generator_registry.py tests/test_pipeline_model_choices.py -q`
Expected: FAIL，因为当前代码还不支持这两个模型。

**Step 3: Write minimal implementation**

先只修改注册分发表和 CLI 枚举，让测试转绿。

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_image_generator_registry.py tests/test_pipeline_model_choices.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_image_generator_registry.py tests/test_pipeline_model_choices.py data_generation/scripts/tools/image_generator.py data_generation/scripts/pipeline.py
git commit -m "test: cover new generator model ids"
```

### Task 2: 新增 Hunyuan-DiT 生成器

**Files:**
- Create: `data_generation/scripts/hunyuan_dit_generator.py`
- Modify: `data_generation/scripts/tools/image_generator.py`

**Step 1: Write the failing test**

扩展已有注册测试，要求 `image_generator()` 在 `hunyuan-dit` 模式下可注册并将参数透传到生成器实例。

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_image_generator_registry.py -q`
Expected: FAIL

**Step 3: Write minimal implementation**

新增 `HunyuanDiTGenerator`：
- 使用 diffusers `HunyuanDiTPipeline`
- 默认 `torch.float16`
- 优先全 GPU
- 支持 `generate(prompt, negative_prompt, steps, guidance_scale, width, height, seed, **kwargs)` 统一接口
- 提供 `cleanup()`

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_image_generator_registry.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add data_generation/scripts/hunyuan_dit_generator.py data_generation/scripts/tools/image_generator.py tests/test_image_generator_registry.py
git commit -m "feat: add hunyuan dit generator"
```

### Task 3: 新增 SD3.5 Large 生成器

**Files:**
- Create: `data_generation/scripts/sd35_large_generator.py`
- Modify: `data_generation/scripts/tools/image_generator.py`

**Step 1: Write the failing test**

扩展注册测试，要求 `sd3.5-large` 也可注册和分发。

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_image_generator_registry.py -q`
Expected: FAIL

**Step 3: Write minimal implementation**

新增 `SD35LargeGenerator`：
- 使用 diffusers `StableDiffusion3Pipeline`
- 默认速度优先：优先 transformer 4bit（若 bitsandbytes 可用）
- 增加可选回退：`use_cpu_offload`
- 统一 `generate(...)` / `cleanup()` 接口

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_image_generator_registry.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add data_generation/scripts/sd35_large_generator.py data_generation/scripts/tools/image_generator.py tests/test_image_generator_registry.py
git commit -m "feat: add sd3.5 large generator"
```

### Task 4: 同步更新调用入口

**Files:**
- Modify: `data_generation/scripts/pipeline.py`
- Modify: `data_generation/scripts/demo_v3_dimension_paired.py`
- Modify: `data_generation/scripts/run_diagnostic.sh`

**Step 1: Write the failing test**

为 CLI choices/脚本字符串添加断言，确保新模型名出现在入口中。

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_pipeline_model_choices.py -q`
Expected: FAIL

**Step 3: Write minimal implementation**

更新 `--model_id` 选项、演示脚本初始化分支，以及诊断脚本中的模型列表与默认建议参数。

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_pipeline_model_choices.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add data_generation/scripts/pipeline.py data_generation/scripts/demo_v3_dimension_paired.py data_generation/scripts/run_diagnostic.sh tests/test_pipeline_model_choices.py
git commit -m "feat: expose new model ids in entrypoints"
```

### Task 5: 端到端静态验证

**Files:**
- Verify only

**Step 1: Run targeted tests**

Run: `pytest tests/test_image_generator_registry.py tests/test_pipeline_model_choices.py -q`

**Step 2: Run syntax validation**

Run: `python -m compileall data_generation/scripts`

**Step 3: Confirm no unintended breakage**

检查修改文件 diff，确认未引入下载逻辑执行路径。

**Step 4: Commit**

```bash
git add docs/plans/2026-03-03-new-model-generators-design.md
git commit -m "docs: add new model generator implementation plan"
```
