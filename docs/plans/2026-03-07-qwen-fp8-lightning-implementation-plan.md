# Qwen FP8 Lightning Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add an isolated scaled-FP8 Qwen-Image-Lightning path that can be benchmarked on a single RTX 4090 without destabilizing the existing bf16 baseline.

**Architecture:** Reuse the existing Qwen generator and add one explicit FP8 experimental profile. The stable bf16 model path remains unchanged. The pipeline and scripts expose the profile, and tests guard both profile wiring and prompt-safety behavior.

**Tech Stack:** Python, Diffusers, PyTorch, existing Qwen generator and pipeline scripts, unittest-based repository tests.

---

### Task 1: Lock profile and safety requirements in tests

**Files:**
- Modify: `tests/test_qwen_runtime_defaults.py`
- Test: `tests/test_qwen_runtime_defaults.py`

**Step 1: Write the failing test**

- Assert that:
  - Qwen generator exposes the FP8 profile marker.
  - Pipeline rejects empty degraded prompts before generation.

**Step 2: Run test to verify it fails**

Run: `python tests/test_qwen_runtime_defaults.py`
Expected: FAIL because FP8 profile strings and/or empty-prompt guard are missing.

**Step 3: Write minimal implementation**

- Only enough test assertions to pin the desired code shape.

**Step 4: Run test to verify it passes**

Run: `python tests/test_qwen_runtime_defaults.py`
Expected: PASS

### Task 2: Add explicit empty negative prompt guard

**Files:**
- Modify: `data_generation/scripts/pipeline.py`
- Modify: `data_generation/scripts/llm_prompt_degradation.py`
- Test: `tests/test_qwen_runtime_defaults.py`

**Step 1: Write the failing test**

- Reuse the Task 1 test requirement for empty prompt guard if still failing.

**Step 2: Run test to verify it fails**

Run: `python tests/test_qwen_runtime_defaults.py`
Expected: FAIL on missing empty prompt guard.

**Step 3: Write minimal implementation**

- In `pipeline.py`, stop before image generation if `negative_prompt` is empty or whitespace.
- In `llm_prompt_degradation.py`, treat an empty post-validated string as an error.

**Step 4: Run test to verify it passes**

Run: `python tests/test_qwen_runtime_defaults.py`
Expected: PASS

### Task 3: Add isolated scaled-FP8 Qwen runtime path

**Files:**
- Modify: `data_generation/scripts/qwen_image_lightning_generator.py`
- Modify: `data_generation/scripts/tools/image_generator.py`
- Test: `tests/test_qwen_runtime_defaults.py`

**Step 1: Write the failing test**

- Extend tests to assert the new FP8 profile hook exists in the generator wiring.

**Step 2: Run test to verify it fails**

Run: `python tests/test_qwen_runtime_defaults.py`
Expected: FAIL because the profile branch does not exist yet.

**Step 3: Write minimal implementation**

- Add a dedicated FP8 experimental profile branch.
- Keep the existing bf16 path intact.
- Ensure profile metadata is logged in generation info.

**Step 4: Run test to verify it passes**

Run: `python tests/test_qwen_runtime_defaults.py`
Expected: PASS

### Task 4: Expose FP8 profile in scripts without changing stable defaults blindly

**Files:**
- Modify: `data_generation/scripts/run_pipeline_qwen_image_lightning.sh`
- Modify: `data_generation/scripts/run_diagnostic_tri_models_small_batch.sh`
- Test: `tests/test_run_scripts.py`

**Step 1: Write the failing test**

- Add assertions only if script exposure changes are needed for the new profile.

**Step 2: Run test to verify it fails**

Run: `python tests/test_run_scripts.py`
Expected: FAIL if expected profile strings are not present.

**Step 3: Write minimal implementation**

- Expose the FP8 experimental profile as an explicit opt-in.
- Do not silently replace the stable baseline unless runtime testing justifies it.

**Step 4: Run test to verify it passes**

Run: `python tests/test_run_scripts.py`
Expected: PASS

### Task 5: Static regression pass

**Files:**
- Test: `tests/test_qwen_runtime_defaults.py`
- Test: `tests/test_run_scripts.py`
- Test: `tests/test_pipeline_model_choices.py`

**Step 1: Run targeted tests**

Run: `python tests/test_qwen_runtime_defaults.py`
Expected: PASS

**Step 2: Run script regression**

Run: `python tests/test_run_scripts.py`
Expected: PASS

**Step 3: Run model-choice regression**

Run: `python tests/test_pipeline_model_choices.py`
Expected: PASS

### Task 6: 4090 runtime benchmark

**Files:**
- Modify: none required unless runtime findings demand adjustment

**Step 1: Run bf16 baseline smoke**

Run a 1024x1024 4-step single-image smoke on the verified bf16 path.
Expected: correct image, benchmark captured.

**Step 2: Run scaled-FP8 smoke**

Run the same smoke on the FP8 path.
Expected: no corrupted image, benchmark captured.

**Step 3: Compare**

- elapsed time
- peak VRAM
- obvious visual failures

**Step 4: Decide final default**

- If FP8 is both correct and materially faster, recommend promoting it.
- Otherwise keep it experimental.
