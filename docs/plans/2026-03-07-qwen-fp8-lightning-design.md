# Qwen FP8 Lightning Design

**Goal:** Add an isolated FP8 loading path for Qwen-Image-Lightning on a single RTX 4090, targeting better throughput than the current offload-heavy bf16 path without breaking the verified bf16 baseline.

**Current Baseline**

- The verified stable path is `Qwen/Qwen-Image` bf16 base + `Qwen-Image-Lightning-4steps-V2.0.safetensors`.
- This path produces correct images when loaded with the official Diffusers-style scheduler and LoRA flow.
- A full-GPU bf16 load was tested on the 24GB RTX 4090 and OOMed at about 23.47 GiB used.
- The current low-memory runtime uses CPU offload and only peaks around 1.5 GiB VRAM in the degradation smoke test, which is too conservative for the throughput target.

**Constraints**

- Do not regress the currently working bf16 path.
- Keep the FP8 path separate and explicitly opt-in.
- Prefer the official compatibility guidance from ModelTC/Qwen-Image-Lightning over community variants.
- Choose the FP8 variant that balances speed and image correctness, not just minimum VRAM.

**Recommended FP8 Variant**

- Use the `scaled FP8` route first.
- Rationale: ModelTC documents that `scaled fp8 base + LoRA trained on bf16 base` is compatible, while raw fp8 base mixed with bf16-trained LoRA is not.
- This makes `scaled FP8` the safest experimental upgrade from the currently validated bf16-Lightning setup.

**Approach**

- Keep `qwen-image-lightning` as the stable bf16 model id.
- Add a separate FP8 experiment path through either:
  - a new `model_id`, or
  - a new Qwen-specific runtime profile that is only used when an FP8 base/weight is configured.
- Start with the smallest change surface: reuse the existing Qwen generator file, but branch internally on a dedicated FP8 profile.

**Runtime Profiles**

- `fit-24g`
  - Stable baseline
  - bf16 base
  - model/sequential CPU offload
- `fast-gpu-24g`
  - Existing experimental full-GPU attempt
  - allowed to fall back when OOM
- `fp8-scaled-24g`
  - New experimental profile
  - uses scaled FP8 assets only
  - optimized for higher GPU utilization and lower VRAM than full bf16

**Behavioral Requirements**

- Empty degraded prompts must be rejected before image generation.
- FP8 must never silently replace the stable bf16 path.
- Logs must record which Qwen runtime profile was used.
- If FP8 loading fails or produces incompatible asset errors, the failure should be explicit.

**Validation**

- Static tests:
  - profile exposure in scripts and defaults
  - empty negative prompt guard
- Runtime smoke tests on 4090:
  - 1024x1024, 4 steps, `true_cfg_scale=1.0`
  - confirm image is not corrupted
  - record elapsed time
  - record VRAM peak

**Success Criteria**

- FP8 path loads with the intended assets.
- FP8 path produces a visually sane image.
- FP8 path uses meaningfully more GPU than the current offload baseline while staying within 24GB.
- Existing bf16 path and existing tests remain green.
