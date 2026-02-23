# Degradation Strategy Optimization Report

## 1. Problem Identification
The initial "Append-Only" strategy for generating technical degradations (adding descriptors like "overexposed", "plastic texture" to the end of prompts) proved ineffective for SDXL.
- **Reason**: SDXL is highly capable of reconciling conflicting instructions. If a prompt says "detailed, high quality photo... (plus) plastic texture", SDXL often interprets it as a *stylistic choice* (e.g., a high-quality 3D render, a high-key photo) rather than a technical failure.
- **Symptom**: Generated images looked too good. "Overexposed" looked like artistic bright lighting; "Plastic" looked like expensive 3D art; "Color Cast" looked like a filter.

## 2. Solution: "Injection/Rewrite" Strategy
The new strategy forces the LLM to **rewrite the core logic** of the prompt.
- **Action**: Instead of appending `", overexposed"`, we now instruct the LLM to:
    1.  **Replace** attributes: "vibrant flower" -> "washed-out flower".
    2.  **Delete** contradictions: Remove words like "detailed", "sharp", "cinematic", "balanced".
    3.  **Inject** flaws: "soft skin" -> "waxy vinyl skin", "golden hour" -> "sickly yellow haze".

## 3. Updates Implemented
I have applied this strategy to the following dimensions in `technical_quality.yaml`:

### A. Overexposure & Underexposure
- **Old**: "Add 'overexposed' to the end."
- **New**: "Replace 'soft lighting' with 'harsh glare'. Delete 'detailed'. Replace 'vibrant' with 'washed-out'."

### B. Color Cast
- **Old**: "Add 'with a color tint'."
- **New**: "Replace 'beautiful sunset' with 'radioactive orange haze'. Replace 'cool blue' with 'dull teal'. Delete 'cinematic'."
- **Palette**: Shifted from generic "tint" to visceral terms: "mucus green", "sewage brown", "stained beige".

### C. Plastic/Waxy Texture
- **Old**: "Add 'plastic texture' to the end."
- **New**: "Replace 'skin' with 'vinyl doll material'. Replace 'wood' with 'brown plastic'. Delete 'furry', 'rough'. Use terms like 'cheap 3D render', 'melted wax'."

## 4. Next Steps
- Re-generate datasets for these attributes.
- Use the `llm_prompt_degradation.py` script to validate.
- Monitor `blur` and `desaturation` results; they may require similar updates if they prove too subtle.
