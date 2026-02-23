# 退化效果检查报告 (Aesthetic + Technical Quality)

**检查日期**: 2026-01-10
**检查数据**: `/root/autodl-tmp/demo_v3_dimension_paired_20260108_170412/aesthetic_quality`
**模板文件**: `/root/ImageReward/data_generation/config/prompt_templates_v3/aesthetic_quality.yaml`

---

## 一、检查结果汇总

| 维度 | 状态 | 问题描述 |
|------|------|----------|
| amateur_look | ❌ 失败 | 风格变化（艺术画风→照片风格）而非视觉质量退化 |
| cluttered_scene | ✅ 通过 | 杂乱感明显增加，退化效果符合预期 |
| color_clash | ✅ 通过 | 颜色冲突明显，退化效果符合预期 |
| dull_palette | ⚠️ 一般 | 颜色暗淡效果不够明显，两图颜色差异不大 |
| flat_lighting | ❌ 失败 | **正负样本质量反转**：正样本是灰度素描，负样本反而是漂亮的彩色照片 |
| harsh_lighting | ✅ 通过 | 负样本确实有更强烈的过曝效果 |
| lack_of_depth | ✅ 通过 | 负样本呈现明显的纸片感，退化效果显著 |
| lighting_imbalance | ⚠️ 一般 | 光线不均匀效果不够强烈 |
| overprocessed_look | ❌ 失败 | 风格变化（宫崎骏动画→真实照片）而非过度处理效果 |
| poor_composition_centered | ❌ 失败 | 风格差异大（写实照片vs扁平插画），构图问题不明显 |
| poor_composition_cropped | ❌ 失败 | 裁切问题未体现，两图内容差异过大 |
| style_inconsistency | ❌ 失败 | 风格混乱未体现，两图都是类似的亚洲艺术风格 |
| unbalanced_layout | ❌ 失败 | 布局不平衡未体现，负样本构图反而更对称 |

**统计**: 4通过 / 2一般 / 7失败

---

## 二、根本原因分析

### 原因1: 正负样本使用不同seed（已修复）

**代码位置**: `demo_v3_dimension_paired.py:479`
```python
# 原代码
neg_seed = pos_seed + (SEVERITIES.index(severity) + 1) * 10000
```

**影响**: 不同seed导致SDXL生成完全不同风格/内容的图像，掩盖了prompt退化效果。

**修复**: 已将代码修改为 `neg_seed = pos_seed`，确保正负样本使用相同seed。

### 原因2: 部分模板退化策略错误

某些模板的退化策略引导LLM改变了图像的艺术风格，而非引入视觉质量退化。

**典型问题**:
- `amateur_look`: 模板说"simple photo"、"phone camera quality"，导致LLM将艺术风格转换为照片风格
- `overprocessed_look`: 退化描述不够具体，被LLM理解为风格变化
- 构图类维度: 描述不够强烈，SDXL难以理解

---

## 三、各问题维度详细分析与解决方案

### 3.1 amateur_look ❌

**当前问题**:
- 正样本: 波斯细密画风格的市场街道（艺术画风）
- 负样本: 变成了真实照片风格的市场街道
- 核心问题: 退化变成了"艺术风格→照片风格"的转换

**当前模板问题** (severe级别):
```
Add extreme amateur: "terrible amateur snapshot", "worst possible phone camera photo"
```
- 使用了"photo"、"camera"等词，引导LLM改变艺术风格

**解决方案**:
```yaml
# 核心原则: 保持原始艺术风格不变，只添加渲染质量退化
CRITICAL RULES:
- Keep the EXACT same artistic style
- DO NOT change the medium or convert between styles
- Only add quality degradation descriptors

# 改进后的策略:
1. Add rendering degradation: "with rough unfinished edges", "visible rendering artifacts"
2. Add quality issues: "hastily rendered", "draft quality rendering", "with jagged aliased edges"
3. Describe low-quality generation: "like a failed render", "with broken fragmented elements"
```

---

### 3.2 flat_lighting ❌

**当前问题**:
- 正样本: 灰度素描风格（老人在树下看书）
- 负样本: 彩色照片风格，有漂亮的光线效果
- 核心问题: **正负样本视觉质量完全反转**

**当前模板问题**:
```
Replace dramatic lighting: "golden hour" → "overcast afternoon"
```
- 这种替换策略可能被LLM错误理解，导致风格变化

**解决方案**:
```yaml
# 核心原则: 明确添加平光描述词，不改变场景风格
CRITICAL RULES:
- Keep the EXACT same artistic style and scene content
- Only ADD flat lighting descriptors, don't replace style terms

# 改进后的策略:
1. Append flat lighting terms: add "under completely flat even lighting" at the end
2. Add specific AIGC flat lighting artifacts: "with no shadows", "uniformly lit with no depth"
3. Use SDXL-friendly terms: "flat ambient lighting only", "no directional light source"
```

---

### 3.3 overprocessed_look ❌

**当前问题**:
- 正样本: 宫崎骏风格动画（小孩被气球带飞）
- 负样本: 变成了类似HDR真实照片的风格
- 核心问题: 风格变化而非过度处理

**当前模板问题**:
```
"Instagram filter cranked up", "too much Lightroom editing"
```
- 这些是针对照片后处理的描述，对AIGC图像不适用

**解决方案**:
```yaml
# 核心原则: 保持原始风格，添加AIGC特有的过度处理特征
CRITICAL RULES:
- Keep the EXACT same artistic style
- Add AIGC-specific over-processing artifacts

# 改进后的策略:
1. Add saturation issues: "with oversaturated colors", "excessive color vibrancy"
2. Add sharpening artifacts: "with over-sharpened edges creating halos", "artificial edge enhancement"
3. Add contrast issues: "with crushed blacks and blown highlights", "excessive contrast"
4. AIGC specific: "with AI upscaling artifacts", "over-enhanced details looking artificial"
```

---

### 3.4 poor_composition_centered ❌

**当前问题**:
- 正样本: 购物中心俯瞰图（写实照片）
- 负样本: 扁平插画风格的购物中心
- 核心问题: 风格差异大，构图问题不明显

**当前模板问题**:
```
"passport photo composition", "subject floating in center"
```
- 描述不够强烈，SDXL难以理解

**解决方案**:
```yaml
# 核心原则: 强制居中构图，保持风格不变
CRITICAL RULES:
- Keep the EXACT same artistic style
- Force extreme centering through explicit positioning

# 改进后的策略:
1. Use explicit positioning: "with subject placed exactly in the dead center of the image"
2. Add boring composition: "perfectly symmetrical and static composition"
3. Add negative space description: "surrounded by excessive empty space on all sides"
4. Physical centering: "viewed straight-on from directly in front"
```

---

### 3.5 poor_composition_cropped ❌

**当前问题**:
- 两图内容完全不同，裁切问题未体现

**当前模板问题**:
```
"half the subject cut off", "severely cropped at edges"
```
- SDXL难以理解抽象的裁切概念

**解决方案**:
```yaml
# 核心原则: 使用SDXL能理解的物理描述
CRITICAL RULES:
- Keep the EXACT same content and style
- Use concrete physical descriptions instead of abstract "cropping"

# 改进后的策略:
1. Describe partial visibility: "showing only the lower half of [subject]", "with the top portion cut off"
2. Use close-up framing: "extreme close-up showing only part of the scene"
3. Physical obstruction: "partially obscured by frame edge", "extending beyond visible area"
4. Specific cut-offs: "with head cropped at forehead level", "feet not visible in frame"
```

---

### 3.6 style_inconsistency ❌

**当前问题**:
- 两图都是类似的亚洲艺术风格，没有风格混乱

**当前模板问题**:
```
"realistic face, cartoon body, watercolor background, pixel art accessories"
```
- 描述过于具体，LLM可能无法正确应用到任意prompt

**解决方案**:
```yaml
# 核心原则: 强制在同一图像中混合不同渲染风格
CRITICAL RULES:
- Keep the same scene content
- Force explicit style mixing within the image

# 改进后的策略:
1. Mix rendering styles: "with the foreground in photorealistic style and background in watercolor style"
2. Element-specific styles: "subject rendered in 3D style while environment is flat 2D illustration"
3. Texture inconsistency: "mixing smooth digital art textures with rough painterly strokes"
4. Force AI glitch: "with inconsistent rendering quality across different parts"
```

---

### 3.7 unbalanced_layout ❌

**当前问题**:
- 负样本构图反而更对称，布局不平衡未体现

**当前模板问题**:
```
"everything piled on one side", "massive empty void on half the image"
```
- 描述不够具体

**解决方案**:
```yaml
# 核心原则: 强制物理上的不平衡布局
CRITICAL RULES:
- Keep the EXACT same content and style
- Force physical positioning to create imbalance

# 改进后的策略:
1. Explicit positioning: "with all elements crowded into the left third of the image"
2. Empty space: "leaving the entire right side completely empty"
3. Weight imbalance: "all visual weight concentrated in bottom-left corner"
4. Tilted horizon: "with noticeably tilted horizon line"
```

---

### 3.8 dull_palette ⚠️

**当前问题**:
- 颜色暗淡效果不够明显

**当前模板问题**:
```
"slightly muted colors", "subdued palette"
```
- 描述太温和

**解决方案**:
```yaml
# 改进策略: 使用更强烈的颜色描述
1. Severe级别增强: "in completely desaturated muddy grey-brown tones"
2. Add specific color removal: "with all vibrant colors drained to dull grey"
3. Use atmosphere: "in dreary overcast grey color palette"
4. AIGC specific: "with washed-out faded colors like a failed color calibration"
```

---

### 3.9 lighting_imbalance ⚠️

**当前问题**:
- 光线不均匀效果不够强烈

**当前模板问题**:
```
"one side overlit other side dark"
```
- 描述不够极端

**解决方案**:
```yaml
# 改进策略: 使用更极端的光照对比描述
1. Extreme contrast: "with left half in complete darkness and right half blown out bright"
2. Hotspots: "with harsh bright hotspot on one area ruining the exposure"
3. Split lighting: "dramatically split lighting with no gradation"
4. AIGC specific: "with AI-generated lighting inconsistency across the scene"
```

---

## 四、通用改进原则

根据 CLAUDE.md 中的指导：

> "注意生成的负样本，还是需要保证在内容和风格尽量差不多的情况下，有一个明显的相应维度的退化才行，然后还需要考虑的是，你生成出来的负样本，应该尽量接近于生成图像的质量退化，而不是模拟拍摄图像的退化"

### 核心原则

1. **保持风格不变**: 每个模板都必须强调 "Keep the EXACT same artistic style"
2. **AIGC特有退化**: 关注AI生成图像的质量问题，而非传统摄影问题
3. **具体物理描述**: 使用SDXL能理解的具体描述，避免抽象概念
4. **追加而非替换**: 在原prompt后追加退化描述，而非替换风格词

### 推荐模板结构

```yaml
dimension_name:
  severe: |
    # Task
    You are given a text-to-image prompt. Your task is to rewrite this prompt to introduce [specific degradation].

    CRITICAL RULES:
    - Keep the EXACT same artistic style (illustration stays illustration, painting stays painting)
    - DO NOT change the medium or convert between styles
    - DO NOT add "photo", "photograph", "camera" unless the original has them
    - Only ADD degradation descriptors at the end of the prompt

    Strategies:
    1. [Specific strategy with SDXL-friendly keywords]
    2. [Another strategy]
    3. [AIGC-specific degradation if applicable]

    Output only the rewritten prompt.
```

---

## 五、下一步行动

1. **修改模板文件**: 按照上述解决方案修改 `aesthetic_quality.yaml`
2. **重新生成样本**: 使用修改后的模板 + 相同seed策略重新生成
3. **验证效果**: 检查新生成的正负样本是否符合预期
4. **迭代优化**: 根据验证结果继续调整模板

---

## 六、附录：已完成的代码修改

### seed策略修改

**文件**: `demo_v3_dimension_paired.py:479`

```python
# 修改前
neg_seed = pos_seed + (SEVERITIES.index(severity) + 1) * 10000

# 修改后
neg_seed = pos_seed  # 正负样本使用相同seed，确保唯一变量是prompt差异
```

**效果**: 正负样本使用相同seed，消除因seed不同导致的风格/内容差异。
