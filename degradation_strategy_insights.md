# Prompt退化体系的全面启示与策略升级
*基于 AAS Distillation 数据集分析的综合报告 (修订版：单一变量原则)*

本文档基于对 AAS Distillation 数据集（800+ 高效退化样本）的深度分析，结合 `/root/ImageReward/data_generation/config/quality_dimensions_active.json` 中定义的完整质量体系，总结出的核心方法论升级与全维度策略建议。

**重要修订**：根据单一变量控制原则，退化策略必须在极大化“病态特征”的同时，**严格避免跨维度污染**（例如：颜色退化不应导致画面模糊）。

## 1. 核心方法论升级

### A. 从“单一形容词”到“病态特征描述” (Pathological Features)
SDXL 等模型对抽象评价词（如 "bad quality"）不敏感。有效的退化必须描述**视觉上的病态特征**。
*   **旧策略**: "A boring image."
*   **新策略**: "A flat, textureless image with distinct jpeg compression artifacts and jagged edges."

### B. 单一变量精准打击 (Single Variable Precision)
**拒绝协同退化**。每个维度的退化必须是“外科手术式”的，只破坏目标属性，保护其他质量维度和主体内容。
*   **错误示例**: 为了做 "Amateur Look" 而添加 `blurry` 和 `noise`（这会污染 Technical Quality）。
*   **正确策略**: "Amateur Look" 应专注于操作失误（如 `bad framing`, `flash glare`, `flat lighting`），这才是审美层面的业余。

### C. 显著性降低 (Saliency Reduction)
通过降低主体的视觉显著性来造成感官上的退化，特别适用于构图类和主体完整性维度。
*   **手段**: 使用 `hard to notice` (难以注意), `lost in the background` (迷失在背景中), `tiny and peripheral` (极小且边缘化)。

---

## 2. Technical Quality (技术质量) 维度启示

*目标：低层视觉信号退化。必须确保不改变物体结构。*

| 子维度 | 关键“破坏性”词汇与策略 (单一变量原则) |
| :--- | :--- |
| **Blur (模糊)** | **策略**: 仅针对对焦和清晰度，保留颜色和光照。<br>**词汇**: `out of focus`, `motion smear` (运动拖影), `indistinct edges`, `soft focus`. |
| **Overexposure (过曝)** | **策略**: 仅增加亮度至信息丢失，不改变色相。<br>**词汇**: `blown out highlights`, `blinding glare`, `washed out whites`, `detail loss in bright areas`. |
| **Underexposure (欠曝)** | **策略**: 仅降低亮度至暗部死黑。<br>**词汇**: `crushed shadows`, `swallowed by darkness`, `insufficient dynamic range`, `muddy blacks`. |
| **Low Contrast (低对比)** | **策略**: 压缩直方图，不改变平均亮度。<br>**词汇**: `uniform grey wash`, `muddy midtones`, `no tonal separation`, `flat histogram`. |
| **Color Cast (偏色)** | **策略**: 叠加滤镜，但不改变饱和度（除非是 desaturation）。<br>**词汇**: `sickly green tint`, `broken white balance`, `heavy monochromatic filter`. |
| **Plastic/Waxy (塑料感)** | **策略**: 仅平滑纹理，不改变物体形状。<br>**词汇**: `rubber-like skin`, `melted plastic texture`, `over-smoothed`, `loss of pores`. |

---

## 3. Aesthetic Quality (美学质量) 维度启示

*目标：审美与艺术表达退化。严禁引入 Technical 缺陷（如 blur/noise）。*

| 子维度 | 关键“破坏性”词汇与策略 |
| :--- | :--- |
| **Composition (构图)**<br>*(Centered/Cropped/Unbalanced)* | **策略**: 仅改变构图框位置，**严禁切割或改变物体本身的完整性**（除非是 content truncation）。<br>**词汇**: `tiny subject in far corner`, `90% empty space`, `subject touching edge`, `awkwardly framed`. |
| **Lighting (美学光影)**<br>*(Flat/Imbalance)* | **策略**: 改变光线分布。Flat = 无立体感；Imbalance = 光比失控。<br>**词汇**: `2D cutout look`, `paper-flat`, `flash bulb glare`, `erratic hot spots`. |
| **Color Harmony (色彩)**<br>*(Dull/Clash/Eclectic)* | **策略**: 仅攻击色相搭配和饱和度。Dull = 无生气；Clash = 不和谐。<br>**词汇**: `muddy olive and sickly yellow` (Dull), `violent color clash`, `neon vs earth tones` (Clash). |
| **Style (风格)**<br>*(Amateur/Inconsistency)* | **策略**: 模拟**人为操作失误**而非设备故障。<br>**词汇**: `on-camera flash`, `snapshot aesthetic`, `unflattering angle`, `messy background`. (避免使用 `low res` 或 `blurry`) |

---

## 4. Semantic Rationality (语义合理性) 维度启示

*目标：语义理解与物理逻辑错误。通过视觉描述诱导模型生成错误结构。*

| 子维度 | AAS 启示与新策略 |
| :--- | :--- |
| **Object Integrity**<br>*(Deformation, Fusion)* | **策略**: 描述物理状态的不稳定。<br>**词汇**: `melting`, `liquid edges`, `shapeless blobs`, `fused together`. |
| **Anatomy (解剖与手)**<br>*(Hands, Face, Limbs)* | **策略**: 描述结构的扭曲，而非画质的降低。<br>**词汇**: `mangled limbs`, `fused fingers`, `distorted facial features`, `anatomically incorrect`. |
| **Spatial Logic**<br>*(Perspective, Floating)* | **策略**: 描述几何关系的错误。<br>**词汇**: `impossible geometry`, `Escher-like perspective`, `floating with no gravity`, `mismatched scales`. |
| **Context Consistency**<br>*(Context, Time)* | **策略**: 强制并置不兼容的元素。<br>**词汇**: `randomly assembled`, `dream-like incoherence`, `nonsensical placement`. |

---

## 5. 全局行动指南

1.  **Isolation Check (隔离检查)**: 在生成任何 Prompt 之前，反问：“这个词会引起其他维度的下降吗？”
    *   例如：想做 `Dull Palette`，如果要用 `faded`，必须确保它不会被模型理解为 `old photo` 从而加上划痕和噪点。建议使用更纯粹的颜色描述如 `desaturated colors`。

2.  **Visual Rewrite (视觉重写)**: 依然推荐将形容词嵌入主体描述中，但必须精准。
    *   *Good*: "A **flatly lit** cat." (Amateur Look)
    *   *Bad*: "A **low quality** cat." (可能引发 Blur/Noise)

3.  **Content Preservation (内容保护)**:
    在 Semantic 以外的维度，任何导致物体变成另一个物体（如猫变成狗，或车变成烂铁）的 Prompt 都是失败的。必须保持 Semantic ID 不变。

此文档为最终指导方针，直接指导 `prompt_templates_v3` 的修改。
