# AIGC图像质量退化维度体系（三视角统一分类）

## 概述

本分类体系融合了**研究预设维度**（`config/quality_dimensions.json`，5类23属性）与**500张图像实证归纳**（2332个失真实例，8个维度），按照三个主视角组织：

- **Technical Quality（技术质量）**：低层视觉信号退化，可在不理解语义的情况下判断
- **Aesthetic Quality（美学）**：构图、光影、色彩风格等主观审美因素
- **Semantic/Rationality（语义/合理性）**：解剖结构、物体属性、空间关系、物理光学、场景上下文等高层一致性

**配置文件**：`config/quality_dimensions_3perspectives.json`

**统计数据**：
- 3个主视角
- 16个维度组（dimension groups）
- 168个具体属性（attributes）
- 完整保留研究v1的23个属性
- 覆盖实证数据的8个维度和289种失真类型

---

## 重叠与缺口分析

### 研究v1与实证数据的重叠

| 维度类型 | 研究v1 | 实证数据 | 结论 |
|---------|--------|---------|------|
| 解剖/人体 | anatomical_accuracy | Anatomical_Defects (39类型) | 核心一致，实证数据更细粒度 |
| 纹理细节 | texture_detail | Texture_Issues (95类型) | 核心一致，实证揭示大量材质实例 |
| 光照 | exposure_issues + poor_lighting | Lighting_Issues (36类型) | **跨视角混合**，需拆分 |
| 颜色 | 3个分散属性 | Color_Issues (19类型) | **跨视角混合**，需拆分 |
| 结构/语义 | structural_plausibility | 3个独立维度 | 实证数据更细分 |

### 研究v1的关键缺口（已补充）

实证数据暴露的研究v1未覆盖的内容：

1. **文本与符号语义**（`text_symbol_semantics`）
   - 实证：Text Legibility, Illegible Text, Iconography Error等
   - v2新增：8个文本相关属性

2. **背景合成与融合**（`compositing_background`）
   - 实证：Background Blurring, Blending, Depth Issues等
   - v2新增：10个合成相关属性

3. **噪声与压缩**（`noise_compression`）
   - 研究v1缺少专门的噪声/压缩维度
   - v2新增：8个技术退化属性

4. **物体位置关系细项**（`spatial_geometry_positioning`）
   - 实证：对齐、摆放、朝向等细粒度问题
   - v2从perspective_error扩展为12个属性

5. **关键要素缺失**（`missing_objects`）
   - 研究v1有hallucination（无中生有）但无missing（该有却没有）
   - v2在object_integrity中显式添加

---

## 最终Taxonomy结构

### 三视角组织原则

**1. Technical Quality（技术质量，权重0.20）**
- 6个维度组，60个属性
- 特征：可通过像素级分析判断，无需理解语义
- 示例：blur, noise, color_banding, texture_artifacts

**2. Aesthetic Quality（美学，权重0.30）**
- 4个维度组，35个属性
- 特征：主观审美判断，即便语义正确也可能评价较差
- 示例：poor_composition, flat_lighting, unharmonious_colors

**3. Semantic/Rationality（语义/合理性，权重0.50）**
- 6个维度组，73个属性
- 特征：涉及对象理解、物理规律、空间逻辑、上下文一致性
- 示例：hand_deformity, floating_objects, contextual_mismatch

### 16个维度组清单

#### Technical Quality
1. `clarity_resolution` - 清晰度与分辨率（10属性）
2. `noise_compression` - 噪声与压缩伪影（8属性）
3. `exposure_contrast` - 曝光与对比（10属性）
4. `color_tone_fidelity` - 色彩与色调保真（10属性）
5. `texture_detail_rendering` - 纹理与细节渲染（12属性）
6. `compositing_background` - 合成与背景渲染（10属性）

#### Aesthetic Quality
7. `composition_framing` - 构图与取景（10属性）
8. `lighting_atmosphere` - 光影与氛围（8属性）
9. `color_harmony_style` - 色彩协调与风格（9属性）
10. `overall_visual_appeal` - 整体观感与吸引力（8属性）

#### Semantic/Rationality
11. `anatomy_biology` - 解剖与生物合理性（14属性）
12. `object_integrity_attributes` - 物体完整性与属性一致（14属性）
13. `spatial_geometry_positioning` - 空间几何与位置关系（12属性）
14. `physical_plausibility_optics` - 物理规律与光学一致性（11属性）
15. `scene_context_consistency` - 场景与上下文一致性（14属性）
16. `text_symbol_semantics` - 文本与符号语义（8属性）

---

## 跨视角问题的处理

某些失真类型横跨多个视角，需按根因分类：

### Lighting的三视角拆分

| 视角 | 维度组 | 典型属性 | 判断依据 |
|-----|--------|---------|---------|
| Technical | exposure_contrast | overexposure, low_contrast | 曝光/动态范围问题 |
| Aesthetic | lighting_atmosphere | flat_lighting, harsh_lighting | 氛围/塑形/美感问题 |
| Semantic | physical_plausibility_optics | shadow_mismatch, inconsistent_light_source | 光源/阴影/反射逻辑矛盾 |

### Color的三视角拆分

| 视角 | 维度组 | 典型属性 | 判断依据 |
|-----|--------|---------|---------|
| Technical | color_tone_fidelity | white_balance_shift, color_bleeding | 白平衡/色带/渗色等成像保真 |
| Aesthetic | color_harmony_style | unharmonious_colors, poor_color_grading | 配色/调色/风格统一性 |
| Semantic | object_integrity | illogical_colors | 违背常识的物体颜色 |

### Background的双视角拆分

| 视角 | 维度组 | 典型属性 | 判断依据 |
|-----|--------|---------|---------|
| Technical | compositing_background | background_blur_artifacts, blending_seams | 虚化/融合/接缝/噪点等渲染问题 |
| Semantic | scene_context_consistency | background_discrepancy | 背景语义与主体/提示词冲突 |

---

## 映射策略

### 研究v1 → v2映射

所有23个研究v1属性完整保留并扩展：

```
研究v1                                  → v2
─────────────────────────────────────────────
technical_quality.blur                  → clarity_resolution.blur (扩展为4个子类型)
technical_quality.exposure_issues       → exposure_contrast.exposure_issues (扩展为5个子类型)
texture_detail.over_smoothing           → texture_detail_rendering.over_smoothing (扩展为3个子类型)
aesthetic_quality.poor_composition      → composition_framing.poor_composition (扩展为9个子类型)
structural_plausibility.object_deformation → object_integrity_attributes.object_deformation (扩展为8个子类型)
anatomical_accuracy.hand_deformity      → anatomy_biology.hand_deformity (扩展为4个子类型)
... (完整映射见JSON的mapping字段)
```

### 实证8维度 → v2映射

```
实证维度                    → v2主要归属
─────────────────────────────────────────────
Anatomical_Defects          → anatomy_biology
Texture_Issues              → texture_detail_rendering
Object_Positioning_Errors   → spatial_geometry_positioning
Semantic_Inconsistencies    → scene_context_consistency

跨视角维度（多重映射）：
Lighting_Issues             → exposure_contrast + lighting_atmosphere + physical_plausibility_optics
Color_Issues                → color_tone_fidelity + color_harmony_style + object_integrity
Background_Inconsistencies  → compositing_background + scene_context_consistency
Unique_Issues               → text_symbol_semantics + compositing_background
```

---

## 使用指南

### 1. 标注失真类型

```python
# 示例：标注一个失真
distortion = {
    "primary": "semantic_rationality.anatomy_biology.hand_deformity",
    "secondary": ["semantic_rationality.anatomy_biology.finger_count_error"],
    "severity": "severe",
    "description": "手部出现6根手指，指间分离不清"
}
```

**标注规则**：
- 允许多标签，但需指定primary（根因）
- 跨视角问题优先选择根因视角
- severity: mild/moderate/severe

### 2. 生成退化数据

基于新taxonomy生成对比数据集：

```python
from llm_prompt_degradation import LLMPromptDegradation

degrader = LLMPromptDegradation(
    quality_dimensions_path="config/quality_dimensions_3perspectives.json"
)

# 指定维度组和属性
negative_prompt = degrader.degrade_prompt(
    original_prompt="A woman in a red dress",
    perspective="semantic_rationality",
    dimension="anatomy_biology",
    attribute="hand_deformity",
    severity="severe"
)
```

### 3. 评测模型能力

按三视角分别评测AIGC模型：

```python
results = {
    "technical_quality": evaluate_technical(model, test_set),
    "aesthetic_quality": evaluate_aesthetic(model, test_set),
    "semantic_rationality": evaluate_semantic(model, test_set)
}
# 加权得分 = 0.20*technical + 0.30*aesthetic + 0.50*semantic
```

### 4. 扩展新属性

在维度组下添加新属性：

```json
{
  "perspectives": {
    "technical_quality": {
      "dimensions": {
        "clarity_resolution": {
          "attributes": {
            "new_artifact_type": {
              "desc": "新发现的伪影类型",
              "empirical": "Empirical Type Name"
            }
          }
        }
      }
    }
  }
}
```

---

## 设计理由

### 三视角强约束的优势

1. **可解释性**：技术质量（可测）/美学（主观）/语义合理性（理解）对应不同研究问题
2. **解决跨类混杂**：实证数据的Lighting/Color/Background混合现象得到清晰拆分
3. **评测适配**：不同视角对应不同评测方法（技术指标/人工审美/语义理解）

### 层级可扩展

- 大量实证的对象化条目（如"Mouse Ear Texture"）作为示例/别名映射到leaf attribute
- 不会导致taxonomy膨胀失控
- 便于后续添加新维度组和属性

### 兼容性与扩展性

- 完整保留研究v1的23个属性作为锚点
- 扩展到168个属性，覆盖实证数据的289种失真类型
- 适配prompt退化模板、合成策略与评测指标的设计

---

## 后续工作

1. **更新prompt模板**：为新增的维度组创建`config/prompt_templates/`下的YAML模板
2. **批量映射实证数据**：将289种失真类型通过规则映射到168个属性
3. **评测baseline**：在新taxonomy上评测现有AIGC模型的各视角能力
4. **人工验证**：对自动归纳的维度进行人工审核和细化

---

## 参考

- 研究v1配置：`config/quality_dimensions.json`
- 实证数据：`/root/autodl-tmp/aigc_distortion_analysis/runs/dataset_500/quality_dimensions.json`
- Prompt模板：`config/prompt_templates/*.yaml`
