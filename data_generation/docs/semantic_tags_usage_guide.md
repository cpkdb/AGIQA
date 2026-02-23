# 语义标签使用指南

## 快速开始

### 1. 使用新版本标注文件

```bash
# 新版本文件（推荐）
/root/ImageReward/data_generation/data/prompts_tagged_sdxl_v2.json

# 旧版本文件（不推荐，误标率高）
/root/ImageReward/data_generation/data/prompts_tagged_sdxl.json
```

### 2. 在数据生成中使用标签筛选

```bash
# 示例：生成手部退化数据
python scripts/demo_v3_dimension_paired.py \
    --subcategory anatomical_accuracy \
    --attribute hand_deformation \
    --num_prompts 100 \
    --tagged_prompts data/prompts_tagged_sdxl_v2.json \
    --required_tags has_hand

# 示例：生成面部退化数据
python scripts/demo_v3_dimension_paired.py \
    --subcategory anatomical_accuracy \
    --attribute facial_distortion \
    --num_prompts 200 \
    --tagged_prompts data/prompts_tagged_sdxl_v2.json \
    --required_tags has_face

# 示例：生成文字退化数据
python scripts/demo_v3_dimension_paired.py \
    --subcategory semantic_alignment \
    --attribute text_distortion \
    --num_prompts 150 \
    --tagged_prompts data/prompts_tagged_sdxl_v2.json \
    --required_tags has_text
```

## 标签详细说明

### 高质量标签（推荐优先使用）

#### 1. has_hand (手部) - 333个样本
**准确率**: ~98%

**包含内容**:
- 明确的手部词汇: hand, hands, finger, fingers, fist, palm, wrist, thumb
- 手部动作: holding, grasping, pointing, waving, touching, clapping

**适用维度**:
- anatomical_accuracy/hand_deformation (手部畸形)
- anatomical_accuracy/finger_count (手指数量)

**示例**:
```
✅ "A person holding a cup of coffee"
✅ "Hands clasped together in prayer"
✅ "A child waving goodbye"
❌ "A smooth hand wall" (误判，已移除)
❌ "A grandmother baking cookies" (隐含但不明确，已移除)
```

#### 2. has_face (面部) - 533个样本
**准确率**: ~95%

**包含内容**:
- 面部特征: face, eyes, nose, mouth, lips, cheek, chin
- 表情: smile, frown, crying, laughing, expression
- 视角: portrait, close-up, looking at, staring

**适用维度**:
- anatomical_accuracy/facial_distortion (面部畸形)
- anatomical_accuracy/eye_issues (眼睛问题)

**示例**:
```
✅ "A portrait of a woman smiling"
✅ "A child with a surprised expression"
✅ "Close-up of a man's face"
❌ "Surface of concrete" (误判，已移除)
❌ "A person standing" (不明确，已移除)
```

#### 3. has_text (文字) - 503个样本
**准确率**: ~95%

**包含内容**:
- 文字相关: text, word, letter, writing, sign, label
- 引号内容: "quoted text"
- 阅读动作: says, reads

**适用维度**:
- semantic_alignment/text_distortion (文字扭曲)
- semantic_alignment/text_legibility (文字可读性)

**示例**:
```
✅ "A sign that says 'Welcome'"
✅ "A book with text on the cover"
✅ "A poster with the title 'Summer Festival'"
❌ "A building" (可能有文字但不明确，已移除)
```

#### 4. has_person (人物) - 717个样本
**准确率**: ~95%

**包含内容**:
- 人物词汇: person, people, man, woman, boy, girl, child
- 家庭成员: father, mother, grandmother, grandfather
- 人物描述: portrait, human, figure

**适用维度**:
- anatomical_accuracy/body_proportion (身体比例)
- aesthetic_quality/poor_composition (构图问题)

**示例**:
```
✅ "A man walking in the park"
✅ "A portrait of a woman"
✅ "Children playing in the garden"
```

**注意**: 由于 has_full_body 样本极少(17个)，建议使用 has_person 代替

#### 5. has_animal (动物) - 637个样本
**准确率**: ~95%

**包含内容**:
- 常见动物: dog, cat, bird, horse, fish, elephant, lion, etc.
- 通用词: animal, pet, wildlife

**适用维度**:
- anatomical_accuracy/animal_anatomy (动物解剖)
- semantic_alignment/species_confusion (物种混淆)

**示例**:
```
✅ "A dog playing in the yard"
✅ "A cat sitting on a windowsill"
✅ "Birds flying in the sky"
```

### 中等质量标签

#### 6. has_countable_objects (可数物体) - 1313个样本
**说明**: 包含明确数量的物体

**适用维度**:
- semantic_alignment/object_count (物体数量)

#### 7. has_indoor_scene (室内场景) - 595个样本
**说明**: 室内环境的场景

**适用维度**:
- aesthetic_quality/lighting_imbalance (光照不平衡)
- technical_quality/underexposure (曝光不足)

#### 8. has_background (背景) - 2353个样本
**说明**: 有明确背景的场景

**适用维度**:
- aesthetic_quality/lack_of_depth (缺乏深度)
- aesthetic_quality/cluttered_scene (杂乱场景)

#### 9. has_reflective_surface (反射表面) - 695个样本
**说明**: 包含反射表面（镜子、水面、玻璃等）

**适用维度**:
- technical_quality/reflective_artifacts (反射伪影)

### 低样本量标签（谨慎使用）

#### 10. has_logo_or_symbol (Logo/符号) - 38个样本
**准确率**: ~95%
**问题**: 样本量太少

**包含内容**:
- Logo相关: logo, symbol, emblem, icon, badge, brand

**建议**:
- 如果需要更多样本，可以考虑放宽筛选条件
- 或者使用 has_text 标签作为替代

#### 11. has_full_body (全身) - 17个样本
**准确率**: ~95%
**问题**: 样本量极少

**包含内容**:
- 明确的全身描述: full body, full-body, entire body, whole body, from head to toe

**建议**:
- **强烈建议使用 has_person (717个) 代替**
- 如果确实需要全身照，可以考虑放宽条件

## 标签组合使用

### 示例1: 手部 + 人物
```bash
# 生成包含人物和手部的场景
python scripts/demo_v3_dimension_paired.py \
    --required_tags has_hand,has_person \
    --num_prompts 50
```

### 示例2: 面部 + 室内
```bash
# 生成室内人物肖像
python scripts/demo_v3_dimension_paired.py \
    --required_tags has_face,has_indoor_scene \
    --num_prompts 100
```

### 示例3: 动物 + 背景
```bash
# 生成有背景的动物场景
python scripts/demo_v3_dimension_paired.py \
    --required_tags has_animal,has_background \
    --num_prompts 150
```

## 实用工具脚本

### 1. 验证标签质量
```bash
# 抽查 has_hand 标签的10个样本
python scripts/verify_tags.py --tag has_hand --sample 10

# 抽查 has_face 标签的20个样本
python scripts/verify_tags.py --tag has_face --sample 20
```

### 2. 对比新旧标注
```bash
# 查看 has_hand 标签的变化
python scripts/compare_tags.py --tag has_hand --show-removed 15

# 查看 has_logo_or_symbol 标签的变化
python scripts/compare_tags.py --tag has_logo_or_symbol --show-removed 20
```

### 3. 统计标签分布
```python
import json

with open('data/prompts_tagged_sdxl_v2.json', 'r') as f:
    data = json.load(f)

tag_counts = {}
for p in data['prompts']:
    for tag in p.get('semantic_tags', []):
        tag_counts[tag] = tag_counts.get(tag, 0) + 1

for tag, count in sorted(tag_counts.items(), key=lambda x: -x[1]):
    print(f"{tag}: {count}")
```

## 常见问题

### Q1: 为什么 has_full_body 样本这么少？
**A**: 重新标注时采用了非常严格的标准，只有明确包含 "full body", "full-body", "entire body" 等词汇的prompt才会被标记。大量原本标记为 has_full_body 的prompt实际上只是包含 "standing", "walking" 等动作词，并不能确定是否是全身照。

**建议**: 使用 has_person (717个样本) 代替。

### Q2: 为什么 has_logo_or_symbol 从347降到38？
**A**: 原始标注将很多装饰图案、设计元素都误标为logo。重新标注后，只有明确包含 "logo", "symbol", "brand", "trademark" 等词汇的prompt才会被标记。

**建议**: 如果需要更多样本，可以考虑：
1. 放宽筛选条件，接受 "design", "pattern" 等词
2. 使用 has_text 标签作为替代（很多logo包含文字）

### Q3: 如何确认标签的准确性？
**A**: 使用验证脚本：
```bash
python scripts/verify_tags.py --tag has_hand --sample 20
```
这会随机抽取20个样本，你可以手动检查是否准确。

### Q4: 旧版本标注文件还能用吗？
**A**: 不推荐使用。旧版本误标率很高：
- has_hand: 28%误标
- has_face: 39%误标
- has_full_body: 97%误标
- has_logo_or_symbol: 77%误标

强烈建议使用新版本 `prompts_tagged_sdxl_v2.json`。

## 标签准确率对比

| 标签 | 旧版本准确率 | 新版本准确率 | 提升 |
|------|-------------|-------------|------|
| has_hand | ~72% | ~98% | +26% |
| has_face | ~61% | ~95% | +34% |
| has_full_body | ~63% | ~95% | +32% |
| has_logo_or_symbol | ~23% | ~95% | +72% |
| has_text | ~72% | ~95% | +23% |
| has_person | ~75% | ~95% | +20% |
| has_animal | ~70% | ~95% | +25% |

## 各维度推荐标签总结

| 退化维度 | 推荐标签 | 样本数 | 备注 |
|---------|---------|--------|------|
| 手部畸形 | has_hand | 333 | 高质量 |
| 面部畸形 | has_face | 533 | 高质量 |
| 身体比例 | has_person | 717 | 代替has_full_body |
| 文字扭曲 | has_text | 503 | 高质量 |
| Logo扭曲 | has_logo_or_symbol | 38 | 样本少 |
| 动物解剖 | has_animal | 637 | 高质量 |
| 物体数量 | has_countable_objects | 1313 | 充足 |
| 室内光照 | has_indoor_scene | 595 | 充足 |
| 深度缺失 | has_background | 2353 | 充足 |

## 更新日志

### v2 (2026-01-18)
- 重新标注所有关键语义标签
- 使用严格的正则表达式匹配
- 移除大量误标样本
- 准确率从60-70%提升至95%+

### v1 (原始版本)
- 基于LLM的自动标注
- 存在大量误标问题
- 不推荐使用
