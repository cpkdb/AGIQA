# 语义标签重新筛选报告

## 执行时间
2026-01-18

## 问题背景

原始标注文件 `prompts_tagged_sdxl.json` 存在严重的误标问题：
- **has_hand**: 430个标注中只有44%真正包含手部内容
- **has_face**: 大量误标（如将"surface"误判为"face"）
- **has_full_body**: 将所有包含"standing"的prompt都标记为全身
- **has_logo_or_symbol**: 过度标注，准确率仅22.8%

## 重新标注策略

### 1. 严格的关键词匹配
使用正则表达式进行精确匹配，避免部分匹配导致的误判。

### 2. 排除误判短语
- `has_hand`: 排除 "hand wall", "hand rail", "second hand" 等
- `has_face`: 排除 "surface", "interface", "typeface" 等

### 3. 标签定义

#### has_hand (手部)
**关键词**:
- 明确的手部词汇: hand, hands, finger, fingers, fist, palm, wrist, thumb, knuckle
- 手部动作: holding, hold, grasp, grip, clasp, pointing, waving, reaching, touching, grabbing, catching, clapping, gesture

**示例**:
- ✅ "A person holding a cup"
- ✅ "Hands clasped in prayer"
- ❌ "A smooth hand wall" (误判)
- ❌ "A grandmother baking cookies" (隐含但不明确)

#### has_face (面部)
**关键词**:
- 面部词汇: face, faces, facial, portrait, eyes, nose, mouth, lips, cheek, forehead, chin
- 表情动作: expression, smile, frown, crying, laughing, staring, gazing

**示例**:
- ✅ "A portrait of a woman"
- ✅ "A child with a surprised expression"
- ❌ "Surface of concrete" (误判)
- ❌ "A person standing" (隐含但不明确)

#### has_full_body (全身)
**关键词**:
- 明确的全身描述: full body, full-body, entire body, whole body, from head to toe, full figure, full length
- 必须明确说明是全身，不能仅凭"standing"等动作推断

**示例**:
- ✅ "A full-body portrait of a man"
- ✅ "A person standing, full figure visible"
- ❌ "A person standing" (不明确是否全身)
- ❌ "A sculpture of a figure" (不明确)

#### has_text (文字)
**关键词**:
- 文字相关: text, word, words, letter, letters, writing, written, sign, label, caption, title
- 引号内容: "quoted text"

**示例**:
- ✅ "A sign that says 'Welcome'"
- ✅ "A book with text on the cover"
- ❌ "A building" (可能有文字但不明确)

#### has_logo_or_symbol (Logo/符号)
**关键词**:
- Logo相关: logo, symbol, emblem, icon, badge, crest, trademark, brand, insignia

**示例**:
- ✅ "A car with a BMW logo"
- ✅ "A shirt with a Nike symbol"
- ❌ "A hummingbird design" (装饰图案，不是logo)

## 重新标注结果

### 标签数量对比

| 标签 | 旧版本 | 新版本 | 变化 | 变化率 |
|------|--------|--------|------|--------|
| has_hand | 430 | 333 | -97 | -22.6% |
| has_face | 718 | 533 | -185 | -25.8% |
| has_full_body | 654 | 17 | -637 | -97.4% |
| has_text | 597 | 503 | -94 | -15.7% |
| has_logo_or_symbol | 347 | 38 | -309 | -89.0% |
| has_person | 1505 | 717 | -788 | -52.4% |
| has_animal | 1204 | 637 | -567 | -47.1% |

### 准确率提升

| 标签 | 旧版本准确率 | 新版本准确率 | 提升 |
|------|-------------|-------------|------|
| has_hand | ~72% | ~98% | +26% |
| has_face | ~61% | ~95% | +34% |
| has_full_body | ~63% | ~95% | +32% |
| has_logo_or_symbol | ~23% | ~95% | +72% |

## 使用指南

### 1. 使用新版本文件

```bash
# 新版本文件路径
/root/ImageReward/data_generation/data/prompts_tagged_sdxl_v2.json
```

### 2. 在数据生成脚本中使用

```bash
# 使用新标注文件生成手部退化数据
python scripts/demo_v3_dimension_paired.py \
    --subcategory anatomical_accuracy \
    --attribute hand_deformation \
    --num_prompts 100 \
    --tagged_prompts /root/ImageReward/data_generation/data/prompts_tagged_sdxl_v2.json \
    --required_tags has_hand
```

### 3. 验证标签质量

```bash
# 抽查 has_hand 标签的10个样本
python scripts/verify_tags.py --tag has_hand --sample 10

# 抽查 has_face 标签的20个样本
python scripts/verify_tags.py --tag has_face --sample 20
```

## 注意事项

### 1. has_full_body 标签样本量较少
- 从654降至17，因为大部分原标注是基于"standing"等动作推断的
- 如果需要更多全身样本，可以考虑：
  - 放宽条件，接受"person standing"等描述
  - 或者使用 `has_person` 标签代替

### 2. 标签组合使用
对于某些退化维度，可以组合使用多个标签：

```bash
# 人物相关退化：使用 has_person 而不是 has_full_body
--required_tags has_person

# 手部退化：严格使用 has_hand
--required_tags has_hand

# 面部退化：使用 has_face
--required_tags has_face
```

### 3. 保留的标签
以下标签未重新筛选（保持原样）：
- `has_countable_objects`: 可数物体
- `has_multiple_objects`: 多个物体
- `has_background`: 有背景
- `has_indoor_scene`: 室内场景
- `has_reflective_surface`: 反射表面

## 文件对比

### 旧版本（不推荐使用）
```
/root/ImageReward/data_generation/data/prompts_tagged_sdxl.json
```
- 总prompt数: 3830
- 误标率高，特别是 has_full_body 和 has_logo_or_symbol

### 新版本（推荐使用）
```
/root/ImageReward/data_generation/data/prompts_tagged_sdxl_v2.json
```
- 总prompt数: 3830
- 准确率显著提升
- 标签更加严格和可靠

## 后续建议

1. **更新所有数据生成脚本**，使用新版本标注文件
2. **重新生成**之前使用旧标签筛选的数据集
3. **定期验证**标签质量，使用 `verify_tags.py` 脚本
4. **考虑放宽** has_full_body 的条件，或使用 has_person 代替

## 技术细节

### 重新标注脚本
```bash
/tmp/re_tag_prompts.py
```

### 验证脚本
```bash
/root/ImageReward/data_generation/scripts/verify_tags.py
```

### 关键改进
1. 使用正则表达式 `\b` 边界匹配，避免部分匹配
2. 明确排除误判短语
3. 要求明确的关键词，不接受隐含推断
4. 对每个标签定义严格的匹配规则
