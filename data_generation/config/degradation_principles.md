# 质量退化原则与实施细则

本文档定义了AIGC图像质量评估数据集构建中负样本prompt生成的核心原则和实施细则。

## 1. 核心原则

### 1.1 修改方式限制

**必须遵守**:
- ✅ **仅基于替换或添加形容词/修饰语**
  - 添加退化描述词（如：blurry, noisy, poor lighting）
  - 替换属性词（如：red → blue, running → walking）
  - 移除质量提升词（如：masterpiece, high quality, detailed）

**严格禁止**:
- ❌ 删除或修改prompt的主体对象
- ❌ 改变prompt的核心语义结构
- ❌ 引入与原始内容无关的新对象
- ❌ 改变场景的基本设定

**示例**:
```
✅ 正确:
原始: "a beautiful sunset over the ocean"
退化: "a beautiful sunset over the ocean, blurry, out of focus"

❌ 错误:
原始: "a beautiful sunset over the ocean"
退化: "a mountain landscape"  ← 改变了主体场景
```

### 1.2 随机性与多样性

**随机选择策略**:
- 每次生成时从关键词库中**随机采样**，而非使用固定例子
- 支持**单一关键词**或**组合多个关键词**（50%概率各一）
- 退化关键词的**插入位置**随机化（70%末尾，30%开头）

**组合策略**:
- 单一关键词（50%概率）：从severity_keywords中随机选择1个
- 组合关键词（50%概率）：从severity_keywords中随机选择2-3个

**示例**:
```python
# 单一关键词
"a portrait, blurry"

# 组合关键词（增强退化效果）
"a portrait, blurry, out of focus, soft"
```

### 1.3 维度隔离

**隔离原则**:
- 仅对**指定的质量维度**进行退化
- 其他未指定的质量维度**保持不受影响**
- 避免跨维度的退化叠加

**示例**:
```
退化维度: blur (模糊)
原始: "a red car on the street, high quality"
退化: "a red car on the street, blurry"
     ↑ 保持颜色red不变
     ↑ 仅添加模糊描述
```

### 1.4 自然性保证

**语言要求**:
- 生成后的prompt在**语言上自然流畅**
- 语法结构完整，符合英语表达习惯
- 退化描述与原prompt和谐融合
- 逗号、连词使用正确

**检查标准**:
- [ ] 语法正确，无明显错误
- [ ] 读起来流畅自然
- [ ] 退化词与原prompt不冲突
- [ ] 逗号分隔恰当

## 2. 退化程度分布

退化程度分为三个级别，遵循以下分布比例：

| 程度 | 英文 | 描述 | 占比 | 关键词示例 |
|------|------|------|------|-----------|
| 轻微 | mild | 需要仔细观察才能发现的质量差异 | **20%** | slightly blurry, minor noise, subtle grain |
| 中等 | moderate | 明显可见但不至于完全破坏图像 | **40%** | noticeable blur, visible noise, poor lighting |
| 严重 | severe | 显著的质量问题，容易区分 | **40%** | extremely blurry, heavy noise, terrible lighting |

**选择方式**:
```python
severities = ["mild", "moderate", "severe"]
weights = [0.2, 0.4, 0.4]
severity = random.choices(severities, weights=weights, k=1)[0]
```

## 3. 具体实施策略

### 3.1 视觉质量退化

#### 3.1.1 单一关键词策略（50%概率）

**流程**:
1. 从 `quality_dimensions.json` 获取对应severity的关键词列表
2. 随机选择**1个**关键词
3. 插入到prompt中（70%末尾，30%开头）

**代码示例**:
```python
severity_keywords = ["slightly blurry", "minor blur", "soft focus"]
degradation_text = random.choice(severity_keywords)
# 结果: "minor blur"
```

**生成示例**:
```
原始: "a beautiful mountain landscape, masterpiece, high quality"
处理: 移除"masterpiece, high quality"
退化: "a beautiful mountain landscape, minor blur"
```

#### 3.1.2 组合关键词策略（50%概率）

**流程**:
1. 从关键词列表中随机选择**2-3个**关键词
2. 用逗号连接
3. 插入到prompt中

**代码示例**:
```python
severity_keywords = ["blurry", "out of focus", "soft", "unfocused"]
num_keywords = random.randint(2, 3)
selected = random.sample(severity_keywords, num_keywords)
degradation_text = ", ".join(selected)
# 结果: "blurry, out of focus, soft"
```

**生成示例**:
```
原始: "a beautiful mountain landscape, masterpiece, high quality"
退化: "a beautiful mountain landscape, blurry, out of focus, soft"
```

#### 3.1.3 插入位置策略

**末尾添加（70%概率）**:
```python
negative_prompt = f"{cleaned_prompt}, {degradation_text}"
# "a sunset, blurry, out of focus"
```

**开头添加（30%概率）**:
```python
negative_prompt = f"{degradation_text}, {cleaned_prompt}"
# "blurry, out of focus, a sunset"
```

### 3.2 对齐度退化

#### 3.2.1 替换策略

**原则**:
- 识别prompt中的目标元素（颜色、对象、属性等）
- 从相同类别中随机选择替换项
- 确保替换后语义连贯

**示例**:

| 退化类型 | 原始元素 | 替换元素 | 示例 |
|---------|---------|---------|------|
| 颜色替换 | red | blue | "a red car" → "a blue car" |
| 对象替换 | dog | cat | "a dog playing" → "a cat playing" |
| 动作替换 | running | walking | "person running" → "person walking" |
| 状态替换 | blooming | wilted | "blooming flower" → "wilted flower" |

**注意事项**:
- 替换时保持语法一致（单复数、时态等）
- 确保替换后的元素在语义上合理
- 严重程度通过替换元素的差异度控制

#### 3.2.2 数量修改策略

**示例**:
```
mild:     "three cats" → "four cats"      (数量接近)
moderate: "three cats" → "two cats"       (数量明显不同)
severe:   "three cats" → "one cat"        (数量差异极大)
```

#### 3.2.3 空间关系替换

**示例**:
```
原始: "cat on the left of dog"

mild:     "cat on the far left of dog"        (微调位置)
moderate: "cat in the center with dog"        (改变关系)
severe:   "cat on the right of dog"           (完全反转)
```

## 4. 质量控制检查清单

生成的负样本prompt**必须满足**以下所有条件：

### 4.1 结构检查
- [ ] 语法结构完整，无明显语法错误
- [ ] 逗号使用正确，分隔恰当
- [ ] 没有多余的空格或符号

### 4.2 内容检查
- [ ] **仅修改了指定的退化维度**
- [ ] 未破坏原prompt的核心语义
- [ ] 主体对象保持不变（视觉质量退化时）
- [ ] 退化描述与原prompt不冲突

### 4.3 退化效果检查
- [ ] 退化程度符合severity要求
- [ ] 退化方式符合modification_type定义
- [ ] 退化描述清晰明确

### 4.4 自然性检查
- [ ] prompt读起来自然流畅
- [ ] 退化描述与原prompt和谐融合
- [ ] 没有重复或冗余的描述

## 5. 完整示例

### 示例1: 视觉质量退化 - blur

**正样本prompt**:
```
"a beautiful sunset over the ocean, masterpiece, best quality, high resolution, detailed"
```

**处理步骤**:
1. 移除质量提升词: "a beautiful sunset over the ocean"
2. 选择退化类型: blur, severity=moderate
3. 获取关键词列表: ["noticeable blur", "out of focus", "blurred"]
4. 随机决定组合策略: 组合（50%概率命中）
5. 随机选择2-3个关键词: ["noticeable blur", "out of focus"]
6. 组合: "noticeable blur, out of focus"
7. 随机决定插入位置: 末尾（70%概率命中）

**负样本prompt (moderate)**:
```
"a beautiful sunset over the ocean, noticeable blur, out of focus"
```

**负样本prompt (severe, 组合3个关键词，开头插入)**:
```
"extremely blurry, heavily blurred, very out of focus, a beautiful sunset over the ocean"
```

### 示例2: 对齐度退化 - 颜色替换

**正样本prompt**:
```
"a red sports car on a highway, masterpiece, best quality"
```

**处理步骤**:
1. 识别目标元素: "red" (颜色)
2. 选择severity: moderate
3. 查找替换选项: "orange" (相近颜色，moderate难度)
4. 执行替换: "red" → "orange"

**负样本prompt**:
```
"a orange sports car on a highway, masterpiece, best quality"
```

### 示例3: 视觉质量退化 - poor_lighting

**正样本prompt**:
```
"portrait of a woman in the garden, professional photography, sharp focus"
```

**处理步骤**:
1. 移除质量提升词: "portrait of a woman in the garden"
2. 退化类型: poor_lighting, severity=severe
3. 关键词列表: ["terrible lighting", "harsh shadows, inconsistent lighting", "very poor lighting"]
4. 单一关键词策略: 选择 "harsh shadows, inconsistent lighting"
5. 末尾插入

**负样本prompt**:
```
"portrait of a woman in the garden, harsh shadows, inconsistent lighting"
```

### 示例4: 对齐度退化 - 动作替换

**正样本prompt**:
```
"a happy dog running in the park, high quality, detailed"
```

**处理步骤**:
1. 识别目标元素: "running" (动作)
2. Severity: severe
3. 替换选项: "sitting" (完全相反的动作)
4. 执行替换

**负样本prompt**:
```
"a happy dog sitting in the park, high quality, detailed"
```

## 6. 常见错误与避免方法

### 错误1: 跨维度退化

**错误示例**:
```
原始: "a red car"
退化维度: blur
错误: "a blue car, blurry"  ← 同时改了颜色和模糊度
```

**正确做法**:
```
"a red car, blurry"  ← 仅添加模糊描述
```

### 错误2: 破坏主体对象

**错误示例**:
```
原始: "a dog playing with a ball"
退化维度: visual_quality
错误: "a cat playing, blurry"  ← 改变了主体对象
```

**正确做法**:
```
"a dog playing with a ball, blurry"
```

### 错误3: 语法不自然

**错误示例**:
```
"a sunset, blurry, noisy, poor lighting, terrible"  ← 过多堆砌
```

**正确做法**:
```
"a sunset, blurry, noisy"  ← 适度组合2-3个
```

### 错误4: 使用固定例子

**错误示例**:
```python
# 总是使用examples中的固定替换
"red apple" → "green apple"  # 每次都是green
```

**正确做法**:
```python
# 随机选择
colors = ["green", "yellow", "orange", "dark red"]
new_color = random.choice(colors)
"red apple" → f"{new_color} apple"
```

## 7. 实施代码规范

### 7.1 退化关键词选择函数

```python
def select_degradation_keywords(
    severity_keywords: List[str],
    allow_combination: bool = True
) -> str:
    """
    随机选择或组合退化关键词

    Args:
        severity_keywords: 该severity级别的关键词列表
        allow_combination: 是否允许组合多个关键词

    Returns:
        退化描述文本
    """
    if not severity_keywords:
        return ""

    # 50%概率组合，50%概率单一
    if allow_combination and len(severity_keywords) > 1 and random.random() < 0.5:
        # 组合2-3个关键词
        num_keywords = random.randint(2, min(3, len(severity_keywords)))
        selected = random.sample(severity_keywords, num_keywords)
        return ", ".join(selected)
    else:
        # 单一关键词
        return random.choice(severity_keywords)
```

### 7.2 退化程度随机选择函数

```python
def select_severity_random(
    distribution: Dict[str, float] = {"mild": 0.2, "moderate": 0.4, "severe": 0.4}
) -> str:
    """
    根据配置的分布随机选择退化程度

    Args:
        distribution: 退化程度分布 {severity: probability}

    Returns:
        选择的severity级别
    """
    severities = list(distribution.keys())
    weights = list(distribution.values())
    return random.choices(severities, weights=weights, k=1)[0]
```

### 7.3 质量提升词移除函数

```python
def remove_quality_boost_words(prompt: str) -> str:
    """
    移除prompt中的质量提升词

    Args:
        prompt: 原始prompt

    Returns:
        移除质量提升词后的prompt
    """
    quality_boost_words = [
        "masterpiece", "best quality", "high quality", "detailed",
        "sharp", "clear", "high resolution", "4k", "8k",
        "professional", "award winning", "stunning", "perfect"
    ]

    cleaned = prompt
    for word in quality_boost_words:
        # 处理各种可能的位置
        cleaned = re.sub(rf",\s*{word}\b", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(rf"\b{word}\s*,", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(rf"\b{word}\b", "", cleaned, flags=re.IGNORECASE)

    # 清理多余空格和逗号
    cleaned = re.sub(r'\s+', ' ', cleaned)
    cleaned = re.sub(r',\s*,', ',', cleaned)
    cleaned = cleaned.strip().strip(',').strip()

    return cleaned
```

## 8. 总结

本文档定义了AIGC图像质量评估数据集构建中负样本prompt生成的完整规则体系：

**核心要点**:
1. ✅ 修改方式仅限于添加/替换形容词修饰语
2. ✅ 随机选择或组合关键词，避免固定模式
3. ✅ 维度隔离，不影响其他质量维度
4. ✅ 保证生成prompt自然流畅

**退化分布**:
- mild: 20%
- moderate: 40%
- severe: 40%

**质量控制**:
- 使用检查清单验证每个生成的prompt
- 确保语法正确、语义连贯、退化明确

遵循本文档的原则，可以生成高质量、多样化的负样本prompt，为构建大规模自监督对比数据集提供坚实基础。
