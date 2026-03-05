# AIGC Quality Degradation Dimensions 

---

## 1. Technical Quality (技术质量)
**Description:** 低层视觉信号退化：清晰度、噪声、曝光、颜色保真、纹理细节等

| Dimension | 维度 | 退化方向（效果） | 目前问题 |
| :--- | :--- | :--- | :--- |
| **blur** | 模糊（散焦/运动） | 图像整体或局部缺乏清晰度，出现散焦或动态模糊 | |
| **overexposure** | 过曝光 | 画面过亮，高光溢出，亮部细节严重丢失 | |
| **underexposure** | 欠曝光 | 画面过暗，阴影死黑，暗部细节严重丢失 | |
| **low_contrast** | 低对比度 | 灰度直方图压缩，画面灰蒙蒙，缺乏明暗对比 | |
| **color_cast** | 偏色/色调偏移 | 画面蒙上不自然的整体色罩（如惨绿、泥黄），白平衡失准 | |
| **desaturation** | 欠饱和/褪色 | 色彩暗淡无光，饱和度过低，看似陈旧或褪色 | |
| **plastic_waxy_texture** | 塑料/蜡质质感 | 皮肤或物体表面过度平滑，丢失纹理细节，呈现塑料/蜡质感 |flux效果一般 |

---

## 2. Aesthetic Quality (美学质量)
**Description:** 审美与艺术表达：构图、光影氛围、色彩风格、整体观感等

| Dimension | 维度 | 退化方向（效果） | 目前问题 |
| :--- | :--- | :--- | :--- |
| **awkward_positioning** | 布局死板/尴尬 | 主体在画面中拍摄距离失当（过远显得渺小、过近导致形变），或主体由于比例过小而被放置在不合理的边缘位置 | |
| **awkward_framing** | 取景/透视畸变 | 镜头角度刁钻或透视变形导致主体呈现不美观（如大饼脸透视） | SDXL不兼容（content_drift），仅Flux |
| **unbalanced_layout** | 画面失衡 | 画面视觉重心严重偏移，构图极度不平衡 | SDXL模型能力限制 仅Flux |
| **cluttered_scene** | 画面杂乱 | 背景充斥着杂乱琐碎的物体，干扰主体视觉呈现 | 通过背景出现新物体达到杂乱效果，需要保持主题一致|
| **lighting_imbalance** | 光照不均 | 画面光照分布混乱，出现意外的亮斑或死黑，破坏整体氛围 |  |
| **color_clash** | 配色冲突 | 配色刺眼或不和谐（如高饱和红配绿），视觉上令人不适 | |
| **dull_palette** | 色调沉闷 | 色调沉闷、压抑或乏味，缺乏吸引力 | |

---

## 3. Semantic Rationality (语义/合理性)
**Description:** 语义理解与现实合理性：解剖结构、物体属性、空间关系、物理光学、场景上下文等

### 3.1 Anatomy & Biology (解剖/生物)

| Dimension | 维度 | 退化方向（效果） | 目前问题 |
| :--- | :--- | :--- | :--- |
| **hand_malformation** | 手部畸形 | 手部结构错误（手指数量异常、粘连、关节扭曲） | 正prompt需要能表现出手部（prompt、生成模型需要稳定） |
| **face_asymmetry** | 面部不对称/异常 | 面部五官严重不对称、崩坏或结构塌陷 | |
| **expression_mismatch** | 表情与语境不符 | 人物表情与场景氛围或行为逻辑完全矛盾 | |
| **body_proportion_error** | 身体比例失调 | 人体比例严重失调（如长臂猿、头身比异常） | SDXL难以生成有问题的身体比例 |
| **extra_limbs** | 多余/重复肢体 | 出现多余的肢体（三只手、多条腿） |  |
| **impossible_pose** | 不可能姿态 | 人体姿态违反生理结构（关节反向弯曲、非自然扭转） | SDXL能力有限，仅flux |
| **animal_anatomy_error** | 动物解剖错误 | 动物出现错误的解剖特征（如混合物种特征、器官错位） |  |

### 3.2 Object Integrity (物体完整性)

| Dimension | 维度 | 退化方向（效果） | 目前问题 |
| :--- | :--- | :--- | :--- |
| **object_shape_error** | 物体形状/结构错误 | 常见物体形状扭曲、融化或结构崩坏 | |
| **extra_objects** | 多余/重复物体 | 画面中出现重复的、不属于原场景的、或与上下文逻辑完全无关的冗余插入物体 | |
| **count_error** | 数量错误 | 生成的物体数量与Prompt描述不符 | |
| **illogical_colors** | 逻辑色彩错误 | 物体呈现反常识的颜色（如蓝色火焰、紫色草地） |  |

### 3.3 Spatial & Geometry (空间/几何)

| Dimension | 维度 | 退化方向（效果） | 目前问题 |
| :--- | :--- | :--- | :--- |

| **scale_inconsistency** | 尺度不一致 | 物体间的大小比例严重反常识（如巨型昆虫） | |
| **floating_objects** | 悬浮无支撑 | 应受重力影响的物体悬浮在空中，无支撑 |  |
| **penetration_overlap** | 穿透/重叠异常 | 固体物体之间发生物理上不可能的穿插/模型重叠 |  |

### 3.4 Physical & Optical (物理/光学)

| Dimension | 维度 | 退化方向（效果） | 目前问题 |
| :--- | :--- | :--- | :--- |
| **shadow_mismatch** | 阴影矛盾 | 阴影缺失、方向错误或与投射物形状不符 | |
| **reflection_error** | 反射/镜像错误 | 镜像/水中倒影与本体不一致或内容错误 | 暂时跳过 |

### 3.5 Scene & Context (场景/上下文)

| Dimension | 维度 | 退化方向（效果） | 目前问题 |
| :--- | :--- | :--- | :--- |
| **context_mismatch** | 上下文不匹配 | 主体出现在极度不合理的场景环境中（如沙漠企鹅） |  |
| **time_inconsistency** | 时间/季节矛盾 | 画面同时出现冲突的时间/季节特征（如日月同辉） | |
| **scene_layout_error** | 场景布局错误 | 物体出现在逻辑不合理的位置（如马桶在厨房、冰箱在卧室） | 策略聚焦位置逻辑，排除悬浮/颠倒/比例 |



### 3.6 Text & Symbols (文字/符号)

| Dimension | 维度 | 退化方向（效果） | 目前问题 |
| :--- | :--- | :--- | :--- |
| **text_error** | 文字错误 | 生成的文字内容拼写错误、乱码或不可读 |   |
| **logo_symbol_error** | 图标/符号错误 | 画面中出现非预期的、干扰性的图标（如条形码、二维码）或品牌Logo水印 | |

---

## 4. Statistics Summary (统计总计)

| Category (类别) | Dimension Count (维度数量) | Description (说明) |
| :--- | :--- | :--- |
| **1. Technical Quality** | 7 | 低层视觉信号退化（清晰度、曝光、对比度等） |
| **2. Aesthetic Quality** | 7 | 高层审美与艺术表达（构图、光影、色彩搭配等） |
| **3. Semantic Rationality** | 21 | 语义逻辑与现实合理性（解剖、物体、空间、物理等） |
| **Total (总计)** | **35** | 有效维度总数（已删除维度不计入） |


