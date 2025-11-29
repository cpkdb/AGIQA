# 正样本 Prompt 分类法

该体系分为三大维度：核心主体 (Content)、视觉属性 (Appearance) 和 关系与结构 (Structure)。

## 维度一：核心主体 (Subject Matter)

这是 Prompt 的名词部分，定义了画面中"有什么"。CROC 定义了 13 个大类：

| 一级分类 | 二级分类示例 (Sub-classes) | 实体示例 (Entities) |
|---------|---------------------------|---------------------|
| 1. Nature (自然) | 风景, 植物, 动物, 天气, 水下 | Mountain, Tree, Deer, Lightning, Coral |
| 2. People (人物) | 肖像, 群体, 活动, 文化 | Adult, Friends, Athlete, Dancer |
| 3. Animals (动物) | 野生动物, 家养动物, 神话生物 | Lion, Dog, Dragon |
| 4. Architecture (建筑) | 住宅, 商业, 地标, 基建 | House, Skyscraper, Castle, Bridge |
| 5. Objects (物体) | 家居, 食物, 艺术品, 工具 | Chair, Fruit Bowl, Sculpture, Hammer |
| 6. Fantasy/Sci-Fi (幻想/科幻) | 神话世界, 未来城市, 太空, 魔法 | Enchanted Forest, Hover Car, Starship, Wizard |
| 7. Vehicles (交通工具) | 陆地, 空中, 水上, 未来载具 | Car, Airplane, Boat, Hoverboard |
| 8. Technology (科技) | 电子设备, 机器人, AI, 可穿戴 | Laptop, Robot Arm, Neural Network, VR Headset |
| 9. Abstract (抽象) | 几何抽象, 色块, 概念抽象 | Circle, Gradient, Minimalist Line Art |
| 10. Events (事件) | 节日, 体育, 演出, 历史事件 | Fireworks, Soccer Ball, Concert, Vintage Clothing |
| 11. Space (太空) | 行星, 星系, 航天器, 天文事件 | Earth, Galaxy, Rocket, Solar Eclipse |
| 12. Historical (历史) | 古文明, 中世纪, 工业时代, 现代史 | Pyramid, Knight Armor, Steam Engine, Vintage Car |
| 13. Everyday Life (日常生活) | 家庭, 办公, 休闲, 交通 | Family Pet, Office Desk, Book, Bus |

## 维度二：视觉属性 (Visual Attributes)

这是 Prompt 的形容词部分，定义了主体"长什么样"。这些属性是您后续进行属性对齐 (Attribute Alignment) 退化的基础。

| 属性类别 | 定义/示例 |
|---------|----------|
| 1. Medium (媒介) | 摄影 (Photography), 插画 (Illustration), 3D渲染 (3D Rendering), 油画 (Painting), 动漫 (Anime) |
| 2. Color (颜色) | 单色 (Monochrome), 鲜艳 (Vibrant), 具体色 (Red, Blue, Green...) |
| 3. Texture (纹理) | 光滑 (Smooth), 粗糙 (Rough), 反光 (Reflective) |
| 4. Shape (形状) | 几何 (Geometric), 有机 (Organic) |
| 5. Material (材质) | 金属 (Metallic), 木质 (Wooden), 织物 (Fabric), 玻璃 (Glass), 塑料 (Plastic), 石头 (Stone) |
| 6. Style (风格) | 写实 (Realistic), 印象派 (Impressionistic), 极简 (Minimalist) |
| 7. Lighting (光照) | 自然光 (Natural), 人造光 (Artificial), 高对比度 (High Contrast) |
| 8. Layout (构图) | 居中 (Centered), 三分法 (Rule of Thirds), 不对称 (Asymmetrical) |

## 维度三：关系与交互 (Relation & Interaction)

这是 Prompt 的动词和介词部分，定义了主体与环境或他物的关系。这是您后续进行组合交互 (Composition Interaction) 退化的基础。

| 属性类别 | 子类别与描述 |
|---------|-------------|
| 1. Action (动作) | 手势 (Gesture): Pointing, Waving; 全身运动 (Full-Body): Running, Dancing, Jumping |
| 2. Spatial (空间) | 位置 (Position): Left-of, Right-of, Above, Below, Inside; 景深: Foreground, Background |
| 3. Scale (比例) | 夸张: Giant, Miniature; 写实: Life-Size |
