# 曝光退化分析与改进报告 (Exposure Degradation Analysis)

基于 `/root/autodl-tmp/technical_quality/overexposure_20260115_192414` 和 `/root/autodl-tmp/technical_quality/underexposure_20260115_192919` 的实测结果分析。

## 1. 现状问题诊断

### Overexposure (过曝) - 偏差严重
*   **当前 Prompt**: "Extremely too bright, nearly white-washed, ... bright areas turning white"
*   **观测现象**: 纪念碑图片 (`negative_42_severe.png`) 变得非常明亮、洁白，但细节依然清晰，光影关系仍然合理。SDXL 将 "bright" 和 "white-washed" 理解为了一种**高调摄影风格 (High-key Photography)** 或 **强光照环境**，而不是**传感器过载**造成的错误。
*   **根本原因**: 使用的词汇描绘了“明亮的内容”而非“损坏的信号”。
*   **改进方向**: 必须强制强调**信息丢失 (Information Loss)** 和 **光溢出 (Blooming/Glare)**。

### Underexposure (欠曝) - 程度不足
*   **当前 Prompt**: "Severely underexposed, ... details lost in deep shadows"
*   **观测现象**: 花园图片 (`negative_42_severe.png`) 确实变暗了，但仍然是一个可接受的“黄昏/阴天”场景。暗部没有完全死黑，中间调虽然暗但依然可见。
*   **根本原因**: "Underexposed" 对模型来说可能只是“暗一点的曝光”。
*   **改进方向**: 需要更暴力的词汇来描述**死黑 (Crushed Blacks)** 和 **不可见 (Unintelligible)**。

## 2. 改进策略 (Refined Strategy)

### Overexposure (过曝) 2.0
从“变亮”转向“高光溢出与细节毁灭”。

*   **核心词汇**: `blown-out highlights` (高光溢出), `clipping` (截断), `detail loss in whites` (白色区域细节丢失), `blinding glare` (致盲眩光), `harsh bloom` (强烈光晕).
*   **Negative Prompt 增强**: 将 "high dynamic range", "detailed highlights" 加入 Negative Prompt。
*   **New Template (Severe)**: 
    *   Prompt: "ruined by overexposure, blown-out highlights, blinding glare destroying all texture, pure white clipping, loss of detail in bright areas."

### Underexposure (欠曝) 2.0
从“变暗”转向“暗部死黑与阴影吞噬”。

*   **核心词汇**: `crushed blacks` (死黑), `shadow clipping` (阴影截断), `swallowed by darkness` (被黑暗吞噬), `barely visible` (几乎不可见), `muddy dark tones` (泥泞暗调).
*   **New Template (Severe)**:
    *   Prompt: "ruined by extreme underexposure, crushed blacks, image swallowed by darkness, shadow clipping, object barely visible in the dark."

## 3. 下一步行动
立即更新 `technical_quality.yaml` 中的 `overexposure` 和 `underexposure` 模板。
