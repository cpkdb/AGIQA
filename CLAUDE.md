# CLAUDE.md
# 永远使用中文回答
本文件为 Claude Code (claude.ai/code) 在此代码库中工作时提供指导。


## 项目概述

ImageReward 是一个文本到图像的人类偏好奖励模型（NeurIPS 2023）。它根据人类偏好评估生成图像与文本提示的匹配程度。该项目包括：

1. **ImageReward 模型**：图像-文本对齐的评分系统
2. **ReFL (Reward Feedback Learning，奖励反馈学习)**：Stable Diffusion 模型的微调框架
3. **基线对比工具**：CLIP、BLIP 和 Aesthetic 评分模型

## 核心架构

### 模型流程 (ImageReward/ImageReward.py:71-174)

ImageReward 模型由两个主要组件构成：

1. **BLIP (Bootstrapping Language-Image Pre-training，自举语言-图像预训练)**：视觉-语言编码器
   - 视觉编码器：处理 224x224 图像
   - 文本编码器：对提示词进行分词（最多 35 个 token），并与图像特征进行交叉注意力

2. **MLP 头**：将联合嵌入映射到偏好分数
   - 架构：768 → 1024 → 128 → 64 → 16 → 1
   - 输出使用 mean=0.167, std=1.033 进行标准化

### ReFL 训练算法 (ImageReward/ReFL.py:344-826)

ReFL 使用 ImageReward 反馈优化 Stable Diffusion 模型：

1. 从预训练 SD 模型初始化（例如 "CompVis/stable-diffusion-v1-4"）
2. 每个训练步骤：
   - 生成部分去噪（在 40 步中的随机 30-39 步停止）
   - 将潜在编码解码为图像
   - 使用冻结的 ImageReward 模型计算奖励分数
   - 反向传播损失：`F.relu(-rewards+2) * grad_scale`
3. 仅训练 UNet；VAE、文本编码器和奖励模型保持冻结

**核心洞察**：通过提前停止去噪，梯度只需流经更少的扩散步骤，使训练更高效。

## 常用命令

### 安装

```bash
# 基础 ImageReward 使用
pip install image-reward

# ReFL 训练（Stable Diffusion v1.x）
pip install image-reward diffusers==0.16.0 accelerate==0.16.0 datasets==2.11.0

# SDXL 训练
pip install -r requirements_refl_sdxl.txt
```

### 运行 ImageReward 评分

```bash
# 基础示例
python example.py

# 在基准数据集上测试（论文表 1）
bash scripts/test-benchmark.sh

# 在验证数据集上测试（论文表 3）
bash scripts/test.sh
```

### 使用 ReFL 训练

```bash
# 标准 ReFL（SD v1.4）
bash scripts/train_refl.sh

# SDXL 版本
bash scripts/train_refl_sdxl.sh

# 自定义训练
python refl.py --train_batch_size 2 --gradient_accumulation_steps 4 \
  --max_train_steps 100 --output_dir checkpoint/custom
```

关键 ReFL 参数：
- `--grad_scale`：奖励损失的缩放因子（默认：0.001）
- `--learning_rate`：默认 1e-5
- `--use_ema`：为 UNet 权重启用指数移动平均
- `--mixed_precision`：使用 "fp16" 或 "bf16" 以提高内存效率

### 训练 ImageReward 模型

```bash
cd train
# 1. 从 Hugging Face 下载数据集（THUDM/ImageRewardDB）
# 2. 准备数据集
python src/make_dataset.py
# 3. 配置：编辑 train/src/config/config.yaml
# 4. 训练
bash scripts/train_one_node.sh
```

## 代码结构说明

### 模型加载 (ImageReward/utils.py:45-83)

模型默认缓存在 `~/.cache/ImageReward/`。`load()` 函数：
- 如果未缓存，则从 Hugging Face 下载检查点
- 需要 `med_config.json` 用于 BLIP 配置
- 返回在指定设备上处于评估模式的模型

可用模型：
- `ImageReward-v1.0`：主要奖励模型
- 基线模型（通过 `load_score()`）："CLIP"、"BLIP"、"Aesthetic"

### ReFL 的数据集格式

训练数据应为包含图像-文本对的 JSON：
```json
[
  {"image": "path/to/image1.jpg", "text": "a description"},
  {"image": "path/to/image2.jpg", "text": "another prompt"}
]
```

参见 `data/refl_data.json` 查看示例格式。

### Stable Diffusion Web UI 集成

脚本 `sdwebui/image_reward.py` 提供：
- 生成图像的实时评分
- 按分数阈值自动过滤
- 将分数嵌入 PNG 元数据

安装：复制到 `stable-diffusion-webui/scripts/` 目录。

## 重要实现细节

### ReFL 中的梯度计算

奖励模型通过 `score_gard()` 计算梯度（注意：方法名拼写错误，应为 "grad"）：
- 处理中间扩散输出（而非最终图像）
- 损失将奖励推向 2.0 的阈值以上
- 梯度缩放通常为 0.001，以防止扩散训练不稳定

### 多 GPU 训练

ReFL 使用 Hugging Face Accelerate：
```bash
accelerate launch --multi_gpu --num_processes=8 refl.py [args]
```

有效批次大小 = `train_batch_size * gradient_accumulation_steps * num_processes`

### SDXL 与 SD 1.x 的区别

- `ReFL_SDXL.py`：完整的 SDXL 微调
- `ReFL_SDXL_LoRA.py`：参数高效的 LoRA 微调
- SDXL 需要不同的依赖版本（见 requirements_refl_sdxl.txt）
- SDXL 分辨率通常为 1024x1024，SD 1.x 为 512x512

## 测试与基准评估

代码库包含两种评估协议：

1. **基准数据集** (scripts/test-benchmark.sh)：
   - 来自各种文本到图像模型的 100 个提示词 × 10 张图像
   - 从 Hugging Face/清华云下载
   - 比较 ImageReward 与 CLIP/BLIP/Aesthetic 的排序准确性

2. **验证数据集** (scripts/test.sh)：
   - `data/test.json` 中的人工标注偏好
   - 用于快速验证的较小数据集

两者都会自动下载所需数据和基线模型。
