# AIGC图像失真分析工具使用指南 (v2.0)

## 核心改进

### ✅ v2.0 新特性
1. **默认使用LLM语义聚类** - 移除预设维度框架，完全基于数据归纳
2. **自动版本管理** - 每次运行使用时间戳标识，避免结果覆盖
3. **支持分批运行** - 处理大规模数据时自动合并结果
4. **快速访问** - `latest/` 符号链接指向最新运行结果

---

## 输出目录结构

```
/root/autodl-tmp/aigc_distortion_analysis/
├── runs/
│   ├── 20260105_191507/              # 单次运行
│   │   ├── sampled_prompts.json
│   │   ├── images/
│   │   │   └── sample_*.png
│   │   ├── raw_analysis.json
│   │   ├── quality_dimensions.json   # ✨ LLM语义聚类结果
│   │   └── dimension_report.md       # ✨ 中文报告
│   │
│   ├── 20260105_200000/              # 分批运行
│   │   ├── sampled_prompts.json      # 共享
│   │   ├── batch_1/
│   │   │   ├── images/
│   │   │   └── raw_analysis.json
│   │   ├── batch_2/
│   │   │   ├── images/
│   │   │   └── raw_analysis.json
│   │   ├── merged_raw_analysis.json  # 合并后的原始数据
│   │   ├── quality_dimensions.json   # 基于所有批次的聚类结果
│   │   └── dimension_report.md
│   │
│   └── my_custom_run/                # 自定义run_id
│       └── ...
│
└── latest -> runs/20260105_200000/   # 符号链接指向最新运行
```

---

## 使用场景

### 场景1：标准单次运行（100张图像）

```bash
cd /root/ImageReward/data_generation

# 自动生成run_id（时间戳）
python scripts/analyze_aigc_distortions.py \
    --num-samples 100 \
    --num-inference-steps 25

# 指定自定义run_id
python scripts/analyze_aigc_distortions.py \
    --num-samples 100 \
    --run-id experiment_v1 \
    --num-inference-steps 25
```

**输出**：
- 目录：`/root/autodl-tmp/aigc_distortion_analysis/runs/20260105_HHMMSS/`
- 快速访问：`/root/autodl-tmp/aigc_distortion_analysis/latest/`

**预计时间**：
- 100张图像生成：~8-10分钟
- GPT-4o分析：~8-10分钟
- LLM语义聚类：~3-5分钟
- **总计：~20-25分钟**

---

### 场景2：分批运行（1000张图像）

当图像数量很大时，分批运行可以：
- 避免单次运行时间过长
- 中途失败可以恢复
- 最后合并所有批次进行聚类

#### **步骤1：运行第1批（1-200张）**
```bash
python scripts/analyze_aigc_distortions.py \
    --num-samples 200 \
    --run-id large_run_1000 \
    --batch-id 1 \
    --seed 42
```

**注意**：
- 使用相同的 `--run-id` 对所有批次
- 使用不同的 `--batch-id`（1, 2, 3...）
- 不同批次可以使用不同 `--seed` 采样不同的prompts

#### **步骤2：运行第2批（201-400张）**
```bash
python scripts/analyze_aigc_distortions.py \
    --num-samples 200 \
    --run-id large_run_1000 \
    --batch-id 2 \
    --seed 43
```

#### **步骤3：运行第3-5批**
```bash
# 批次3（401-600）
python scripts/analyze_aigc_distortions.py \
    --num-samples 200 \
    --run-id large_run_1000 \
    --batch-id 3 \
    --seed 44

# 批次4（601-800）
python scripts/analyze_aigc_distortions.py \
    --num-samples 200 \
    --run-id large_run_1000 \
    --batch-id 4 \
    --seed 45

# 批次5（801-1000）
python scripts/analyze_aigc_distortions.py \
    --num-samples 200 \
    --run-id large_run_1000 \
    --batch-id 5 \
    --seed 46
```

#### **步骤4：合并并聚类**
```bash
python scripts/analyze_aigc_distortions.py \
    --run-id large_run_1000 \
    --merge-only
```

**目录结构**：
```
runs/large_run_1000/
├── sampled_prompts.json
├── batch_1/
│   ├── images/ (200张)
│   └── raw_analysis.json (200条)
├── batch_2/
│   ├── images/ (200张)
│   └── raw_analysis.json (200条)
├── batch_3/ ... batch_5/
├── merged_raw_analysis.json     # 1000条失真记录
├── quality_dimensions.json      # 基于1000条的聚类结果
└── dimension_report.md
```

**聚类时间**：
- 1000条失真 ÷ 250（batch_size） = 4批
- 每批聚类：~2分钟
- 合并聚类：~1分钟
- **总计：~10分钟**

---

### 场景3：快速测试（20张）

```bash
python scripts/analyze_aigc_distortions.py \
    --num-samples 20 \
    --num-inference-steps 20 \
    --run-id quick_test
```

**预计时间**：~5分钟

---

## 参数说明

### **数据集参数**
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--num-samples` | 80 | 生成的图像数量 |
| `--seed` | 42 | 随机种子（用于采样prompts） |
| `--dataset-path` | data/image_quality_train.json | 本地数据集路径 |

### **SDXL参数**
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--num-inference-steps` | 30 | 生成步数（减少可加速） |
| `--width` / `--height` | 1024 | 图像尺寸 |
| `--device` | cuda | GPU设备 |

### **LLM参数**
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--phase2-batch-size` | 250 | 语义聚类批大小 |
| `--model` | None | GPT模型覆盖（默认使用配置文件） |

### **运行管理参数**
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--run-id` | 时间戳 | 运行标识符（自定义名称） |
| `--batch-id` | None | 批次ID（分批运行时使用） |
| `--merge-only` | False | 仅合并批次并聚类 |
| `--resume` | False | 从中断处继续 |

---

## 常见问题

### Q1: 如何查看最新的分析结果？
```bash
cd /root/autodl-tmp/aigc_distortion_analysis/latest
cat dimension_report.md
```

### Q2: 如何继续之前失败的分批运行？
```bash
# 假设batch_3失败了，重新运行它
python scripts/analyze_aigc_distortions.py \
    --run-id large_run_1000 \
    --batch-id 3 \
    --resume
```

### Q3: 如何比较不同run_id的结果？
```bash
# 查看所有运行
ls /root/autodl-tmp/aigc_distortion_analysis/runs/

# 对比两个运行的维度数量
wc -l runs/20260105_191507/quality_dimensions.json
wc -l runs/20260105_200000/quality_dimensions.json
```

### Q4: 分批运行时，如何确保不重复采样prompts？
使用不同的 `--seed`：
```bash
# 批次1
--seed 42 --num-samples 200

# 批次2
--seed 43 --num-samples 200

# 批次3
--seed 44 --num-samples 200
```

### Q5: 语义聚类的batch_size如何选择？
- **250**（默认）：适用于大多数情况（最多处理约500-1000条失真）
- **200**：LLM上下文接近限制时使用
- **300**：失真描述较短时可以增大

---

## 输出文件说明

### `quality_dimensions.json`
```json
{
  "dimensions": [
    {
      "name": "Anatomical_and_Object_Distortions",
      "description": "涉及人体、动物或物体的比例、位置不自然的失真。",
      "distortion_types": [...]
    }
  ],
  "note": "从 N 个失真归纳得出，无预设框架",
  "generated_at": "2026-01-05T...",
  "total_distortions_analyzed": N
}
```

### `dimension_report.md`
- 中文格式的可读报告
- 包含维度说明、失真类型列表、统计信息

### `raw_analysis.json` / `merged_raw_analysis.json`
- GPT-4o的原始分析结果
- 包含每张图像的所有失真详情
- 用于后续重新聚类或调试

---

## 最佳实践

### 1. 小样本验证
先运行20-50张图像验证配置是否正确：
```bash
python scripts/analyze_aigc_distortions.py \
    --num-samples 20 \
    --run-id test_config
```

### 2. 分批运行大规模任务
- 每批200-300张图像
- 使用不同seed避免重复
- 保持相同run_id

### 3. 保存关键运行
使用有意义的run_id：
```bash
--run-id baseline_sdxl_1000
--run-id experiment_a_500
--run-id final_v1_2000
```

### 4. 定期清理
删除不需要的运行：
```bash
rm -rf /root/autodl-tmp/aigc_distortion_analysis/runs/old_run_id
```

---

## 技术细节

### LLM语义聚类流程
1. 收集所有失真描述（label + description）
2. 按batch_size分批（默认250条）
3. 每批调用GPT-4o进行语义聚类
4. 合并所有批次的聚类结果
5. 统一命名和去重
6. 输出8-15个最终维度

### 与v1.0的区别
| 特性 | v1.0 | v2.0 |
|------|------|------|
| 归类方法 | 关键词匹配（阶段一） | LLM语义聚类（阶段二） |
| 结果管理 | 单一目录，易覆盖 | 时间戳目录，自动版本 |
| 分批支持 | 无 | 完整支持 |
| 默认维度 | 9个预设 | 完全归纳 |
