# Judge & Repair Agent 刷新实现计划

> **执行说明：** 按任务顺序逐步实现，并在每一步完成后进行最小验证。

**目标：** 刷新 `Judge & Repair Agent`，使其评判准则来自最新的退化 taxonomy、退化维度文档以及 prompt template 策略，同时**不引入**基于 severity 的判定，也**不引入**基于分数阈值的二次 gating。

**架构：** 保持当前基于 VLM 的 pair-level judging 流程和 JSON 输出 schema 不变，但把过去写死在代码中的 dimension guideline 中心，替换为基于当前 source-of-truth 文件动态合成的评判准则。新的 judge prompt 应同时融合维度效果定义、prompt strategy、template 策略线索以及更清晰的正样本适配性提示。

**技术栈：** Python 3.10、标准库（`json`、`re`、`pathlib`）、`yaml`、现有 `degradation_judge.py`、`unittest`

---

### 任务 1：先补 source-of-truth 驱动准则的失败测试

**涉及文件：**
- 新建：`tests/test_degradation_judge_criteria.py`
- 测试：`tests/test_degradation_judge_criteria.py`

**步骤 1：先写失败测试**

补充测试，明确以下行为：
- 维度准则应从当前 taxonomy 与文档中加载，而不是只依赖写死的字符串
- template strategy cues 应从 `prompt_templates_v3` 中提取
- judge prompt 应包含最新的 effect definition 和 strategy references

测试应断言：
- `object_structure_error` 的准则反映当前维度名称和 effect definition
- `material_mismatch` 的准则包含当前模板中更强的材质策略线索
- `extra_limbs` 的 judge prompt 包含当前模板中偏向额外手臂的策略线索

**步骤 2：运行测试并确认失败**

执行：

```bash
python -m unittest tests.test_degradation_judge_criteria -v
```

预期：FAIL，因为当前 judge 仍依赖旧的硬编码 guideline 中心。

**步骤 3：编写最小实现**

在 `degradation_judge.py` 中增加辅助函数，用于：
- 从 `quality_dimensions_active.json` 加载 taxonomy metadata
- 从 `degradation_dimensions.md` 解析 effect definitions
- 扫描 `prompt_templates_v3/*.yaml` 并提取简洁的策略线索
- 合成每个维度的 judge criteria

**步骤 4：再次运行测试并确认通过**

执行：

```bash
python -m unittest tests.test_degradation_judge_criteria -v
```

预期：PASS。

### 任务 2：补正样本适配性提示与新 judge prompt 结构的失败测试

**涉及文件：**
- 修改：`tests/test_degradation_judge_criteria.py`
- 测试：`tests/test_degradation_judge_criteria.py`

**步骤 1：先写失败测试**

补充测试，明确以下行为：
- judge prompt 包含维度级正样本适配性提示
- prompt 需要显式暴露：
  - effect definition
  - prompt strategy reference
  - template strategy cues
  - compatibility hint

**步骤 2：运行测试并确认失败**

执行：

```bash
python -m unittest tests.test_degradation_judge_criteria -v
```

预期：FAIL，因为当前 prompt 仍使用较老的通用描述方式。

**步骤 3：编写最小实现**

更新 `_build_judge_prompt(...)`，让它通过新的辅助函数构造 criteria，并插入维度级 compatibility hint，同时不改变外部工具 API。

**步骤 4：再次运行测试并确认通过**

执行：

```bash
python -m unittest tests.test_degradation_judge_criteria -v
```

预期：PASS。

### 任务 3：运行聚焦回归验证

**涉及文件：**
- 修改：`data_generation/scripts/tools/degradation_judge.py`
- 测试：`tests/test_degradation_judge_criteria.py`
- 测试：`tests/test_api_client_proxy_independence.py`

**步骤 1：运行聚焦验证**

执行：

```bash
python -m unittest \
  tests.test_degradation_judge_criteria \
  tests.test_api_client_proxy_independence \
  -v
```

预期：PASS，且没有失败用例。

**步骤 2：确认范围边界**

确认本次刷新**不做**以下事情：
- 不增加基于 severity 的判定
- 不增加基于分数阈值的重分类
- 不修改 `degradation_judge(...)` 的调用签名
- 不改变在线 retry 控制流

本次只提升 judge prompt 如何获取并表达评判准则。

### 后续待实现

以下能力暂时只在计划中标注，当前版本**不实现**：
- selective judge mode
- 按维度/模型选择性启用 judge
- 面向大规模生产的 judge 成本控制策略

• 具体是这样做的，核心都在 degradation_judge.py:279。

  1. 先从 3 个最新来源里读信息

  - 从 active taxonomy 读维度元数据
    用 _load_active_taxonomy() 直接读 quality_dimensions_active.json。
    代码在 degradation_judge.py:279。

    这里主要取的是：
      - perspective
      - zh_name
      - prompt_strategy
      - controllability

    真正抽取维度信息的是 _find_dimension_metadata()，在 degradation_judge.py:413。
  - 从退化维度文档读“官方效果定义”
    用 _load_degradation_dimension_effects() 去解析 degradation_dimensions.md。
    代码在 degradation_judge.py:285。

    它的做法很直接：
      - 扫描 markdown 每一行
      - 只抓表格里以 | **dimension** | 开头的行
      - 取第 3 列，也就是“退化方向（效果）”

    所以 judge 里的 Official effect definition，现在就是从这个文档来的。
  - 从模板目录读“退化策略线索”
    用 _load_prompt_template_entries() 扫描整个 prompt_templates_v3。
    代码在 degradation_judge.py:301。

    它会把每个维度对应的：
      - 所属 yaml 文件
      - 所属 group
      - moderate/severe 的模板文本

    都收进来。

  2. 再从模板文本里抽“策略线索”

  这一步是 _extract_template_strategy_cues()，在 degradation_judge.py:319。

  它不是把整个模板原文塞给 judge，而是做了一层启发式抽取：

  - 优先看 moderate、severe、mild
  - 一行一行清洗
  - 去掉编号、短句、空行、系统性说明
  - 只保留像这种“有操作方向”的句子：
      - 含 prefer
      - 含 focus
      - 含 describe
      - 含 use
      - 含 make
      - 或一些定向关键词，比如：
          - waxy rubber
          - extra visible arms
          - the mismatch should
          - the size error

  最后最多保留 4 条 cue。

  所以 Template strategy cues 本质上是：

  - 从当前模板里抽出来的“Judge 该关注什么方向”
  - 不是整段模板复述

  3. 再拼成一个维度级 criteria 对象

  这一步是 _get_dimension_criteria()，在 degradation_judge.py:436。

  它把前面三路信息合并成：

  - dimension
  - zh_name
  - perspective
  - prompt_strategy
  - effect_definition
  - template_group
  - template_strategy_cues
  - compatibility_hint

  这里你要注意一点：

  前 3 项来源是动态的，但 compatibility_hint 目前还是人工维护的。

  compatibility_hint 现在来自 _get_compatibility_hint()，在 degradation_judge.py:359。
  这部分还是一个手写字典，比如：

  - body_proportion_error
      - 要求单人、头身和腿部比例可读
  - material_mismatch
      - 要求一个显著的结构化非生物物体
  - scale_inconsistency
      - 要求场景里有可比较的参照物

  所以目前 Judge 是：

  - effect / strategy / template cues 动态读取
  - positive compatibility hint 仍是少量硬编码补充

  4. 最后把这些 criteria 塞进 judge prompt

  这一步是 _build_judge_prompt()，在 degradation_judge.py:458。

  它会把刚才合成的 criteria 显式写进 prompt：

  - Official effect definition
  - Prompt strategy reference
  - Positive compatibility hint
  - Template strategy cues

  然后要求 VLM 按固定顺序判断：

  1. 正样本是否适合这个维度
  2. 如果适合，再判断这对图是否真的实现了目标退化
  3. 如果失败，优先归因：
      - style_drift
      - content_drift
      - insufficient_effect

  5. 目前还不是“完全配置化”的地方

  现在这套 Judge 已经比之前强很多，但还没到完全 declarative：

  - compatibility_hint 还是代码里的字典，不在配置文件里
  - template_strategy_cues 是启发式抽取，不是结构化模板字段
  - effect_definition 的解析依赖 markdown 表格格式稳定
  - 如果某个维度在文档里找不到，就会 fallback 到旧的 LEGACY_DIMENSION_GUIDELINES

  所以更准确地说：

  - Judge 现在已经不再主要依赖旧硬编码 guideline
  - 但也还没有做到 100% 全部从配置文件自动驱动
  - 当前是：
      - 动态 source-of-truth 为主
      - 少量人工 compatibility hint 补强
      - 旧 guideline 只做兜底

  一句话总结
  它现在的实现方式是：

  - 从 quality_dimensions_active.json 取“这个维度想怎么退化”
  - 从 degradation_dimensions.md 取“这个维度最终应该长什么样”
  - 从 prompt_templates_v3 取“当前模板实际在往哪个方向改写”
  - 再加一层手写的 positive compatibility hint
  - 最后组合成 Judge prompt 去判图
