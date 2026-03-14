# AIGC 图像质量退化数据生产 6-Agent Workflow 设计稿

> 本文档的定位是 **paper-oriented design proposal**，用于把当前仓库中的真实生产系统包装成一个更清晰、可展示、可评估的 agent-flow framework。
> 它不是在声明“仓库里已经完整实现了一个 6-agent runtime”，而是在当前离线资源工厂 + 在线闭环生成链路的基础上，抽象出一套更适合论文叙事的系统设计。

## 1. 目标

本文给出一套面向 AIGC 图像质量退化数据生产的 agent workflow 设计稿。该设计不是从零发明一个全新的系统，而是在当前仓库中已经落地的正样本 prompt 构建、退化 prompt 改写、图像生成、VLM 判别和闭环重试链路之上，进一步抽象出一个更清晰、更可展示、更适合论文包装的 `agent-flow system`。

设计目标有三点：

1. 把当前分散在离线脚本、主 pipeline 和工具层中的能力，上升为职责稳定的 specialized agents。
2. 保留当前系统的受控生成特性，避免为了“agent 化”而引入高风险的开放式决策。
3. 让系统既能支撑真实的数据生产，也能在论文中呈现为一个完整的自动化 workflow。

---

## 2. 设计定位

这套系统更适合被定义为一种 **受控的 agent-flow system**，而不是完全自由探索的通用 agent。

原因很直接：

- 当前任务的核心目标是稳定地产出大规模、高可用的正负图像对。
- 图像生成、VLM 判别和 API 调用成本都较高，不适合让 agent 大量自由试错。
- 当前仓库已经拥有较成熟的脚本化能力和工具封装，最合理的方向是“agent 做任务组织与状态流转，工具做具体执行”。

因此，本设计借鉴 SafetyFlow 的思路，但不照搬其安全 benchmark 场景，而是将其核心优势转化为以下四点：

1. 明确的 agent 分工
2. 强工具约束
3. 可统计、可评估的阶段化 workflow
4. 可迭代更新的数据生产闭环

---

## 3. 当前系统的真实基础

当前仓库已经具备两条核心链路，而且这两条链路的执行节奏并不相同。

### 3.1 正样本资源构建链路

当前正样本系统已经不是简单的 prompt 收集，而是一个 **离线资源生产系统**，主要包含：

- 公共 AIGC prompt 源规划与导入
- 清洗、去重、粗粒度语义打标
- 30k public-only working pool 分层抽样
- 宏观 taxonomy 统计
- 按退化维度构建正样本子池
- LLM 定向补齐
- 高约束维度语义筛查

当前主链路最重要的产物已经包括：

- `merged_working_pool_cleaned_v1.jsonl`
- `dimension_subpools_cleaned_v1/index.json`
- `anatomy_screened_dimension_subpools_cleaned_v2/index.json`
- `merged_working_pool_sd35_turbo_clipsafe_v1.jsonl`
- `sd35_turbo_dimension_subpools_clipsafe_v2/index.json`
- `semantic_screening_runs/*`

这意味着系统已经拥有了“agent 可调用的正样本基础设施”，但它更像一个 **offline resource factory**，而不是每次生成 run 都会完整重新执行的在线 agent。

### 3.2 退化数据闭环生成链路

当前 [pipeline.py](/root/ImageReward/data_generation/scripts/pipeline.py) 已经实现了一个可运行的 **在线闭环 workflow**：

1. 按维度从正样本子池中选择正样本 prompt
2. 调 `prompt_degrader` 改写负样本 prompt
3. 调图像生成工具生成正图与负图
4. 调 `degradation_judge` 检验图像对
5. 根据失败类型进行反馈修正和重试
6. 输出 dataset、full log 和 validation report

这条链路已经具备：

- 维度感知采样
- prompt-driven 退化生成
- 同 seed 内容对齐
- VLM 多维判别
- 局部闭环修正
- 断点续跑

因此，当前系统的问题不是“没有 workflow”，而是：

- 离线资源链路和在线闭环链路还没有被统一包装
- agent 分工还没有被显式命名
- run-level artifact、memory、curation 还没有被整理为论文可陈述的模块

---

## 4. 总体架构

建议将整个系统抽象为 6 个 specialized agents，并由一个轻量 orchestrator 统一编排。
但更准确的组织方式不是一条单链串行，而是 **双层 workflow**。

### 4.1 总体结构

```text
Layer A: Offline Resource Factory

Prompt Sources / Existing Registries
          ↓
  Agent 1: Prompt Pool Agent
          ↓
  Agent 2: Taxonomy & Allocation Agent
          ↓
  Agent 6: Curator Agent (cross-run update / registry / filtering)
          ↓
  Versioned Prompt Pools + Allocation Plans + Registry


Layer B: Online Closed-loop Pair Generation

Versioned Prompt Pools + Allocation Plans
          ↓
  Agent 3: Prompt Degradation Agent
          ↓
  Agent 4: Image Pair Generation Agent
          ↓
  Agent 5: Judge & Repair Agent
          ↺
     local retry loop
          ↓
  Curated Pair Dataset + Run Reports + Failure Logs
```

### 4.2 Orchestrator 的角色

Orchestrator 不负责“创造性生成”，只负责：

- 维护 run state
- 调度 agent 的执行顺序
- 管理输入输出 artifact
- 控制 retry、resume、sampling budget 和 model budget

在当前设计中，Orchestrator 也不要求变成一个“会自由决策的大 agent”。  
它更适合作为一个 **tool-constrained state machine / workflow coordinator**，负责：

- 资源版本选择
- 运行配置注入
- 阶段间 artifact 传递
- retry / resume / budget 控制

### 是否需要模型

当前建议 **不需要单独模型**。  
Orchestrator 更适合作为一个确定性的 workflow coordinator / state machine，而不是 reasoning-heavy agent。  
它的职责是调度和状态管理，不应把 LLM 推理引入在线主路径。

如果后续确实希望增加一点“规划感”，也应优先通过：

- 显式 allocation plan
- 历史统计驱动的规则调度
- 简单的 policy engine

而不是把它做成一个会频繁调用 LLM 的中心大脑。

也就是说，系统的智能体感主要来自分工明确的 agents，而不是一个无边界的大 agent。

### 当前实现状态

这一层现在已经有了一个第一版显式实现：

- [orchestrator.py](/root/ImageReward/data_generation/scripts/orchestrator.py)
- [orchestrator_tools.py](/root/ImageReward/data_generation/scripts/orchestrator_tools.py)
- [orchestrator.md](/root/ImageReward/data_generation/docs/orchestrator.md)

它当前已经能：

- 读取当前 active prompt pool / dimension index
- 生成标准化 `run_config.json`
- 生成 `launch_command.sh`
- 生成 `run_registry.json`
- 支持 `dry_run`
- 支持最小版 `execute`
- 在 `run_registry.json` 中记录 `planned -> running -> completed/failed`

但它目前还没有做到：

- 读取 `allocation_plan.json`
- 管理多任务 / 多卡调度
- 统一 resume / queue / stage orchestration
- 自动串联 `Curator Agent` 和后续 memory 更新

---

## 5. 六个 Agent 的职责设计

## 5.1 Agent 1: Prompt Pool Agent

### 职责

负责构建和更新正样本 prompt 资源池，为后续各退化维度提供可调用的正样本入口。

### 输入

- 公共 AIGC prompt 数据源
- 采样规划
- 清洗与打标规则
- LLM 补齐计划
- 高约束维度筛查策略

### 输出

- 主正样本池
- 维度兼容子池
- 高约束维度筛查池
- 对应的索引文件和统计报告

### Primary artifacts

- `merged_working_pool_cleaned_v1.jsonl`
- `dimension_subpools_cleaned_v1/index.json`
- `anatomy_screened_dimension_subpools_cleaned_v2/index.json`
- `sd35_turbo_dimension_subpools_clipsafe_v2/index.json`
- prompt pool sizing / coverage reports

### 对应当前已有模块

- [prompt_source_plan.py](/root/ImageReward/data_generation/scripts/prompt_source_plan.py)
- [prompt_source_downloader.py](/root/ImageReward/data_generation/scripts/prompt_source_downloader.py)
- [prompt_candidate_cleaner.py](/root/ImageReward/data_generation/scripts/prompt_candidate_cleaner.py)
- [prompt_candidate_tagger.py](/root/ImageReward/data_generation/scripts/prompt_candidate_tagger.py)
- [prompt_candidate_sampler.py](/root/ImageReward/data_generation/scripts/prompt_candidate_sampler.py)
- [dimension_subpool_builder.py](/root/ImageReward/data_generation/scripts/dimension_subpool_builder.py)
- [positive_prompt_backfill_executor.py](/root/ImageReward/data_generation/scripts/positive_prompt_backfill_executor.py)
- [positive_prompt_semantic_screening.py](/root/ImageReward/data_generation/scripts/positive_prompt_semantic_screening.py)

### 在系统中的价值

它解决的是“正样本从哪里来、怎么变成可调度资源”的问题，是整个数据生产系统的上游基础设施层。
在论文叙事里，这个 agent 应该被明确说明为 **offline / cross-run agent**。

### 成熟度与实现建议

这是当前最成熟的 offline agent 之一。  
从工程角度看，它已经具备真实生产所需的大部分核心能力；后续更值得补的不是“再加一个更聪明的模型”，而是：

- pool manifest / registry
- lineage 与 provenance
- 维度级 blacklist / whitelist
- stale pool refresh policy

这一层现在也已经有了一个第一版的显式 Agent 实现，但仍然是 **analysis-only**，还没有接入在线 orchestrator。  
当前已有独立入口：

- [prompt_pool_agent.py](/root/ImageReward/data_generation/scripts/prompt_pool_agent.py)
- [prompt_pool_agent_tools.py](/root/ImageReward/data_generation/scripts/prompt_pool_agent_tools.py)
- [prompt_pool_agent.md](/root/ImageReward/data_generation/docs/prompt_pool_agent.md)

它当前已经能：

- 解析各模型当前真实会使用的 source pool / subpool index
- 统计 active index 下各维度当前 pool 大小
- 扫描现有 pool / subpool 目录并输出 inventory
- 区分 `active` / `fallback` / `stale_candidate`
- 根据 routing 配置生成 32 维最终版 pool routing 视图
- 输出需要规则召回 + LLM 语义筛选的新维度计划
- 输出共享父池 / 轻规则池的 screening spec，供后续 builder 和 LLM 语义筛选复用

并且已经开始补共享父池 builder：

- [shared_prompt_pool_family_builder.py](/root/ImageReward/data_generation/scripts/shared_prompt_pool_family_builder.py)

当前共享父池的 turbo 路线也已经收口成更省成本的方案：

- common 版本负责 family-level LLM 语义筛查
- `sd3.5-large-turbo` 不再单独重复做一遍 family-level 语义筛查
- turbo 版本直接复用 common 的 `pass` 集
- 再额外施加 token / clipsafe 过滤，生成 turbo 复用池

这样能显著减少重复筛查成本，同时保持 turbo 版本与 common 版本的语义基底一致。

当前第一步先支持：
- `human_full_body_realistic`
- `structured_object_primary`
- `multi_object_reference`

其中 `multi_object_reference` 的含义已经进一步收口为：
- 先做轻量规则预筛
- 再做 family-level LLM 语义筛查
- 实体可以是物体、人或动物
- 目标是保留“多个可读实体”或“一个主体加清楚参照关系”的 prompt，而不是把它误解成“必须多物体”的池

它的定位不是“直接产出最终维度子池”，而是：
- 先从 active cleaned source pool 中做规则召回 + 轻量签名过滤
- 生成共享父池候选集
- 生成后续 LLM 语义筛查输入
- 使用统一命名落盘，方便后面继续派生到具体维度子池

当前这层已经把“共享父池构造”和“后续语义筛查准备”连起来了：
- 共享父池 builder 先输出 `{family_name}_candidates.jsonl`
- 同时输出 `{family_name}_screening_input.jsonl`
- screening input 中会携带 family 级的 `screening_goal`、`llm_prompt_focus`、`target_dimensions` 和启发式信息

family screening 完成后，又新增了最终固化脚本：

- [semantic_screened_pool_finalizer.py](/root/ImageReward/data_generation/scripts/semantic_screened_pool_finalizer.py)

它负责：
- 把 completed family screening 的 `pass` 结果固化成 common 共享父池
- 从共享父池派生出对应维度的 common 子池
- 再基于 common 结果派生 `sd3.5-large-turbo` 的 token / clipsafe 过滤版本

最终统一命名收口为：
- common 共享父池：`shared_family_screened_pools_cleaned_v1`
- common 维度子池：`semantic_screened_dimension_subpools_cleaned_v1`
- turbo 共享父池：`sd35_turbo_shared_family_screened_pools_clipsafe_v1`
- turbo 维度子池：`sd35_turbo_semantic_screened_dimension_subpools_clipsafe_v1`

这样后续真正接入 LLM 语义筛查时，不需要再从零拼 family 上下文，而可以直接沿用 builder 产出的标准化输入。

到目前为止，这一层已经完成了第一轮真实落地：

- `human_full_body_realistic`、`structured_object_primary`、`multi_object_reference` 三个 common shared family 的语义筛查已经完成
- `semantic_screened_pool_finalizer.py` 已经把筛查 `pass` 结果固化成 common 共享父池与 common 维度子池
- `sd3.5-large-turbo` 已经改成复用 common `pass` 集，再施加 token / clipsafe 过滤派生 turbo 版本
- 7 个新维度池已经正式落盘：
  - `body_proportion_error`
  - `extra_limbs`
  - `object_structure_error`
  - `material_mismatch`
  - `scale_inconsistency`
  - `penetration_overlap`
  - `floating_objects`
- 主运行脚本已经把：
  - `semantic_screened_dimension_subpools_cleaned_v1/index.json`
  - `sd35_turbo_semantic_screened_dimension_subpools_clipsafe_v1/index.json`
  切到最高优先级
- 首批冻结的旧 pool 目录已经在 cut over 后完成清理

因此，Prompt Pool Agent 当前已经不再只是“分析层 + 准备层”，而是已经完成了第一轮从规划到执行的资源收口。

但它目前仍然没有做到：

- 自动增量触发新的 family screening run
- 自动维护长期 registry / lineage 快照
- 被显式 orchestrator 消费为统一的资源状态输入
- 自动决定更多历史目录是否可以物理删除

### 是否需要模型

可以继续复用现有的 LLM 补齐与语义筛查模块，但不需要新增一个独立的“Prompt Pool 大脑模型”。  
更合适的做法是：**资源构建以脚本和规则为主，LLM 只在 backfill 和 high-constraint screening 中作为受控工具调用。**

---

## 5.2 Agent 2: Taxonomy & Allocation Agent

### 职责

负责维护质量退化维度体系，并围绕维度分布、宏观类别覆盖和模型分配制定生产计划。

### 输入

- 当前退化维度 taxonomy
- 各维度正样本池规模
- 各维度历史有效产出率
- 目标数据量和模型配置

### 输出

- 维度采样计划
- severity 配比
- 维度与模型的分配策略
- 低覆盖维度补齐计划

### Primary artifacts

- `quality_dimensions_active.json`
- `coverage_summary.json`
- `model_dimension_stats.json`
- `speed_summary.json`
- `allocation_plan.template.json`
- `allocation_insights.md`
- `manifest.json`

### 对应当前已有基础

- [degradation_dimensions.md](/root/ImageReward/data_generation/docs/degradation_dimensions.md)
- [quality_dimensions_active.json](/root/ImageReward/data_generation/config/quality_dimensions_active.json)
- [prompt_pool_sizing.py](/root/ImageReward/data_generation/scripts/prompt_pool_sizing.py)
- [prompt_pool_coverage_report.py](/root/ImageReward/data_generation/scripts/prompt_pool_coverage_report.py)

### 当前缺口

这一层目前已经有了一个第一版的显式 Agent 实现，但仍然是 **analysis-only**，还没有接入在线 orchestrator。  
当前已有独立入口：

- [taxonomy_allocation_agent.py](/root/ImageReward/data_generation/scripts/taxonomy_allocation_agent.py)
- [taxonomy_allocation_tools.py](/root/ImageReward/data_generation/scripts/taxonomy_allocation_tools.py)
- [taxonomy_allocation_agent.md](/root/ImageReward/data_generation/docs/taxonomy_allocation_agent.md)

它当前已经能：

- 解析当前实际运行会使用的 prompt pool / subpool
- 读取当前 active taxonomy
- 聚合历史 run 成功率与失败类型
- 粗粒度估算各模型生成速度
- 输出可人工编辑的 allocation 计划模板

但它还没有做到：

- 自动做生产策略决策
- 被 orchestrator 或 pipeline 直接消费
- 自动输出最终的 `allocation_plan.json`

当前版本尤其应该对齐两个事实：

- 当前 active taxonomy 是 **32 个维度**，按 `technical_quality / aesthetic_quality / semantic_rationality` 三大 perspective 组织
- 生产调度实际上已经存在模型差异化资源分配，例如 `sd3.5-large-turbo` 使用单独的 clipsafe prompt pool

### 是否需要模型

我建议第一版 **默认不用模型**。  
这层最自然的实现方式是基于：

- active taxonomy
- pool coverage
- 历史成功率
- model × dimension 兼容统计

来输出 allocation plan。  
后续如果想要“agent 感”更强，可以只在离线分析阶段引入一个廉价模型做总结和建议，但不建议把它放到在线热路径里。

当前实现状态也已经符合这个判断：

- 第一版是 **无模型、确定性、离线分析型 Agent**
- 目前不替用户自动决定：
  - 哪些维度多采样
  - 哪些维度只跑某些模型
  - 哪些维度只保留 `severe`
  - 哪些维度启用 judge

### 在大规模生产中的必要性

对于百万对、尤其是多卡并行生产，这一层是 **必不可少** 的。  
原因不是为了论文包装，而是为了避免：

- 某些维度过采样
- 某些维度覆盖不足
- 某些模型持续浪费在低成功率维度上
- severe / moderate 配比失控

因此，这一层应当被实现为一个轻量但明确的 **planning layer**，而不是可有可无的分析脚本。

---

## 5.3 Agent 3: Prompt Degradation Agent

### 职责

根据目标维度、严重度和模型上下文，将正样本 prompt 改写成负样本 prompt。

### 输入

- 正样本 prompt
- 目标退化维度
- 目标严重度
- 模型 ID
- 历史失败反馈

### 输出

- 负样本 prompt
- 退化策略元数据

### 对应当前已有模块

- [llm_prompt_degradation.py](/root/ImageReward/data_generation/scripts/llm_prompt_degradation.py)
- [prompt_degrader.py](/root/ImageReward/data_generation/scripts/tools/prompt_degrader.py)
- [prompt_templates_v3](/root/ImageReward/data_generation/config/prompt_templates_v3)

### 当前强项

这是当前最成熟的 agent 候选模块之一，因为它已经具备：

- 维度感知模板选择
- severity 控制
- 失败反馈驱动重写
- 对生成模型差异的适配入口

### 论文包装建议

这部分可以明确写成一个 **Prompt Rewriting Agent**，强调它不是自由生成，而是“在 taxonomy、模板和模型约束下进行目标明确的退化改写”。

在当前系统里，它还具备两个对论文叙事很重要的特征：

- degradation prompt 与 target model 绑定，例如 `sd3.5-large-turbo` 有长度约束和专用 prompt pool
- 可接收 judge 反馈后重写，因此不是静态模板替换，而是受控的 local repair

### 成熟度与实现建议

这是当前最成熟的在线 agent 候选模块之一。  
从工程角度看，后续更值得持续优化的是：

- model-specific prompt constraints
- dimension-specific template evolution
- retry failure memory

而不是重写它的主体结构。

---

## 5.4 Agent 4: Image Pair Generation Agent

### 职责

负责调用不同图像生成模型，根据正负 prompt 构造内容对齐的图像对。

### 输入

- 正样本 prompt
- 负样本 prompt
- seed
- model profile
- runtime profile

### 输出

- 正样本图像
- 负样本图像
- 生成元数据

### 对应当前已有模块

- [pipeline.py](/root/ImageReward/data_generation/scripts/pipeline.py)
- [tools/image_generator.py](/root/ImageReward/data_generation/scripts/tools/image_generator.py)
- 各种 generator 脚本，如 [sdxl_generator.py](/root/ImageReward/data_generation/scripts/sdxl_generator.py)、[flux_generator.py](/root/ImageReward/data_generation/scripts/flux_generator.py)、[qwen_image_lightning_generator.py](/root/ImageReward/data_generation/scripts/qwen_image_lightning_generator.py)

### 当前强项

这一层已经天然具备“工具化”特征，而且支持多模型，是 agent 系统很好的执行层。
当前最重要的运行模型包括：

- `flux-schnell`
- `sd3.5-large-turbo`
- `qwen-image-lightning`

### 成熟度与实现建议

这也是当前最成熟的在线执行层之一。  
如果后续走多卡并行生产，最核心的新增工作不会是“再 agent 化”，而是：

- multi-GPU task queue / worker dispatch
- run sharding
- output registry aggregation
- 失败样本的异步回收

### 当前缺口

目前它更像“生成工具集合”，还没上升到一个显式的 generation agent。  
后续可增加：

- model-level 成功率统计
- 每维度在不同模型上的 yield 估计
- 生成预算调度

---

## 5.5 Agent 5: Judge & Repair Agent

### 职责

负责判定当前图像对是否为有效训练样本，并在失败时给出修复方向，驱动局部闭环重试。

### 输入

- 正负图像
- 目标维度
- 正负 prompt

### 输出

- valid / invalid 判定
- failure type
- 多维评分
- 修复反馈

### 对应当前已有模块

- [tools/degradation_judge.py](/root/ImageReward/data_generation/scripts/tools/degradation_judge.py)
- [pipeline.py](/root/ImageReward/data_generation/scripts/pipeline.py) 中的 `generate_pair_with_retry()`

### 当前强项

这是当前系统最接近完整 agent 行为的一层，因为它已经形成了：

- 判别
- 诊断
- 修复建议
- 局部重试

### 论文包装建议

可以把它明确定义为一个 **Judge-and-Repair Agent**，并强调其作用不是做最终打分，而是做“样本可用性把关 + 失败闭环修正”。

从实现上看，这一层已经是最接近 runtime agent 的模块：

- 它接收图像对和目标维度
- 它输出 structured scores 与 failure types
- 它反向驱动局部重试

这也是论文里最容易展示出“agent feedback loop”特点的一层。

### 当前判别逻辑

当前 Judge & Repair 并不是单纯二分类，而是一个带优先级的 structured judge。  
它会优先判断：

1. 正样本内容是否根本不适合该维度  
2. 正样本风格是否使目标退化不可见  
3. 负样本是否发生 style drift / content drift / insufficient effect

当前输出包含：

- `valid`
- `failure`
- 4 维评分：
  - `content_preservation`
  - `style_consistency`
  - `degradation_intensity`
  - `dimension_accuracy`

### 当前准则来源

当前 Judge 的评判准则不再主要依赖旧的硬编码 dimension guideline，而是动态融合以下四类来源：

- [degradation_dimensions.md](/root/ImageReward/data_generation/docs/degradation_dimensions.md)
  - 提供每个维度的 `Official effect definition`
- [prompt_templates_v3](/root/ImageReward/data_generation/config/prompt_templates_v3)
  - 提供 `Template strategy cues`
- [judge_compatibility_hints_v1.yaml](/root/ImageReward/data_generation/config/judge_compatibility_hints_v1.yaml)
  - 提供 `Positive compatibility hint`
- [quality_dimensions_active.json](/root/ImageReward/data_generation/config/quality_dimensions_active.json)
  - 提供 taxonomy metadata，如 perspective / 中文名 / controllability

其中：

- `Official effect definition` 由 `degradation_dimensions.md` 的表格“退化方向（效果）”列解析得到
- `Template strategy cues` 由 Judge 从当前模板中启发式抽取少量关键策略句
- `Positive compatibility hint` 已经迁移到独立配置文件，不再放在 Python 代码中

Judge 当前构造的 prompt 会显式包含：

- `Official effect definition`
- `Positive compatibility hint`
- `Template strategy cues`

而不会再把 `Prompt strategy reference` 暴露到最终 judge prompt 中。

### 当前边界

当前 Judge 刷新后的边界是：

- **不使用 severity 参与判定**
- **不做分数阈值二次 gating**
- **不改变现有 `degradation_judge(...)` 调用签名**
- 仍然以 pair-level acceptance 为主，而不是 dataset-level curation

因此，当前 Judge 的核心定位仍然是：

- 判断当前图像对是否可用
- 给出失败类型
- 驱动局部重试闭环

### 风险与改进方向

当前仍有一个需要警惕的问题：  
`valid` 主要还是由 VLM 直接给出，而系统还没有在所有维度上叠加足够严格的 dimension-specific score thresholds。  
因此，如果后续要提高严谨性，更合理的方向是：

- 对高风险维度增加更严格的 acceptance policy
- 补充 dimension-specific judge hints
- 从“VLM 直接 valid”逐步过渡到“VLM + 规则阈值”的混合判定

### 在实际大规模生产中的建议

这层在论文叙事里可以保持为完整闭环。  
但在实际百万对生产中，我不建议对所有维度全量开启。更合理的策略是：

- 对高成功率、低风险维度：只做抽样 judge
- 对低成功率、高风险维度：开启 full Judge & Repair
- 对新模板 / 新模型 / 新 pool：先做 canary run 和 selective validation

也就是说，Judge & Repair 在生产系统里更适合作为 **selectively activated quality gate**，而不是所有 pair 的强制在线环节。

---

## 5.6 Agent 6: Curator Agent

### 职责

负责最终样本入库、统计分析、失败归因、冗余控制和后续增量更新。

### 输入

- 所有 pair 的生成结果
- judge 结果
- 历史成功率与失败统计

### 输出

- 最终数据集
- run-level 报告
- 维度分布报告
- 失败模式分析
- 后续补池与再生产建议

### Primary artifacts

- `dataset.json`
- `full_log.json`
- `validation_report.json`
- `run_registry.json`（建议新增）
- `memory_stats.json`（建议新增）
- `curation_decisions.jsonl`（建议新增）

### 对应当前已有基础

- `dataset.json`
- `full_log.json`
- `validation_report.json`
- 各类统计脚本和报告脚本

### 当前实现状态

这一层现在已经有了第一版显式实现：

- [curator_agent.py](/root/ImageReward/data_generation/scripts/curator_agent.py)
- [curator_agent_tools.py](/root/ImageReward/data_generation/scripts/curator_agent_tools.py)
- [curator_agent.md](/root/ImageReward/data_generation/docs/curator_agent.md)

它当前已经能：

- 扫描 `runs_root` 下已有 run artifact
- 优先以 `full_log.json` 为主做 pair-level failure memory 提取
- 输出：
  - `failure_pattern_summary.json`
  - `failure_pattern_by_prompt.json`
  - `blacklist.json`
  - `curation_decisions.jsonl`
  - `memory_stats.json`
- 在 prompt 级别识别重复失败模式，为后续模板和 pool 反哺提供输入

但它目前仍然缺：

- pair-level dedup / diversity filtering
- difficulty-aware retention
- dimension-level yield balancing
- 更强的 registry / memory 回写

所以 Curator Agent 现在已经从“纯概念层”推进到了“可运行的第一版数据整理层”，但还没有发展到完整的数据工厂后处理层。

### 是否需要模型

第一版我建议 **默认不用模型**。  
Curator 更适合优先做成：

- 统计驱动
- embedding / rule 驱动
- 异步批处理驱动

的后处理层。  
如果后续需要更强的总结能力，可以只在 run-level report generation 或 failure pattern summarization 中引入模型，而不把它放到主数据生产热路径。

### 在大规模生产中的必要性

对于多卡并行和百万对生产，这一层也是 **必不可少** 的。  
它不是为了“看起来更像 agent”，而是为了：

- 跨 worker 聚合结果
- 做 pair-level 去重与多样性控制
- 管理高失败 prompt blacklist
- 生成下一轮 allocation 的反馈信号

因此，它非常重要，但应当被设计成 **异步 / cross-run layer**，而不是在线阻塞式模块。

---

## 6. 与 SafetyFlow 的对应关系

如果和 SafetyFlow 做结构对应，可以得到下面这个映射。

| SafetyFlow | 当前系统建议对应 | 说明 |
|---|---|---|
| Ingestion Agent | Prompt Pool Agent | 公共源导入与资源池构建 |
| Categorization Agent | Taxonomy & Allocation Agent | taxonomy、维度分配与覆盖规划 |
| Generation Agent | Prompt Degradation Agent + Image Pair Generation Agent | 你这里生成分成两层，更合理 |
| Augmentation Agent | Prompt Pool Agent / Prompt Degradation Agent | 前者做正样本补池，后者做负样本改写 |
| Deduplication Agent | Curator Agent | 当前这部分还偏弱 |
| Filtration Agent | Judge & Repair Agent + Curator Agent | 一部分在线判别，一部分最终过滤 |
| Dynamic Evaluation Agent | Taxonomy & Allocation Agent + Curator Agent | 当前尚未显式实现 |

这说明你并不是缺少 workflow，而是缺少一套更清楚的 agent 分工和系统包装。
但要明确：你这里最自然的对应关系是 **hybrid workflow**，不是和 SafetyFlow 完全同构的一条串行 runtime。

---

## 7. 当前系统离“成熟 agent workflow”还差什么

## 7.1 已经具备的能力

当前已经具备的能力包括：

- 多源正样本资源构建
- 维度兼容子池
- prompt-driven 退化生成
- 多模型图像生成
- VLM 判别
- 局部闭环修复
- 断点续跑
- 基础统计报告

这些能力足以支撑真实生产。

从实现成熟度上，可以进一步区分为：

- **已实现并在主链路运行**：
  - versioned prompt pools
  - explicit lightweight orchestrator
  - prompt-driven degradation
  - multi-model image generation
  - judge-driven local retry
  - checkpoint / resume
  - dataset / log / report 输出
  - first curator layer for cross-run failure memory
- **已有雏形但未形成稳定 agent**：
  - taxonomy-aware allocation
  - strategy evolution / template optimization
  - richer cross-run memory / registry feedback
- **尚缺显式实现**：
  - curator-level retention / diversity control
  - dynamic distribution balancing

## 7.2 还缺的关键能力

但如果目标是一个像 SafetyFlow 一样“完整、可展示、可评估”的 agent-flow system，还缺以下五类能力：

### A. 统一的 Artifact / Run Registry

需要一个显式 registry 来记录：

- 正样本池版本
- taxonomy 版本
- 子池索引版本
- run 配置
- 使用的模型和 judge
- 成本、耗时、有效率

### B. 真正的跨运行 Memory

当前文档里设计过 `KnowledgeBase`，但仓库内尚无稳定落地实现。  
需要把以下经验持久化：

- `dimension × model` 成功率
- `dimension × template` 失败率
- 高频 failure type
- 哪类正样本子池在某些维度上更稳定

### C. Curator 层的过滤与保留机制

需要补上：

- pair-level 近重复过滤
- 低信息量 pair 过滤
- 各维度视觉多样性控制
- 高失败 prompt blacklist

### D. 动态难度与分布调节

需要能根据历史产出情况自动调节：

- 哪些维度多采样
- 哪些维度要再补齐正样本
- 哪些模型不适合某些维度
- 哪些维度的 severe 产出不稳定

### E. 系统级指标与 ablation 能力

需要固定输出：

- stage / agent success rate
- 平均 retry 次数
- 每维度有效产出率
- 每模型有效产出率
- 每次 run 的时间和 API 成本
- 各 agent 对最终数据量和有效率的贡献

这部分会直接决定它能否被包装成像 SafetyFlow 那样“有说服力的自动化系统”。

## 7.3 生产模式与论文模式的区别

为了避免系统设计在论文叙事和真实生产之间相互冲突，建议明确区分两种运行模式。

### 论文展示模式

- 展示完整 6-Agent workflow
- 强调 Judge-and-Repair 的闭环能力
- 展示 cross-run registry / memory / curator 的闭环更新

### 大规模生产模式

- Prompt Pool Agent / Taxonomy & Allocation Agent / Curator Agent 主要作为离线或异步层
- Orchestrator 采用确定性调度，不引入额外大模型
- Prompt Degradation Agent 和 Image Pair Generation Agent 保持为主力在线路径
- Judge & Repair 选择性启用，重点覆盖低成功率和高风险维度

这样的表达既不会削弱论文里的系统完整性，也更符合真实的大规模生产约束。

---

## 8. 最值得优先包装的三个点

如果只做最小改动，我建议优先包装下面三个点。

### 8.1 把当前系统显式命名为 6-Agent Workflow

这是成本最低但收益最大的包装。

你不需要马上把所有模块改成真正多 agent runtime。  
只要把当前已有模块重新组织、命名、画图，就已经会比“一个 pipeline 脚本”强很多。

### 8.2 做一个 Run Registry + Memory 层

这是让系统真正像“会学习、会进化的 agent 工厂”的关键。

它不需要很复杂，第一版只要记录：

- 运行配置
- 维度产出
- 失败类型
- 模型兼容性

就已经能显著提高系统完整度。

### 8.3 做 Curator 层

也就是把“最终留下哪些样本、为什么留下、为什么丢弃”从隐式过程变成一个显式模块。  
SafetyFlow 很强的一点就是它不只负责生成，还负责最终 benchmark 质量。  
你这里也需要一个对应层。

---

## 9. 推荐的落地路线

## 第一阶段：先完成论文级包装

目标：让当前系统在叙事上成为一个完整的 agent-flow framework，同时不夸大当前已实现部分。

建议产出：

- 一张双层总框架图（offline resource factory / online closed-loop generation）
- 一张 agent-tool 对照表
- 一张当前已实现 vs 部分实现 vs 待完善能力表
- 一套统一的 agent 命名

## 第二阶段：补最关键的系统能力

优先实现：

1. Run Registry
2. Knowledge Memory
3. Curator / Filtration

## 第三阶段：再考虑 runtime 级 agent 化

如果后续确实需要，可以再考虑：

- 将 orchestrator 改成更显式的 agent scheduler
- 为各 agent 增加统一输入输出 schema
- 引入更完整的 tool invocation traces

但这一步不是当前最优先的。

---

## 10. 总结

当前系统已经具备了一个高质量自动化 workflow 的核心能力，尤其是在正样本资源构建、退化 prompt 改写、图像对生成和 judge 闭环修复方面，已经明显超过了普通脚本流水线的层级。  
它距离一个真正“成熟的 agent-flow system”并不遥远，主要缺口不在于更多的生成功能，而在于：

- 更清晰的 agent 分工
- 更统一的状态与 artifact 管理
- 更完整的 curator 与 memory 层
- 更系统化的指标和 ablation 输出

因此，最合理的方向不是推翻当前 pipeline，而是在当前代码基础上，把它上升为一个 **面向 AIGC 图像质量退化数据生产的受控、双层 agent-flow system**：

- 上层是离线资源工厂，负责 prompt pool、taxonomy、allocation 与 curator 更新
- 下层是在线闭环生成链路，负责 degradation、generation、judge 与 local repair

这一路线既与现有工作高度兼容，也最适合论文包装与后续大规模生产。
