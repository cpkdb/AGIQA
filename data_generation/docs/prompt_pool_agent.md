# Prompt Pool Agent

`Prompt Pool Agent` 是当前 AIGC 质量数据生产流程中的离线资源整理与构建 agent。

第一版目标很收敛：
- 自动解析当前主链路实际使用的 source pool / subpool index
- 自动扫描现有 prompt pool / subpool 目录与文件
- 自动输出 active manifest、inventory、routing、screening plan、cleanup candidates
- 为后续 32 维最终 pool 收口提供结构化依据

在当前阶段，它已经不只是分析层，还完成了首批共享父池语义筛查、最终子池固化、主脚本 cut over 和冻结旧目录清理。

## CLI

```bash
python scripts/prompt_pool_agent.py \
  --output_dir /tmp/prompt_pool_agent_smoke
```

可选参数：

```bash
python scripts/prompt_pool_agent.py \
  --output_dir /tmp/prompt_pool_agent_smoke \
  --taxonomy_path /root/ImageReward/data_generation/config/quality_dimensions_active.json \
  --routing_config_path /root/ImageReward/data_generation/config/prompt_pool_routing_v1.json \
  --inventory_roots /root/autodl-tmp/AGIQA/data/prompt_sources_workspace/backfill_merge_runs/all_dimensions_v1_full,/root/autodl-tmp/AGIQA/data/prompt_sources_workspace/backfill_merge_runs \
  --model_filter flux-schnell,sd3.5-large-turbo,qwen-image-lightning
```

## 输出产物

第一版基础产物会输出：

- `active_pool_manifest.json`
- `prompt_pool_inventory.json`
- `prompt_pool_routing_v1.json`
- `prompt_pool_screening_plan_v1.json`
- `prompt_pool_screening_spec_v1.json`
- `prompt_pool_build_targets_v1.json`
- `prompt_pool_cleanup_candidates_v1.json`
- `manifest.json`

当前也已经补上了共享父池 builder：

- [shared_prompt_pool_family_builder.py](/root/ImageReward/data_generation/scripts/shared_prompt_pool_family_builder.py)

它当前先支持：
- `human_full_body_realistic`
- `structured_object_primary`
- `multi_object_reference`

并且采用标准化命名：
- 共享父池目录：
  - `shared_family_screened_pools_cleaned_v1`
  - `sd35_turbo_shared_family_screened_pools_clipsafe_v1`
- 最终维度子池目录：
  - `semantic_screened_dimension_subpools_cleaned_v1`
  - `sd35_turbo_semantic_screened_dimension_subpools_clipsafe_v1`
- 父池候选文件：
  - `{family_name}_candidates.jsonl`
- 供后续 LLM 语义筛查使用的输入文件：
  - `{family_name}_screening_input.jsonl`

当前 `screening_input` 里已经会带：
- `family_name`
- `pool_variant`
- `base_tags`
- `screening_goal`
- `llm_prompt_focus`
- `target_dimensions`
- `heuristics`

当前 turbo 版本的共享父池策略也已经收口：
- 不再单独做 family-level LLM 语义筛查
- 直接复用 common 版本的 `pass` 集
- 再对 turbo 版本做 token / clipsafe 过滤
- 并统一落到 `sd35_turbo_*_clipsafe_v1` 目录

另外，当前仓库里已经冻结并执行过一轮清理参考清单：

- [prompt_pool_cleanup_freeze_v1.json](/root/ImageReward/data_generation/config/prompt_pool_cleanup_freeze_v1.json)

## 当前实现状态

它当前已经能：
- 解析各模型当前真实会使用的 source pool / dimension subpool index
- 统计这些 active index 中各维度当前的 pool 大小
- 扫描现有 pool / subpool 目录并区分 `active` / `fallback` / `stale_candidate`
- 根据 `prompt_pool_routing_v1.json` 生成 32 维 routing 视图
- 输出需要新增规则召回 + LLM 语义筛选的维度计划
- 输出共享父池 / 轻规则池的 screening spec，供后续 builder 直接使用
- 输出规范化的共享父池 / 维度子池落盘命名方案
- 配合冻结清单区分 `keep_active` / `keep_fallback_until_cutover` / `keep_historical_reference` / `pending_delete_after_confirmation`
- 生成 3 个共享父池的标准化 LLM 语义筛查输入：
  - `human_full_body_realistic`
  - `structured_object_primary`
  - `multi_object_reference`
- 完成 common 版本的 family-level 语义筛查，并固化 family `pass` 池
- 通过 [semantic_screened_pool_finalizer.py](/root/ImageReward/data_generation/scripts/semantic_screened_pool_finalizer.py) 派生 7 个新维度子池：
  - `body_proportion_error`
  - `extra_limbs`
  - `object_structure_error`
  - `material_mismatch`
  - `scale_inconsistency`
  - `penetration_overlap`
  - `floating_objects`
- 为 `sd3.5-large-turbo` 复用 common `pass` 集并施加 token / clipsafe 过滤，生成 turbo 版共享父池和 turbo 版维度子池
- 把新的 semantic screened index 接入主运行脚本的最高优先级
- 按冻结清单清理已确认废弃的旧 pool 目录

当前已落地的最终目录是：

- common 共享父池：
  - `shared_family_screened_pools_cleaned_v1`
- common 维度子池：
  - `semantic_screened_dimension_subpools_cleaned_v1`
- turbo 共享父池：
  - `sd35_turbo_shared_family_screened_pools_clipsafe_v1`
- turbo 维度子池：
  - `sd35_turbo_semantic_screened_dimension_subpools_clipsafe_v1`

当前已接入主运行链路的新 index 是：

- common：
  - `semantic_screened_dimension_subpools_cleaned_v1/index.json`
- turbo：
  - `sd35_turbo_semantic_screened_dimension_subpools_clipsafe_v1/index.json`

共享父池 builder 当前也只负责：
- 从 active cleaned source pool 里按 routing/spec 做规则召回
- 生成标准化的共享父池候选集
- 生成后续 LLM 语义筛查输入

family screening 完成后，已经由：

- [semantic_screened_pool_finalizer.py](/root/ImageReward/data_generation/scripts/semantic_screened_pool_finalizer.py)

把 `pass` 结果固化成最终共享父池与维度子池，并为 turbo 版本派生 token / clipsafe 过滤后的复用池。

当前 family-specific 过滤边界大致是：
- `human_full_body_realistic`
  - 在规则层只做 very light 预过滤；正式 family-level 语义筛查允许多人，只要求至少有一个可读主人物，且身体/肢体/头身比例线索足够清楚
- `structured_object_primary`
  - 过滤掉带明显生物主体信号的样本，尽量保留非生物结构化主物体
- `multi_object_reference`
  - 先做轻量规则预筛，再进入 family-level LLM 语义筛查；实体可以是物体、人或动物，不再隐含成“多物体池”

当前还没有继续自动化的部分主要是：
- 自动触发新的 family screening run
- 自动执行后续增量补池
- 自动更新文档与 registry 中的“当前生效 index”快照
- 自动决定哪些旧 pool 可以物理删除

## 当前范围

当前这一版已经完成了“分析 → family screening → finalization → cut over → cleanup”的第一轮闭环。

但它仍然不是一个完全自动执行的资源工厂，仍保留这些人工确认点：
- 是否对新的 family / 维度继续补池
- 是否切换更多脚本或 fallback 路径
- 是否物理删除更多历史目录
- 是否把后续大规模补池直接纳入 orchestrator

后续更适合继续补的，是 registry / lineage / active-manifest 的长期维护，而不是重新推翻当前命名和目录结构。
