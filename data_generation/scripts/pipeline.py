#!/usr/bin/env python3
"""
Data Generation Pipeline (Closed-Loop)
基于 smolagents tools 的数据生成主脚本
实现 生成 → 判别 → 诊断 → 修正 → 重新生成 的闭环流程

6阶段 Workflow (可选启用):
  Stage 1: 数据筛选 (prompt_filter)           --tagged_prompts + --tag_requirements
  Stage 2: 策略选择 (StrategyOptimizer)        (自动)
  Stage 3: 质量退化 (LLM prompt rewrite)       (现有)
  Stage 4: 图像生成 (SDXL/Flux)                (现有)
  Stage 5: 正负样本检验 (VLM Judge)             (现有，多维度评分)
  Stage 6: 反馈优化 (KnowledgeBase)            --knowledge_base_dir
"""

import argparse
import concurrent.futures
import json
import logging
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Set, Tuple

sys.path.insert(0, str(Path(__file__).parent))

from tools import prompt_degrader, image_generator, degradation_judge

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataGenerationPipeline:
    """
    闭环数据生成流水线
    流程: prompt退化 → 图像生成 → VLM判别 → [失败: 诊断→修正→重新生成] → 保存
    """

    # 语义子组 → mapping_from_v2 键
    DIMENSION_SUBGROUPS = {
        "semantic_anatomy": ["semantic_rationality.anatomy_biology"],
        "semantic_object": ["semantic_rationality.object_integrity_attributes"],
        "semantic_spatial": [
            "semantic_rationality.spatial_geometry_positioning",
            "semantic_rationality.physical_plausibility_optics",
            "semantic_rationality.scene_context_consistency",
        ],
        "semantic_text": ["semantic_rationality.text_symbol_semantics"],
    }

    # dimension → 模板分组名（与 config/prompt_templates_v3/*.yaml 的顶级 key 对应）
    DIMENSION_TO_TEMPLATE_GROUP = {
        # semantic_anatomy
        "hand_malformation": "semantic_anatomy",
        "face_asymmetry": "semantic_anatomy",
        "expression_mismatch": "semantic_anatomy",
        "body_proportion_error": "semantic_anatomy",
        "extra_limbs": "semantic_anatomy",
        "impossible_pose": "semantic_anatomy",
        "animal_anatomy_error": "semantic_anatomy",
        # semantic_object
        "object_shape_error": "semantic_object",
        "extra_objects": "semantic_object",
        "count_error": "semantic_object",
        "illogical_colors": "semantic_object",
        # semantic_spatial
        "scale_inconsistency": "semantic_spatial",
        "floating_objects": "semantic_spatial",
        "penetration_overlap": "semantic_spatial",
        "shadow_mismatch": "semantic_spatial",
        "reflection_error": "semantic_spatial",
        "context_mismatch": "semantic_spatial",
        "time_inconsistency": "semantic_spatial",
        "scene_layout_error": "semantic_spatial",
        # semantic_text
        "text_error": "semantic_text",
        "logo_symbol_error": "semantic_text",
    }

    # 维度 → 正样本 prompt 所需的 semantic_tags（任一匹配即可）
    # 未列出的维度不做过滤，使用全量 prompt 池
    DIMENSION_REQUIRED_TAGS = {
        # 需要人脸
        "face_asymmetry": ["has_face"],
        "expression_mismatch": ["has_face"],
        # 需要人物（全身/半身）
        "body_proportion_error": ["has_person"],
        "extra_limbs": ["has_person"],
        "impossible_pose": ["has_person"],
        # 需要手部
        "hand_malformation": ["has_hand"],
        # 需要动物
        "animal_anatomy_error": ["has_animal"],
        # 需要可数物体
        "count_error": ["has_countable_objects"],
        # 需要多物体（尺度对比）
        "scale_inconsistency": ["has_multiple_objects"],
        # 需要有背景的场景
        "scene_layout_error": ["has_background"],
        "context_mismatch": ["has_background"],
        # 需要反射面
        "reflection_error": ["has_reflective_surface"],
        # 需要自然物体/生物（草地、动物、人、水果等可变色目标）
        "illogical_colors": ["has_person", "has_animal", "has_background"],
    }

    def _get_template_subcategory(self, dimension: str, perspective: str = None) -> str:
        """获取维度对应的模板分组名，用于 prompt_degrader 的 subcategory 参数"""
        return self.DIMENSION_TO_TEMPLATE_GROUP.get(dimension, perspective or dimension)

    def __init__(
        self,
        output_dir: str,
        quality_dimensions_path: str = None,
        model_id: str = "sdxl",
        model_path: str = None,
        max_retries: int = 2,
        enable_routing: bool = False,
        enable_feedback: bool = False,
        knowledge_base_dir: str = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "images").mkdir(exist_ok=True)

        self.model_id = model_id
        self.model_path = model_path
        self.max_retries = max_retries
        self.pair_counter = 0

        # 加载维度配置
        if quality_dimensions_path is None:
            quality_dimensions_path = Path(__file__).parent.parent / "config" / "quality_dimensions_v3.json"
        with open(quality_dimensions_path, 'r', encoding='utf-8') as f:
            self.dimensions_config = json.load(f)

        # 生成配置（由 main 注入）
        self.generation_config = {}

        self.severities = ["moderate", "severe"]

        # Stage 1: SemanticRouter（可选）
        self.router = None
        if enable_routing:
            from semantic_router import SemanticRouter
            self.router = SemanticRouter()
            logger.info("Stage 1 (SemanticRouter) enabled")

        # Stage 6: KnowledgeBase（可选）
        self.knowledge_base = None
        if enable_feedback:
            from knowledge_base import KnowledgeBase
            kb_dir = knowledge_base_dir or str(self.output_dir / "knowledge_base")
            self.knowledge_base = KnowledgeBase(path=kb_dir)
            logger.info(f"Stage 6 (KnowledgeBase) enabled: {kb_dir}")

        # 断点续跑: 已完成的 pair
        self._completed_pairs: Set[str] = set()
        self._load_checkpoint()

        # 统计
        self.stats = {
            "total_pairs": 0,
            "valid_pairs": 0,
            "invalid_pairs": 0,
            "total_attempts": 0,
            "retry_stats": {
                "first_try_success": 0,
                "retry_success": 0,
                "all_retries_failed": 0
            },
            "failure_types": {},
            "by_dimension": {},
            "by_degradation_level": {}
        }

        # 完整日志（所有 pair，含全部 attempts）
        self.full_log = []
        # 数据集（仅成功 pair）
        self.dataset = {
            "metadata": {
                "version": "3.0",
                "generator": "closed_loop_pipeline",
                "created_at": datetime.now().isoformat(),
                "model_id": model_id,
                "max_retries": max_retries
            },
            "pairs": []
        }

    def _load_checkpoint(self):
        """从 full_log.json 加载已完成的 pair，支持断点续跑"""
        full_log_path = self.output_dir / "full_log.json"
        if not full_log_path.exists():
            return

        try:
            with open(full_log_path, 'r', encoding='utf-8') as f:
                existing_log = json.load(f)
            for entry in existing_log:
                if entry.get("success"):
                    key = self._make_pair_key(
                        entry.get("positive_prompt", ""),
                        entry.get("dimension", ""),
                        entry.get("severity", ""),
                    )
                    self._completed_pairs.add(key)
            if self._completed_pairs:
                logger.info(f"Checkpoint: {len(self._completed_pairs)} completed pairs loaded, will skip")
            # 恢复 full_log 和 dataset
            self.full_log = existing_log
            dataset_path = self.output_dir / "dataset.json"
            if dataset_path.exists():
                with open(dataset_path, 'r', encoding='utf-8') as f:
                    self.dataset = json.load(f)
            # 恢复 pair_counter
            self.pair_counter = len(existing_log)
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")

    @staticmethod
    def _make_pair_key(prompt: str, dimension: str, severity: str) -> str:
        return f"{prompt[:100]}|{dimension}|{severity}"

    def get_available_dimensions(self, perspective: str = None) -> List[Dict]:
        """获取可用维度列表（仅 L1, L2 可控维度）"""
        dimensions = []
        perspectives = self.dimensions_config.get("perspectives", {})

        for persp_name, persp_data in perspectives.items():
            if perspective and persp_name != perspective:
                continue
            for dim_name, dim_data in persp_data.get("dimensions", {}).items():
                controllability = dim_data.get("controllability", "L1")
                if controllability in ["L1", "L2"]:
                    dimensions.append({
                        "perspective": persp_name,
                        "dimension": dim_name,
                        "controllability": controllability
                    })
        return dimensions

    def select_random_dimension(self) -> Dict:
        dimensions = self.get_available_dimensions()
        return random.choice(dimensions)

    def select_random_severity(self) -> str:
        return random.choice(self.severities)

    def _resolve_filter(self, name: str) -> List[Dict]:
        """解析过滤器为维度列表，支持 perspective / sub-group / 单维度名"""
        perspectives = self.dimensions_config.get("perspectives", {})

        # 1. perspective (technical_quality, aesthetic_quality, semantic_rationality)
        if name in perspectives:
            return self.get_available_dimensions(perspective=name)

        # 2. sub-group (semantic_anatomy, semantic_object, ...)
        group_keys = self.DIMENSION_SUBGROUPS.get(name)
        if group_keys:
            mapping = self.dimensions_config.get("mapping_from_v2", {})
            dims = []
            seen = set()
            for key in group_keys:
                for dim_name in mapping.get(key, []):
                    if dim_name in seen:
                        continue
                    persp = self._find_perspective(dim_name)
                    if persp:
                        dims.append({"dimension": dim_name, "perspective": persp})
                        seen.add(dim_name)
            return dims

        # 3. 单维度 (blur, hand_malformation, ...)
        persp = self._find_perspective(name)
        if persp:
            return [{"dimension": name, "perspective": persp}]

        logger.warning(f"Unknown filter: {name}")
        return []

    # ------------------------------------------------------------------ #
    #  Style Anchoring (画风锚定)
    # ------------------------------------------------------------------ #

    STYLE_ANCHORS = {
        "realistic": "realistic style, photorealistic, lifelike,",
        "illustration": "illustration style, digital art, stylized,",
        "painting": "painting style, artistic,",
    }

    def _apply_style_anchor(self, negative_prompt: str, detected_style: str) -> str:
        """给 negative prompt 添加画风锚定前缀"""
        anchor = self.STYLE_ANCHORS.get(detected_style, "")
        if anchor:
            return f"{anchor} {negative_prompt}"
        return negative_prompt

    # ------------------------------------------------------------------ #
    #  Core: Closed-Loop Generation
    # ------------------------------------------------------------------ #

    def generate_pair_with_retry(
        self,
        positive_prompt: str,
        dimension: str,
        severity: str,
        seed: int,
        perspective: str = None,
        prompt_signature: Dict = None,
        prefetched_negative: Dict = None,
        positive_image_path: str = None,
    ) -> Dict:
        """
        闭环生成：生成 → 判别 → 失败则诊断修正重试

        Args:
            prefetched_negative: 预取的 LLM 退化结果，有值时跳过首次 LLM 调用
            positive_image_path: 已有正样本路径，有值时跳过正样本生成（跨 severity 复用）

        Returns:
            包含所有 attempt 详情的字典
        """
        pair_id = f"pair_{self.pair_counter:04d}"
        self.pair_counter += 1
        # 输出目录结构：dimension/model_id/severity/pair_id
        pair_dir = self.output_dir / "images" / dimension / self.model_id / severity / pair_id
        pair_dir.mkdir(parents=True, exist_ok=True)

        gen_cfg = self.generation_config
        attempts = []

        # ---- 生成正样本（跨 severity 复用） ---- #
        if positive_image_path and os.path.exists(positive_image_path):
            pos_image_path = positive_image_path
            logger.info(f"[{pair_id}] Reusing positive image: {pos_image_path}")
        else:
            pos_image_path = str(pair_dir / f"positive_seed{seed}.png")
            try:
                image_generator(
                    prompt=positive_prompt,
                    seed=seed,
                    model_id=self.model_id,
                    negative_prompt="",
                    output_path=pos_image_path,
                    model_path=self.model_path or gen_cfg.get('model_path'),
                    steps=gen_cfg.get('steps', 35),
                    cfg=gen_cfg.get('cfg', 7.5),
                    width=gen_cfg.get('width', 1024),
                    height=gen_cfg.get('height', 1024),
                    optimize=gen_cfg.get('optimize', False)
                )
            except Exception as e:
                logger.error(f"[{pair_id}] Positive image generation failed: {e}")
                return self._make_error_result(pair_id, positive_prompt, dimension, severity, seed, str(e))

        # ---- 闭环重试 ---- #
        failed_negative_prompt = None
        current_feedback = None
        last_failure_type = None
        detected_style = None
        last_judge_scores = None
        last_degradation_info = None
        positive_anchored = False  # 正样本是否已做过画风锚定
        negative_anchored = False  # 负样本是否已做过画风锚定
        total_allowed = 1 + self.max_retries  # 首次 + 重试次数

        for attempt_idx in range(total_allowed):
            is_retry = attempt_idx > 0
            attempt_record = {
                "attempt": attempt_idx,
                "is_retry": is_retry
            }

            # Step 1: 获取 negative prompt
            if is_retry and last_failure_type == "style_drift" and detected_style and not negative_anchored:
                # style_drift 首次: 只用画风锚定，不调用 LLM
                negative_anchored = True
                negative_prompt = self._apply_style_anchor(failed_negative_prompt, detected_style)
                degradation_info = last_degradation_info.copy() if last_degradation_info else {}
                degradation_info["rewrite_method"] = "style_anchor_only"
                degradation_info["anchored_style"] = detected_style
                attempt_record["negative_prompt"] = negative_prompt
                attempt_record["degradation_info"] = degradation_info
                attempt_record["rewrite_method"] = "style_anchor_only"
                logger.info(f"[{pair_id}] Style anchor applied: {detected_style}")
            elif is_retry:
                # content_drift / insufficient_effect: 只调用 LLM 重写，不锚定风格
                try:
                    degrade_kwargs = {
                        "positive_prompt": positive_prompt,
                        "subcategory": self._get_template_subcategory(dimension, perspective),
                        "attribute": dimension,
                        "severity": severity,
                        "failed_negative_prompt": failed_negative_prompt,
                        "feedback": current_feedback,
                        "model_id": self.model_id,
                    }
                    # 传入上次 judge 评分
                    if last_judge_scores:
                        degrade_kwargs["judge_scores"] = json.dumps(last_judge_scores)

                    degrade_result = json.loads(prompt_degrader(**degrade_kwargs))
                    negative_prompt = degrade_result["negative_prompt"]
                    degradation_info = degrade_result["degradation_info"]
                    degradation_info["rewrite_method"] = "llm_rewrite"

                    attempt_record["negative_prompt"] = negative_prompt
                    attempt_record["degradation_info"] = degradation_info
                    attempt_record["rewrite_method"] = "llm_rewrite"
                    logger.info(f"[{pair_id}] LLM rewrite (failure={last_failure_type})")
                except Exception as e:
                    logger.error(f"[{pair_id}] Prompt degradation failed (attempt {attempt_idx}): {e}")
                    attempt_record["status"] = "error"
                    attempt_record["error"] = f"degradation_failed: {e}"
                    attempts.append(attempt_record)
                    continue
            else:
                # 首次生成: 使用预取结果或调用 LLM 退化
                try:
                    if prefetched_negative:
                        degrade_result = prefetched_negative
                        logger.info(f"[{pair_id}] Using prefetched negative prompt")
                    else:
                        degrade_kwargs = {
                            "positive_prompt": positive_prompt,
                            "subcategory": self._get_template_subcategory(dimension, perspective),
                            "attribute": dimension,
                            "severity": severity,
                            "model_id": self.model_id,
                        }
                        if prompt_signature:
                            degrade_kwargs["prompt_signature"] = json.dumps(prompt_signature)

                        degrade_result = json.loads(prompt_degrader(**degrade_kwargs))
                    negative_prompt = degrade_result["negative_prompt"]
                    degradation_info = degrade_result["degradation_info"]

                    attempt_record["negative_prompt"] = negative_prompt
                    attempt_record["degradation_info"] = degradation_info
                    attempt_record["rewrite_method"] = "llm"
                except Exception as e:
                    logger.error(f"[{pair_id}] Prompt degradation failed (attempt {attempt_idx}): {e}")
                    attempt_record["status"] = "error"
                    attempt_record["error"] = f"degradation_failed: {e}"
                    attempts.append(attempt_record)
                    continue

            # Step 1.5: 检测 SKIP_THIS_PROMPT
            if "SKIP_THIS_PROMPT" in negative_prompt:
                logger.info(f"[{pair_id}] LLM returned SKIP_THIS_PROMPT, skipping pair")
                attempt_record["status"] = "skipped"
                attempt_record["failure_type"] = "prompt_incompatible"
                attempts.append(attempt_record)
                break

            # Step 2: 生成负样本图像
            neg_image_path = str(pair_dir / f"attempt_{attempt_idx}_negative_seed{seed}.png")
            try:
                image_generator(
                    prompt=negative_prompt,
                    seed=seed,
                    model_id=self.model_id,
                    negative_prompt="",
                    output_path=neg_image_path,
                    model_path=self.model_path or gen_cfg.get('model_path'),
                    steps=gen_cfg.get('steps', 35),
                    cfg=gen_cfg.get('cfg', 7.5),
                    width=gen_cfg.get('width', 1024),
                    height=gen_cfg.get('height', 1024),
                    optimize=gen_cfg.get('optimize', False)
                )
                attempt_record["negative_image_path"] = neg_image_path
            except Exception as e:
                logger.error(f"[{pair_id}] Negative image generation failed (attempt {attempt_idx}): {e}")
                attempt_record["status"] = "error"
                attempt_record["error"] = f"generation_failed: {e}"
                attempts.append(attempt_record)
                continue

            # Step 3: VLM 判别
            try:
                judge_result_str = degradation_judge(
                    positive_image_path=pos_image_path,
                    negative_image_path=neg_image_path,
                    expected_dimension=dimension,
                    positive_prompt=positive_prompt,
                    negative_prompt=negative_prompt,
                    expected_attribute=dimension
                )
                validation = json.loads(judge_result_str)
                attempt_record["validation"] = validation
            except Exception as e:
                logger.error(f"[{pair_id}] Judge failed (attempt {attempt_idx}): {e}")
                attempt_record["status"] = "error"
                attempt_record["error"] = f"judge_failed: {e}"
                attempts.append(attempt_record)
                continue

            # Step 4: 判定结果
            if validation.get("valid", False):
                attempt_record["status"] = "success"
                attempts.append(attempt_record)
                logger.info(f"[{pair_id}] Valid pair (attempt {attempt_idx})")
                break
            else:
                attempt_record["status"] = "failed"
                failure_type = validation.get("failure", "unknown")
                attempt_record["failure_type"] = failure_type
                attempts.append(attempt_record)

                # 更新失败类型统计
                self.stats["failure_types"][failure_type] = self.stats["failure_types"].get(failure_type, 0) + 1

                # 内容不兼容：prompt 本身无法承载该维度，直接跳过
                if failure_type == "positive_content_mismatch":
                    logger.warning(f"[{pair_id}] Positive content incompatible with {dimension}, skipping (not a style issue)")
                    break

                # 风格不兼容: positive_incompatible - 修正正样本并重新开始
                if failure_type == "positive_incompatible":
                    if positive_anchored:
                        logger.warning(f"[{pair_id}] Positive still incompatible after style anchoring, giving up")
                        break

                    recommended_style = validation.get("recommended_style")
                    if not recommended_style:
                        logger.warning(f"[{pair_id}] VLM did not provide recommended_style, defaulting to realistic")
                        recommended_style = 'realistic'

                    logger.info(f"[{pair_id}] Positive sample incompatible with {dimension}: {validation.get('notes', '')}")
                    logger.info(f"[{pair_id}] VLM recommends: {recommended_style} style")

                    # 根据 VLM 建议的风格添加定语
                    style_anchors = {
                        'realistic': 'realistic style, photorealistic, ',
                        'illustration': 'illustration style, ',
                        'painting': 'painting style, '
                    }

                    style_anchor = style_anchors.get(recommended_style, 'realistic style, photorealistic, ')
                    positive_prompt = style_anchor + positive_prompt
                    positive_anchored = True

                    logger.info(f"[{pair_id}] Adding style anchor to positive prompt: {style_anchor}")

                    # 重新生成正样本
                    try:
                        image_generator(
                            prompt=positive_prompt,
                            seed=seed,
                            model_id=self.model_id,
                            negative_prompt="",
                            output_path=pos_image_path,
                            model_path=self.model_path or gen_cfg.get('model_path'),
                            steps=gen_cfg.get('steps', 35),
                            cfg=gen_cfg.get('cfg', 7.5),
                            width=gen_cfg.get('width', 1024),
                            height=gen_cfg.get('height', 1024),
                            optimize=gen_cfg.get('optimize', False)
                        )
                        logger.info(f"[{pair_id}] Positive sample regenerated with style anchor")

                        # 重置状态，重新开始整个流程
                        failed_negative_prompt = None
                        last_failure_type = None
                        last_degradation_info = None
                        detected_style = None
                        current_feedback = None

                        # 继续下一次尝试（会重新退化 → 生成负样本 → Judge）
                        continue

                    except Exception as e:
                        logger.error(f"[{pair_id}] Failed to regenerate positive sample: {e}")
                        attempt_record["error"] = f"positive_regeneration_failed: {e}"
                        break

                # 保存状态，准备重试
                failed_negative_prompt = negative_prompt
                last_failure_type = failure_type
                last_degradation_info = degradation_info
                detected_style = validation.get("style_type")
                last_judge_scores = validation.get("scores")
                current_feedback = self._build_feedback(validation)

                if attempt_idx < total_allowed - 1:
                    if failure_type == "style_drift" and detected_style:
                        logger.info(f"[{pair_id}] Style drift detected (style={detected_style}), will apply style anchor...")
                    else:
                        logger.info(f"[{pair_id}] Failed (type={failure_type}), retrying with LLM ({attempt_idx+1}/{self.max_retries})...")
                else:
                    logger.warning(f"[{pair_id}] All {total_allowed} attempts exhausted, marking as failed")

        # ---- 汇总结果 ---- #
        final_attempt = attempts[-1] if attempts else {}
        success = final_attempt.get("status") == "success"

        result = {
            "pair_id": pair_id,
            "positive_prompt": positive_prompt,
            "positive_image_path": pos_image_path,
            "dimension": dimension,
            "perspective": perspective,
            "severity": severity,
            "seed": seed,
            "success": success,
            "total_attempts": len(attempts),
            "attempts": attempts,
            "final_negative_prompt": final_attempt.get("negative_prompt"),
            "final_negative_image_path": final_attempt.get("negative_image_path"),
            "final_validation": final_attempt.get("validation")
        }

        # 更新统计
        self._update_stats(result)

        return result

    def _build_feedback(self, validation: Dict) -> str:
        """根据 judge 诊断构建给退化 LLM 的反馈"""
        failure_type = validation.get("failure")
        notes = validation.get("notes", "")

        parts = []

        if failure_type == "positive_content_mismatch":
            parts.append("The positive image content cannot carry this degradation dimension (e.g., no face for expression_mismatch). Skip this prompt.")
        elif failure_type == "positive_incompatible":
            parts.append("The positive image style is not suitable for this degradation dimension. This should not happen in retry - check logic.")
        elif failure_type == "style_drift":
            parts.append("The negative image changed rendering style. Keep the EXACT SAME style as the positive prompt. Only degrade quality in the target dimension.")
        elif failure_type == "content_drift":
            parts.append("The negative image changed subject/content too much. Keep the SAME subject and scene composition. Only introduce degradation in the target dimension.")
        elif failure_type == "insufficient_effect":
            parts.append("The degradation effect is too subtle or invisible. Make the degradation STRONGER and MORE OBVIOUS. Use more explicit degradation keywords.")
        elif failure_type:
            parts.append(f"Failure type: {failure_type}")

        if notes:
            parts.append(f"VLM notes: {notes}")

        return " | ".join(parts) if parts else "Previous attempt was rejected. Please try a different approach."

    def _update_stats(self, result: Dict):
        """更新统计数据"""
        self.stats["total_pairs"] += 1
        self.stats["total_attempts"] += result["total_attempts"]

        dim = result["dimension"]
        if dim not in self.stats["by_dimension"]:
            self.stats["by_dimension"][dim] = {"valid": 0, "invalid": 0, "attempts": 0}
        self.stats["by_dimension"][dim]["attempts"] += result["total_attempts"]

        if result["success"]:
            self.stats["valid_pairs"] += 1
            self.stats["by_dimension"][dim]["valid"] += 1

            # 新格式不再有 degradation_level，记录为 "valid"
            self.stats["by_degradation_level"]["valid"] = self.stats["by_degradation_level"].get("valid", 0) + 1

            if result["total_attempts"] == 1:
                self.stats["retry_stats"]["first_try_success"] += 1
            else:
                self.stats["retry_stats"]["retry_success"] += 1
        else:
            self.stats["invalid_pairs"] += 1
            self.stats["by_dimension"][dim]["invalid"] += 1
            self.stats["retry_stats"]["all_retries_failed"] += 1

    def _make_error_result(self, pair_id, positive_prompt, dimension, severity, seed, error_msg):
        """生成错误结果"""
        self.stats["total_pairs"] += 1
        self.stats["invalid_pairs"] += 1
        self.stats["failure_types"]["error"] = self.stats["failure_types"].get("error", 0) + 1
        return {
            "pair_id": pair_id,
            "positive_prompt": positive_prompt,
            "dimension": dimension,
            "severity": severity,
            "seed": seed,
            "success": False,
            "total_attempts": 0,
            "attempts": [],
            "error": error_msg
        }

    # ------------------------------------------------------------------ #
    #  Pipeline Orchestration
    # ------------------------------------------------------------------ #

    def run(
        self,
        prompts: List,
        num_pairs_per_prompt: int = 5,
        dimensions: List[str] = None,
        perspectives: List[str] = None,
        base_seed: int = 42,
        systematic: bool = False,
    ) -> Dict:
        """运行闭环数据生成流水线

        Args:
            prompts: List[str] 或 List[dict]，dict 格式 {"prompt": str, "tags": list}
        """
        if systematic:
            return self._run_systematic(prompts, num_pairs_per_prompt, dimensions, perspectives, base_seed)
        logger.info(f"Starting closed-loop pipeline: {len(prompts)} prompts x {num_pairs_per_prompt} pairs (max_retries={self.max_retries})")
        if self.router:
            logger.info("Stage 1 (SemanticRouter) active")
        if self.knowledge_base:
            logger.info("Stage 6 (KnowledgeBase) active")

        seed_counter = base_seed
        total_target = len(prompts) * num_pairs_per_prompt
        skipped_routing = 0
        skipped_checkpoint = 0
        skipped_paused = 0
        skipped_incompat = 0

        for prompt_idx, prompt_item in enumerate(prompts):
            positive_prompt = prompt_item["prompt"] if isinstance(prompt_item, dict) else prompt_item
            logger.info(f"\n{'='*60}")
            logger.info(f"Prompt [{prompt_idx+1}/{len(prompts)}]: {positive_prompt[:80]}...")

            # Stage 1: 预分析 prompt 语义标签
            signature = None
            if self.router:
                signature = self.router.analyze(positive_prompt)

            for pair_idx in range(num_pairs_per_prompt):
                current = prompt_idx * num_pairs_per_prompt + pair_idx + 1
                logger.info(f"\n--- Target pair {current}/{total_target} ---")

                # 选择维度
                if dimensions:
                    dim_name = random.choice(dimensions)
                    persp = self._find_perspective(dim_name)
                    dim_info = {"dimension": dim_name, "perspective": persp}
                elif perspectives:
                    persp = random.choice(perspectives)
                    available_dims = self.get_available_dimensions(perspective=persp)
                    if not available_dims:
                        logger.warning(f"No available dimensions in perspective '{persp}', using random")
                        dim_info = self.select_random_dimension()
                    else:
                        dim_info = random.choice(available_dims)
                else:
                    dim_info = self.select_random_dimension()

                severity = self.select_random_severity()
                seed = seed_counter
                seed_counter += 1

                dimension = dim_info["dimension"]

                # 断点续跑：跳过已完成的 pair
                pair_key = self._make_pair_key(positive_prompt, dimension, severity)
                if pair_key in self._completed_pairs:
                    skipped_checkpoint += 1
                    logger.info(f"[Checkpoint] Skipping completed: {dimension}/{severity}")
                    continue

                # Stage 1: SemanticRouter 兼容性筛选
                if self.router and signature:
                    compatible, reason = self.router.is_compatible(signature, dimension)
                    if not compatible:
                        skipped_routing += 1
                        logger.info(f"[Stage 1] Skipped: {dimension} ({reason})")
                        continue

                # Stage 6: Circuit Breaker
                if self.knowledge_base and self.knowledge_base.is_dimension_paused(dimension):
                    skipped_paused += 1
                    logger.info(f"[Stage 6] Skipped paused: {dimension}")
                    continue

                # Stage 6: 模型兼容性
                if self.knowledge_base and not self.knowledge_base.is_model_compatible(self.model_id, dimension):
                    skipped_incompat += 1
                    logger.info(f"[Stage 6] Skipped incompat: {self.model_id}/{dimension}")
                    continue

                # Stage 2-5: 闭环生成
                result = self.generate_pair_with_retry(
                    positive_prompt=positive_prompt,
                    dimension=dimension,
                    severity=severity,
                    seed=seed,
                    perspective=dim_info.get("perspective"),
                    prompt_signature=signature,
                )

                # Stage 6: 记录到知识库
                if self.knowledge_base:
                    validation = result.get("final_validation") or {}
                    self.knowledge_base.report_outcome(
                        dimension=dimension,
                        severity=severity,
                        model_id=self.model_id,
                        template_id=f"{dimension}_{severity}",
                        success=result["success"],
                        scores=validation.get("scores"),
                        failure_type=validation.get("failure"),
                    )

                # 记录到完整日志
                self.full_log.append(result)

                # 保存成功 pair
                if result["success"]:
                    final = result["attempts"][-1]
                    pair_data = {
                        "id": result["pair_id"],
                        "positive": {
                            "prompt": positive_prompt,
                            "image_path": result["positive_image_path"],
                            "seed": seed
                        },
                        "negative": {
                            "prompt": result["final_negative_prompt"],
                            "image_path": result["final_negative_image_path"],
                            "seed": seed
                        },
                        "degradation": final.get("degradation_info", {}),
                        "validation": result["final_validation"],
                        "retries_needed": result["total_attempts"] - 1
                    }
                    self.dataset["pairs"].append(pair_data)

                # 定期保存
                if current % 10 == 0:
                    self._save_results()
                    if self.knowledge_base:
                        self.knowledge_base.save()

        # 最终保存
        self._save_results()
        if self.knowledge_base:
            self.knowledge_base.save()

        if skipped_routing or skipped_checkpoint or skipped_paused or skipped_incompat:
            logger.info(
                f"Skipped: routing={skipped_routing}, checkpoint={skipped_checkpoint}, "
                f"paused={skipped_paused}, incompat={skipped_incompat}"
            )

        return {
            "stats": self.stats,
            "output_dir": str(self.output_dir),
            "dataset_path": str(self.output_dir / "dataset.json"),
            "full_log_path": str(self.output_dir / "full_log.json"),
            "report_path": str(self.output_dir / "validation_report.json")
        }

    def _prefetch_negatives(
        self,
        tasks: List[Dict],
        max_workers: int = 4,
    ) -> Dict[str, Dict]:
        """并发预取一批 LLM 退化 prompt

        Args:
            tasks: [{positive_prompt, dimension, perspective, severity, model_id}, ...]
            max_workers: 线程池大小（控制 API 并发）

        Returns:
            {cache_key: degrade_result_dict, ...}  失败的 key 不包含在内
        """
        cache = {}
        if not tasks:
            return cache

        def _call(t):
            key = t["cache_key"]
            try:
                result_str = prompt_degrader(
                    positive_prompt=t["positive_prompt"],
                    subcategory=self._get_template_subcategory(t["dimension"], t["perspective"]),
                    attribute=t["dimension"],
                    severity=t["severity"],
                    model_id=t["model_id"],
                )
                return key, json.loads(result_str)
            except Exception as e:
                logger.warning(f"[Prefetch] Failed for {t['dimension']}/{t['severity']}: {e}")
                return key, None

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(_call, t) for t in tasks]
            for fut in concurrent.futures.as_completed(futures):
                key, result = fut.result()
                if result is not None:
                    cache[key] = result

        logger.info(f"[Prefetch] {len(cache)}/{len(tasks)} negative prompts cached")
        return cache

    def _run_systematic(
        self,
        prompts: List,
        num_prompts_per_dim: int,
        dimensions: List[str] = None,
        perspectives: List[str] = None,
        base_seed: int = 42,
    ) -> Dict:
        """系统化遍历：维度 → prompt → severity，确保每个维度都被覆盖

        优化：
        1. LLM 预批量：每个维度开始前，并发预取所有 (prompt, severity) 的退化 prompt
        2. 正样本复用：同 prompt 不同 severity 共享同一个 seed 和正样本图像
        """

        # 1. 统一解析过滤器
        target_dims = []
        filters = dimensions or perspectives or []
        seen = set()
        for f in filters:
            for d in self._resolve_filter(f):
                if d["dimension"] not in seen:
                    target_dims.append(d)
                    seen.add(d["dimension"])
        if not filters:
            target_dims = self.get_available_dimensions()

        total_target = len(target_dims) * num_prompts_per_dim * len(self.severities)
        logger.info(
            f"Systematic mode: {len(target_dims)} dims × {num_prompts_per_dim} prompts × "
            f"{len(self.severities)} severities = {total_target} pairs (max_retries={self.max_retries})"
        )

        # 用 model_id 扰动 seed，确保不同模型选不同 prompt
        model_seed = base_seed + hash(self.model_id) % 10000
        rng = random.Random(model_seed)
        seed_counter = base_seed
        current = 0

        for dim_info in target_dims:
            dimension = dim_info["dimension"]
            persp = dim_info.get("perspective")

            # Circuit breaker（维度级别）
            if self.knowledge_base and self.knowledge_base.is_dimension_paused(dimension):
                skipped = num_prompts_per_dim * len(self.severities)
                current += skipped
                logger.info(f"[CB] Paused: {dimension}, skipping {skipped} pairs")
                continue

            # 按标签过滤 prompt 池，再随机选取 N 个
            required_tags = self.DIMENSION_REQUIRED_TAGS.get(dimension, [])
            if required_tags:
                pool = [p for p in prompts if any(t in p.get("semantic_tags", p.get("tags", [])) for t in required_tags)]
                if not pool:
                    logger.warning(f"[TagFilter] {dimension} requires {required_tags} but no matching prompts, using full pool")
                    pool = prompts
                else:
                    logger.info(f"[TagFilter] {dimension}: {len(pool)}/{len(prompts)} prompts match {required_tags}")
            else:
                pool = prompts
            selected = rng.sample(pool, min(num_prompts_per_dim, len(pool)))
            dim_prompts = [p["prompt"] if isinstance(p, dict) else p for p in selected]

            # --- 优化1: LLM 预批量 --- #
            prefetch_tasks = []
            for positive_prompt in dim_prompts:
                for severity in self.severities:
                    pair_key = self._make_pair_key(positive_prompt, dimension, severity)
                    if pair_key in self._completed_pairs:
                        continue
                    cache_key = f"{positive_prompt[:100]}|{dimension}|{severity}"
                    prefetch_tasks.append({
                        "cache_key": cache_key,
                        "positive_prompt": positive_prompt,
                        "dimension": dimension,
                        "perspective": persp,
                        "severity": severity,
                        "model_id": self.model_id,
                    })

            prefetch_cache = self._prefetch_negatives(prefetch_tasks) if prefetch_tasks else {}

            # --- 遍历 prompt → severity（同 prompt 共享 seed/正样本） --- #
            for positive_prompt in dim_prompts:
                # 同 prompt 共享一个 seed
                prompt_seed = seed_counter
                seed_counter += 1
                shared_pos_path = None  # 首个 severity 生成后复用

                for severity in self.severities:
                    current += 1
                    logger.info(f"\n--- [{current}/{total_target}] {dimension}/{severity} ---")
                    logger.info(f"Prompt: {positive_prompt[:80]}...")

                    # 断点续跑
                    pair_key = self._make_pair_key(positive_prompt, dimension, severity)
                    if pair_key in self._completed_pairs:
                        logger.info(f"[Checkpoint] Skip: {dimension}/{severity}")
                        continue

                    # 从预取缓存取退化结果
                    cache_key = f"{positive_prompt[:100]}|{dimension}|{severity}"
                    prefetched = prefetch_cache.get(cache_key)

                    result = self.generate_pair_with_retry(
                        positive_prompt=positive_prompt,
                        dimension=dimension,
                        severity=severity,
                        seed=prompt_seed,
                        perspective=persp,
                        prefetched_negative=prefetched,
                        positive_image_path=shared_pos_path,
                    )

                    # 记录正样本路径供后续 severity 复用
                    if shared_pos_path is None and result.get("positive_image_path"):
                        shared_pos_path = result["positive_image_path"]

                    # KnowledgeBase 记录
                    if self.knowledge_base:
                        validation = result.get("final_validation") or {}
                        self.knowledge_base.report_outcome(
                            dimension=dimension,
                            severity=severity,
                            model_id=self.model_id,
                            template_id=f"{dimension}_{severity}",
                            success=result["success"],
                            scores=validation.get("scores"),
                            failure_type=validation.get("failure"),
                        )

                    self.full_log.append(result)

                    if result["success"]:
                        final = result["attempts"][-1]
                        self.dataset["pairs"].append({
                            "id": result["pair_id"],
                            "positive": {
                                "prompt": positive_prompt,
                                "image_path": result["positive_image_path"],
                                "seed": prompt_seed
                            },
                            "negative": {
                                "prompt": result["final_negative_prompt"],
                                "image_path": result["final_negative_image_path"],
                                "seed": prompt_seed
                            },
                            "degradation": final.get("degradation_info", {}),
                            "validation": result["final_validation"],
                            "retries_needed": result["total_attempts"] - 1
                        })

                    if current % 10 == 0:
                        self._save_results()
                        if self.knowledge_base:
                            self.knowledge_base.save()

        self._save_results()
        if self.knowledge_base:
            self.knowledge_base.save()

        return {
            "stats": self.stats,
            "output_dir": str(self.output_dir),
            "dataset_path": str(self.output_dir / "dataset.json"),
            "full_log_path": str(self.output_dir / "full_log.json"),
            "report_path": str(self.output_dir / "validation_report.json")
        }

    def _find_perspective(self, dimension: str) -> Optional[str]:
        """根据维度名查找所属 perspective"""
        for persp_name, persp_data in self.dimensions_config.get("perspectives", {}).items():
            if dimension in persp_data.get("dimensions", {}):
                return persp_name
        return None

    def _save_results(self):
        """保存数据集、完整日志和统计报告"""
        # 1. dataset.json（仅成功 pair）
        self.dataset["metadata"]["total_pairs"] = len(self.dataset["pairs"])
        self.dataset["metadata"]["completed_at"] = datetime.now().isoformat()
        with open(self.output_dir / "dataset.json", 'w', encoding='utf-8') as f:
            json.dump(self.dataset, f, ensure_ascii=False, indent=2)

        # 2. full_log.json（所有 pair 含全部 attempts）
        with open(self.output_dir / "full_log.json", 'w', encoding='utf-8') as f:
            json.dump(self.full_log, f, ensure_ascii=False, indent=2)

        # 3. validation_report.json
        total = max(self.stats["total_pairs"], 1)
        validation_rate = self.stats["valid_pairs"] / total
        avg_attempts = self.stats["total_attempts"] / total if total else 0

        report = {
            "summary": {
                "total_pairs": self.stats["total_pairs"],
                "valid_pairs": self.stats["valid_pairs"],
                "invalid_pairs": self.stats["invalid_pairs"],
                "validation_rate": round(validation_rate, 4),
                "avg_attempts_per_pair": round(avg_attempts, 2),
                "max_retries_setting": self.max_retries
            },
            "retry_stats": self.stats["retry_stats"],
            "failure_types": self.stats["failure_types"],
            "by_degradation_level": self.stats["by_degradation_level"],
            "by_dimension": self.stats["by_dimension"],
            "generated_at": datetime.now().isoformat()
        }
        with open(self.output_dir / "validation_report.json", 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved: {self.stats['valid_pairs']}/{self.stats['total_pairs']} valid pairs ({validation_rate*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="AIGC Quality Degradation Data Generation Pipeline (Closed-Loop)")

    parser.add_argument("--source_prompts", "--positive_source", type=str, required=True,
                        help="Path to JSON file containing source prompts")
    parser.add_argument("--output_dir", type=str, default="/root/autodl-tmp/pipeline_output",
                        help="Output directory")
    parser.add_argument("--quality_dimensions", type=str, default=None,
                        help="Path to quality dimensions JSON")

    # 维度筛选
    parser.add_argument("--subcategory_filter", type=str, default=None,
                        help="Filter by perspective (e.g., 'technical_quality')")
    parser.add_argument("--attribute_filter", type=str, default=None,
                        help="Filter by dimension (e.g., 'blur')")
    parser.add_argument("--severities", type=str, default="moderate,severe",
                        help="Severities to use, comma-separated (default: moderate,severe)")

    # 数量配置
    parser.add_argument("--num_pairs_per_prompt", type=int, default=3,
                        help="Negative pairs per prompt (default: 3)")
    parser.add_argument("--max_prompts", type=int, default=None,
                        help="Maximum prompts to process")
    parser.add_argument("--shuffle", action="store_true",
                        help="Shuffle prompts before processing")
    parser.add_argument("--seed", "--base_seed", type=int, default=42,
                        help="Base random seed (default: 42)")

    # 闭环重试配置
    parser.add_argument("--max_retries", type=int, default=2,
                        help="Max retries per pair on judge failure (default: 2)")
    parser.add_argument("--systematic", action="store_true",
                        help="Systematic mode: iterate all dims × prompts × severities deterministically")

    # 图像生成配置
    parser.add_argument("--model_id", type=str, default="sdxl", choices=["sdxl", "flux", "flux-schnell"],
                        help="Generation model: sdxl, flux or flux-schnell (default: sdxl)")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Model path (SDXL: /root/ckpts/sd_xl_base_1.0.safetensors, Flux: /root/autodl-tmp/flux-1-dev)")
    parser.add_argument("--steps", type=int, default=35,
                        help="Inference steps (default: 35 for SDXL, 28 for Flux)")
    parser.add_argument("--cfg", type=float, default=7.5,
                        help="CFG scale (default: 7.5 for SDXL, 3.5 for Flux)")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--optimize", action="store_true",
                        help="Enable speed optimization for Flux-Schnell (T5 4-bit + FP8 + compile)")

    # Prompt 筛选
    parser.add_argument("--model_filter", type=str, default="sdxl",
                        help="Filter prompts by model name")

    # Stage 1: 标签筛选（可选）
    parser.add_argument("--tagged_prompts", type=str, default=None,
                        help="Path to tagged prompts JSON (enables Stage 1 filtering)")
    parser.add_argument("--tag_requirements", type=str, default=None,
                        help="Path to tag requirements JSON (requires --tagged_prompts)")

    # Stage 6: 知识库（可选）
    parser.add_argument("--knowledge_base_dir", type=str, default=None,
                        help="Path to knowledge base directory (enables Stage 6 feedback)")

    args = parser.parse_args()

    severities = [s.strip() for s in args.severities.split(',')]

    # 加载 prompts
    with open(args.source_prompts, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if isinstance(data, list):
        prompts_data = data
    elif isinstance(data, dict) and "prompts" in data:
        prompts_data = data["prompts"]
    else:
        raise ValueError("Invalid prompts file format")

    prompts = []
    for item in prompts_data:
        if isinstance(item, str):
            prompts.append({"prompt": item, "tags": []})
        elif isinstance(item, dict):
            model = item.get("model", "")
            if args.model_filter and args.model_filter not in model:
                continue
            prompts.append({
                "prompt": item.get("prompt", ""),
                "tags": item.get("semantic_tags", []),
            })

    if args.shuffle:
        random.shuffle(prompts)

    if args.max_prompts:
        prompts = prompts[:args.max_prompts]

    logger.info(f"Loaded {len(prompts)} prompts from {args.source_prompts}")

    # 创建闭环流水线
    pipeline = DataGenerationPipeline(
        output_dir=args.output_dir,
        quality_dimensions_path=args.quality_dimensions,
        model_id=args.model_id,
        model_path=args.model_path,
        max_retries=args.max_retries,
        enable_feedback=bool(args.knowledge_base_dir),
        knowledge_base_dir=args.knowledge_base_dir,
    )

    pipeline.generation_config = {
        "model_path": args.model_path,
        "steps": args.steps,
        "cfg": args.cfg,
        "width": args.width,
        "height": args.height,
        "optimize": args.optimize
    }
    pipeline.severities = severities

    dimensions = [x.strip() for x in args.attribute_filter.split(',')] if args.attribute_filter else None
    perspectives = [x.strip() for x in args.subcategory_filter.split(',')] if args.subcategory_filter else None

    results = pipeline.run(
        prompts=prompts,
        num_pairs_per_prompt=args.num_pairs_per_prompt,
        dimensions=dimensions,
        perspectives=perspectives,
        base_seed=args.seed,
        systematic=args.systematic,
    )

    logger.info(f"\nResults: {results['output_dir']}")
    logger.info(f"Dataset ({pipeline.stats['valid_pairs']} pairs): {results['dataset_path']}")
    logger.info(f"Full log: {results['full_log_path']}")
    logger.info(f"Report: {results['report_path']}")


if __name__ == "__main__":
    main()
