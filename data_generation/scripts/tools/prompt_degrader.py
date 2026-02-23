"""
Prompt Degrader Tool
封装现有的 LLMPromptDegradation，使用 smolagents @tool 装饰器
"""

import json
import sys
from pathlib import Path
from typing import Optional

# 添加 scripts 目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from smolagents import tool

# 单例模式：避免重复初始化
_degrader_instance = None


def _get_degrader():
    """获取 LLMPromptDegradation 单例"""
    global _degrader_instance
    if _degrader_instance is None:
        from llm_prompt_degradation import LLMPromptDegradation

        config_dir = Path(__file__).parent.parent.parent / "config"
        _degrader_instance = LLMPromptDegradation(
            llm_config_path=str(config_dir / "llm_config.yaml"),
            quality_dimensions_path=str(config_dir / "quality_dimensions_v3.json")
        )
    return _degrader_instance


@tool
def prompt_degrader(
    positive_prompt: str,
    subcategory: str,
    attribute: Optional[str] = None,
    severity: str = "moderate",
    failed_negative_prompt: Optional[str] = None,
    feedback: Optional[str] = None,
    judge_scores: Optional[str] = None,
    model_id: str = "sdxl",
    prompt_signature: Optional[str] = None,
) -> str:
    """
    Generate a degraded (lower quality) version of the input prompt using LLM.

    This tool uses GPT-4o to rewrite the prompt with quality degradation in the
    specified dimension. The degradation strategy is determined by templates
    loaded from config/prompt_templates_v3/.

    For retry mode, provide failed_negative_prompt and feedback to help the LLM
    correct the previous failure.

    Args:
        positive_prompt: The original high-quality prompt to degrade.
        subcategory: The degradation category. Available options include:
                     - technical_quality: blur, overexposure, underexposure, low_contrast, etc.
                     - aesthetic_quality: awkward_positioning, flat_lighting, color_clash, etc.
                     - semantic_rationality: hand_malformation, face_asymmetry, object_shape_error, etc.
        attribute: Specific attribute to degrade (e.g., 'blur', 'hand_malformation').
                   If None, randomly selects one from the subcategory.
        severity: Degradation intensity - 'mild', 'moderate', or 'severe'.
        failed_negative_prompt: [Retry mode] The negative prompt that failed in previous attempt.
        feedback: [Retry mode] Description of why the previous attempt failed.
        judge_scores: [Retry mode] JSON string of multi-dimensional scores from VLM judge.
        model_id: Target generation model identifier (e.g., 'sdxl', 'flux-schnell').
        prompt_signature: JSON string of semantic tags from SemanticRouter.

    Returns:
        JSON string containing:
        - negative_prompt: The degraded prompt
        - degradation_info: Metadata about the degradation applied including
          category, subcategory, attribute, severity, method, and is_retry flag
    """
    degrader = _get_degrader()

    # 解析可选的 JSON 字符串参数
    parsed_scores = None
    if judge_scores:
        try:
            parsed_scores = json.loads(judge_scores) if isinstance(judge_scores, str) else judge_scores
        except (json.JSONDecodeError, TypeError):
            pass

    parsed_sig = None
    if prompt_signature:
        try:
            parsed_sig = json.loads(prompt_signature) if isinstance(prompt_signature, str) else prompt_signature
        except (json.JSONDecodeError, TypeError):
            pass

    negative_prompt, degradation_info = degrader.generate_negative_prompt(
        positive_prompt=positive_prompt,
        subcategory=subcategory,
        attribute=attribute,
        severity=severity,
        failed_negative_prompt=failed_negative_prompt,
        feedback=feedback,
        judge_scores=parsed_scores,
        model_id=model_id,
        prompt_signature=parsed_sig,
    )

    return json.dumps({
        "negative_prompt": negative_prompt,
        "degradation_info": degradation_info
    }, ensure_ascii=False)


def cleanup():
    """清理资源"""
    global _degrader_instance
    _degrader_instance = None


if __name__ == "__main__":
    # 测试
    result = prompt_degrader(
        positive_prompt="a beautiful sunset over the ocean, masterpiece, high quality",
        subcategory="technical_quality",
        attribute="blur",
        severity="moderate"
    )
    print("Result:", result)
