"""
Degradation Judge Tool
使用 VLM (通过 OpenAI 兼容 API) 判别图像对是否展示了有效的质量退化
"""

import json
import os
import base64
import logging
import yaml
import time
import httpx
import re
from pathlib import Path
from typing import Optional, Dict, Any
from PIL import Image
import io
from functools import lru_cache

from smolagents import tool

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_GENERATION_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = DATA_GENERATION_ROOT / "config"
PROMPT_TEMPLATES_ROOT = CONFIG_ROOT / "prompt_templates_v3"
ACTIVE_TAXONOMY_PATH = CONFIG_ROOT / "quality_dimensions_active.json"
JUDGE_COMPATIBILITY_HINTS_PATH = CONFIG_ROOT / "judge_compatibility_hints_v1.yaml"
DEGRADATION_DIMENSIONS_DOC_PATH = DATA_GENERATION_ROOT / "docs" / "degradation_dimensions.md"

# 尝试导入 OpenAI
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("openai not installed. Install with: pip install openai")


# 单例配置和客户端
_config = None
_client = None


def _load_config() -> Dict:
    """加载判别配置"""
    global _config
    if _config is not None:
        return _config

    override_path = os.environ.get("JUDGE_CONFIG_PATH")
    if override_path:
        config_path = Path(override_path).expanduser().resolve()
    else:
        config_path = Path(__file__).parent.parent.parent / "config" / "judge_config.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        _config = yaml.safe_load(f)

    return _config


def _get_client():
    """获取 OpenAI 兼容客户端"""
    global _client
    if _client is not None:
        return _client

    if not OPENAI_AVAILABLE:
        raise ImportError("openai not installed")

    config = _load_config()
    api_key = config['vlm'].get('api_key')
    api_base = config['vlm'].get('api_base')
    timeout = config['vlm'].get('timeout')

    if not api_key:
        raise ValueError("API key not set in judge_config.yaml")

    _client = OpenAI(
        api_key=api_key,
        base_url=api_base,
        http_client=httpx.Client(
            timeout=float(timeout) if timeout else None,
            trust_env=False,
        ),
    )

    model_name = config['vlm'].get('model', 'gemini-3-pro-preview')
    logger.info(f"Initialized OpenAI-compatible client with model: {model_name}, base_url: {api_base}")
    return _client


def _load_image_as_pil(image_path: str) -> Image.Image:
    """加载图像为 PIL Image"""
    return Image.open(image_path).convert('RGB')


def _image_to_base64(img: Image.Image, jpeg_quality: int = 85) -> str:
    """将 PIL Image 转换为 base64 字符串（JPEG 格式压缩）"""
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG', quality=jpeg_quality)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


# 细节敏感维度：需要更高分辨率
DETAIL_SENSITIVE_DIMENSIONS = {
    "hand_malformation", "expression_mismatch",
    "text_error", "plastic_waxy_texture", "logo_symbol_error",
}


def _concat_images_horizontally(
    img1: Image.Image, img2: Image.Image, dimension: str = None
) -> tuple:
    """水平拼接两张图像（左正右负），按维度自适应缩放"""
    if dimension in DETAIL_SENSITIVE_DIMENSIONS:
        target_size, quality = 768, 90
    else:
        target_size, quality = 512, 85

    img1 = img1.resize((target_size, target_size), Image.LANCZOS)
    img2 = img2.resize((target_size, target_size), Image.LANCZOS)

    total_width = img1.width + img2.width
    combined = Image.new('RGB', (total_width, target_size))
    combined.paste(img1, (0, 0))
    combined.paste(img2, (img1.width, 0))

    return combined, quality


# Legacy fallback guidelines used only if newer source-of-truth extraction fails.
LEGACY_DIMENSION_GUIDELINES = {
    # === Technical Quality ===
    "blur": "Lack of sharpness overall or locally, with defocus or motion blur effects",
    "overexposure": "Overly bright image with blown highlights and severe loss of bright area details",
    "underexposure": "Overly dark image with crushed blacks and severe loss of shadow details",
    "low_contrast": "Compressed histogram, hazy/flat appearance lacking tonal contrast",
    "color_cast": "Unnatural overall color tint (e.g., greenish, yellowish), white balance error",
    "desaturation": "Dull, low saturation colors appearing faded or aged",
    "plastic_waxy_texture": "Overly smooth skin/surfaces losing texture details, plastic/waxy appearance",

    # === Aesthetic Quality ===
    "awkward_positioning": "Subject at wrong distance (too far appearing tiny, too close causing distortion) or placed at awkward edge positions",
    "awkward_framing": "Unflattering camera angle or perspective distortion (e.g., wide-angle face distortion)",
    "unbalanced_layout": "Severely off-center visual weight, extremely unbalanced composition",
    "cluttered_scene": "Background filled with cluttered/distracting objects interfering with subject",
    "lighting_imbalance": "Chaotic light distribution with unexpected bright spots or dead blacks",
    "color_clash": "Jarring or disharmonious color combinations (e.g., saturated red-green clash)",
    "dull_palette": "Dull, depressing, or boring color tones lacking appeal",

    # === Semantic Rationality: Anatomy ===
    "hand_malformation": "Hand structure errors (abnormal finger count, fusion, twisted joints)",
    "expression_mismatch": "Facial expression contradicting scene mood or action logic",
    "body_proportion_error": "Severely distorted body proportions (e.g., gibbon arms, abnormal head-body ratio)",
    "extra_limbs": "Extra limbs appearing (three hands, multiple legs)",
    "animal_anatomy_error": "Animals with wrong anatomical features (mixed species traits, misplaced organs)",

    # === Semantic Rationality: Object ===
    "object_structure_error": "Common structured objects with obvious shape, alignment, attachment, orientation, or component-count failures",
    "material_mismatch": "Object keeps its shape but has an obviously wrong material appearance",
    "extra_objects": "Duplicate, out-of-context, or logically irrelevant redundant objects in scene",
    "count_error": "Generated object count not matching expected quantity",
    "illogical_colors": "Objects with counter-intuitive colors (e.g., blue flames, purple grass)",

    # === Semantic Rationality: Spatial ===
    "scale_inconsistency": "Severely counter-intuitive size ratios between objects (e.g., giant insects)",
    "floating_objects": "Gravity-affected objects floating in air without support",
    "penetration_overlap": "Impossible overlap, merge, biting edges, or wrong attachment between solid objects",

    # === Semantic Rationality: Scene ===
    "context_mismatch": "Subject appearing in extremely illogical environment (e.g., penguin in desert)",
    "time_inconsistency": "Conflicting time/season features in same image (e.g., sun and moon together)",
    "scene_layout_error": "Scene layout violating common sense (e.g., toilet in kitchen)",

    # === Semantic Rationality: Text ===
    "text_error": "Generated text with spelling errors, garbled characters, or unreadable content",
    "logo_symbol_error": "Unexpected distracting icons (barcodes, QR codes) or brand logo watermarks",
}

def _repair_and_parse_json(text: str) -> dict:
    """
    尝试修复 VLM 返回的格式有问题的 JSON。
    常见问题：尾逗号、截断、缺少闭合括号、多余文本等。
    
    Raises:
        ValueError: 如果修复后仍无法解析
    """
    # 1. 提取第一个 { 到最后一个 } 之间的内容
    start = text.find('{')
    end = text.rfind('}')
    if start == -1:
        raise ValueError("No JSON object found in response")
    
    if end > start:
        text = text[start:end + 1]
    else:
        # 没有闭合的 }，说明被截断了
        text = text[start:]
    
    # 2. 移除尾逗号（"key": value, } 这种情况）
    text = re.sub(r',\s*}', '}', text)
    text = re.sub(r',\s*$', '', text)
    
    # 3. 补齐缺失的闭合括号
    open_braces = text.count('{') - text.count('}')
    open_brackets = text.count('[') - text.count(']')
    
    # 如果截断在字符串值中间，尝试关闭字符串
    if open_braces > 0:
        # 检查是否有未闭合的字符串
        quote_count = text.count('"') - text.count('\\"')
        if quote_count % 2 == 1:
            text += '"'
        # 移除最后一个不完整的 key-value 对
        text = re.sub(r',\s*"[^"]*"?\s*:?\s*"?[^"]*$', '', text)
        text += '}' * open_braces
    
    if open_brackets > 0:
        text += ']' * open_brackets
    
    # 4. 再次清理尾逗号
    text = re.sub(r',\s*}', '}', text)
    text = re.sub(r',\s*]', ']', text)
    
    # 5. 尝试解析
    try:
        result = json.loads(text)
        logger.info("JSON 修复成功")
        return result
    except json.JSONDecodeError as e:
        logger.warning(f"JSON 修复后仍无法解析: {e}")
        logger.warning(f"修复后文本: {text[:300]}")
        
        # 6. 最后手段：用正则逐字段提取
        return _extract_fields_by_regex(text)


def _extract_fields_by_regex(text: str) -> dict:
    """通过正则表达式从半损坏的 JSON 中提取关键字段"""
    result = {}
    
    # valid
    valid_match = re.search(r'"valid"\s*:\s*(true|false)', text, re.IGNORECASE)
    result['valid'] = valid_match.group(1).lower() == 'true' if valid_match else False
    
    # style_type
    style_match = re.search(r'"style_type"\s*:\s*"([^"]*)"', text)
    result['style_type'] = style_match.group(1) if style_match else 'realistic'
    
    # failure
    failure_match = re.search(r'"failure"\s*:\s*"([^"]*)"', text)
    if failure_match:
        result['failure'] = failure_match.group(1)
    else:
        null_match = re.search(r'"failure"\s*:\s*null', text)
        result['failure'] = None if null_match else 'unknown'
    
    # recommended_style
    rec_match = re.search(r'"recommended_style"\s*:\s*"([^"]*)"', text)
    result['recommended_style'] = rec_match.group(1) if rec_match else None
    
    # scores
    scores = {}
    for key in ('content_preservation', 'style_consistency', 'degradation_intensity', 'dimension_accuracy'):
        score_match = re.search(rf'"{key}"\s*:\s*([\d.]+)', text)
        scores[key] = float(score_match.group(1)) if score_match else 0.0
    result['scores'] = scores
    
    # notes
    notes_match = re.search(r'"notes"\s*:\s*"([^"]*)"', text)
    result['notes'] = notes_match.group(1) if notes_match else ''
    
    logger.info(f"通过正则提取 JSON 字段: valid={result['valid']}, failure={result['failure']}")
    return result


@lru_cache(maxsize=1)
def _load_active_taxonomy() -> Dict[str, Any]:
    with ACTIVE_TAXONOMY_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@lru_cache(maxsize=1)
def _load_degradation_dimension_effects() -> Dict[str, str]:
    effects: Dict[str, str] = {}
    if not DEGRADATION_DIMENSIONS_DOC_PATH.exists():
        return effects

    for raw_line in DEGRADATION_DIMENSIONS_DOC_PATH.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line.startswith("| **"):
            continue
        cells = [cell.strip() for cell in line.split("|")[1:-1]]
        if len(cells) < 3:
            continue
        dimension_cell, _zh_name, effect = cells[:3]
        match = re.match(r"\*\*([a-z0-9_]+)\*\*", dimension_cell)
        if not match:
            continue
        effects[match.group(1)] = effect
    return effects


@lru_cache(maxsize=1)
def _load_compatibility_hints() -> Dict[str, str]:
    if not JUDGE_COMPATIBILITY_HINTS_PATH.exists():
        return {}

    with JUDGE_COMPATIBILITY_HINTS_PATH.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}

    if not isinstance(payload, dict):
        return {}

    return {str(key): str(value) for key, value in payload.items()}


@lru_cache(maxsize=1)
def _load_prompt_template_entries() -> Dict[str, Dict[str, Any]]:
    entries: Dict[str, Dict[str, Any]] = {}
    for yaml_path in sorted(PROMPT_TEMPLATES_ROOT.glob("*.yaml")):
        with yaml_path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        if not isinstance(payload, dict):
            continue
        for group_name, group_payload in payload.items():
            if not isinstance(group_payload, dict):
                continue
            for dimension, dimension_payload in group_payload.items():
                if not isinstance(dimension_payload, dict):
                    continue
                entries[dimension] = {
                    "template_group": group_name,
                    "yaml_file": str(yaml_path),
                    "severity_prompts": dimension_payload,
                }
    return entries


def _extract_template_strategy_cues(severity_prompts: Dict[str, str]) -> list[str]:
    cues: list[str] = []
    ordered_keys = [key for key in ("moderate", "severe", "mild") if key in severity_prompts]
    if not ordered_keys:
        ordered_keys = list(severity_prompts.keys())

    for severity in ordered_keys:
        text = str(severity_prompts.get(severity, "") or "")
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            line = re.sub(r"^\d+[a-z]?\.\s*", "", line)
            line = re.sub(r"^-\s*", "", line)
            if not line:
                continue
            if line.startswith(("You are given", "Avoid making", "Output only", "CRITICAL:", "FORBIDDEN")):
                continue
            if len(line) < 24:
                continue
            if "e.g." in line:
                line = line.split("e.g.", 1)[1].strip()
            keep = (
                "prefer" in line.lower()
                or "focus" in line.lower()
                or "describe" in line.lower()
                or "use " in line.lower()
                or "explicitly" in line.lower()
                or "make " in line.lower()
                or "keep the mismatch" in line.lower()
                or "the mismatch should" in line.lower()
                or "the size error" in line.lower()
                or "waxy rubber" in line.lower()
                or "extra visible arms" in line.lower()
            )
            if not keep:
                continue
            line = re.sub(r"\s+", " ", line).strip()
            if len(line) > 220:
                line = line[:217].rstrip() + "..."
            if line not in cues:
                cues.append(line)
            if len(cues) >= 4:
                return cues
    return cues


def _get_compatibility_hint(dimension: str) -> str:
    hints = _load_compatibility_hints()
    return hints.get(
        dimension,
        f"The positive image must contain the core subject matter required to visually express {dimension}."
    )


def _find_dimension_metadata(dimension: str) -> Dict[str, Any]:
    taxonomy = _load_active_taxonomy()
    perspectives = taxonomy.get("perspectives", {})
    for perspective_name, perspective_payload in perspectives.items():
        dimensions = (perspective_payload or {}).get("dimensions", {})
        if dimension in dimensions:
            meta = dimensions[dimension]
            return {
                "perspective": perspective_name,
                "zh_name": meta.get("zh"),
                "prompt_strategy": meta.get("prompt_strategy"),
                "controllability": meta.get("controllability"),
            }
    return {
        "perspective": None,
        "zh_name": None,
        "prompt_strategy": None,
        "controllability": None,
    }


def _get_dimension_criteria(dimension: str) -> Dict[str, Any]:
    metadata = _find_dimension_metadata(dimension)
    effects = _load_degradation_dimension_effects()
    template_entry = _load_prompt_template_entries().get(dimension, {})
    severity_prompts = template_entry.get("severity_prompts", {})

    effect_definition = effects.get(dimension) or LEGACY_DIMENSION_GUIDELINES.get(
        dimension, f"Check for visible {dimension} degradation in the image"
    )

    return {
        "dimension": dimension,
        "zh_name": metadata.get("zh_name"),
        "perspective": metadata.get("perspective"),
        "prompt_strategy": metadata.get("prompt_strategy"),
        "effect_definition": effect_definition,
        "template_group": template_entry.get("template_group"),
        "template_strategy_cues": _extract_template_strategy_cues(severity_prompts),
        "compatibility_hint": _get_compatibility_hint(dimension),
    }


def _build_judge_prompt(dimension: str, positive_prompt: str = None, negative_prompt: str = None, attribute: str = None) -> str:
    """Build judge prompt (hybrid approach: overall judgment + conditional diagnosis)"""
    criteria = _get_dimension_criteria(dimension)
    attr_text = f" ({attribute})" if attribute else ""

    prompt_section = ""
    if positive_prompt and negative_prompt:
        prompt_section = f"""
Prompts:
- Positive: "{positive_prompt}"
- Negative: "{negative_prompt}"
"""

    cue_lines = criteria.get("template_strategy_cues") or []
    if cue_lines:
        template_cues = "\n".join(f"- {cue}" for cue in cue_lines)
    else:
        template_cues = "- No template cues found; rely on the official effect definition and prompt strategy."

    return f"""You are an image quality judge. Compare LEFT (positive) vs RIGHT (negative).
{prompt_section}
Expected: {dimension}{attr_text} degradation

Dimension profile:
- Official effect definition: {criteria.get("effect_definition")}
- Positive compatibility hint: {criteria.get("compatibility_hint")}
Template strategy cues:
{template_cues}

**EVALUATE IN STRICT PRIORITY ORDER (stop at first failure)**:

**Step 1 — Positive Compatibility (CHECK FIRST)**:
Is the positive image compatible with {dimension} degradation?

1a. **Content compatibility**: Does the positive image contain the subject/element required for {dimension}?
   - Use the positive compatibility hint above as the primary content prerequisite.
   - If the image content CANNOT carry {dimension} degradation regardless of style, then failure = "positive_content_mismatch", STOP here

1b. **Style compatibility**: Is the rendering style suitable for {dimension}?
   - Judge whether the current rendering style makes the target degradation clearly visible or hides it.
   - If the content is suitable but the style makes {dimension} hard to perceive → failure = "positive_incompatible", provide recommended_style, STOP here

**Step 2 — Pair Evaluation (only if Step 1 passes)**:
a. Content: Are subjects/scenes similar between positive and negative?
b. Style: Is the MAJOR rendering category (realistic/illustration/painting) preserved in negative?
   - Focus on major category, minor sub-style variations are acceptable
c. Degradation: Is there CLEAR, VISIBLE {dimension} degradation?
   - Must be obvious enough for contrastive learning
   - Must specifically match the official effect definition above
   - Use the template strategy cues above to judge whether the degradation direction is the intended one, not just any generic corruption

If Step 2 fails, pick the FIRST matching failure:
1. style_drift: negative changed major style category (realistic↔illustration↔painting)
2. content_drift: subject/scene changed significantly
3. insufficient_effect: degradation too subtle or wrong dimension

Score all aspects (0.0-1.0), even if invalid:
- content_preservation: scene/subject consistency
- style_consistency: rendering style consistency
- degradation_intensity: strength of quality degradation
- dimension_accuracy: match with expected {dimension}

Output JSON only:
{{
  "valid": bool,
  "scores": {{
    "content_preservation": 0.0-1.0,
    "style_consistency": 0.0-1.0,
    "degradation_intensity": 0.0-1.0,
    "dimension_accuracy": 0.0-1.0
  }},
  "style_type": "realistic" | "illustration" | "painting",
  "failure": "positive_content_mismatch" | "positive_incompatible" | "style_drift" | "content_drift" | "insufficient_effect" | null,
  "recommended_style": "realistic" | "illustration" | "painting" | null,
  "notes": "Chinese diagnosis (required if invalid, explain what went wrong and why)"
}}"""


@tool
def degradation_judge(
    positive_image_path: str,
    negative_image_path: str,
    expected_dimension: str,
    positive_prompt: str = None,
    negative_prompt: str = None,
    expected_attribute: str = None
) -> str:
    """
    Judge whether an image pair shows valid visual quality degradation using VLM.

    This tool uses VLM (via OpenAI-compatible API) to evaluate if the generated
    image pair constitutes a valid training sample for a preference model.

    It checks for:
    1. Quality gap: The negative image should have visibly lower quality
    2. Content consistency: Both images should depict similar scenes
    3. Style consistency: Both images should have similar artistic style
    4. Target dimension: The degradation should match the expected dimension

    Only pairs with "moderate" or "obvious" degradation are accepted as valid.
    If the pair is invalid, a failure_type is provided for the rewriter.

    Args:
        positive_image_path: Path to the high-quality (positive) image.
        negative_image_path: Path to the degraded (negative) image.
        expected_dimension: The degradation category that was applied.
        positive_prompt: The prompt used for positive image (for diagnosis).
        negative_prompt: The prompt used for negative image (for diagnosis).
        expected_attribute: The specific attribute degraded (optional).

    Returns:
        JSON string containing:
        - valid: bool - Whether this is a valid training pair
        - failure_type: str or null - Type of failure for rewriter
        - diagnosis: dict - Detailed diagnosis of quality_gap, content/style consistency
        - degradation_level: str - 'none', 'subtle', 'moderate', 'obvious'
        - notes: str - Specific diagnosis for rewriter (in Chinese)
    """
    config = _load_config()
    client = _get_client()

    # 加载图像
    pos_img = _load_image_as_pil(positive_image_path)
    neg_img = _load_image_as_pil(negative_image_path)

    # 拼接图像（左正右负），按维度自适应缩放
    combined_img, jpeg_quality = _concat_images_horizontally(
        pos_img, neg_img, dimension=expected_dimension
    )

    # 转换为 base64（JPEG 压缩）
    img_base64 = _image_to_base64(combined_img, jpeg_quality=jpeg_quality)

    # 构建 prompt（传入正负 prompt 用于诊断）
    prompt = _build_judge_prompt(
        expected_dimension,
        positive_prompt=positive_prompt,
        negative_prompt=negative_prompt,
        attribute=expected_attribute
    )

    # 调用 OpenAI 兼容 Vision API
    max_retries = config['vlm'].get('max_retries', 3)
    retry_delay = config['vlm'].get('retry_delay', 2)
    model_name = config['vlm'].get('model', 'gemini-3-pro-preview')

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{img_base64}"
                                }
                            }
                        ]
                    }
                ],
                temperature=config['vlm'].get('temperature', 0.2),
                max_tokens=config['vlm'].get('max_tokens', 500),
                top_p=config['vlm'].get('top_p', 0.9)
            )

            # 解析响应
            response_text = response.choices[0].message.content.strip()

            # 移除可能的 markdown 代码块
            if response_text.startswith('```'):
                response_text = response_text.split('```')[1]
                if response_text.startswith('json'):
                    response_text = response_text[4:]
                response_text = response_text.strip()

            # 尝试解析 JSON，失败则尝试修复
            try:
                result = json.loads(response_text)
            except json.JSONDecodeError:
                result = _repair_and_parse_json(response_text)

            # 确保所有必要字段存在
            result.setdefault('valid', False)
            result.setdefault('style_type', 'realistic')
            result.setdefault('failure', None)
            result.setdefault('recommended_style', None)
            result.setdefault('notes', '')

            # 确保 scores 字段存在（向后兼容：无 scores 时补默认值）
            if 'scores' not in result:
                result['scores'] = {}
            for sk in ('content_preservation', 'style_consistency',
                       'degradation_intensity', 'dimension_accuracy'):
                result['scores'].setdefault(sk, 0.0)

            # valid 由 VLM 直接判断，不需要重新计算
            # 但如果 invalid 且没有 failure，尝试从 notes 推断
            if not result['valid'] and result['failure'] is None:
                notes_lower = result.get('notes', '').lower()
                if '风格' in notes_lower or 'style' in notes_lower:
                    result['failure'] = 'style_drift'
                elif '内容' in notes_lower or 'content' in notes_lower or '主体' in notes_lower:
                    result['failure'] = 'content_drift'
                else:
                    result['failure'] = 'insufficient_effect'

            if config['judge'].get('verbose_logging', False):
                logger.info(f"Judge result: valid={result['valid']}, failure={result['failure']}, style={result['style_type']}")

            return json.dumps(result, ensure_ascii=False)

        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse response (attempt {attempt + 1}): {e}")
            logger.warning(f"Raw response: {response_text[:300]}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                # 返回默认失败结果（新格式）
                return json.dumps({
                    "valid": False,
                    "scores": {
                        "content_preservation": 0.0,
                        "style_consistency": 0.0,
                        "degradation_intensity": 0.0,
                        "dimension_accuracy": 0.0
                    },
                    "style_type": "realistic",
                    "failure": None,
                    "recommended_style": None,
                    "notes": f"Failed to parse VLM response: {str(e)}"
                }, ensure_ascii=False)

        except Exception as e:
            logger.error(f"VLM API call failed (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                return json.dumps({
                    "valid": False,
                    "scores": {
                        "content_preservation": 0.0,
                        "style_consistency": 0.0,
                        "degradation_intensity": 0.0,
                        "dimension_accuracy": 0.0
                    },
                    "style_type": "realistic",
                    "failure": None,
                    "recommended_style": None,
                    "notes": f"API call failed: {str(e)}"
                }, ensure_ascii=False)


def cleanup():
    """清理资源"""
    global _config, _client
    _config = None
    _client = None


if __name__ == "__main__":
    # 测试代码（需要有实际图像）
    import sys
    if len(sys.argv) >= 3:
        result = degradation_judge(
            positive_image_path=sys.argv[1],
            negative_image_path=sys.argv[2],
            expected_dimension="blur"
        )
        print("Result:", result)
    else:
        print("Usage: python degradation_judge.py <positive_img> <negative_img>")
