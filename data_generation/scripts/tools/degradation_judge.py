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
from pathlib import Path
from typing import Optional, Dict, Any
from PIL import Image
import io

from smolagents import tool

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

    if not api_key:
        raise ValueError("API key not set in judge_config.yaml")

    _client = OpenAI(
        api_key=api_key,
        base_url=api_base
    )

    model_name = config['vlm'].get('model', 'gemini-3-pro-preview')
    logger.info(f"Initialized OpenAI-compatible client with model: {model_name}, base_url: {api_base}")
    return _client


def _load_image_as_pil(image_path: str) -> Image.Image:
    """加载图像为 PIL Image"""
    return Image.open(image_path).convert('RGB')


def _image_to_base64(img: Image.Image) -> str:
    """将 PIL Image 转换为 base64 字符串"""
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def _concat_images_horizontally(img1: Image.Image, img2: Image.Image) -> Image.Image:
    """水平拼接两张图像（左正右负）"""
    # 确保高度一致
    max_height = max(img1.height, img2.height)
    if img1.height != max_height:
        ratio = max_height / img1.height
        img1 = img1.resize((int(img1.width * ratio), max_height), Image.LANCZOS)
    if img2.height != max_height:
        ratio = max_height / img2.height
        img2 = img2.resize((int(img2.width * ratio), max_height), Image.LANCZOS)

    # 拼接
    total_width = img1.width + img2.width
    combined = Image.new('RGB', (total_width, max_height))
    combined.paste(img1, (0, 0))
    combined.paste(img2, (img1.width, 0))

    return combined


# Dimension-specific check guidelines (from degradation_dimensions.md)
DIMENSION_GUIDELINES = {
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
    "lack_of_depth": "Flat paper-like image lacking foreground/background depth layers",
    "flat_lighting": "Monotonous uniform lighting lacking shadow/highlight dimension (ID photo-like)",
    "lighting_imbalance": "Chaotic light distribution with unexpected bright spots or dead blacks",
    "color_clash": "Jarring or disharmonious color combinations (e.g., saturated red-green clash)",
    "dull_palette": "Dull, depressing, or boring color tones lacking appeal",

    # === Semantic Rationality: Anatomy ===
    "hand_malformation": "Hand structure errors (abnormal finger count, fusion, twisted joints)",
    "face_asymmetry": "Severely asymmetric facial features, collapsed or broken facial structure",
    "expression_mismatch": "Facial expression contradicting scene mood or action logic",
    "body_proportion_error": "Severely distorted body proportions (e.g., gibbon arms, abnormal head-body ratio)",
    "extra_limbs": "Extra limbs appearing (three hands, multiple legs)",
    "impossible_pose": "Anatomically impossible poses (reverse-bending joints, unnatural twists)",
    "animal_anatomy_error": "Animals with wrong anatomical features (mixed species traits, misplaced organs)",

    # === Semantic Rationality: Object ===
    "object_shape_error": "Common objects with distorted, melted, or collapsed shapes",
    "object_fusion": "Independent objects unreasonably fused or merged together",
    "missing_parts": "Objects missing critical components (e.g., car without wheels)",
    "extra_objects": "Duplicate, out-of-context, or logically irrelevant redundant objects in scene",
    "count_error": "Generated object count not matching expected quantity",
    "illogical_colors": "Objects with counter-intuitive colors (e.g., blue flames, purple grass)",

    # === Semantic Rationality: Spatial ===
    "perspective_error": "Incorrect spatial perspective violating geometric logic (Escher-like illusions)",
    "scale_inconsistency": "Severely counter-intuitive size ratios between objects (e.g., giant insects)",
    "floating_objects": "Gravity-affected objects floating in air without support",
    "penetration_overlap": "Physically impossible interpenetration or overlap of solid objects",

    # === Semantic Rationality: Physical ===
    "shadow_mismatch": "Missing shadows, wrong direction, or shape not matching casting object",
    "reflection_error": "Mirror/water reflections inconsistent with or incorrectly showing the source",

    # === Semantic Rationality: Scene ===
    "context_mismatch": "Subject appearing in extremely illogical environment (e.g., penguin in desert)",
    "time_inconsistency": "Conflicting time/season features in same image (e.g., sun and moon together)",
    "scene_layout_error": "Scene layout violating common sense (e.g., toilet in kitchen)",

    # === Semantic Rationality: Text ===
    "text_error": "Generated text with spelling errors, garbled characters, or unreadable content",
    "logo_symbol_error": "Unexpected distracting icons (barcodes, QR codes) or brand logo watermarks",
}

# Dimension category mapping
DIMENSION_CATEGORIES = {
    "technical_quality": ["blur", "overexposure", "underexposure", "low_contrast",
                          "color_cast", "desaturation", "plastic_waxy_texture"],
    "aesthetic_quality": ["awkward_positioning", "awkward_framing", "unbalanced_layout",
                          "cluttered_scene", "lack_of_depth", "flat_lighting",
                          "lighting_imbalance", "color_clash", "dull_palette"],
    "semantic_rationality": ["hand_malformation", "face_asymmetry", "expression_mismatch",
                             "body_proportion_error", "extra_limbs", "impossible_pose",
                             "animal_anatomy_error", "object_shape_error", "object_fusion",
                             "missing_parts", "extra_objects", "count_error", "illogical_colors",
                             "perspective_error", "scale_inconsistency", "floating_objects",
                             "penetration_overlap", "shadow_mismatch", "reflection_error",
                             "context_mismatch", "time_inconsistency", "scene_layout_error",
                             "text_error", "logo_symbol_error"]
}


import re


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


def _get_dimension_guideline(dimension: str) -> str:
    """Get dimension-specific check guideline"""
    return DIMENSION_GUIDELINES.get(dimension, f"Check for visible {dimension} degradation in the image")


def _build_judge_prompt(dimension: str, positive_prompt: str = None, negative_prompt: str = None, attribute: str = None) -> str:
    """Build judge prompt (hybrid approach: overall judgment + conditional diagnosis)"""
    guideline = _get_dimension_guideline(dimension)
    attr_text = f" ({attribute})" if attribute else ""

    prompt_section = ""
    if positive_prompt and negative_prompt:
        prompt_section = f"""
Prompts:
- Positive: "{positive_prompt}"
- Negative: "{negative_prompt}"
"""

    return f"""You are an image quality judge. Compare LEFT (positive) vs RIGHT (negative).
{prompt_section}
Expected: {dimension}{attr_text} degradation - {guideline}

**EVALUATE IN STRICT PRIORITY ORDER (stop at first failure)**:

**Step 1 — Positive Compatibility (CHECK FIRST)**:
Is the positive image compatible with {dimension} degradation?

1a. **Content compatibility**: Does the positive image contain the subject/element required for {dimension}?
   - Face/expression dimensions need a visible face; hand dimensions need visible hands; text dimensions need text present; shadow dimensions need objects casting shadows, etc.
   - If the image content CANNOT carry {dimension} degradation regardless of style (e.g., landscape for face_asymmetry, abstract art for expression_mismatch) → failure = "positive_content_mismatch", STOP here

1b. **Style compatibility**: Is the rendering style suitable for {dimension}?
   - blur/exposure need photorealistic; anatomy errors need realistic depiction
   - If the content is suitable but the style makes {dimension} invisible → failure = "positive_incompatible", provide recommended_style, STOP here

**Step 2 — Pair Evaluation (only if Step 1 passes)**:
a. Content: Are subjects/scenes similar between positive and negative?
b. Style: Is the MAJOR rendering category (realistic/illustration/painting) preserved in negative?
   - Focus on major category, minor sub-style variations are acceptable
c. Degradation: Is there CLEAR, VISIBLE {dimension} degradation?
   - Must be obvious enough for contrastive learning
   - Must specifically match {dimension}

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

    # 拼接图像（左正右负）
    combined_img = _concat_images_horizontally(pos_img, neg_img)

    # 转换为 base64
    img_base64 = _image_to_base64(combined_img)

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
                                    "url": f"data:image/png;base64,{img_base64}"
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
