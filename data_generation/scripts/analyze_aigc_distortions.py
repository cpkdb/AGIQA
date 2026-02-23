"""
End-to-end pipeline for discovering AIGC-specific distortions from SDXL outputs.

Pipeline:
1) download_dataset: fetch image_quality_train.json from HuggingFace
2) sample_prompts: randomly select 50-100 prompts for a pilot
3) generate_images: wrap scripts/sdxl_generator.py batch generation
4) analyze_with_gpt4o: open-ended GPT-4o vision analysis (no predefined categories)
5) extract_distortion_patterns: cluster raw GPT outputs into recurring patterns
6) build_dimension_taxonomy: induce a hierarchical quality dimension taxonomy
7) generate_reports: write JSON + markdown artifacts

Resumable via --resume and intermediate JSON checkpoints.
"""

import argparse
import base64
import json
import logging
import os
import random
import re
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add parent directory to path for imports
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from scripts.llm_prompt_degradation import LLMPromptDegradation
from scripts.sdxl_generator import SDXLGenerator

logger = logging.getLogger("aigc_distortion_analysis")
logging.basicConfig(level=logging.INFO)

# Constants and default paths
DATASET_URL = (
    "https://huggingface.co/datasets/meituan-longcat/Q-Eval-100K/"
    "resolve/main/Images_train_jsons/image_quality_train.json"
)
DEFAULT_DATASET_PATH = Path("data/image_quality_train.json")
BASE_OUTPUT_DIR = Path("/root/autodl-tmp/aigc_distortion_analysis")
LLM_CONFIG_PATH = Path("config/llm_config.yaml")

# 动态路径将在运行时根据run_id设置
OUTPUT_DIR = None
SAMPLED_PROMPTS_PATH = None
IMAGE_OUTPUT_DIR = None
IMAGE_METADATA_PATH = None
RAW_ANALYSIS_PATH = None
TAXONOMY_PATH = None
REPORT_PATH = None


def download_dataset(
    url: str = DATASET_URL,
    destination: Path = DEFAULT_DATASET_PATH,
    force: bool = False,
) -> Path:
    """
    Download image_quality_train.json from HuggingFace.
    Respects existing file unless force=True.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not force:
        logger.info("Dataset already present at %s", destination)
        return destination

    logger.info("Downloading dataset from %s", url)
    try:
        import requests
    except ImportError as exc:
        raise ImportError("requests is required to download the dataset") from exc

    try:
        with requests.get(url, stream=True, timeout=60) as resp:
            resp.raise_for_status()
            with open(destination, "wb") as fout:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        fout.write(chunk)
        logger.info("Dataset saved to %s", destination)
        return destination
    except Exception as exc:
        logger.error("Failed to download dataset: %s", exc)
        raise


def _extract_prompt(entry: Dict[str, Any]) -> Optional[str]:
    """Try multiple fields to extract a text prompt from a dataset entry."""
    for key in ["prompt", "text", "caption", "raw_prompt", "Prompt", "description"]:
        if key in entry and entry[key]:
            return entry[key]
    return None


def sample_prompts(
    dataset_path: Path,
    sample_size: int = 80,
    seed: int = 42,
    save_path: Path = SAMPLED_PROMPTS_PATH,
    resume: bool = False,
) -> List[Dict[str, Any]]:
    """Select a random subset of prompts for a pilot run."""
    if resume and save_path.exists():
        logger.info("Loading previously sampled prompts from %s", save_path)
        with open(save_path, "r", encoding="utf-8") as f:
            return json.load(f)

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    candidates = []
    if isinstance(data, dict):
        candidates = data.get("data") or data.get("images") or data.get("items") or []
    elif isinstance(data, list):
        candidates = data

    prompts: List[Dict[str, Any]] = []
    for idx, entry in enumerate(candidates):
        prompt_text = _extract_prompt(entry) if isinstance(entry, dict) else None
        if prompt_text:
            prompts.append({"id": idx, "prompt": prompt_text})

    if not prompts:
        raise ValueError("No prompts found in dataset; check schema and _extract_prompt.")

    rng = random.Random(seed)
    rng.shuffle(prompts)
    selected = prompts[:sample_size]

    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(selected, f, ensure_ascii=False, indent=2)
    logger.info("Sampled %d prompts (seed=%d) -> %s", len(selected), seed, save_path)
    return selected


def generate_images(
    prompts: List[Dict[str, Any]],
    output_dir: Path = IMAGE_OUTPUT_DIR,
    metadata_path: Path = IMAGE_METADATA_PATH,
    resume: bool = False,
    num_inference_steps: int = 30,
    guidance_scale: float = 7.5,
    width: int = 1024,
    height: int = 1024,
    device: str = "cuda",
    seed: Optional[int] = 1234,
) -> List[Dict[str, Any]]:
    """Generate images for sampled prompts via SDXLGenerator."""
    output_dir.mkdir(parents=True, exist_ok=True)
    existing: Dict[str, Any] = {}
    if resume and metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            existing = {str(item["id"]): item for item in json.load(f)}
        logger.info("Resuming generation with %d existing items", len(existing))

    generator = SDXLGenerator(device=device)
    results: List[Dict[str, Any]] = []
    try:
        for idx, item in enumerate(prompts):
            record_id = str(item.get("id", idx))
            image_path = output_dir / f"sample_{int(record_id):05d}.png"

            if resume and record_id in existing and image_path.exists():
                results.append(existing[record_id])
                continue

            image, info = generator.generate(
                prompt=item["prompt"],
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
                seed=seed + idx if seed is not None else None,
            )
            image.save(image_path)
            result = {
                "id": item.get("id", idx),
                "prompt": item["prompt"],
                "image_path": str(image_path),
                "generation_info": info,
            }
            results.append(result)
    finally:
        generator.cleanup()

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info("Generated %d images -> %s", len(results), metadata_path)
    return results


def _init_llm_client(config_path: Path = LLM_CONFIG_PATH):
    """Reuse OpenAI client initialization from LLMPromptDegradation without legacy dimensions."""
    stub = LLMPromptDegradation.__new__(LLMPromptDegradation)
    stub.llm_config_path = str(config_path)
    config = LLMPromptDegradation._load_llm_config(stub)
    stub.config = config
    client = LLMPromptDegradation._init_llm_client(stub)
    return client, config


def _encode_image_to_base64(image_path: Path, max_size: int = 768) -> str:
    """Encode image to base64 with compression to reduce payload size."""
    from PIL import Image

    img = Image.open(image_path)

    # Resize if larger than max_size
    if max(img.size) > max_size:
        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

    # Convert to RGB if needed (for JPEG encoding)
    if img.mode in ("RGBA", "P"):
        rgb_img = Image.new("RGB", img.size, (255, 255, 255))
        rgb_img.paste(img, mask=img.split()[3] if img.mode == "RGBA" else None)
        img = rgb_img

    # Encode as JPEG with quality 85
    import io
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=85)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return f"data:image/jpeg;base64,{encoded}"


def _build_system_prompt() -> str:
    return (
        "You are a professional image quality assessor analyzing AI-generated images.\n\n"
        "TASK: Identify ALL visual quality issues and semantic problems in the image.\n\n"
        "SCOPE:\n"
        "1. Visual Quality Issues: Technical defects, rendering artifacts, unnatural appearance\n"
        "2. Semantic Problems: Content that contradicts common sense or prompt expectations\n"
        "3. EXCLUDE: Consistency/alignment issues (multi-image comparisons)\n\n"
        "CRITICAL REQUIREMENTS:\n"
        "- AI-generated images always have SOME defects - find them all\n"
        "- Describe issues in your OWN WORDS without predefined categories\n"
        "- Report both obvious defects AND subtle quality degradations\n"
        "- Be thorough but honest - don't invent problems that don't exist\n\n"
        "Respond strictly in JSON format."
    )


def _build_user_content(prompt: str, b64_image: str) -> List[Dict[str, Any]]:
    text_block = (
        "Comprehensively evaluate this AI-generated image for ALL perceptible quality issues.\n\n"
        f"Reference prompt (for semantic context): {prompt}\n\n"
        "INSTRUCTIONS:\n"
        "1. Examine the entire image systematically for defects\n"
        "2. Identify EVERY issue you can observe - both major and minor\n"
        "3. Describe each issue naturally in your own words (avoid predefined jargon)\n"
        "4. Include BOTH types of problems:\n"
        "   - Visual/Technical: Rendering artifacts, unnatural textures, lighting errors, etc.\n"
        "   - Semantic: Content that violates common sense or looks implausible\n"
        "5. Be thorough - AI images typically have multiple subtle defects\n\n"
        "For each defect provide:\n"
        "- label: Short descriptive name (your own phrasing)\n"
        "- description: What you observe and why it's problematic\n"
        "- severity: low (subtle/minor) | medium (noticeable) | high (obvious) | critical (severe)\n"
        "- confidence: 0.0-1.0 (certainty this is a genuine defect)\n"
        "- evidence: Specific visual details that support your assessment\n\n"
        "Return JSON in this exact format:\n"
        "{\n"
        '  "summary": "brief overall assessment of image quality",\n'
        '  "distortions": [\n'
        '    {\n'
        '      "label": "your descriptive name",\n'
        '      "description": "what you observe",\n'
        '      "severity": "low|medium|high|critical",\n'
        '      "confidence": 0.85,\n'
        '      "evidence": "specific visual detail"\n'
        '    }\n'
        '  ]\n'
        "}"
    )
    return [
        {"type": "text", "text": text_block},
        {"type": "image_url", "image_url": {"url": b64_image}},
    ]


def analyze_with_gpt4o(
    records: List[Dict[str, Any]],
    raw_output_path: Path = RAW_ANALYSIS_PATH,
    resume: bool = False,
    llm_config_path: Path = LLM_CONFIG_PATH,
    model_override: Optional[str] = None,
    request_interval: float = 0.0,
) -> List[Dict[str, Any]]:
    """Send images to GPT-4o for open-ended distortion analysis with retry and validation."""
    existing: Dict[str, Any] = {}
    if resume and raw_output_path.exists():
        with open(raw_output_path, "r", encoding="utf-8") as f:
            existing = {str(item["id"]): item for item in json.load(f)}
        logger.info("Resuming GPT-4o analysis with %d existing results", len(existing))

    client, config = _init_llm_client(llm_config_path)
    model_name = model_override or config["llm"]["model"]
    system_prompt = _build_system_prompt()

    # Auto rate limit from config if not specified
    if request_interval == 0.0:
        rpm_limit = config.get("llm", {}).get("rate_limit_rpm", 60)
        request_interval = max(60.0 / rpm_limit, 1.0) if rpm_limit > 0 else 1.0

    max_retries = 3
    results: List[Dict[str, Any]] = []

    for record in records:
        record_id = str(record["id"])

        # Check if already completed successfully
        if record_id in existing:
            existing_record = existing[record_id]
            if existing_record.get("status") == "success":
                # Verify image still exists
                if Path(record["image_path"]).exists():
                    results.append(existing_record)
                    continue
                else:
                    logger.warning("Image missing for id=%s, re-analyzing", record_id)

        # Verify image exists before processing
        image_path = Path(record["image_path"])
        if not image_path.exists():
            logger.error("Image not found for id=%s: %s", record_id, image_path)
            result = {
                "id": record["id"],
                "prompt": record["prompt"],
                "image_path": record["image_path"],
                "status": "error",
                "error": "Image file not found",
                "response_text": None,
                "parsed": None,
            }
            results.append(result)
            continue

        b64_image = _encode_image_to_base64(image_path)
        user_content = _build_user_content(record["prompt"], b64_image)

        # Retry loop
        parsed = None
        content = None
        last_error = None

        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                    temperature=0.2,
                    max_tokens=600,
                )
                content = response.choices[0].message.content
                parsed = _parse_llm_json(content)

                # Validate parsed response
                if parsed and "distortions" in parsed:
                    logger.info("GPT-4o analysis succeeded for id=%s (attempt %d)", record_id, attempt + 1)
                    break
                else:
                    logger.warning("GPT-4o returned invalid JSON for id=%s (attempt %d): missing 'distortions'", record_id, attempt + 1)
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)  # Exponential backoff

            except Exception as exc:
                last_error = str(exc)
                logger.error("GPT-4o analysis failed for id=%s (attempt %d): %s", record_id, attempt + 1, exc)
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff

        # Build result
        if parsed and "distortions" in parsed:
            status = "success"
        else:
            status = "failed"
            logger.error("GPT-4o analysis ultimately failed for id=%s after %d attempts", record_id, max_retries)

        result = {
            "id": record["id"],
            "prompt": record["prompt"],
            "image_path": record["image_path"],
            "status": status,
            "error": last_error if status == "failed" else None,
            "response_text": content,
            "parsed": parsed,
        }
        results.append(result)

        _persist_json(raw_output_path, results)  # incremental for resumability
        time.sleep(request_interval)

    success_count = sum(1 for r in results if r.get("status") == "success")
    logger.info("Completed GPT-4o analysis: %d/%d successful -> %s", success_count, len(results), raw_output_path)
    return results


def _persist_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _parse_llm_json(content: Optional[str]) -> Optional[Dict[str, Any]]:
    if not content:
        return None
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", content, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None


def extract_distortion_patterns(
    raw_analysis: List[Dict[str, Any]],
    save_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Aggregate open-ended distortions into recurring pattern buckets with normalization."""
    pattern_buckets: Dict[str, Dict[str, Any]] = {}
    total_findings = 0

    for record in raw_analysis:
        parsed = record.get("parsed") or {}
        distortions = parsed.get("distortions") or []
        for dist in distortions:
            name = (
                dist.get("label")
                or dist.get("type")
                or dist.get("distortion")
                or dist.get("name")
                or "unspecified"
            )
            normalized = _normalize_pattern_name(name)
            bucket = pattern_buckets.setdefault(
                normalized,
                {
                    "canonical_label": name,
                    "count": 0,
                    "severity_histogram": Counter(),
                    "confidence_sum": 0.0,
                    "examples": [],
                },
            )
            bucket["count"] += 1
            total_findings += 1

            # Normalize severity field
            severity_raw = (dist.get("severity") or "").lower().strip()
            severity = _normalize_severity(severity_raw)
            bucket["severity_histogram"][severity] += 1

            # Normalize confidence field (clamp to 0-1)
            confidence_raw = dist.get("confidence") or 0.0
            confidence = _normalize_confidence(confidence_raw)
            bucket["confidence_sum"] += confidence

            if len(bucket["examples"]) < 5:
                bucket["examples"].append(
                    {
                        "description": dist.get("description") or dist.get("note") or "",
                        "evidence": dist.get("evidence") or "",
                        "severity": severity,
                        "confidence": confidence,
                        "image_id": record.get("id"),
                        "image_path": record.get("image_path"),
                        "prompt": record.get("prompt"),
                    }
                )

    patterns = []
    for name, info in pattern_buckets.items():
        avg_conf = info["confidence_sum"] / info["count"] if info["count"] else 0.0
        patterns.append(
            {
                "pattern": name,
                "canonical_label": info["canonical_label"],
                "count": info["count"],
                "average_confidence": round(avg_conf, 3),
                "severity_histogram": dict(info["severity_histogram"]),
                "examples": info["examples"],
            }
        )

    patterns.sort(key=lambda x: x["count"], reverse=True)
    payload = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "total_images": len(raw_analysis),
        "total_findings": total_findings,
        "patterns": patterns,
    }
    if save_path:
        _persist_json(save_path, payload)
        logger.info("Extracted %d unique distortion patterns -> %s", len(patterns), save_path)
    return payload


def _normalize_pattern_name(name: str) -> str:
    normalized = re.sub(r"[^a-z0-9\s]+", " ", name.lower()).strip()
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized or "unspecified"


def _normalize_severity(severity: str) -> str:
    """Normalize severity to standard values."""
    severity_map = {
        "low": "low",
        "minor": "low",
        "slight": "low",
        "mild": "low",
        "medium": "medium",
        "moderate": "medium",
        "med": "medium",
        "high": "high",
        "major": "high",
        "severe": "high",
        "critical": "critical",
        "extreme": "critical",
    }
    return severity_map.get(severity, "unspecified")


def _normalize_confidence(confidence: Any) -> float:
    """Normalize and clamp confidence to 0-1 range."""
    try:
        conf = float(confidence)
        return max(0.0, min(1.0, conf))
    except (TypeError, ValueError):
        return 0.0


def build_dimension_taxonomy(
    pattern_summary: Dict[str, Any],
    save_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Induce a hierarchical taxonomy from open-ended patterns using keyword heuristics."""
    roots = _taxonomy_roots()
    for pattern in pattern_summary.get("patterns", []):
        bucket = _assign_root(pattern["pattern"], roots)
        bucket["directions"].append(
            {
                "name": pattern["canonical_label"],
                "normalized_name": pattern["pattern"],
                "count": pattern["count"],
                "average_confidence": pattern["average_confidence"],
                "severity_histogram": pattern["severity_histogram"],
                "examples": pattern["examples"],
            }
        )

    taxonomy = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "note": "Induced taxonomy from open-ended GPT-4o distortion analysis; no legacy dimensions used.",
        "dimensions": roots,
    }
    if save_path:
        _persist_json(save_path, taxonomy)
        logger.info("Built induced taxonomy with %d root dimensions -> %s", len(roots), save_path)
    return taxonomy


def _taxonomy_roots() -> List[Dict[str, Any]]:
    return [
        {
            "name": "anatomy_and_form",
            "description": "Body/structure integrity, limb counts, facial coherence, pose plausibility.",
            "keywords": ["anatom", "limb", "finger", "hand", "face", "body", "pose", "proportion", "symmetry"],
            "directions": [],
        },
        {
            "name": "texture_and_materials",
            "description": "Surface detail, fabric/skin/metal/wood realism, melting/bleeding textures.",
            "keywords": ["texture", "material", "fabric", "skin", "metal", "plastic", "wood", "leather", "porcelain"],
            "directions": [],
        },
        {
            "name": "lighting_and_rendering",
            "description": "Light transport, shadows, reflections, specular highlights, volumetrics.",
            "keywords": ["light", "shadow", "highlight", "reflection", "specular", "glow", "exposure", "illumination"],
            "directions": [],
        },
        {
            "name": "scene_logic_and_physics",
            "description": "Perspective, occlusion, gravity/physics breaks, impossible geometry.",
            "keywords": ["perspective", "physics", "gravity", "impossible", "occlusion", "intersection", "floating"],
            "directions": [],
        },
        {
            "name": "composition_and_layout",
            "description": "Framing, cropping, clutter, spatial relationships, hierarchy.",
            "keywords": ["composition", "layout", "framing", "crop", "balance", "clutter", "spacing"],
            "directions": [],
        },
        {
            "name": "text_and_symbols",
            "description": "Rendered text/typography, signage, logos, glyph correctness.",
            "keywords": ["text", "word", "typography", "logo", "sign", "letter", "glyph"],
            "directions": [],
        },
        {
            "name": "style_and_coherence",
            "description": "Style clashes, aesthetic drift, mixed modalities (photo/CGI/sketch).",
            "keywords": ["style", "aesthetic", "cartoon", "realistic", "sketch", "painterly", "render"],
            "directions": [],
        },
        {
            "name": "model_artifacts",
            "description": "Diffusion artifacts: banding, tiling, smudging, watermark ghosts, duplicated limbs.",
            "keywords": ["artifact", "banding", "tiling", "moire", "noise", "smudge", "watermark", "duplicate", "glitch"],
            "directions": [],
        },
        {
            "name": "color_and_tone",
            "description": "Hue/saturation shifts, unnatural skin tones, grading inconsistencies.",
            "keywords": ["color", "hue", "saturation", "tone", "grading", "white balance", "oversaturated", "muted"],
            "directions": [],
        },
        {
            "name": "emergent_other",
            "description": "Patterns that do not align with other roots; flagged for further review.",
            "keywords": [],
            "directions": [],
        },
    ]


def _assign_root(pattern_name: str, roots: List[Dict[str, Any]]) -> Dict[str, Any]:
    for root in roots:
        if any(keyword in pattern_name for keyword in root["keywords"]):
            return root
    return roots[-1]  # emergent_other


def generate_reports(
    taxonomy: Dict[str, Any],
    pattern_summary: Dict[str, Any],
    raw_analysis: List[Dict[str, Any]],
    report_path: Path = REPORT_PATH,
) -> None:
    """Produce a markdown report with taxonomy overview and example snippets."""
    lines: List[str] = []
    lines.append("# AIGC图像失真分类体系报告")
    lines.append("")
    lines.append(f"- 生成时间: {datetime.utcnow().isoformat()}Z")
    lines.append(f"- 分析图像数量: {pattern_summary.get('total_images', 0)}")
    lines.append(f"- 发现失真实例: {pattern_summary.get('total_findings', 0)}")
    lines.append(f"- 独特失真模式: {len(pattern_summary.get('patterns', []))}")
    lines.append("")
    lines.append("## 根维度分类")
    for root in taxonomy.get("dimensions", []):
        lines.append(f"### {root['name']}")
        lines.append(root["description"])
        lines.append("")
        top_dirs = sorted(root["directions"], key=lambda x: x["count"], reverse=True)[:5]
        for direction in top_dirs:
            lines.append(
                f"- {direction['name']} (出现次数={direction['count']}, 平均置信度={direction['average_confidence']})"
            )
        lines.append("")

    lines.append("## 失真模式示例")
    for pattern in pattern_summary.get("patterns", [])[:15]:
        lines.append(f"### {pattern['canonical_label']} (出现次数={pattern['count']})")
        for ex in pattern["examples"][:3]:
            lines.append(
                f"- 图像ID={ex['image_id']} | Prompt={ex.get('prompt','')[:80]}...\n"
                f"  描述: {ex.get('description','')} | 证据: {ex.get('evidence','')}"
            )
        lines.append("")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info("Markdown报告已生成 -> %s", report_path)


def semantic_clustering_phase2(
    raw_analysis: List[Dict[str, Any]],
    output_path: Path,
    llm_config_path: Path = LLM_CONFIG_PATH,
    batch_size: int = 250,
) -> Dict[str, Any]:
    """阶段二：使用LLM对所有失真进行分批语义聚类，归纳质量维度

    Args:
        raw_analysis: GPT-4o分析的原始结果
        output_path: 聚类结果输出路径
        llm_config_path: LLM配置文件路径
        batch_size: 每批处理的失真数量（建议200-300，避免超出上下文限制）

    Returns:
        包含维度和方向的字典
    """
    logger.info("=== 阶段二：语义聚类归纳质量维度 ===")

    # 1. 收集所有失真描述
    all_distortions = []
    for record in raw_analysis:
        if record.get("status") != "success":
            continue
        for dist in record.get("parsed", {}).get("distortions", []):
            all_distortions.append({
                "label": dist.get("label", ""),
                "description": dist.get("description", ""),
                "image_id": record.get("id"),
                "severity": dist.get("severity", ""),
                "confidence": dist.get("confidence", 0.0),
            })

    total_distortions = len(all_distortions)
    logger.info("收集到 %d 个失真描述，准备分批聚类", total_distortions)

    if total_distortions == 0:
        logger.warning("没有失真数据，跳过阶段二")
        return {"dimensions": []}

    # 2. 分批处理
    client, config = _init_llm_client(llm_config_path)
    num_batches = (total_distortions + batch_size - 1) // batch_size
    batch_results = []

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_distortions)
        batch = all_distortions[start_idx:end_idx]

        logger.info("处理批次 %d/%d (失真 %d-%d)", batch_idx + 1, num_batches, start_idx, end_idx)

        # 构建批次摘要（只包含label和简短description）
        batch_summary = []
        for i, dist in enumerate(batch, start=start_idx):
            batch_summary.append(
                f"{i+1}. [{dist['label']}] {dist['description'][:100]}"
            )

        # 调用LLM进行聚类
        cluster_prompt = f"""你是一个AIGC图像质量专家。现在有 {len(batch)} 个AI生成图像的失真描述（总共 {total_distortions} 个中的第 {start_idx+1}-{end_idx} 个）。

失真描述列表：
{chr(10).join(batch_summary[:200])}

任务：
1. 将这些失真按照**语义相似性**进行聚类分组
2. 为每个聚类定义一个**维度名称**和**描述**
3. 每个维度下列出属于该维度的**失真类型**（使用原始label）

要求：
- 维度命名使用英文（如 anatomical_defects, texture_issues）
- 描述用中文
- 基于数据归纳，不要使用预设框架
- 如果某个失真很独特无法归类，单独成为一个维度

返回JSON格式：
{{
  "dimensions": [
    {{
      "name": "维度英文名",
      "description": "该维度覆盖什么类型的失真（中文）",
      "distortion_labels": ["原始label1", "原始label2", ...]
    }}
  ]
}}"""

        try:
            response = client.chat.completions.create(
                model=config["llm"]["model"],
                messages=[{"role": "user", "content": cluster_prompt}],
                temperature=0.3,
                max_tokens=3000,
                response_format={"type": "json_object"},
            )

            result_text = response.choices[0].message.content
            batch_result = json.loads(result_text)
            batch_results.append(batch_result)
            logger.info("批次 %d 聚类完成，发现 %d 个维度", batch_idx + 1, len(batch_result.get("dimensions", [])))

        except Exception as e:
            logger.error("批次 %d 聚类失败: %s", batch_idx + 1, e)
            continue

    # 3. 合并所有批次的结果
    logger.info("合并 %d 个批次的聚类结果", len(batch_results))

    merge_prompt = f"""你是一个AIGC图像质量专家。现在有 {len(batch_results)} 批失真聚类结果需要合并成统一的维度体系。

各批次发现的维度：
{json.dumps(batch_results, ensure_ascii=False, indent=2)}

任务：
1. 识别不同批次中**语义相同**的维度并合并
2. 去除重复，统一命名
3. 构建最终的质量维度分类体系

要求：
- 维度名称使用英文，描述用中文
- 合并时保留所有失真类型，不丢失信息
- 维度数量控制在8-15个之间（合并相似的）

返回JSON格式：
{{
  "dimensions": [
    {{
      "name": "最终维度英文名",
      "description": "该维度的完整描述（中文）",
      "distortion_types": ["失真类型1", "失真类型2", ...]
    }}
  ],
  "note": "从 {total_distortions} 个失真归纳得出，无预设框架"
}}"""

    try:
        response = client.chat.completions.create(
            model=config["llm"]["model"],
            messages=[{"role": "user", "content": merge_prompt}],
            temperature=0.3,
            max_tokens=4000,
            response_format={"type": "json_object"},
        )

        final_taxonomy = json.loads(response.choices[0].message.content)
        final_taxonomy["generated_at"] = datetime.utcnow().isoformat() + "Z"
        final_taxonomy["total_distortions_analyzed"] = total_distortions

        # 保存结果
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(final_taxonomy, f, ensure_ascii=False, indent=2)

        logger.info("阶段二聚类完成 -> %s", output_path)
        logger.info("最终归纳出 %d 个质量维度", len(final_taxonomy.get("dimensions", [])))

        return final_taxonomy

    except Exception as e:
        logger.error("合并聚类结果失败: %s", e)
        return {"dimensions": []}


def setup_output_paths(run_id: str, batch_id: Optional[str] = None) -> None:
    """设置输出路径，支持分批运行"""
    global OUTPUT_DIR, SAMPLED_PROMPTS_PATH, IMAGE_OUTPUT_DIR, IMAGE_METADATA_PATH
    global RAW_ANALYSIS_PATH, TAXONOMY_PATH, REPORT_PATH

    # 主运行目录
    run_dir = BASE_OUTPUT_DIR / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    if batch_id:
        # 分批运行：每批有独立的子目录
        batch_dir = run_dir / f"batch_{batch_id}"
        batch_dir.mkdir(parents=True, exist_ok=True)

        OUTPUT_DIR = run_dir
        SAMPLED_PROMPTS_PATH = run_dir / "sampled_prompts.json"  # 共享
        IMAGE_OUTPUT_DIR = batch_dir / "images"
        IMAGE_METADATA_PATH = batch_dir / "metadata.json"
        RAW_ANALYSIS_PATH = batch_dir / "raw_analysis.json"
    else:
        # 单次运行
        OUTPUT_DIR = run_dir
        SAMPLED_PROMPTS_PATH = run_dir / "sampled_prompts.json"
        IMAGE_OUTPUT_DIR = run_dir / "images"
        IMAGE_METADATA_PATH = run_dir / "metadata.json"
        RAW_ANALYSIS_PATH = run_dir / "raw_analysis.json"

    # 最终输出（合并后的结果）
    TAXONOMY_PATH = run_dir / "quality_dimensions.json"
    REPORT_PATH = run_dir / "dimension_report.md"

    # 创建latest符号链接
    latest_link = BASE_OUTPUT_DIR / "latest"
    if latest_link.exists() or latest_link.is_symlink():
        latest_link.unlink()
    latest_link.symlink_to(run_dir.relative_to(BASE_OUTPUT_DIR))

    logger.info("输出目录设置: %s", run_dir)
    if batch_id:
        logger.info("批次ID: %s", batch_id)


def merge_batch_results(run_id: str) -> List[Dict[str, Any]]:
    """合并所有批次的分析结果"""
    run_dir = BASE_OUTPUT_DIR / "runs" / run_id
    batch_dirs = sorted(run_dir.glob("batch_*"))

    if not batch_dirs:
        logger.warning("未找到批次目录，可能是单次运行")
        return []

    logger.info("发现 %d 个批次，开始合并结果", len(batch_dirs))

    all_results = []
    for batch_dir in batch_dirs:
        batch_file = batch_dir / "raw_analysis.json"
        if batch_file.exists():
            with open(batch_file, "r", encoding="utf-8") as f:
                batch_results = json.load(f)
                all_results.extend(batch_results)
                logger.info("合并批次 %s: %d 条记录", batch_dir.name, len(batch_results))

    # 保存合并结果
    merged_path = run_dir / "merged_raw_analysis.json"
    with open(merged_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    logger.info("合并完成，共 %d 条记录 -> %s", len(all_results), merged_path)
    return all_results


def run_pipeline(args: argparse.Namespace) -> None:
    # 设置输出路径
    setup_output_paths(args.run_id, args.batch_id)

    # 如果指定了merge-only，直接跳到合并和聚类步骤
    if args.merge_only:
        logger.info("仅执行合并和聚类...")
        analyzed = merge_batch_results(args.run_id)
        if not analyzed:
            # 不是批次运行，使用单次结果
            with open(RAW_ANALYSIS_PATH, "r", encoding="utf-8") as f:
                analyzed = json.load(f)
    else:
        # 正常流程：下载数据集、生成图像、分析
        # 下载数据集
        dataset_path = download_dataset(
            url=args.dataset_url,
            destination=Path(args.dataset_path),
            force=args.force_download,
        )

        # 采样prompts
        prompts = sample_prompts(
            dataset_path=dataset_path,
            sample_size=args.num_samples,
            seed=args.seed,
            resume=args.resume,
            save_path=SAMPLED_PROMPTS_PATH,
        )

        # 生成图像
        generated = generate_images(
            prompts=prompts,
            output_dir=IMAGE_OUTPUT_DIR,
            metadata_path=IMAGE_METADATA_PATH,
            resume=args.resume,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            width=args.width,
            height=args.height,
            device=args.device,
            seed=args.seed,
        )

        # GPT-4o分析
        analyzed = analyze_with_gpt4o(
            records=generated,
            raw_output_path=RAW_ANALYSIS_PATH,
            resume=args.resume,
            llm_config_path=Path(args.llm_config_path),
            model_override=args.model,
            request_interval=args.request_interval,
        )

        # 如果是分批运行，仅执行到这里
        if args.batch_id:
            logger.info("=== 批次 %s 完成，等待所有批次完成后执行聚类 ===", args.batch_id)
            logger.info("若要合并并聚类，运行: python %s --run-id %s --merge-only", __file__, args.run_id)
            return

    # 执行LLM语义聚类（默认启用）
    logger.info("=== 开始LLM语义聚类归纳质量维度 ===")
    taxonomy = semantic_clustering_phase2(
        raw_analysis=analyzed,
        output_path=TAXONOMY_PATH,
        llm_config_path=Path(args.llm_config_path),
        batch_size=args.phase2_batch_size,
    )

    # 生成报告
    generate_phase2_report(
        taxonomy_phase2=taxonomy,
        report_path=REPORT_PATH,
    )

    logger.info("=== 流程完成 ===")
    logger.info("结果保存在: %s", OUTPUT_DIR)
    logger.info("快速访问最新结果: %s", BASE_OUTPUT_DIR / "latest")


def generate_phase2_report(
    taxonomy_phase2: Dict[str, Any],
    report_path: Path,
) -> None:
    """生成阶段二语义聚类的报告"""
    lines: List[str] = []
    lines.append("# AIGC图像失真分类体系报告（阶段二：LLM语义聚类）")
    lines.append("")
    lines.append(f"- 生成时间: {taxonomy_phase2.get('generated_at', 'N/A')}")
    lines.append(f"- 分析失真总数: {taxonomy_phase2.get('total_distortions_analyzed', 0)}")
    lines.append(f"- 归纳维度数量: {len(taxonomy_phase2.get('dimensions', []))}")
    lines.append(f"- 归纳方法: {taxonomy_phase2.get('note', 'LLM语义聚类，无预设框架')}")
    lines.append("")

    lines.append("## 归纳的质量维度")
    for idx, dim in enumerate(taxonomy_phase2.get("dimensions", []), 1):
        lines.append(f"### {idx}. {dim.get('name', 'N/A')}")
        lines.append(f"**描述**: {dim.get('description', 'N/A')}")
        lines.append("")
        lines.append("**包含的失真类型**:")
        for dtype in dim.get("distortion_types", [])[:10]:
            lines.append(f"- {dtype}")
        if len(dim.get("distortion_types", [])) > 10:
            lines.append(f"- ...（共{len(dim.get('distortion_types', []))}种）")
        lines.append("")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info("阶段二报告已生成 -> %s", report_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze SDXL outputs with GPT-4o to induce AIGC distortion taxonomy via LLM semantic clustering."
    )

    # 数据集参数
    parser.add_argument("--dataset-url", default=DATASET_URL, help="Dataset download URL.")
    parser.add_argument("--dataset-path", default=str(DEFAULT_DATASET_PATH), help="Local dataset path.")
    parser.add_argument("--num-samples", type=int, default=80, help="Number of prompts to analyze.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling and generation.")
    parser.add_argument("--force-download", action="store_true", help="Force re-download of dataset.")
    parser.add_argument("--resume", action="store_true", help="Resume from existing intermediates.")

    # SDXL参数
    parser.add_argument("--device", default="cuda", help="Device for SDXL inference.")
    parser.add_argument("--width", type=int, default=1024, help="Image width.")
    parser.add_argument("--height", type=int, default=1024, help="Image height.")
    parser.add_argument("--num-inference-steps", type=int, default=30, help="SDXL inference steps.")
    parser.add_argument("--guidance-scale", type=float, default=7.5, help="SDXL guidance scale.")

    # LLM参数
    parser.add_argument("--llm-config-path", default=str(LLM_CONFIG_PATH), help="LLM config YAML path.")
    parser.add_argument("--model", default=None, help="Optional GPT model override.")
    parser.add_argument("--request-interval", type=float, default=0.0, help="Sleep seconds between GPT requests.")
    parser.add_argument("--phase2-batch-size", type=int, default=250, help="Batch size for semantic clustering (to avoid context limit).")

    # 运行管理参数
    parser.add_argument("--run-id", default=None, help="Run ID for organizing outputs (default: timestamp YYYYMMDD_HHMMSS).")
    parser.add_argument("--batch-id", default=None, help="Batch ID for split runs (e.g., '1', '2'). If set, only runs analysis without clustering.")
    parser.add_argument("--merge-only", action="store_true", help="Only merge batch results and run clustering (requires --run-id).")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # 如果未指定run_id，使用时间戳
    if args.run_id is None:
        from datetime import datetime
        args.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger.info("自动生成 run_id: %s", args.run_id)

    run_pipeline(args)
