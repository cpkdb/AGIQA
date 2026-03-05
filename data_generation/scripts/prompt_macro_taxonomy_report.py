#!/usr/bin/env python3
"""
Macro taxonomy tagging and macro-by-dimension coverage reporting.

This stage adds a coarse production-oriented macro scene category to each prompt
in the current working pool, then computes:
- global macro distribution
- macro-by-dimension compatibility matrix
"""

from __future__ import annotations

import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Dict, Iterable, List, Optional


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
TAXONOMY_PATH = DATA_DIR / "prompt_macro_taxonomy_v1.json"
TAG_CONFIG_PATH = SCRIPT_DIR.parent / "config" / "semantic_tag_requirements.json"

PORTRAIT_WORDS = {
    "portrait", "headshot", "close-up", "close up", "selfie", "studio portrait",
    "looking at camera", "looking at viewer",
}
ACTION_WORDS = {
    "running", "walking", "dancing", "jumping", "holding", "pointing", "hugging",
    "fighting", "talking", "working", "typing", "cooking", "playing", "posing",
    "smiling", "laughing",
}
TEXT_MEDIA_WORDS = {
    "sign", "signage", "poster", "label", "logo", "menu", "screen", "display",
    "storefront", "banner", "book", "cover", "packaging", "receipt", "interface",
    "title", "headline", "billboard",
}
FOOD_WORDS = {
    "food", "dish", "meal", "plate", "bowl", "sushi", "burger", "pizza", "salad",
    "coffee", "tea", "drink", "dessert", "cake", "restaurant", "dining", "table",
}
ANIMAL_WORDS = {
    "cat", "dog", "bird", "horse", "tiger", "lion", "wolf", "fox", "bear", "animal",
    "creature", "monkey", "fish", "insect",
}
INDOOR_WORDS = {
    "room", "living room", "bedroom", "kitchen", "office", "classroom", "studio",
    "indoors", "interior", "hallway", "shop", "store", "cafe", "restaurant",
}
URBAN_WORDS = {
    "street", "city", "road", "sidewalk", "plaza", "station", "downtown", "storefront",
    "traffic", "crosswalk", "building exterior", "parking lot",
}
NATURE_WORDS = {
    "mountain", "forest", "lake", "river", "beach", "ocean", "waterfall", "meadow",
    "park", "nature", "sunset", "sunrise", "valley", "snow", "desert",
}
EVENT_WORDS = {
    "concert", "festival", "wedding", "ceremony", "parade", "performance", "stage",
    "sports", "match", "crowd", "celebration", "party",
}
OBJECT_WORDS = {
    "tool", "device", "product", "chair", "table", "lamp", "phone", "camera", "bottle",
    "book", "box", "package", "toy", "machine", "watch", "furniture",
}


def _read_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> List[Dict]:
    records: List[Dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _write_jsonl(path: Path, records: Iterable[Dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _load_prompt_filter_module():
    module_path = SCRIPT_DIR / "prompt_filter.py"
    spec = spec_from_file_location("prompt_filter", module_path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _contains_any(text: str, phrases: Iterable[str]) -> bool:
    lower = text.lower()
    return any(phrase in lower for phrase in phrases)


def assign_macro_category(record: Dict) -> str:
    """Assign one macro taxonomy category using coarse heuristic rules."""
    prompt = record.get("prompt", "")
    tags = set(record.get("semantic_tags", []))

    if {"has_text", "has_logo_or_symbol"} & tags or _contains_any(prompt, TEXT_MEDIA_WORDS):
        return "commercial_text_media"

    if _contains_any(prompt, FOOD_WORDS):
        return "food_tabletop"

    if "has_face" in tags or _contains_any(prompt, PORTRAIT_WORDS):
        return "human_portrait"

    if "has_person" in tags and (
        _contains_any(prompt, ACTION_WORDS) or "has_multiple_objects" in tags
    ):
        return "human_activity_interaction"

    if "has_animal" in tags or _contains_any(prompt, ANIMAL_WORDS):
        return "animals_creatures"

    if _contains_any(prompt, INDOOR_WORDS):
        return "indoor_structured_space"

    if _contains_any(prompt, URBAN_WORDS):
        return "urban_built_outdoor"

    if _contains_any(prompt, NATURE_WORDS) or "has_reflective_surface" in tags:
        return "natural_landscape"

    if _contains_any(prompt, EVENT_WORDS):
        return "events_performance_culture"

    if _contains_any(prompt, OBJECT_WORDS):
        return "objects_products_tools"

    return "objects_products_tools"


def build_macro_reports(
    records: Iterable[Dict],
    dimensions: Optional[List[str]] = None,
) -> Dict:
    """Tag records with macro categories and build distribution + matrix reports."""
    taxonomy = _read_json(TAXONOMY_PATH)
    categories = list(taxonomy["categories"].keys())
    prompt_filter = _load_prompt_filter_module()
    tag_requirements = prompt_filter.load_tag_requirements(TAG_CONFIG_PATH)
    if dimensions is None:
        dimensions = list(tag_requirements.keys())

    tagged_records: List[Dict] = []
    distribution_counts = {name: 0 for name in categories}

    for record in records:
        item = dict(record)
        macro_category = assign_macro_category(item)
        item["macro_taxonomy"] = macro_category
        tagged_records.append(item)
        distribution_counts[macro_category] += 1

    total_prompts = len(tagged_records)
    distribution = {
        "total_prompts": total_prompts,
        "categories": {
            name: {
                "count": distribution_counts[name],
                "coverage": round((distribution_counts[name] / total_prompts) * 100, 2)
                if total_prompts
                else 0.0,
            }
            for name in categories
        },
    }

    matrix_dimensions: Dict[str, Dict] = {}
    for dimension in dimensions:
        by_macro = {name: 0 for name in categories}
        total_available = 0
        for item in tagged_records:
            compatible, _ = prompt_filter.is_prompt_compatible(
                prompt=item.get("prompt", ""),
                tags=item.get("semantic_tags", []),
                dimension=dimension,
                tag_requirements=tag_requirements,
            )
            if compatible:
                total_available += 1
                by_macro[item["macro_taxonomy"]] += 1

        matrix_dimensions[dimension] = {
            "total_available": total_available,
            "by_macro_category": by_macro,
        }

    matrix = {
        "total_prompts": total_prompts,
        "dimensions": matrix_dimensions,
        "categories": categories,
    }

    return {
        "tagged_records": tagged_records,
        "distribution": distribution,
        "matrix": matrix,
    }


def write_macro_reports(
    input_path: str,
    output_dir: str,
    dimensions: Optional[List[str]] = None,
) -> Dict[str, str]:
    """Read a working pool file, write macro-tagged prompts and report artifacts."""
    input_file = Path(input_path).resolve()
    output_root = Path(output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    records = _read_jsonl(input_file)
    result = build_macro_reports(records, dimensions=dimensions)

    tagged_path = output_root / "working_pool_macro_tagged.jsonl"
    distribution_path = output_root / "macro_taxonomy_distribution_report.json"
    matrix_path = output_root / "macro_dimension_matrix_report.json"

    _write_jsonl(tagged_path, result["tagged_records"])
    distribution_path.write_text(
        json.dumps(result["distribution"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    matrix_path.write_text(
        json.dumps(result["matrix"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "tagged_path": str(tagged_path),
        "distribution_path": str(distribution_path),
        "matrix_path": str(matrix_path),
    }


__all__ = [
    "assign_macro_category",
    "build_macro_reports",
    "write_macro_reports",
]
