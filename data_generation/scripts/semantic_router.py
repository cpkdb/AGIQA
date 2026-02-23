#!/usr/bin/env python3
"""
SemanticRouter: Stage 1 - 数据筛选
对正样本 prompt 进行语义分析，判断 (prompt, dimension) 兼容性。

可独立运行：
  python scripts/semantic_router.py --input data/prompts.json --output data/prompts_tagged.json

也可被 pipeline.py 导入使用。
"""

import argparse
import json
import logging
import re
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

CONFIG_DIR = Path(__file__).parent.parent / "config"


class SemanticRouter:
    """对正样本 prompt 进行语义分析 + 维度兼容性筛选"""

    def __init__(
        self,
        tag_config_path: str = None,
        dimension_req_path: str = None,
    ):
        if tag_config_path is None:
            tag_config_path = str(CONFIG_DIR / "semantic_tag_requirements.json")
        if dimension_req_path is None:
            dimension_req_path = str(CONFIG_DIR / "dimension_requirements.yaml")

        with open(tag_config_path, "r", encoding="utf-8") as f:
            self.tag_config = json.load(f)

        with open(dimension_req_path, "r", encoding="utf-8") as f:
            self.dimension_reqs = yaml.safe_load(f) or {}

        self._keyword_index = self._build_keyword_index()

    # ------------------------------------------------------------------ #
    #  关键词索引构建
    # ------------------------------------------------------------------ #

    def _build_keyword_index(self) -> Dict[str, List[str]]:
        """tag -> keyword list"""
        idx = {}
        for tag_name, tag_info in self.tag_config.get("tags", {}).items():
            kws = tag_info.get("detection_keywords", [])
            idx[tag_name] = [k.lower() for k in kws]
        return idx

    # ------------------------------------------------------------------ #
    #  单条分析
    # ------------------------------------------------------------------ #

    def analyze(self, prompt: str) -> Dict:
        """
        对单条 prompt 进行快速关键词标签化。

        Returns:
            PromptSignature dict，例如：
            {
                "tags": ["has_person", "has_hand"],
                "has_person": True,
                "has_hand": True,
                ...
            }
        """
        prompt_lower = prompt.lower()
        tags = []

        for tag_name, keywords in self._keyword_index.items():
            for kw in keywords:
                if kw in prompt_lower:
                    tags.append(tag_name)
                    break

        # 特殊处理: has_quoted_text
        if re.search(r'[\'"][^\'"]{2,}[\'"]', prompt):
            tags.append("has_quoted_text")

        tags = list(set(tags))

        sig = {"tags": tags}
        for tag_name in self.tag_config.get("tags", {}):
            sig[tag_name] = tag_name in tags
        sig["has_quoted_text"] = "has_quoted_text" in tags
        return sig

    # ------------------------------------------------------------------ #
    #  兼容性判定
    # ------------------------------------------------------------------ #

    def is_compatible(
        self, signature: Dict, dimension: str
    ) -> Tuple[bool, str]:
        """
        判断 (prompt_signature, dimension) 是否兼容。

        Returns:
            (compatible: bool, reason: str)
        """
        reqs = self.dimension_reqs.get(dimension)
        if reqs is None:
            return True, "no_requirements"

        required = reqs.get("required_tags")
        if not required:
            return True, "preferred_only"

        prompt_tags = set(signature.get("tags", []))
        # OR 逻辑: 至少一个 required_tag 匹配即可
        if any(t in prompt_tags for t in required):
            return True, "ok"

        return False, f"missing_required:{','.join(required)}"

    def has_preferred(self, signature: Dict, dimension: str) -> bool:
        """检查是否有优先标签（用于排序而非过滤）"""
        reqs = self.dimension_reqs.get(dimension)
        if reqs is None:
            return True
        preferred = reqs.get("preferred_tags")
        if not preferred:
            return True
        prompt_tags = set(signature.get("tags", []))
        return any(t in prompt_tags for t in preferred)

    # ------------------------------------------------------------------ #
    #  批量标签化
    # ------------------------------------------------------------------ #

    def batch_analyze(self, prompts: List[Dict]) -> List[Dict]:
        """
        批量标签化（纯关键词，无 LLM 调用）。

        Args:
            prompts: [{"prompt": "...", ...}, ...]

        Returns:
            同结构列表，每项增加 "semantic_tags" 和 "signature" 字段
        """
        results = []
        for item in prompts:
            item = item.copy()
            text = item.get("prompt", item.get("text", ""))
            sig = self.analyze(text)
            item["semantic_tags"] = sig["tags"]
            item["signature"] = sig
            results.append(item)
        return results

    def filter_prompts_for_dimension(
        self,
        tagged_prompts: List[Dict],
        dimension: str,
    ) -> List[Dict]:
        """筛选与指定维度兼容的 prompts"""
        compatible = []
        for item in tagged_prompts:
            sig = item.get("signature")
            if sig is None:
                text = item.get("prompt", item.get("text", ""))
                sig = self.analyze(text)
            ok, _ = self.is_compatible(sig, dimension)
            if ok:
                compatible.append(item)
        return compatible

    # ------------------------------------------------------------------ #
    #  覆盖率统计
    # ------------------------------------------------------------------ #

    def coverage_report(
        self, tagged_prompts: List[Dict], dimensions: List[str] = None
    ) -> Dict:
        """生成各维度的 prompt 覆盖率报告"""
        if dimensions is None:
            dimensions = list(self.dimension_reqs.keys())

        total = len(tagged_prompts)
        report = {}
        for dim in dimensions:
            compatible = self.filter_prompts_for_dimension(tagged_prompts, dim)
            report[dim] = {
                "available": len(compatible),
                "total": total,
                "coverage": round(len(compatible) / total * 100, 2) if total else 0,
            }
        return report


# ====================================================================== #
#  CLI 入口
# ====================================================================== #

def main():
    parser = argparse.ArgumentParser(description="SemanticRouter: prompt 标签化 + 维度兼容性筛选")
    parser.add_argument("--input", type=str, required=True, help="输入 prompts JSON 文件")
    parser.add_argument("--output", type=str, default=None, help="输出标签化后的 JSON 文件")
    parser.add_argument("--dimension", type=str, default=None, help="筛选指定维度的兼容 prompts")
    parser.add_argument("--report", action="store_true", help="输出覆盖率报告")
    args = parser.parse_args()

    # 加载 prompts
    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        prompts = data
    elif isinstance(data, dict):
        prompts = data.get("prompts", data.get("data", []))
    else:
        raise ValueError("Unsupported format")

    router = SemanticRouter()
    tagged = router.batch_analyze(prompts)
    logger.info(f"Tagged {len(tagged)} prompts")

    if args.dimension:
        filtered = router.filter_prompts_for_dimension(tagged, args.dimension)
        logger.info(f"Dimension '{args.dimension}': {len(filtered)}/{len(tagged)} compatible")

    if args.report:
        report = router.coverage_report(tagged)
        print(json.dumps(report, indent=2, ensure_ascii=False))

    if args.output:
        out_data = {
            "version": "2.0",
            "source": args.input,
            "total_prompts": len(tagged),
            "prompts": tagged,
        }
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(out_data, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
