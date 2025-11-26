#!/usr/bin/env python3
"""
比较多个 sentence-transformers 模型在仓库示例对上的相似度统计（similarity_statistics）

输出：`demo/output/compare_<model_id_sanitized>.json`，包含 similarity_statistics 和原始样本相似度列表。

用法：
  python tools\compare_models.py --models "sentence-transformers/paraphrase-multilingual-mpnet-base-v2,Alibaba-NLP/gte-multilingual-base"

注意：模型会从 Hugging Face 下载，可能需要网络或 token（私有模型）。
如果模型需要执行远程代码，确保已在 `core/processor.py` 中开启 `trust_remote_code=True`（已在仓库中修改）。
"""

import sys
import time
import json
import argparse
from pathlib import Path
from typing import List

import numpy as np

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bilingual_aligner.core.processor import get_text_processor
from bilingual_aligner.core.punctuation import (
    calculate_punctuation_similarity,
    PunctuationHandler,
)


def clear_caches(processor):
    try:
        processor.sentence_embedding_cache.clear()
        processor.similarity_cache.clear()
    except Exception:
        pass


def compute_stats(similarities: List[float]) -> dict:
    arr = np.array(similarities, dtype=float)
    if arr.size == 0:
        return {
            "count": 0,
            "mean": 0.0,
            "median": 0.0,
            "stdev": 0.0,
            "1%_low": 0.0,
            "5%_low": 0.0,
            "10%_low": 0.0,
            "25%_low": 0.0,
            "min": 0.0,
            "max": 0.0,
        }

    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "stdev": float(np.std(arr, ddof=0)),
        "1%_low": float(np.percentile(arr, 1)),
        "5%_low": float(np.percentile(arr, 5)),
        "10%_low": float(np.percentile(arr, 10)),
        "25%_low": float(np.percentile(arr, 25)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


class ModelComparator:
    def __init__(
        self, src_path: Path, tgt_path: Path, model_name: str, method: str = "paragraph"
    ):
        self.src_path = src_path
        self.tgt_path = tgt_path
        self.model_name = model_name
        self.method = method
        self.processor = get_text_processor(model_name=model_name)

        # preload model if possible
        try:
            self.processor._load_model()
            _ = self.processor.get_normalized_embedding_by_sentences(
                "Hello world.", method="mean"
            )
        except Exception:
            pass

    def load_lines(self):
        with open(self.src_path, "r", encoding="utf-8") as f:
            src_lines = [l.rstrip("\n\r") for l in f if l.strip()]
        with open(self.tgt_path, "r", encoding="utf-8") as f:
            tgt_lines = [l.rstrip("\n\r") for l in f if l.strip()]
        min_len = min(len(src_lines), len(tgt_lines))
        return src_lines[:min_len], tgt_lines[:min_len]

    def calc_punct_weight(self, s: str, t: str) -> float:
        return calculate_punctuation_similarity(s, t)

    def run(self) -> dict:
        src_lines, tgt_lines = self.load_lines()
        similarities = []
        total = len(src_lines)
        for idx, (s, t) in enumerate(zip(src_lines, tgt_lines), start=1):
            try:
                clear_caches(self.processor)
                sim = self.processor.calculate_similarity(s, t, method=self.method)
                weight = self.calc_punct_weight(s, t)
                final = max(0.0, float(sim) * float(weight))
            except Exception:
                final = 0.0
            similarities.append(final)
            if idx % 50 == 0 or idx == total:
                print(f"[{self.model_name}] processed {idx}/{total}")

        stats = compute_stats(similarities)
        return {
            "model": self.model_name,
            "method": self.method,
            "similarity_statistics": stats,
            "samples": similarities,
        }


def main():
    parser = argparse.ArgumentParser(
        description="Compare models and compute similarity_statistics on demo files"
    )
    parser.add_argument(
        "--models",
        help="Comma-separated model ids (Hugging Face or local paths)",
        default=(
            "sentence-transformers/paraphrase-multilingual-mpnet-base-v2,"
            "Alibaba-NLP/gte-multilingual-base"
        ),
    )
    parser.add_argument(
        "--method", choices=["paragraph", "sentence"], default="paragraph"
    )
    parser.add_argument(
        "--out-dir", default="demo/output", help="Directory to save JSON reports"
    )

    args = parser.parse_args()

    repo_root = Path(__file__).parent
    src_file = repo_root / "sample_zh.md"
    tgt_file = repo_root / "output" / "sample_en_repaired.txt"

    if not src_file.exists():
        print(f"Source file not found: {src_file}")
        return 1
    if not tgt_file.exists():
        print(f"Target file not found: {tgt_file}")
        return 1

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for m in models:
        print(f"\nStarting evaluation for model: {m}\n")
        comp = ModelComparator(src_file, tgt_file, model_name=m, method=args.method)
        result = comp.run()

        safe = m.replace("/", "_")
        out_path = out_dir / f"compare_{safe}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"Saved {out_path}\nSummary: {result['similarity_statistics']}\n")

    print("All done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
