#!/usr/bin/env python3
"""
统计 demo 原文与修复后译文逐行相关性与标点信息的脚本

用法：
  python demo/stats_repaired.py
可选参数：--source --repaired --out --model --k

输出：CSV 文件，包含每行的语义相似度、标点数、容差与是否超出容差，以及综合得分（语义 * 标点权重）
支持过滤空行，并输出综合得分的详细统计信息
"""
import sys
import csv
import argparse
import statistics
from pathlib import Path

# Ensure project root is importable when running from repo
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from bilingual_aligner.core.punctuation import (
    PunctuationHandler,
    calculate_punctuation_similarity,
)
from bilingual_aligner.core.processor import get_text_processor


def safe_calculate_similarity(processor, a: str, b: str) -> float:
    try:
        return processor.calculate_similarity(a, b)
    except Exception:
        # Fallback: exact match -> 1.0 else 0.0
        a2 = a.strip()
        b2 = b.strip()
        return 1.0 if a2 and b2 and a2 == b2 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=str,
        default=str(project_root / "demo" / "sample_zh.md"),
        help="源文件（包含空行）",
    )
    parser.add_argument(
        "--repaired",
        type=str,
        default=str(project_root / "demo" / "output" / "sample_en_repaired.txt"),
        help="修复后译文文件（按源行对齐）",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(project_root / "demo" / "repair_stats.csv"),
        help="输出 CSV 路径",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="可选 sentence-transformers 模型名（留空使用默认工厂）",
    )
    parser.add_argument(
        "--k",
        type=float,
        default=0.8,
        help="标点容差参数 k（默认 0.8）",
    )

    args = parser.parse_args()

    src_path = Path(args.source)
    repaired_path = Path(args.repaired)
    out_path = Path(args.out)

    if not src_path.exists():
        print(f"源文件不存在: {src_path}")
        return 2
    if not repaired_path.exists():
        print(f"修复文件不存在: {repaired_path}")
        return 2

    src_lines = [l.rstrip("\n\r") for l in src_path.open("r", encoding="utf-8")]
    repaired_lines = [
        l.rstrip("\n\r") for l in repaired_path.open("r", encoding="utf-8")
    ]

    # Prepare text processor (may download model on first run)
    processor = get_text_processor(model_name=args.model)

    rows = []
    combined_scores = []  # For statistical analysis
    semantic_scores = []
    punct_sim_scores = []
    total_combined = 0.0
    total_sem = 0.0
    total_punct_sim = 0.0
    count = 0

    max_lines = max(len(src_lines), len(repaired_lines))
    for idx in range(max_lines):
        src = src_lines[idx] if idx < len(src_lines) else ""
        tgt = repaired_lines[idx] if idx < len(repaired_lines) else ""

        src_str = src.strip()
        tgt_str = tgt.strip()

        # Filter empty lines (only process lines where at least one side is non-empty)
        if not (src_str or tgt_str):
            continue

        # Count punctuation in source and target lines
        p_src = PunctuationHandler.count_punctuation_line(src)
        p_tgt = PunctuationHandler.count_punctuation_line(tgt)
        punct_diff = abs(p_src - p_tgt)

        # Punctuation similarity weight (using new formula)
        punct_sim = calculate_punctuation_similarity(src, tgt)

        # Semantic similarity (embedding-based), safe fallback if model missing
        sem_sim = (
            safe_calculate_similarity(processor, src_str, tgt_str)
            if (src_str or tgt_str)
            else 0.0
        )

        combined = sem_sim * punct_sim

        rows.append(
            {
                "line_index": idx + 1,
                "source": src,
                "repaired": tgt,
                "semantic_similarity": round(sem_sim, 6),
                "punct_src": p_src,
                "punct_tgt": p_tgt,
                "punct_diff": punct_diff,
                "punct_similarity": round(punct_sim, 6),
                "combined_score": round(combined, 6),
            }
        )

        # Collect scores for statistics
        combined_scores.append(combined)
        semantic_scores.append(sem_sim)
        punct_sim_scores.append(punct_sim)

        total_combined += combined
        total_sem += sem_sim
        total_punct_sim += punct_sim
        count += 1

    # Write CSV
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "line_index",
        "semantic_similarity",
        "punct_src",
        "punct_tgt",
        "punct_diff",
        "punct_similarity",
        "combined_score",
        "source",
        "repaired",
    ]

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r[k] for k in fieldnames})

    avg_combined = total_combined / count if count else 0.0
    avg_sem = total_sem / count if count else 0.0
    avg_punct_sim = total_punct_sim / count if count else 0.0

    # Calculate comprehensive statistics for combined scores
    if combined_scores:
        combined_sorted = sorted(combined_scores)
        combined_stats = {
            "count": len(combined_scores),
            "mean": statistics.mean(combined_scores),
            "median": statistics.median(combined_scores),
            "stdev": (
                statistics.stdev(combined_scores) if len(combined_scores) > 1 else 0.0
            ),
            "min": min(combined_scores),
            "max": max(combined_scores),
            "q25": combined_sorted[len(combined_sorted) // 4],
            "q75": combined_sorted[3 * len(combined_sorted) // 4],
            "q90": combined_sorted[int(0.9 * len(combined_sorted))],
            "q95": combined_sorted[int(0.95 * len(combined_sorted))],
        }
    else:
        combined_stats = {
            k: 0.0
            for k in [
                "count",
                "mean",
                "median",
                "stdev",
                "min",
                "max",
                "q25",
                "q75",
                "q90",
                "q95",
            ]
        }

    print(f"Wrote stats to: {out_path}")
    print(f"Lines processed: {count}")
    print(f"Empty lines filtered: True")
    print()
    print("=== 综合得分统计信息 ===")
    print(f"样本数量: {combined_stats['count']}")
    print(f"平均值: {combined_stats['mean']:.4f}")
    print(f"中位数: {combined_stats['median']:.4f}")
    print(f"标准差: {combined_stats['stdev']:.4f}")
    print(f"最小值: {combined_stats['min']:.4f}")
    print(f"最大值: {combined_stats['max']:.4f}")
    print(f"25%分位数: {combined_stats['q25']:.4f}")
    print(f"75%分位数: {combined_stats['q75']:.4f}")
    print(f"90%分位数: {combined_stats['q90']:.4f}")
    print(f"95%分位数: {combined_stats['q95']:.4f}")
    print()
    print("=== 基础指标平均值 ===")
    print(
        f"Avg semantic sim: {avg_sem:.4f}, avg punct sim: {avg_punct_sim:.4f}, avg combined: {avg_combined:.4f}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
