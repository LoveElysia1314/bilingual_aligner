#!/usr/bin/env python3
"""
Tool specifically for testing the impact of misalignment on punctuation similarity weights

Test content:
1. Baseline punctuation similarity weights for correct alignment
2. Reduction in punctuation similarity weights for various misalignment modes
3. Punctuation count statistics
4. Deviation statistics comparison: standard method vs square root denominator method

Usage:
    python test_misalignment_impact.py
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from bilingual_aligner.core.punctuation import (
    PunctuationHandler,
    calculate_punctuation_similarity,
)


def load_sample_texts():
    """Load sample texts, filter empty lines"""
    try:
        # Source: Chinese
        with open(
            Path(__file__).parent / "demo" / "sample_zh.md", "r", encoding="utf-8"
        ) as f:
            zh_lines = [line.strip() for line in f.readlines() if line.strip()]

        # Translation: English (repaired)
        with open(
            Path(__file__).parent / "demo" / "output" / "sample_en_repaired.txt",
            "r",
            encoding="utf-8",
        ) as f:
            en_lines = [line.strip() for line in f.readlines() if line.strip()]

        # Match the number of lines in both files (count full text)
        n_lines = min(len(en_lines), len(zh_lines))
        print(f"加载全文：{n_lines} 对句子")
        return en_lines[:n_lines], zh_lines[:n_lines]
    except FileNotFoundError as e:
        print(f"Error: Sample file not found {e}")
        return None, None


def calculate_punctuation_stats_and_weights(en_lines, zh_lines):
    """Calculate punctuation statistics and punctuation similarity weights for various alignment modes"""
    n_lines = len(en_lines)

    # Store punctuation counts
    en_punct_counts = []
    zh_punct_counts = []

    # Correct alignment weights
    correct_weights = []

    # Misalignment weights
    misalignment_weights = {
        "ZH[i] vs EN[i+1]": [],
        "ZH[i]+ZH[i+1] vs EN[i]": [],
        "EN[i] vs ZH[i+1]": [],
    }

    # Calculate punctuation counts and correct alignment weights
    for i in range(n_lines):
        en_punct = PunctuationHandler.count_punctuation_line(en_lines[i])
        zh_punct = PunctuationHandler.count_punctuation_line(zh_lines[i])

        en_punct_counts.append(en_punct)
        zh_punct_counts.append(zh_punct)

        weight = calculate_punctuation_similarity(zh_lines[i], en_lines[i])
        correct_weights.append(weight)

    # Calculate misalignment weights
    for i in range(n_lines):
        # ZH[i] vs EN[i+1]
        tgt_idx = (i + 1) % n_lines
        weight = calculate_punctuation_similarity(zh_lines[i], en_lines[tgt_idx])
        misalignment_weights["ZH[i] vs EN[i+1]"].append(weight)

        # ZH[i]+ZH[i+1] vs EN[i]
        src_combined = zh_lines[i] + " " + zh_lines[(i + 1) % n_lines]
        weight = calculate_punctuation_similarity(src_combined, en_lines[i])
        misalignment_weights["ZH[i]+ZH[i+1] vs EN[i]"].append(weight)

        # EN[i] vs ZH[i+1]
        tgt_idx = (i + 1) % n_lines
        weight = calculate_punctuation_similarity(en_lines[i], zh_lines[tgt_idx])
        misalignment_weights["EN[i] vs ZH[i+1]"].append(weight)

    return en_punct_counts, zh_punct_counts, correct_weights, misalignment_weights


def analyze_results(
    en_punct_counts, zh_punct_counts, correct_weights, misalignment_weights
):
    """Analyze and display results, including probability density plots"""
    print("Punctuation similarity weight test results")
    print("=" * 80)

    # Punctuation count statistics
    print("Punctuation count statistics:")
    print(f"  Total English punctuation: {sum(en_punct_counts)}")
    print(f"  Total Chinese punctuation: {sum(zh_punct_counts)}")
    print(
        f"  Average English punctuation: {np.mean(en_punct_counts):.2f} ± {np.std(en_punct_counts):.2f}"
    )
    print(
        f"  Average Chinese punctuation: {np.mean(zh_punct_counts):.2f} ± {np.std(zh_punct_counts):.2f}"
    )
    print(f"  Test sample pairs: {len(correct_weights)}")
    print()

    # Calculate deviations (without absolute value) and average punctuation counts
    deviations = []  # Signed deviations: (EN - ZH) / avg
    deviations_sqrt = []  # Square root denominator deviations: (EN - ZH) / sqrt(avg)
    deviations_const = (
        []
    )  # Constant denominator deviations: (EN - ZH) / C (C is global average)
    absolute_deviations = []  # Absolute deviations: |EN - ZH|
    avg_puncts = []

    # Calculate global average as constant denominator
    global_avg = np.mean(
        [
            (en_punct_counts[i] + zh_punct_counts[i]) / 2.0
            for i in range(len(correct_weights))
        ]
    )

    for i in range(len(correct_weights)):
        zh_punct = zh_punct_counts[i]
        en_punct = en_punct_counts[i]
        avg_punct = (zh_punct + en_punct) / 2.0
        abs_diff = abs(en_punct - zh_punct)

        if avg_punct > 0:
            deviation = (
                en_punct - zh_punct
            ) / avg_punct  # Relative deviation: (EN - ZH) / avg
            deviation_sqrt = (en_punct - zh_punct) / np.sqrt(
                avg_punct
            )  # Square root deviation: (EN - ZH) / sqrt(avg)
            deviation_const = (
                (en_punct - zh_punct) / global_avg if global_avg > 0 else 0
            )  # Constant denominator: (EN - ZH) / C

            deviations.append(deviation)
            deviations_sqrt.append(deviation_sqrt)
            deviations_const.append(deviation_const)
            absolute_deviations.append(abs_diff)
            avg_puncts.append(avg_punct)

    print("Deviation statistics comparison:")
    print(f"\nGlobal average punctuation (as constant denominator C): {global_avg:.4f}")

    print("\n1. Standard relative deviation (EN - ZH) / avg:")
    print(f"  Mean: {np.mean(deviations):.4f} ± {np.std(deviations):.4f}")
    print(f"  Range: min={min(deviations):.4f}, max={max(deviations):.4f}")
    print(
        f"  Absolute mean: {np.mean(np.abs(deviations)):.4f} ± {np.std(np.abs(deviations)):.4f}"
    )

    print("\n2. Square root denominator deviation (EN - ZH) / sqrt(avg):")
    print(f"  Mean: {np.mean(deviations_sqrt):.4f} ± {np.std(deviations_sqrt):.4f}")
    print(f"  Range: min={min(deviations_sqrt):.4f}, max={max(deviations_sqrt):.4f}")
    print(
        f"  Absolute mean: {np.mean(np.abs(deviations_sqrt)):.4f} ± {np.std(np.abs(deviations_sqrt)):.4f}"
    )

    print(
        "\n3. Constant denominator deviation (EN - ZH) / C (C={:.4f}):".format(
            global_avg
        )
    )
    print(f"  Mean: {np.mean(deviations_const):.4f} ± {np.std(deviations_const):.4f}")
    print(f"  Range: min={min(deviations_const):.4f}, max={max(deviations_const):.4f}")
    print(
        f"  Absolute mean: {np.mean(np.abs(deviations_const)):.4f} ± {np.std(np.abs(deviations_const)):.4f}"
    )

    print("\n4. Absolute deviation |EN - ZH| (no normalization):")
    print(
        f"  Mean: {np.mean(absolute_deviations):.4f} ± {np.std(absolute_deviations):.4f}"
    )
    print(
        f"  Range: min={min(absolute_deviations):.4f}, max={max(absolute_deviations):.4f}"
    )

    # Compare the differences of the four methods
    print("\n5. Comparison of the four methods:")
    print(
        f"  sqrt/standard absolute ratio: {np.mean(np.abs(deviations_sqrt)) / np.mean(np.abs(deviations)):.4f}"
    )
    print(
        f"  constant/standard absolute ratio: {np.mean(np.abs(deviations_const)) / np.mean(np.abs(deviations)):.4f}"
    )
    print(
        f"  sqrt/定值 的绝对值比值: {np.mean(np.abs(deviations_sqrt)) / np.mean(np.abs(deviations_const)):.4f}"
    )
    print()

    # Correct alignment weight statistics
    correct_avg = np.mean(correct_weights)
    correct_std = np.std(correct_weights)
    print(f"Correct alignment weight baseline: {correct_avg:.4f} ± {correct_std:.4f}")
    print(
        f"  Weight distribution: min={min(correct_weights):.4f}, max={max(correct_weights):.4f}"
    )
    high_weight_count = sum(1 for w in correct_weights if w > 0.8)
    print(
        f"  High weight (>0.8) ratio: {high_weight_count}/{len(correct_weights)} ({high_weight_count/len(correct_weights)*100:.1f}%)"
    )
    print()

    # Create chart: Compare four deviation calculation methods (2x2 layout)
    axes = plt.subplots(2, 2, figsize=(16, 12))[1]

    # Chart 1: Standard relative deviation vs average punctuation count
    ax1 = axes[0, 0]
    ax1.scatter(avg_puncts, deviations, alpha=0.6, edgecolors="k", s=50, color="blue")
    ax1.axhline(y=0, color="r", linestyle="--", alpha=0.5)
    ax1.set_xlabel("Average punctuation count")
    ax1.set_ylabel("Deviation (EN - ZH) / avg")
    ax1.set_title(
        "Method 1: Standard Relative Deviation\nMean |dev| = {:.4f}".format(
            np.mean(np.abs(deviations))
        )
    )
    ax1.grid(True, alpha=0.3)

    # Add trend line
    if len(avg_puncts) > 1:
        z = np.polyfit(avg_puncts, np.abs(deviations), 1)
        p = np.poly1d(z)
        ax1.plot(
            sorted(avg_puncts),
            p(sorted(avg_puncts)),
            "r--",
            alpha=0.8,
            label=f"Trend of |dev|",
        )
        ax1.legend()

    # Chart 2: Square root denominator deviation vs average punctuation count
    ax2 = axes[0, 1]
    ax2.scatter(
        avg_puncts, deviations_sqrt, alpha=0.6, edgecolors="k", s=50, color="orange"
    )
    ax2.axhline(y=0, color="r", linestyle="--", alpha=0.5)
    ax2.set_xlabel("Average punctuation count")
    ax2.set_ylabel("Deviation (EN - ZH) / sqrt(avg)")
    ax2.set_title(
        "Method 2: Sqrt Denominator Deviation\nMean |dev| = {:.4f}".format(
            np.mean(np.abs(deviations_sqrt))
        )
    )
    ax2.grid(True, alpha=0.3)

    # Add trend line
    if len(avg_puncts) > 1:
        z = np.polyfit(avg_puncts, np.abs(deviations_sqrt), 1)
        p = np.poly1d(z)
        ax2.plot(
            sorted(avg_puncts),
            p(sorted(avg_puncts)),
            "r--",
            alpha=0.8,
            label=f"Trend of |dev|",
        )
        ax2.legend()

    # Chart 3: Constant denominator deviation vs average punctuation count
    ax3 = axes[1, 0]
    ax3.scatter(
        avg_puncts, deviations_const, alpha=0.6, edgecolors="k", s=50, color="green"
    )
    ax3.axhline(y=0, color="r", linestyle="--", alpha=0.5)
    ax3.set_xlabel("Average punctuation count")
    ax3.set_ylabel("Deviation (EN - ZH) / C")
    ax3.set_title(
        "Method 3: Constant Denominator (C={:.4f})\nMean |dev| = {:.4f}".format(
            global_avg, np.mean(np.abs(deviations_const))
        )
    )
    ax3.grid(True, alpha=0.3)

    # Add trend line
    if len(avg_puncts) > 1:
        z = np.polyfit(avg_puncts, np.abs(deviations_const), 1)
        p = np.poly1d(z)
        ax3.plot(
            sorted(avg_puncts),
            p(sorted(avg_puncts)),
            "r--",
            alpha=0.8,
            label=f"Trend of |dev|",
        )
        ax3.legend()

    # Chart 4: Absolute deviation vs average punctuation count
    ax4 = axes[1, 1]
    ax4.scatter(
        avg_puncts, absolute_deviations, alpha=0.6, edgecolors="k", s=50, color="purple"
    )
    ax4.set_xlabel("Average punctuation count")
    ax4.set_ylabel("Absolute Deviation |EN - ZH|")
    ax4.set_title(
        "Method 4: Absolute Deviation (No Normalization)\nMean = {:.4f}".format(
            np.mean(absolute_deviations)
        )
    )
    ax4.grid(True, alpha=0.3)

    # Add trend line
    if len(avg_puncts) > 1:
        z = np.polyfit(avg_puncts, absolute_deviations, 1)
        p = np.poly1d(z)
        ax4.plot(
            sorted(avg_puncts),
            p(sorted(avg_puncts)),
            "r--",
            alpha=0.8,
            label=f"Trend of |dev|",
        )
        ax4.legend()

    plt.tight_layout()

    # Save chart
    plot_path = Path(__file__).parent / "punctuation_error_analysis.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"统计图已保存到: {plot_path}")
    plt.close()

    # Misalignment mode analysis
    print("Misalignment mode analysis:")
    higher_count = 0
    total_comparisons = 0

    for name, weights in misalignment_weights.items():
        avg_weight = np.mean(weights)
        std_weight = np.std(weights)

        impact = correct_avg - avg_weight
        impact_pct = (impact / correct_avg * 100) if correct_avg > 0 else 0
        status = "↓" if impact > 0 else "↑"

        print(
            f"  {name:<25} - Mean: {avg_weight:.4f} ± {std_weight:.4f} ({status}{abs(impact):.4f}, {status}{abs(impact_pct):.1f}%)"
        )

        # Count the number of times with higher scores
        for i, weight in enumerate(weights):
            if weight > correct_weights[i]:
                higher_count += 1
            total_comparisons += 1

    print()
    if total_comparisons > 0:
        higher_percentage = (higher_count / total_comparisons) * 100
        print(
            f"统计结果: {higher_count}/{total_comparisons} ({higher_percentage:.1f}%) Misalignment weights高于正确对齐"
        )

        if higher_count == 0:
            print("✅ 所有错位对齐都正确降低了标点相似度权重")
        else:
            print(f"⚠️  {higher_count} 次Misalignment weights意外高于正确对齐")

    return higher_count == 0


def main():
    """主函数"""
    print("开始测试标点相似度权重...")
    print()

    # Load texts
    en_lines, zh_lines = load_sample_texts()
    if en_lines is None or zh_lines is None:
        return 1

    # Calculate punctuation statistics and weights
    en_punct_counts, zh_punct_counts, correct_weights, misalignment_weights = (
        calculate_punctuation_stats_and_weights(en_lines, zh_lines)
    )

    # Analysis results
    success = analyze_results(
        en_punct_counts, zh_punct_counts, correct_weights, misalignment_weights
    )

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
