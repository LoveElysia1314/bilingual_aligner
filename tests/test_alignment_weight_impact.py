#!/usr/bin/env python3
"""
专门测试错位对齐对标点相似度权重影响的工具

测试内容：
1. 正确对齐的标点相似度权重基准
2. 各种错位模式的标点相似度权重降低
3. 标点数统计
4. 偏差统计对比：标准方法 vs 根号分母方法

使用方法：
    python test_misalignment_impact.py
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from bilingual_aligner.core.punctuation import (
    PunctuationHandler,
    calculate_punctuation_similarity,
)


def load_sample_texts():
    """加载样本文本，过滤空行"""
    try:
        # 原文：中文
        with open(
            Path(__file__).parent / "demo" / "sample_zh.md", "r", encoding="utf-8"
        ) as f:
            zh_lines = [line.strip() for line in f.readlines() if line.strip()]

        # 译文：英文（repaired）
        with open(
            Path(__file__).parent / "demo" / "output" / "sample_en_repaired.txt",
            "r",
            encoding="utf-8",
        ) as f:
            en_lines = [line.strip() for line in f.readlines() if line.strip()]

        # 匹配两个文件的行数（统计全文）
        n_lines = min(len(en_lines), len(zh_lines))
        print(f"加载全文：{n_lines} 对句子")
        return en_lines[:n_lines], zh_lines[:n_lines]
    except FileNotFoundError as e:
        print(f"错误：找不到样本文件 {e}")
        return None, None


def calculate_punctuation_stats_and_weights(en_lines, zh_lines):
    """计算标点数统计和各种对齐模式的标点相似度权重"""
    n_lines = len(en_lines)

    # 存储标点数
    en_punct_counts = []
    zh_punct_counts = []

    # 正确对齐权重
    correct_weights = []

    # 错位权重
    misalignment_weights = {
        "ZH[i] vs EN[i+1]": [],
        "ZH[i]+ZH[i+1] vs EN[i]": [],
        "EN[i] vs ZH[i+1]": [],
    }

    # 计算标点数和正确对齐权重
    for i in range(n_lines):
        en_punct = PunctuationHandler.count_punctuation_line(en_lines[i])
        zh_punct = PunctuationHandler.count_punctuation_line(zh_lines[i])

        en_punct_counts.append(en_punct)
        zh_punct_counts.append(zh_punct)

        weight = calculate_punctuation_similarity(zh_lines[i], en_lines[i])
        correct_weights.append(weight)

    # 计算错位权重
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
    """分析和显示结果，包括概率密度图"""
    print("标点相似度权重测试结果")
    print("=" * 80)

    # 标点数统计
    print("标点数统计:")
    print(f"  英文总标点数: {sum(en_punct_counts)}")
    print(f"  中文总标点数: {sum(zh_punct_counts)}")
    print(
        f"  英文平均标点数: {np.mean(en_punct_counts):.2f} ± {np.std(en_punct_counts):.2f}"
    )
    print(
        f"  中文平均标点数: {np.mean(zh_punct_counts):.2f} ± {np.std(zh_punct_counts):.2f}"
    )
    print(f"  测试样本数: {len(correct_weights)} 对")
    print()

    # 计算偏差（不取绝对值）和平均标点数
    deviations = []  # 有符号的偏差：(EN - ZH) / avg
    deviations_sqrt = []  # 取分母根号的偏差：(EN - ZH) / sqrt(avg)
    deviations_const = []  # 分母为定值的偏差：(EN - ZH) / C（C为全局平均）
    absolute_deviations = []  # 绝对偏差：|EN - ZH|
    avg_puncts = []

    # 计算全局平均值作为定值分母
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
            deviation = (en_punct - zh_punct) / avg_punct  # 相对偏差：(EN - ZH) / avg
            deviation_sqrt = (en_punct - zh_punct) / np.sqrt(
                avg_punct
            )  # 根号偏差：(EN - ZH) / sqrt(avg)
            deviation_const = (
                (en_punct - zh_punct) / global_avg if global_avg > 0 else 0
            )  # 定值分母：(EN - ZH) / C

            deviations.append(deviation)
            deviations_sqrt.append(deviation_sqrt)
            deviations_const.append(deviation_const)
            absolute_deviations.append(abs_diff)
            avg_puncts.append(avg_punct)

    print("偏差统计对比：")
    print(f"\n全局平均标点数（作为定值分母C）: {global_avg:.4f}")

    print("\n1. 标准相对偏差 (EN - ZH) / avg：")
    print(f"  平均: {np.mean(deviations):.4f} ± {np.std(deviations):.4f}")
    print(f"  范围: min={min(deviations):.4f}, max={max(deviations):.4f}")
    print(
        f"  绝对值平均: {np.mean(np.abs(deviations)):.4f} ± {np.std(np.abs(deviations)):.4f}"
    )

    print("\n2. 根号分母偏差 (EN - ZH) / sqrt(avg)：")
    print(f"  平均: {np.mean(deviations_sqrt):.4f} ± {np.std(deviations_sqrt):.4f}")
    print(f"  范围: min={min(deviations_sqrt):.4f}, max={max(deviations_sqrt):.4f}")
    print(
        f"  绝对值平均: {np.mean(np.abs(deviations_sqrt)):.4f} ± {np.std(np.abs(deviations_sqrt)):.4f}"
    )

    print("\n3. 定值分母偏差 (EN - ZH) / C（C={:.4f}）：".format(global_avg))
    print(f"  平均: {np.mean(deviations_const):.4f} ± {np.std(deviations_const):.4f}")
    print(f"  范围: min={min(deviations_const):.4f}, max={max(deviations_const):.4f}")
    print(
        f"  绝对值平均: {np.mean(np.abs(deviations_const)):.4f} ± {np.std(np.abs(deviations_const)):.4f}"
    )

    print("\n4. 绝对偏差 |EN - ZH|（不做任何归一化）：")
    print(
        f"  平均: {np.mean(absolute_deviations):.4f} ± {np.std(absolute_deviations):.4f}"
    )
    print(
        f"  范围: min={min(absolute_deviations):.4f}, max={max(absolute_deviations):.4f}"
    )

    # 对比四种方法的差异
    print("\n5. 四种方法的对比：")
    print(
        f"  sqrt/标准 的绝对值比值: {np.mean(np.abs(deviations_sqrt)) / np.mean(np.abs(deviations)):.4f}"
    )
    print(
        f"  定值/标准 的绝对值比值: {np.mean(np.abs(deviations_const)) / np.mean(np.abs(deviations)):.4f}"
    )
    print(
        f"  sqrt/定值 的绝对值比值: {np.mean(np.abs(deviations_sqrt)) / np.mean(np.abs(deviations_const)):.4f}"
    )
    print()

    # 正确对齐权重统计
    correct_avg = np.mean(correct_weights)
    correct_std = np.std(correct_weights)
    print(f"正确对齐权重基准: {correct_avg:.4f} ± {correct_std:.4f}")
    print(f"  权重分布: min={min(correct_weights):.4f}, max={max(correct_weights):.4f}")
    high_weight_count = sum(1 for w in correct_weights if w > 0.8)
    print(
        f"  高权重(>0.8)比例: {high_weight_count}/{len(correct_weights)} ({high_weight_count/len(correct_weights)*100:.1f}%)"
    )
    print()

    # 创建图表：对比四种偏差计算方法（2x2布局）
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 图1：标准相对偏差 vs 平均标点数
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

    # 添加趋势线
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

    # 图2：根号分母偏差 vs 平均标点数
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

    # 添加趋势线
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

    # 图3：定值分母偏差 vs 平均标点数
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

    # 添加趋势线
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

    # 图4：绝对偏差 vs 平均标点数
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

    # 添加趋势线
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

    # 保存图表
    plot_path = Path(__file__).parent / "punctuation_error_analysis.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"统计图已保存到: {plot_path}")
    plt.close()

    # 错位模式分析
    print("错位模式分析:")
    higher_count = 0
    total_comparisons = 0

    for name, weights in misalignment_weights.items():
        avg_weight = np.mean(weights)
        std_weight = np.std(weights)

        impact = correct_avg - avg_weight
        impact_pct = (impact / correct_avg * 100) if correct_avg > 0 else 0
        status = "↓" if impact > 0 else "↑"

        print(
            f"  {name:<25} - 平均: {avg_weight:.4f} ± {std_weight:.4f} ({status}{abs(impact):.4f}, {status}{abs(impact_pct):.1f}%)"
        )

        # 统计得分更高的次数
        for i, weight in enumerate(weights):
            if weight > correct_weights[i]:
                higher_count += 1
            total_comparisons += 1

    print()
    if total_comparisons > 0:
        higher_percentage = (higher_count / total_comparisons) * 100
        print(
            f"统计结果: {higher_count}/{total_comparisons} ({higher_percentage:.1f}%) 错位权重高于正确对齐"
        )

        if higher_count == 0:
            print("✅ 所有错位对齐都正确降低了标点相似度权重")
        else:
            print(f"⚠️  {higher_count} 次错位权重意外高于正确对齐")

    return higher_count == 0


def main():
    """主函数"""
    print("开始测试标点相似度权重...")
    print()

    # 加载文本
    en_lines, zh_lines = load_sample_texts()
    if en_lines is None or zh_lines is None:
        return 1

    # 计算标点统计和权重
    en_punct_counts, zh_punct_counts, correct_weights, misalignment_weights = (
        calculate_punctuation_stats_and_weights(en_lines, zh_lines)
    )

    # 分析结果
    success = analyze_results(
        en_punct_counts, zh_punct_counts, correct_weights, misalignment_weights
    )

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
