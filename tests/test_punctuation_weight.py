"""
验证新的容差=2.5的二次函数权重
展示与原有线性函数的对比
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from bilingual_aligner.core.punctuation import (
    PunctuationHandler,
    calculate_punctuation_similarity,
)


def weight_original_linear(p_src, p_tgt, tolerance=2.0):
    """原有线性函数（仅用于对比参考）"""
    avg = (p_src + p_tgt) / 2.0
    if avg <= 0:
        return 1.0
    punct_diff = abs(p_src - p_tgt)
    effective_diff = max(0, punct_diff - tolerance)
    return max(1 - effective_diff / avg, 0)


def main():
    print("=" * 80)
    print("新权重函数验证 (容差=2.5, 二次函数)")
    print("=" * 80)
    print()

    # 测试用例展示
    test_cases = [
        ("perfect match", "Hello, world!", "Hello, world!"),
        ("within tolerance (Δp=1)", "Hello, world!", "Hello world"),
        ("within tolerance (Δp=2)", "Hello, world! How are you?", "Hello world"),
        ("at boundary (Δp=2.5)", "Test sentence.", "Test"),
        ("just beyond (Δp=3)", "Hello, world! How?", "Hello world"),
        ("larger diff (Δp=4)", "Test.?!-", "Test."),
        ("larger diff (Δp=5)", "A.?!-,", "A."),
    ]

    for label, src, tgt in test_cases:
        src_count = PunctuationHandler.count_punctuation_line(src)
        tgt_count = PunctuationHandler.count_punctuation_line(tgt)
        diff = abs(src_count - tgt_count)
        weight_new = calculate_punctuation_similarity(src, tgt)
        weight_old = weight_original_linear(src_count, tgt_count, tolerance=2.0)

        print(f"{label:30s}")
        print(
            f"  Src: {src_count} punctuation, Tgt: {tgt_count} punctuation, |Δp|={diff}"
        )
        print(f"  新权重 (二次): {weight_new:.4f}  |  原有 (线性): {weight_old:.4f}")
        print()

    print("=" * 80)
    print("总结")
    print("=" * 80)
    print(
        """
新的二次函数权重特点：
✓ 容差=2.5: |Δp| ≤ 2.5时无惩罚，|Δp| ≥ 3时开始扣分
✓ 二次曲线: 惩罚力度随差异平方递增，对大偏差更严格
✓ 完全公平: 不受句长影响，同样差异同样惩罚
✓ 参数直观: 只需调整容差(tolerance)和除数(divisor)

权重等级：
- [0.90, 1.00]: 高置信 (|Δp| ≤ 3.5)
- [0.64, 0.90]: 中置信 (3.5 < |Δp| ≤ 4.5)
- [0.36, 0.64]: 低置信 (4.5 < |Δp| ≤ 5.5)
- [0.00, 0.36]: 不信任 (|Δp| > 5.5)
"""
    )


if __name__ == "__main__":
    main()
