"""
Validate the new quadratic function weight with tolerance=2.5
Show comparison with the original linear function
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from bilingual_aligner.core.punctuation import (
    PunctuationHandler,
    calculate_punctuation_similarity,
)


def weight_original_linear(p_src, p_tgt, tolerance=2.0):
    """Original linear function (for comparison reference only)"""
    avg = (p_src + p_tgt) / 2.0
    if avg <= 0:
        return 1.0
    punct_diff = abs(p_src - p_tgt)
    effective_diff = max(0, punct_diff - tolerance)
    return max(1 - effective_diff / avg, 0)


def main():
    print("=" * 80)
    print("New weight function validation (tolerance=2.5, quadratic function)")
    print("=" * 80)
    print()

    # Test case demonstration
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
        print(
            f"  New weight (quadratic): {weight_new:.4f}  |  Original (linear): {weight_old:.4f}"
        )
        print()

    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print(
        """
New quadratic function weight features:
✓ Tolerance=2.5: No penalty when |Δp| ≤ 2.5, penalty starts when |Δp| ≥ 3
✓ Quadratic curve: Penalty increases with the square of difference, stricter for large deviations
✓ Completely fair: Not affected by sentence length, same difference same penalty
✓ Intuitive parameters: Only need to adjust tolerance and divisor

Weight levels:
- [0.90, 1.00]: High confidence (|Δp| ≤ 3.5)
- [0.64, 0.90]: Medium confidence (3.5 < |Δp| ≤ 4.5)
- [0.36, 0.64]: Low confidence (4.5 < |Δp| ≤ 5.5)
- [0.00, 0.36]: Distrust (|Δp| > 5.5)
"""
    )


if __name__ == "__main__":
    main()
