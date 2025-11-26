#!/usr/bin/env python3
"""
Encoding Method Comparison Tool

Compares two encoding strategies:
1. Whole-paragraph encoding (baseline)
2. Sentence-by-sentence encoding with aggregation

Tests accuracy, stability, speed, and robustness.

Usage:
    python tools/encoding_comparison.py [--method paragraph|sentence] [--benchmark] [--model MODEL_NAME]

Available models:
    - Alibaba-NLP/gte-multilingual-base (default)
    - sentence-transformers/paraphrase-multilingual-mpnet-base-v2
    - Any other sentence-transformers model
"""

import sys
import time
import argparse
import numpy as np
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bilingual_aligner.core.processor import get_text_processor
from bilingual_aligner.core.punctuation import calculate_punctuation_similarity


class EncodingComparator:
    """Tool for comparing encoding methods"""

    def __init__(self, model_name="Alibaba-NLP/gte-multilingual-base"):
        self.model_name = model_name
        self.processor = get_text_processor(model_name=model_name)

    def test_basic_functionality(self, method="paragraph"):
        """Test basic encoding functionality"""
        print(f"Testing {method} encoding...")

        test_cases = [
            ("English text", "The quick brown fox jumps over the lazy dog."),
            ("Chinese text", "快速的棕色狐狸跳过懒狗。"),
            ("Mixed content", "Project: 双语对齐 (Bilingual Alignment)"),
        ]

        all_valid = True
        for name, text in test_cases:
            try:
                if method == "paragraph":
                    emb = self.processor.get_normalized_embedding(text)
                else:
                    emb = self.processor.get_normalized_embedding_by_sentences(
                        text, method="mean"
                    )
                is_valid = emb is not None and len(emb) > 0
                print(
                    f"  {name:<20} - Embedding dim: {len(emb) if is_valid else 'N/A'}"
                )
                if not is_valid:
                    all_valid = False
            except Exception as e:
                print(f"  {name:<20} - Error: {e}")
                all_valid = False

        print(f"✅ {method.capitalize()} encoding: {'PASS' if all_valid else 'FAIL'}")
        return all_valid

    def test_similarity_computation(self):
        """Test similarity computation consistency"""
        print("Testing similarity computation...")

        text_pairs = [
            ("Identical text", "Hello world", "Hello world"),
            ("Similar text", "The quick fox", "The quick dog"),
            ("Different text", "Hello", "Goodbye"),
        ]

        all_valid = True
        for name, src, tgt in text_pairs:
            try:
                # Paragraph method
                para_sim = self.processor.calculate_similarity(src, tgt)

                # Sentence method
                src_emb = self.processor.get_normalized_embedding_by_sentences(
                    src, method="mean"
                )
                tgt_emb = self.processor.get_normalized_embedding_by_sentences(
                    tgt, method="mean"
                )
                sent_sim = float(np.dot(src_emb, tgt_emb))

                is_valid = 0 <= para_sim <= 1 and 0 <= sent_sim <= 1
                print(
                    f"  {name:<20} - Paragraph: {para_sim:.4f}, Sentence: {sent_sim:.4f}"
                )

                if not is_valid:
                    all_valid = False
            except Exception as e:
                print(f"  {name:<20} - Error: {e}")
                all_valid = False

        print(f"✅ Similarity computation: {'PASS' if all_valid else 'FAIL'}")
        return all_valid

    def test_misalignment_impact(self):
        """Test the impact of misalignment on similarity scores"""
        print("Testing misalignment impact on similarity scores...")

        # Load sample texts
        try:
            with open(
                Path(__file__).parent.parent / "demo" / "sample_en.md",
                "r",
                encoding="utf-8",
            ) as f:
                en_lines = [line.strip() for line in f.readlines() if line.strip()]

            with open(
                Path(__file__).parent.parent / "demo" / "sample_zh.md",
                "r",
                encoding="utf-8",
            ) as f:
                zh_lines = [line.strip() for line in f.readlines() if line.strip()]
        except FileNotFoundError:
            print("  Sample files not found, skipping misalignment test")
            return False

        # Use first 20 lines for testing
        en_lines = en_lines[:20]
        zh_lines = zh_lines[:20]
        n_lines = min(len(en_lines), len(zh_lines))

        print(f"  Testing with {n_lines} line pairs")

        correct_similarities = []
        misalignment_similarities = []

        # Test correct alignments
        for i in range(n_lines):
            try:
                sim = self.processor.calculate_similarity(en_lines[i], zh_lines[i])
                correct_similarities.append(sim)
            except Exception as e:
                print(f"  Error calculating similarity for line {i}: {e}")
                continue

        # Test misalignment patterns
        misalignment_types = [
            (
                "EN[i] vs ZH[i+1]+ZH[i+2]",
                lambda i: (
                    en_lines[i],
                    zh_lines[(i + 1) % n_lines] + " " + zh_lines[(i + 2) % n_lines],
                ),
            ),
            (
                "EN[i] vs ZH[i-1]+ZH[i-2]",
                lambda i: (
                    en_lines[i],
                    zh_lines[i - 1 if i > 0 else n_lines - 1]
                    + " "
                    + zh_lines[i - 2 if i > 1 else n_lines - 2],
                ),
            ),
            (
                "ZH[i] vs EN[i+1]+EN[i+2]",
                lambda i: (
                    zh_lines[i],
                    en_lines[(i + 1) % n_lines] + " " + en_lines[(i + 2) % n_lines],
                ),
            ),
            (
                "ZH[i] vs EN[i-1]+EN[i-2]",
                lambda i: (
                    zh_lines[i],
                    en_lines[i - 1 if i > 0 else n_lines - 1]
                    + " "
                    + en_lines[i - 2 if i > 1 else n_lines - 2],
                ),
            ),
        ]

        for misalignment_name, get_pair_func in misalignment_types:
            similarities = []
            for i in range(n_lines):
                try:
                    src, tgt = get_pair_func(i)
                    sim = self.processor.calculate_similarity(src, tgt)
                    similarities.append(sim)
                except Exception as e:
                    continue

            avg_sim = sum(similarities) / len(similarities) if similarities else 0
            misalignment_similarities.append((misalignment_name, avg_sim))

        # Calculate overall statistics
        correct_avg = (
            sum(correct_similarities) / len(correct_similarities)
            if correct_similarities
            else 0
        )
        correct_std = (
            (
                sum((x - correct_avg) ** 2 for x in correct_similarities)
                / len(correct_similarities)
            )
            ** 0.5
            if correct_similarities
            else 0
        )
        print(f"  1:1 alignment baseline: {correct_avg:.4f} ± {correct_std:.4f}")

        print("\n  Misalignment patterns:")
        higher_count = 0
        total_comparisons = 0

        for name, avg_sim in misalignment_similarities:
            # Calculate std for this misalignment pattern
            similarities = []
            for i in range(n_lines):
                try:
                    misalignment_name, get_pair_func = [
                        (n, f) for n, f in misalignment_types if n == name
                    ][0]
                    src, tgt = get_pair_func(i)
                    sim = self.processor.calculate_similarity(src, tgt)
                    similarities.append(sim)
                    # Count how many times misalignment score > correct score
                    if sim > correct_similarities[i]:
                        higher_count += 1
                    total_comparisons += 1
                except Exception:
                    continue

            std_sim = (
                (sum((x - avg_sim) ** 2 for x in similarities) / len(similarities))
                ** 0.5
                if similarities
                else 0
            )
            impact = correct_avg - avg_sim
            impact_pct = (impact / correct_avg * 100) if correct_avg > 0 else 0
            status = "↓" if impact > 0 else "↑"
            print(
                f"  {name:<25} - Avg: {avg_sim:.4f} ± {std_sim:.4f} ({status}{abs(impact):.4f}, {status}{abs(impact_pct):.1f}%)"
            )

        if total_comparisons > 0:
            higher_percentage = (higher_count / total_comparisons) * 100
            print(
                f"\n  Summary: {higher_count}/{total_comparisons} ({higher_percentage:.1f}%) misalignment scores were higher than 1:1 baseline"
            )

        # Check if misalignments generally reduce scores
        misalignments_reduced = all(
            avg_sim < correct_avg for _, avg_sim in misalignment_similarities
        )

        print(
            f"\n✅ Misalignment impact test: {'PASS' if misalignments_reduced else 'FAIL'}"
        )
        print("  (All misalignments should reduce similarity scores)")

        return misalignments_reduced

    def run_tests(
        self,
        method="paragraph",
        benchmark=False,
        model_name="Alibaba-NLP/gte-multilingual-base",
    ):
        """Run selected tests"""
        print("Encoding Method Comparison Tool")
        print(f"Model: {model_name}")
        print("=" * 50)

        results = []
        results.append(self.test_basic_functionality(method))
        results.append(self.test_similarity_computation())
        results.append(self.test_misalignment_impact())
        if benchmark:
            results.append(self.benchmark_performance())

        print("\n" + "=" * 50)
        passed = sum(results)
        total = len(results)
        print(f"Tests passed: {passed}/{total}")

        if passed == total:
            print("🎉 All tests passed!")
            return True
        else:
            print(f"⚠️  {total - passed} test(s) failed")
            return False


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Compare encoding methods")
    parser.add_argument(
        "--method",
        choices=["paragraph", "sentence"],
        default="paragraph",
        help="Encoding method to test (default: paragraph)",
    )
    parser.add_argument(
        "--benchmark", action="store_true", help="Include performance benchmarking"
    )
    parser.add_argument(
        "--model",
        default="Alibaba-NLP/gte-multilingual-base",
        help="Model name to use (default: Alibaba-NLP/gte-multilingual-base)",
    )

    args = parser.parse_args()

    try:
        comparator = EncodingComparator(model_name=args.model)
        success = comparator.run_tests(
            method=args.method, benchmark=args.benchmark, model_name=args.model
        )
        return 0 if success else 1
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
