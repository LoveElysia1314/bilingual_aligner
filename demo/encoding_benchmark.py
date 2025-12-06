#!/usr/bin/env python3
"""
编码方法性能基准测试 - 精简版
用于验证句子级编码 vs 整段编码的性能优势

核心对比：
  - 准确性：相似度得分（越高越好）
  - 稳定性：结果标准差（越低越好）
  - 速度：处理时间（越快越好）
  - 鲁棒性：句子数不匹配时的表现
"""

import os
import sys
import time
import numpy as np
from pathlib import Path
from typing import Tuple, List, Callable, Dict

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bilingual_aligner.core.processor import get_text_processor
from bilingual_aligner.core.punctuation import (
    calculate_punctuation_similarity,
    PunctuationHandler,
)


def clear_caches(processor):
    """统一清除缓存，避免重复代码"""
    try:
        processor.sentence_embedding_cache.clear()
        processor.similarity_cache.clear()
    except Exception:
        pass


class EncodingBenchmark:
    """精简的编码方法对比测试"""

    def __init__(self, source_file: str, target_file: str):
        self.source_file = source_file
        self.target_file = target_file
        self.processor = get_text_processor()

        # Warm-up and preload model
        try:
            self.processor._load_model()
            _ = self.processor.get_normalized_embedding_by_sentences(
                "Hello world.", method="mean"
            )
        except Exception:
            pass  # fallback to on-the-fly loading

    def load_files(self) -> Tuple[List[str], List[str]]:
        with open(self.source_file, "r", encoding="utf-8") as f:
            source_lines = [line.rstrip("\n\r") for line in f if line.strip()]
        with open(self.target_file, "r", encoding="utf-8") as f:
            target_lines = [line.rstrip("\n\r") for line in f if line.strip()]

        min_len = min(len(source_lines), len(target_lines))
        source_lines, target_lines = source_lines[:min_len], target_lines[:min_len]

        return source_lines, target_lines

    def is_multi_sentence(self, text: str) -> bool:
        return len(self.processor.split_sentences(text)) > 1

    def calculate_punctuation_weight(self, src: str, tgt: str) -> float:
        return calculate_punctuation_similarity(src, tgt)

    # === Encoding Strategies ===

    def _run_with_timing(self, func: Callable[[], float]) -> Tuple[float, float]:
        start = time.perf_counter()
        clear_caches(self.processor)
        score = func()
        elapsed = time.perf_counter() - start
        return max(0.0, score), elapsed

    def encode_paragraph(self, src: str, tgt: str) -> Tuple[float, float]:
        def _compute():
            score = self.processor.calculate_similarity(src, tgt, method="paragraph")
            return score * self.calculate_punctuation_weight(src, tgt)

        return self._run_with_timing(_compute)

    def encode_sentences(self, src: str, tgt: str) -> Tuple[float, float]:
        def _compute():
            src_emb = self.processor.get_normalized_embedding_by_sentences(
                src, method="mean"
            )
            tgt_emb = self.processor.get_normalized_embedding_by_sentences(
                tgt, method="mean"
            )
            score = float(np.dot(src_emb, tgt_emb))
            return score * self.calculate_punctuation_weight(src, tgt)

        return self._run_with_timing(_compute)

    # === Benchmark Execution ===

    def run_benchmark(self) -> bool:
        print("\n" + "=" * 70)
        print("📊 编码方法性能基准测试")
        print("=" * 70)

        print("\n📁 加载数据...")
        source_lines, target_lines = self.load_files()

        multi_sent_pairs = [
            (s, t)
            for s, t in zip(source_lines, target_lines)
            if self.is_multi_sentence(s) and self.is_multi_sentence(t)
        ]

        print(f"   总行数: {len(source_lines)}")
        print(f"   多句子行对: {len(multi_sent_pairs)} (用于对比测试)")

        if not multi_sent_pairs:
            print("❌ 没有足够的多句子行对用于测试")
            return False

        # Define methods
        methods: Dict[str, Callable[[str, str], Tuple[float, float]]] = {
            "paragraph": self.encode_paragraph,
            "sentence": self.encode_sentences,
        }

        # Run benchmark
        print("\n⚙️  运行对比测试...")
        all_scores: Dict[str, List[float]] = {name: [] for name in methods}
        all_times: Dict[str, List[float]] = {name: [] for name in methods}

        for src, tgt in multi_sent_pairs:
            for name, func in methods.items():
                try:
                    score, elapsed = func(src, tgt)
                    all_scores[name].append(score)
                    all_times[name].append(elapsed)
                except Exception as e:
                    print(f"   ⚠️  {name} 错误: {e}")
                    continue

        # Convert to numpy arrays
        scores = {k: np.array(v) for k, v in all_scores.items() if v}
        times = {k: np.array(v) for k, v in all_times.items() if v}

        # Print main results
        self._print_results(scores, times)

        # Additional tests
        self._test_mismatch(source_lines, target_lines)
        self._test_misalignment(source_lines, target_lines, multi_sent_pairs)

        return True

    def _print_results(
        self, scores: Dict[str, np.ndarray], times: Dict[str, np.ndarray]
    ):
        print("\n" + "-" * 70)
        print("【测试 1】正确对齐性能对比")
        print("-" * 70)

        # Accuracy
        means = {k: np.mean(v) for k, v in scores.items()}
        print("\n📈 准确性 (相似度得分)")
        for k, v in means.items():
            print(f"   {k}: {v:.4f}")

        para_mean = means.get("paragraph", 0.0)
        sent_mean = means.get("sentence", 0.0)
        accuracy_gain = (sent_mean - para_mean) / (para_mean or 1.0) * 100
        if accuracy_gain > 0:
            print(f"   ✅ 句子编码相对于段落提升 {accuracy_gain:.2f}%")
        else:
            print(f"   ⚠️ 段落编码相对于句子编码提升 {-accuracy_gain:.2f}%")

        # Stability
        stds = {k: np.std(v) for k, v in scores.items()}
        print("\n🎯 稳定性 (标准差)")
        for k, v in stds.items():
            print(f"   {k}: {v:.4f}")

        para_std = stds.get("paragraph", float("inf"))
        sent_std = stds.get("sentence", float("inf"))
        stability_gain = (para_std - sent_std) / (para_std or 1.0) * 100
        if stability_gain > 0:
            print(f"   ✅ 句子编码稳定性提升 {stability_gain:.2f}%")
        else:
            print(f"   ⚠️  段落编码更稳定 {-stability_gain:.2f}%")

        # Speed
        totals = {k: np.sum(v) for k, v in times.items()}
        per_item_ms = {
            k: (tot / len(scores[k]) * 1000 if len(scores[k]) > 0 else 0.0)
            for k, tot in totals.items()
        }
        print("\n⚡ 处理速度")
        for k in totals:
            print(f"   {k}: {totals[k]:.3f}s ({per_item_ms[k]:.2f}ms/行)")

        fastest = min(totals.items(), key=lambda x: x[1])
        if fastest[0] == "paragraph":
            print(f"   ✅ 段落编码最快 ({fastest[1]:.3f}s)")
        else:
            print(f"   ✅ 句子编码最快 ({fastest[1]:.3f}s)")

        # Summary comparison
        print("\n" + "-" * 70)
        print("综合比较（paragraph / sentence）")
        print("-" * 70)
        print(f"平均相似度: " + ", ".join(f"{k}={v:.4f}" for k, v in means.items()))
        print(f"标准差: " + ", ".join(f"{k}={stds[k]:.4f}" for k in means))
        print(f"总耗时: " + ", ".join(f"{k}={totals[k]:.3f}s" for k in totals))

        # Scoring by metric wins
        para_score = 0
        sent_score = 0
        # Accuracy
        if sent_mean > para_mean:
            sent_score += 1
        elif para_mean > sent_mean:
            para_score += 1
        # Stability
        if sent_std < para_std:
            sent_score += 1
        elif para_std < sent_std:
            para_score += 1
        # Speed
        if totals.get("sentence", float("inf")) < totals.get("paragraph", float("inf")):
            sent_score += 1
        elif totals.get("paragraph", float("inf")) < totals.get(
            "sentence", float("inf")
        ):
            para_score += 1

        print(f"得分: paragraph={para_score}, sentence={sent_score}")

    def _test_mismatch(self, source_lines: List[str], target_lines: List[str]):
        print("\n" + "-" * 70)
        print("【测试 2】句子数不匹配鲁棒性")
        print("-" * 70)

        mismatch_pairs = []
        for src, tgt in zip(source_lines, target_lines):
            if not (self.is_multi_sentence(src) and self.is_multi_sentence(tgt)):
                continue
            src_count = len(self.processor.split_sentences(src))
            tgt_count = len(self.processor.split_sentences(tgt))
            if src_count != tgt_count:
                mismatch_pairs.append((src, tgt))

        if not mismatch_pairs:
            print("   ℹ️  未发现句子数不匹配的情况")
            return

        print(f"\n   发现 {len(mismatch_pairs)} 对句子数不匹配的行")
        methods = {
            "paragraph": self.encode_paragraph,
            "sentence": self.encode_sentences,
        }

        results = {name: [] for name in methods}
        for src, tgt in mismatch_pairs:
            for name, func in methods.items():
                try:
                    score, _ = func(src, tgt)
                    results[name].append(score)
                except:
                    continue

        para_scores = np.array(results["paragraph"])
        print(f"\n   段落编码平均相似度: {np.mean(para_scores):.4f}")
        if results["sentence"]:
            sent_scores = np.array(results["sentence"])
            better = np.sum(sent_scores > para_scores[: len(sent_scores)])
            win_rate = better / len(sent_scores) * 100
            print(
                f"   句子编码平均相似度: {np.mean(sent_scores):.4f}（更优 {better}/{len(sent_scores)} ({win_rate:.0f}%)）"
            )

    def _test_misalignment(
        self,
        source_lines: List[str],
        target_lines: List[str],
        multi_sent_pairs: List[Tuple[str, str]],
    ):
        print("\n" + "-" * 70)
        print("【测试 3】误对齐检测能力")
        print("-" * 70)

        sample_size = min(20, len(multi_sent_pairs))
        if sample_size == 0:
            return

        methods = {
            "paragraph": self.encode_paragraph,
            "sentence": self.encode_sentences,
        }

        correct = {k: [] for k in methods}
        incorrect = {k: [] for k in methods}

        for i in range(sample_size):
            src, tgt = multi_sent_pairs[i]
            # Correct
            for name, func in methods.items():
                try:
                    score, _ = func(src, tgt)
                    correct[name].append(score)
                except:
                    pass
            # Incorrect (shifted)
            src_wrong = source_lines[(i + 1) % len(source_lines)]
            for name, func in methods.items():
                try:
                    score, _ = func(src_wrong, tgt)
                    incorrect[name].append(score)
                except:
                    pass

        print("\n   正确对齐 vs 错误对齐（平均相似度）")
        para_gap = 0.0
        sent_gap = 0.0
        for name in methods:
            if correct[name] and incorrect[name]:
                c_mean = np.mean(correct[name])
                i_mean = np.mean(incorrect[name])
                gap = c_mean - i_mean
                if name == "paragraph":
                    para_gap = gap
                elif name == "sentence":
                    sent_gap = gap
                print(f"\n   {name}:")
                print(f"      正确: {c_mean:.4f}")
                print(f"      错误: {i_mean:.4f}")
                print(f"      区分度: {gap:.4f}")

        if para_gap > 0 and sent_gap > para_gap:
            gain = (sent_gap / para_gap - 1) * 100
            print(f"\n   ✅ 句子编码在区分能力上优于段落 (增益 {gain:.1f}%)")
        else:
            print(f"\n   ⚠️ 段落编码在区分能力上更强或持平")


def main() -> int:
    demo_dir = Path(__file__).parent
    source_file = demo_dir / "sample_zh.md"
    target_file = demo_dir / "output" / "sample_en_repaired.txt"

    if not source_file.exists():
        print(f"❌ 源文件不存在: {source_file}")
        return 1
    if not target_file.exists():
        print(f"❌ 目标文件不存在: {target_file}")
        return 1

    try:
        benchmark = EncodingBenchmark(str(source_file), str(target_file))
        success = benchmark.run_benchmark()
        if success:
            print("\n" + "=" * 70)
            print("✅ 测试完成")
            print("=" * 70 + "\n")
            return 0
        else:
            return 1
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
