#!/usr/bin/env python3
"""
Encoding Method Analyzer

快速对比两种编码方法在特定文本上的表现。
用于理解两种方法的差异和优缺点。

Usage:
    python tools/encoding_analyzer.py  # 使用默认示例文本
    python tools/encoding_analyzer.py "Text" "Translation"
    python tools/encoding_analyzer.py --file source.txt target.txt
    python tools/encoding_analyzer.py --detailed "Text 1" "Text 2"
"""

import sys
import argparse
import time
import numpy as np
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bilingual_aligner.core.processor import get_text_processor


class EncodingAnalyzer:
    """编码方法分析工具"""

    def __init__(self):
        self.processor = get_text_processor()

    def analyze_text(self, text):
        """分析单个文本"""
        sentences = self.processor.split_sentences(text)

        return {
            "text": text,
            "length": len(text),
            "sentence_count": len(sentences),
            "avg_sentence_length": len(text) / len(sentences) if sentences else 0,
            "sentences": sentences,
        }

    def benchmark_encoding(self, text, method="paragraph"):
        """对编码方法计时"""
        iterations = 3
        times = []

        for _ in range(iterations):
            start = time.perf_counter()

            if method == "paragraph":
                _ = self.processor.get_normalized_embedding(text)
            else:
                _ = self.processor.get_normalized_embedding_by_sentences(
                    text, method="mean"
                )

            times.append(time.perf_counter() - start)

        return {
            "method": method,
            "avg_time": np.mean(times),
            "min_time": np.min(times),
            "max_time": np.max(times),
            "std_time": np.std(times),
        }

    def compute_similarity(self, text1, text2, method="paragraph"):
        """计算相似度"""
        if method == "paragraph":
            emb1 = self.processor.get_normalized_embedding(text1)
            emb2 = self.processor.get_normalized_embedding(text2)
        else:
            emb1 = self.processor.get_normalized_embedding_by_sentences(
                text1, method="mean"
            )
            emb2 = self.processor.get_normalized_embedding_by_sentences(
                text2, method="mean"
            )

        return float(np.dot(emb1, emb2))

    def print_text_analysis(self, text_info):
        """打印文本分析"""
        print(f"\n{'='*70}")
        print("文本分析")
        print(f"{'='*70}")

        print(f"\n长度:        {text_info['length']} 字符")
        print(f"句子数:      {text_info['sentence_count']}")
        print(f"平均句长:    {text_info['avg_sentence_length']:.1f} 字符/句")

        if text_info["sentence_count"] > 0:
            print(f"\n句子详情:")
            for i, sent in enumerate(text_info["sentences"][:5], 1):
                display_sent = sent[:50] + "..." if len(sent) > 50 else sent
                print(f"  {i}. {display_sent}")
            if len(text_info["sentences"]) > 5:
                print(f"  ... 共 {text_info['sentence_count']} 句")

    def print_benchmark(self, results):
        """打印性能基准"""
        print(f"\n{'='*70}")
        print("编码性能基准")
        print(f"{'='*70}\n")

        print(f"{'方法':<20} {'平均时间':<15} {'最小':<12} {'最大':<12} {'标准差':<12}")
        print("-" * 70)

        for result in results:
            print(
                f"{result['method']:<20} "
                f"{result['avg_time']*1000:.2f}ms{'':<8} "
                f"{result['min_time']*1000:.2f}ms{'':<5} "
                f"{result['max_time']*1000:.2f}ms{'':<5} "
                f"{result['std_time']*1000:.2f}ms"
            )

        # 计算速度比
        times = [r["avg_time"] for r in results]
        if len(times) == 2:
            speedup = times[0] / times[1]
            faster = "整段编码" if speedup > 1 else "句子编码"
            print(f"\n✓ {faster} 快 {abs(speedup):.2f}x")

    def print_similarity_comparison(self, text1, text2):
        """打印相似度对比"""
        para_sim = self.compute_similarity(text1, text2, "paragraph")
        sent_sim = self.compute_similarity(text1, text2, "sentence")

        print(f"\n{'='*70}")
        print("相似度对比")
        print(f"{'='*70}\n")

        print(f"{'方法':<20} {'相似度':<15} {'排名':<10}")
        print("-" * 70)

        if para_sim > sent_sim:
            print(f"{'整段编码':<20} {para_sim:<15.6f} {'1 (更高)':<10}")
            print(f"{'句子编码':<20} {sent_sim:<15.6f} {'2':<10}")
            diff = para_sim - sent_sim
            improve = diff / sent_sim * 100
            print(f"\n整段编码领先 {diff:.6f} ({improve:.2f}%)")
        else:
            print(f"{'句子编码':<20} {sent_sim:<15.6f} {'1 (更高)':<10}")
            print(f"{'整段编码':<20} {para_sim:<15.6f} {'2':<10}")
            diff = sent_sim - para_sim
            improve = diff / para_sim * 100
            print(f"\n句子编码领先 {diff:.6f} ({improve:.2f}%)")

    def test_basic_functionality(self, method="paragraph"):
        """测试编码方法的基本功能"""
        print(f"测试 {method} 编码基本功能...")

        test_cases = [
            ("英文文本", "The quick brown fox jumps over the lazy dog."),
            ("中文文本", "快速的棕色狐狸跳过懒狗。"),
            ("混合内容", "Project: 双语对齐 (Bilingual Alignment)"),
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
                print(f"  {name:<15} - 嵌入维度: {len(emb) if is_valid else 'N/A'}")
                if not is_valid:
                    all_valid = False
            except Exception as e:
                print(f"  {name:<15} - 错误: {e}")
                all_valid = False

        status = "✅ 通过" if all_valid else "❌ 失败"
        print(f"{method.capitalize()} 编码: {status}")
        return all_valid

    def test_similarity_consistency(self):
        """测试相似度计算一致性"""
        print("测试相似度计算一致性...")

        text_pairs = [
            ("相同文本", "Hello world", "Hello world"),
            ("相似文本", "The quick fox", "The quick dog"),
            ("不同文本", "Hello", "Goodbye"),
        ]

        all_valid = True
        for name, src, tgt in text_pairs:
            try:
                # 段落方法
                para_sim = self.processor.calculate_similarity(src, tgt)

                # 句子方法
                src_emb = self.processor.get_normalized_embedding_by_sentences(
                    src, method="mean"
                )
                tgt_emb = self.processor.get_normalized_embedding_by_sentences(
                    tgt, method="mean"
                )
                sent_sim = float(np.dot(src_emb, tgt_emb))

                is_valid = 0 <= para_sim <= 1 and 0 <= sent_sim <= 1
                print(f"  {name:<15} - 段落: {para_sim:.4f}, 句子: {sent_sim:.4f}")

                if not is_valid:
                    all_valid = False
            except Exception as e:
                print(f"  {name:<15} - 错误: {e}")
                all_valid = False

        status = "✅ 通过" if all_valid else "❌ 失败"
        print(f"相似度计算: {status}")
        return all_valid

    def run_tests(self, method="paragraph"):
        """运行编码方法测试"""
        print("编码方法对比测试")
        print("=" * 50)

        results = []
        results.append(self.test_basic_functionality(method))
        results.append(self.test_similarity_consistency())

        print("\n" + "=" * 50)
        passed = sum(results)
        total = len(results)
        print(f"测试通过: {passed}/{total}")

        if passed == total:
            print("🎉 所有测试通过！")
            return True
        else:
            print(f"⚠️  {total - passed} 个测试失败")
            return False


def main():
    """主程序"""
    parser = argparse.ArgumentParser(
        description="分析编码方法的性能和特性",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 快速对比
  python tools/encoding_analyzer.py "Text 1" "Text 2"
  
  # 详细分析
  python tools/encoding_analyzer.py --detailed "Text 1" "Text 2"
  
  # 从文件读取
  python tools/encoding_analyzer.py --file source.txt target.txt
  
  # 运行编码方法测试
  python tools/encoding_analyzer.py --test
  python tools/encoding_analyzer.py --test --method sentence
        """,
    )

    parser.add_argument("text1", nargs="?", help="第一段文本（可选，默认使用示例）")
    parser.add_argument("text2", nargs="?", help="第二段文本（可选，默认使用示例）")
    parser.add_argument(
        "--file", action="store_true", help="从文件读取（text1 和 text2 为文件路径）"
    )
    parser.add_argument("--detailed", action="store_true", help="显示详细分析")
    parser.add_argument(
        "--test",
        action="store_true",
        help="运行编码方法测试（验证基本功能和相似度计算）",
    )
    parser.add_argument(
        "--method",
        choices=["paragraph", "sentence"],
        default="paragraph",
        help="测试时使用的编码方法（默认: paragraph）",
    )

    try:
        args = parser.parse_args()

        # 默认示例文本
        default_text1 = "Hello world! This is a test sentence."
        default_text2 = "Hello there! This is another test sentence."

        # 读取文本
        if args.file:
            try:
                with open(args.text1, "r", encoding="utf-8") as f:
                    text1 = f.read()
                if args.text2:
                    with open(args.text2, "r", encoding="utf-8") as f:
                        text2 = f.read()
                else:
                    print("❌ 文件模式需要两个文件路径")
                    return 1
            except Exception as e:
                print(f"❌ 无法读取文件: {e}")
                return 1
        else:
            text1 = args.text1 if args.text1 else default_text1
            text2 = args.text2 if args.text2 else default_text2

        analyzer = EncodingAnalyzer()

        if args.test:
            # 运行测试模式
            success = analyzer.run_tests(method=args.method)
            return 0 if success else 1
        elif args.detailed:
            analyzer.run_detailed_analysis(text1, text2)
        else:
            analyzer.print_similarity_comparison(text1, text2)

        return 0

    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
