#!/usr/bin/env python3
"""
Quick Text Similarity Tool

快速计算两段文本的相似度。支持两种编码方法：
- 整段编码：快速、简洁
- 句子编码：精确、稳定

Usage:
    python tools/similarity.py  # 使用默认示例文本
    python tools/similarity.py "Source text" "Target text"
    python tools/similarity.py --method sentence "Source" "Target"
    python tools/similarity.py --compare "Source" "Target"  # 对比两种方法
"""

import sys
import argparse
import numpy as np
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bilingual_aligner.core.processor import get_text_processor


class SimilarityTool:
    """文本相似度快速工具"""

    def __init__(self):
        self.processor = get_text_processor()

    def _get_punct_weight(self, text1, text2):
        """计算标点权重"""
        from bilingual_aligner.core.punctuation import (
            calculate_punctuation_similarity,
        )

        return calculate_punctuation_similarity(text1, text2)

    def similarity_paragraph(self, text1, text2):
        """整段编码相似度"""
        score = self.processor.calculate_similarity(text1, text2)
        weight = self._get_punct_weight(text1, text2)
        return max(0, score * weight)

    def similarity_sentence(self, text1, text2):
        """句子编码相似度"""
        emb1 = self.processor.get_normalized_embedding_by_sentences(
            text1, method="mean"
        )
        emb2 = self.processor.get_normalized_embedding_by_sentences(
            text2, method="mean"
        )
        score = float(np.dot(emb1, emb2))
        weight = self._get_punct_weight(text1, text2)
        return max(0, score * weight)

    def print_result(self, method, similarity):
        """打印结果"""
        print(f"\n{'='*50}")
        print(f"方法: {method}")
        print(f"相似度: {similarity:.6f}")
        print(f"{'='*50}\n")

    def run_quick(self, text1, text2, method="paragraph"):
        """快速相似度计算"""
        if method == "paragraph":
            similarity = self.similarity_paragraph(text1, text2)
            self.print_result("整段编码", similarity)
        elif method == "sentence":
            similarity = self.similarity_sentence(text1, text2)
            self.print_result("句子编码", similarity)

    def run_comparison(self, text1, text2):
        """对比两种方法"""
        para_sim = self.similarity_paragraph(text1, text2)
        sent_sim = self.similarity_sentence(text1, text2)

        print(f"\n{'='*60}")
        print("相似度对比")
        print(f"{'='*60}")
        print(f"文本1长度: {len(text1)} 字符")
        print(f"文本2长度: {len(text2)} 字符")
        print(f"\n{'方法':<20} {'相似度':<15} {'差异':<15}")
        print("-" * 60)
        print(f"{'整段编码':<20} {para_sim:<15.6f}")
        print(f"{'句子编码':<20} {sent_sim:<15.6f}")
        print(f"{'差异':<20} {abs(para_sim - sent_sim):<15.6f}")

        if sent_sim > para_sim:
            improvement = (sent_sim - para_sim) / para_sim * 100
            print(f"\n✓ 句子编码提升 {improvement:.2f}%")
        elif para_sim > sent_sim:
            improvement = (para_sim - sent_sim) / sent_sim * 100
            print(f"\n✓ 整段编码提升 {improvement:.2f}%")
        else:
            print(f"\n= 两种方法结果相同")

        print(f"{'='*60}\n")


def main():
    """主程序"""
    parser = argparse.ArgumentParser(
        description="快速计算文本相似度",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用整段编码（默认）
  python tools/similarity.py "Hello world" "Hello there"
  
  # 使用句子编码
  python tools/similarity.py --method sentence "Text 1" "Text 2"
  
  # 对比两种方法
  python tools/similarity.py --compare "Chinese text" "English text"
        """,
    )

    parser.add_argument("text1", nargs="?", help="第一段文本（可选，默认使用示例）")
    parser.add_argument("text2", nargs="?", help="第二段文本（可选，默认使用示例）")
    parser.add_argument(
        "--method",
        choices=["paragraph", "sentence"],
        default="paragraph",
        help="编码方法（默认: paragraph）",
    )
    parser.add_argument("--compare", action="store_true", help="对比两种编码方法")

    try:
        args = parser.parse_args()

        # 默认示例文本
        default_text1 = "Hello world! This is a test sentence."
        default_text2 = "Hello there! This is another test sentence."

        # 如果没有提供文本，使用默认
        if not args.text1:
            args.text1 = default_text1
        if not args.text2:
            args.text2 = default_text2

        tool = SimilarityTool()

        if args.compare:
            tool.run_comparison(args.text1, args.text2)
        else:
            tool.run_quick(args.text1, args.text2, args.method)

        return 0

    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
