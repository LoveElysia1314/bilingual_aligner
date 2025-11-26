#!/usr/bin/env python3
"""
Punctuation Analysis Tool

分析文本的标点符号特性，计算标点权重。
用于理解标点符号对文本相似度的影响。

Usage:
    python tools/punctuation_analyzer.py  # 使用默认示例文本
    python tools/punctuation_analyzer.py "Text with punctuation."
    python tools/punctuation_analyzer.py --compare "Text 1" "Text 2"
    python tools/punctuation_analyzer.py --file corpus.txt
"""

import sys
import argparse
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bilingual_aligner.core.punctuation import (
    PunctuationHandler,
    calculate_punctuation_similarity,
)


class PunctuationAnalyzer:
    """标点符号分析工具"""

    # 标点符号集合
    PUNCTUATION_CN = "。，；：？！…—·「」『』【】（）《》、"
    PUNCTUATION_EN = ".,;:?!-()[]{}\"'" "''"
    PUNCTUATION_ALL = PUNCTUATION_CN + PUNCTUATION_EN

    def __init__(self):
        self.avg_punct = 4.0  # 默认平均标点数

    def analyze_text(self, text):
        """分析单个文本的标点符号"""
        punct_count = PunctuationHandler.count_punctuation_line(text)

        # 统计各类标点
        cn_puncts = sum(1 for c in text if c in self.PUNCTUATION_CN)
        en_puncts = sum(1 for c in text if c in self.PUNCTUATION_EN)

        # 找到所有标点符号
        puncts_found = [c for c in text if c in self.PUNCTUATION_ALL]

        return {
            "text": text,
            "total_punct": punct_count,
            "chinese_punct": cn_puncts,
            "english_punct": en_puncts,
            "puncts_found": puncts_found,
            "length": len(text),
            "punct_ratio": punct_count / len(text) if len(text) > 0 else 0,
        }

    def calculate_weight(self, text1, text2):
        """计算两段文本间的标点权重"""
        weight = calculate_punctuation_similarity(text1, text2)

        return {
            "weight": weight,
            "weight_percent": weight * 100,
        }

    def print_analysis(self, analysis):
        """打印单个文本的分析"""
        print(f"\n{'='*60}")
        print("标点符号分析")
        print(f"{'='*60}\n")

        print(f"文本长度:     {analysis['length']} 字符")
        print(f"标点总数:     {analysis['total_punct']} 个")
        print(f"标点比例:     {analysis['punct_ratio']*100:.2f}%")

        print(f"\n标点分类:")
        print(f"  中文标点:   {analysis['chinese_punct']} 个")
        print(f"  英文标点:   {analysis['english_punct']} 个")

        if analysis["puncts_found"]:
            print(f"\n找到的标点符号: {''.join(set(analysis['puncts_found']))}")
        else:
            print(f"\n找到的标点符号: 无")

        # 显示完整文本的前100字符
        preview = analysis["text"][:100]
        if len(analysis["text"]) > 100:
            preview += "..."
        print(f"\n文本预览:    {preview}")

    def print_comparison(self, text1, text2):
        """打印两段文本的对比分析"""
        analysis1 = self.analyze_text(text1)
        analysis2 = self.analyze_text(text2)
        weight_info = self.calculate_weight(text1, text2)

        print(f"\n{'='*70}")
        print("标点符号对比分析")
        print(f"{'='*70}\n")

        # 标点统计对比表
        print(f"{'指标':<20} {'文本 1':<20} {'文本 2':<20} {'差异':<15}")
        print("-" * 70)

        punct_diff = abs(analysis1["total_punct"] - analysis2["total_punct"])
        print(
            f"{'标点总数':<20} {analysis1['total_punct']:<20} {analysis2['total_punct']:<20} {punct_diff:<15}"
        )

        length_diff = abs(analysis1["length"] - analysis2["length"])
        print(
            f"{'文本长度':<20} {analysis1['length']:<20} {analysis2['length']:<20} {length_diff:<15}"
        )

        ratio1 = analysis1["punct_ratio"] * 100
        ratio2 = analysis2["punct_ratio"] * 100
        ratio_diff = abs(ratio1 - ratio2)
        print(
            f"{'标点比例(%)':<20} {ratio1:<20.2f} {ratio2:<20.2f} {ratio_diff:<15.2f}"
        )

        print(
            f"\n{'中文标点':<20} {analysis1['chinese_punct']:<20} {analysis2['chinese_punct']:<20}"
        )
        print(
            f"{'英文标点':<20} {analysis1['english_punct']:<20} {analysis2['english_punct']:<20}"
        )

        # 权重计算详情
        print(f"\n{'='*70}")
        print("标点权重计算")
        print(f"{'='*70}\n")

        print(f"权重:           {weight_info['weight']:.6f}")
        print(f"权重百分比:     {weight_info['weight_percent']:.2f}%")

        # 解释
        print(f"\n{'='*70}")
        print("权重解释")
        print(f"{'='*70}\n")

        if weight_info["weight"] >= 0.9:
            print("✓ 权重很高（≥0.9）")
            print("  标点差异很小，不会显著影响相似度计算")
        elif weight_info["weight"] >= 0.7:
            print("✓ 权重较高（≥0.7）")
            print("  标点差异在可接受范围内，影响较小")
        elif weight_info["weight"] >= 0.5:
            print("⚠ 权重中等（≥0.5）")
            print("  标点差异明显，会对相似度产生一定影响")
        else:
            print("❌ 权重较低（<0.5）")
            print("  标点差异很大，会显著降低相似度")

    def print_file_analysis(self, filepath):
        """打印文件的标点分析"""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                lines = [line.rstrip("\n\r") for line in f if line.strip()]

            print(f"\n{'='*70}")
            print("文件标点符号统计")
            print(f"{'='*70}\n")

            print(f"文件: {filepath}")
            print(f"行数: {len(lines)}\n")

            # 统计整体
            total_punct = sum(
                PunctuationHandler.count_punctuation_line(line) for line in lines
            )
            total_chars = sum(len(line) for line in lines)
            avg_punct_per_line = total_punct / len(lines) if lines else 0

            print(f"总标点数:     {total_punct}")
            print(f"总字符数:     {total_chars}")
            print(f"平均行标点数: {avg_punct_per_line:.2f}")
            print(f"整体标点比例: {total_punct/total_chars*100:.2f}%")

            # 按标点数分布
            print(f"\n标点数分布:")
            punct_ranges = [(0, 0), (1, 2), (3, 5), (6, 10), (11, 100)]

            for range_min, range_max in punct_ranges:
                count = sum(
                    1
                    for line in lines
                    if range_min
                    <= PunctuationHandler.count_punctuation_line(line)
                    <= range_max
                )
                percentage = count / len(lines) * 100
                print(
                    f"  {range_min:>2}-{range_max:<2} 个标点: {count:>3} 行 ({percentage:>5.1f}%)"
                )

            # 保存统计结果
            self.avg_punct = avg_punct_per_line
            print(f"\n✓ 平均标点数已更新为: {self.avg_punct:.2f}")

        except Exception as e:
            print(f"❌ 读取文件失败: {e}")


def main():
    """主程序"""
    parser = argparse.ArgumentParser(
        description="分析文本的标点符号特性",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 分析单个文本
  python tools/punctuation_analyzer.py "Hello, world!"
  
  # 对比两段文本的标点
  python tools/punctuation_analyzer.py --compare "Text 1." "Text 2?"
  
  # 分析文件
  python tools/punctuation_analyzer.py --file corpus.txt
        """,
    )

    parser.add_argument(
        "text1", nargs="?", help="第一段文本或文件路径（可选，默认使用示例）"
    )
    parser.add_argument("text2", nargs="?", help="第二段文本（对比模式，可选）")
    parser.add_argument("--compare", action="store_true", help="对比两段文本的标点符号")
    parser.add_argument(
        "--file", action="store_true", help="text1 为文件路径，分析整个文件"
    )

    try:
        args = parser.parse_args()

        # 默认示例文本
        default_text1 = "Hello, world! This is a test sentence."
        default_text2 = "Hi there? This is another test sentence!"

        analyzer = PunctuationAnalyzer()

        if args.file:
            analyzer.print_file_analysis(args.text1 or default_text1)
        elif args.compare and (args.text2 or args.text1):
            text1 = args.text1 or default_text1
            text2 = args.text2 or default_text2
            analyzer.print_comparison(text1, text2)
        elif args.text2:
            text1 = args.text1 or default_text1
            text2 = args.text2
            analyzer.print_comparison(text1, text2)
        else:
            text = args.text1 or default_text1
            analysis = analyzer.analyze_text(text)
            analyzer.print_analysis(analysis)

        return 0

    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
