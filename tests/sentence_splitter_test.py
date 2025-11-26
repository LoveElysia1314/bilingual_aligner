#!/usr/bin/env python3
"""
测试脚本：测试句子分句功能
调用项目代码，对硬编码文本进行硬分句和软分句，输出结果
支持命令行参数：如果提供文本参数，则测试该文本；否则执行完整测试
"""

import sys
import os
import argparse

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from bilingual_aligner.core.processor import TextProcessor


def single_text_test(text):
    """测试单个文本的分句功能（辅助函数，避免 pytest 将带参数的函数当作 fixture）"""

    # 初始化 TextProcessor
    processor = TextProcessor()

    print("=== 单个文本分句测试 ===\n")
    print(f"测试文本: {text}")
    print("-" * 50)

    # 硬分句（句子结束标点）
    hard_splits = processor.find_hard_split_points(text)
    print(f"硬分句点: {hard_splits}")

    # 软分句（逗号、冒号等）
    soft_splits = processor.find_soft_split_points(text)
    print(f"软分句点: {soft_splits}")

    # 演示实际分句结果
    sentences = processor.split_sentences(text)
    print(f"分句结果 ({len(sentences)} 个句子):")
    for j, sentence in enumerate(sentences, 1):
        print(f"  {j}. {sentence}")

    print("\n" + "=" * 80 + "\n")


def test_single_text_default():
    """pytest wrapper: 使用内置示例文本运行单文本分句测试"""
    example = "It's a test. I'm here. You're coming."
    single_text_test(example)


def test_full_splitting():
    """测试句子分句功能（完整测试）"""

    # 初始化 TextProcessor
    processor = TextProcessor()

    # 硬编码测试文本（中英文混合）
    test_texts = [
        '"It seems… maybe we should wrap things up a bit earlier?" Masachika suggested to Elisa, watching Maria, utterly exhausted and lost in thought, and Irene, slumped on the ground flapping her hands about.',
        "It's a test. I'm here. You're coming.",
        "「看来……稍微提早结束比较好吗？」 看着精疲力尽正在恍神的玛利亚以及坐在地上甩动双手的依礼奈，政近向艾莉莎这么说。",
        "Price is $1,000.50 today. That's expensive!",
        'He said: "This is important." Then he left.',
        '未闭合引号的文本：某些"不完整的内容仍然有效',
    ]

    print("=== 句子分句完整测试 ===\n")

    for i, text in enumerate(test_texts, 1):
        print(f"测试文本 {i}: {text}")
        print("-" * 50)

        # 硬分句（句子结束标点）
        hard_splits = processor.find_hard_split_points(text)
        print(f"硬分句点: {hard_splits}")

        # 软分句（逗号、冒号等）
        soft_splits = processor.find_soft_split_points(text)
        print(f"软分句点: {soft_splits}")

        # 演示实际分句结果
        sentences = processor.split_sentences(text)
        print(f"分句结果 ({len(sentences)} 个句子):")
        for j, sentence in enumerate(sentences, 1):
            print(f"  {j}. {sentence}")

        print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="测试句子分句功能")
    parser.add_argument("text", nargs="?", help="要测试的文本（可选）")
    args = parser.parse_args()

    if args.text:
        single_text_test(args.text)
    else:
        test_full_splitting()
