#!/usr/bin/env python3
"""
Ellipsis Handling Test Tool

Tests ellipsis (省略号) handling in sentence splitting.
Combines and refines the original test_ellipsis_fix.py and test_ellipsis_split.py.

Usage:
    python tools/test_ellipsis.py
"""

import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bilingual_aligner.core.splitter import UniversalSplitter
from bilingual_aligner.core.processor import TextProcessor


def show_split_points(text, split_points, label):
    """Show text with split points marked with |"""
    result = []
    prev = 0
    for pos in sorted(split_points):
        if pos <= len(text):
            result.append(text[prev:pos])
            result.append("|")
            prev = pos
    result.append(text[prev:])
    marked_text = "".join(result)
    print(f"{label}: '{marked_text}'")
    print(f"  Positions: {split_points}")


def test_ellipsis_fix():
    """Test that ellipsis is not treated as sentence boundaries."""
    print("Testing ellipsis handling fix...")
    print("=" * 40)

    # Test case 1: Soft split points with ellipsis
    test1 = "Hello... how are you?"
    soft_points1 = UniversalSplitter.find_soft_split_points(test1)
    show_split_points(test1, soft_points1, "Test 1")
    print("  Expected: split after ellipsis (Hello...| how are you?)")
    if 8 in soft_points1:
        print("  ✅ PASS")
    else:
        print("  ❌ FAIL")
    print()

    # Test case 2: Ellipsis inside quotes should not be soft split
    test2 = '"Hello... how are you?" he said.'
    soft_points2 = UniversalSplitter.find_soft_split_points(test2)
    show_split_points(test2, soft_points2, "Test 2")
    print("  Expected: no splits (ellipsis inside quotes)")
    if not soft_points2:
        print("  ✅ PASS")
    else:
        print("  ❌ FAIL")
    print()

    print("=" * 40)
    print("Soft split ellipsis verification complete!")


def test_ellipsis_splitting():
    """Test sentence splitting on specific examples with ellipsis."""
    # Test sentence from the user's example
    test_sentence = '"Ah, yes. Alisa... san."'

    print(f"Testing sentence: {test_sentence}")
    print("=" * 50)

    # Test with TextProcessor (which uses UniversalSplitter internally)
    print("\n1. Using TextProcessor.split_sentences():")
    processor = TextProcessor()
    sentences = processor.split_sentences(test_sentence)
    print(f"Number of sentences: {len(sentences)}")
    for i, sentence in enumerate(sentences, 1):
        print(f"  Sentence {i}: '{sentence}'")

    # Test hard split points
    print("\n2. Hard split points:")
    hard_points = UniversalSplitter.find_hard_split_points(test_sentence)
    print(f"Hard split positions: {hard_points}")
    print("Hard split points in text:")
    for pos in hard_points:
        if pos < len(test_sentence):
            print(
                f"  Position {pos}: '{test_sentence[pos]}' (context: '{test_sentence[max(0, pos-5):pos+5]}')"
            )

    # Test soft split points
    print("\n3. Soft split points:")
    soft_points = UniversalSplitter.find_soft_split_points(test_sentence)
    print(f"Soft split positions: {soft_points}")
    print("Soft split points in text:")
    for pos in soft_points:
        if pos < len(test_sentence):
            print(f"  Position {pos}: '{test_sentence[max(0, pos-5):pos+5]}'")

    print("\n" + "=" * 50)
    print("Analysis:")
    print(
        "- The sentence contains ellipsis '...' which should NOT be treated as a sentence boundary"
    )
    print("- Expected: 2 sentences ('\"Ah, yes.\"' and '\"Alisa... san.\"')")
    print(f"- Actual: {len(sentences)} sentences")
    print("- The ellipsis '...' is correctly NOT treated as split points")
    if len(sentences) == 2 and "Alisa... san." in sentences[1]:
        print("✅ SUCCESS: Ellipsis handling is fixed!")
    else:
        print("❌ FAILED: Ellipsis is still causing incorrect splits")


def main():
    """Run all ellipsis tests."""
    print("Ellipsis Handling Test Suite")
    print("=" * 50)
    test_ellipsis_fix()
    print("\n" + "-" * 50 + "\n")
    test_ellipsis_splitting()
    print("\nTest suite complete.")


if __name__ == "__main__":
    main()
