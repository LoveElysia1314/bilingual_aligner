#!/usr/bin/env python3
"""
Tools Package - Quick Toolset

This directory contains command-line tools for quick testing and analysis.
These tools can be used directly without writing complex code to utilize project features.

Tool list:

1. similarity.py - Text similarity tool
   Quickly calculate the similarity between two texts, supporting comparison of two encoding methods.

   Usage:
     python tools/similarity.py "Source text" "Target text"
     python tools/similarity.py --method sentence "Source" "Target"
     python tools/similarity.py --compare "Source" "Target"

   Features:
     • Supports both paragraph encoding and sentence encoding methods
     • Can directly compare the differences between the two methods
     • Automatically calculates punctuation weight

2. encoding_analyzer.py - Encoding method analysis tool
   Deeply analyze the performance and characteristics of two encoding methods, supporting test mode.

   Usage:
     python tools/encoding_analyzer.py "Text 1" "Text 2"
     python tools/encoding_analyzer.py --detailed "Text 1" "Text 2"
     python tools/encoding_analyzer.py --file source.txt target.txt
     python tools/encoding_analyzer.py --test  # Run encoding method test

   Features:
     • Text language feature analysis (number of sentences, length, etc.)
     • Encoding performance benchmark testing
     • Similarity comparison and improvement percentage
     • Encoding method function testing
     • Detailed recommendation suggestions

3. punctuation_analyzer.py - Punctuation analysis tool
   Analyze the punctuation characteristics and weight impact of texts.

   Usage:
     python tools/punctuation_analyzer.py "Text with punctuation."
     python tools/punctuation_analyzer.py --compare "Text 1" "Text 2"
     python tools/punctuation_analyzer.py --file corpus.txt

   Features:
     • Punctuation statistics (Chinese/English classification)
     • Punctuation weight calculation
     • File-level punctuation distribution analysis
     • Weight impact explanation

Quick start examples:

# Quick similarity query
python tools/similarity.py "The quick fox" "快速的狐狸"

# Compare encoding methods
python tools/similarity.py --compare "Hello world" "你好世界"

# Detailed performance analysis
python tools/encoding_analyzer.py --detailed "Source text" "Target text"

# Run encoding method test
python tools/encoding_analyzer.py --test

# Analyze punctuation characteristics of file
python tools/punctuation_analyzer.py --file corpus.txt

All tools support --help for detailed help information.
"""
