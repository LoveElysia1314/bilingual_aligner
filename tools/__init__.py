#!/usr/bin/env python3
"""
Tools Package - 快速工具集

此目录包含用于快速测试和分析的命令行工具。
这些工具无需编写复杂代码，可直接使用项目功能。

工具列表:

1. similarity.py - 文本相似度工具
   快速计算两段文本的相似度，支持两种编码方法对比。

   用法:
     python tools/similarity.py "Source text" "Target text"
     python tools/similarity.py --method sentence "Source" "Target"
     python tools/similarity.py --compare "Source" "Target"

   特性:
     • 支持整段编码和句子编码两种方法
     • 可直接对比两种方法的差异
     • 自动计算标点权重

2. encoding_analyzer.py - 编码方法分析工具
   深入分析两种编码方法的性能和特性，支持测试模式。

   用法:
     python tools/encoding_analyzer.py "Text 1" "Text 2"
     python tools/encoding_analyzer.py --detailed "Text 1" "Text 2"
     python tools/encoding_analyzer.py --file source.txt target.txt
     python tools/encoding_analyzer.py --test  # 运行编码方法测试

   特性:
     • 文本语言特性分析（句子数、长度等）
     • 编码性能基准测试
     • 相似度对比和改进百分比
     • 编码方法功能测试
     • 详细推荐建议

3. punctuation_analyzer.py - 标点符号分析工具
   分析文本的标点符号特性和权重影响。

   用法:
     python tools/punctuation_analyzer.py "Text with punctuation."
     python tools/punctuation_analyzer.py --compare "Text 1" "Text 2"
     python tools/punctuation_analyzer.py --file corpus.txt

   特性:
     • 标点符号统计（中文/英文分类）
     • 标点权重计算
     • 文件级别的标点分布分析
     • 权重影响解释

快速开始示例:

# 快速相似度查询
python tools/similarity.py "The quick fox" "快速的狐狸"

# 对比编码方法
python tools/similarity.py --compare "Hello world" "你好世界"

# 详细性能分析
python tools/encoding_analyzer.py --detailed "Source text" "Target text"

# 运行编码方法测试
python tools/encoding_analyzer.py --test

# 分析文件的标点特性
python tools/punctuation_analyzer.py --file corpus.txt

所有工具都支持 --help 获取详细帮助信息。
"""
