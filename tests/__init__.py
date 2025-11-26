"""
Test Scripts - 测试脚本集合

此目录包含用于测试和验证功能的脚本。
这些脚本主要用于开发和调试目的。

脚本列表:

1. repair_test.py - 对齐修复集成测试
   使用硬编码的双语文本测试完整的对齐修复流程。

   用法:
     python tests/repair_test.py

   特性:
     • 完整的对齐修复流程测试
     • 自动创建临时文件
     • 生成修复报告

2. sentence_splitter_test.py - 句子分句测试
   测试句子分句功能的硬分句和软分句能力。

   用法:
     python tests/sentence_splitter_test.py

   特性:
     • 测试硬分句点（句子结束标点）
     • 测试软分句点（逗号、冒号等）
     • 支持中英文混合文本

3. test_ellipsis.py - 省略号处理测试
   测试省略号在句子分句中的正确处理。

   用法:
     python tests/test_ellipsis.py

   特性:
     • 验证省略号不被误认为句子边界
     • 测试引号内的省略号处理
     • 结合了原有的ellipsis相关测试

运行所有测试:
  python -m pytest tests/ -v

注意: 这些脚本主要用于开发验证，生产环境建议使用正式的单元测试框架。
"""
