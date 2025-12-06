#!/usr/bin/env python3
"""
整合测试文件 - 使用pytest框架
包含所有模块的基础测试
"""

import pytest
from bilingual_aligner.alignment import Alignment
from bilingual_aligner.analyzer import AnalysisResult
from bilingual_aligner.repair import RepairType, RepairLog, RepairExecutor


# 对齐模块测试
def test_alignment_dataclass():
    """测试 Alignment 数据类"""
    alignment = Alignment(
        source_indices=(0, 1),
        target_indices=(0, 1),
        score=0.9,
    )

    assert alignment.source_indices == (0, 1)
    assert alignment.target_indices == (0, 1)
    assert alignment.score == 0.9


def test_alignment_one_to_one():
    """测试一对一对齐判断"""
    alignment = Alignment(
        source_indices=(0,),
        target_indices=(0,),
        score=0.95,
    )
    assert alignment.is_one_to_one


def test_alignment_one_to_many():
    """测试一对多对齐判断"""
    alignment = Alignment(
        source_indices=(0,),
        target_indices=(0, 1),
        score=0.8,
    )
    assert not alignment.is_one_to_one


def test_alignment_operation_type():
    """测试对齐操作类型"""
    align_1_1 = Alignment((0,), (0,), 0.9)
    assert align_1_1.operation_type == "1:1"

    align_2_1 = Alignment((0, 1), (0,), 0.8)
    assert align_2_1.operation_type == "2:1"

    align_1_2 = Alignment((0,), (0, 1), 0.85)
    assert align_1_2.operation_type == "1:2"


def test_alignment_repr():
    """测试对齐表示字符串"""
    alignment = Alignment((0, 1), (0, 1), 0.9)
    repr_str = repr(alignment)
    assert "src=" in repr_str
    assert "tgt=" in repr_str
    assert "score=" in repr_str


# 分析器模块测试
def test_analysis_result_creation():
    """测试Analysis results创建"""
    result = AnalysisResult(
        tool_name="test",
        result_type="test_type",
        data={"key": "value"},
        metadata={"meta": "data"},
    )

    assert result.tool_name == "test"
    assert result.result_type == "test_type"
    assert result.data == {"key": "value"}


def test_analysis_result_serialization():
    """测试Analysis results序列化"""
    result = AnalysisResult(
        tool_name="test",
        result_type="test_type",
        data={"value": 0.5},
    )

    result_dict = result.to_dict()
    assert result_dict["tool"] == "test"
    assert result_dict["type"] == "test_type"


def test_similarity_analyzer_init(similarity_analyzer):
    """测试相似度分析器初始化"""
    assert similarity_analyzer is not None
    assert similarity_analyzer.name == "similarity"


def test_encoding_analyzer_init(encoding_analyzer):
    """测试编码分析器初始化"""
    assert encoding_analyzer is not None
    assert encoding_analyzer.name == "encoding"


def test_punctuation_analyzer_init(punctuation_analyzer):
    """测试标点符号分析器初始化"""
    assert punctuation_analyzer is not None
    assert punctuation_analyzer.name == "punctuation"


def test_punctuation_analyzer_analyze(punctuation_analyzer):
    """测试标点符号分析"""
    text = "Hello, world! This is a test."
    result = punctuation_analyzer.analyze(text)

    assert result is not None
    assert result.tool_name == "punctuation"
    assert "text1_punctuation" in result.data


# 修复模块测试
def test_repair_log_creation():
    """测试修复日志创建"""
    log = RepairLog(
        repair_type=RepairType.MERGE_LINES,
        description="Test merge",
        source_text="Hello",
        target_before="你好",
        target_after="你好 世界",
        similarity_before=0.7,
        similarity_after=0.85,
    )

    assert log.repair_type == RepairType.MERGE_LINES
    assert log.description == "Test merge"
    assert log.source_text == "Hello"


def test_repair_log_serialization():
    """测试修复日志序列化"""
    log = RepairLog(
        repair_type=RepairType.SPLIT_LINE,
        description="Test split",
        source_text="Hello",
        target_before="你好世界",
        target_after="你好 世界",
        similarity_before=0.8,
        similarity_after=0.9,
    )

    log_dict = log.to_dict()
    assert "repair_type" in log_dict
    assert "timestamp" in log_dict


def test_repair_executor_init(repair_applier):
    """测试修复执行器初始化"""
    assert repair_applier is not None
    assert hasattr(repair_applier, "apply_repairs")
    assert isinstance(repair_applier, RepairExecutor)


def test_repair_coordinator_init(repair_coordinator):
    """测试修复协调器初始化"""
    assert repair_coordinator is not None
    assert hasattr(repair_coordinator, "decide_repairs")


# 句子分句测试
def test_single_text_default(processor):
    """测试单个文本分句（默认示例）"""
    example = "It's a test. I'm here. You're coming."
    sentences = processor.split_sentences(example)
    assert len(sentences) == 3
    assert "It's a test." in sentences
    assert "I'm here." in sentences
    assert "You're coming." in sentences


def test_full_splitting(processor):
    """测试句子分句功能（完整测试）"""
    test_texts = [
        '"It seems… maybe we should wrap things up a bit earlier?" Masachika suggested to Elisa, watching Maria, utterly exhausted and lost in thought, and Irene, slumped on the ground flapping her hands about.',
        "It's a test. I'm here. You're coming.",
        "「看来……稍微提早结束比较好吗？」 看着精疲力尽正在恍神的玛利亚以及坐在地上甩动双手的依礼奈，政近向艾莉莎这么说。",
        "Price is $1,000.50 today. That's expensive!",
        'He said: "This is important." Then he left.',
        '未闭合引号的文本：某些"不完整的内容仍然有效',
    ]

    for text in test_texts:
        sentences = processor.split_sentences(text)
        assert isinstance(sentences, list)
        assert len(sentences) > 0
