"""Pytest 配置和 fixtures

定义所有测试共享的 fixtures 和配置。
"""

import pytest
from pathlib import Path

from bilingual_aligner.core.processor import get_text_processor
from bilingual_aligner.analyzer import (
    TextSimilarityAnalyzer,
    TextEncodingAnalyzer,
    TextPunctuationAnalyzer,
)
from bilingual_aligner.repair import RepairApplier, RepairCoordinator
from bilingual_aligner.corpus import BilingualCorpus


@pytest.fixture(scope="session")
def processor():
    """获取文本处理器 fixture"""
    return get_text_processor()


@pytest.fixture
def project_root():
    """获取项目根目录"""
    return Path(__file__).parent.parent


@pytest.fixture
def demo_dir(project_root):
    """获取 demo 目录"""
    return project_root / "demo"


@pytest.fixture
def similarity_analyzer(processor):
    """创建相似度分析器"""
    return TextSimilarityAnalyzer(processor)


@pytest.fixture
def encoding_analyzer(processor):
    """创建编码分析器"""
    return TextEncodingAnalyzer(processor)


@pytest.fixture
def punctuation_analyzer(processor):
    """创建标点符号分析器"""
    return TextPunctuationAnalyzer(processor)


@pytest.fixture
def repair_applier(processor):
    """创建修复应用器"""
    corpus = BilingualCorpus(processor)
    config = {}
    return RepairApplier(corpus, config)


@pytest.fixture
def repair_coordinator(processor):
    """创建修复协调器"""
    return RepairCoordinator(processor=processor)
