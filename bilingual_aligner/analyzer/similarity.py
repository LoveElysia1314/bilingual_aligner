"""相似度分析器

从 tools/similarity.py 迁移到统一的分析器框架。
"""

from typing import Dict, Any, Optional
import numpy as np

from .base import SimilarityAnalyzer, AnalysisResult


class TextSimilarityAnalyzer(SimilarityAnalyzer):
    """文本相似度分析器

    支持两种编码方法的相似度计算和对比。
    """

    def __init__(self, processor=None):
        super().__init__(processor)

    def _get_punct_weight(self, text1: str, text2: str) -> float:
        """计算标点权重"""
        try:
            from bilingual_aligner.core.punctuation import (
                calculate_punctuation_similarity,
            )

            return calculate_punctuation_similarity(text1, text2)
        except Exception:
            return 1.0

    def similarity_paragraph(self, text1: str, text2: str) -> float:
        """整段编码相似度"""
        if not self.processor:
            return 0.0

        score = self.processor.calculate_similarity(text1, text2)
        weight = self._get_punct_weight(text1, text2)
        return max(0.0, score * weight)

    def similarity_sentence(self, text1: str, text2: str) -> float:
        """句子编码相似度"""
        if not self.processor:
            return 0.0

        try:
            emb1 = self.processor.get_normalized_embedding_by_sentences(
                text1, method="mean"
            )
            emb2 = self.processor.get_normalized_embedding_by_sentences(
                text2, method="mean"
            )
            score = float(np.dot(emb1, emb2))
            weight = self._get_punct_weight(text1, text2)
            return max(0.0, score * weight)
        except Exception:
            return 0.0

    def analyze(self, text1: str, text2: str, method: str = "both") -> AnalysisResult:
        """分析相似度

        Args:
            text1: 第一段文本
            text2: 第二段文本
            method: 编码方法 ("paragraph", "sentence", "both")

        Returns:
            Analysis results
        """
        data = {
            "text1_length": len(text1),
            "text2_length": len(text2),
        }

        if method in ("paragraph", "both"):
            para_sim = self.similarity_paragraph(text1, text2)
            data["paragraph_similarity"] = para_sim

        if method in ("sentence", "both"):
            sent_sim = self.similarity_sentence(text1, text2)
            data["sentence_similarity"] = sent_sim

        # 计算改进（如果两种方法都计算了）
        if method == "both":
            if "paragraph_similarity" in data and "sentence_similarity" in data:
                improvement = data["sentence_similarity"] - data["paragraph_similarity"]
                data["improvement"] = improvement
                data["improvement_percent"] = (
                    (improvement / data["paragraph_similarity"] * 100)
                    if data["paragraph_similarity"] != 0
                    else 0
                )

        return AnalysisResult(
            tool_name="similarity",
            result_type="text_similarity",
            data=data,
            metadata={"method": method},
        )
