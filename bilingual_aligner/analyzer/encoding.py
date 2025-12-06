"""编码方法分析器

从 tools/encoding_analyzer.py 和 tools/encoding_comparison.py 迁移。
分析不同编码方法的性能和特性。
"""

from typing import Dict, Any, Optional, List
import time

from .base import EncodingAnalyzer, AnalysisResult


class TextEncodingAnalyzer(EncodingAnalyzer):
    """文本编码方法分析器

    分析和对比不同编码方法的性能。
    """

    def __init__(self, processor=None):
        super().__init__(processor)

    def analyze_text_structure(self, text: str) -> Dict[str, Any]:
        """分析文本结构"""
        if not self.processor:
            return {"text": text, "length": len(text)}

        try:
            sentences = self.processor.split_sentences(text)
        except Exception:
            sentences = text.split("。") + text.split(".")

        return {
            "text_length": len(text),
            "sentence_count": len(sentences),
            "avg_sentence_length": len(text) / len(sentences) if sentences else 0,
        }

    def benchmark_encoding(
        self, text: str, method: str = "paragraph", iterations: int = 3
    ) -> Dict[str, Any]:
        """对编码方法进行基准测试"""
        if not self.processor:
            return {"method": method, "time_ms": 0.0, "iterations": iterations}

        times = []
        for _ in range(iterations):
            start = time.time()
            try:
                if method == "paragraph":
                    _ = self.processor.get_embedding(text)
                elif method == "sentence":
                    _ = self.processor.get_embeddings_by_sentences(text)
            except Exception:
                pass
            end = time.time()
            times.append((end - start) * 1000)  # 转换为毫秒

        avg_time = sum(times) / len(times) if times else 0.0
        return {
            "method": method,
            "time_ms": avg_time,
            "iterations": iterations,
            "min_time_ms": min(times) if times else 0.0,
            "max_time_ms": max(times) if times else 0.0,
        }

    def analyze(self, text1: str, text2: Optional[str] = None) -> AnalysisResult:
        """分析编码方法

        Args:
            text1: 第一段文本
            text2: 第二段文本（可选）

        Returns:
            Analysis results
        """
        data = {}

        # 分析第一段文本结构
        data["text1_structure"] = self.analyze_text_structure(text1)

        # 如果提供了第二段文本
        if text2:
            data["text2_structure"] = self.analyze_text_structure(text2)

        return AnalysisResult(
            tool_name="encoding",
            result_type="text_analysis",
            data=data,
        )

    def compare_methods(self, text1: str, text2: str) -> AnalysisResult:
        """对比两种编码方法

        Args:
            text1: 第一段文本
            text2: 第二段文本

        Returns:
            Analysis results
        """
        if not self.processor:
            return AnalysisResult(
                tool_name="encoding",
                result_type="comparison",
                data={"error": "processor not available"},
            )

        try:
            # 计算两种方法的相似度
            para_sim = self.processor.calculate_similarity(text1, text2)

            sent_emb1 = self.processor.get_normalized_embedding_by_sentences(
                text1, method="mean"
            )
            sent_emb2 = self.processor.get_normalized_embedding_by_sentences(
                text2, method="mean"
            )
            sent_sim = float((sent_emb1 * sent_emb2).sum())
        except Exception as e:
            self.logger.debug(f"Failed to compare methods: {e}")
            sent_sim = para_sim = 0.0

        data = {
            "paragraph_similarity": para_sim,
            "sentence_similarity": sent_sim,
            "difference": abs(para_sim - sent_sim),
        }

        if para_sim != 0:
            improvement = (sent_sim - para_sim) / para_sim * 100
            data["improvement_percent"] = improvement
            data["better_method"] = "sentence" if improvement > 0 else "paragraph"

        return AnalysisResult(
            tool_name="encoding",
            result_type="method_comparison",
            data=data,
        )
