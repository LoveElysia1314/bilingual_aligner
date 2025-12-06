"""
API module for bilingual text alignment.
Provides high-level interface for easy integration.
"""

from typing import List, Dict, Any, Optional
import os
import time
from .core.repairer import TextAligner as _TextAligner
from .core.processor import TextProcessor, get_text_processor
from .repair.models import RepairLog, RepairType


class TextAligner:
    """Clean API wrapper around core.TextAligner.

    Public methods are intentionally close to the previous API to avoid
    breaking examples: `repair()`, `save_results(...)`, `print_report(...)`.
    Internally delegates to `bilingual_aligner.core.repairer.TextAligner`.
    """

    def __init__(
        self,
        source_path: str,
        target_path: str,
        model_name: str = "Alibaba-NLP/GTE-multilingual-base",
        text_processor: Optional[TextProcessor] = None,
        **config,
    ):
        merged_config = {**config}
        if model_name is not None:
            merged_config["model_name"] = model_name

        if text_processor is not None:
            self._aligner = _TextAligner(
                source_path, target_path, text_processor=text_processor, **merged_config
            )
        else:
            self._aligner = _TextAligner(source_path, target_path, **merged_config)

        self.start_time = time.time()

    def repair(self) -> Dict[str, Any]:
        result = self._aligner.repair()

        # Override processing time to include model loading
        result["stats"]["processing_time_seconds"] = round(
            time.time() - self.start_time, 2
        )

        # Perform validation checks
        exceptions = []
        stats = result.get("stats", {})
        repair_logs = result.get("repair_logs", [])

        # Use line_difference from stats (initial difference)
        line_difference = stats.get("line_difference", 0)
        if line_difference != 0:
            exceptions.append(
                f"LineDifferenceException: line_difference={line_difference}"
            )

        # Check for deletions or insertions (INSERT_LINE and DELETE_LINE are dangerous operations)
        has_deletion_or_insertion = any(
            r.repair_type in (RepairType.INSERT_LINE, RepairType.DELETE_LINE)
            for r in repair_logs
        )
        if has_deletion_or_insertion:
            exceptions.append("OperationException: deletions or insertions detected")

        # Check repair rate
        total_repairs = stats.get("total_repairs", 0)
        source_content_lines = stats.get("source_content_lines", 0)
        if source_content_lines > 0:
            repair_rate = total_repairs / source_content_lines
            if repair_rate > 0.05:
                exceptions.append(
                    f"RepairRateException: repair_rate={repair_rate:.2%} > 5%"
                )

        # Low-similarity bulk check removed: individual low-similarity cases
        # are still recorded per-repair in repair_logs but are not elevated
        # to an overall exception threshold here.

        # Add validation to result
        result["state"] = {"exceptions": exceptions}

        return result

    def save_results(
        self,
        result: Dict[str, Any],
        output_dir: Optional[str] = None,
        logs_file: Optional[str] = None,
        repaired_file: Optional[str] = None,
        include_texts: bool = True,
    ):
        """Save repair results.

        Path resolution logic (handled by core layer):
        - Absolute paths are used as-is
        - Relative paths are resolved against output_dir
        - Simple filenames are written to output_dir
        - Returned paths in result["io"] are always absolute paths

        Args:
            result: result dict from `repair()`
            output_dir: Base directory for output files. All relative paths are resolved
                       against this directory. If None, uses current working directory.
            logs_file: Output path for repair logs (JSON format). Can be:
                      - Absolute path (used as-is)
                      - Relative path (joined with output_dir)
                      - Simple filename (written to output_dir)
                      - None (defaults to "repair_logs.json" in output_dir)
            repaired_file: Output path for repaired target file. Can be:
                          - Absolute path (used as-is)
                          - Relative path (joined with output_dir)
                          - Simple filename (written to output_dir)
                          - None (defaults to "<target_basename>_repaired.txt" in output_dir)
            include_texts: whether to include full source/target texts in repair logs

        Returns:
            result dict with "io" key containing:
            - "repaired_path": absolute path to repaired file
            - "logs_path": absolute path to logs file
            - "output_base": absolute path to output_dir
            - "generated_at": ISO timestamp
        """
        # Normalize and validate output_dir
        if output_dir is None:
            output_dir = os.getcwd()
        output_dir = os.path.normpath(os.path.abspath(output_dir))

        # Set defaults for optional file parameters
        if logs_file is None:
            logs_file = "repair_logs.json"

        if repaired_file is None:
            base_name = os.path.splitext(os.path.basename(self._aligner.target_file))[0]
            repaired_file = f"{base_name}_repaired.txt"

        # ✅ Delegate ALL path resolution to core/repairer.py
        # Do NOT process paths here - pass them as-is to avoid double-joining
        return self._aligner.save_results(
            result,
            output_dir,
            logs_file=logs_file,
            repaired_file=repaired_file,
            include_texts=include_texts,
        )

    def print_report(
        self,
        result: Dict[str, Any],
        output_dir: Optional[str] = None,
        show_sample_logs: int = 3,
        level: str = "normal",
    ):
        """Print a summary report of the repair results

        Args:
            result: Repair result dict
            output_dir: Optional output directory for context
            show_sample_logs: Deprecated parameter (kept for compatibility)
            level: Output level (minimal, normal, verbose)
        """
        return self._aligner.print_report(
            result, output_dir, show_sample_logs, level=level
        )


def calculate_similarity(
    text1: str, text2: str, model_name: Optional[str] = None
) -> float:
    """
    Convenience helper to compute similarity between two texts.

    This reuses the cached `TextProcessor` instances via `get_text_processor`.
    Falls back to a simple equality check if embeddings fail to compute.
    """
    try:
        processor = get_text_processor(model_name=model_name)
        return processor.calculate_similarity(text1, text2)
    except Exception:
        t1 = text1.strip()
        t2 = text2.strip()
        return 1.0 if t1 == t2 else 0.0
