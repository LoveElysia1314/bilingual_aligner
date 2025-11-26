import os
import json
import time
import logging
import hashlib
from datetime import datetime
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np
from .processor import TextProcessor, get_text_processor
from .splitter import UniversalSplitter
from ..corpus import (
    LineObject,
    BilingualCorpus,
    RepairType,
    SplitType,
    AlignmentState,
    RepairLog,
)
from ..repair_applier import RepairApplier
from ..position import LocationRange


class TextAligner:
    """Main coordinator class for bilingual text alignment repair system.

    This class orchestrates the entire process of repairing misalignments between
    bilingual text files (source and target languages). It loads the input files,
    analyzes alignment quality, performs automatic repairs, and provides results
    in various formats.

    Attributes:
        source_file (str): Path to the source language text file.
        target_file (str): Path to the target language text file.
        config (dict): Configuration parameters for alignment analysis and repair.
        corpus (BilingualCorpus): Container for loaded text data.

    Args:
        source_file (str): Path to the source language file. Must exist.
        target_file (str): Path to the target language file. Must exist.
        **config: Optional configuration overrides for DEFAULT_CONFIG parameters.

    Raises:
        FileNotFoundError: If either source_file or target_file does not exist.

    Example:
        >>> aligner = TextAligner("source.txt", "target.txt")
        >>> result = aligner.repair()
        >>> aligner.save_results(result, "output/")
        >>> aligner.print_report(result)
    """

    DEFAULT_CONFIG = {
        # Repair decision parameters
        "soft_split_penalty": 0.05,  # Penalty for soft splits (between sentences)
        "insert_fallback_score": 0.6,  # Fixed score for insert fallback option
        "delete_penalty": 0.05,  # Penalty for delete operations (future use)
    }

    def __init__(
        self,
        source_file: str,
        target_file: str,
        text_processor: Optional[TextProcessor] = None,
        **config,
    ):
        self.logger = logging.getLogger(__name__)
        self.source_file = source_file
        self.target_file = target_file
        self.config = {**self.DEFAULT_CONFIG, **config}

        # Check if input files exist
        if not os.path.exists(source_file):
            raise FileNotFoundError(f"Source file '{source_file}' does not exist.")
        if not os.path.exists(target_file):
            raise FileNotFoundError(f"Target file '{target_file}' does not exist.")

        # Initialize components (allow overriding model via config 'model_name')
        model_name = self.config.get("model_name")
        # Allow injection of a pre-created TextProcessor to enable reuse/caching
        if text_processor is not None:
            self._text_processor = text_processor
        else:
            # Use factory which may return a cached instance
            self._text_processor = get_text_processor(model_name=model_name)
        self.corpus = BilingualCorpus(self._text_processor)
        self.repair_applier = RepairApplier(self.corpus, self.config)

        # Load files
        self._load_files()

    @property
    def text_processor(self):
        return self._text_processor

    @text_processor.setter
    def text_processor(self, value):
        self._text_processor = value
        self.corpus.text_processor = value

    def _load_files(self):
        """Load source and target files into the corpus"""
        self.corpus.load_source(self.source_file)
        self.corpus.load_target(self.target_file)
        self.logger.info(
            f"Loading completed: {len(self.corpus.source_lines)} source lines, "
            f"{len(self.corpus.target_lines)} target lines"
        )
        self.logger.info(
            f"Total source lines (including empty): {len(self.corpus.source_with_empty)}"
        )

    def repair(self) -> Dict[str, Any]:
        """Execute the repair process and return results"""
        start_time = time.time()

        # Execute repairs using new DP-based pipeline
        result = self.repair_applier.apply_repairs()

        # Add timing and state info
        result["stats"]["total_time_seconds"] = time.time() - start_time
        result["state"]["exceptions"] = result["state"].get("exceptions", [])
        return result

    def save_results(
        self,
        result: Dict[str, Any],
        output_dir: str,
        logs_file: Optional[str] = None,
        repaired_file: Optional[str] = None,
        include_texts: bool = True,
    ):
        """Save repair results to files"""
        os.makedirs(output_dir, exist_ok=True)

        # Build output aligned to SOURCE file structure (preserve source empty lines)
        # `result['repaired_lines']` is produced aligned to source filtered indices
        # (one entry per non-empty source line). To produce a human-usable repaired
        # target file that corresponds line-for-line with the source file, we
        # map each repaired entry back into the source file positions stored in
        # `self.corpus.source_with_empty` via `self.corpus.content_line_map`.
        repaired_with_empty = list(self.corpus.source_with_empty)

        # Map repaired lines (indexed by source filtered idx) back to original
        # source positions (original_idx is 1-indexed). If repaired list is
        # shorter than expected, skip missing entries.
        for src_filtered_idx, src_original_idx in self.corpus.content_line_map.items():
            if src_filtered_idx < len(result.get("repaired_lines", [])):
                repaired_with_empty[src_original_idx - 1] = result["repaired_lines"][
                    src_filtered_idx
                ]

        # Ensure repaired_with_empty has same length as source_with_empty
        # (should normally be equal, but guard against unexpected cases)
        while len(repaired_with_empty) > len(self.corpus.source_with_empty):
            repaired_with_empty.pop()
        while len(repaired_with_empty) < len(self.corpus.source_with_empty):
            repaired_with_empty.append("")

        # Save repaired file
        # If caller provided `repaired_file`, interpret it similarly to `logs_file`:
        # - if it's an absolute path, use as-is
        # - if it contains directory components, treat as relative to output_dir
        # - if it's a simple filename, write into output_dir
        if repaired_file:
            if os.path.isabs(repaired_file):
                # Absolute path: use as-is
                repaired_path = repaired_file
            elif os.path.dirname(repaired_file):
                # Contains directory components: treat as relative to output_dir
                repaired_path = os.path.join(output_dir, repaired_file)
                # Ensure parent directory exists
                os.makedirs(os.path.dirname(repaired_path), exist_ok=True)
            else:
                # Simple filename: write into output_dir
                repaired_path = os.path.join(output_dir, repaired_file)
        else:
            base_name = os.path.splitext(os.path.basename(self.target_file))[0]
            repaired_path = os.path.join(output_dir, f"{base_name}_repaired.txt")

        with open(repaired_path, "w", encoding="utf-8") as f:
            f.write("\n".join(repaired_with_empty))

        # Determine logs format and path
        # If caller provides a simple filename (no directory component), write it
        # into the provided `output_dir`. If a path is provided, treat it as relative
        # to `output_dir` to maintain consistency with CLI expectations.
        if logs_file:
            if os.path.isabs(logs_file):
                # Absolute path: use as-is
                logs_path = logs_file
            elif os.path.dirname(logs_file):
                # Contains directory components: treat as relative to output_dir
                logs_path = os.path.join(output_dir, logs_file)
                # Ensure parent directory exists
                os.makedirs(os.path.dirname(logs_path), exist_ok=True)
            else:
                # Simple filename: write into output_dir
                logs_path = os.path.join(output_dir, logs_file)
            logs_format = "json" if logs_path.endswith(".json") else "ndjson"
        else:
            logs_format = "json"
            logs_path = os.path.join(output_dir, "repair_logs.json")

        # Calculate total improvement (from repair logs)
        logs = result.get("repair_logs", [])
        total_improvement = sum(
            log.similarity_after - log.similarity_before for log in logs
        )

        # Use stats from result directly (already calculated in apply_repairs)
        stats = result.get("stats", {})

        # Extract line number mapping from result if available
        line_number_mapping = result.get("line_number_mapping")

        # Prepare metadata
        metadata = {"generated_at": datetime.utcnow().isoformat() + "Z"}

        # Prepare repair records with improved position format
        def _prepare_record(log: RepairLog, idx: int) -> Dict[str, Any]:
            """
            Prepare repair record with unified position format using LocationRange.

            Format: src@{original_lines}[content_indices] → tgt@{original_lines}[content_indices]

            Uses original line numbers for user to locate text in original files,
            and content indices for internal reference.

            Example: src@{317,319}[193,194] → tgt@{320}[195]
            - 317,319 are original file line numbers
            - 193,194 are internal filtered indices
            """
            # Calculate improvement for this record
            improvement = log.similarity_after - log.similarity_before

            # Map RepairType to error_type string
            type_map = {
                RepairType.MERGE_LINES: "merge",
                RepairType.SPLIT_LINE: "split",
                RepairType.INSERT_LINE: "insert",
                RepairType.DELETE_LINE: "delete",
            }
            error_type = type_map.get(log.repair_type, "unknown")

            # Build unified position format using LocationRange
            src_range = None
            tgt_range = None

            if log.source_orig_line_numbers:
                src_range = LocationRange(
                    is_source=True,
                    line_numbers=log.source_orig_line_numbers,
                    content_indices=log.source_filtered_position,
                )

            if log.target_orig_line_numbers:
                tgt_range = LocationRange(
                    is_source=False,
                    line_numbers=log.target_orig_line_numbers,
                    content_indices=log.target_filtered_position,  # Use target filtered position
                )

            # Format position string
            if src_range and tgt_range:
                position_str = f"{str(src_range)} → {str(tgt_range)}"
            elif src_range:
                position_str = str(src_range)
            elif tgt_range:
                position_str = str(tgt_range)
            else:
                position_str = ""

            # Calculate line count change
            line_change = 0
            if log.repair_type == RepairType.MERGE_LINES:
                line_change = -1  # Two lines become one
            elif log.repair_type == RepairType.SPLIT_LINE:
                line_change = 1  # One line becomes two
            elif log.repair_type == RepairType.INSERT_LINE:
                line_change = 1  # One line added
            elif log.repair_type == RepairType.DELETE_LINE:
                line_change = -1  # One line removed

            # Use placeholders if text is missing
            src_txt = log.source_text or ""
            tgt_before_txt = log.target_before or ""
            tgt_after_txt = log.target_after or ""

            rec = {
                "id": idx + 1,
                "type": error_type,
                # compact position string for human reading and compactness
                "position": position_str,
            }

            # Include texts only when requested (to keep logs compact by default)
            if include_texts:
                rec["source_text"] = src_txt
                rec["target_before"] = tgt_before_txt
                rec["target_after"] = tgt_after_txt

            # Add split type for SPLIT_LINE repairs
            if log.repair_type == RepairType.SPLIT_LINE and log.split_type:
                rec["split_type"] = log.split_type.value

            return rec

        # Sort logs by source original line numbers to order repairs by line number
        sorted_logs = sorted(
            logs,
            key=lambda log: (
                log.source_orig_line_numbers[0] if log.source_orig_line_numbers else 0
            ),
        )

        # Calculate comprehensive similarity statistics across ALL repaired alignments.
        # NOTE: This is already calculated in repair_applier.apply_repairs(), but we
        # ensure it's also computed here for save_results in case it's called separately.
        repaired_lines = result.get("repaired_lines", [])
        all_sims = []

        for i, src_line in enumerate(self.corpus.source_lines):
            if i < len(repaired_lines):
                tgt_text = repaired_lines[i]
            else:
                tgt_text = ""
            if not tgt_text:
                continue
            try:
                tgt_line = self.corpus._create_or_get_line(tgt_text, is_source=False)
                sim = self.corpus.get_similarity(
                    src_line, tgt_line, use_punctuation_weight=False
                )
                all_sims.append(sim)
            except Exception:
                # If similarity computation fails for a pair, skip it
                continue

        # Calculate statistics on ALL similarities
        similarity_stats = self._compute_similarity_statistics(all_sims)

        # Build comprehensive stats dict aligned with the compact schema
        total_repairs = stats.get("total_repairs", 0)
        similarity_improvement = round(stats.get("similarity_improvement", 0.0), 4)

        # Derive processing time breakdown: total, encoding (inferred), dp, postprocess
        total_time = round(stats.get("total_time_seconds", 0.0), 4)
        dp_time = round(stats.get("dp_time_seconds", 0.0), 4)
        post_time = round(stats.get("postprocess_time_seconds", 0.0), 4)
        encoding_time = total_time - dp_time - post_time
        if encoding_time < 0:
            # fallback guard
            encoding_time = 0.0

        processing_time = {
            "total": round(total_time, 4),
            "encoding": round(encoding_time, 4),
            "dp": dp_time,
            "postprocess": post_time,
        }

        # File summary: structured and compact string
        # Build compact file summary only (src/tgt compact strings)
        src_total = stats.get("source_total_lines", 0)
        src_content = stats.get("source_content_lines", 0)
        tgt_total_before = stats.get("target_total_lines", 0)
        tgt_total_after = stats.get("target_total_lines_after", 0)
        tgt_before = stats.get("target_content_lines_before", 0)
        tgt_after = stats.get("target_content_lines_after", 0)

        file_summary = {
            "src": f"src@{{{src_total}}}[{src_content}]",
            "tgt": f"tgt@{{{tgt_total_before}->{tgt_total_after}}}[{tgt_before}->{tgt_after}]",
        }

        repair_types = {
            "merge": sum(
                1 for log in sorted_logs if log.repair_type == RepairType.MERGE_LINES
            ),
            "split": sum(
                1 for log in sorted_logs if log.repair_type == RepairType.SPLIT_LINE
            ),
            "insert": sum(
                1 for log in sorted_logs if log.repair_type == RepairType.INSERT_LINE
            ),
            "delete": sum(
                1 for log in sorted_logs if log.repair_type == RepairType.DELETE_LINE
            ),
        }

        stats_dict = {
            "total_repairs": total_repairs,
            "similarity_improvement": similarity_improvement,
            "processing_time": processing_time,
            "file_summary": file_summary,
            "repair_types": repair_types,
            "similarity_statistics": similarity_stats,
            "source_content_lines": src_content,
            "target_content_lines_after": tgt_after,
        }

        # Update result['stats'] with the new stats dict (adds similarity_statistics)
        result["stats"] = stats_dict

        # Build structured exceptions list using neighbor-index validation.
        #
        # NOTE: previous implementation produced exceptions based on global
        # percentiles of similarity (mean / 25% / 10% / 5% / 1%), which caused
        # frequent false positives. Per request, remove those percentile-based
        # anomaly rules but keep `similarity_statistics` intact. Instead we now
        # detect local index-shift anomalies by comparing for each source line i
        # the similarity to target lines at indices (i), (i-1), (i+1).
        #
        # Logic:
        # - If similarity(src_i, tgt_i) is the highest among available neighbors,
        #   the alignment at i is considered normal (no exception).
        # - If similarity(src_i, tgt_{i-1}) is highest, report a "TARGET_BEFORE"
        #   anomaly (translation appears earlier than expected).
        # - If similarity(src_i, tgt_{i+1}) is highest, report a "TARGET_AFTER"
        #   anomaly (translation appears later than expected).
        #
        # We avoid head/tail comparisons when neighbors are missing.
        exceptions = []

        # Helper to compute similarity between a source LineObject and a target
        # text string (returns None on failure)
        def _sim_to_text(src_line_obj, tgt_text: str) -> Optional[float]:
            if not tgt_text:
                return None
            try:
                tgt_line_obj = self.corpus._create_or_get_line(
                    tgt_text, is_source=False
                )
                return self.corpus.get_similarity(
                    src_line_obj, tgt_line_obj, use_punctuation_weight=False
                )
            except Exception:
                return None

        anomaly_by_index = {}
        # Iterate over source indices and compare with neighbors when present
        for i, src_line in enumerate(self.corpus.source_lines):
            # Skip if no repaired target at this index
            if i >= len(repaired_lines):
                continue

            sim_center = _sim_to_text(src_line, repaired_lines[i])
            sim_prev = None
            sim_next = None

            if i - 1 >= 0 and (i - 1) < len(repaired_lines):
                sim_prev = _sim_to_text(src_line, repaired_lines[i - 1])
            if i + 1 < len(repaired_lines):
                sim_next = _sim_to_text(src_line, repaired_lines[i + 1])

            # Collect available sims and find the argmax
            sims = []  # list of tuples (label, sim)
            if sim_prev is not None:
                sims.append(("prev", sim_prev))
            if sim_center is not None:
                sims.append(("center", sim_center))
            if sim_next is not None:
                sims.append(("next", sim_next))

            if not sims:
                continue

            # Determine max
            best_label, best_sim = max(sims, key=lambda x: x[1])

            # If center is best, it's normal; otherwise record anomaly
            if best_label == "center":
                continue

            # Build example snippet
            try:
                src_txt = src_line.text
            except Exception:
                src_txt = ""

            def _example_for(idx: int) -> str:
                if 0 <= idx < len(repaired_lines):
                    return repaired_lines[idx]
                return ""

            if best_label == "prev":
                code = "TARGET_BEFORE"
                message = "Translation appears before source (likely index shift: target at i-1 matches source i)"
                examples = [
                    {
                        "index": i,
                        "source": src_txt,
                        "target_prev": _example_for(i - 1),
                        "target_center": _example_for(i),
                    }
                ]
                value = round(best_sim - (sim_center or 0.0), 4)
            else:
                code = "TARGET_AFTER"
                message = "Translation appears after source (likely index shift: target at i+1 matches source i)"
                examples = [
                    {
                        "index": i,
                        "source": src_txt,
                        "target_center": _example_for(i),
                        "target_next": _example_for(i + 1),
                    }
                ]
                value = round(best_sim - (sim_center or 0.0), 4)

            # Store anomaly info by index; final exceptions will only include
            # anomalies that are part of runs of length >= 3
            anomaly_by_index[i] = {
                "code": code,
                "message": message,
                "value": value,
                "examples": examples,
            }

        # Build exceptions only for consecutive runs of indices of length >= 3
        exceptions_from_runs = []
        if anomaly_by_index:
            sorted_idxs = sorted(anomaly_by_index.keys())
            runs = []
            current_run = [sorted_idxs[0]]
            for idx in sorted_idxs[1:]:
                if idx == current_run[-1] + 1:
                    current_run.append(idx)
                else:
                    runs.append(current_run)
                    current_run = [idx]
            runs.append(current_run)

            for run in runs:
                if len(run) < 3:
                    continue

                # Determine if codes are uniform across the run
                codes = [anomaly_by_index[i]["code"] for i in run]
                uniform_code = codes.count(codes[0]) == len(codes)
                if uniform_code:
                    code = codes[0]
                    message = f"Consecutive index-shift anomalies detected: {code} for {len(run)} lines"
                else:
                    code = "TARGET_MIXED_SHIFT"
                    message = f"Consecutive index-shift anomalies detected (mixed types) for {len(run)} lines"

                # Aggregate examples (take up to 3 examples from the run)
                examples = []
                for i in run[:3]:
                    ex = anomaly_by_index[i].get("examples", [])
                    if ex:
                        examples.append(ex[0])

                # Aggregate value as average improvement over center
                avg_value = round(
                    sum(anomaly_by_index[i]["value"] for i in run) / len(run), 4
                )

                exceptions_from_runs.append(
                    {
                        "level": "WARNING",
                        "code": code,
                        "message": message,
                        "value": avg_value,
                        "affected_lines": run,
                        "examples": examples,
                    }
                )

        # Attach neighbor-based exceptions (only groups >=3)
        exceptions.extend(exceptions_from_runs)

        # Create config snapshot: include non-default overrides and model name
        try:
            defaults = self.DEFAULT_CONFIG
        except Exception:
            defaults = {}
        config_snapshot = {}
        for k, v in self.config.items():
            if k not in defaults or defaults.get(k) != v:
                config_snapshot[k] = v
        # Always include model name if available from processor
        try:
            model_name = getattr(self._text_processor, "model_name", None)
            if model_name:
                config_snapshot["model_name"] = model_name
        except Exception:
            pass

        # Save logs
        if logs_format == "ndjson":
            # Write to temporary file first
            temp_logs_path = logs_path + ".tmp"
            with open(temp_logs_path, "w", encoding="utf-8") as f:
                # Write header with metadata and summary
                header = {
                    "metadata": metadata,
                    "stats": stats_dict,
                }
                f.write(json.dumps(header, ensure_ascii=False) + "\n")
                for idx, log in enumerate(sorted_logs):
                    f.write(
                        json.dumps(_prepare_record(log, idx), ensure_ascii=False) + "\n"
                    )
            # Atomic rename
            os.replace(temp_logs_path, logs_path)
        else:
            # JSON payload (compact schema)
            payload = {
                "metadata": metadata,
                "config_snapshot": config_snapshot,
                "stats": stats_dict,
                "exceptions": exceptions,
                "repairs": [
                    _prepare_record(log, idx) for idx, log in enumerate(sorted_logs)
                ],
            }

            # Write to temporary file first
            temp_logs_path = logs_path + ".tmp"
            with open(temp_logs_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)

            # Atomic rename
            os.replace(temp_logs_path, logs_path)

            # Record actual written paths in the result for programmatic use
            result.setdefault("io", {})
            result["io"]["repaired_path"] = os.path.abspath(repaired_path)
            result["io"]["logs_path"] = os.path.abspath(logs_path)
            result["io"]["output_base"] = os.path.abspath(output_dir)
            result["io"]["generated_at"] = metadata.get("generated_at")

            # Do not log a console "Results saved to" here; printing is handled
            # by `print_report` so that file-save notification appears after
            # pipeline completion messages in console output.
            return result

    def print_report(
        self,
        result: Dict[str, Any],
        output_dir: Optional[str] = None,
        show_sample_logs: int = 20,
    ):
        """Print a summary report of the repair results"""
        stats = result.get("stats", {})
        print("\nRepair Results:")
        print(f"  Repairs performed: {stats.get('total_repairs', 0)}")
        print(f"  Source lines: {stats.get('source_content_lines', 0)}")
        print(
            f"  Target lines: {stats.get('target_content_lines_after', stats.get('target_line_count', 0))}"
        )
        total_time = stats.get("total_time_seconds")
        if total_time is not None:
            print(f"  Processing time: {total_time:.2f}s")

        # Detailed repair listing omitted from console summary to keep output concise.
        # Full repair logs are available in the JSON/ndjson output saved to disk.

        if output_dir:
            print(f"\nResults saved to: {os.path.abspath(output_dir)}")

    def _compute_similarity_statistics(
        self, similarities: List[float]
    ) -> Dict[str, float]:
        """
        Compute similarity statistics optimized for anomaly detection.

        Focus on detecting low-quality alignments:
        - Lower percentiles (1%_low, 5%_low, 10%_low, 25%_low) to find weak spots
        - Central tendency (mean, median, stdev) for overall quality
        - Extreme values (min, max) for outliers

        Args:
            similarities: List of similarity scores for all aligned line pairs

        Returns:
            Dict with statistics focused on anomaly detection
        """
        if not similarities:
            return {
                "count": 0,
                "mean": 0.0,
                "median": 0.0,
                "stdev": 0.0,
                "1%_low": 0.0,
                "5%_low": 0.0,
                "10%_low": 0.0,
                "25%_low": 0.0,
                "min": 0.0,
                "max": 0.0,
            }

        import statistics

        sims = sorted(similarities)
        n = len(sims)

        # Mean
        mean = sum(sims) / n

        # Median
        if n % 2 == 1:
            median = sims[n // 2]
        else:
            median = (sims[n // 2 - 1] + sims[n // 2]) / 2

        # Standard deviation
        try:
            stdev = statistics.stdev(sims) if n > 1 else 0.0
        except Exception:
            stdev = 0.0

        # Percentiles helper
        def percentile(data: List[float], p: float) -> float:
            """Calculate p-th percentile (0-100)"""
            if not data:
                return 0.0
            sorted_data = sorted(data)
            idx = (p / 100) * (len(sorted_data) - 1)
            lower = int(idx)
            upper = lower + 1
            if upper >= len(sorted_data):
                return sorted_data[lower]
            weight = idx - lower
            return sorted_data[lower] * (1 - weight) + sorted_data[upper] * weight

        return {
            "count": n,
            "mean": round(mean, 4),
            "median": round(median, 4),
            "stdev": round(stdev, 4),
            "1%_low": round(percentile(sims, 1), 4),
            "5%_low": round(percentile(sims, 5), 4),
            "10%_low": round(percentile(sims, 10), 4),
            "25%_low": round(percentile(sims, 25), 4),
            "min": round(min(sims), 4),
            "max": round(max(sims), 4),
        }
