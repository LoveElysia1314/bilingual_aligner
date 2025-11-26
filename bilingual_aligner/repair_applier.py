"""
New repair pipeline: Integrate DP validation with targeted post-processing.

This module orchestrates the complete repair workflow:
1. Run DP global alignment to get baseline
2. Identify 2:1 mappings requiring post-processing
3. Simulate repairs and select best options
4. Generate standardized repair logs
5. Apply repairs to corpus and return results
"""

import logging
import time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np

from .aligners.enum_pruning_aligner import DPAligner, AlignmentStep, OperationType
from .corpus import BilingualCorpus, RepairLog, RepairType, SplitType
from .position import build_line_number_mapping, LineNumberMapping, LocationRange


logger = logging.getLogger(__name__)


@dataclass
class PostProcessResult:
    """Result of post-processing"""

    source_ahead_count: int
    repairs_generated: int
    repair_logs: List[RepairLog]
    total_time: float


class RepairApplier:
    """
    New unified repair application pipeline combining DP and post-processing.

    Replaces the old executor-based approach with DP-driven strategy.
    """

    def __init__(self, corpus: BilingualCorpus, config: Dict):
        self.corpus = corpus
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.line_number_mapping: Optional[LineNumberMapping] = (
            None  # Set during apply_repairs
        )

    def apply_repairs(self) -> Dict:
        """
        Execute complete repair pipeline:
        1. DP alignment
        2. Post-processing
        3. Generate results

        Returns:
            result dict with repaired_lines, repair_logs, stats
        """

        start_time = time.time()

        # Record initial state before any repairs
        initial_target_line_count = len(self.corpus.target_lines)

        # Build line number mapping from current corpus state
        line_number_mapping = build_line_number_mapping(
            self.corpus.source_lines,
            self.corpus.target_lines,
            self.corpus.source_with_empty,
            self.corpus.target_with_empty,
            self.corpus.content_line_map,
            self.corpus.target_content_line_map,
        )

        # Store for use in other methods
        self.line_number_mapping = line_number_mapping

        # Phase 1: DP alignment
        dp_start = time.time()
        dp_aligner = DPAligner(self.corpus, self.config)
        dp_result = dp_aligner.run()
        dp_time = time.time() - dp_start

        # Support both old return (list of ops) and new dict return with stages
        if isinstance(dp_result, dict):
            # Prefer Stage1 for post-processing (it identifies non-1:1 candidates)
            alignment_steps = dp_result.get("stage1", [])
        else:
            alignment_steps = dp_result

        if not alignment_steps:
            self.logger.error("DP alignment failed")
            return self._create_empty_result()

        self.logger.info(
            f"DP alignment completed: {len(alignment_steps)} steps ({dp_time:.3f}s)"
        )

        # Phase 2: Post-processing for non-1:1 cases (SOURCE_AHEAD / TARGET_SPLIT)
        pp_start = time.time()
        pp_result = self._postprocess_source_ahead(alignment_steps)
        pp_time = time.time() - pp_start

        self.logger.info(
            f"Post-processing completed: {pp_result.source_ahead_count} source-ahead cases, "
            f"{pp_result.repairs_generated} repairs ({pp_time:.3f}s)"
        )

        # Phase 3: Prepare final result
        total_time = time.time() - start_time

        # Calculate similarity improvement
        total_improvement = sum(
            log.similarity_after - log.similarity_before
            for log in pp_result.repair_logs
        )

        # Calculate comprehensive similarity statistics across ALL repaired alignments
        repaired_lines = self._build_repaired_aligned_to_source()
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
                continue

        similarity_stats = self._compute_similarity_statistics(all_sims)

        # Update line number mapping after repairs (line_number_mapping is stored
        # in result.stats separately; avoid adding dynamic attributes to mapping)

        # Check for alignment exceptions
        exceptions = self._check_alignment_exceptions(alignment_steps)

        result = {
            # Build `repaired_lines` aligned to source filtered indices.
            # We perform a greedy matching: for each source filtered line, find
            # the most similar remaining target line (by embedding cosine).
            "repaired_lines": repaired_lines,
            "repair_logs": pp_result.repair_logs,
            "line_number_mapping": self.line_number_mapping,  # Include mapping in result
            "stats": {
                "total_repairs": len(pp_result.repair_logs),
                "source_total_lines": len(self.corpus.source_with_empty),
                "target_total_lines": len(self.corpus.target_with_empty),
                "target_total_lines_after": len(self.corpus.source_with_empty),
                "source_content_lines": len(self.corpus.source_lines),
                "target_content_lines_before": initial_target_line_count,
                "target_content_lines_after": len(self.corpus.target_lines),
                "line_difference": len(self.corpus.target_lines)
                - len(self.corpus.source_lines),
                "similarity_improvement": round(total_improvement, 4),
                "dp_time_seconds": dp_time,
                "postprocess_time_seconds": pp_time,
                "total_time_seconds": total_time,
                "dp_operations": self._count_operations(alignment_steps),
                "similarity_statistics": similarity_stats,
            },
            "state": {"exceptions": exceptions},
        }

        self.logger.info(f"Repair pipeline complete ({total_time:.3f}s total)")
        return result

    def _check_alignment_exceptions(self, alignment_steps: List) -> List[str]:
        """
        Check for alignment exceptions: consecutive non-1:1 operations of different types.
        """
        exceptions = []
        prev_op = None

        for step in alignment_steps:
            current_op = step.operation

            if current_op != OperationType.NO_SHIFT:
                if (
                    prev_op is not None
                    and prev_op != OperationType.NO_SHIFT
                    and prev_op != current_op
                ):
                    exceptions.append(
                        f"Consecutive non-1:1 operations: {prev_op.value} followed by {current_op.value} "
                        f"at src[{step.src_start}:{step.src_end}] tgt[{step.tgt_start}:{step.tgt_end}]"
                    )
                prev_op = current_op
            else:
                prev_op = None  # Reset on 1:1

        return exceptions

    def _build_repaired_aligned_to_source(self) -> List[str]:
        """
        Build a list of repaired target texts aligned to source filtered indices.

        Strategy:
        - For each source filtered LineObject, compute cosine similarity to each
          target LineObject (using precomputed normalized embeddings).
        - Greedily assign the highest-similarity unused target to the source.
        - If no suitable target remains, leave the source slot empty.
        This produces a one-to-one mapping from source filtered indices to
        repaired target texts, which can then be mapped to original source
        file positions for output.
        """
        src_lines = self.corpus.source_lines
        tgt_lines = self.corpus.target_lines

        # Conservative, order-preserving mapping:
        # If the corpus has been repaired to have the same number of target lines
        # as source lines, assume a one-to-one correspondence by index: source[i]
        # maps to target[i]. This preserves original target order and prevents
        # global reordering that moves sentences to the end of the file.
        if len(tgt_lines) == len(src_lines):
            return [t.text for t in tgt_lines]

        # Fallback conservative behavior: do not perform global reordering.
        # Build a list aligned to source filtered indices where available; if no
        # suitable target exists at that index, leave empty string.
        repaired = [""] * len(src_lines)
        min_len = min(len(src_lines), len(tgt_lines))
        for i in range(min_len):
            repaired[i] = tgt_lines[i].text

        # If target shorter than source, remaining entries stay empty. If target
        # longer, extra target lines are left unused (do not move them to the end).
        self.logger.debug(
            "Repaired aligned to source conservatively: "
            f"src_lines={len(src_lines)}, tgt_lines={len(tgt_lines)}"
        )

        return repaired

    def _get_repair_sort_key(self, repair_info: Tuple) -> int:
        """
        Extract sort key from repair_info tuple for explicit sorting.

        Repairs MUST be applied in descending order of target indices to avoid
        index shifting issues. This method extracts the key explicitly for clarity.

                Args:
                        repair_info: Tuple with one of these formats:
                            - SOURCE_AHEAD (relabelled for postprocessing):
                                (src_idx1, src_idx2, tgt_idx, repair, log, "NON_2_TO_1_REPAIR")
                            - TARGET_SPLIT (relabelled for postprocessing):
                                (src_idx, tgt_idx1, tgt_idx2, repair, log, "NON_1_TO_2_REPAIR")

        Returns:
            Maximum target index for sorting (higher indices processed first)
        """
        operation_type = repair_info[-1]

        # Accept legacy labels and new postprocessing labels for compatibility
        if operation_type in ("SOURCE_AHEAD", "NON_2_TO_1_REPAIR"):
            # SOURCE_AHEAD / NON_2_TO_1_REPAIR: (src_idx1, src_idx2, tgt_idx, repair, log, _)
            _, _, tgt_idx, _, _, _ = repair_info
            return tgt_idx
        elif operation_type in ("TARGET_SPLIT", "NON_1_TO_2_REPAIR"):
            # TARGET_SPLIT / NON_1_TO_2_REPAIR: (src_idx, tgt_idx1, tgt_idx2, repair, log, _)
            _, tgt_idx1, tgt_idx2, _, _, _ = repair_info
            return max(tgt_idx1, tgt_idx2)
        else:
            raise ValueError(f"Unknown repair operation type: {operation_type}")

    def _rebuild_target_indices(self) -> None:
        """
        Rebuild target line indices and content line map.

        This MUST be called after any modification to self.corpus.target_lines
        to ensure consistency between:
        - LineObject.current_index
        - target_content_line_map

        Failure to call this leads to index mismatches and silent data corruption.
        """
        self.corpus.target_content_line_map.clear()
        for idx, line in enumerate(self.corpus.target_lines):
            line.current_index = idx
            self.corpus.target_content_line_map[idx] = line.original_line_number
        self.logger.debug(
            f"Rebuilt indices for {len(self.corpus.target_lines)} target lines"
        )

    def _verify_repairs_applied(self) -> None:
        """
        Verify that all repairs were applied correctly.

        Validation checks:
        1. Index continuity: All indices 0..len-1 are present
        2. Map consistency: target_content_line_map matches target_lines
        3. Line validity: original_line_number in valid range

        Raises:
            ValueError: If any validation check fails
        """
        errors = []

        # Check 1: Index continuity
        expected_indices = set(range(len(self.corpus.target_lines)))
        actual_indices = {line.current_index for line in self.corpus.target_lines}
        if expected_indices != actual_indices:
            missing = expected_indices - actual_indices
            extra = actual_indices - expected_indices
            if missing:
                errors.append(f"Missing indices: {sorted(missing)}")
            if extra:
                errors.append(f"Extra indices: {sorted(extra)}")

        # Check 2: Map size consistency
        if len(self.corpus.target_content_line_map) != len(self.corpus.target_lines):
            errors.append(
                f"Map size {len(self.corpus.target_content_line_map)} != "
                f"target_lines size {len(self.corpus.target_lines)}"
            )

        # Check 3: Map key-value consistency
        for idx, orig_line_num in self.corpus.target_content_line_map.items():
            if idx >= len(self.corpus.target_lines):
                errors.append(
                    f"Map has index {idx} beyond target_lines length {len(self.corpus.target_lines)}"
                )
            elif self.corpus.target_lines[idx].current_index != idx:
                errors.append(
                    f"Index mismatch at {idx}: "
                    f"map says {idx} but LineObject.current_index={self.corpus.target_lines[idx].current_index}"
                )

        if errors:
            error_msg = "Repair verification FAILED:\n" + "\n".join(
                f"  {e}" for e in errors
            )
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        self.logger.info(
            f"Repair verification PASSED: {len(self.corpus.target_lines)} lines indices consistent"
        )

    def _postprocess_source_ahead(
        self, alignment_steps: List[AlignmentStep]
    ) -> PostProcessResult:
        """
        Post-process all alignment operations requiring repair.

        Modified behavior:
        - SOURCE_AHEAD (2:1): ALWAYS repaired (required)
        - TARGET_SPLIT (1:2): NOT repaired (left as-is)

        Reason: TARGET_SPLIT repairs (merging) conflict with DP alignment decisions.
        If DP decided to separate two target lines (1:2), we should respect that decision.

        Process:
        1. Scan all alignment_steps and collect SOURCE_AHEAD repairs
        2. Apply SOURCE_AHEAD repairs one by one (no sorting needed)
        3. Verify all repairs applied correctly
        """

        source_ahead_count = 0
        repair_logs = []

        # Phase A: Collect all non-1:1 repairs first (no corpus modifications yet)
        repairs_to_apply = []

        for step_idx, step in enumerate(alignment_steps):
            # Handle SOURCE_AHEAD (2:1) -> try split first, insert only as fallback
            if step.operation == OperationType.SOURCE_AHEAD:
                source_ahead_count += 1
                src_start, src_end = step.src_start, step.src_end
                tgt_idx = step.tgt_start

                # Verify 2:1 mapping
                if (
                    src_end - src_start + 1 != 2
                    or step.tgt_end - step.tgt_start + 1 != 1
                ):
                    self.logger.warning(
                        f"SOURCE_AHEAD at step {step_idx}: expected 2:1, got {src_end-src_start+1}:{step.tgt_end-step.tgt_start+1}"
                    )
                    continue

                # Simulate and select best repair (corpus state unchanged)
                best_repair, repair_log = self._find_best_repair(
                    src_start, src_end, tgt_idx
                )

                if best_repair and repair_log:
                    repairs_to_apply.append(
                        (
                            src_start,
                            src_end,
                            tgt_idx,
                            best_repair,
                            repair_log,
                            "NON_2_TO_1_REPAIR",
                        )
                    )
                    repair_logs.append(repair_log)

            # Handle TARGET_SPLIT (1:2) -> create merge repair (mandatory to restore 1:1)
            elif step.operation == OperationType.TARGET_SPLIT:
                # Expect src is single index, target covers two indices
                src_idx = step.src_start
                tgt_idx1 = step.tgt_start
                tgt_idx2 = step.tgt_end

                # Basic validation
                if (tgt_idx2 - tgt_idx1 + 1) != 2 or (
                    step.src_end - step.src_start + 1
                ) != 1:
                    self.logger.warning(
                        f"TARGET_SPLIT at step {step_idx}: expected 1:2, got {step.src_end-step.src_start+1}:{tgt_idx2-tgt_idx1+1}"
                    )
                    continue

                merge_repair, merge_log = self._find_merge_repair(
                    src_idx, tgt_idx1, tgt_idx2
                )
                if merge_repair and merge_log:
                    repairs_to_apply.append(
                        (
                            src_idx,
                            tgt_idx1,
                            tgt_idx2,
                            merge_repair,
                            merge_log,
                            "NON_1_TO_2_REPAIR",
                        )
                    )
                    repair_logs.append(merge_log)

        # Phase B: Sort repairs by target indices descending to avoid index shift issues
        repairs_to_apply.sort(key=lambda r: self._get_repair_sort_key(r), reverse=True)

        # Apply repairs one by one in sorted order
        for repair_idx, repair_info in enumerate(repairs_to_apply):
            operation_type = repair_info[-1]
            if operation_type in ("SOURCE_AHEAD", "NON_2_TO_1_REPAIR"):
                src_idx1, src_idx2, tgt_idx, best_repair, repair_log, _ = repair_info
                self._apply_repair_to_corpus(best_repair, src_idx1, src_idx2, tgt_idx)
            elif operation_type in ("TARGET_SPLIT", "NON_1_TO_2_REPAIR"):
                src_idx, tgt_idx1, tgt_idx2, merge_repair, merge_log, _ = repair_info
                self._apply_merge_to_corpus(merge_repair, tgt_idx1, tgt_idx2)
            else:
                self.logger.warning(
                    f"Unknown repair type in apply loop: {operation_type}"
                )

            # CRITICAL: Rebuild indices after EVERY repair to ensure consistency
            self._rebuild_target_indices()
            self.logger.debug(
                f"Applied repair {repair_idx+1}/{len(repairs_to_apply)}: type={operation_type}"
            )

        # Phase C: Verify all repairs applied correctly
        try:
            self._verify_repairs_applied()
        except ValueError as e:
            self.logger.error(f"Repairs verification failed: {e}")
            raise

        return PostProcessResult(
            source_ahead_count=source_ahead_count,
            repairs_generated=len(repair_logs),
            repair_logs=repair_logs,
            total_time=0.0,
        )

    def _find_best_repair(
        self, src_idx1: int, src_idx2: int, tgt_idx: int
    ) -> Tuple[Optional[Dict], Optional[RepairLog]]:
        """
        Find best repair for 2:1 mapping (SOURCE_AHEAD).

        V2 Logic: Compare three parallel repair strategies:
        - Strategy A: SPLIT target line (tgt→tgt_a|tgt_b)
        - Strategy B: INSERT virtual source before src2 (0.6 fixed score)
        - Strategy C: INSERT virtual target after tgt (0.6 fixed score)

        Returns:
            (repair_dict, repair_log) or (None, None)
        """

        source_lines = self.corpus.source_lines
        target_lines = self.corpus.target_lines

        if src_idx1 >= len(source_lines) or src_idx2 >= len(source_lines):
            return None, None
        if tgt_idx >= len(target_lines):
            return None, None

        src1_line = source_lines[src_idx1]
        src2_line = source_lines[src_idx2]
        tgt_line = target_lines[tgt_idx]

        src1 = src1_line.text
        src2 = src2_line.text
        tgt = tgt_line.text

        # Calculate baseline similarities (before repair)
        base_sim1 = self.corpus.get_similarity(
            src1_line, tgt_line, use_punctuation_weight=False
        )
        base_sim2 = self.corpus.get_similarity(
            src2_line, tgt_line, use_punctuation_weight=False
        )
        base_avg = (base_sim1 + base_sim2) / 2

        candidates = []
        # Unified config key: use insert_fallback_score (matches core.TextAligner.DEFAULT_CONFIG)
        virtual_score = self.config.get("insert_fallback_score", 0.6)

        # ========== STRATEGY A: Split target line ==========
        # Try hard splits
        hard_points = self.corpus.text_processor.find_hard_split_points(tgt)
        for split_pos in hard_points:
            candidate = self._simulate_split(
                src1, src2, tgt, split_pos, SplitType.HARD_SPLIT, base_avg
            )
            if candidate:
                candidates.append(candidate)

        # Try soft splits if no good hard splits
        if not candidates or max(c["score"] for c in candidates) < base_avg:
            soft_points = self.corpus.text_processor.find_soft_split_points(tgt)
            for split_pos in soft_points:
                candidate = self._simulate_split(
                    src1, src2, tgt, split_pos, SplitType.SOFT_SPLIT, base_avg
                )
                if candidate:
                    candidates.append(candidate)

        # Only consider INSERT fallback strategies if no split candidates were found
        if not candidates:
            # ========== STRATEGY B: Insert virtual source before src2 ==========
            sim_src2_tgt = self.corpus.get_similarity(
                src2_line, tgt_line, use_punctuation_weight=False
            )
            avg_B = (virtual_score + sim_src2_tgt) / 2

            strategy_b = {
                "type": RepairType.INSERT_LINE,
                "split_type": None,
                "score": avg_B,
                "split_pos": None,
                "description": f"Insert virtual source (score: {virtual_score:.2f} + {sim_src2_tgt:.4f})",
                "strategy": "insert_virtual_source_before",
            }
            candidates.append(strategy_b)

            # ========== STRATEGY C: Insert virtual target after tgt ==========
            sim_src1_tgt = self.corpus.get_similarity(
                src1_line, tgt_line, use_punctuation_weight=False
            )
            avg_C = (sim_src1_tgt + virtual_score) / 2

            strategy_c = {
                "type": RepairType.INSERT_LINE,
                "split_type": None,
                "score": avg_C,
                "split_pos": None,
                "description": f"Insert virtual target (score: {sim_src1_tgt:.4f} + {virtual_score:.2f})",
                "strategy": "insert_virtual_target_after",
            }
            candidates.append(strategy_c)

        # Select best by score
        if not candidates:
            return None, None

        best_repair = max(candidates, key=lambda c: c["score"])

        # Log which strategy was selected
        strategy_name = best_repair.get("strategy", "split_target")

        # Create unified position format
        src_range = LocationRange.from_original_lines(
            is_source=True,
            line_numbers=(
                self.corpus.source_lines[src_idx1].original_line_number,
                self.corpus.source_lines[src_idx2].original_line_number,
            ),
        )
        tgt_range = LocationRange.from_original_lines(
            is_source=False,
            line_numbers=self.corpus.target_lines[tgt_idx].original_line_number,
        )

        self.logger.info(
            f"Selected repair for 2:1 at {src_range} → {tgt_range}: "
            f"{strategy_name} with score {best_repair['score']:.4f}"
        )

        # Create repair log
        repair_log = self._create_repair_log(
            src_idx1, src_idx2, tgt_idx, best_repair, base_avg
        )

        return best_repair, repair_log

    def _simulate_split(
        self,
        src1: str,
        src2: str,
        tgt: str,
        split_pos: int,
        split_type: SplitType,
        base_avg: float,
    ) -> Optional[Dict]:
        """
        Simulate splitting target at position and evaluate (STRATEGY A).

        Strategy A: Split target into tgt1, tgt2
        - Calculates: avg(sim(src1, tgt1), sim(src2, tgt2))
        - For SOFT_SPLIT, applies penalty

        Returns: dict with score and split info, or None if invalid
        """
        try:
            tgt1 = tgt[:split_pos].strip()
            tgt2 = tgt[split_pos:].strip()

            if not tgt1 or not tgt2:
                return None

            # Create temporary LineObject instances for similarity calculation
            from .corpus import LineObject

            src1_line = self.corpus._create_or_get_line(src1, is_source=True)
            src2_line = self.corpus._create_or_get_line(src2, is_source=True)
            tgt1_line = self.corpus._create_or_get_line(tgt1, is_source=False)
            tgt2_line = self.corpus._create_or_get_line(tgt2, is_source=False)

            # New similarity after split
            sim1 = self.corpus.get_similarity(
                src1_line, tgt1_line, use_punctuation_weight=False
            )
            sim2 = self.corpus.get_similarity(
                src2_line, tgt2_line, use_punctuation_weight=False
            )
            new_avg = (sim1 + sim2) / 2

            # Apply penalty for soft split using configurable parameter
            if split_type == SplitType.SOFT_SPLIT:
                penalty = self.config.get("soft_split_penalty", 0.05)
                new_avg -= penalty

            return {
                "type": RepairType.SPLIT_LINE,
                "split_type": split_type,
                "score": new_avg,
                "split_pos": split_pos,
                "description": f"Split at {split_pos} chars",
                "strategy": "split_target",
            }

        except Exception as e:
            self.logger.debug(f"Split simulation failed: {e}")
            return None

    def _create_repair_log(
        self,
        src_idx1: int,
        src_idx2: int,
        tgt_idx: int,
        repair: Dict,
        sim_before: float,
    ) -> RepairLog:
        """
        Create repair log for post-processing record (2:1 operation).

        Supports all three repair strategies:
        - SPLIT_LINE: split target into two parts
        - INSERT_LINE (strategy B): insert virtual source before src2
        - INSERT_LINE (strategy C): insert virtual target after tgt

        Position tracking:
        - source_orig_line_numbers: (src_orig_1, src_orig_2) - original file line numbers
        - target_orig_line_numbers: (tgt_orig,) - original file line number
        - source_filtered_position: (src_idx1, src_idx2) - internal filtered indices
        - target_filtered_position: (tgt_idx,) - internal filtered index
        """

        src1 = self.corpus.source_lines[src_idx1]
        src2 = self.corpus.source_lines[src_idx2]
        tgt = self.corpus.target_lines[tgt_idx]

        source_text = f"{src1.text} | {src2.text}"
        target_before = tgt.text

        # Target after depends on repair strategy
        strategy = repair.get("strategy", "split_target")

        if repair["type"] == RepairType.SPLIT_LINE:
            # Strategy A: Split target
            split_pos = repair["split_pos"]
            tgt1 = target_before[:split_pos].strip()
            tgt2 = target_before[split_pos:].strip()
            target_after = f"{tgt1} | {tgt2}"
        elif strategy == "insert_virtual_source_before":
            # Strategy B: Insert virtual source before src2
            # Target remains same; src side gets virtual line with source text as placeholder
            target_after = f"{target_before} | [PLACEHOLDER from src: {src2.text}]"
        elif strategy == "insert_virtual_target_after":
            # Strategy C: Insert virtual target after tgt
            # Source remains same; tgt side gets virtual line with source text as placeholder
            target_after = f"{target_before} | [PLACEHOLDER from src: {src2.text}]"
        else:
            # Fallback INSERT
            target_after = target_before

        # Get original line numbers from LineObjects
        src1_orig = src1.original_line_number
        src2_orig = src2.original_line_number
        tgt_orig = tgt.original_line_number

        return RepairLog(
            repair_type=repair["type"],
            description=repair["description"],
            source_text=source_text,
            target_before=target_before,
            target_after=target_after,
            similarity_before=sim_before,
            similarity_after=repair["score"],
            source_orig_line_numbers=(
                src1_orig,
                src2_orig,
            ),  # Original file line numbers
            target_orig_line_numbers=(tgt_orig,),  # Original file line number
            source_filtered_position=(
                src_idx1,
                src_idx2,
            ),  # Internal filtered indices (0-based)
            target_filtered_position=(tgt_idx,),  # Internal filtered index (0-based)
            split_type=repair.get("split_type"),
            is_fallback=repair["type"] == RepairType.INSERT_LINE,
        )

    def _find_merge_repair(
        self, src_idx: int, tgt_idx1: int, tgt_idx2: int
    ) -> Tuple[Optional[Dict], Optional[RepairLog]]:
        """
        Find merge repair for 1:2 mapping (TARGET_SPLIT).

        IMPORTANT: Merging is MANDATORY for line count balance, not optional.
        Even if merging doesn't improve similarity, it must be done.
        We always merge and return a repair, never return None.

        Returns:
            (repair_dict, repair_log) - always returns a merge repair
        """

        source_lines = self.corpus.source_lines
        target_lines = self.corpus.target_lines

        if src_idx >= len(source_lines):
            return None, None
        if tgt_idx1 >= len(target_lines) or tgt_idx2 >= len(target_lines):
            return None, None

        src_line = source_lines[src_idx]
        tgt1_line = target_lines[tgt_idx1]
        tgt2_line = target_lines[tgt_idx2]

        src = src_line.text
        tgt1 = tgt1_line.text
        tgt2 = tgt2_line.text

        # Calculate baseline similarity (keeping two target lines separate)
        sim1_separate = self.corpus.get_similarity(
            src_line, tgt1_line, use_punctuation_weight=False
        )
        sim2_separate = self.corpus.get_similarity(
            src_line, tgt2_line, use_punctuation_weight=False
        )
        base_sim = (sim1_separate + sim2_separate) / 2

        # Merge with space separator
        merged_text = f"{tgt1} {tgt2}"
        merged_line = self.corpus._create_or_get_line(merged_text, is_source=False)
        sim_merged = self.corpus.get_similarity(
            src_line, merged_line, use_punctuation_weight=False
        )

        # ALWAYS return merge repair (mandatory for balance, not conditional on similarity)
        merge_candidate = {
            "type": RepairType.MERGE_LINES,
            "score": sim_merged,
            "description": "Merge target lines (mandatory for line count balance)",
        }

        # Get original line numbers from LineObjects
        src_orig = src_line.original_line_number
        tgt1_orig = tgt1_line.original_line_number
        tgt2_orig = tgt2_line.original_line_number

        # Create merge log with both target indices
        merge_log = RepairLog(
            repair_type=RepairType.MERGE_LINES,
            description="Merge target lines (mandatory for line count balance)",
            source_text=src,
            target_before=f"{tgt1} | {tgt2}",
            target_after=merged_text,
            similarity_before=base_sim,
            similarity_after=sim_merged,
            source_orig_line_numbers=(src_orig,),  # Original file line number
            target_orig_line_numbers=(
                tgt1_orig,
                tgt2_orig,
            ),  # Both original line numbers
            split_type=None,
            is_fallback=False,
        )

        return merge_candidate, merge_log

    def _apply_repair_to_corpus(
        self, repair: Dict, src_idx1: int, src_idx2: int, tgt_idx: int
    ):
        """
        Apply repair to corpus (modify target_lines only).

        NOTE: Caller is responsible for rebuilding indices via _rebuild_target_indices()
        after this method returns.
        """

        if repair["type"] == RepairType.SPLIT_LINE:
            split_pos = repair["split_pos"]
            tgt_line = self.corpus.target_lines[tgt_idx]
            tgt1_text = tgt_line.text[:split_pos].strip()
            tgt2_text = tgt_line.text[split_pos:].strip()

            # Create new lines
            tgt1 = self.corpus._create_or_get_line(tgt1_text, is_source=False)
            tgt1.original_line_number = tgt_line.original_line_number

            tgt2 = self.corpus._create_or_get_line(tgt2_text, is_source=False)
            tgt2.original_line_number = tgt_line.original_line_number

            # Replace in corpus (caller handles index rebuild)
            self.corpus.target_lines = (
                self.corpus.target_lines[:tgt_idx]
                + [tgt1, tgt2]
                + self.corpus.target_lines[tgt_idx + 1 :]
            )

        elif repair["type"] == RepairType.INSERT_LINE:
            # Insert placeholder using corresponding SOURCE FILE text (original Chinese/source language)
            # For SOURCE_AHEAD (2:1) repair: src_idx1 and src_idx2 map to tgt_idx
            # We insert a placeholder after tgt_idx for the second source line
            # Use the original source text (not translation, but actual source file content)
            src_idx2_line = self.corpus.source_lines[src_idx2]
            # Get the original source file text using the source line's original line number
            placeholder_text = (
                src_idx2_line.text
            )  # This is the original source (Chinese) text
            new_line = self.corpus._create_or_get_line(
                placeholder_text, is_source=False
            )
            # Mark original line number as -1 to indicate synthetic line
            new_line.original_line_number = -1

            # Insert after current target (caller handles index rebuild)
            self.corpus.target_lines = (
                self.corpus.target_lines[: tgt_idx + 1]
                + [new_line]
                + self.corpus.target_lines[tgt_idx + 1 :]
            )

    def _apply_merge_to_corpus(self, merge: Dict, tgt_idx1: int, tgt_idx2: int):
        """
        Apply merge repair to corpus (merge two target lines into one).

        NOTE: Caller is responsible for rebuilding indices via _rebuild_target_indices()
        after this method returns.
        """

        if merge["type"] == RepairType.MERGE_LINES:
            tgt1_line = self.corpus.target_lines[tgt_idx1]
            tgt2_line = self.corpus.target_lines[tgt_idx2]

            # Merge text with space separator
            merged_text = f"{tgt1_line.text} {tgt2_line.text}"

            # Create merged line
            merged_line = self.corpus._create_or_get_line(merged_text, is_source=False)
            merged_line.original_line_number = tgt1_line.original_line_number

            # Replace in corpus (caller handles index rebuild)
            self.corpus.target_lines = (
                self.corpus.target_lines[:tgt_idx1]
                + [merged_line]
                + self.corpus.target_lines[tgt_idx2 + 1 :]
            )

    def _count_operations(self, alignment_steps: List[AlignmentStep]) -> Dict[str, int]:
        """Count operations by type"""
        counts = {}
        for step in alignment_steps:
            op = step.operation.value
            counts[op] = counts.get(op, 0) + 1
        return counts

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

    def _create_empty_result(self) -> Dict:
        """Create empty result dict"""
        return {
            "repaired_lines": [line.text for line in self.corpus.target_lines],
            "repair_logs": [],
            "stats": {
                "total_repairs": 0,
                "source_content_lines": len(self.corpus.source_lines),
                "target_line_count": len(self.corpus.target_lines),
                "dp_time_seconds": 0,
                "postprocess_time_seconds": 0,
                "total_time_seconds": 0,
                "dp_operations": {},
            },
            "state": {"exceptions": ["DP alignment failed"]},
        }
