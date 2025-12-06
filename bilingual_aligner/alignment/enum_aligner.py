"""
Single-Stage Dynamic Programming Algorithm for Bilingual Text Alignment.

This implementation computes a stable set of reachable nodes using hard
bandwidth constraints and node-level soft filtering, then runs a standard
DAG DP (level-by-level) to obtain a single optimal alignment path that
maximizes the total similarity (sum of edge weights). Average similarity
is computed after path extraction for reporting. Although earlier design
notes reference enumeration and pruning theorems, this version performs
a single-pass DP with parent pointers and returns the optimal path;
enumeration/branch-and-bound style enumeration and early-stop logic are
not implemented here.

Core Design:
1. Phase 0: Compute stable reachable node set with:
    - Hard constraints: ceil(i/2) <= j <= 2*i, ceil((n-i)/2) <= m-j <= 2*(n-i)
    - Node-level soft constraints based on per-line 1:1 best similarities
    - Forward/backward reachability filtering until convergence

2. Phase 1: Compute edge weights and run standard DAG DP (level order)
    to obtain a single best path (max total similarity) and convert it to
    alignment operations. Final structural checks are applied afterwards.

Notes:
- Node-level soft filtering treats all operation types uniformly.
- This file documents and implements the current DP behavior (single
  optimal path). If enumeration/early-stopping is desired in future,
  extend the DP stage accordingly.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Dict, Tuple, Set
from math import ceil
import logging
import numpy as np


class AlignmentError(Exception):
    """Exception raised for alignment quality issues"""

    pass


class ConsecutiveNonOneToOneError(AlignmentError):
    """Exception raised when consecutive different non-1:1 operations are detected"""

    def __init__(
        self,
        op1_idx: int,
        op2_idx: int,
        op1_type: Tuple[int, int],
        op2_type: Tuple[int, int],
        distance: int,
    ):
        self.op1_idx = op1_idx
        self.op2_idx = op2_idx
        self.op1_type = op1_type
        self.op2_type = op2_type
        self.distance = distance
        super().__init__(
            f"Consecutive different non-1:1 operations detected: "
            f"op{op1_idx} ({op1_type[0]}:{op1_type[1]}) and op{op2_idx} ({op2_type[0]}:{op2_type[1]}) "
            f"at distance {distance}"
        )


class OperationType(Enum):
    """Three possible alignment operations"""

    NO_SHIFT = "NO_SHIFT"  # 1:1 mapping
    TARGET_SPLIT = "TARGET_SPLIT"  # 1:2 mapping (1 source, 2 target)
    SOURCE_AHEAD = "SOURCE_AHEAD"  # 2:1 mapping (2 source, 1 target)


@dataclass
class AlignmentStep:
    """One step in the alignment path"""

    operation: OperationType
    src_start: int
    src_end: int
    tgt_start: int
    tgt_end: int
    score: float  # Score of this step


@dataclass
class EnumPruningAligner:
    """
    Single-Stage Dynamic Programming algorithm for bilingual alignment.

    Implements node-level soft constraints and a standard DAG DP to find
    the single best alignment path (maximizing total similarity — the sum of edge weights).
    Average similarity is computed after path extraction for reporting.
    Soft constraints treat all operation types uniformly and do not
    implement operation-type penalties here.
    """

    # Soft constraint parameters (Optimized v2.1)
    # CONSECUTIVE_NON_1TO1_PENALTY_MAX removed - final structural validation will raise exception when applicable
    CONSECUTIVE_NON_1TO1_LOOKAHEAD = 5  # Lookahead steps for consecutive detection
    # Node-level relative threshold (fraction of best 1:1 score for involved lines)
    NODE_RELATIVE_THRESHOLD = 0.8

    def __init__(self, corpus, config):
        self.corpus = corpus
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.src_len = len(corpus.source_lines)
        self.tgt_len = len(corpus.target_lines)

        # Configuration parameters
        # Removed legacy non_one_to_one_raw_threshold; node-level filtering used instead
        # Removed consecutive_non_1to1_penalty_max - final structural validation will report violations
        self.consecutive_non_1to1_lookahead = config.get(
            "consecutive_non_1to1_lookahead", self.CONSECUTIVE_NON_1TO1_LOOKAHEAD
        )
        self.node_relative_threshold = config.get(
            "node_relative_threshold", self.NODE_RELATIVE_THRESHOLD
        )
        # Collect structural exceptions found during validation (no longer raise)
        self.structural_exceptions: List[str] = []

    def run(self) -> List[AlignmentStep]:
        """
        Execute single-stage DP algorithm with fair treatment of all operation types.

        Maximizes total similarity (sum of edge weights) without discriminating between 1:1, 2:1, and 1:2 operations.
        Note: average similarity is computed after path extraction for reporting; the DP objective is total sum.

        Returns:
                List of alignment steps (complete path from source 0 to src_len, target 0 to tgt_len)
        """
        self.logger.info(
            f"Starting Fair Single-Stage DP Algorithm...\n"
            f"Source: {self.src_len} lines, Target: {self.tgt_len} lines"
        )

        # Stage 0: Compute hard and soft constraint nodes
        self.logger.info("\n" + "=" * 70)
        self.logger.info("Stage 0: Compute Constraint Nodes")
        self.logger.info("=" * 70)

        hard_nodes = self._compute_hard_constraint_nodes()
        self.logger.info(f"Hard constraint nodes |S_all|: {len(hard_nodes)}")

        soft_nodes = self._apply_soft_constraints(hard_nodes)
        self.logger.info(f"After soft constraints |S'|: {len(soft_nodes)}")

        stable_nodes = self._compute_stable_reachable_set(soft_nodes)
        self.logger.info(f"Stable reachable set |S*|: {len(stable_nodes)}")

        if len(stable_nodes) == 0:
            self.logger.error("No stable nodes found!")
            return []

        # Compute edge weights with soft constraints
        self.logger.info("Computing edge weights...")
        edges = self._compute_edge_weights(stable_nodes)
        self.logger.info(f"Computed {len(edges)} edges")

        best_ops, search_stats = self._run_stage1_dp_single_path(
            stable_nodes, edges
        )

        # Compute and log node ratio statistics for selected path nodes
        selected_nodes = self._extract_path_nodes(best_ops)
        node_ratio_stats = self._compute_node_ratios_stats(selected_nodes, hard_nodes)

        total_score = sum(op.score for op in best_ops)
        avg_sim = total_score / len(best_ops) if best_ops else 0.0

        self.logger.info(f"✓ Optimal path found")
        self.logger.info(f"  Operations: {len(best_ops)}")
        self.logger.info(f"  Total score: {total_score:.6f}")
        self.logger.info(f"  Average similarity: {avg_sim:.6f}")
        self.logger.info(f"  Paths checked: {search_stats['paths_checked']}")
        self.logger.info(f"  Early stop supported: {search_stats['early_stop']}")

        # Final validation: raise if final operations contain consecutive different non-1:1
        # within lookahead distance (this moves rejecting to final check).
        self._validate_final_ops_raise_on_consecutive_non_1to1(best_ops)

        # Collect DP debug statistics to return alongside operations
        dp_stats = {
            "S_all": len(hard_nodes),
            "S_prime": len(soft_nodes),
            "S_star": len(stable_nodes),
            "edges_count": len(edges),
            "selected_path_nodes": len(selected_nodes),
        }

        # Merge node ratio stats if available
        if node_ratio_stats:
            dp_stats.update(node_ratio_stats)

        # Return results in compatible format (including dp_stats)
        return {
            "stage1": best_ops,
            "stage2": best_ops,  # For backward compatibility
            "T_max": avg_sim,
            "dp_stats": dp_stats,
            "structural_exceptions": self.structural_exceptions,
        }

    def _compute_hard_constraint_nodes(self) -> Set[Tuple[int, int]]:
        """Compute all nodes satisfying hard constraints"""
        hard_nodes = set()

        for i in range(self.src_len + 1):
            for j in range(self.tgt_len + 1):
                # Start reachability
                start_lower = ceil(i / 2)
                start_upper = 2 * i

                if not (start_lower <= j <= start_upper):
                    continue

                # End reachability
                end_lower = ceil((self.src_len - i) / 2)
                end_upper = 2 * (self.src_len - i)
                remaining_tgt = self.tgt_len - j

                if not (end_lower <= remaining_tgt <= end_upper):
                    continue

                hard_nodes.add((i, j))

        return hard_nodes

    def _apply_soft_constraints(
        self, hard_nodes: Set[Tuple[int, int]]
    ) -> Set[Tuple[int, int]]:
        """
        Apply soft constraints at node level.

        New rule (node-level):
        - Compute 1:1 (single-line) raw similarities for all possible 1:1 nodes.
        - For each source line index s, record `src_best[s]` = max 1:1 score where source==s.
        - For each target line index t, record `tgt_best[t]` = max 1:1 score where target==t.
        - For each node (i,j) compute the node score = max raw similarity across its possible
          outgoing operations (1:1, 2:1, 1:2).
                - For the node, take `element_best` = min(src_best[i] if i < src_len, tgt_best[j] if j < tgt_len)
                    (if both sides available). If only one side is available, use that side's best.
        - If node_score < node_relative_threshold * element_best, disable (filter out) this node.

        After this node filtering, the stable reachable set computation will be applied by the
        caller (`run`) via forward/backward filtering.
        This moves the earlier non-1:1 edge-level hard filter into a node-level soft filter.
        """
        nodes = set(hard_nodes)

        # If no nodes or trivial, return copy
        if not nodes:
            return set()

        src_best = [0.0] * self.src_len
        tgt_best = [0.0] * self.tgt_len

        # Compute 1:1 (single-line) raw similarities for all possible 1:1 nodes
        for i, j in list(nodes):
            # only consider valid 1:1 operation if next indices within bounds
            if i < self.src_len and j < self.tgt_len and (i + 1, j + 1) in nodes:
                raw = self._score_operation(i, i, j, j)
                if raw is None:
                    raw = 0.0
                if raw > src_best[i]:
                    src_best[i] = raw
                if raw > tgt_best[j]:
                    tgt_best[j] = raw

        # Now compute node scores (max over possible outgoing ops) and filter nodes
        nodes_to_remove = set()

        for i, j in list(nodes):
            # Compute node score: best outgoing operation raw similarity
            best_out_raw = 0.0
            for di, dj in [(1, 1), (2, 1), (1, 2)]:
                ni, nj = i + di, j + dj
                if (ni, nj) not in nodes:
                    continue
                # score operation from i..ni-1, j..nj-1
                raw = self._score_operation(i, ni - 1, j, nj - 1)
                if raw is None:
                    raw = 0.0
                if raw > best_out_raw:
                    best_out_raw = raw

            # Determine element_best from involved single-line bests.
            # New policy: use the MIN of the two per-element bests when both
            # are available; if only one side is available, use that side's best.
            if i < self.src_len and j < self.tgt_len:
                element_best = min(src_best[i], tgt_best[j])
            elif i < self.src_len:
                element_best = src_best[i]
            elif j < self.tgt_len:
                element_best = tgt_best[j]
            else:
                element_best = 0.0

            # If element_best is zero or negative, we cannot judge; keep the node
            if element_best <= 0.0:
                continue

            if best_out_raw < self.node_relative_threshold * element_best:
                nodes_to_remove.add((i, j))

        if nodes_to_remove:
            for n in nodes_to_remove:
                nodes.discard(n)

        return nodes

    def _compute_stable_reachable_set(
        self, soft_nodes: Set[Tuple[int, int]], max_iterations: int = 20
    ) -> Set[Tuple[int, int]]:
        """Compute stable reachable set via forward-backward filtering"""
        current = soft_nodes.copy()

        for _ in range(max_iterations):
            forward = self._compute_forward_reachability(current)
            backward = self._compute_backward_reachability(current)
            stable = forward & backward & current

            if stable == current:
                break

            current = stable

        return current

    def _extract_path_nodes(
        self, operations: List[AlignmentStep]
    ) -> List[Tuple[int, int]]:
        """Extract the sequence of nodes visited in the optimal path"""
        nodes = [(0, 0)]  # Start node
        current_i, current_j = 0, 0

        for op in operations:
            # Each operation advances by (src_end - src_start + 1, tgt_end - tgt_start + 1)
            di = op.src_end - op.src_start + 1
            dj = op.tgt_end - op.tgt_start + 1
            current_i += di
            current_j += dj
            nodes.append((current_i, current_j))

        return nodes

    def _compute_node_ratios_stats(
        self, selected_nodes: List[Tuple[int, int]], hard_nodes: Set[Tuple[int, int]]
    ):
        """Compute and log statistics of node ratios (node_score / element_best) for threshold tuning"""
        if not selected_nodes:
            return

        # Compute src_best and tgt_best from hard_nodes (same as in _apply_soft_constraints)
        src_best = [0.0] * self.src_len
        tgt_best = [0.0] * self.tgt_len

        for i, j in list(hard_nodes):
            if i < self.src_len and j < self.tgt_len and (i + 1, j + 1) in hard_nodes:
                raw = self._score_operation(i, i, j, j)
                if raw is None:
                    raw = 0.0
                if raw > src_best[i]:
                    src_best[i] = raw
                if raw > tgt_best[j]:
                    tgt_best[j] = raw

        # Compute ratios for selected path nodes
        ratios = []
        for i, j in selected_nodes:
            # Compute node score: best outgoing operation raw similarity
            best_out_raw = 0.0
            for di, dj in [(1, 1), (2, 1), (1, 2)]:
                ni, nj = i + di, j + dj
                if (ni, nj) not in hard_nodes:  # Use hard_nodes as possible next nodes
                    continue
                raw = self._score_operation(i, ni - 1, j, nj - 1)
                if raw is None:
                    raw = 0.0
                if raw > best_out_raw:
                    best_out_raw = raw

            # Determine element_best
            # New policy: element_best is the MIN of the per-element bests when
            # both are available; otherwise fall back to the available side.
            if i < self.src_len and j < self.tgt_len:
                element_best = min(src_best[i], tgt_best[j])
            elif i < self.src_len:
                element_best = src_best[i]
            elif j < self.tgt_len:
                element_best = tgt_best[j]
            else:
                element_best = 0.0

            if element_best > 0.0:
                ratio = best_out_raw / element_best
                ratios.append(ratio)

        if ratios:
            ratios_array = np.array(ratios)
            mean_ratio = np.mean(ratios_array)
            std_ratio = np.std(ratios_array)
            min_ratio = np.min(ratios_array)
            max_ratio = np.max(ratios_array)
            percentile_1 = np.percentile(ratios_array, 1)
            percentile_5 = np.percentile(ratios_array, 5)
            percentile_10 = np.percentile(ratios_array, 10)
            percentile_25 = np.percentile(ratios_array, 25)
            percentile_50 = np.percentile(ratios_array, 50)
            percentile_75 = np.percentile(ratios_array, 75)
            percentile_95 = np.percentile(ratios_array, 95)
            percentile_99 = np.percentile(ratios_array, 99)

            # Stage 1: Find optimal path (DP maximizes total similarity; average reported afterwards)
            self.logger.info("\n" + "=" * 70)
            self.logger.info("Stage 1: Find Optimal Path (Maximize Total Similarity; average reported afterwards)")
            self.logger.info("=" * 70)
            self.logger.info(f"Number of selected path nodes: {len(ratios)}")
            self.logger.info(f"Mean ratio: {mean_ratio:.4f}")
            self.logger.info(f"Std ratio: {std_ratio:.4f}")
            self.logger.info(f"Min ratio: {min_ratio:.4f}")
            self.logger.info(f"Max ratio: {max_ratio:.4f}")
            self.logger.info(f"1% percentile: {percentile_1:.4f}")
            self.logger.info(f"5% percentile: {percentile_5:.4f}")
            self.logger.info(f"95% percentile: {percentile_95:.4f}")
            self.logger.info(f"99% percentile: {percentile_99:.4f}")
            self.logger.info(f"Current threshold: {self.node_relative_threshold}")
            self.logger.info(
                f"Nodes below current threshold: {np.sum(ratios_array < self.node_relative_threshold)} ({100 * np.sum(ratios_array < self.node_relative_threshold) / len(ratios):.1f}%)"
            )

            # Return collected stats for external consumption
            return {
                "selected_nodes_count": len(ratios),
                "mean_ratio": round(float(mean_ratio), 4),
                "std_ratio": round(float(std_ratio), 4),
                "min_ratio": round(float(min_ratio), 4),
                "max_ratio": round(float(max_ratio), 4),
                "1%_percentile": round(float(percentile_1), 4),
                "5%_percentile": round(float(percentile_5), 4),
                "10%_percentile": round(float(percentile_10), 4),
                "25%_percentile": round(float(percentile_25), 4),
                "50%_percentile": round(float(percentile_50), 4),
                "75%_percentile": round(float(percentile_75), 4),
                "95%_percentile": round(float(percentile_95), 4),
                "99%_percentile": round(float(percentile_99), 4),
                "current_threshold": float(self.node_relative_threshold),
            }

        # No ratios collected -> return empty dict
        return {}

    def _compute_forward_reachability(
        self, nodes: Set[Tuple[int, int]]
    ) -> Set[Tuple[int, int]]:
        """Compute forward-reachable nodes"""
        forward = set()
        queue = [(0, 0)]
        visited = {(0, 0)}

        while queue:
            i, j = queue.pop(0)
            forward.add((i, j))

            for di, dj in [(2, 1), (1, 2), (1, 1)]:
                ni, nj = i + di, j + dj

                if (ni, nj) in nodes and (ni, nj) not in visited:
                    visited.add((ni, nj))
                    queue.append((ni, nj))

        return forward

    def _compute_backward_reachability(
        self, nodes: Set[Tuple[int, int]]
    ) -> Set[Tuple[int, int]]:
        """Compute backward-reachable nodes"""
        backward = set()
        queue = [(self.src_len, self.tgt_len)]
        visited = {(self.src_len, self.tgt_len)}

        while queue:
            i, j = queue.pop(0)
            backward.add((i, j))

            for di, dj in [(2, 1), (1, 2), (1, 1)]:
                pi, pj = i - di, j - dj

                if (pi, pj) in nodes and (pi, pj) not in visited:
                    visited.add((pi, pj))
                    queue.append((pi, pj))

        return backward

    def _compute_edge_weights(
        self,
        stable_nodes: Set[Tuple[int, int]],
    ) -> Dict[Tuple[Tuple[int, int], Tuple[int, int]], float]:
        """
        Compute edge weights uniformly for all operation types.

        All edges kept with their actual similarity scores.
        No discrimination by operation type (1:1, 2:1, 1:2).
        """
        edges = {}
        nodes_list = sorted(list(stable_nodes))

        for i, j in nodes_list:
            for di, dj in [(2, 1), (1, 2), (1, 1)]:
                ni, nj = i + di, j + dj

                if (ni, nj) not in stable_nodes:
                    continue

                # Score operation: average similarity across all pairs
                raw_weight = self._score_operation(i, ni - 1, j, nj - 1)

                # Keep all edges uniformly
                final_weight = raw_weight

                edges[((i, j), (ni, nj))] = final_weight

        return edges

    def _score_operation(
        self, src_start: int, src_end: int, tgt_start: int, tgt_end: int
    ) -> float:
        """Calculate average similarity score for an operation"""
        src_lines = self.corpus.source_lines[src_start : src_end + 1]
        tgt_lines = self.corpus.target_lines[tgt_start : tgt_end + 1]

        total_sim = 0.0
        count = 0

        for src in src_lines:
            for tgt in tgt_lines:
                total_sim += self.corpus.get_similarity(
                    src, tgt, use_punctuation_weight=False
                )
                count += 1

        return total_sim / count if count > 0 else 0.0

    def _extract_operations(
        self,
        path: List[Tuple[int, int]],
        edges: Dict[Tuple[Tuple[int, int], Tuple[int, int]], float],
    ) -> List[AlignmentStep]:
        """Extract operations from path"""
        operations = []

        for idx in range(len(path) - 1):
            from_node = path[idx]
            to_node = path[idx + 1]

            i, j = from_node
            ni, nj = to_node
            di, dj = ni - i, nj - j

            edge_key = (from_node, to_node)
            score = edges.get(edge_key, 0.0)

            # Determine operation type
            if di == 1 and dj == 1:
                op_type = OperationType.NO_SHIFT
            elif di == 2 and dj == 1:
                op_type = OperationType.SOURCE_AHEAD
            elif di == 1 and dj == 2:
                op_type = OperationType.TARGET_SPLIT
            else:
                continue

            operations.append(
                AlignmentStep(
                    operation=op_type,
                    src_start=i,
                    src_end=ni - 1,
                    tgt_start=j,
                    tgt_end=nj - 1,
                    score=score,
                )
            )

        return operations

    def _is_one_to_one(self, op: AlignmentStep) -> bool:
        """Check if operation is 1:1"""
        return op.operation == OperationType.NO_SHIFT

    def _validate_final_ops_raise_on_consecutive_non_1to1(
        self, ops: List[AlignmentStep]
    ) -> None:
        """
        Validate final operations and RECORD ConsecutiveNonOneToOne issues if
        consecutive different non-1:1 operations are found within lookahead.

        Changed behavior: do not raise; append a readable message to
        `self.structural_exceptions` for later reporting.
        """
        last_non_1to1_type = None
        last_non_1to1_idx = None

        for idx, op in enumerate(ops):
            is_non_1to1 = not self._is_one_to_one(op)

            if is_non_1to1:
                op_type = (
                    op.src_end - op.src_start + 1,
                    op.tgt_end - op.tgt_start + 1,
                )

                if (
                    last_non_1to1_type is not None
                    and last_non_1to1_idx is not None
                    and op_type != last_non_1to1_type
                ):
                    distance = idx - last_non_1to1_idx
                    if distance <= self.consecutive_non_1to1_lookahead:
                        # Build a readable message and record it instead of raising
                        try:
                            prev_op = ops[last_non_1to1_idx]
                            cur_op = op
                            msg = (
                                f"Consecutive different non-1:1 operations detected: "
                                f"op{last_non_1to1_idx} ({last_non_1to1_type[0]}:{last_non_1to1_type[1]}) "
                                f"and op{idx} ({op_type[0]}:{op_type[1]}) at distance {distance}; "
                                f"src[{prev_op.src_start}:{prev_op.src_end}]->tgt[{prev_op.tgt_start}:{prev_op.tgt_end}] vs "
                                f"src[{cur_op.src_start}:{cur_op.src_end}]->tgt[{cur_op.tgt_start}:{cur_op.tgt_end}]"
                            )
                        except Exception:
                            msg = (
                                f"Consecutive different non-1:1 operations: indices {last_non_1to1_idx} and {idx} "
                                f"types {last_non_1to1_type} vs {op_type} distance {distance}"
                            )

                        self.structural_exceptions.append(msg)
                        # Log as warning for visibility
                        try:
                            logging.getLogger(__name__).warning(msg)
                        except Exception:
                            pass

                last_non_1to1_type = op_type
                last_non_1to1_idx = idx

    def _run_stage1_dp_single_path(
        self,
        stable_nodes: Set[Tuple[int, int]],
        edges: Dict[Tuple[Tuple[int, int], Tuple[int, int]], float],
    ) -> Tuple[List[AlignmentStep], Dict]:
        """
        Stage 1: Standard DAG DP that computes the single best complete path.

        Implementation details:
        - Runs a level-ordered DP to compute `dp[(i,j)]` = maximum total score
          to reach node (i,j) using available `edges` on `stable_nodes`.
        - Maintains `parent` pointers to reconstruct the single best path to
          the target node `(src_len, tgt_len)` if available.
        - Converts the reconstructed node sequence into `AlignmentStep`s and
          returns that single best operation sequence.

        Note: This function returns only one optimal path (the path implied
        by the DP parent pointers). Enumeration of multiple complete paths
        or branch-and-bound early-stopping is NOT implemented here.

        Returns:
            (best_operations, {'paths_checked': int, 'early_stop': bool})
        """

        # Step 1: Run standard DP to get all scores
        dp = {}
        parent = {}
        dp[(0, 0)] = 0.0

        nodes_by_level = {}
        for i, j in stable_nodes:
            level = i + j
            if level not in nodes_by_level:
                nodes_by_level[level] = []
            nodes_by_level[level].append((i, j))

        for level in sorted(nodes_by_level.keys()):
            for i, j in nodes_by_level[level]:
                if (i, j) not in dp:
                    continue

                current_sum = dp[(i, j)]

                for di, dj in [(2, 1), (1, 2), (1, 1)]:
                    ni, nj = i + di, j + dj

                    if (ni, nj) not in stable_nodes:
                        continue

                    edge_key = ((i, j), (ni, nj))
                    if edge_key not in edges:
                        continue

                    weight = edges[edge_key]
                    new_sum = current_sum + weight

                    if (ni, nj) not in dp or new_sum > dp[(ni, nj)]:
                        dp[(ni, nj)] = new_sum
                        parent[(ni, nj)] = (i, j)

        # Step 2: Reconstruct and sort all COMPLETE paths (reaching target) by total score (descending)
        def reconstruct_path(node):
            """Reconstruct path from (0,0) to node"""
            path = []
            current = node
            while current in parent:
                path.append(current)
                current = parent[current]
            path.append((0, 0))
            path.reverse()
            return path

        all_paths_with_scores = []
        target_node = (self.src_len, self.tgt_len)

        # Only consider complete paths that reach the target
        if target_node in dp:
            path = reconstruct_path(target_node)
            score = dp[target_node]
            all_paths_with_scores.append((score, path, target_node))

        # Sort by score in descending order
        all_paths_with_scores.sort(key=lambda x: x[0], reverse=True)

        # Step 3: We have only one complete path; return it
        best_path = None
        paths_checked = 0
        early_stop_triggered = False

        if all_paths_with_scores:
            paths_checked = 1
            score, path_nodes = all_paths_with_scores[0][:2]

            # Convert node sequence to operations
            ops = self._extract_operations(path_nodes, edges)
            best_path = ops

        if best_path is None:
            # No complete path found - return empty list
            best_path = []

        return best_path, {
            "paths_checked": paths_checked,
            "early_stop": early_stop_triggered,
        }
