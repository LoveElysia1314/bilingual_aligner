"""
Single-Stage Dynamic Programming for Bilingual Text Alignment

This implementation uses a single-stage DP approach:
- Find maximum average similarity path

Key features:
- Uses sentence-level encoding consistent with the project (get_normalized_embedding_by_sentences)
- Precomputes all line embeddings for efficiency
- Produces near-perfect 1:1 alignment while maintaining high semantic quality
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Dict, Tuple, Set
from math import ceil
import logging
import numpy as np


class OperationType(Enum):
    """Three possible alignment operations"""

    NO_SHIFT = "NO_SHIFT"  # 1:1 mapping
    TARGET_SPLIT = "TARGET_SPLIT"  # 1:2 mapping
    SOURCE_AHEAD = "SOURCE_AHEAD"  # 2:1 mapping


@dataclass
class AlignmentStep:
    """One step in the alignment path"""

    operation: OperationType
    src_start: int
    src_end: int
    tgt_start: int
    tgt_end: int
    score: float


class DPAligner:
    """
    Single-stage DP algorithm for optimal bilingual alignment.

    Configuration parameters:
    - min_quality_threshold (T_min): Minimum acceptable average similarity
    """

    # Configuration parameters
    MIN_QUALITY_THRESHOLD = 0.75  # T_min: minimum average similarity

    def __init__(self, corpus, config):
        self.corpus = corpus
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.src_len = len(corpus.source_lines)
        self.tgt_len = len(corpus.target_lines)

        # Configuration parameters
        self.min_quality_threshold = config.get(
            "min_quality_threshold", self.MIN_QUALITY_THRESHOLD
        )

    def run(self) -> List[AlignmentStep]:
        """
        Execute single-stage DP algorithm.

        Find maximum average similarity path

        Returns:
                List of alignment steps
        """
        self.logger.info(
            f"Starting Single-Stage DP Algorithm\n"
            f"Source: {self.src_len} lines, Target: {self.tgt_len} lines\n"
            f"Quality threshold T_min: {self.min_quality_threshold:.4f}"
        )

        # Compute hard and soft constraint nodes
        self.logger.info("\n" + "=" * 70)
        self.logger.info("Compute Constraint Nodes")
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

        # Precompute line embeddings
        self.logger.info("\nPrecomputing line embeddings...")
        src_embeddings, tgt_embeddings = self._precompute_embeddings()

        # Compute edge weights
        self.logger.info("Computing edge weights...")
        edges = self._compute_edge_weights(stable_nodes, src_embeddings, tgt_embeddings)
        self.logger.info(f"Computed {len(edges)} edges")

        # Find maximum average similarity
        self.logger.info("\n" + "=" * 70)
        self.logger.info("Find Maximum Average Similarity")
        self.logger.info("=" * 70)

        T_max, path = self._run_maximize_score(stable_nodes, edges)
        ops = self._extract_operations(path, edges)
        non_1to1 = sum(1 for op in ops if not self._is_one_to_one(op))

        self.logger.info(f"✓ Maximum T_max: {T_max:.6f}")
        self.logger.info(f"  Path: {len(ops)} operations ({non_1to1} non-1:1)")

        if T_max >= self.min_quality_threshold:
            self.logger.info(
                f"  ✓ Quality threshold satisfied ({T_max:.6f} >= {self.min_quality_threshold:.6f})"
            )
        else:
            self.logger.warning(
                f"  ⚠ Quality threshold NOT satisfied ({T_max:.6f} < {self.min_quality_threshold:.6f})"
            )

        # Return result
        return ops

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
        Apply soft constraints. Currently returns all hard constraint nodes.
        """
        return hard_nodes.copy()

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

    def _precompute_embeddings(
        self,
    ) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
        """Precompute normalized embeddings for all lines using sentence-level encoding"""
        src_embeddings = {}
        tgt_embeddings = {}

        # Ensure all lines have embeddings computed
        for i, line in enumerate(self.corpus.source_lines):
            line.ensure_features(self.corpus.text_processor)
            src_embeddings[i] = line.embedding

        for j, line in enumerate(self.corpus.target_lines):
            line.ensure_features(self.corpus.text_processor)
            tgt_embeddings[j] = line.embedding

        return src_embeddings, tgt_embeddings

    def _compute_edge_weights(
        self,
        stable_nodes: Set[Tuple[int, int]],
        src_embeddings: Dict[int, np.ndarray],
        tgt_embeddings: Dict[int, np.ndarray],
    ) -> Dict[Tuple[Tuple[int, int], Tuple[int, int]], float]:
        """
        Compute edge weights.
        """
        edges = {}
        nodes_list = sorted(list(stable_nodes))

        for i, j in nodes_list:
            for di, dj in [(2, 1), (1, 2), (1, 1)]:
                ni, nj = i + di, j + dj

                if (ni, nj) not in stable_nodes:
                    continue

                # Sum source embeddings
                src_combined = None
                for si in range(i, min(i + di, self.src_len)):
                    if si in src_embeddings:
                        emb = src_embeddings[si]
                        if src_combined is None:
                            src_combined = emb.copy()
                        else:
                            src_combined = src_combined + emb

                # Sum target embeddings
                tgt_combined = None
                for sj in range(j, min(j + dj, self.tgt_len)):
                    if sj in tgt_embeddings:
                        emb = tgt_embeddings[sj]
                        if tgt_combined is None:
                            tgt_combined = emb.copy()
                        else:
                            tgt_combined = tgt_combined + emb

                # Compute cosine similarity
                if src_combined is not None and tgt_combined is not None:
                    src_norm = np.linalg.norm(src_combined)
                    tgt_norm = np.linalg.norm(tgt_combined)

                    if src_norm > 0 and tgt_norm > 0:
                        src_normalized = src_combined / src_norm
                        tgt_normalized = tgt_combined / tgt_norm
                        raw_weight = float(np.dot(src_normalized, tgt_normalized))
                    else:
                        raw_weight = 0.0
                else:
                    raw_weight = 0.0

                edges[((i, j), (ni, nj))] = raw_weight

        return edges

    def _run_maximize_score(
        self,
        stable_nodes: Set[Tuple[int, int]],
        edges: Dict[Tuple[Tuple[int, int], Tuple[int, int]], float],
    ) -> Tuple[float, List[Tuple[int, int]]]:
        """Find maximum average similarity"""
        dp = {}
        parent = {}
        dp[(0, 0)] = 0.0

        # Process nodes in topological order
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

        # Reconstruct path
        path = []
        current = (self.src_len, self.tgt_len)
        while current in parent:
            path.append(current)
            current = parent[current]
        path.append((0, 0))
        path.reverse()

        # Calculate average similarity
        total_score = dp.get((self.src_len, self.tgt_len), 0.0)
        num_operations = len(path) - 1
        avg_similarity = total_score / num_operations if num_operations > 0 else 0.0

        return avg_similarity, path

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
