"""Top-level shim: re-export aligner implementation from package aligners."""

from .aligners.enum_pruning_aligner import *  # noqa: F401,F403

        New rule (node-level):
        - Compute 1:1 (single-line) raw similarities for all possible 1:1 nodes.
        - For each source line index s, record `src_best[s]` = max 1:1 score where source==s.
        - For each target line index t, record `tgt_best[t]` = max 1:1 score where target==t.
        - For each node (i,j) compute the node score = max raw similarity across its possible
          outgoing operations (1:1, 2:1, 1:2).
        - For the node, take `element_best` = max(src_best[i] if i < src_len, tgt_best[j] if j < tgt_len).
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

            # Determine element_best from involved single-line bests
            element_best = 0.0
            if i < self.src_len:
                element_best = max(element_best, src_best[i])
            if j < self.tgt_len:
                element_best = max(element_best, tgt_best[j])

            # If element_best is zero, we cannot judge; keep the node
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

    def _run_stage1_maximize_score(
        self,
        stable_nodes: Set[Tuple[int, int]],
        edges: Dict[Tuple[Tuple[int, int], Tuple[int, int]], float],
    ) -> Tuple[float, List[Tuple[int, int]]]:
        """Stage 1: Find maximum average similarity"""
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

    def _validate_final_ops_raise_on_consecutive_non_1to1(
        self, ops: List[AlignmentStep]
    ) -> None:
        """
        Validate final operations and raise ConsecutiveNonOneToOneError if
        consecutive different non-1:1 operations are found within lookahead.
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
                        raise ConsecutiveNonOneToOneError(
                            last_non_1to1_idx,
                            idx,
                            last_non_1to1_type,
                            op_type,
                            distance,
                        )

                last_non_1to1_type = op_type
                last_non_1to1_idx = idx

    def _run_stage1_with_penalty_sorting(
        self,
        stable_nodes: Set[Tuple[int, int]],
        edges: Dict[Tuple[Tuple[int, int], Tuple[int, int]], float],
    ) -> Tuple[List[AlignmentStep], Dict]:
        """
        Single-stage DP with pruning theorem (v3.0).

        Algorithm:
        1. Run standard DP to get dp[(i,j)] = total_score
        2. Enumerate paths in order of score (descending)
        3. Track maximum average similarity seen
        4. EARLY STOP when no remaining path can beat current best
        5. Return best path found

        Mathematical Guarantee:
        - Paths sorted by score descending
        - For any remaining path: score ≤ previous score
        - Maximum possible avg_sim for remaining paths ≤ previous score / min_ops
        - If current best avg_sim > this upper bound, stop early

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
            score, path_nodes, final_node = all_paths_with_scores[0]
            
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

