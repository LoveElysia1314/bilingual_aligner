"""Unified output formatting for repair logs and console reports"""

from typing import Dict, Any, List, Optional
from statistics import stdev
import json


class OutputFormatter:
    """Formats repair results into unified schema and console output"""

    # Output levels
    OUTPUT_LEVELS = {"minimal", "normal", "verbose"}
    DEFAULT_LEVEL = "normal"

    @staticmethod
    def build_summary(
        stats: Dict[str, Any], repair_breakdown: Optional[Dict[str, int]] = None
    ) -> Dict[str, Any]:
        """Build summary layer from raw stats

        Args:
            stats: Raw statistics dict from executor
            repair_breakdown: Optional dict with repair type counts {split, merge, insert, delete}

        Returns:
            Structured summary dict with total_repairs, repair_breakdown, files info
        """
        src_total = stats.get("source_total_lines", 0)
        src_content = stats.get("source_content_lines", 0)
        tgt_total_before = stats.get("target_total_lines", 0)
        tgt_total_after = stats.get("target_total_lines_after", 0)
        tgt_before = stats.get("target_content_lines_before", 0)
        tgt_after = stats.get("target_content_lines_after", 0)

        # Use provided repair_breakdown or extract from stats
        if repair_breakdown is None:
            repair_breakdown = stats.get("dp_operations", {}) or {}
            if not repair_breakdown and "repair_types" in stats:
                # Fallback to repair_types if present
                repair_breakdown = stats.get("repair_types", {})

        return {
            "total_repairs": stats.get("total_repairs", 0),
            "repair_breakdown": {
                "split": repair_breakdown.get("split", 0),
                "merge": repair_breakdown.get("merge", 0),
                "insert": repair_breakdown.get("insert", 0),
                "delete": repair_breakdown.get("delete", 0),
            },
            "files": {
                "source": {
                    "total_lines": src_total,
                    "content_lines": src_content,
                },
                "target": {
                    "total_lines_before": tgt_total_before,
                    "total_lines_after": tgt_total_after,
                    "content_lines_before": tgt_before,
                    "content_lines_after": tgt_after,
                },
            },
        }

    @staticmethod
    def build_performance(stats: Dict[str, Any]) -> Dict[str, Any]:
        """Build performance layer from raw stats

        Args:
            stats: Raw statistics dict from executor

        Returns:
            Structured performance dict with timing, alignment_stats, dp_analysis
        """
        # Timing breakdown
        total_time = stats.get("total_time_seconds", 0.0)
        encoding_time = stats.get("encoding_time_seconds", 0.0)
        dp_time = stats.get("dp_time_seconds", 0.0)
        post_time = stats.get("postprocess_time_seconds", 0.0)

        timing_seconds = {
            "total": round(total_time, 4),
            "encoding": round(encoding_time, 4),
            "dp_alignment": round(dp_time, 4),
            "post_processing": round(post_time, 4),
        }

        # Alignment stats
        sim_stats = stats.get("similarity_statistics", {})
        if sim_stats:
            percentiles = {
                "p1": sim_stats.get("1%_percentile", 0),
                "p5": sim_stats.get("5%_percentile", 0),
                "p10": sim_stats.get("10%_percentile", 0),
                "p25": sim_stats.get("25%_percentile", 0),
                "p50": sim_stats.get("50%_percentile", 0),
                "p75": sim_stats.get("75%_percentile", 0),
            }
        else:
            percentiles = {
                "p1": 0,
                "p5": 0,
                "p10": 0,
                "p25": 0,
                "p95": 0,
                "p99": 0,
            }

        alignment_stats = {
            "total_aligned_pairs": sim_stats.get("count", 0),
                "similarity": {
                "mean": round(sim_stats.get("mean", 0), 4),
                "stdev": round(sim_stats.get("stdev", 0), 4),
                "min": round(sim_stats.get("min", 0), 4),
                "max": round(sim_stats.get("max", 0), 4),
                "count": sim_stats.get("count", 0),
                "percentiles": {k: round(v, 4) for k, v in percentiles.items()},
            },
        }

        # DP analysis
        dp_stats = stats.get("dp_stats", {})
        if dp_stats:
            dp_percentiles = {
                "p1": dp_stats.get("1%_percentile", 0),
                "p5": dp_stats.get("5%_percentile", 0),
                "p10": dp_stats.get("10%_percentile", 0),
                "p25": dp_stats.get("25%_percentile", 0),
                "p50": dp_stats.get("50%_percentile", 0),
                "p75": dp_stats.get("75%_percentile", 0),
            }
        else:
            dp_percentiles = {"p1": 0, "p5": 0, "p95": 0, "p99": 0}

        dp_analysis = {
            "total_candidates": dp_stats.get("S_all", 0),
            "pruned_candidates": dp_stats.get("S_prime", 0),
            "selected_candidates": dp_stats.get("S_star", 0),
            "selected_path": {
                "nodes_evaluated": dp_stats.get("selected_path_nodes", 0),
                "nodes_selected": dp_stats.get("selected_nodes_count", 0),
            },
                "quality_metrics": {
                "mean_ratio": round(dp_stats.get("mean_ratio", 0), 4),
                "std_ratio": round(dp_stats.get("std_ratio", 0), 4),
                "percentiles": {k: round(v, 4) for k, v in dp_percentiles.items()},
            },
        }

        return {
            "timing_seconds": timing_seconds,
            "alignment_stats": alignment_stats,
            "dp_analysis": dp_analysis,
        }

    @staticmethod
    def format_console(result: Dict[str, Any], level: str = "normal") -> str:
        """Format result as console output

        Args:
            result: Repair result dict
            level: Output level (minimal, normal, verbose)

        Returns:
            Formatted console output string
        """
        if level not in OutputFormatter.OUTPUT_LEVELS:
            level = OutputFormatter.DEFAULT_LEVEL

        lines = []

        if level == "minimal":
            return OutputFormatter._format_minimal(result)
        elif level == "normal":
            return OutputFormatter._format_normal(result)
        else:  # verbose
            return OutputFormatter._format_verbose(result)

    @staticmethod
    def _format_minimal(result: Dict[str, Any]) -> str:
        """Minimal console output - summary only"""
        summary = result.get("summary", {})
        total_repairs = summary.get("total_repairs", 0)
        output_dir = result.get("io", {}).get("output_base", "output/")

        lines = [
            f"[OK] Repairs completed: {total_repairs} repairs",
            f"     Saved to: {output_dir}",
        ]
        return "\n".join(lines)

    @staticmethod
    def _format_normal(result: Dict[str, Any]) -> str:
        """Normal console output - compact summary only"""
        lines = []

        # Repair stats
        summary = result.get("summary", {})
        breakdown = summary.get("repair_breakdown", {})
        total_repairs = summary.get("total_repairs", 0)
        src_lines = summary.get("files", {}).get("source", {}).get("content_lines", 0)

        if total_repairs == 0:
            lines.extend(
                [
                    "[OK] No repairs needed - alignment quality is good",
                ]
            )
        else:
            repair_rate = (total_repairs / src_lines * 100) if src_lines > 0 else 0
            repair_types = [t for t, c in breakdown.items() if c > 0]
            types_str = (
                " / ".join([f"{t}x{breakdown[t]}" for t in repair_types])
                if repair_types
                else "none"
            )

            lines.extend(
                [
                    f"[OK] Repairs completed: {total_repairs} repairs ({types_str}) - {repair_rate:.1f}%",
                ]
            )

        # Results saved message
        io_info = result.get("io", {})
        output_base = io_info.get("output_base", "output/")
        lines.append(f"     Saved to: {output_base}")

        return "\n".join(lines)

    @staticmethod
    def _format_verbose(result: Dict[str, Any]) -> str:
        """Verbose console output - normal + detailed stats + sample repairs"""
        # Start with minimal info
        lines = [OutputFormatter._format_normal(result).rstrip(""), ""]

        # Add detailed stats section
        summary = result.get("summary", {})
        breakdown = summary.get("repair_breakdown", {})
        src_lines = summary.get("files", {}).get("source", {}).get("content_lines", 0)

        files = summary.get("files", {})
        src = files.get("source", {})
        tgt = files.get("target", {})

        performance = result.get("performance", {})
        timing = performance.get("timing_seconds", {})
        alignment = performance.get("alignment_stats", {})
        sim = alignment.get("similarity", {})

        lines.extend(
            [
                "--- DETAILED STATS ---",
                f"  Files: source {src.get('total_lines', 0)} lines ({src.get('content_lines', 0)} content)",
                f"         target {tgt.get('total_lines_before', 0)}=>{tgt.get('total_lines_after', 0)} lines",
                f"  Similarity: mean={sim.get('mean', 0):.4f}, median={sim.get('median', 0):.4f}, range={sim.get('min', 0):.4f}~{sim.get('max', 0):.4f}",
                f"  Timing: total={timing.get('total', 0):.2f}s (encoding={timing.get('encoding', 0):.2f}s, dp={timing.get('dp_alignment', 0):.2f}s, post={timing.get('post_processing', 0):.2f}s)",
                "",
            ]
        )

        # Show repair details
        repairs = result.get("repairs", [])
        if repairs:
            lines.append("REPAIR DETAILS")
            for i, repair in enumerate(repairs[:5], 1):
                repair_type = repair.get("type", "unknown").upper()
                position = repair.get("position", "")
                split_type = repair.get("split_type", "")

                lines.append(f"  [{i}] {repair_type} {position}")
                if split_type:
                    lines.append(f"       type={split_type}")

                src_text = repair.get("source_text", "")
                tgt_before = repair.get("target_before", "")
                tgt_after = repair.get("target_after", "")

                if src_text:
                    if len(src_text) > 80:
                        src_text = src_text[:77] + "..."
                    lines.append(f"       src: {src_text}")

                if tgt_before and tgt_after:
                    if len(tgt_before) > 80:
                        tgt_before = tgt_before[:77] + "..."
                    if len(tgt_after) > 80:
                        tgt_after = tgt_after[:77] + "..."
                    lines.append(f"       bef: {tgt_before}")
                    lines.append(f"       aft: {tgt_after}")

            if len(repairs) > 5:
                lines.append(
                    f"  ... {len(repairs) - 5} more repairs in repair_logs.json"
                )

        return "\n".join(lines)

    @staticmethod
    def compute_similarity_statistics(similarities: List[float]) -> Dict[str, Any]:
        """Compute similarity statistics

        Args:
            similarities: List of similarity scores

        Returns:
            Dict with statistics
        """
        if not similarities:
            return {
                "count": 0,
                "mean": 0.0,
                "stdev": 0.0,
                "1%_percentile": 0.0,
                "5%_percentile": 0.0,
                "10%_percentile": 0.0,
                "25%_percentile": 0.0,
                "50%_percentile": 0.0,
                "75%_percentile": 0.0,
                "min": 0.0,
                "max": 0.0,
            }

        sims = sorted(similarities)
        n = len(sims)

        mean = sum(sims) / n

        # 50th percentile (median) will be obtained via percentile helper

        try:
            stdev_val = stdev(sims) if n > 1 else 0.0
        except Exception:
            stdev_val = 0.0

        def percentile(data: List[float], p: float) -> float:
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
            "stdev": round(stdev_val, 4),
            "1%_percentile": round(percentile(sims, 1), 4),
            "5%_percentile": round(percentile(sims, 5), 4),
            "10%_percentile": round(percentile(sims, 10), 4),
            "25%_percentile": round(percentile(sims, 25), 4),
            "50%_percentile": round(percentile(sims, 50), 4),
            "75%_percentile": round(percentile(sims, 75), 4),
            "min": round(min(sims), 4),
            "max": round(max(sims), 4),
        }
