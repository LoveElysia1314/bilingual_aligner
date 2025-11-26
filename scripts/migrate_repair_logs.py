"""
Migrate old repair_logs.json (legacy) to the new compact schema defined in docs/repair_log_schema.md
Usage:
  python scripts/migrate_repair_logs.py path/to/repair_logs.json
Outputs: path/to/repair_logs.migrated.json
"""

import sys
import json
import os
from datetime import datetime


def load_old(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    if len(sys.argv) < 2:
        print("Usage: migrate_repair_logs.py path/to/repair_logs.json")
        return 1
    in_path = sys.argv[1]
    out_path = os.path.splitext(in_path)[0] + ".migrated.json"

    old = load_old(in_path)
    metadata = old.get("metadata", {})
    old_stats = old.get("stats", {})
    old_repairs = old.get("repairs", [])

    # Stats mapping with fallbacks
    total_repairs = old_stats.get("total_repairs", len(old_repairs))
    similarity_improvement = old_stats.get(
        "similarity_improvement", old_stats.get("improvement", 0.0)
    )
    avg_similarity_after = old_stats.get(
        "avg_similarity_after", old_stats.get("average_score", 0.0)
    )

    # processing time: try multiple keys
    # normalize potential dict or scalar values to floats
    def _to_num(v):
        if isinstance(v, dict):
            # accept {'total': x} or {'processing_time': {'total': x}}
            return float(v.get("total") or v.get("processing_time") or 0.0)
        try:
            return float(v or 0.0)
        except Exception:
            return 0.0

    total_time = _to_num(
        old_stats.get(
            "processing_time_seconds",
            old_stats.get("total_time_seconds", old_stats.get("processing_time", 0.0)),
        )
    )
    dp_time = _to_num(old_stats.get("dp_time_seconds", old_stats.get("dp_time", 0.0)))
    post_time = _to_num(
        old_stats.get(
            "postprocess_time_seconds", old_stats.get("postprocess_time", 0.0)
        )
    )
    encoding_time = max(0.0, total_time - dp_time - post_time)

    processing_time = {
        "total": round(float(total_time or 0.0), 4),
        "encoding": round(float(encoding_time or 0.0), 4),
        "dp": round(float(dp_time or 0.0), 4),
        "postprocess": round(float(post_time or 0.0), 4),
    }

    # file summary: support both flat keys and nested 'file_summary' structure
    if isinstance(old_stats.get("file_summary"), dict):
        fs = old_stats.get("file_summary")
        src_fs = fs.get("src", {})
        tgt_fs = fs.get("tgt", {})
        src_total = src_fs.get("file_lines", 0)
        src_content = src_fs.get("content_lines", 0)
        tgt_total = tgt_fs.get("file_lines", 0)
        tgt_total_after = tgt_fs.get("file_lines_after", src_total)
        tgt_before = tgt_fs.get("content_lines_before", tgt_fs.get("content_lines", 0))
        tgt_after = tgt_fs.get("content_lines_after", tgt_before)
    else:
        src_total = old_stats.get(
            "source_file_lines_total", old_stats.get("source_total_lines", 0)
        )
        src_content = old_stats.get(
            "source_content_lines", old_stats.get("source_content_lines", 0)
        )
        tgt_total = old_stats.get(
            "target_file_lines_total", old_stats.get("target_total_lines", 0)
        )
        tgt_total_after = old_stats.get("target_total_lines_after", src_total)
        tgt_before = old_stats.get(
            "target_content_lines_before",
            old_stats.get("target_content_lines_before", 0),
        )
        tgt_after = old_stats.get(
            "target_content_lines_after",
            old_stats.get("target_content_lines_after", tgt_before),
        )

    # produce compact file_summary only
    file_summary = {
        "src": f"src@{{{src_total}}}[{src_content}]",
        "tgt": f"tgt@{{{tgt_total}->{tgt_total_after}}}[{tgt_before}->{tgt_after}]",
    }

    # exceptions: try to convert any existing exceptions; otherwise create per rules
    exceptions = []
    old_ex = old_stats.get("exceptions", []) or old.get("exceptions", [])
    for e in old_ex:
        # if it's already structured, keep; else wrap
        if isinstance(e, dict) and "code" in e:
            exceptions.append(e)
        else:
            exceptions.append({"code": "LEGACY_EXCEPTION", "message": str(e)})

    # derive structured exceptions per schema
    src_content_val = src_content or 0
    repair_rate = (total_repairs / src_content_val) if src_content_val else 0.0
    if repair_rate > 0.05:
        exceptions.append(
            {
                "code": "REPAIR_RATE_EXCEEDED",
                "message": "Repair rate > 5%",
                "value": round(repair_rate, 4),
            }
        )
    if src_content_val != tgt_after:
        exceptions.append(
            {
                "code": "LINE_COUNT_MISMATCH",
                "message": "Source/target content lines mismatch",
                "value": {"src": src_content_val, "tgt": tgt_after},
            }
        )
    if avg_similarity_after and avg_similarity_after < 0.7:
        exceptions.append(
            {
                "code": "LOW_AVG_SIMILARITY",
                "message": "Average similarity after repairs < 0.7",
                "value": avg_similarity_after,
            }
        )

    # convert repairs
    new_repairs = []
    for idx, r in enumerate(old_repairs):
        # position conversion: try to reuse existing 'position' if present
        position = r.get("position") or ""
        score_before = (
            r.get("score_before")
            or r.get("similarity_before")
            or r.get("similarity_before")
        )
        score_after = (
            r.get("score_after")
            or r.get("similarity_after")
            or r.get("similarity_after")
        )
        rec = {
            "id": idx + 1,
            "type": r.get("type", "unknown"),
            "position": position,
            "score_before": round(float(score_before or 0.0), 4),
            "score_after": round(float(score_after or 0.0), 4),
            "line_change": r.get("line_change", 0),
        }
        # copy split_type if present
        if "split_type" in r:
            rec["split_type"] = r["split_type"]
        # include texts
        rec["source_text"] = r.get("source_text")
        rec["target_before"] = r.get("target_before")
        rec["target_after"] = r.get("target_after")
        new_repairs.append(rec)

    payload = {
        "metadata": metadata or {"generated_at": datetime.utcnow().isoformat() + "Z"},
        "config_snapshot": {},
        "stats": {
            "total_repairs": total_repairs,
            "similarity_improvement": similarity_improvement,
            "avg_similarity_after": avg_similarity_after,
            "processing_time": processing_time,
            "file_summary": file_summary,
            "repair_types": old_stats.get("repair_types", {}),
        },
        "exceptions": exceptions,
        "repairs": new_repairs,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Migrated: {in_path} -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
