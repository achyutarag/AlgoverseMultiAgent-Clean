"""
Analyze correlation between query oscillation and F1/EM from a diffusion debug report.

Usage:
    python tools/oscillation_correlation.py --report results/diffusion_debug_report_YYYYMMDD_HHMMSS.json
"""

import argparse
import json
from typing import List, Dict, Any


def load_report(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def summarize_bucket(name: str, items: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not items:
        return {"bucket": name, "count": 0}
    f1s = [r["f1"] for r in items]
    ems = [r["em"] for r in items]
    return {
        "bucket": name,
        "count": len(items),
        "avg_f1": round(sum(f1s) / len(f1s), 3),
        "avg_em": round(sum(ems) / len(ems), 3),
        "avg_osc": round(sum(r["avg_osc"] for r in items) / len(items), 3),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        required=True,
        help="Path to diffusion_debug_report_*.json",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Oscillation threshold for high vs low (default: 0.5)",
    )
    args = parser.parse_args()

    data = load_report(args.report)
    rows = []
    for t in data.get("traces", []):
        osc_vals = []
        for hop in t.get("hops", []):
            osc = hop.get("debug_metadata", {}).get("oscillation", {})
            across = osc.get("across_hop")
            if across and across.get("token_jaccard_distance") is not None:
                osc_vals.append(across["token_jaccard_distance"])
        if osc_vals:
            rows.append(
                {
                    "question": t.get("question", "")[:80],
                    "f1": t.get("f1", 0.0),
                    "em": t.get("em", 0.0),
                    "avg_osc": sum(osc_vals) / len(osc_vals),
                    "max_osc": max(osc_vals),
                    "count_hops": len(osc_vals),
                }
            )

    TH = args.threshold
    high = [r for r in rows if r["avg_osc"] > TH]
    low = [r for r in rows if r["avg_osc"] <= TH]

    print("Summary by oscillation bucket (threshold =", TH, ")")
    for bucket in (summarize_bucket("high_osc", high), summarize_bucket("low_osc", low)):
        print(bucket)

    print("\nTop 5 highest-oscillation questions:")
    for r in sorted(rows, key=lambda x: x["avg_osc"], reverse=True)[:5]:
        print(f"{r['avg_osc']:.2f}  F1={r['f1']:.2f}  EM={r['em']:.2f}  {r['question']}")


if __name__ == "__main__":
    main()

