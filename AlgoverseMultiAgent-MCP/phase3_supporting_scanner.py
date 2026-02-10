"""
Phase 3 Supporting Facts Scanner

Scans the first N MuSiQue examples and reports whether the gold answer
appears in supporting paragraphs (is_supporting=True).
"""

import argparse
import os
import sys

# Ensure MCP folder is first on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.musique_document_loader import _load_musique_from_github


def scan_supporting(split: str, limit: int, out_path: str | None):
    examples = _load_musique_from_github(split)
    total = min(limit, len(examples))
    found_indices = []

    for i in range(total):
        ex = examples[i]
        answer = (ex.get("answer") or "").strip().lower()
        if not answer:
            continue
        paragraphs = ex.get("paragraphs", []) or []
        supporting = [p for p in paragraphs if isinstance(p, dict) and p.get("is_supporting")]
        found = any(answer in (p.get("paragraph_text") or "").lower() for p in supporting)

        status = "FOUND" if found else "MISSING"
        question = (ex.get("question") or "")[:80]
        print(f"{status}\tidx={i}\tq={question}")
        if found:
            found_indices.append(i)

    print(f"\nSummary: FOUND {len(found_indices)}/{total} examples")
    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            for idx in found_indices:
                f.write(f"{idx}\n")
        print(f"✅ Wrote indices to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Scan MuSiQue supporting paragraphs for gold answer.")
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--out", type=str, default="supporting_found_indices.txt")
    args = parser.parse_args()

    scan_supporting(args.split, args.limit, args.out)


if __name__ == "__main__":
    main()
