"""
Phase 3 Hit@K Retrieval Audit

Runs retrieval-only over the first N MuSiQue examples and reports
the rank of the first passage containing the gold answer.
"""

import argparse
import os
import sys
import json
import asyncio
import unicodedata

# Ensure MCP folder is first on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.musique_document_loader import _load_musique_from_github, load_musique_example_context_as_documents
from agents.retriever_agent import RetrieverAgent


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return "".join(ch.lower() for ch in text if ch.isalnum() or ch.isspace())


def _find_answer_rank(docs: list, answer: str) -> int | None:
    ans = _normalize(answer.strip())
    if not ans:
        return None
    for idx, doc in enumerate(docs, start=1):
        text = _normalize(doc.get("page_content") or "")
        if ans in text:
            return idx
    return None


def _load_indices(path: str) -> list[int]:
    indices = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            try:
                indices.append(int(raw))
            except ValueError:
                continue
    return indices


async def run_hitk(split: str, limit: int, k: int, indices_file: str | None):
    examples = _load_musique_from_github(split)
    if indices_file:
        indices = [i for i in _load_indices(indices_file) if 0 <= i < len(examples)]
    else:
        indices = list(range(min(limit, len(examples))))
    total = len(indices)
    hit_counts = {5: 0, 10: 0, 20: 0, 50: 0}
    checked = 0

    for i in indices:
        ex = examples[i]
        question = ex.get("question", "")
        answer = ex.get("answer", "")
        if not question or not answer:
            continue

        documents = load_musique_example_context_as_documents(ex)
        retriever = RetrieverAgent(documents=documents)
        result = await retriever.process({"query": question, "k": k, "min_similarity": 0.0})
        payload = json.loads(result.content)
        docs = payload.get("documents", [])

        rank = _find_answer_rank(docs, answer)
        checked += 1

        if rank is not None:
            if rank <= 5:
                hit_counts[5] += 1
            if rank <= 10:
                hit_counts[10] += 1
            if rank <= 20:
                hit_counts[20] += 1
            if rank <= 50:
                hit_counts[50] += 1

        rank_str = str(rank) if rank is not None else "miss"
        print(f"idx={i}\trank={rank_str}\tq={question[:80]}")

    subset_label = f" (subset from {indices_file})" if indices_file else ""
    print(f"\nHit@K Summary (retrieval-only){subset_label}")
    for key in [5, 10, 20, 50]:
        rate = (hit_counts[key] / checked) if checked else 0.0
        print(f"Hit@{key}: {hit_counts[key]}/{checked} = {rate:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Hit@K audit for MuSiQue retrieval.")
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--limit", type=int, default=200,
                        help="Number of examples to scan if --indices-file is not provided.")
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--indices-file", type=str, default=None,
                        help="Optional file with example indices (one per line).")
    args = parser.parse_args()

    asyncio.run(run_hitk(args.split, args.limit, args.k, args.indices_file))


if __name__ == "__main__":
    main()
