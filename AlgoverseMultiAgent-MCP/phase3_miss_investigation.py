"""
Phase 3 Miss Investigation

For specific indices, checks:
- whether the gold answer appears in supporting paragraphs or anywhere in the example
- whether the gold answer appears in top-K retrieved docs
- prints top-K titles/snippets for manual inspection
"""

import argparse
import asyncio
import json
import os
import sys
import unicodedata

# Ensure MCP folder is first on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.musique_document_loader import (
    _load_musique_from_github,
    load_musique_example_context_as_documents,
)
from agents.retriever_agent import RetrieverAgent


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return "".join(ch.lower() for ch in text if ch.isalnum() or ch.isspace())


def _parse_indices(raw: str) -> list[int]:
    indices = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            indices.append(int(part))
        except ValueError:
            continue
    return indices


def _load_indices_file(path: str) -> list[int]:
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


async def investigate(split: str, indices: list[int], k: int):
    examples = _load_musique_from_github(split)

    for idx in indices:
        if idx < 0 or idx >= len(examples):
            print(f"\n== idx {idx} (out of range) ==")
            continue

        ex = examples[idx]
        question = ex.get("question", "")
        answer = ex.get("answer", "") or ""
        paras = ex.get("paragraphs", []) or []
        supporting = [p for p in paras if isinstance(p, dict) and p.get("is_supporting")]

        answer_norm = _normalize(answer)
        supporting_text = " ".join(p.get("paragraph_text", "") for p in supporting)
        all_text = " ".join(
            p.get("paragraph_text", "") for p in paras if isinstance(p, dict)
        )

        in_supporting = answer_norm in _normalize(supporting_text)
        in_any = answer_norm in _normalize(all_text)

        documents = load_musique_example_context_as_documents(ex)
        retriever = RetrieverAgent(documents=documents)
        result = await retriever.process({"query": question, "k": k, "min_similarity": 0.0})
        payload = json.loads(result.content)
        docs = payload.get("documents", [])

        rank = None
        for i, d in enumerate(docs, start=1):
            content = d.get("page_content", "")
            if answer_norm and answer_norm in _normalize(content):
                rank = i
                break

        print(f"\n== idx {idx} ==")
        print(f"question: {question}")
        print(f"answer: {answer}")
        print(f"answer_in_supporting: {in_supporting}")
        print(f"answer_in_any_paragraph: {in_any}")
        print(f"answer_rank_in_top{k}: {rank if rank is not None else 'miss'}")
        print("top-K previews:")
        for i, d in enumerate(docs, start=1):
            title = d.get("metadata", {}).get("title", "")
            snippet = (d.get("page_content", "") or "").replace("\n", " ")[:160]
            print(f"  {i}. {title} | {snippet}")


def main():
    parser = argparse.ArgumentParser(description="Investigate retrieval misses by index.")
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--indices", type=str, default="")
    parser.add_argument("--indices-file", type=str, default=None)
    parser.add_argument("--k", type=int, default=50)
    args = parser.parse_args()

    indices = []
    if args.indices:
        indices.extend(_parse_indices(args.indices))
    if args.indices_file:
        indices.extend(_load_indices_file(args.indices_file))

    indices = sorted(set(indices))
    if not indices:
        raise SystemExit("Provide --indices or --indices-file.")

    asyncio.run(investigate(args.split, indices, args.k))


if __name__ == "__main__":
    main()
