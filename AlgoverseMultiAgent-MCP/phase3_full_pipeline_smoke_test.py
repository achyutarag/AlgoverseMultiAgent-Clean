"""
Phase 3 Full Pipeline Smoke Test

Runs a single MuSiQue example through the full pipeline to exercise:
- Breadcrumb scoping (StepDefiner + RetrieverAgent re-ranking)
- Context stitching (ExtractorAgent uses chunk relationships)
"""

import argparse
import asyncio
import logging
import os
import sys

# Ensure MCP folder is first on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets import load_dataset
from agents.mixed_model_orchestrator import run_optimized_marag_pipeline
from agents.musique_document_loader import (
    load_musique_example_context_as_documents,
    _load_musique_from_github,
)


def load_single_musique_example(split: str, example_index: int):
    """
    Load one MuSiQue example (question + paragraphs).
    Tries HuggingFace first, falls back to local JSONL.
    """
    try:
        dataset = load_dataset("allenai/musique-v1")
        split_data = dataset[split]
        return split_data[example_index]
    except Exception:
        examples = _load_musique_from_github(split)
        return examples[example_index]


async def run_full_smoke(split: str, example_index: int, sleep_seconds: int):
    example = load_single_musique_example(split, example_index)
    question = example.get("question", "").strip()
    if not question:
        raise ValueError("Loaded MuSiQue example missing 'question'")

    documents = load_musique_example_context_as_documents(example)
    gold_answer = example.get("answer", "")
    print(f"✅ Loaded example {example_index} from {split} with {len(documents)} documents")
    print(f"Question: {question}")
    if gold_answer:
        print(f"Gold answer: {gold_answer}")

    result = await run_optimized_marag_pipeline(question, documents=documents)
    final_answer = getattr(result, "final_answer", None) or getattr(result, "content", None)
    confidence = getattr(result, "confidence", None)

    print("\n✅ Pipeline completed")
    if final_answer:
        print(f"Answer: {final_answer}")
    if gold_answer:
        match = str(final_answer).strip().lower() == str(gold_answer).strip().lower()
        print(f"Exact match: {match}")
    if confidence is not None:
        print(f"Confidence: {confidence}")

    if sleep_seconds:
        print(f"\n⏳ Sleeping {sleep_seconds}s to avoid rate limits...")
        await asyncio.sleep(sleep_seconds)


def main():
    parser = argparse.ArgumentParser(description="Phase 3 full pipeline smoke test")
    parser.add_argument("--split", type=str, default="validation", help="MuSiQue split")
    parser.add_argument("--example-index", type=int, default=0, help="Example index to run")
    parser.add_argument("--sleep", type=int, default=0, help="Seconds to sleep after run")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    asyncio.run(run_full_smoke(args.split, args.example_index, args.sleep))


if __name__ == "__main__":
    main()
