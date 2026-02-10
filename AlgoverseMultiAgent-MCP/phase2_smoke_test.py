"""
Phase 2 Smoke Test

Runs the full MA-RAG pipeline on a small set of questions to ensure:
- No crashes
- Phase 2 retrieval integration executes
"""

import argparse
import asyncio
import logging
import os
import sys

# Ensure the MCP folder is first on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.mixed_model_orchestrator import run_optimized_marag_pipeline
from agents.musique_document_loader import load_musique_context_as_documents


async def run_smoke_test(queries, sleep_seconds, documents):
    print("\n" + "=" * 80)
    print("PHASE 2 SMOKE TEST")
    print("=" * 80)

    for i, query in enumerate(queries, start=1):
        print("\n" + "-" * 80)
        print(f"[{i}/{len(queries)}] Query: {query}")
        print("-" * 80)
        try:
            result = await run_optimized_marag_pipeline(query, documents=documents)
            final_answer = getattr(result, "final_answer", None) or getattr(result, "content", None)
            confidence = getattr(result, "confidence", None)

            print("✅ Pipeline completed")
            if final_answer:
                print(f"Answer: {final_answer}")
            if confidence is not None:
                print(f"Confidence: {confidence}")
        except Exception as exc:
            print("❌ Pipeline failed")
            print(f"Error: {exc}")
            raise
        if sleep_seconds and i < len(queries):
            print(f"⏳ Sleeping {sleep_seconds}s to avoid rate limits...")
            await asyncio.sleep(sleep_seconds)


def main():
    parser = argparse.ArgumentParser(description="Phase 2 smoke test")
    parser.add_argument(
        "--queries",
        nargs="*",
        default=[
            "Where is the DLR headquarters located?",
            "Which country is Cologne in?",
            "What state is Tamaulipas in?"
        ],
        help="Queries to run through the pipeline"
    )
    parser.add_argument(
        "--sleep",
        type=int,
        default=0,
        help="Seconds to sleep between queries (rate-limit friendly)"
    )
    parser.add_argument(
        "--use-musique",
        action="store_true",
        help="Load MuSiQue dataset and use it as the retrieval corpus"
    )
    parser.add_argument(
        "--musique-split",
        type=str,
        default="validation",
        help="MuSiQue split to load (train/validation)"
    )
    parser.add_argument(
        "--musique-examples",
        type=int,
        default=200,
        help="Number of MuSiQue examples to load"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    documents = None
    if args.use_musique:
        print(f"Loading MuSiQue ({args.musique_split}) with {args.musique_examples} examples...")
        documents = load_musique_context_as_documents(
            dataset_split=args.musique_split,
            num_examples=args.musique_examples
        )
        print(f"✅ Loaded {len(documents)} MuSiQue documents")

    asyncio.run(run_smoke_test(args.queries, args.sleep, documents))


if __name__ == "__main__":
    main()
