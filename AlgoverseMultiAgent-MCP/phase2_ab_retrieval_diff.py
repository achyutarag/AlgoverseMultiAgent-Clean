"""
Phase 2 A/B Retrieval Diff

Runs RetrieverAgent with and without breadcrumb_scope and prints the top results
to compare ordering and scores.
"""

import asyncio
import os
import sys

# Ensure the MCP folder is first on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from langchain.schema import Document
from agents.retriever_agent import RetrieverAgent


async def run_ab_diff():
    print("\n" + "=" * 80)
    print("PHASE 2 A/B RETRIEVAL DIFF")
    print("=" * 80)

    documents = [
        Document(
            page_content="DLR headquarters is located in Cologne, Germany.",
            metadata={"id": "doc_match", "breadcrumb_path": ["NASA", "Centers", "DLR"], "breadcrumb_confidence": 0.9}
        ),
        Document(
            page_content="Cologne is a city in Germany with a famous cathedral.",
            metadata={"id": "doc_city", "breadcrumb_path": ["Germany", "Cities"], "breadcrumb_confidence": 0.7}
        ),
        Document(
            page_content="Tamaulipas is a state in northeastern Mexico.",
            metadata={"id": "doc_other", "breadcrumb_path": ["Mexico", "States"], "breadcrumb_confidence": 0.7}
        )
    ]

    retriever = RetrieverAgent(
        documents=documents,
        model_config={"use_cuda": False},
        top_k=3,
        min_similarity=0.1
    )

    query = "Where is the DLR headquarters located?"
    print(f"Query: {query}")

    # Without scope
    no_scope = await retriever.process({
        "query": query,
        "k": 3,
        "min_similarity": 0.1,
        "breadcrumb_scope": None
    })

    # With scope
    with_scope = await retriever.process({
        "query": query,
        "k": 3,
        "min_similarity": 0.1,
        "breadcrumb_scope": ["NASA", "Centers"]
    })

    def print_results(label, response):
        docs = response.metadata.get("documents", [])
        print("\n" + label)
        print("-" * 60)
        for i, doc in enumerate(docs[:3], start=1):
            meta = doc.get("metadata", {})
            print(f"{i}. id={meta.get('id')} score={doc.get('score', 0.0):.3f} breadcrumb={meta.get('breadcrumb_path')}")

    print_results("A) No Scope", no_scope)
    print_results("B) With Scope", with_scope)

    print("\n✅ Compare ordering between A and B for Phase 2 verification")


def main():
    asyncio.run(run_ab_diff())


if __name__ == "__main__":
    main()
