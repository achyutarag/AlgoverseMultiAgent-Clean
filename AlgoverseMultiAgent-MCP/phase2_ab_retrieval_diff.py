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
        # Doc A (The Trap) - high semantic similarity but wrong breadcrumb
        Document(
            page_content="DLR Headquarters is in Mexico.",
            metadata={"id": "doc_trap", "breadcrumb_path": ["Mexico"], "breadcrumb_confidence": 0.9}
        ),
        # Doc B (The Truth) - medium semantic similarity with correct breadcrumb
        Document(
            page_content="The HQ is located in Cologne.",
            metadata={"id": "doc_truth", "breadcrumb_path": ["NASA", "Centers", "DLR"], "breadcrumb_confidence": 0.9}
        ),
        # Doc C (The Noise) - low semantic similarity and unrelated breadcrumb
        Document(
            page_content="DLR stands for German Aerospace Center.",
            metadata={"id": "doc_noise", "breadcrumb_path": ["Glossary"], "breadcrumb_confidence": 0.7}
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

    async def compute_posteriors(scope):
        docs_and_scores = retriever.vector_store.similarity_search_with_score(
            query=query,
            k=3
        )
        semantic_pairs = [(doc, 1.0 / (1.0 + score)) for doc, score in docs_and_scores]
        reranked = retriever._bayesian_rerank_by_breadcrumb(
            semantic_pairs,
            breadcrumb_scope=scope,
            heuristic_conf=0.62
        )
        return semantic_pairs, reranked

    print_results("A) No Scope", no_scope)
    print_results("B) With Scope", with_scope)

    semantic_pairs, reranked = await compute_posteriors(["NASA", "Centers"])
    print("\nPosterior Breakdown (scope=['NASA','Centers'])")
    print("-" * 60)
    for doc, posterior in reranked:
        meta = doc.metadata or {}
        chunk_breadcrumb = meta.get("breadcrumb_path", [])
        match_level = retriever._calculate_breadcrumb_match_level(chunk_breadcrumb, ["NASA", "Centers"])
        prior = retriever._calculate_structural_prior(
            match_level,
            meta.get("breadcrumb_confidence", 0.5),
            heuristic_conf=0.62
        )
        semantic_score = next(
            (score for d, score in semantic_pairs if d == doc),
            0.0
        )
        print(
            f"id={meta.get('id')} | semantic={semantic_score:.3f} | "
            f"prior={prior:.3f} | posterior={posterior:.3f} | "
            f"breadcrumb={chunk_breadcrumb}"
        )

    print("\n✅ Compare ordering between A and B for Phase 2 verification")


def main():
    asyncio.run(run_ab_diff())


if __name__ == "__main__":
    main()
