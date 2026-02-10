"""
Phase 3 StepDefiner + Rerank Debug

Uses StepDefinerAgent to generate subqueries (entity_name, target_type,
breadcrumb_scope) and then runs RetrieverAgent twice:
1) without breadcrumb_scope
2) with breadcrumb_scope from StepDefiner

This helps diagnose entity preservation, scope emission, and reranking behavior.
Requires LLM credentials for StepDefiner.
"""

import argparse
import asyncio
import json
import os
import sys

from langchain.schema import Document

# Ensure MCP folder is first on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.step_definer_agent import StepDefinerAgent
from agents.retriever_agent import RetrieverAgent


def _build_mock_docs() -> list[Document]:
    docs = [
        Document(
            page_content="DLR Headquarters is in Mexico.",
            metadata={"title": "Trap Doc", "breadcrumb_path": ["Mexico"]},
        ),
        Document(
            page_content="The HQ is located in Cologne. DLR is the German Aerospace Center.",
            metadata={"title": "Truth Doc", "breadcrumb_path": ["NASA", "Centers", "DLR"]},
        ),
        Document(
            page_content="DLR stands for German Aerospace Center.",
            metadata={"title": "Noise Doc", "breadcrumb_path": ["Glossary"]},
        ),
    ]
    return docs


async def run_debug(query: str, expected_entity: str | None, expected_type: str | None, manual_scope: str | None):
    # Minimal plan + step
    plan = {
        "main_question": query,
        "disambiguated_query": query,
        "query_type": "unknown",
    }
    step = {
        "id": "step_1",
        "description": query,
        "objective": "Answer the question",
        "dependencies": [],
        "critical": True,
        "expected_output": "Short factual answer",
    }

    # Provide a synthetic previous answer with breadcrumb metadata
    previous_answers = {
        "step_0": {
            "documents": [
                {
                    "metadata": {
                        "breadcrumb_path": ["NASA", "Centers", "DLR"]
                    }
                }
            ]
        }
    }

    agent = StepDefinerAgent()
    response = await agent.process(
        {
            "step": step,
            "plan": plan,
            "history": [],
            "context": {},
            "previous_answers": previous_answers,
            "max_subqueries": 1,
        }
    )

    print("\n=== Raw StepDefiner JSON ===")
    print(response.content)
    try:
        parsed = json.loads(response.content)
    except json.JSONDecodeError as e:
        print(f"\n❌ Failed to parse StepDefiner JSON: {e}")
        return
    sub_queries = parsed.get("sub_queries", []) or []
    if not sub_queries:
        raise RuntimeError("No sub_queries returned by StepDefinerAgent.")

    subq = sub_queries[0]
    breadcrumb_scope = subq.get("breadcrumb_scope") or []
    if manual_scope:
        breadcrumb_scope = [s.strip() for s in manual_scope.split(",") if s.strip()]

    print("\n=== StepDefiner Output ===")
    print(f"query: {subq.get('query')}")
    print(f"entity_name: {subq.get('entity_name')}")
    print(f"target_type: {subq.get('target_type')}")
    print(f"breadcrumb_scope: {breadcrumb_scope}")
    print(f"structural_priority: {subq.get('structural_priority')}")
    print(f"scope_rationale: {subq.get('scope_rationale')}")

    if expected_entity and expected_entity.lower() not in (subq.get("query") or "").lower():
        print(f"⚠️ Expected entity not preserved in query: {expected_entity}")
    if expected_type and (subq.get("target_type") or "").upper() != expected_type.upper():
        print(f"⚠️ Expected target_type '{expected_type}' but got '{subq.get('target_type')}'")

    # Run retriever without and with scope
    documents = _build_mock_docs()
    retriever = RetrieverAgent(documents=documents)

    print("\n=== Retrieval without scope ===")
    no_scope = await retriever.process({"query": subq.get("query", ""), "k": 3, "min_similarity": 0.0})
    no_scope_docs = json.loads(no_scope.content).get("documents", [])
    for i, d in enumerate(no_scope_docs, start=1):
        meta = d.get("metadata", {})
        print(f"{i}. title={meta.get('title')} score={d.get('score'):.3f} path={meta.get('breadcrumb_path')}")

    if breadcrumb_scope:
        print("\n=== Retrieval with scope ===")
        with_scope = await retriever.process(
            {
                "query": subq.get("query", ""),
                "k": 3,
                "min_similarity": 0.0,
                "breadcrumb_scope": breadcrumb_scope,
            }
        )
        with_scope_docs = json.loads(with_scope.content).get("documents", [])
        for i, d in enumerate(with_scope_docs, start=1):
            meta = d.get("metadata", {})
            print(f"{i}. title={meta.get('title')} score={d.get('score'):.3f} path={meta.get('breadcrumb_path')}")
    else:
        print("\n⚠️ No breadcrumb_scope emitted; reranking not exercised.")


def main():
    parser = argparse.ArgumentParser(description="Debug StepDefiner + reranking interactions.")
    parser.add_argument(
        "--query",
        type=str,
        default="Where is the DLR headquarters located?",
    )
    parser.add_argument("--expected-entity", type=str, default="DLR")
    parser.add_argument("--expected-type", type=str, default="LOC")
    parser.add_argument("--manual-scope", type=str, default=None,
                        help="Comma-separated breadcrumb scope override")
    args = parser.parse_args()

    asyncio.run(run_debug(args.query, args.expected_entity, args.expected_type, args.manual_scope))


if __name__ == "__main__":
    main()
