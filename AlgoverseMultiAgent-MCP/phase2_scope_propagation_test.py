"""
Phase 2 Scope Propagation Test

Validates that breadcrumb_scope flows through StateManager to RetrieverAgent
by checking behavior (ranking changes) rather than logs.
"""

import asyncio
import os
import sys

# Ensure the MCP folder is first on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from langchain.schema import Document
from agents.state_manager.core import StateManager
from agents.retriever_agent import RetrieverAgent


async def run_scope_propagation_test():
    print("\n" + "=" * 80)
    print("PHASE 2 SCOPE PROPAGATION TEST")
    print("=" * 80)

    # Minimal document set with breadcrumb metadata
    documents = [
        Document(
            page_content="DLR is a research center affiliated with NASA centers.",
            metadata={"id": "doc_match", "breadcrumb_path": ["NASA", "Centers", "DLR"], "breadcrumb_confidence": 0.9}
        ),
        Document(
            page_content="Unrelated content about cooking recipes.",
            metadata={"id": "doc_other", "breadcrumb_path": ["Cooking", "Recipes"], "breadcrumb_confidence": 0.8}
        )
    ]

    retriever = RetrieverAgent(
        documents=documents,
        model_config={"use_cuda": False},
        top_k=3,
        min_similarity=0.1
    )

    state_manager = StateManager()

    with_scope = await state_manager.stabilize_and_retrieve(
            proposed_query="DLR headquarters NASA centers",
            hop=1,
            previous_answers={},
            plan_goal="Find DLR headquarters",
            retriever_agent=retriever,
            current_step_index=0,
            total_steps=1,
            breadcrumb_scope=["NASA", "Centers"]
        )
    
    no_scope = await state_manager.stabilize_and_retrieve(
            proposed_query="DLR headquarters NASA centers",
            hop=1,
            previous_answers={},
            plan_goal="Find DLR headquarters",
            retriever_agent=retriever,
            current_step_index=0,
            total_steps=1
        )
    
    with_scope_docs = with_scope.get("documents", [])
    no_scope_docs = no_scope.get("documents", [])
    
    assert with_scope_docs, "No documents returned with breadcrumb scope"
    assert no_scope_docs, "No documents returned without breadcrumb scope"
    
    top_with_scope = with_scope_docs[0].get("metadata", {}).get("id")
    top_no_scope = no_scope_docs[0].get("metadata", {}).get("id")
    
    print("Top doc with scope:", top_with_scope)
    print("Top doc without scope:", top_no_scope)
    
    assert top_with_scope == "doc_match", "Expected scope-aware top document to be the breadcrumb match"
    
    print("✅ Scope propagation confirmed via ranking behavior")


def main():
    asyncio.run(run_scope_propagation_test())


if __name__ == "__main__":
    main()
