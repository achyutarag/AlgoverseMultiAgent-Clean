"""
Phase 2 Scope Propagation Test

Validates that breadcrumb_scope flows through StateManager to RetrieverAgent
by checking for the audit logs emitted in RetrieverAgent.
"""

import asyncio
import io
import logging
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

    log_stream = io.StringIO()
    handler = logging.StreamHandler(log_stream)
    logger = logging.getLogger("agents.retriever_agent")
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)

    try:
        _ = await state_manager.stabilize_and_retrieve(
            proposed_query="DLR headquarters NASA centers",
            hop=1,
            previous_answers={},
            plan_goal="Find DLR headquarters",
            retriever_agent=retriever,
            current_step_index=0,
            total_steps=1,
            breadcrumb_scope=["NASA", "Centers"]
        )
    finally:
        logger.removeHandler(handler)

    logs = log_stream.getvalue()

    print("Captured audit logs:")
    print(logs.strip() or "(no logs captured)")

    assert "Structural Intent Detected" in logs, "Missing structural intent audit log"
    assert "Retrieval pool expanded" in logs, "Missing retrieval pool expansion log"

    print("✅ Scope propagation confirmed via audit logs")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    asyncio.run(run_scope_propagation_test())


if __name__ == "__main__":
    main()
