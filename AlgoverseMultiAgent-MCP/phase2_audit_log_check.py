"""
Phase 2 Audit Log Check

Confirms that the audit log lines are emitted when breadcrumb_scope is provided.
"""

import io
import logging
import os
import sys
import asyncio

# Ensure the MCP folder is first on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from langchain.schema import Document
from agents.retriever_agent import RetrieverAgent


async def run_audit_log_check():
    print("\n" + "=" * 80)
    print("PHASE 2 AUDIT LOG CHECK")
    print("=" * 80)

    documents = [
        Document(
            page_content="DLR headquarters is located in Cologne, Germany.",
            metadata={"id": "doc_match", "breadcrumb_path": ["NASA", "Centers", "DLR"], "breadcrumb_confidence": 0.9}
        )
    ]

    retriever = RetrieverAgent(
        documents=documents,
        model_config={"use_cuda": False},
        top_k=3,
        min_similarity=0.1
    )

    log_stream = io.StringIO()
    handler = logging.StreamHandler(log_stream)
    logger = logging.getLogger("agents.retriever_agent")
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)

    try:
        await retriever.process({
            "query": "Where is the DLR headquarters located?",
            "k": 3,
            "min_similarity": 0.1,
            "breadcrumb_scope": ["NASA", "Centers"]
        })
    finally:
        logger.removeHandler(handler)

    logs = log_stream.getvalue()
    print("Captured audit logs:")
    print(logs.strip() or "(no logs captured)")

    assert "Structural Intent Detected" in logs, "Missing structural intent audit log"
    assert "Retrieval pool expanded" in logs, "Missing retrieval pool expansion log"

    print("✅ Audit logs present")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    asyncio.run(run_audit_log_check())


if __name__ == "__main__":
    main()
