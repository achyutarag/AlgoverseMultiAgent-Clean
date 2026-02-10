"""
Phase 3 Integration Test: ExtractorAgent Context Stitching

Validates horizontal stitching (prev/next chunks) and vertical stitching
(breadcrumb/parent context) with a logical dependency example.
"""

import asyncio
import os
import sys

# Ensure MCP folder is first on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.extractor_agent import ExtractorAgent


async def run_extractor_integration_test():
    print("\n" + "=" * 80)
    print("PHASE 3 EXTRACTOR INTEGRATION TEST")
    print("=" * 80)

    agent = ExtractorAgent()

    # Logical dependency example:
    # Current chunk references "This facility", which is defined in the previous chunk.
    all_docs = [
        {
            "page_content": "NASA and DLR established the Propulsion Lab in 2024.",
            "metadata": {"chunk_id": "1_0"}
        },
        {
            "page_content": "This facility is headquartered in Cologne.",
            "metadata": {
                "chunk_id": "1_1",
                "previous_chunk_id": "1_0",
                "breadcrumb_path": ["NASA", "Centers", "DLR"]
            }
        }
    ]

    current_doc = all_docs[1]

    # Directly call the stitching helper
    stitched = agent._get_stitched_context(current_doc, all_docs)

    print("\n--- STITCHED CONTEXT ---")
    print("PREVIOUS:", stitched.get("previous", ""))
    print("CURRENT:", stitched.get("current", ""))
    print("NEXT:", stitched.get("next", ""))
    print("PARENT:", stitched.get("parent", ""))

    # Assertions for "zero-loss" transfer
    assert "Propulsion Lab" in stitched.get("previous", ""), "Missing previous chunk context"
    assert "facility is headquartered in Cologne" in stitched.get("current", ""), "Missing current chunk context"
    assert "NASA" in stitched.get("parent", "") or "Centers" in stitched.get("parent", ""), "Missing parent context"

    print("✅ Horizontal + vertical stitching verified")


def main():
    asyncio.run(run_extractor_integration_test())


if __name__ == "__main__":
    main()
