"""
Unit Tests for ExtractorAgent Context Stitching

Tests the _get_stitched_context() method to ensure:
- Previous, current, and next chunks are correctly retrieved
- Parent context is extracted from breadcrumb_path
- Edge cases are handled correctly
"""

import sys
import os

# Ensure the MCP folder is first on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.extractor_agent import ExtractorAgent


def test_get_stitched_context_full():
    """Test 1: Full stitched context with all chunks available"""
    print("\n[Test 1] Full stitched context")
    print("-" * 60)
    
    agent = ExtractorAgent()
    
    # Create mock documents with relationships
    all_docs = [
        {
            "page_content": "Previous chunk content about NASA centers",
            "metadata": {"chunk_id": "example_1_0"}
        },
        {
            "page_content": "Current chunk content about DLR headquarters",
            "metadata": {
                "chunk_id": "example_1_1",
                "previous_chunk_id": "example_1_0",
                "next_chunk_id": "example_1_2",
                "breadcrumb_path": ["NASA", "Centers", "DLR"],
                "breadcrumb_string": "NASA > Centers > DLR"
            }
        },
        {
            "page_content": "Next chunk content about DLR facilities",
            "metadata": {"chunk_id": "example_1_2"}
        }
    ]
    
    current_doc = all_docs[1]
    stitched = agent._get_stitched_context(current_doc, all_docs)
    
    assert stitched['previous'] == "Previous chunk content about NASA centers"
    assert stitched['current'] == "Current chunk content about DLR headquarters"
    assert stitched['next'] == "Next chunk content about DLR facilities"
    assert "NASA" in stitched['parent'] and "Centers" in stitched['parent']
    
    print(f"✅ Full context stitched:")
    print(f"   - Previous: {stitched['previous'][:50]}...")
    print(f"   - Current: {stitched['current'][:50]}...")
    print(f"   - Next: {stitched['next'][:50]}...")
    print(f"   - Parent: {stitched['parent']}")


def test_get_stitched_context_no_previous():
    """Test 2: No previous chunk (first chunk)"""
    print("\n[Test 2] No previous chunk")
    print("-" * 60)
    
    agent = ExtractorAgent()
    
    all_docs = [
        {
            "page_content": "First chunk content",
            "metadata": {
                "chunk_id": "example_1_0",
                "next_chunk_id": "example_1_1",
                "breadcrumb_path": ["NASA"]
            }
        },
        {
            "page_content": "Second chunk content",
            "metadata": {"chunk_id": "example_1_1"}
        }
    ]
    
    current_doc = all_docs[0]
    stitched = agent._get_stitched_context(current_doc, all_docs)
    
    assert stitched['previous'] == "", "Expected empty previous for first chunk"
    assert stitched['current'] == "First chunk content"
    assert stitched['next'] == "Second chunk content"
    
    print(f"✅ No previous chunk handled correctly")


def test_get_stitched_context_no_next():
    """Test 3: No next chunk (last chunk)"""
    print("\n[Test 3] No next chunk")
    print("-" * 60)
    
    agent = ExtractorAgent()
    
    all_docs = [
        {
            "page_content": "First chunk content",
            "metadata": {"chunk_id": "example_1_0"}
        },
        {
            "page_content": "Last chunk content",
            "metadata": {
                "chunk_id": "example_1_1",
                "previous_chunk_id": "example_1_0",
                "breadcrumb_path": ["NASA", "Centers"]
            }
        }
    ]
    
    current_doc = all_docs[1]
    stitched = agent._get_stitched_context(current_doc, all_docs)
    
    assert stitched['previous'] == "First chunk content"
    assert stitched['current'] == "Last chunk content"
    assert stitched['next'] == "", "Expected empty next for last chunk"
    
    print(f"✅ No next chunk handled correctly")


def test_get_stitched_context_no_parent():
    """Test 4: No parent context (single-level breadcrumb)"""
    print("\n[Test 4] No parent context")
    print("-" * 60)
    
    agent = ExtractorAgent()
    
    all_docs = [
        {
            "page_content": "Current chunk",
            "metadata": {
                "chunk_id": "example_1_0",
                "breadcrumb_path": ["NASA"],  # Single level, no parent
                "breadcrumb_string": "NASA"
            }
        }
    ]
    
    current_doc = all_docs[0]
    stitched = agent._get_stitched_context(current_doc, all_docs)
    
    assert stitched['current'] == "Current chunk"
    # Parent should be empty or minimal for single-level breadcrumb
    print(f"✅ No parent context handled (breadcrumb: {all_docs[0]['metadata']['breadcrumb_path']})")


def test_get_stitched_context_missing_chunk_ids():
    """Test 5: Missing chunk IDs in document map"""
    print("\n[Test 5] Missing chunk IDs")
    print("-" * 60)
    
    agent = ExtractorAgent()
    
    all_docs = [
        {
            "page_content": "Current chunk",
            "metadata": {
                "chunk_id": "example_1_0",
                "previous_chunk_id": "example_0_9",  # Doesn't exist
                "next_chunk_id": "example_1_1",  # Doesn't exist
                "breadcrumb_path": ["NASA"]
            }
        }
    ]
    
    current_doc = all_docs[0]
    stitched = agent._get_stitched_context(current_doc, all_docs)
    
    assert stitched['previous'] == "", "Expected empty when previous chunk not found"
    assert stitched['next'] == "", "Expected empty when next chunk not found"
    assert stitched['current'] == "Current chunk"
    
    print(f"✅ Missing chunk IDs handled gracefully")


def test_get_stitched_context_no_metadata():
    """Test 6: Document with no metadata"""
    print("\n[Test 6] Document with no metadata")
    print("-" * 60)
    
    agent = ExtractorAgent()
    
    all_docs = [
        {
            "page_content": "Current chunk",
            # No metadata
        }
    ]
    
    current_doc = all_docs[0]
    stitched = agent._get_stitched_context(current_doc, all_docs)
    
    assert stitched['previous'] == ""
    assert stitched['current'] == "Current chunk"
    assert stitched['next'] == ""
    assert stitched['parent'] == ""
    
    print(f"✅ No metadata handled correctly")


def test_get_stitched_context_parent_extraction():
    """Test 7: Parent context extraction from breadcrumb_path"""
    print("\n[Test 7] Parent context extraction")
    print("-" * 60)
    
    agent = ExtractorAgent()
    
    all_docs = [
        {
            "page_content": "Current chunk",
            "metadata": {
                "chunk_id": "example_1_0",
                "breadcrumb_path": ["NASA", "Centers", "DLR", "Headquarters"]
            }
        }
    ]
    
    current_doc = all_docs[0]
    stitched = agent._get_stitched_context(current_doc, all_docs)
    
    # Parent should be path without last element
    assert "NASA" in stitched['parent'] and "Centers" in stitched['parent']
    assert "DLR" in stitched['parent']
    assert "Headquarters" not in stitched['parent'], "Parent should not include last element"
    
    print(f"✅ Parent context extracted: {stitched['parent']}")


def test_get_stitched_context_multiple_documents():
    """Test 8: Stitching with multiple documents in collection"""
    print("\n[Test 8] Multiple documents in collection")
    print("-" * 60)
    
    agent = ExtractorAgent()
    
    all_docs = [
        {"page_content": "Doc 0", "metadata": {"chunk_id": "doc_0"}},
        {"page_content": "Doc 1", "metadata": {"chunk_id": "doc_1"}},
        {
            "page_content": "Doc 2 (target)",
            "metadata": {
                "chunk_id": "doc_2",
                "previous_chunk_id": "doc_1",
                "next_chunk_id": "doc_3",
                "breadcrumb_path": ["A", "B"]
            }
        },
        {"page_content": "Doc 3", "metadata": {"chunk_id": "doc_3"}},
        {"page_content": "Doc 4", "metadata": {"chunk_id": "doc_4"}},
    ]
    
    current_doc = all_docs[2]
    stitched = agent._get_stitched_context(current_doc, all_docs)
    
    assert stitched['previous'] == "Doc 1"
    assert stitched['current'] == "Doc 2 (target)"
    assert stitched['next'] == "Doc 3"
    
    print(f"✅ Correctly found chunks in large collection")


def test_get_stitched_context_empty_documents():
    """Test 9: Empty document collection"""
    print("\n[Test 9] Empty document collection")
    print("-" * 60)
    
    agent = ExtractorAgent()
    
    all_docs = []
    current_doc = {
        "page_content": "Current chunk",
        "metadata": {
            "chunk_id": "example_1_0",
            "previous_chunk_id": "example_0_9",
            "next_chunk_id": "example_1_1"
        }
    }
    
    stitched = agent._get_stitched_context(current_doc, all_docs)
    
    assert stitched['previous'] == ""
    assert stitched['current'] == "Current chunk"
    assert stitched['next'] == ""
    
    print(f"✅ Empty collection handled correctly")


def run_all_tests():
    """Run all context stitching tests"""
    print("=" * 60)
    print("ExtractorAgent Context Stitching Tests")
    print("=" * 60)
    
    tests = [
        test_get_stitched_context_full,
        test_get_stitched_context_no_previous,
        test_get_stitched_context_no_next,
        test_get_stitched_context_no_parent,
        test_get_stitched_context_missing_chunk_ids,
        test_get_stitched_context_no_metadata,
        test_get_stitched_context_parent_extraction,
        test_get_stitched_context_multiple_documents,
        test_get_stitched_context_empty_documents,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"❌ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

