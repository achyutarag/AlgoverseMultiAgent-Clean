"""
Unit Tests for StepDefinerAgent Breadcrumb Scope Extraction

Tests the _extract_breadcrumb_scope_from_history() method to ensure
breadcrumb scope is correctly extracted from previous_answers and history.
"""

import asyncio
import sys
import os
import json

# Ensure the MCP folder is first on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.step_definer_agent import StepDefinerAgent


async def test_breadcrumb_extraction_from_previous_answers():
    """Test 1: Extract breadcrumb scope from previous_answers with documents"""
    print("\n[Test 1] Extract from previous_answers with documents")
    print("-" * 60)
    
    agent = StepDefinerAgent()
    
    previous_answers = {
        "step_1": {
            "documents": [
                {
                    "metadata": {
                        "breadcrumb_path": ["NASA", "Centers", "DLR"],
                        "breadcrumb_string": "NASA > Centers > DLR"
                    }
                }
            ]
        }
    }
    
    scope = agent._extract_breadcrumb_scope_from_history(previous_answers, [])
    
    assert scope == ["NASA", "Centers", "DLR"], f"Expected ['NASA', 'Centers', 'DLR'], got {scope}"
    print(f"✅ Extracted scope: {scope}")


async def test_breadcrumb_extraction_from_history_json():
    """Test 2: Extract breadcrumb scope from history JSON content"""
    print("\n[Test 2] Extract from history JSON content")
    print("-" * 60)
    
    agent = StepDefinerAgent()
    
    history = [
        {
            "role": "assistant",
            "content": json.dumps({
                "documents": [
                    {
                        "metadata": {
                            "breadcrumb_path": ["Italian", "Armed Forces"],
                            "breadcrumb_string": "Italian > Armed Forces"
                        }
                    }
                ]
            })
        }
    ]
    
    scope = agent._extract_breadcrumb_scope_from_history({}, history)
    
    assert scope == ["Italian", "Armed Forces"], f"Expected ['Italian', 'Armed Forces'], got {scope}"
    print(f"✅ Extracted scope: {scope}")


async def test_breadcrumb_extraction_no_scope_available():
    """Test 3: Handle case when no breadcrumb scope is available"""
    print("\n[Test 3] No breadcrumb scope available")
    print("-" * 60)
    
    agent = StepDefinerAgent()
    
    scope = agent._extract_breadcrumb_scope_from_history({}, [])
    
    assert scope is None, f"Expected None, got {scope}"
    print(f"✅ Correctly returned None when no scope available")


async def test_breadcrumb_extraction_multiple_documents():
    """Test 4: Extract from multiple documents (should use first valid)"""
    print("\n[Test 4] Extract from multiple documents")
    print("-" * 60)
    
    agent = StepDefinerAgent()
    
    previous_answers = {
        "step_1": {
            "documents": [
                {
                    "metadata": {
                        "breadcrumb_path": ["NASA", "Centers"],
                        "breadcrumb_string": "NASA > Centers"
                    }
                },
                {
                    "metadata": {
                        "breadcrumb_path": ["Other", "Topic"],
                        "breadcrumb_string": "Other > Topic"
                    }
                }
            ]
        }
    }
    
    scope = agent._extract_breadcrumb_scope_from_history(previous_answers, [])
    
    assert scope == ["NASA", "Centers"], f"Expected ['NASA', 'Centers'], got {scope}"
    print(f"✅ Extracted first valid scope: {scope}")


async def test_breadcrumb_extraction_empty_breadcrumb_path():
    """Test 5: Handle empty breadcrumb_path"""
    print("\n[Test 5] Handle empty breadcrumb_path")
    print("-" * 60)
    
    agent = StepDefinerAgent()
    
    previous_answers = {
        "step_1": {
            "documents": [
                {
                    "metadata": {
                        "breadcrumb_path": [],
                        "breadcrumb_string": "Unknown"
                    }
                }
            ]
        }
    }
    
    scope = agent._extract_breadcrumb_scope_from_history(previous_answers, [])
    
    assert scope is None, f"Expected None for empty breadcrumb_path, got {scope}"
    print(f"✅ Correctly returned None for empty breadcrumb_path")


async def test_breadcrumb_extraction_from_sources():
    """Test 6: Extract from 'sources' field (alternative to 'documents')"""
    print("\n[Test 6] Extract from 'sources' field")
    print("-" * 60)
    
    agent = StepDefinerAgent()
    
    previous_answers = {
        "step_1": {
            "sources": [
                {
                    "metadata": {
                        "breadcrumb_path": ["Grant", "Green"],
                        "breadcrumb_string": "Grant > Green"
                    }
                }
            ]
        }
    }
    
    scope = agent._extract_breadcrumb_scope_from_history(previous_answers, [])
    
    assert scope == ["Grant", "Green"], f"Expected ['Grant', 'Green'], got {scope}"
    print(f"✅ Extracted scope from 'sources' field: {scope}")


async def test_breadcrumb_extraction_max_depth():
    """Test 7: Verify max depth limit (should be 3)"""
    print("\n[Test 7] Verify max depth limit")
    print("-" * 60)
    
    agent = StepDefinerAgent()
    
    previous_answers = {
        "step_1": {
            "documents": [
                {
                    "metadata": {
                        "breadcrumb_path": ["Level1", "Level2", "Level3", "Level4", "Level5"],
                        "breadcrumb_string": "Level1 > Level2 > Level3 > Level4 > Level5"
                    }
                }
            ]
        }
    }
    
    scope = agent._extract_breadcrumb_scope_from_history(previous_answers, [])
    
    assert len(scope) == 3, f"Expected max depth 3, got {len(scope)}"
    assert scope == ["Level1", "Level2", "Level3"], f"Expected first 3 levels, got {scope}"
    print(f"✅ Correctly limited to max depth 3: {scope}")


async def run_all_tests():
    """Run all breadcrumb extraction tests"""
    print("=" * 60)
    print("StepDefinerAgent Breadcrumb Extraction Tests")
    print("=" * 60)
    
    tests = [
        test_breadcrumb_extraction_from_previous_answers,
        test_breadcrumb_extraction_from_history_json,
        test_breadcrumb_extraction_no_scope_available,
        test_breadcrumb_extraction_multiple_documents,
        test_breadcrumb_extraction_empty_breadcrumb_path,
        test_breadcrumb_extraction_from_sources,
        test_breadcrumb_extraction_max_depth,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            await test()
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
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)

