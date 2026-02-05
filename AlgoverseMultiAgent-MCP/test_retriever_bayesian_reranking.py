"""
Unit Tests for RetrieverAgent Bayesian Re-ranking

Tests the breadcrumb-based Bayesian re-ranking functionality:
- _calculate_breadcrumb_match_level()
- _calculate_structural_prior()
- _bayesian_rerank_by_breadcrumb()
"""

import sys
import os

# Ensure the MCP folder is first on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.retriever_agent import RetrieverAgent
from langchain.schema import Document


def test_breadcrumb_match_level_perfect_match():
    """Test 1: Perfect prefix match"""
    print("\n[Test 1] Perfect prefix match")
    print("-" * 60)
    
    agent = RetrieverAgent()
    
    match = agent._calculate_breadcrumb_match_level(
        chunk_breadcrumb=["NASA", "Centers", "DLR"],
        target_scope=["NASA", "Centers"]
    )
    
    assert match == 1.0, f"Expected 1.0, got {match}"
    print(f"✅ Perfect match: {match}")


def test_breadcrumb_match_level_partial_match():
    """Test 2: Partial match (scope is longer than chunk)"""
    print("\n[Test 2] Partial match")
    print("-" * 60)
    
    agent = RetrieverAgent()
    
    match = agent._calculate_breadcrumb_match_level(
        chunk_breadcrumb=["NASA"],
        target_scope=["NASA", "Centers"]
    )
    
    assert match == 0.5, f"Expected 0.5, got {match}"
    print(f"✅ Partial match (50%): {match}")


def test_breadcrumb_match_level_no_match():
    """Test 3: No match"""
    print("\n[Test 3] No match")
    print("-" * 60)
    
    agent = RetrieverAgent()
    
    match = agent._calculate_breadcrumb_match_level(
        chunk_breadcrumb=["DLR"],
        target_scope=["NASA", "Centers"]
    )
    
    assert match == 0.0, f"Expected 0.0, got {match}"
    print(f"✅ No match: {match}")


def test_breadcrumb_match_level_case_insensitive():
    """Test 4: Case insensitive matching"""
    print("\n[Test 4] Case insensitive matching")
    print("-" * 60)
    
    agent = RetrieverAgent()
    
    match = agent._calculate_breadcrumb_match_level(
        chunk_breadcrumb=["nasa", "centers"],
        target_scope=["NASA", "Centers"]
    )
    
    assert match == 1.0, f"Expected 1.0 (case insensitive), got {match}"
    print(f"✅ Case insensitive match: {match}")


def test_breadcrumb_match_level_empty_scope():
    """Test 5: Empty scope"""
    print("\n[Test 5] Empty scope")
    print("-" * 60)
    
    agent = RetrieverAgent()
    
    match = agent._calculate_breadcrumb_match_level(
        chunk_breadcrumb=["NASA", "Centers"],
        target_scope=[]
    )
    
    assert match == 0.0, f"Expected 0.0 for empty scope, got {match}"
    print(f"✅ Empty scope handled: {match}")


def test_structural_prior_high_match():
    """Test 6: Structural prior with high match"""
    print("\n[Test 6] Structural prior - high match")
    print("-" * 60)
    
    agent = RetrieverAgent()
    
    prior = agent._calculate_structural_prior(
        breadcrumb_match_level=1.0,
        breadcrumb_confidence=0.8,
        heuristic_conf=0.62
    )
    
    # Should be close to heuristic_conf * match_level * confidence
    expected = 0.1 + (0.62 - 0.1) * (1.0 * 0.8)
    assert abs(prior - expected) < 0.01, f"Expected ~{expected}, got {prior}"
    print(f"✅ High match prior: {prior:.3f}")


def test_structural_prior_low_match():
    """Test 7: Structural prior with low match"""
    print("\n[Test 7] Structural prior - low match")
    print("-" * 60)
    
    agent = RetrieverAgent()
    
    prior = agent._calculate_structural_prior(
        breadcrumb_match_level=0.0,
        breadcrumb_confidence=0.8,
        heuristic_conf=0.62
    )
    
    # Should be close to epsilon (0.1)
    assert prior < 0.2, f"Expected prior < 0.2 for low match, got {prior}"
    print(f"✅ Low match prior: {prior:.3f}")


def test_structural_prior_medium_match():
    """Test 8: Structural prior with medium match"""
    print("\n[Test 8] Structural prior - medium match")
    print("-" * 60)
    
    agent = RetrieverAgent()
    
    prior = agent._calculate_structural_prior(
        breadcrumb_match_level=0.5,
        breadcrumb_confidence=0.7,
        heuristic_conf=0.62
    )
    
    # Should be between epsilon and heuristic_conf
    assert 0.1 < prior < 0.62, f"Expected prior between 0.1 and 0.62, got {prior}"
    print(f"✅ Medium match prior: {prior:.3f}")


def test_bayesian_reranking_with_scope():
    """Test 9: Bayesian re-ranking boosts matching documents"""
    print("\n[Test 9] Bayesian re-ranking with scope")
    print("-" * 60)
    
    agent = RetrieverAgent()
    
    # Create mock documents
    doc1 = Document(
        page_content="DLR headquarters information",
        metadata={
            "breadcrumb_path": ["NASA", "Centers", "DLR"],
            "breadcrumb_confidence": 0.8,
            "breadcrumb_string": "NASA > Centers > DLR"
        }
    )
    doc2 = Document(
        page_content="Random information",
        metadata={
            "breadcrumb_path": ["Other", "Topic"],
            "breadcrumb_confidence": 0.8,
            "breadcrumb_string": "Other > Topic"
        }
    )
    
    # doc2 has higher semantic score, but doc1 matches scope
    docs_and_scores = [
        (doc1, 0.7),  # Lower semantic score, but matches scope
        (doc2, 0.9),  # Higher semantic score, but doesn't match scope
    ]
    
    # Re-rank with scope
    reranked = agent._bayesian_rerank_by_breadcrumb(
        docs_and_scores,
        breadcrumb_scope=["NASA", "Centers"],
        heuristic_conf=0.62
    )
    
    # doc1 should be ranked higher despite lower semantic score
    assert reranked[0][0] == doc1, "Expected doc1 (matching scope) to be ranked first"
    assert reranked[0][1] > reranked[1][1], "Expected doc1 posterior > doc2 posterior"
    
    print(f"✅ Re-ranking successful:")
    print(f"   - Doc1 (match): semantic=0.7, posterior={reranked[0][1]:.3f}")
    print(f"   - Doc2 (no match): semantic=0.9, posterior={reranked[1][1]:.3f}")


def test_bayesian_reranking_no_scope():
    """Test 10: Bayesian re-ranking without scope (should return original)"""
    print("\n[Test 10] Bayesian re-ranking without scope")
    print("-" * 60)
    
    agent = RetrieverAgent()
    
    doc1 = Document(
        page_content="Document 1",
        metadata={"breadcrumb_path": ["A", "B"]}
    )
    doc2 = Document(
        page_content="Document 2",
        metadata={"breadcrumb_path": ["C", "D"]}
    )
    
    docs_and_scores = [
        (doc1, 0.8),
        (doc2, 0.6),
    ]
    
    # Re-rank without scope
    reranked = agent._bayesian_rerank_by_breadcrumb(
        docs_and_scores,
        breadcrumb_scope=None,
        heuristic_conf=0.62
    )
    
    # Should return original order (no re-ranking)
    assert reranked == docs_and_scores, "Expected original order when no scope provided"
    print(f"✅ No re-ranking when scope is None")


def test_bayesian_reranking_empty_scope():
    """Test 11: Bayesian re-ranking with empty scope"""
    print("\n[Test 11] Bayesian re-ranking with empty scope")
    print("-" * 60)
    
    agent = RetrieverAgent()
    
    doc1 = Document(
        page_content="Document 1",
        metadata={"breadcrumb_path": ["A", "B"]}
    )
    
    docs_and_scores = [(doc1, 0.8)]
    
    # Re-rank with empty scope
    reranked = agent._bayesian_rerank_by_breadcrumb(
        docs_and_scores,
        breadcrumb_scope=[],
        heuristic_conf=0.62
    )
    
    # Empty scope => uniform prior => posterior equals likelihood
    assert len(reranked) == 1, "Expected 1 document"
    assert reranked[0][1] == docs_and_scores[0][1], "Expected posterior to equal likelihood for empty scope"
    print(f"✅ Empty scope handled (posterior equals likelihood)")


def test_bayesian_reranking_multiple_documents():
    """Test 12: Bayesian re-ranking with multiple documents"""
    print("\n[Test 12] Bayesian re-ranking with multiple documents")
    print("-" * 60)
    
    agent = RetrieverAgent()
    
    # Create documents with varying match levels
    docs = [
        Document(
            page_content=f"Document {i}",
            metadata={
                "breadcrumb_path": ["NASA", "Centers"] if i == 0 else ["Other", f"Topic{i}"],
                "breadcrumb_confidence": 0.8
            }
        )
        for i in range(5)
    ]
    
    docs_and_scores = [(doc, 0.5 + i * 0.1) for i, doc in enumerate(docs)]
    
    # Re-rank with scope
    reranked = agent._bayesian_rerank_by_breadcrumb(
        docs_and_scores,
        breadcrumb_scope=["NASA", "Centers"],
        heuristic_conf=0.62
    )
    
    # First document should be ranked first (matches scope)
    assert reranked[0][0] == docs[0], "Expected matching document to be ranked first"
    print(f"✅ Multiple documents re-ranked correctly")
    print(f"   - Top document matches scope: {reranked[0][0].metadata.get('breadcrumb_path')}")


def run_all_tests():
    """Run all Bayesian re-ranking tests"""
    print("=" * 60)
    print("RetrieverAgent Bayesian Re-ranking Tests")
    print("=" * 60)
    
    tests = [
        test_breadcrumb_match_level_perfect_match,
        test_breadcrumb_match_level_partial_match,
        test_breadcrumb_match_level_no_match,
        test_breadcrumb_match_level_case_insensitive,
        test_breadcrumb_match_level_empty_scope,
        test_structural_prior_high_match,
        test_structural_prior_low_match,
        test_structural_prior_medium_match,
        test_bayesian_reranking_with_scope,
        test_bayesian_reranking_no_scope,
        test_bayesian_reranking_empty_scope,
        test_bayesian_reranking_multiple_documents,
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

