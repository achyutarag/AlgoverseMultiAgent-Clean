"""
Phase 1: Comprehensive Unit Test Runner

Runs all Phase 1 unit tests for the hybrid metadata management system:
1. StepDefinerAgent breadcrumb extraction
2. RetrieverAgent Bayesian re-ranking
3. ExtractorAgent context stitching
"""

import asyncio
import sys
import os

# Ensure the MCP folder is first on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


async def run_step_definer_tests():
    """Run StepDefinerAgent tests"""
    print("\n" + "=" * 80)
    print("PHASE 1.1: StepDefinerAgent Breadcrumb Extraction Tests")
    print("=" * 80)
    
    try:
        from test_step_definer_breadcrumb_extraction import run_all_tests
        return await run_all_tests()
    except Exception as e:
        print(f"❌ Failed to run StepDefiner tests: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_retriever_tests():
    """Run RetrieverAgent tests"""
    print("\n" + "=" * 80)
    print("PHASE 1.2: RetrieverAgent Bayesian Re-ranking Tests")
    print("=" * 80)
    
    try:
        from test_retriever_bayesian_reranking import run_all_tests
        return run_all_tests()
    except Exception as e:
        print(f"❌ Failed to run Retriever tests: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_extractor_tests():
    """Run ExtractorAgent tests"""
    print("\n" + "=" * 80)
    print("PHASE 1.3: ExtractorAgent Context Stitching Tests")
    print("=" * 80)
    
    try:
        from test_extractor_context_stitching import run_all_tests
        return run_all_tests()
    except Exception as e:
        print(f"❌ Failed to run Extractor tests: {e}")
        import traceback
        traceback.print_exc()
        return False


async def run_all_phase1_tests():
    """Run all Phase 1 unit tests"""
    print("=" * 80)
    print("PHASE 1: UNIT TESTS - Hybrid Metadata Management System")
    print("=" * 80)
    print("\nTesting components in isolation:")
    print("  1. StepDefinerAgent: Breadcrumb scope extraction")
    print("  2. RetrieverAgent: Bayesian re-ranking with breadcrumb scope")
    print("  3. ExtractorAgent: Context stitching (i±1 chunks)")
    print("=" * 80)
    
    results = {}
    
    # Test 1: StepDefinerAgent
    results['step_definer'] = await run_step_definer_tests()
    
    # Test 2: RetrieverAgent
    results['retriever'] = run_retriever_tests()
    
    # Test 3: ExtractorAgent
    results['extractor'] = run_extractor_tests()
    
    # Summary
    print("\n" + "=" * 80)
    print("PHASE 1 TEST SUMMARY")
    print("=" * 80)
    
    all_passed = True
    for component, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {component.upper():20s}: {status}")
        if not passed:
            all_passed = False
    
    print("=" * 80)
    
    if all_passed:
        print("\n🎉 All Phase 1 unit tests passed!")
        print("   Ready to proceed to Phase 2 (Integration Tests)")
    else:
        print("\n⚠️  Some tests failed. Please fix issues before proceeding.")
    
    return all_passed


if __name__ == "__main__":
    success = asyncio.run(run_all_phase1_tests())
    sys.exit(0 if success else 1)

