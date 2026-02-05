# Phase 1 Testing Guide: Hybrid Metadata Management System

## Overview

Phase 1 tests validate each component **in isolation** to ensure they work correctly before integration. This helps isolate bugs and build confidence incrementally.

## Test Files

1. **`test_step_definer_breadcrumb_extraction.py`** - Tests breadcrumb scope extraction
2. **`test_retriever_bayesian_reranking.py`** - Tests Bayesian re-ranking logic
3. **`test_extractor_context_stitching.py`** - Tests context stitching (i±1 chunks)
4. **`test_phase1_all.py`** - Runs all Phase 1 tests together

## Running Tests

### Run Individual Component Tests

```bash
# Test StepDefinerAgent breadcrumb extraction
python test_step_definer_breadcrumb_extraction.py

# Test RetrieverAgent Bayesian re-ranking
python test_retriever_bayesian_reranking.py

# Test ExtractorAgent context stitching
python test_extractor_context_stitching.py
```

### Run All Phase 1 Tests

```bash
# Run all Phase 1 tests at once
python test_phase1_all.py
```

## Test Coverage

### StepDefinerAgent (7 tests)
- ✅ Extract from previous_answers with documents
- ✅ Extract from history JSON content
- ✅ Handle no breadcrumb scope available
- ✅ Extract from multiple documents (use first valid)
- ✅ Handle empty breadcrumb_path
- ✅ Extract from 'sources' field (alternative)
- ✅ Verify max depth limit (3 levels)

### RetrieverAgent (12 tests)
- ✅ Perfect prefix match
- ✅ Partial match (scope longer than chunk)
- ✅ No match
- ✅ Case insensitive matching
- ✅ Empty scope handling
- ✅ Structural prior with high match
- ✅ Structural prior with low match
- ✅ Structural prior with medium match
- ✅ Bayesian re-ranking boosts matching documents
- ✅ No re-ranking when scope is None
- ✅ Empty scope handling
- ✅ Multiple documents re-ranking

### ExtractorAgent (9 tests)
- ✅ Full stitched context (all chunks available)
- ✅ No previous chunk (first chunk)
- ✅ No next chunk (last chunk)
- ✅ No parent context (single-level breadcrumb)
- ✅ Missing chunk IDs in document map
- ✅ Document with no metadata
- ✅ Parent context extraction from breadcrumb_path
- ✅ Multiple documents in collection
- ✅ Empty document collection

## Expected Output

When tests pass, you should see:

```
================================================================================
PHASE 1: UNIT TESTS - Hybrid Metadata Management System
================================================================================

[Component 1 Tests]
✅ Test 1 passed
✅ Test 2 passed
...

================================================================================
PHASE 1 TEST SUMMARY
================================================================================
  STEP_DEFINER        : ✅ PASSED
  RETRIEVER           : ✅ PASSED
  EXTRACTOR           : ✅ PASSED
================================================================================

🎉 All Phase 1 unit tests passed!
   Ready to proceed to Phase 2 (Integration Tests)
```

## Troubleshooting

### Import Errors
If you get import errors, make sure you're running from the project root:
```bash
cd AlgoverseMultiAgent-MCP
python test_step_definer_breadcrumb_extraction.py
```

### Test Failures
If a test fails:
1. Read the error message carefully
2. Check which assertion failed
3. Verify the component implementation matches expected behavior
4. Fix the issue and re-run the test

### Common Issues

**"Module not found" errors:**
- Ensure you're in the correct directory
- Check that `agents/` directory is accessible

**"AttributeError" errors:**
- Verify the method exists in the component
- Check method name spelling

**Assertion failures:**
- Review the expected vs actual values
- Check component logic matches test expectations

## Next Steps

After all Phase 1 tests pass:
1. ✅ Verify each component works in isolation
2. ✅ Fix any bugs found
3. ✅ Proceed to Phase 2 (Integration Tests)

## Notes

- These tests use **mock data** - no actual document loading required
- Tests are **fast** - should complete in seconds
- Tests are **isolated** - each component tested independently
- Tests are **deterministic** - same input = same output

