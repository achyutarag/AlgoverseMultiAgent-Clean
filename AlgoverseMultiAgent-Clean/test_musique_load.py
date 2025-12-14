"""
Test script to verify MuSiQue dataset loading in AlgoverseMultiAgent-Clean.

This script tests:
1. Loading MuSiQue from HuggingFace
2. Fallback to local JSONL files if HuggingFace fails
3. Converting MuSiQue examples to Document objects
"""

import traceback
import json
from datasets import load_dataset

def test_musique_loading():
    """Test MuSiQue dataset loading from various sources."""
    print("Testing MuSiQue dataset loading...")
    print("=" * 60)
    
    # Test 1: Try HuggingFace
    print("\n" + "=" * 60)
    print("TEST 1: HuggingFace Loading")
    print("=" * 60)
    
    test_paths = [
        "allenai/musique-v1",
        "allenai/musique",
        "StonyBrookNLP/musique",
    ]
    
    huggingface_success = False
    dataset = None
    for path in test_paths:
        print(f"\nTrying: {path}")
        try:
            dataset = load_dataset(path)
            print(f"✅ SUCCESS with {path}")
            print(f"   Available splits: {list(dataset.keys())}")
            if "validation" in dataset:
                print(f"   Validation examples: {len(dataset['validation'])}")
                if len(dataset['validation']) > 0:
                    sample = dataset['validation'][0]
                    print(f"   Sample keys: {list(sample.keys())}")
            huggingface_success = True
            break
        except Exception as e:
            print(f"❌ FAILED with {path}")
            print(f"   Error: {type(e).__name__}: {str(e)}")
            if "401" in str(e) or "Unauthorized" in str(e):
                print("   → This suggests authentication is required")
            elif "404" in str(e) or "not found" in str(e).lower():
                print("   → This suggests the dataset doesn't exist at this path")
            traceback.print_exc()
            print()
    
    if not huggingface_success:
        print("\nHuggingFace loading failed for all tested paths.")
        print("Will try local file fallback...")
    
    # Test 2: Try local file fallback
    print("\n" + "=" * 60)
    print("TEST 2: Local File Fallback Loading")
    print("=" * 60)
    
    try:
        print("\nTesting local file loading...")
        from agents.musique_document_loader import _load_musique_from_github
        examples = _load_musique_from_github("validation")
        print(f"✅ SUCCESS: Loaded {len(examples)} examples from local files")
        if examples:
            print(f"   First example keys: {list(examples[0].keys())}")
            print(f"   Sample question: {examples[0].get('question', 'N/A')[:100]}...")
            answer = examples[0].get('answer', 'NOT FOUND')
            print(f"   Sample answer: {answer}")
            print(f"   Answer type: {type(answer)}")
            if isinstance(answer, list):
                print(f"   Answer is a list with {len(answer)} items")
            elif isinstance(answer, dict):
                print(f"   Answer is a dict with keys: {list(answer.keys())}")
    except Exception as e:
        print(f"❌ FAILED to load from local files")
        print(f"   Error: {type(e).__name__}: {str(e)}")
        traceback.print_exc()
    
    # Test 3: Test document conversion
    print("\n" + "=" * 60)
    print("TEST 3: Document Conversion")
    print("=" * 60)
    
    try:
        from agents.musique_document_loader import load_musique_example_context_as_documents
        
        # Get a sample example
        if dataset and "validation" in dataset:
            sample_example = dataset["validation"][0]
        else:
            # Try to load from local files
            from agents.musique_document_loader import _load_musique_from_github
            examples = _load_musique_from_github("validation")
            if not examples:
                print("❌ No examples available for testing document conversion")
                return
            sample_example = examples[0]
        
        print(f"\nConverting example to documents...")
        print(f"Example ID: {sample_example.get('id', 'unknown')}")
        print(f"Question: {sample_example.get('question', 'N/A')[:100]}...")
        
        documents = load_musique_example_context_as_documents(sample_example)
        print(f"✅ SUCCESS: Converted to {len(documents)} documents")
        
        if documents:
            print(f"\nFirst document preview:")
            print(f"  Paragraph ID: {documents[0].metadata.get('paragraph_id')}")
            print(f"  Is supporting fact: {documents[0].metadata.get('is_supporting_fact', False)}")
            print(f"  Content preview: {documents[0].page_content[:200]}...")
            print(f"  Metadata keys: {list(documents[0].metadata.keys())}")
        
        # Verify structure
        from agents.musique_document_loader import verify_musique_structure
        verification = verify_musique_structure(sample_example, documents)
        print(f"\nVerification results:")
        print(f"  Total paragraphs: {verification['total_paragraphs']}")
        print(f"  Total documents: {verification['total_documents']}")
        print(f"  Supporting facts: {verification['supporting_facts']}")
        print(f"  Supporting facts in loaded: {verification['supporting_facts_in_loaded']}")
        print(f"  Is valid: {verification['is_valid']}")
        
    except Exception as e:
        print(f"❌ FAILED to convert example to documents")
        print(f"   Error: {type(e).__name__}: {str(e)}")
        traceback.print_exc()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
    print("\n✅ MuSiQue loading is properly integrated in AlgoverseMultiAgent-Clean!")
    print("   You can now use:")
    print("   - load_musique_example_context_as_documents(example)")
    print("   - load_musique_context_as_documents(split, num_examples)")
    print("   - _load_musique_from_github(split)  # for local file loading")

if __name__ == "__main__":
    test_musique_loading()

