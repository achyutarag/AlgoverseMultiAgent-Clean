"""
Test script to validate breadcrumb extraction quality.

Tests on sample MuSiQue examples and reports:
- Extraction success rate
- Breadcrumb accuracy (manual review)
- Entity extraction quality
"""

import asyncio
import json
from pathlib import Path
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.musique_document_loader import load_musique_context_as_documents
from agents.breadcrumb_extractor import BreadcrumbExtractor

async def test_breadcrumb_quality(num_examples: int = 10):
    """Test breadcrumb extraction on sample examples."""
    
    print("=" * 80)
    print("BREADCRUMB EXTRACTION QUALITY TEST")
    print("=" * 80)
    
    # Load sample documents
    print(f"\n📚 Loading {num_examples} examples from MuSiQue...")
    documents = load_musique_context_as_documents(
        dataset_split="validation",
        num_examples=num_examples,
        include_metadata=True
    )
    
    print(f"✅ Loaded {len(documents)} documents")
    
    # Group by example_id
    examples = {}
    for doc in documents:
        example_id = doc.metadata.get("example_id", "unknown")
        if example_id not in examples:
            examples[example_id] = []
        examples[example_id].append(doc)
    
    print(f"✅ Grouped into {len(examples)} examples\n")
    
    # Extract breadcrumbs for all documents
    extractor = BreadcrumbExtractor()
    
    # Process documents with breadcrumb extraction
    # Track example-level root entities for deeper hierarchies
    processed_docs = []
    example_root_entities = {}  # Track root entity per example
    
    for example_id, docs in examples.items():
        previous_breadcrumbs = None
        example_root_entity = None  # Track first paragraph's main entity as root
        
        for i, doc in enumerate(docs):
            # Extract breadcrumbs with root entity support
            breadcrumb_result = extractor.extract_breadcrumbs(
                text=doc.page_content,
                paragraph_id=doc.metadata.get("paragraph_id"),
                example_id=example_id,
                previous_breadcrumbs=previous_breadcrumbs,
                example_root_entity=example_root_entity  # Pass root entity for depth
            )
            
            # If this is the first paragraph, extract and store root entity
            if i == 0 and breadcrumb_result["breadcrumb_path"]:
                # Use first entity from first paragraph as root
                first_entity = breadcrumb_result["breadcrumb_path"][0]
                if len(first_entity) > 2 and first_entity.lower() not in extractor.COMMON_PHRASES:
                    example_root_entity = first_entity
                    example_root_entities[example_id] = example_root_entity
            
            # Add breadcrumb metadata to doc
            doc.metadata.update({
                "breadcrumb_path": breadcrumb_result["breadcrumb_path"],
                "breadcrumb_string": breadcrumb_result["breadcrumb_string"],
                "breadcrumb_depth": breadcrumb_result["breadcrumb_depth"],
                "breadcrumb_confidence": breadcrumb_result["confidence"],
                "breadcrumb_entities": breadcrumb_result["entities"],
                "example_root_entity": example_root_entity  # Store for reference
            })
            
            processed_docs.append(doc)
            previous_breadcrumbs = breadcrumb_result["breadcrumb_path"]
    
    # Analyze breadcrumb quality
    stats = {
        "total_docs": len(processed_docs),
        "docs_with_breadcrumbs": 0,
        "docs_without_breadcrumbs": 0,
        "avg_confidence": 0.0,
        "avg_depth": 0.0,
        "examples_analyzed": 0
    }
    
    # Show sample breadcrumbs
    print("=" * 80)
    print("SAMPLE BREADCRUMB EXTRACTIONS")
    print("=" * 80)
    
    sample_count = 0
    for example_id, docs in list(examples.items())[:5]:  # Show first 5 examples
        print(f"\n📋 Example: {example_id}")
        print("-" * 80)
        
        # Get processed docs for this example
        example_docs = [d for d in processed_docs if d.metadata.get("example_id") == example_id]
        
        for i, doc in enumerate(example_docs[:3]):  # Show first 3 paragraphs per example
            breadcrumb_path = doc.metadata.get("breadcrumb_path", [])
            breadcrumb_string = doc.metadata.get("breadcrumb_string", "Unknown")
            confidence = doc.metadata.get("breadcrumb_confidence", 0.0)
            entities = doc.metadata.get("breadcrumb_entities", [])
            
            # Show preview
            text_preview = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
            
            print(f"\n  Paragraph {i+1}:")
            print(f"    Text: {text_preview}")
            print(f"    Breadcrumb: {breadcrumb_string}")
            print(f"    Path: {breadcrumb_path}")
            print(f"    Entities: {entities[:3]}")  # Show first 3
            print(f"    Confidence: {confidence:.2f}")
            print(f"    Depth: {len(breadcrumb_path)}")
            
            sample_count += 1
            if sample_count >= 15:  # Limit output
                break
        
        if sample_count >= 15:
            break
    
    # Calculate statistics for all documents (ONCE - fix double-counting bug)
    for doc in processed_docs:
        breadcrumb_path = doc.metadata.get("breadcrumb_path", [])
        confidence = doc.metadata.get("breadcrumb_confidence", 0.0)
        
        if breadcrumb_path and len(breadcrumb_path) > 0:  # Check it's not empty
            stats["docs_with_breadcrumbs"] += 1
            stats["avg_confidence"] += confidence
            stats["avg_depth"] += len(breadcrumb_path)
        else:
            stats["docs_without_breadcrumbs"] += 1
    
    # Calculate averages
    if stats["docs_with_breadcrumbs"] > 0:
        stats["avg_confidence"] /= stats["docs_with_breadcrumbs"]
        stats["avg_depth"] /= stats["docs_with_breadcrumbs"]
    
    # Print summary
    print("\n" + "=" * 80)
    print("QUALITY SUMMARY")
    print("=" * 80)
    print(f"Total Documents: {stats['total_docs']}")
    print(f"Documents with Breadcrumbs: {stats['docs_with_breadcrumbs']} ({stats['docs_with_breadcrumbs']/stats['total_docs']*100:.1f}%)")
    print(f"Documents without Breadcrumbs: {stats['docs_without_breadcrumbs']} ({stats['docs_without_breadcrumbs']/stats['total_docs']*100:.1f}%)")
    print(f"Average Confidence: {stats['avg_confidence']:.2f}")
    print(f"Average Depth: {stats['avg_depth']:.2f}")
    
    # Quality assessment
    print("\n" + "=" * 80)
    print("QUALITY ASSESSMENT")
    print("=" * 80)
    
    success_rate = stats['docs_with_breadcrumbs'] / stats['total_docs'] if stats['total_docs'] > 0 else 0
    
    if success_rate >= 0.8:
        print("✅ EXCELLENT: >80% extraction success rate")
    elif success_rate >= 0.6:
        print("⚠️  GOOD: 60-80% extraction success rate (may need refinement)")
    else:
        print("❌ NEEDS IMPROVEMENT: <60% extraction success rate")
    
    if stats['avg_confidence'] >= 0.7:
        print("✅ EXCELLENT: High confidence scores")
    elif stats['avg_confidence'] >= 0.5:
        print("⚠️  MODERATE: Medium confidence scores")
    else:
        print("❌ LOW: Low confidence scores")
    
    # Recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    
    if success_rate >= 0.7 and stats['avg_confidence'] >= 0.6:
        print("✅ Breadcrumb extraction quality is GOOD ENOUGH for testing")
        print("   → Proceed with Phase 2 (Search Schema + Post-filtering)")
    else:
        print("⚠️  Consider improvements:")
        if success_rate < 0.7:
            print("   - Add more entity extraction patterns")
            print("   - Improve relationship detection")
        if stats['avg_confidence'] < 0.6:
            print("   - Refine confidence calculation")
            print("   - Add more hierarchical patterns")
    
    return stats

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test breadcrumb extraction quality")
    parser.add_argument("--num_examples", type=int, default=10, help="Number of examples to test")
    args = parser.parse_args()
    
    asyncio.run(test_breadcrumb_quality(num_examples=args.num_examples))

