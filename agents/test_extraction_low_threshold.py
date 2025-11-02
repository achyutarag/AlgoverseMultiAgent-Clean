"""
Test script to verify extraction works with lower thresholds.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import logging
import json
from agents.hotpotqa_document_loader import load_hotpotqa_context_as_documents
from agents.extractor_agent import ExtractorAgent

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

async def test_extraction_with_low_threshold():
    """Test extraction with very low relevance threshold."""
    
    print("🔍 Testing Extraction with Low Threshold...")
    print("=" * 50)
    
    try:
        # Load documents
        print("📚 Loading documents...")
        docs = load_hotpotqa_context_as_documents("validation", num_examples=3)
        print(f"✅ Loaded {len(docs)} documents")
        
        # Initialize extractor
        print("\n🔧 Initializing extractor...")
        extractor = ExtractorAgent(
            model_name="gemini-2.5-flash",
            model_config={"use_cuda": False},
            temperature=0.1
        )
        print("✅ Extractor initialized")
        
        # Test extraction with very low threshold
        test_query = "Find the nationality of Scott Derrickson"
        
        # Find relevant documents
        relevant_docs = []
        for doc in docs:
            if "Scott Derrickson" in doc.page_content:
                relevant_docs.append({
                    "id": doc.metadata.get("article_id", "unknown"),
                    "page_content": doc.page_content,
                    "metadata": doc.metadata
                })
        
        print(f"\n🔍 Testing extraction for: '{test_query}'")
        print(f"📄 Found {len(relevant_docs)} relevant documents")
        
        if relevant_docs:
            # Test extraction with very low threshold
            extraction_input = {
                "query": test_query,
                "documents": relevant_docs[:2],  # Use first 2 documents
                "min_relevance": 0.1  # Very low threshold
            }
            
            response = await extractor.process(extraction_input)
            
            if response.metadata.get("error"):
                print(f"❌ Error: {response.metadata.get('error')}")
            else:
                print("✅ Extraction completed!")
                
                # Try to parse the JSON
                try:
                    result = json.loads(response.content)
                    print("✅ JSON parsing successful!")
                    print(f"📊 Query: {result.get('query', 'N/A')}")
                    print(f"📊 Extracted passages: {len(result.get('extracted_passages', []))}")
                    print(f"📊 Aggregated evidence: {result.get('aggregated_evidence', 'N/A')[:100]}...")
                    
                    # Show extracted passages
                    passages = result.get('extracted_passages', [])
                    if passages:
                        print(f"\n📄 Extracted passages:")
                        for i, passage in enumerate(passages):
                            print(f"   {i+1}. Relevance: {passage.get('relevance', 0.0):.2f}")
                            print(f"      Text: {passage.get('text', 'N/A')[:100]}...")
                            print(f"      Reasoning: {passage.get('reasoning', 'N/A')}")
                    else:
                        print("❌ No passages extracted even with low threshold")
                        print(f"📄 Raw response: {response.content[:500]}...")
                    
                except json.JSONDecodeError as e:
                    print(f"❌ JSON parsing failed: {e}")
                    print(f"📄 Raw response: {response.content[:500]}...")
        else:
            print("❌ No relevant documents found for testing")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        logger.error(f"Test failed: {str(e)}")

if __name__ == "__main__":
    asyncio.run(test_extraction_with_low_threshold())

