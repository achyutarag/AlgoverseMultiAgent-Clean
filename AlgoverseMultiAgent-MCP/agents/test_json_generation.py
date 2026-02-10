"""
Test script to verify JSON generation from LLM agents.
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
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_json_generation():
    """Test JSON generation from extractor agent."""
    
    print("🔍 Testing JSON Generation...")
    print("=" * 40)
    
    try:
        # Load documents
        print("📚 Loading documents...")
        docs = load_hotpotqa_context_as_documents("validation", num_examples=2)
        print(f"✅ Loaded {len(docs)} documents")
        
        # Initialize extractor
        print("\n🔧 Initializing extractor...")
        extractor = ExtractorAgent(
            model_name="models/gemini-flash-lite-latest",
            model_config={"use_cuda": False},
            temperature=0.1
        )
        print("✅ Extractor initialized")
        
        # Test extraction with a simple query
        test_query = "What is Scott Derrickson's nationality?"
        
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
            # Test extraction
            extraction_input = {
                "query": test_query,
                "documents": relevant_docs[:2],  # Use first 2 documents
                "min_relevance": 0.3
            }
            
            response = await extractor.process(extraction_input)
            
            if response.metadata.get("error"):
                print(f"❌ Error: {response.metadata.get('error')}")
            else:
                print("✅ Extraction completed!")
                print(f"📋 Response length: {len(response.content)} characters")
                
                # Try to parse the JSON
                try:
                    result = json.loads(response.content)
                    print("✅ JSON parsing successful!")
                    print(f"📊 Query: {result.get('query', 'N/A')}")
                    print(f"📊 Extracted passages: {len(result.get('extracted_passages', []))}")
                    print(f"📊 Aggregated evidence: {result.get('aggregated_evidence', 'N/A')[:100]}...")
                    
                    # Show first extracted passage
                    passages = result.get('extracted_passages', [])
                    if passages:
                        first_passage = passages[0]
                        print(f"\n📄 First passage:")
                        print(f"   Text: {first_passage.get('text', 'N/A')[:100]}...")
                        print(f"   Relevance: {first_passage.get('relevance', 0.0)}")
                        print(f"   Reasoning: {first_passage.get('reasoning', 'N/A')}")
                    
                except json.JSONDecodeError as e:
                    print(f"❌ JSON parsing failed: {e}")
                    print(f"📄 Raw response: {response.content[:500]}...")
        else:
            print("❌ No relevant documents found for testing")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        logger.error(f"Test failed: {str(e)}")

if __name__ == "__main__":
    asyncio.run(test_json_generation())

