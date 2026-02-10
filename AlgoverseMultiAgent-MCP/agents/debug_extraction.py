"""
Debug script to see what the LLM is actually generating for extraction.
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

async def debug_extraction():
    """Debug what the LLM is generating."""
    
    print("🔍 Debugging LLM Extraction Output...")
    print("=" * 50)
    
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
            # Show the document content
            print(f"\n📄 Document content:")
            print(f"Title: {relevant_docs[0]['metadata'].get('title', 'Unknown')}")
            print(f"Content: {relevant_docs[0]['page_content'][:200]}...")
            
            # Test extraction with very low threshold
            extraction_input = {
                "query": test_query,
                "documents": relevant_docs[:1],  # Use just 1 document
                "min_relevance": 0.1  # Very low threshold
            }
            
            response = await extractor.process(extraction_input)
            
            print(f"\n📋 Raw LLM Response:")
            print(f"Length: {len(response.content)} characters")
            print(f"Content: {response.content}")
            
            # Try to parse the JSON
            try:
                result = json.loads(response.content)
                print(f"\n✅ JSON parsing successful!")
                print(f"📊 Query: {result.get('query', 'N/A')}")
                print(f"📊 Extracted passages count: {len(result.get('extracted_passages', []))}")
                
                passages = result.get('extracted_passages', [])
                if passages:
                    print(f"\n📄 Extracted passages:")
                    for i, passage in enumerate(passages):
                        print(f"   {i+1}. Relevance: {passage.get('relevance', 0.0)}")
                        print(f"      Text: {passage.get('text', 'N/A')}")
                        print(f"      Reasoning: {passage.get('reasoning', 'N/A')}")
                else:
                    print(f"\n❌ No passages in JSON response")
                    print(f"📊 Extraction reasoning: {result.get('extraction_reasoning', 'N/A')}")
                    print(f"📊 Aggregated evidence: {result.get('aggregated_evidence', 'N/A')}")
                
            except json.JSONDecodeError as e:
                print(f"❌ JSON parsing failed: {e}")
                
        else:
            print("❌ No relevant documents found for testing")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        logger.error(f"Test failed: {str(e)}")

if __name__ == "__main__":
    asyncio.run(debug_extraction())
