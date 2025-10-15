"""
Test script to verify document retrieval is working without hitting API limits.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import logging
from agents.hotpotqa_document_loader import load_hotpotqa_context_as_documents
from agents.retriever_agent import RetrieverAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_retrieval_only():
    """Test just the retrieval component without LLM agents."""
    
    print("🔍 Testing Document Retrieval (No API Calls)...")
    print("=" * 50)
    
    try:
        # Load documents
        print("📚 Loading documents...")
        docs = load_hotpotqa_context_as_documents("validation", num_examples=5)
        print(f"✅ Loaded {len(docs)} documents")
        
        # Initialize retriever with lower threshold
        print("\n🔧 Initializing retriever...")
        retriever = RetrieverAgent(
            documents=docs,
            model_name="all-MiniLM-L6-v2",
            model_config={"use_cuda": False},
            top_k=5,
            min_similarity=0.4  # Lower threshold
        )
        print("✅ Retriever initialized")
        
        # Test the exact queries from the pipeline
        test_queries = [
            "Scott Derrickson nationality",
            "Ed Wood nationality",
            "Scott Derrickson birthplace"
        ]
        
        for query in test_queries:
            print(f"\n🔍 Testing query: '{query}'")
            
            retrieval_input = {
                "query": query,
                "k": 5,
                "min_similarity": 0.4
            }
            
            response = await retriever.process(retrieval_input)
            
            if response.metadata.get("error"):
                print(f"  ❌ Error: {response.metadata.get('error')}")
            else:
                # Parse the response
                import json
                try:
                    result = json.loads(response.content)
                    num_docs = len(result.get("documents", []))
                    avg_similarity = result.get("average_similarity", 0.0)
                    print(f"  ✅ Retrieved {num_docs} documents (avg similarity: {avg_similarity:.3f})")
                    
                    # Show retrieved documents
                    for i, doc in enumerate(result.get("documents", [])[:3]):  # Show first 3
                        title = doc.get("metadata", {}).get("title", "Unknown")
                        score = doc.get("score", 0.0)
                        content_preview = doc.get("page_content", "")[:100] + "..."
                        print(f"    📄 {i+1}. '{title}' (score: {score:.3f})")
                        print(f"       Content: {content_preview}")
                        
                except json.JSONDecodeError as e:
                    print(f"  ❌ JSON parse error: {e}")
                    print(f"  Raw response: {response.content[:200]}...")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        logger.error(f"Test failed: {str(e)}")

if __name__ == "__main__":
    asyncio.run(test_retrieval_only())
