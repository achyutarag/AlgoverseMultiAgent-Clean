"""
Test script to check similarity scores and debug retrieval issues.
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

async def test_retrieval_similarity():
    """Test retrieval with different similarity thresholds."""
    
    print("🔍 Testing Retrieval Similarity Scores...")
    print("=" * 50)
    
    try:
        # Load documents
        print("📚 Loading documents...")
        docs = load_hotpotqa_context_as_documents("validation", num_examples=5)
        print(f"✅ Loaded {len(docs)} documents")
        
        # Initialize retriever
        print("\n🔧 Initializing retriever...")
        retriever = RetrieverAgent(
            documents=docs,
            model_name="all-MiniLM-L6-v2",
            model_config={"use_cuda": False},  # Use CPU
            top_k=10,
            min_similarity=0.0  # Start with no threshold
        )
        print("✅ Retriever initialized")
        
        # Test queries
        test_queries = [
            "Scott Derrickson nationality",
            "Ed Wood nationality", 
            "Scott Derrickson",
            "Ed Wood",
            "American director"
        ]
        
        for query in test_queries:
            print(f"\n🔍 Testing query: '{query}'")
            
            # Test with different thresholds
            for threshold in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
                retrieval_input = {
                    "query": query,
                    "k": 5,
                    "min_similarity": threshold
                }
                
                response = await retriever.process(retrieval_input)
                
                if response.metadata.get("error"):
                    print(f"  ❌ Threshold {threshold}: Error - {response.metadata.get('error')}")
                else:
                    # Parse the response
                    import json
                    clean_response = response.content
                    try:
                        result = json.loads(clean_response)
                        num_docs = len(result.get("documents", []))
                        avg_similarity = result.get("average_similarity", 0.0)
                        print(f"  📊 Threshold {threshold}: {num_docs} docs, avg similarity: {avg_similarity:.3f}")
                        
                        # Show first document if any
                        if num_docs > 0:
                            first_doc = result["documents"][0]
                            title = first_doc.get("metadata", {}).get("title", "Unknown")
                            score = first_doc.get("score", 0.0)
                            print(f"    📄 First doc: '{title}' (score: {score:.3f})")
                    except json.JSONDecodeError:
                        print(f"  ❌ Threshold {threshold}: JSON parse error")
            
            print()  # Empty line between queries
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        logger.error(f"Test failed: {str(e)}")

if __name__ == "__main__":
    asyncio.run(test_retrieval_similarity())
