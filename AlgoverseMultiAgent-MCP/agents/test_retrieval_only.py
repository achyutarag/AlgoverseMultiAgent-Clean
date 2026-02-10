"""
Enhanced test script to debug document retrieval issues.
Tests retrieval with actual failing queries and provides detailed diagnostics.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import logging
import json
from agents.hotpotqa_document_loader import load_hotpotqa_context_as_documents
from agents.retriever_agent import RetrieverAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Add this function to your test_retrieval_only.py file (around line 19, before test_retrieval_only)

async def find_and_load_chipmunks_documents():
    """Search the full dataset for Chipmunks question and load its context documents."""
    from datasets import load_dataset
    from agents.hotpotqa_document_loader import load_hotpotqa_example_context_as_documents
    
    print("🔍 Searching full dataset for Chipmunks question...")
    dataset = load_dataset("hotpot_qa", "distractor")
    validation_data = dataset["validation"]
    
    # Search for Chipmunks-related questions
    chipmunks_keywords = ["chipmunk", "chipmunks", "alvin", "bagdasarian", "david seville", "seville"]
    
    matching_examples = []
    for i, example in enumerate(validation_data):
        question_lower = example['question'].lower()
        answer_lower = example.get('answer', '').lower()
        
        # Check if question or answer mentions Chipmunks
        if any(kw in question_lower or kw in answer_lower for kw in chipmunks_keywords):
            matching_examples.append((i, example))
            print(f"   ✅ Found match at index {i}:")
            print(f"      Question: {example['question']}")
            print(f"      Answer: {example.get('answer', 'N/A')}")
    
    if matching_examples:
        print(f"\n   Found {len(matching_examples)} Chipmunks-related examples!")
        # Load context documents from the first matching example
        example_idx, example = matching_examples[0]
        print(f"   Loading context documents from example {example_idx}...")
        docs = load_hotpotqa_example_context_as_documents(example)
        print(f"   ✅ Loaded {len(docs)} context documents")
        return docs
    else:
        print("   ❌ No Chipmunks questions found in validation set!")
        print("   Checking train set...")
        train_data = dataset["train"]
        for i, example in enumerate(train_data):
            question_lower = example['question'].lower()
            if any(kw in question_lower for kw in chipmunks_keywords):
                print(f"   ✅ Found in train set at index {i}: {example['question']}")
                docs = load_hotpotqa_example_context_as_documents(example)
                print(f"   ✅ Loaded {len(docs)} context documents")
                return docs
        return None


async def test_retrieval_only():
    """Test retrieval with actual failing queries and comprehensive diagnostics."""
    
    print("🔍 Testing Document Retrieval (Enhanced Debugging)...")
    print("=" * 70)
    
    try:
        # Load MORE documents to ensure we have the failing query's documents
        print("📚 Loading documents...")
        chipmunks_docs = await find_and_load_chipmunks_documents()
        if chipmunks_docs:
            print("\n✅ Using Chipmunks-specific documents from dataset!")
            docs = chipmunks_docs
            # Also load some other documents for comparison
            other_docs = load_hotpotqa_context_as_documents("validation", num_examples=20)
            docs.extend(other_docs)  # Combine them
        else:
            print("\n⚠️  Chipmunks documents not found, loading general corpus...")
            docs = load_hotpotqa_context_as_documents("validation", num_examples=100)  # Load more
        print(f"✅ Loaded {len(docs)} documents")
        
        # Check if corpus contains relevant documents
        print("\n🔍 Analyzing corpus...")
        corpus_titles = [doc.metadata.get("title", "") for doc in docs]
        print(f"   Total unique articles: {len(set(corpus_titles))}")
        
        # Search for Chipmunks-related documents
        chipmunks_keywords = ["chipmunk", "chipmunks", "alvin", "bagdasarian", "david seville"]
        relevant_titles = [title for title in corpus_titles if any(kw in title.lower() for kw in chipmunks_keywords)]
        if relevant_titles:
            print(f"   ✅ Found {len(relevant_titles)} Chipmunks-related documents:")
            for title in relevant_titles[:5]:
                print(f"      - {title}")
        else:
            print(f"   ⚠️  No Chipmunks-related documents found in corpus!")
            print(f"   This might explain why retrieval fails.")
        
        # Initialize retriever
        print("\n🔧 Initializing retriever...")
        retriever = RetrieverAgent(
            documents=docs,
            model_name="all-MiniLM-L6-v2",
            model_config={"use_cuda": False},
            top_k=10,  # Match orchestrator's k
            min_similarity=0.3  # Match orchestrator's threshold
        )
        print("✅ Retriever initialized")
        
        # Test with ACTUAL failing queries from your logs
        test_queries = [
            # The actual failing queries
            "who created The Chipmunks",
            "who created Alvin and the Chipmunks",
            "Ross Bagdasarian Sr. creator of The Chipmunks",
            "creator of The Chipmunks name",
            # Also test with known good queries
            "Scott Derrickson nationality",
            "Ed Wood nationality",
        ]
        
        # Test with different similarity thresholds
        similarity_thresholds = [0.3, 0.4, 0.5, 0.6]
        
        for query in test_queries:
            print(f"\n{'='*70}")
            print(f"🔍 Testing query: '{query}'")
            print(f"{'='*70}")
            
            for min_sim in similarity_thresholds:
                print(f"\n  📊 Threshold: {min_sim}")
                
                retrieval_input = {
                    "query": query,
                    "k": 10,  # Match orchestrator
                    "min_similarity": min_sim
                }
                
                response = await retriever.process(retrieval_input)
                
                if response.metadata.get("error"):
                    print(f"    ❌ Error: {response.metadata.get('error')}")
                else:
                    # Parse the response
                    try:
                        result = json.loads(response.content)
                        docs_retrieved = result.get("documents", [])
                        num_docs = len(docs_retrieved)

                        # Get average similarity from METADATA (not JSON content)
                        avg_similarity = response.metadata.get("average_score", 0.0)

                        # Get scores from JSON if available, otherwise from documents
                        scores = result.get("scores", [])
                        if not scores:
                            scores = [doc.get("score", 0.0) for doc in docs_retrieved if doc.get("score")]

                        print(f"    ✅ Retrieved {num_docs} documents (avg similarity: {avg_similarity:.3f})")
                        if scores:
                            print(f"    📈 Score range: {min(scores):.3f} - {max(scores):.3f}")

                        print("    📚 Top retrieved titles:")
                        for doc in docs_retrieved[:10]:
                            title = doc.get("metadata", {}).get("title", "Unknown")
                            score = doc.get("score", 0.0)
                            print(f"       - {title} (score={score:.3f})")
                        
                        # Show top 3 documents with relevance analysis
                        for i, doc in enumerate(docs_retrieved[:3]):
                            title = doc.get("metadata", {}).get("title", "Unknown")
                            score = doc.get("score", 0.0)
                            content_preview = doc.get("page_content", "")[:150]
                            
                            # Check semantic relevance
                            query_lower = query.lower()
                            title_lower = title.lower()
                            content_lower = content_preview.lower()
                            
                            has_query_terms = any(term in content_lower or term in title_lower 
                                                 for term in query_lower.split() if len(term) > 3)
                            
                            relevance_indicator = "✅" if has_query_terms else "❌"
                            
                            print(f"    {relevance_indicator} {i+1}. '{title}' (score: {score:.3f})")
                            print(f"       Content: {content_preview}...")
                            if not has_query_terms:
                                print(f"       ⚠️  No semantic connection to query terms!")
                        
                        # If we found relevant docs, stop testing other thresholds
                        if num_docs > 0 and any(
                            any(term in doc.get("metadata", {}).get("title", "").lower() 
                                for term in ["chipmunk", "alvin", "bagdasarian", "seville"])
                            for doc in docs_retrieved[:5]
                        ):
                            print(f"    ✅ Found relevant documents! Stopping threshold tests for this query.")
                            break
                            
                    except json.JSONDecodeError as e:
                        print(f"    ❌ JSON parse error: {e}")
                        print(f"    Raw response: {response.content[:200]}...")
            
            # Additional check: Search corpus directly for query terms
            print(f"\n  🔎 Direct corpus search for query terms...")
            query_terms = [term.lower() for term in query.split() if len(term) > 3]
            matching_docs = []
            for doc in docs:
                title_lower = doc.metadata.get("title", "").lower()
                content_lower = doc.page_content.lower()
                if any(term in title_lower or term in content_lower for term in query_terms):
                    matching_docs.append(doc.metadata.get("title", "Unknown"))
            
            if matching_docs:
                print(f"    ✅ Found {len(matching_docs)} documents containing query terms:")
                for title in matching_docs[:5]:
                    print(f"       - {title}")
            else:
                print(f"    ❌ No documents in corpus contain query terms!")
                print(f"    This confirms retrieval cannot find relevant documents.")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        logger.error(f"Test failed: {str(e)}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(test_retrieval_only())
