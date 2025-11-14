"""
Simple test script for the HotpotQA pipeline with minimal API calls.
This script tests the pipeline with just one question to avoid quota limits.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import logging
from agents.hotpotqa_document_loader import load_hotpotqa_context_as_documents, get_hotpotqa_sample_questions
from agents.mixed_model_orchestrator import MixedModelOrchestrator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_simple_pipeline():
    """Test the pipeline with a single question to avoid quota limits."""
    
    print("🚀 Testing Simple HotpotQA Pipeline...")
    print("=" * 60)
    
    try:
        # Load documents (smaller sample)
        print("📚 Loading HotpotQA documents...")
        docs = load_hotpotqa_context_as_documents("validation", num_examples=5)
        print(f"✅ Loaded {len(docs)} documents")
        
        # Initialize orchestrator with documents
        print("\n🔧 Initializing orchestrator...")
        orchestrator = MixedModelOrchestrator(
            documents=docs,  # Pass documents to orchestrator
            max_steps=3,  # Reduced steps
            max_subqueries=2,  # Reduced subqueries
            top_k=3,  # Reduced top_k
            min_similarity=0.3  # Even lower similarity threshold
        )
        print("✅ Orchestrator initialized")
        
        # Get one sample question
        print("\n📝 Getting sample question...")
        sample_questions = get_hotpotqa_sample_questions("validation", num_questions=1)
        question_data = sample_questions[0]
        
        print(f"📝 Question: {question_data['question']}")
        print(f"🎯 Expected Answer: {question_data['answer']}")
        print(f"📚 Available Context Documents: {len(question_data['context_documents'])}")
        
        # Test the pipeline
        print("\n🔍 Running pipeline...")
        result = await orchestrator.execute_pipeline(
            query=question_data['question'],
            context={"documents": docs}
        )
        
        print(f"\n✅ Pipeline completed!")
        print(f"📋 Answer: {result.final_answer}")
        print(f"🎯 Expected: {question_data['answer']}")
        print(f"📊 Confidence: {result.confidence}")
        
        # Check if answer matches
        if question_data['answer'].lower() in result.final_answer.lower():
            print("🎉 Answer matches expected result!")
        else:
            print("⚠️ Answer doesn't match expected result")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        logger.error(f"Pipeline test failed: {str(e)}")

if __name__ == "__main__":
    asyncio.run(test_simple_pipeline())
