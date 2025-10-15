"""
Test script for HotpotQA multi-agent pipeline with proper document loading.
This script loads HotpotQA documents and tests the full MA-RAG pipeline.
"""
import sys
import os
# Add parent directory to path so we can import agents module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import json
from agents.planner_agent import PlannerAgent
from agents.retriever_agent import RetrieverAgent
from agents.extractor_agent import ExtractorAgent
from agents.qa_agent import QAAgent
from agents.llm_wrapper import get_llm, LLMConfig
from agents.tokenization_utils import TokenizationUtils
from agents.hotpotqa_document_loader import (
    load_hotpotqa_context_as_documents,
    get_hotpotqa_sample_questions
)
from langchain.schema import Document

async def test_hotpotqa_pipeline():
    """Test the full multi-agent pipeline with HotpotQA documents."""
    print("🚀 Testing HotpotQA Multi-Agent Pipeline...")
    print("=" * 60)
    
    try:
        # Step 1: Load HotpotQA documents
        print("📚 Loading HotpotQA documents...")
        documents = load_hotpotqa_context_as_documents(
            dataset_split="validation", 
            num_examples=50  # Small sample for testing
        )
        print(f"✅ Loaded {len(documents)} documents")
        
        # Step 2: Initialize agents
        print("\n🔧 Initializing agents...")
        
        # LLM Configuration
        llm_config = LLMConfig(
            model_name="gemini-2.5-flash",
            model_type="google_gemini"
        )
        
        # Initialize all agents
        planner = PlannerAgent()
        retriever = RetrieverAgent(
            documents=documents,
            model_config={"use_cuda": False},  # Add model_config
            model_name="all-MiniLM-L6-v2",
            top_k=5,
            min_similarity=0.6
        )
        extractor = ExtractorAgent()
        qa_agent = QAAgent()
        
        print("✅ All agents initialized")
        
        # Step 3: Get sample questions
        print("\n📝 Getting sample questions...")
        sample_questions = get_hotpotqa_sample_questions("validation", num_questions=3)
        
        # Step 4: Test each question
        for i, sample in enumerate(sample_questions):
            print(f"\n{'='*60}")
            print(f"🧪 Testing Question {i+1}/{len(sample_questions)}")
            print(f"{'='*60}")
            
            question = sample['question']
            expected_answer = sample['answer']
            context_docs = sample['context_documents']
            
            print(f"📝 Question: {question}")
            print(f"🎯 Expected Answer: {expected_answer}")
            print(f"📚 Available Context Documents: {len(context_docs)}")
            
            # Step 4a: Test Planner Agent
            print(f"\n🔍 Step 1: Planning...")
            plan_result = await planner.process({'query': question})
            
            if plan_result.metadata.get('error'):
                print(f"❌ Planning failed: {plan_result.metadata['error']}")
                continue
            
            # Parse plan
            try:
                clean_plan = TokenizationUtils.strip_markdown_json(plan_result.content)
                plan_data = json.loads(clean_plan)
                steps = plan_data.get('steps', [])
                print(f"✅ Plan generated with {len(steps)} steps")
                
                # Show first step
                if steps:
                    print(f"📋 First step: {steps[0].get('description', 'No description')}")
                    
            except json.JSONDecodeError as e:
                print(f"❌ Failed to parse plan: {e}")
                continue
            
            # Step 4b: Test Retrieval for first step
            if steps:
                print(f"\n🔍 Step 2: Retrieval...")
                first_step = steps[0]
                step_query = first_step.get('description', question)
                
                retrieval_result = await retriever.process({
                    'query': step_query,
                    'k': 3,
                    'min_similarity': 0.1  # Lower threshold for better results
                })
                
                if retrieval_result.metadata.get('error'):
                    print(f"❌ Retrieval failed: {retrieval_result.metadata['error']}")
                    continue
                
                retrieved_docs = retrieval_result.metadata.get('documents', [])
                print(f"✅ Retrieved {len(retrieved_docs)} documents")
                
                if retrieved_docs:
                    print(f"📄 First retrieved doc: {retrieved_docs[0]['metadata']['title']}")
                    print(f"📄 Content preview: {retrieved_docs[0]['page_content'][:150]}...")
                
                # Step 4c: Test Extraction
                print(f"\n🔍 Step 3: Extraction...")
                extract_result = await extractor.process({
                    'query': step_query,
                    'documents': retrieved_docs,
                    'max_documents': 2,
                    'min_relevance': 0.2  # Much lower relevance threshold
                })
                
                if extract_result.metadata.get('error'):
                    print(f"❌ Extraction failed: {extract_result.metadata['error']}")
                    continue
                
                extracted_passages = extract_result.metadata.get('extracted_passages', [])
                print(f"✅ Extracted {len(extracted_passages)} relevant passages")
                
                # Step 4d: Test QA Agent
                print(f"\n🔍 Step 4: Question Answering...")
                qa_result = await qa_agent.process({
                    'question': question,
                    'context': extracted_passages,
                    'min_confidence': 0.3
                })
                
                if qa_result.metadata.get('error'):
                    print(f"❌ QA failed: {qa_result.metadata['error']}")
                    continue
                
                # Parse QA result
                try:
                    qa_data = json.loads(qa_result.content)
                    answer = qa_data.get('answer', qa_result.content)
                    confidence = qa_data.get('confidence', 0.0)
                except:
                    answer = qa_result.content
                    confidence = 0.5
                
                print(f"✅ Generated Answer: {answer}")
                print(f"📊 Confidence: {confidence:.2f}")
                
                # Compare with expected answer
                print(f"\n📊 Results Summary:")
                print(f"   Expected: {expected_answer}")
                print(f"   Generated: {answer}")
                print(f"   Confidence: {confidence:.2f}")
                
        print(f"\n{'='*60}")
        print("🎉 HotpotQA Pipeline Test Completed!")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"❌ Error in pipeline test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_hotpotqa_pipeline())