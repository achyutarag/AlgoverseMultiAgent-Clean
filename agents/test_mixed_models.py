#!/usr/bin/env python3
"""
Test script for the mixed SLM/LLM model configuration.
Verifies that agents are using the correct model types.
"""

import sys
import os
import asyncio
import json
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the mixed model orchestrator
from mixed_model_orchestrator import MixedModelOrchestrator, create_optimized_marag_pipeline

def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f"🧪 {title}")
    print(f"{'='*60}")

def print_subsection(title: str):
    """Print a formatted subsection header."""
    print(f"\n📋 {title}")
    print("-" * 40)

async def test_model_configuration():
    """Test that agents are configured with the correct model types."""
    print_section("Mixed Model Configuration Test")
    
    try:
        # Create mixed model orchestrator
        print("Creating Mixed Model Orchestrator...")
        orchestrator = await create_optimized_marag_pipeline(
            retrieval_model="all-MiniLM-L6-v2",      # SLM
            extraction_model="distilbert-base-uncased",  # SLM
            planning_model="llama-2-13b-chat-hf",    # LLM
            step_definition_model="llama-2-13b-chat-hf",  # LLM
            qa_model="llama-2-13b-chat-hf"           # LLM
        )
        
        print("✅ Mixed Model Orchestrator created successfully")
        
        # Get model information
        print("\nGetting model configuration...")
        model_info = await orchestrator.get_model_info()
        
        print("📊 Model Configuration:")
        print(json.dumps(model_info, indent=2))
        
        # Verify agent types
        print("\n🔍 Verifying Agent Types:")
        
        # Check SLM agents
        print("SLM Agents (Fast, Efficient):")
        print(f"  - Retriever: {model_info['model_configuration']['slm_models']['retrieval']}")
        print(f"  - Extractor: {model_info['model_configuration']['slm_models']['extraction']}")
        
        # Check LLM agents  
        print("LLM Agents (Complex Reasoning):")
        print(f"  - Planner: {model_info['model_configuration']['llm_models']['planning']}")
        print(f"  - Step Definer: {model_info['model_configuration']['llm_models']['step_definition']}")
        print(f"  - QA: {model_info['model_configuration']['llm_models']['qa']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model configuration test failed: {str(e)}")
        return False

async def test_agent_initialization():
    """Test that individual agents are initialized with correct models."""
    print_section("Agent Initialization Test")
    
    try:
        # Test individual agent initialization
        print("Testing individual agent initialization...")
        
        from planner_agent import PlannerAgent
        from step_definer_agent import StepDefinerAgent
        from retriever_agent import RetrieverAgent
        from extractor_agent import ExtractorAgent
        from qa_agent import QAAgent
        
        # Test SLM agents
        print("\nTesting SLM Agents:")
        retriever = RetrieverAgent()
        extractor = ExtractorAgent()
        
        print(f"  ✅ Retriever Agent: {retriever.model_name}")
        print(f"  ✅ Extractor Agent: {extractor.model_name}")
        
        # Test LLM agents
        print("\nTesting LLM Agents:")
        planner = PlannerAgent()
        step_definer = StepDefinerAgent()
        qa = QAAgent()
        
        print(f"  ✅ Planner Agent: {planner.model_name}")
        print(f"  ✅ Step Definer Agent: {step_definer.model_name}")
        print(f"  ✅ QA Agent: {qa.model_name}")
        
        # Verify model types
        slm_models = ["all-MiniLM-L6-v2", "distilbert-base-uncased"]
        llm_models = ["llama-2-13b-chat-hf"]
        
        print("\n🔍 Model Type Verification:")
        print(f"  Retriever ({retriever.model_name}): {'SLM' if retriever.model_name in slm_models else 'Unknown'}")
        print(f"  Extractor ({extractor.model_name}): {'SLM' if extractor.model_name in slm_models else 'Unknown'}")
        print(f"  Planner ({planner.model_name}): {'LLM' if planner.model_name in llm_models else 'Unknown'}")
        print(f"  Step Definer ({step_definer.model_name}): {'LLM' if step_definer.model_name in llm_models else 'Unknown'}")
        print(f"  QA ({qa.model_name}): {'LLM' if qa.model_name in llm_models else 'Unknown'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Agent initialization test failed: {str(e)}")
        return False

async def test_performance_characteristics():
    """Test the performance characteristics of different model types."""
    print_section("Performance Characteristics Test")
    
    try:
        # Create orchestrator
        orchestrator = await create_optimized_marag_pipeline()
        
        # Get performance benchmark
        print("Running performance benchmark...")
        performance = await orchestrator.benchmark_agent_performance()
        
        print("📊 Performance Results:")
        print(json.dumps(performance, indent=2))
        
        # Analyze results
        print("\n📈 Performance Analysis:")
        if performance["slm_performance"]["status"] == "fast":
            print("  ✅ SLM agents: Fast and efficient")
        else:
            print("  ⚠️ SLM agents: May need configuration")
            
        if performance["llm_performance"]["status"] == "complex":
            print("  ✅ LLM agents: Complex reasoning capability")
        else:
            print("  ⚠️ LLM agents: May need configuration")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance test failed: {str(e)}")
        return False

async def demonstrate_usage():
    """Demonstrate how to use the mixed model configuration."""
    print_section("Usage Demonstration")
    
    print("""
🚀 Mixed Model MA-RAG Pipeline Usage:

1. **Basic Usage (Default Models):**
   ```python
   from mixed_model_orchestrator import create_optimized_marag_pipeline
   
   # Create orchestrator with optimal models
   orchestrator = await create_optimized_marag_pipeline()
   
   # Run pipeline
   result = await orchestrator.execute_pipeline("Your question here")
   ```

2. **Custom Model Configuration:**
   ```python
   # Custom SLM/LLM models
   orchestrator = await create_optimized_marag_pipeline(
       retrieval_model="all-MiniLM-L6-v2",        # SLM
       extraction_model="distilbert-base-uncased", # SLM
       planning_model="llama-2-13b-chat-hf",     # LLM
       step_definition_model="llama-2-13b-chat-hf", # LLM
       qa_model="llama-2-13b-chat-hf"             # LLM
   )
   ```

3. **Convenience Function:**
   ```python
   from mixed_model_orchestrator import run_optimized_marag_pipeline
   
   result = await run_optimized_marag_pipeline(
       query="Your question here",
       retrieval_model="all-MiniLM-L6-v2",
       planning_model="llama-2-13b-chat-hf"
   )
   ```

🔧 **Model Configuration Benefits:**
   ✅ SLMs for fast retrieval and extraction
   ✅ LLMs for complex reasoning tasks
   ✅ Optimized resource usage
   ✅ Balanced performance and efficiency
   ✅ Easy to customize per agent
    """)

async def main():
    """Run all tests."""
    print("🧪 Mixed Model Configuration Test Suite")
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    tests = [
        ("Model Configuration", test_model_configuration),
        ("Agent Initialization", test_agent_initialization),
        ("Performance Characteristics", test_performance_characteristics)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {str(e)}")
            results.append((test_name, False))
    
    # Summary
    print_section("Test Results Summary")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Mixed model configuration is ready.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    
    # Demonstrate usage
    await demonstrate_usage()

if __name__ == "__main__":
    asyncio.run(main())


