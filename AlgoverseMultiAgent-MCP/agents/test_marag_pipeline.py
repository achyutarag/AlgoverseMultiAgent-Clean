#!/usr/bin/env python3
"""
Test script for the complete MA-RAG pipeline integration.
Demonstrates how all components work together seamlessly.
"""

import sys
import os
import asyncio
import json
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the main components
from orchestrator import MARAGOrchestrator, run_marag_pipeline
from state_manager import StateManager
from final_assembler import FinalAssembler

# Import agents
from planner_agent import PlannerAgent
from step_definer_agent import StepDefinerAgent
from retriever_agent import RetrieverAgent
from extractor_agent import ExtractorAgent
from qa_agent import QAAgent

def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f"🧪 {title}")
    print(f"{'='*60}")

def print_subsection(title: str):
    """Print a formatted subsection header."""
    print(f"\n📋 {title}")
    print("-" * 40)

async def test_component_initialization():
    """Test that all components can be initialized properly."""
    print_section("Component Initialization Test")
    
    try:
        # Test individual components
        print("Initializing individual components...")
        
        planner = PlannerAgent()
        step_definer = StepDefinerAgent()
        retriever = RetrieverAgent()
        extractor = ExtractorAgent()
        qa = QAAgent()
        state_manager = StateManager()
        final_assembler = FinalAssembler()
        
        print("✅ All individual components initialized successfully")
        
        # Test orchestrator initialization
        print("\nInitializing MA-RAG Orchestrator...")
        orchestrator = MARAGOrchestrator(
            planner_agent=planner,
            step_definer_agent=step_definer,
            retriever_agent=retriever,
            extractor_agent=extractor,
            qa_agent=qa,
            state_manager=state_manager,
            final_assembler=final_assembler
        )
        
        print("✅ MA-RAG Orchestrator initialized successfully")
        
        # Test status
        status = await orchestrator.get_pipeline_status()
        print(f"Pipeline Status: {json.dumps(status, indent=2)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Component initialization failed: {str(e)}")
        return False

async def test_state_management():
    """Test state management functionality."""
    print_section("State Management Test")
    
    try:
        state_manager = StateManager()
        
        # Test execution initialization
        print("Testing execution initialization...")
        await state_manager.initialize_execution(
            execution_id="test_exec_001",
            main_query="What are the environmental benefits of renewable energy?",
            context={"domain": "environmental_science"}
        )
        
        # Test plan update
        print("Testing plan update...")
        test_plan = {
            "main_question": "What are the environmental benefits of renewable energy?",
            "disambiguated_query": "What are the specific environmental benefits of renewable energy sources?",
            "query_type": "multi-hop",
            "steps": [
                {
                    "id": "step_1",
                    "description": "Research types of renewable energy",
                    "dependencies": [],
                    "critical": True
                },
                {
                    "id": "step_2", 
                    "description": "Analyze environmental benefits of each type",
                    "dependencies": ["step_1"],
                    "critical": True
                }
            ]
        }
        
        await state_manager.update_plan(test_plan)
        
        # Test dependency resolution
        print("Testing dependency resolution...")
        ordered_steps = await state_manager.resolve_step_dependencies(test_plan["steps"])
        print(f"Ordered steps: {[s['id'] for s in ordered_steps]}")
        
        # Test step result addition
        print("Testing step result addition...")
        step_result = {
            "step_id": "step_1",
            "step_description": "Research types of renewable energy",
            "qa_result": {
                "answer": "Renewable energy includes solar, wind, hydro, and geothermal sources.",
                "confidence": 0.85,
                "sources": ["doc_1", "doc_2"]
            },
            "success": True
        }
        
        await state_manager.add_step_result("step_1", step_result)
        
        # Test state retrieval
        print("Testing state retrieval...")
        current_state = await state_manager.get_current_state()
        print(f"Current state: {json.dumps(current_state, indent=2)}")
        
        # Test history retrieval
        history = await state_manager.get_accumulated_history()
        print(f"Accumulated history: {len(history)} entries")
        
        # Test cleanup
        print("Testing cleanup...")
        cleanup_success = await state_manager.cleanup_execution("test_exec_001")
        print(f"Cleanup successful: {cleanup_success}")
        
        print("✅ State management test completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ State management test failed: {str(e)}")
        return False

async def test_final_assembler():
    """Test final answer assembly functionality."""
    print_section("Final Assembler Test")
    
    try:
        final_assembler = FinalAssembler()
        
        # Test data
        assembler_input = {
            "main_query": "What are the environmental benefits of renewable energy?",
            "disambiguated_query": "What are the specific environmental benefits of renewable energy sources?",
            "query_type": "multi-hop",
            "step_results": [
                {
                    "step_id": "step_1",
                    "step_description": "Research types of renewable energy",
                    "result": {
                        "success": True,
                        "qa_result": {
                            "answer": "Renewable energy includes solar, wind, hydro, and geothermal sources.",
                            "confidence": 0.85,
                            "sources": ["doc_1", "doc_2"]
                        }
                    }
                },
                {
                    "step_id": "step_2",
                    "step_description": "Analyze environmental benefits",
                    "result": {
                        "success": True,
                        "qa_result": {
                            "answer": "Renewable energy reduces greenhouse gas emissions and air pollution.",
                            "confidence": 0.90,
                            "sources": ["doc_3", "doc_4"]
                        }
                    }
                }
            ],
            "plan": {
                "main_question": "What are the environmental benefits of renewable energy?",
                "steps": [
                    {"id": "step_1", "description": "Research types of renewable energy"},
                    {"id": "step_2", "description": "Analyze environmental benefits"}
                ]
            }
        }
        
        print("Testing final answer assembly...")
        result = await final_assembler.assemble_final_answer(assembler_input)
        
        print("✅ Final assembler test completed successfully")
        print(f"Final answer preview: {result['final_answer'][:200]}...")
        print(f"Confidence: {result['confidence']}")
        print(f"Sources: {len(result['sources'])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Final assembler test failed: {str(e)}")
        return False

async def test_pipeline_integration():
    """Test the complete pipeline integration (without actual LLM calls)."""
    print_section("Pipeline Integration Test")
    
    try:
        print("Testing pipeline integration...")
        
        # Note: This test doesn't actually run the full pipeline since it requires
        # initialized retriever with documents and LLM connections
        # Instead, we test the orchestration structure
        
        orchestrator = MARAGOrchestrator()
        
        # Test status
        status = await orchestrator.get_pipeline_status()
        print(f"Orchestrator status: {json.dumps(status, indent=2)}")
        
        # Test cancellation (should work even without active execution)
        cancel_result = await orchestrator.cancel_execution()
        print(f"Cancel execution result: {cancel_result}")
        
        print("✅ Pipeline integration test completed successfully")
        print("\n📝 Note: Full pipeline execution requires:")
        print("   - Initialized retriever agent with documents")
        print("   - Working LLM connections")
        print("   - Proper agent configurations")
        
        return True
        
    except Exception as e:
        print(f"❌ Pipeline integration test failed: {str(e)}")
        return False

async def demonstrate_usage():
    """Demonstrate how to use the MA-RAG pipeline."""
    print_section("Usage Demonstration")
    
    print("""
🚀 MA-RAG Pipeline Usage Guide:

1. Initialize the pipeline:
   ```python
   from orchestrator import MARAGOrchestrator
   
   # Create orchestrator with your agents
   orchestrator = MARAGOrchestrator(
       retriever_agent=your_retriever_agent,  # Must be initialized with documents
       # Other agents will use defaults
   )
   ```

2. Execute a query:
   ```python
   result = await orchestrator.execute_pipeline(
       query="Your question here",
       context={"domain": "your_domain"}  # Optional
   )
   ```

3. Access results:
   ```python
   print(result.final_answer)
   print(f"Confidence: {result.confidence}")
   print(f"Sources: {result.sources}")
   print(f"Reasoning: {result.reasoning_trajectory}")
   ```

4. Convenience function:
   ```python
   from orchestrator import run_marag_pipeline
   
   result = await run_marag_pipeline(
       query="Your question here",
       retriever_agent=your_retriever_agent
   )
   ```

🔧 Key Features:
   ✅ Dynamic agent invocation based on plan structure
   ✅ State management throughout reasoning trajectory  
   ✅ Step dependency resolution
   ✅ Comprehensive final answer assembly
   ✅ Consistent tokenization across all agents
   ✅ Error handling and fallback mechanisms
   ✅ Performance tracking and metrics
    """)

async def main():
    """Run all tests."""
    print("🧪 MA-RAG Pipeline Integration Test Suite")
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    tests = [
        ("Component Initialization", test_component_initialization),
        ("State Management", test_state_management),
        ("Final Assembler", test_final_assembler),
        ("Pipeline Integration", test_pipeline_integration)
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
        print("🎉 All tests passed! MA-RAG pipeline is ready for use.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    
    # Demonstrate usage
    await demonstrate_usage()

if __name__ == "__main__":
    asyncio.run(main())


