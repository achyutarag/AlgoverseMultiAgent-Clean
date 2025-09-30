#!/usr/bin/env python3
"""
Test script to verify the package structure works correctly.
Run this from the parent directory (AlgoverseMultiAgent/).
"""

import sys
import os

def test_imports():
    """Test that all imports work correctly."""
    print("🧪 Testing Package Structure...")
    print("=" * 50)
    
    try:
        # Test 1: Import from agents package
        print("1. Testing agents package import...")
        from agents.mixed_model_orchestrator import run_optimized_marag_pipeline
        print("   ✅ agents.mixed_model_orchestrator imported successfully")
        
        # Test 2: Test individual agent imports
        print("\n2. Testing individual agent imports...")
        from agents.planner_agent import PlannerAgent
        from agents.step_definer_agent import StepDefinerAgent
        from agents.retriever_agent import RetrieverAgent
        from agents.extractor_agent import ExtractorAgent
        from agents.qa_agent import QAAgent
        print("   ✅ All agent imports successful")
        
        # Test 3: Test supporting components
        print("\n3. Testing supporting components...")
        from agents.state_manager import StateManager
        from agents.final_assembler import FinalAssembler
        from agents.orchestrator import MARAGOrchestrator
        print("   ✅ All supporting components imported successfully")
        
        # Test 4: Test utility imports
        print("\n4. Testing utility imports...")
        from agents.tokenization_utils import tokenization_utils
        print("   ✅ Utility imports successful")
        
        print("\n🎉 All imports successful! Package structure is correct.")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {str(e)}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        return False

def test_package_info():
    """Test package information."""
    print("\n📦 Package Information:")
    print("=" * 30)
    
    # Check current directory
    current_dir = os.getcwd()
    print(f"Current directory: {current_dir}")
    
    # Check if agents directory exists
    agents_dir = os.path.join(current_dir, "agents")
    if os.path.exists(agents_dir):
        print(f"✅ Agents directory found: {agents_dir}")
        
        # Check for __init__.py
        init_file = os.path.join(agents_dir, "__init__.py")
        if os.path.exists(init_file):
            print("✅ __init__.py found - agents is a proper package")
        else:
            print("❌ __init__.py missing - agents is not a package")
    else:
        print(f"❌ Agents directory not found: {agents_dir}")
    
    # Check Python path
    print(f"\nPython path: {sys.path[:3]}...")  # Show first 3 entries

if __name__ == "__main__":
    print("🚀 MA-RAG Package Structure Test")
    print("=" * 50)
    
    # Test package info
    test_package_info()
    
    # Test imports
    success = test_imports()
    
    if success:
        print("\n✅ Package structure is working correctly!")
        print("\n📋 How to run your test:")
        print("   python -m agents.test_hotpotqa")
    else:
        print("\n❌ Package structure has issues. Check the errors above.")
