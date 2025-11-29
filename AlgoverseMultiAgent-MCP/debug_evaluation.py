#!/usr/bin/env python3
"""
Debug script for evaluation pipeline
Tests with minimal examples and better error handling
"""

import asyncio
import time
import sys
import os

# Add the agents directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'agents'))

from agents.mixed_model_orchestrator import run_optimized_marag_pipeline

async def debug_single_question():
    """Test with just ONE question to see what's happening"""
    
    print("🔍 DEBUG: Testing with ONE question...")
    
    # Simple test question
    test_query = "What is the capital of France?"
    
    try:
        print(f"📝 Query: {test_query}")
        print("⏳ Starting pipeline...")
        
        start_time = time.time()
        
        # Run the pipeline
        result = await run_optimized_marag_pipeline(test_query)
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"✅ Pipeline completed in {duration:.2f} seconds")
        print(f"📊 Result: {result}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print(f"🔍 Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False

async def debug_with_timeout():
    """Test with timeout to prevent hanging"""
    
    print("🔍 DEBUG: Testing with 30-second timeout...")
    
    try:
        # Run with timeout
        result = await asyncio.wait_for(
            debug_single_question(),
            timeout=30.0
        )
        return result
        
    except asyncio.TimeoutError:
        print("⏰ TIMEOUT: Pipeline took longer than 30 seconds!")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting debug evaluation...")
    
    # Test with timeout
    success = asyncio.run(debug_with_timeout())
    
    if success:
        print("✅ Debug completed successfully!")
    else:
        print("❌ Debug failed - check the errors above")
