"""
Simple test to check if the LLM wrapper is working.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import logging
from agents.llm_wrapper import GoogleGeminiLLM, LLMConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_llm_wrapper():
    """Test if the LLM wrapper is working."""
    
    print("🔍 Testing LLM Wrapper...")
    print("=" * 50)
    
    try:
        # Initialize LLM wrapper
        print("🔧 Initializing LLM wrapper...")
        config = LLMConfig(
            model_name="models/gemini-flash-lite-latest",
            model_config={"use_cuda": False}
        )
        llm = GoogleGeminiLLM(config)
        print("✅ LLM wrapper initialized")
        
        # Test simple generation
        print("\n🔍 Testing simple generation...")
        simple_prompt = "What is 2+2? Answer with just the number."
        
        response = await llm.generate(simple_prompt)
        print(f"✅ Response: {response.text}")
        
        # Test JSON generation
        print("\n🔍 Testing JSON generation...")
        json_prompt = """Answer this question in JSON format:
Question: What is Scott Derrickson's nationality?
Evidence: Scott Derrickson is an American director.

Return JSON like: {"answer": "American", "confidence": 0.9}"""
        
        response = await llm.generate(json_prompt)
        print(f"✅ JSON Response: {response.text}")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        logger.error(f"Test failed: {str(e)}")

if __name__ == "__main__":
    asyncio.run(test_llm_wrapper())
