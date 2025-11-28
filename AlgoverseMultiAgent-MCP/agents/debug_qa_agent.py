"""
Debug script to test the QA agent JSON generation.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import logging
import json
from agents.qa_agent import QAAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def debug_qa_agent():
    """Debug QA agent JSON generation."""
    
    print("🔍 Debugging QA Agent JSON Generation...")
    print("=" * 50)
    
    try:
        # Initialize QA agent
        print("🔧 Initializing QA agent...")
        qa_agent = QAAgent(
            model_name="gemini-2.5-flash",
            model_config={"use_cuda": False},
            temperature=0.1
        )
        print("✅ QA agent initialized")
        
        # Test with simple context
        test_question = "What is Scott Derrickson's nationality?"
        test_context = [
            {
                "text": "Scott Derrickson (born July 16, 1966) is an American director, screenwriter and producer.",
                "document_id": "hotpotqa_Scott_Derrickson",
                "relevance": 0.9,
                "reasoning": "Directly states nationality"
            }
        ]
        
        print(f"\n🔍 Testing QA for: '{test_question}'")
        print(f"📄 Context: {test_context[0]['text']}")
        
        # Test QA
        qa_input = {
            "question": test_question,
            "context": test_context,
            "min_confidence": 0.3
        }
        
        response = await qa_agent.process(qa_input)
        
        print(f"\n📋 QA Response:")
        print(f"Length: {len(response.content)} characters")
        print(f"Content: {response.content}")
        
        if response.metadata.get("error"):
            print(f"❌ Error: {response.metadata.get('error')}")
        else:
            print("✅ QA completed successfully!")
            
            # Try to parse the JSON
            try:
                result = json.loads(response.content)
                print(f"✅ JSON parsing successful!")
                print(f"📊 Answer: {result.get('answer', 'N/A')}")
                print(f"📊 Confidence: {result.get('confidence', 0.0)}")
                print(f"📊 Reasoning: {result.get('reasoning', 'N/A')}")
            except json.JSONDecodeError as e:
                print(f"❌ JSON parsing failed: {e}")
                print(f"📄 Raw response: {response.content[:500]}...")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        logger.error(f"Test failed: {str(e)}")

if __name__ == "__main__":
    asyncio.run(debug_qa_agent())
